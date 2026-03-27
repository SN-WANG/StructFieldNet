# Base trainer for StructFieldNet
# Author: Shengning Wang

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn, Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from structfieldnet.utils.hue_logger import hue, logger


class BaseTrainer:
    """Base trainer with checkpointing, AMP, and metric aggregation."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        scalers: Dict[str, object],
        output_dir: Path,
        max_epochs: int,
        patience: int,
        gradient_clip_norm: float,
        use_amp: bool,
        device: str,
        scheduler: Optional[_LRScheduler] = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: Model instance.
            optimizer: Optimizer instance.
            criterion: Loss module.
            scalers: Dictionary of fitted scalers.
            output_dir: Artifact directory.
            max_epochs: Maximum training epochs.
            patience: Early stopping patience.
            gradient_clip_norm: Gradient clipping threshold.
            use_amp: Whether to enable mixed precision.
            device: Target device string.
            scheduler: Optional learning-rate scheduler.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scalers = scalers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_epochs = max_epochs
        self.patience = patience
        self.gradient_clip_norm = gradient_clip_norm
        self.scheduler = scheduler
        self.history = []
        self.best_val_loss = float("inf")
        self.current_epoch = 0

        self.amp_enabled = use_amp and self.device.type == "cuda"
        self.grad_scaler = GradScaler(enabled=self.amp_enabled)

    def _move_batch_to_device(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Move a batch dictionary onto the trainer device.

        Args:
            batch: Input batch dictionary.

        Returns:
            Device-mapped batch dictionary.
        """
        return {key: value.to(self.device, non_blocking=True) if isinstance(value, Tensor) else value for key, value in batch.items()}

    def _compute_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Compute one forward-loss-metrics step.

        Args:
            batch: Device-resident batch dictionary.

        Returns:
            Dictionary containing at least a scalar `loss`.
        """
        raise NotImplementedError("Subclasses must implement _compute_step.")

    def _run_epoch(self, loader: DataLoader, training: bool) -> Dict[str, float]:
        """Run one training or validation epoch.

        Args:
            loader: Data loader for the epoch.
            training: Whether to update model parameters.

        Returns:
            Dictionary of averaged metrics.
        """
        self.model.train(training)
        aggregated: Dict[str, list[float]] = {}
        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            progress = tqdm(loader, leave=False, dynamic_ncols=True, desc="Training" if training else "Validating")
            for batch in progress:
                batch = self._move_batch_to_device(batch)

                with autocast(enabled=self.amp_enabled):
                    outputs = self._compute_step(batch)
                    loss = outputs["loss"]

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()

                log_items = {}
                for key, value in outputs.items():
                    scalar = float(value.detach().item()) if isinstance(value, Tensor) else float(value)
                    aggregated.setdefault(key, []).append(scalar)
                    log_items[key] = f"{scalar:.4e}"
                progress.set_postfix(log_items)

        return {key: float(np.mean(values)) for key, values in aggregated.items()}

    def _save_checkpoint(self, is_best: bool, val_metrics: Dict[str, float]) -> None:
        """Persist training state to disk.

        Args:
            is_best: Whether this checkpoint is the current best.
            val_metrics: Validation metric dictionary.
        """
        state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_metrics": val_metrics,
            "scalers": {name: scaler.state_dict() for name, scaler in self.scalers.items()},
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(state, self.output_dir / "last.pt")
        if is_best:
            torch.save(state, self.output_dir / "best.pt")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Restore trainer state from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scalers" in checkpoint:
            for name, scaler_state in checkpoint["scalers"].items():
                if name in self.scalers:
                    self.scalers[name].load_state_dict(scaler_state)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Run the training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
        """
        logger.info(
            f"start training on {hue.m}{self.device}{hue.q} | "
            f"epochs: {hue.m}{self.max_epochs}{hue.q} | "
            f"amp: {hue.m}{self.amp_enabled}{hue.q}"
        )
        start_time = time.time()
        patience_counter = 0

        for epoch_idx in range(self.max_epochs):
            self.current_epoch = epoch_idx + 1
            epoch_start = time.time()

            train_metrics = self._run_epoch(train_loader, training=True)
            val_metrics = self._run_epoch(val_loader, training=False)

            if self.scheduler is not None:
                self.scheduler.step()

            val_loss = val_metrics["loss"]
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            self._save_checkpoint(is_best=is_best, val_metrics=val_metrics)
            epoch_time = time.time() - epoch_start

            logger.info(
                f"epoch {hue.b}{self.current_epoch:03d}{hue.q} | "
                f"time: {hue.c}{epoch_time:.1f}s{hue.q} | "
                f"train loss: {hue.m}{train_metrics['loss']:.4e}{hue.q} | "
                f"val loss: {hue.m}{val_metrics['loss']:.4e}{hue.q}"
                + (f" {hue.y}(best){hue.q}" if is_best else "")
            )

            self.history.append(
                {
                    "epoch": self.current_epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )

            if patience_counter >= self.patience:
                logger.info(f"early stopping triggered at epoch {hue.m}{self.current_epoch}{hue.q}")
                break

        with (self.output_dir / "history.json").open("w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=2)
        logger.info(f"{hue.g}training finished in {time.time() - start_time:.1f}s{hue.q}")
