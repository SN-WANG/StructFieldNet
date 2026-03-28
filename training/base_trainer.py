# Base trainer with training loop, checkpointing, and evaluation hooks
# Author: Shengning Wang

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.hue_logger import hue, logger


class BaseTrainer:
    """Base trainer for experiment-oriented PyTorch workflows."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        max_epochs: int = 100,
        patience: Optional[int] = None,
        scalers: Optional[Dict[str, object]] = None,
        output_dir: Optional[Union[str, Path]] = "./runs",
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = True,
        gradient_clip_norm: Optional[float] = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.scalers = scalers or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = optimizer if optimizer is not None else Adam(self.model.parameters(), lr=lr)
        self.scheduler = scheduler
        self.criterion = criterion if criterion is not None else nn.MSELoss()

        self.max_epochs = max_epochs
        self.patience = max_epochs if patience is None else patience
        self.gradient_clip_norm = gradient_clip_norm

        self.current_epoch = 0
        self.best_loss = float("inf")
        self.history: List[Dict[str, float]] = []

        self.amp_enabled = bool(use_amp and self.device.type == "cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        else:  # pragma: no cover - compatibility path
            self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    def _move_batch_to_device(self, batch: Any) -> Any:
        """Recursively move tensor batches to the configured device."""
        if isinstance(batch, Tensor):
            return batch.to(self.device)
        if isinstance(batch, dict):
            return {
                key: value.to(self.device) if isinstance(value, Tensor) else value
                for key, value in batch.items()
            }
        if isinstance(batch, (list, tuple)):
            return [value.to(self.device) if isinstance(value, Tensor) else value for value in batch]
        return batch

    def _compute_step(self, batch: Any) -> Dict[str, Union[Tensor, float]]:
        """Compute one forward step. Subclasses must return a dict containing ``loss``."""
        raise NotImplementedError("Subclasses must implement _compute_step().")

    def _on_epoch_start(self, **kwargs) -> None:
        """Optional hook called at the beginning of each epoch."""

    def _on_epoch_end(self, **kwargs) -> None:
        """Optional hook called at the end of each epoch."""

    def _checkpoint_state(self, monitor_loss: float) -> Dict[str, object]:
        """Build the serializable checkpoint state."""
        state: Dict[str, object] = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "monitor_loss": monitor_loss,
        }

        if self.scalers:
            state["scaler_state_dict"] = {
                name: scaler.state_dict() for name, scaler in self.scalers.items()
            }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        return state

    def _save_checkpoint(self, monitor_loss: float, is_best: bool = False) -> None:
        """Save the latest checkpoint and optionally the best checkpoint."""
        state = self._checkpoint_state(monitor_loss)
        torch.save(state, self.output_dir / "ckpt.pt")
        if is_best:
            torch.save(state, self.output_dir / "best.pt")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, object]:
        """Load model, optimizer, scheduler, and scaler state from a checkpoint."""
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scalers and "scaler_state_dict" in checkpoint:
            for name, scaler in self.scalers.items():
                if name in checkpoint["scaler_state_dict"]:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"][name])

        self.current_epoch = int(checkpoint.get("epoch", 0))
        return checkpoint

    def _run_epoch(self, loader: DataLoader, is_training: bool) -> Dict[str, float]:
        """Run one training or validation epoch."""
        self.model.train(is_training)
        aggregated: Dict[str, List[float]] = {}

        context = torch.enable_grad() if is_training else torch.no_grad()
        desc = "Training" if is_training else "Validating"

        with context:
            progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
            for batch in progress:
                batch = self._move_batch_to_device(batch)

                if is_training:
                    self.optimizer.zero_grad(set_to_none=True)

                if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                    autocast_context = torch.amp.autocast(
                        device_type=self.device.type,
                        enabled=self.amp_enabled,
                    )
                else:  # pragma: no cover - compatibility path
                    autocast_context = torch.cuda.amp.autocast(enabled=self.amp_enabled)

                with autocast_context:
                    outputs = self._compute_step(batch)

                if "loss" not in outputs:
                    raise KeyError("Trainer step output must contain a 'loss' key.")

                loss = outputs["loss"]
                if not isinstance(loss, Tensor):
                    raise TypeError("Step output 'loss' must be a torch.Tensor.")
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Encountered non-finite loss at epoch {self.current_epoch}: {loss.item()}"
                    )

                if is_training:
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.unscale_(self.optimizer)

                    if self.gradient_clip_norm is not None and self.gradient_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()

                log_dict: Dict[str, str] = {}
                for key, value in outputs.items():
                    scalar = float(value.detach().item()) if isinstance(value, Tensor) else float(value)
                    aggregated.setdefault(key, []).append(scalar)
                    if key in {"loss", "mse", "r2", "accuracy"}:
                        log_dict[key] = f"{scalar:.4e}" if key != "accuracy" else f"{scalar:.2f}"

                progress.set_postfix(log_dict)

        return {key: float(np.mean(values)) for key, values in aggregated.items()}

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        """Run the full training loop."""
        logger.info(
            f"start training on {hue.m}{self.device}{hue.q} for "
            f"{hue.m}{self.max_epochs}{hue.q} epochs"
        )
        start_time = time.time()
        patience_counter = 0

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()

            self._on_epoch_start()

            train_metrics = self._run_epoch(train_loader, is_training=True)
            val_metrics = self._run_epoch(val_loader, is_training=False) if val_loader is not None else None

            train_loss = train_metrics["loss"]
            val_loss = val_metrics["loss"] if val_metrics is not None else None
            monitor_loss = val_loss if val_loss is not None else train_loss

            self._on_epoch_end(train_metrics=train_metrics, val_metrics=val_metrics)

            if self.scheduler is not None:
                self.scheduler.step()

            is_best = monitor_loss < self.best_loss
            if is_best:
                self.best_loss = monitor_loss
                patience_counter = 0
            else:
                patience_counter += 1

            self._save_checkpoint(monitor_loss, is_best=is_best)

            history_item = {
                "epoch": float(self.current_epoch),
                "train_loss": float(train_loss),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            }
            if val_loss is not None:
                history_item["val_loss"] = float(val_loss)
            self.history.append(history_item)

            duration = time.time() - epoch_start
            val_string = (
                f" | val loss: {hue.m}{val_loss:.4e}{hue.q}"
                f"{f' {hue.y}(best){hue.q}' if is_best and val_loss is not None else ''}"
                if val_loss is not None
                else ""
            )
            logger.info(
                f"epoch {hue.b}{self.current_epoch:03d}{hue.q}"
                f" | time: {hue.c}{duration:.1f}s{hue.q}"
                f" | train loss: {hue.m}{train_loss:.4e}{hue.q}{val_string}"
            )

            if patience_counter >= self.patience:
                logger.info(
                    f"early stopping triggered at epoch {hue.m}{self.current_epoch}{hue.q}"
                )
                break

        with open(self.output_dir / "history.json", "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=2)

        logger.info(f"{hue.g}training finished in {time.time() - start_time:.1f}s{hue.q}")
