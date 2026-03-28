# Trainer for static stress-field reconstruction
# Author: Shengning Wang

import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from training.base_criterion import Metrics
from training.base_trainer import BaseTrainer
from utils.hue_logger import hue, logger


class FieldTrainer(BaseTrainer):
    """Trainer specialized for design-to-field stress regression."""

    def __init__(self, *args, metrics: Optional[Metrics] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metrics = metrics if metrics is not None else Metrics()

    def _compute_step(self, batch: Dict[str, Tensor]) -> Dict[str, Union[Tensor, float]]:
        coords = batch["coords"]
        design = batch["design"]
        target = batch["stress"]

        pred = self.model(coords, design)
        loss = self.criterion(pred, target)
        outputs: Dict[str, Union[Tensor, float]] = {"loss": loss}

        if not self.model.training:
            stress_scaler = self.scalers.get("stress_scaler")
            pred_denorm = (
                stress_scaler.inverse_transform(pred.detach())
                if stress_scaler is not None
                else pred.detach()
            )
            target_denorm = (
                stress_scaler.inverse_transform(target.detach())
                if stress_scaler is not None
                else target.detach()
            )
            outputs.update(self.metrics.compute(pred_denorm, target_denorm))

        return outputs

    def evaluate(
        self,
        loader: DataLoader,
        checkpoint_path: Optional[Union[str, Path]] = None,
        save_name: str = "metrics.json",
    ) -> Dict[str, float]:
        """Evaluate a checkpoint and save aggregated metrics."""
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        self.model.eval()
        aggregated: Dict[str, list[float]] = {}

        with torch.no_grad():
            progress = tqdm(loader, desc="Evaluating", leave=False, dynamic_ncols=True)
            for batch in progress:
                batch = self._move_batch_to_device(batch)
                outputs = self._compute_step(batch)
                log_dict = {}
                for key, value in outputs.items():
                    scalar = float(value.detach().item()) if isinstance(value, Tensor) else float(value)
                    aggregated.setdefault(key, []).append(scalar)
                    if key in {"loss", "mse", "r2", "accuracy"}:
                        log_dict[key] = f"{scalar:.4e}" if key != "accuracy" else f"{scalar:.2f}"
                progress.set_postfix(log_dict)

        metrics = {key: float(np.mean(values)) for key, values in aggregated.items()}
        with open(self.output_dir / save_name, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)

        logger.info(
            f"evaluation finished | mse: {hue.m}{metrics['mse']:.4e}{hue.q}"
            f" | r2: {hue.m}{metrics['r2']:.4f}{hue.q}"
            f" | accuracy: {hue.m}{metrics['accuracy']:.2f}%{hue.q}"
        )
        return metrics
