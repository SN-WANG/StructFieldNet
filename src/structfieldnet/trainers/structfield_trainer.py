# StructFieldNet trainer
# Author: Shengning Wang

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from structfieldnet.losses.field_loss import compute_field_metrics
from structfieldnet.trainers.base_trainer import BaseTrainer
from structfieldnet.utils.hue_logger import hue, logger


class StructFieldTrainer(BaseTrainer):
    """Trainer specialized for scalar nodal stress field reconstruction."""

    def _compute_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Compute loss terms and metrics for one batch.

        Args:
            batch: Batch dictionary with `coords`, `design`, and `stress`.

        Returns:
            Dictionary containing scalar loss terms and metrics.
        """
        coords = batch["coords"]
        design = batch["design"]
        target = batch["stress"]

        pred = self.model(coords, design)
        loss_dict = self.criterion(pred, target)

        pred_denorm = self.scalers["stress"].inverse_transform(pred.detach())
        target_denorm = self.scalers["stress"].inverse_transform(target.detach())
        metric_dict = compute_field_metrics(
            pred=pred_denorm,
            target=target_denorm,
            hotspot_percentile=self.criterion.hotspot_percentile,
        )
        return {**loss_dict, **metric_dict}

    def evaluate(self, loader: DataLoader, checkpoint_path: Path, save_name: str = "metrics.json") -> Dict[str, float]:
        """Evaluate a checkpoint on a data loader.

        Args:
            loader: Evaluation data loader.
            checkpoint_path: Path to the checkpoint file.
            save_name: Name of the metrics JSON file to save.

        Returns:
            Averaged metric dictionary.
        """
        self.load_checkpoint(checkpoint_path)
        self.model.eval()

        aggregated: Dict[str, list[float]] = {}
        with torch.no_grad():
            progress = tqdm(loader, leave=False, dynamic_ncols=True, desc="Evaluating")
            for batch in progress:
                batch = self._move_batch_to_device(batch)
                outputs = self._compute_step(batch)
                log_items = {}
                for key, value in outputs.items():
                    scalar = float(value.detach().item()) if isinstance(value, Tensor) else float(value)
                    aggregated.setdefault(key, []).append(scalar)
                    log_items[key] = f"{scalar:.4e}"
                progress.set_postfix(log_items)

        metrics = {key: float(np.mean(values)) for key, values in aggregated.items()}
        with (self.output_dir / save_name).open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
        logger.info(
            f"evaluation finished | field mae: {hue.m}{metrics['field_mae']:.4e}{hue.q} | "
            f"field rmse: {hue.m}{metrics['field_rmse']:.4e}{hue.q}"
        )
        return metrics
