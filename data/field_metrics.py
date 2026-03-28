# Metrics for StructFieldNet
# Author: Shengning Wang

from typing import Dict

import torch
from torch import Tensor


class FieldMetrics:
    """Metrics for fixed-mesh structural field reconstruction."""

    def __init__(self, hotspot_percentile: float = 0.95, eps: float = 1e-8) -> None:
        if hotspot_percentile <= 0.0 or hotspot_percentile >= 1.0:
            raise ValueError("hotspot_percentile must lie in the open interval (0, 1)")
        self.hotspot_percentile = hotspot_percentile
        self.eps = eps

    def compute(self, pred: Tensor, target: Tensor) -> Dict[str, float]:
        """Compute scalar metrics on one field sample.

        Args:
            pred: Predicted field with shape (num_nodes, num_channels) or
                (batch_size, num_nodes, num_channels).
            target: Ground-truth field. Shape must match pred.
        """
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")

        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)

        diff = pred_flat - target_flat
        abs_diff = diff.abs()
        sq_diff = diff.square()

        mse = torch.mean(sq_diff)
        rmse = torch.sqrt(mse)
        mae = torch.mean(abs_diff)

        ss_res = torch.sum(sq_diff)
        ss_tot = torch.sum((target_flat - torch.mean(target_flat)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + self.eps)

        accuracy = (1.0 - torch.sum(abs_diff) / (torch.sum(target_flat.abs()) + self.eps)) * 100.0
        max_error = torch.max(abs_diff)

        hotspot_threshold = torch.quantile(target_flat, self.hotspot_percentile)
        hotspot_pred = pred_flat >= hotspot_threshold
        hotspot_target = target_flat >= hotspot_threshold
        hotspot_union = (hotspot_pred | hotspot_target).sum()
        hotspot_intersection = (hotspot_pred & hotspot_target).sum()
        hotspot_iou = hotspot_intersection.float() / (hotspot_union.float() + self.eps)

        return {
            "mse": float(mse.item()),
            "rmse": float(rmse.item()),
            "mae": float(mae.item()),
            "r2": float(r2.item()),
            "accuracy": float(accuracy.item()),
            "max_error": float(max_error.item()),
            "hotspot_iou": float(hotspot_iou.item()),
        }
