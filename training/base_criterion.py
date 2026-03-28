# Base criterion and metrics for StructFieldNet
# Author: Shengning Wang

from typing import Dict

import torch
from torch import Tensor, nn


class BaseCriterion(nn.Module):
    """Abstract base class for loss functions."""

    def forward(self, pred: Tensor, target: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError("Subclasses must implement forward().")


class MSECriterion(BaseCriterion):
    """Plain mean squared error loss."""

    def forward(self, pred: Tensor, target: Tensor, **kwargs) -> Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
        return torch.mean((pred - target) ** 2)


class Metrics:
    """Global field metrics for static stress prediction."""

    def __init__(self, hotspot_percentile: float = 0.95, eps: float = 1e-8):
        if not 0.0 < hotspot_percentile < 1.0:
            raise ValueError("hotspot_percentile must be between 0 and 1.")
        self.hotspot_percentile = hotspot_percentile
        self.eps = eps

    def compute(self, pred: Tensor, target: Tensor) -> Dict[str, float]:
        """Compute case-level metrics on a stress field.

        Args:
            pred: Predicted stress tensor of shape ``(N, 1)`` or ``(B, N, 1)``.
            target: Reference stress tensor with the same shape as ``pred``.

        Returns:
            Flat dictionary of averaged metrics.
        """
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        if pred.dim() != 3 or pred.shape[-1] != 1:
            raise ValueError("pred and target must have shape (N, 1) or (B, N, 1).")

        pred_flat = pred.squeeze(-1)
        target_flat = target.squeeze(-1)
        diff = pred_flat - target_flat
        abs_diff = diff.abs()
        sq_diff = diff.square()

        mse = torch.mean(sq_diff, dim=1)
        rmse = torch.sqrt(mse)
        mae = torch.mean(abs_diff, dim=1)

        ss_res = torch.sum(sq_diff, dim=1)
        target_mean = torch.mean(target_flat, dim=1, keepdim=True)
        ss_tot = torch.sum((target_flat - target_mean).square(), dim=1)
        r2 = 1.0 - ss_res / (ss_tot + self.eps)

        accuracy = (
            1.0 - torch.sum(abs_diff, dim=1) / (torch.sum(target_flat.abs(), dim=1) + self.eps)
        ) * 100.0

        relative_l2 = torch.linalg.norm(diff, dim=1) / (
            torch.linalg.norm(target_flat, dim=1) + self.eps
        )

        threshold = torch.quantile(
            target_flat,
            q=self.hotspot_percentile,
            dim=1,
            keepdim=True,
        )
        hotspot_mask = target_flat >= threshold

        hotspot_mse_values = []
        hotspot_mae_values = []
        for batch_idx in range(pred_flat.shape[0]):
            masked_sq = sq_diff[batch_idx][hotspot_mask[batch_idx]]
            masked_abs = abs_diff[batch_idx][hotspot_mask[batch_idx]]
            hotspot_mse_values.append(masked_sq.mean() if masked_sq.numel() > 0 else torch.zeros((), device=pred.device))
            hotspot_mae_values.append(masked_abs.mean() if masked_abs.numel() > 0 else torch.zeros((), device=pred.device))

        hotspot_mse = torch.stack(hotspot_mse_values)
        hotspot_mae = torch.stack(hotspot_mae_values)

        max_stress_error = (
            pred_flat.amax(dim=1) - target_flat.amax(dim=1)
        ).abs()

        return {
            "mse": float(mse.mean().item()),
            "rmse": float(rmse.mean().item()),
            "mae": float(mae.mean().item()),
            "r2": float(r2.mean().item()),
            "accuracy": float(accuracy.mean().item()),
            "relative_l2": float(relative_l2.mean().item()),
            "hotspot_mse": float(hotspot_mse.mean().item()),
            "hotspot_mae": float(hotspot_mae.mean().item()),
            "max_stress_error": float(max_stress_error.mean().item()),
        }
