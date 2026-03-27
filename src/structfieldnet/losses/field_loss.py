# Losses and metrics for StructFieldNet
# Author: Shengning Wang

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def compute_field_metrics(pred: Tensor, target: Tensor, hotspot_percentile: float = 0.95) -> Dict[str, Tensor]:
    """Compute regression metrics on the denormalized stress field.

    Args:
        pred: Predicted stress tensor of shape `(B, N, 1)`.
        target: Reference stress tensor of shape `(B, N, 1)`.
        hotspot_percentile: Percentile threshold used to define hotspots.

    Returns:
        Dictionary of scalar metric tensors.
    """
    error = pred - target
    field_mae = error.abs().mean()
    field_rmse = torch.sqrt((error.square()).mean())

    pred_flat = pred.squeeze(-1)
    target_flat = target.squeeze(-1)
    error_flat = error.squeeze(-1)

    numerator = torch.linalg.norm(error_flat, dim=1)
    denominator = torch.linalg.norm(target_flat, dim=1).clamp_min(1e-8)
    relative_l2 = (numerator / denominator).mean()

    threshold = torch.quantile(target_flat, q=hotspot_percentile, dim=1, keepdim=True)
    hotspot_mask = target_flat >= threshold
    hotspot_values = error_flat.abs()[hotspot_mask]
    hotspot_mae = hotspot_values.mean() if hotspot_values.numel() > 0 else torch.zeros((), device=pred.device)

    max_stress_mae = (pred_flat.amax(dim=1) - target_flat.amax(dim=1)).abs().mean()
    return {
        "field_mae": field_mae,
        "field_rmse": field_rmse,
        "relative_l2": relative_l2,
        "hotspot_mae": hotspot_mae,
        "max_stress_mae": max_stress_mae,
    }


class StructFieldLoss(nn.Module):
    """Combined global and hotspot-weighted MSE loss for field reconstruction."""

    def __init__(
        self,
        global_weight: float = 1.0,
        hotspot_weight: float = 1.0,
        hotspot_percentile: float = 0.95,
        hotspot_boost: float = 4.0,
    ) -> None:
        """Initialize the loss function.

        Args:
            global_weight: Weight of the global field term.
            hotspot_weight: Weight of the hotspot term.
            hotspot_percentile: Percentile threshold defining hotspot nodes.
            hotspot_boost: Multiplicative weight assigned to hotspot nodes.
        """
        super().__init__()
        if not 0.0 < hotspot_percentile < 1.0:
            raise ValueError("hotspot_percentile must be between 0 and 1.")
        if hotspot_boost < 1.0:
            raise ValueError("hotspot_boost must be at least 1.")

        self.global_weight = global_weight
        self.hotspot_weight = hotspot_weight
        self.hotspot_percentile = hotspot_percentile
        self.hotspot_boost = hotspot_boost

    def forward(self, pred: Tensor, target: Tensor) -> Dict[str, Tensor]:
        """Compute the training loss terms.

        Args:
            pred: Predicted stress tensor of shape `(B, N, 1)`.
            target: Target stress tensor of shape `(B, N, 1)`.

        Returns:
            Dictionary with total and component losses.
        """
        global_loss = F.mse_loss(pred, target)

        target_flat = target.squeeze(-1)
        threshold = torch.quantile(target_flat, q=self.hotspot_percentile, dim=1, keepdim=True)
        hotspot_mask = target_flat >= threshold
        weights = torch.ones_like(target_flat)
        weights = weights + hotspot_mask.float() * (self.hotspot_boost - 1.0)
        hotspot_loss = (weights.unsqueeze(-1) * (pred - target).square()).mean()

        total_loss = self.global_weight * global_loss + self.hotspot_weight * hotspot_loss
        return {
            "loss": total_loss,
            "global_loss": global_loss,
            "hotspot_loss": hotspot_loss,
        }
