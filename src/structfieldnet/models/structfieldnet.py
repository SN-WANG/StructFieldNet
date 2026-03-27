# StructFieldNet: Full-field stress reconstruction model
# Author: Shengning Wang

from typing import List

import torch
from torch import nn, Tensor

from structfieldnet.models.components import PointwiseMLP, StructFieldBlock, _trunc_normal_


class StructFieldNet(nn.Module):
    """StructFieldNet for scalar nodal stress field reconstruction.

    Architecture:
        design -> Branch MLP
        coords -> Trunk MLP
        branch x trunk -> Fusion MLP
        stacked Physics-Attention blocks
        output projection -> scalar stress
    """

    def __init__(
        self,
        coord_dim: int,
        design_dim: int,
        output_dim: int,
        hidden_dim: int,
        branch_hidden_dim: int,
        branch_num_layers: int,
        trunk_hidden_dim: int,
        trunk_num_layers: int,
        fusion_hidden_dim: int,
        fusion_num_layers: int,
        depth: int,
        num_heads: int,
        num_slices: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the StructFieldNet model.

        Args:
            coord_dim: Coordinate dimension.
            design_dim: Design vector dimension.
            output_dim: Output field channel dimension.
            hidden_dim: Backbone hidden dimension.
            branch_hidden_dim: Hidden dimension of the branch MLP.
            branch_num_layers: Number of branch MLP layers.
            trunk_hidden_dim: Hidden dimension of the trunk MLP.
            trunk_num_layers: Number of trunk MLP layers.
            fusion_hidden_dim: Hidden dimension of the fusion MLP.
            fusion_num_layers: Number of fusion MLP layers.
            depth: Number of Physics-Attention blocks.
            num_heads: Number of attention heads.
            num_slices: Number of slice tokens.
            mlp_ratio: Expansion ratio in block MLPs.
            dropout: Dropout probability.
        """
        super().__init__()
        self.coord_dim = coord_dim
        self.design_dim = design_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.branch_mlp = PointwiseMLP(
            input_dim=design_dim,
            hidden_dim=branch_hidden_dim,
            output_dim=hidden_dim,
            num_layers=branch_num_layers,
            dropout=dropout,
        )
        self.trunk_mlp = PointwiseMLP(
            input_dim=coord_dim,
            hidden_dim=trunk_hidden_dim,
            output_dim=hidden_dim,
            num_layers=trunk_num_layers,
            dropout=dropout,
        )
        self.fusion_mlp = PointwiseMLP(
            input_dim=hidden_dim,
            hidden_dim=fusion_hidden_dim,
            output_dim=hidden_dim,
            num_layers=fusion_num_layers + 1,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [
                StructFieldBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_slices=num_slices,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.output_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights.

        Args:
            module: Module to initialize.
        """
        if isinstance(module, nn.Linear):
            _trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, coords: Tensor, design: Tensor) -> Tensor:
        """Predict the scalar nodal stress field.

        Args:
            coords: Normalized mesh coordinates of shape `(B, N, 3)` or `(N, 3)`.
            design: Normalized design vector of shape `(B, M)` or `(M,)`.

        Returns:
            Predicted stress field of shape `(B, N, 1)` or `(N, 1)`.

        Raises:
            ValueError: If tensor dimensions are incompatible.
        """
        squeeze_batch = False
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
            squeeze_batch = True
        if design.dim() == 1:
            design = design.unsqueeze(0)

        if coords.dim() != 3:
            raise ValueError(f"coords must have shape (B, N, 3), got {tuple(coords.shape)}")
        if design.dim() != 2:
            raise ValueError(f"design must have shape (B, M), got {tuple(design.shape)}")
        if coords.shape[0] != design.shape[0]:
            raise ValueError(
                f"coords and design batch sizes must match, got {coords.shape[0]} and {design.shape[0]}"
            )
        if coords.shape[-1] != self.coord_dim:
            raise ValueError(f"coords last dimension must be {self.coord_dim}, got {coords.shape[-1]}")
        if design.shape[-1] != self.design_dim:
            raise ValueError(f"design last dimension must be {self.design_dim}, got {design.shape[-1]}")

        branch_feature = self.branch_mlp(design)
        trunk_feature = self.trunk_mlp(coords)
        hidden = self.fusion_mlp(branch_feature.unsqueeze(1) * trunk_feature)

        for block in self.blocks:
            hidden = block(hidden)

        output = self.output_head(hidden)
        return output.squeeze(0) if squeeze_batch else output

    def extra_repr(self) -> str:
        """Return a concise model summary string."""
        return (
            f"coord_dim={self.coord_dim}, design_dim={self.design_dim}, "
            f"output_dim={self.output_dim}, hidden_dim={self.hidden_dim}"
        )
