# StructFieldNet: design-conditioned stress field reconstruction
# Author: Shengning Wang

from typing import Optional

import torch
from torch import Tensor, nn


def _trunc_normal_(tensor: Tensor, std: float = 0.02) -> Tensor:
    """Initialize a tensor with truncated normal noise."""
    with torch.no_grad():
        tensor.normal_(0.0, std)
        while True:
            mask = tensor.abs() > 2.0 * std
            if not mask.any():
                break
            tensor[mask] = torch.empty_like(tensor[mask]).normal_(0.0, std)
    return tensor


class MLP(nn.Module):
    """Pointwise multilayer perceptron on the last tensor dimension."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PhysicsAttention(nn.Module):
    """Slice-space attention adapted for irregular structural meshes."""

    def __init__(
        self,
        width: int,
        num_slices: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"width={width} must be divisible by num_heads={num_heads}")

        self.slice_proj = nn.Linear(width, num_slices)
        self.attention = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Sequential(
            nn.Linear(width, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must have shape (B, N, C), got {tuple(x.shape)}")

        weights = torch.softmax(self.slice_proj(x), dim=-1)
        normalizer = weights.sum(dim=1, keepdim=True).transpose(1, 2).clamp_min(1e-6)
        tokens = torch.bmm(weights.transpose(1, 2), x) / normalizer

        attended_tokens, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        broadcast = torch.bmm(weights, attended_tokens)
        return self.out_proj(broadcast)


class StructFieldBlock(nn.Module):
    """Pre-normalized Physics-Attention block."""

    def __init__(
        self,
        width: int,
        num_slices: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = PhysicsAttention(
            width=width,
            num_slices=num_slices,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * mlp_ratio, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class StructFieldNet(nn.Module):
    """StructFieldNet for scalar von Mises stress reconstruction on a fixed mesh."""

    def __init__(
        self,
        coord_dim: int,
        design_dim: int,
        output_dim: int = 1,
        width: int = 128,
        depth: int = 6,
        num_slices: int = 64,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        branch_hidden_dim: int = 128,
        branch_layers: int = 2,
        trunk_hidden_dim: int = 128,
        trunk_layers: int = 2,
        lifting_hidden_dim: int = 128,
        lifting_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.coord_dim = coord_dim
        self.design_dim = design_dim
        self.output_dim = output_dim
        self.width = width

        self.branch_encoder = MLP(
            input_dim=design_dim,
            hidden_dim=branch_hidden_dim,
            output_dim=width,
            num_layers=branch_layers,
            dropout=dropout,
        )
        self.trunk_encoder = MLP(
            input_dim=coord_dim,
            hidden_dim=trunk_hidden_dim,
            output_dim=width,
            num_layers=trunk_layers,
            dropout=dropout,
        )
        self.lifting = MLP(
            input_dim=width,
            hidden_dim=lifting_hidden_dim,
            output_dim=width,
            num_layers=lifting_layers,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [
                StructFieldBlock(
                    width=width,
                    num_slices=num_slices,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(width)
        self.output_head = nn.Linear(width, output_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            _trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, coords: Tensor, design: Tensor) -> Tensor:
        """Predict nodal stress values from coordinates and a design vector.

        Args:
            coords: Mesh coordinates with shape ``(B, N, 3)`` or ``(N, 3)``.
            design: Thickness design vector with shape ``(B, M)`` or ``(M,)``.

        Returns:
            Predicted stress field with shape ``(B, N, 1)`` or ``(N, 1)``.
        """
        squeeze_batch = False
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
            squeeze_batch = True
        if design.dim() == 1:
            design = design.unsqueeze(0)

        if coords.dim() != 3:
            raise ValueError(f"coords must have shape (B, N, C), got {tuple(coords.shape)}")
        if design.dim() != 2:
            raise ValueError(f"design must have shape (B, M), got {tuple(design.shape)}")
        if coords.shape[0] != design.shape[0]:
            raise ValueError(
                f"batch mismatch between coords and design: {coords.shape[0]} vs {design.shape[0]}"
            )
        if coords.shape[-1] != self.coord_dim:
            raise ValueError(
                f"coords last dimension must be {self.coord_dim}, got {coords.shape[-1]}"
            )
        if design.shape[-1] != self.design_dim:
            raise ValueError(
                f"design last dimension must be {self.design_dim}, got {design.shape[-1]}"
            )

        branch_feature = self.branch_encoder(design).unsqueeze(1)
        trunk_feature = self.trunk_encoder(coords)
        hidden = self.lifting(branch_feature * trunk_feature)

        for block in self.blocks:
            hidden = block(hidden)

        output = self.output_head(self.output_norm(hidden))
        return output.squeeze(0) if squeeze_batch else output

    def predict(self, coords: Tensor, design: Tensor) -> Tensor:
        """Alias for ``forward`` kept for experiment readability."""
        return self.forward(coords, design)

    def extra_repr(self) -> str:
        return (
            f"coord_dim={self.coord_dim}, design_dim={self.design_dim}, "
            f"output_dim={self.output_dim}, width={self.width}"
        )
