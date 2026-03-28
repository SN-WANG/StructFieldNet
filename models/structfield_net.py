# StructFieldNet: Design-Conditioned Structural Field Reconstruction
# Author: Shengning Wang
#
# Fixed-mesh setting:
#   1. Encode the design vector.
#   2. Encode nodal coordinates.
#   3. Fuse design and coordinate features at each node.
#   4. Update nodal features with Physics Attention blocks.
#   5. Regress the scalar stress field.

from typing import List

import torch
from torch import nn, Tensor
from torch.nn import functional as F


# ============================================================
# Basic MLP
# ============================================================

class MLP(nn.Module):
    """Simple multilayer perceptron."""

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
            raise ValueError("num_layers must be at least 1")

        dims: List[int] = [input_dim]
        if num_layers > 1:
            dims += [hidden_dim] * (num_layers - 1)
        dims += [output_dim]

        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != output_dim:
                layers.append(nn.GELU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ============================================================
# Physics Attention
# ============================================================

class PhysicsAttention(nn.Module):
    """Slice-space attention on irregular structural meshes."""

    def __init__(self, width: int, num_slices: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"width={width} must be divisible by num_heads={num_heads}")

        self.slice_proj = nn.Linear(width, num_slices)
        self.attn = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(width, width)

    def forward(self, x: Tensor) -> Tensor:
        """Run slice attention.

        Args:
            x: Node features with shape (batch_size, num_nodes, num_channels).

        Returns:
            Updated node features with shape (batch_size, num_nodes, num_channels).
        """
        weights = F.softmax(self.slice_proj(x), dim=-1)
        weight_sum = weights.sum(dim=1, keepdim=True).transpose(1, 2).clamp_min(1e-6)
        slices = torch.bmm(weights.transpose(1, 2), x) / weight_sum
        slices, _ = self.attn(slices, slices, slices, need_weights=False)
        return self.out_proj(torch.bmm(weights, slices))


# ============================================================
# StructField Block
# ============================================================

class StructFieldBlock(nn.Module):
    """Physics Attention block with pre-normalization."""

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
        self.attn = PhysicsAttention(width, num_slices, num_heads, dropout)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * mlp_ratio, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# StructFieldNet
# ============================================================

class StructFieldNet(nn.Module):
    """StructFieldNet for fixed-mesh stress reconstruction."""

    def __init__(
        self,
        num_nodes: int,
        coord_dim: int,
        design_dim: int,
        output_dim: int = 1,
        width: int = 64,
        depth: int = 4,
        num_slices: int = 32,
        num_heads: int = 4,
        num_bases: int = 32,
        mlp_ratio: int = 4,
        branch_hidden_dim: int = 64,
        branch_layers: int = 2,
        trunk_hidden_dim: int = 64,
        trunk_layers: int = 2,
        lifting_hidden_dim: int = 64,
        lifting_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.coord_dim = coord_dim
        self.design_dim = design_dim
        self.output_dim = output_dim
        self.width = width
        self.num_bases = num_bases

        self.design_encoder = MLP(design_dim, branch_hidden_dim, width, branch_layers, dropout)
        self.coord_encoder = MLP(coord_dim, trunk_hidden_dim, width, trunk_layers, dropout)
        self.fusion = MLP(width, lifting_hidden_dim, width, lifting_layers, dropout)
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
        self.norm = nn.LayerNorm(width)
        self.basis_coeff = nn.Linear(design_dim, num_bases)
        self.basis_fields = nn.Parameter(torch.empty(num_bases, num_nodes, output_dim))
        self.field_bias = nn.Parameter(torch.zeros(num_nodes, output_dim))
        self.residual_head = nn.Linear(width, output_dim)

        self.apply(self._init_weights)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02, a=-0.04, b=0.04)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @torch.no_grad()
    def initialize_basis(self, design: Tensor, stress: Tensor) -> None:
        """Warm-start the fixed-mesh basis decoder from the training split.

        Args:
            design: Scaled design matrix with shape (batch_size, design_dim).
            stress: Scaled target field with shape (batch_size, num_nodes, output_dim).
        """
        if design.ndim != 2:
            raise ValueError(f"design must have shape (batch_size, design_dim), got {tuple(design.shape)}")
        if stress.ndim != 3:
            raise ValueError(
                f"stress must have shape (batch_size, num_nodes, output_dim), got {tuple(stress.shape)}"
            )
        if stress.shape[1] != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got {stress.shape[1]}")
        if stress.shape[2] != self.output_dim:
            raise ValueError(f"Expected output_dim={self.output_dim}, got {stress.shape[2]}")

        target = stress.reshape(stress.shape[0], -1)
        field_mean = target.mean(dim=0, keepdim=True)
        centered = target - field_mean

        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh[: self.num_bases]
        coeff = centered @ basis.transpose(0, 1)

        ones = torch.ones(design.shape[0], 1, dtype=design.dtype, device=design.device)
        design_aug = torch.cat([design, ones], dim=1)
        solution = torch.linalg.lstsq(design_aug, coeff).solution

        self.field_bias.copy_(field_mean.reshape(self.num_nodes, self.output_dim))
        self.basis_fields.copy_(basis.reshape(self.num_bases, self.num_nodes, self.output_dim))
        self.basis_coeff.weight.copy_(solution[:-1].transpose(0, 1))
        self.basis_coeff.bias.copy_(solution[-1])

    def forward(self, coords: Tensor, design: Tensor) -> Tensor:
        """Predict the stress field.

        Args:
            coords: Mesh coordinates with shape (batch_size, num_nodes, coord_dim).
            design: Design vector with shape (batch_size, design_dim).

        Returns:
            Predicted field with shape (batch_size, num_nodes, output_dim).
        """
        if coords.ndim != 3:
            raise ValueError(f"coords must have shape (batch_size, num_nodes, coord_dim), got {tuple(coords.shape)}")
        if design.ndim != 2:
            raise ValueError(f"design must have shape (batch_size, design_dim), got {tuple(design.shape)}")
        if coords.shape[0] != design.shape[0]:
            raise ValueError(f"Batch mismatch: coords={coords.shape[0]} vs design={design.shape[0]}")
        if coords.shape[1] != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got {coords.shape[1]}")
        if coords.shape[-1] != self.coord_dim:
            raise ValueError(f"Expected coord_dim={self.coord_dim}, got {coords.shape[-1]}")
        if design.shape[-1] != self.design_dim:
            raise ValueError(f"Expected design_dim={self.design_dim}, got {design.shape[-1]}")

        coarse = self.field_bias.unsqueeze(0) + torch.einsum(
            "bk,kno->bno",
            self.basis_coeff(design),
            self.basis_fields,
        )

        design_feature = self.design_encoder(design).unsqueeze(1)
        coord_feature = self.coord_encoder(coords)
        hidden = self.fusion(design_feature * coord_feature)

        for block in self.blocks:
            hidden = block(hidden)

        residual = self.residual_head(self.norm(hidden))
        return coarse + residual
