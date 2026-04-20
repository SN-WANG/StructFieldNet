# StructFieldNet for fixed-mesh structural field reconstruction
# Author: Shengning Wang

from typing import List

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# ============================================================
# Building Blocks
# ============================================================


class MLP(nn.Module):
    """
    Compact MLP with an explicit output projection.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int] | None = None,
        activation: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize the MLP.

        Args:
            input_dim (int): Input feature dimension.
            output_dim (int): Output feature dimension.
            hidden_sizes (List[int] | None): Hidden-layer sizes.
            activation (type[nn.Module]): Hidden activation class.
            dropout (float): Dropout rate after hidden layers.
        """
        super().__init__()
        hidden_sizes = hidden_sizes or []

        layers: List[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.output_head = nn.Linear(current_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the MLP.

        Args:
            x (Tensor): Input features. (..., C_IN).

        Returns:
            Tensor: Output features. (..., C_OUT).
        """
        return self.output_head(self.net(x))


class PhysicsAttention(nn.Module):
    """
    Slice-space attention for fixed-mesh node tokens.
    """

    def __init__(self, width: int, num_slices: int, num_heads: int, dropout: float = 0.0) -> None:
        """
        Initialize the physics attention module.

        Args:
            width (int): Node token width.
            num_slices (int): Number of slice tokens.
            num_heads (int): Number of slice-space attention heads.
            dropout (float): Attention dropout rate.
        """
        super().__init__()
        self.slice_proj = nn.Linear(width, num_slices)
        self.attn = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(width, width)

    def forward(self, x: Tensor) -> Tensor:
        """
        Aggregate nodes into slices, update slice states, and broadcast them back.

        Args:
            x (Tensor): Node tokens. (B, N, C).

        Returns:
            Tensor: Updated node tokens. (B, N, C).
        """
        slice_weights = F.softmax(self.slice_proj(x), dim=-1)
        slice_norm = slice_weights.sum(dim=1, keepdim=True).transpose(1, 2).clamp_min(1e-6)
        slice_tokens = torch.bmm(slice_weights.transpose(1, 2), x) / slice_norm
        slice_tokens, _ = self.attn(slice_tokens, slice_tokens, slice_tokens, need_weights=False)
        return self.out_proj(torch.bmm(slice_weights, slice_tokens))


class StructFieldBlock(nn.Module):
    """
    Residual StructFieldNet operator block.
    """

    def __init__(self, width: int, num_slices: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        """
        Initialize one StructFieldNet block.

        Args:
            width (int): Node token width.
            num_slices (int): Number of slice tokens.
            num_heads (int): Number of attention heads.
            mlp_ratio (int): Feed-forward expansion ratio.
            dropout (float): Dropout rate.
        """
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
        """
        Apply one residual operator update.

        Args:
            x (Tensor): Node tokens. (B, N, C).

        Returns:
            Tensor: Updated node tokens. (B, N, C).
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# StructFieldNet
# ============================================================


class StructFieldNet(nn.Module):
    """
    Design-conditioned fixed-mesh structural field reconstructor.
    """

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
        branch_layers: int = 1,
        trunk_hidden_dim: int = 64,
        trunk_layers: int = 1,
        lifting_hidden_dim: int = 64,
        lifting_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize StructFieldNet.

        Args:
            num_nodes (int): Number of mesh nodes.
            coord_dim (int): Coordinate dimension.
            design_dim (int): Design vector dimension.
            output_dim (int): Output field dimension.
            width (int): Hidden token width.
            depth (int): Number of operator blocks.
            num_slices (int): Number of slice tokens.
            num_heads (int): Number of slice-space attention heads.
            num_bases (int): Number of coarse basis fields.
            mlp_ratio (int): Feed-forward expansion ratio.
            branch_hidden_dim (int): Hidden width of the design encoder.
            branch_layers (int): Number of hidden layers in the design encoder.
            trunk_hidden_dim (int): Hidden width of the coordinate encoder.
            trunk_layers (int): Number of hidden layers in the coordinate encoder.
            lifting_hidden_dim (int): Hidden width of the fusion MLP.
            lifting_layers (int): Number of hidden layers in the fusion MLP.
            dropout (float): Dropout rate.
        """
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"width={width} must be divisible by num_heads={num_heads}")

        self.num_nodes = num_nodes
        self.coord_dim = coord_dim
        self.design_dim = design_dim
        self.output_dim = output_dim
        self.width = width
        self.num_bases = num_bases

        self.design_encoder = MLP(
            design_dim,
            width,
            hidden_sizes=[branch_hidden_dim] * branch_layers,
            dropout=dropout,
        )
        self.coord_encoder = MLP(
            coord_dim,
            width,
            hidden_sizes=[trunk_hidden_dim] * trunk_layers,
            dropout=dropout,
        )
        self.fusion = MLP(
            width,
            width,
            hidden_sizes=[lifting_hidden_dim] * lifting_layers,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList([
            StructFieldBlock(
                width=width,
                num_slices=num_slices,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(width)
        self.basis_coeff = nn.Linear(design_dim, num_bases)
        self.basis_fields = nn.Parameter(torch.empty(num_bases, num_nodes, output_dim))
        self.field_bias = nn.Parameter(torch.zeros(num_nodes, output_dim))
        self.residual_head = nn.Linear(width, output_dim)

        self.apply(self._init_weights)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize linear and normalization layers.

        Args:
            module (nn.Module): One submodule.
        """
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02, a=-0.04, b=0.04)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @torch.no_grad()
    def initialize_basis(self, design: Tensor, stress: Tensor) -> None:
        """
        Warm-start the coarse basis decoder from the training split.

        Args:
            design (Tensor): Scaled design matrix. (B, C_DESIGN).
            stress (Tensor): Scaled target fields. (B, N, C_OUT).
        """
        flat_stress = stress.reshape(stress.shape[0], -1)
        field_mean = flat_stress.mean(dim=0, keepdim=True)
        centered_stress = flat_stress - field_mean

        _, _, right_vectors = torch.linalg.svd(centered_stress, full_matrices=False)
        basis_vectors = right_vectors[:self.num_bases]
        basis_coeff = centered_stress @ basis_vectors.transpose(0, 1)

        bias_column = torch.ones(design.shape[0], 1, dtype=design.dtype, device=design.device)
        design_aug = torch.cat([design, bias_column], dim=1)
        coeff_solution = torch.linalg.lstsq(design_aug, basis_coeff).solution

        self.field_bias.copy_(field_mean.reshape(self.num_nodes, self.output_dim))
        self.basis_fields.copy_(basis_vectors.reshape(self.num_bases, self.num_nodes, self.output_dim))
        self.basis_coeff.weight.copy_(coeff_solution[:-1].transpose(0, 1))
        self.basis_coeff.bias.copy_(coeff_solution[-1])

    def forward(self, coords: Tensor, design: Tensor) -> Tensor:
        """
        Predict the structural field on the fixed mesh.

        Args:
            coords (Tensor): Mesh coordinates. (B, N, D).
            design (Tensor): Design vectors. (B, C_DESIGN).

        Returns:
            Tensor: Predicted structural field. (B, N, C_OUT).
        """
        coarse_field = self.field_bias.unsqueeze(0) + torch.einsum(
            "bk,kno->bno",
            self.basis_coeff(design),
            self.basis_fields,
        )

        design_tokens = self.design_encoder(design).unsqueeze(1)
        coord_tokens = self.coord_encoder(coords)
        node_tokens = self.fusion(design_tokens * coord_tokens)

        for block in self.blocks:
            node_tokens = block(node_tokens)

        residual_field = self.residual_head(self.norm(node_tokens))
        return coarse_field + residual_field
