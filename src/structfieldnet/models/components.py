# Core neural building blocks for StructFieldNet
# Author: Shengning Wang

from typing import Optional

import torch
from torch import nn, Tensor


def _trunc_normal_(tensor: Tensor, std: float = 0.02) -> Tensor:
    """Initialize a tensor with truncated normal samples.

    Args:
        tensor: Tensor to initialize in-place.
        std: Standard deviation of the normal distribution.

    Returns:
        Initialized tensor.
    """
    with torch.no_grad():
        tensor.normal_(0.0, std)
        while True:
            mask = tensor.abs() > 2.0 * std
            if not mask.any():
                break
            tensor[mask] = torch.empty_like(tensor[mask]).normal_(0.0, std)
    return tensor


class PointwiseMLP(nn.Module):
    """Pointwise MLP operating on the last tensor dimension."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the pointwise MLP.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden feature dimension.
            output_dim: Output feature dimension.
            num_layers: Number of linear layers.
            dropout: Dropout probability.
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        layers = []
        in_dim = input_dim
        for layer_idx in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the MLP to the last tensor dimension.

        Args:
            x: Input tensor of shape `(..., input_dim)`.

        Returns:
            Output tensor of shape `(..., output_dim)`.
        """
        return self.net(x)


class PhysicsAttention(nn.Module):
    """Physics-aware slice attention for irregular meshes."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        num_slices: int = 64,
    ) -> None:
        """Initialize the physics-attention module.

        Args:
            dim: Hidden feature dimension.
            num_heads: Number of attention heads.
            dim_head: Feature dimension per head.
            dropout: Dropout probability.
            num_slices: Number of slice tokens.
        """
        super().__init__()
        inner_dim = dim_head * num_heads
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, num_slices)
        nn.init.orthogonal_(self.in_project_slice.weight)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        """Apply physics-attention on node features.

        Args:
            x: Node features of shape `(B, N, C)`.

        Returns:
            Updated node features of shape `(B, N, C)`.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape (B, N, C), got {tuple(x.shape)}")

        batch_size, num_nodes, _ = x.shape
        num_heads, dim_head = self.num_heads, self.dim_head

        fx_mid = self.in_project_fx(x).reshape(batch_size, num_nodes, num_heads, dim_head).permute(0, 2, 1, 3)
        x_mid = self.in_project_x(x).reshape(batch_size, num_nodes, num_heads, dim_head).permute(0, 2, 1, 3)

        temperature = torch.clamp(self.temperature, min=0.1, max=5.0)
        slice_weights = self.softmax(self.in_project_slice(x_mid) / temperature)

        slice_norm = slice_weights.sum(dim=2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm.unsqueeze(-1) + 1e-5)

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention_map = self.softmax(attention_scores)
        attention_map = self.dropout(attention_map)
        out_slice = torch.matmul(attention_map, v)

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = out_x.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, num_heads * dim_head)
        return self.to_out(out_x)


class StructFieldBlock(nn.Module):
    """Pre-normalized Physics-Attention block for StructFieldNet."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_slices: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the block.

        Args:
            hidden_dim: Hidden feature dimension.
            num_heads: Number of attention heads.
            num_slices: Number of slice tokens.
            mlp_ratio: Expansion ratio in the pointwise MLP.
            dropout: Dropout probability.
        """
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            num_slices=num_slices,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = PointwiseMLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * mlp_ratio,
            output_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply one StructField block.

        Args:
            x: Hidden node features of shape `(B, N, H)`.

        Returns:
            Updated hidden node features of shape `(B, N, H)`.
        """
        v = x + self.attn(self.ln_1(x))
        return v + self.mlp(self.ln_2(v))
