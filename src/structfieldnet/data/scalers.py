# Tensor scalers for StructFieldNet
# Author: Shengning Wang

from typing import Dict, Optional

import torch
from torch import Tensor


class IdentityScaler:
    """Identity transform used when normalization is disabled."""

    def fit(self, x: Tensor, channel_dim: int = -1) -> "IdentityScaler":
        """Return the scaler itself without changing state.

        Args:
            x: Input tensor.
            channel_dim: Unused channel dimension argument.

        Returns:
            Self instance.
        """
        _ = x
        _ = channel_dim
        return self

    def transform(self, x: Tensor) -> Tensor:
        """Return the input tensor unchanged.

        Args:
            x: Input tensor.

        Returns:
            Unchanged tensor.
        """
        return x

    def inverse_transform(self, x: Tensor) -> Tensor:
        """Return the input tensor unchanged.

        Args:
            x: Input tensor.

        Returns:
            Unchanged tensor.
        """
        return x

    def state_dict(self) -> Dict[str, bool]:
        """Serialize the scaler state.

        Returns:
            Minimal state dictionary.
        """
        return {"identity": True}

    def load_state_dict(self, state_dict: Dict[str, bool]) -> None:
        """Load the scaler state.

        Args:
            state_dict: Stored scaler state.
        """
        _ = state_dict


class TensorStandardScaler:
    """Channel-wise standardization for tensors."""

    def __init__(self, eps: float = 1e-7) -> None:
        """Initialize the scaler.

        Args:
            eps: Small positive value to avoid division by zero.
        """
        self.eps = eps
        self.mean: Optional[Tensor] = None
        self.std: Optional[Tensor] = None
        self.channel_dim: Optional[int] = None
        self.channel_offset_from_end: Optional[int] = None

    def fit(self, x: Tensor, channel_dim: int = -1) -> "TensorStandardScaler":
        """Fit channel-wise mean and standard deviation.

        Args:
            x: Input tensor of shape `(..., C)`.
            channel_dim: Channel dimension index.

        Returns:
            Self instance.
        """
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        if x.numel() == 0:
            raise ValueError("Input tensor cannot be empty.")

        self.channel_dim = channel_dim % x.dim()
        self.channel_offset_from_end = x.dim() - self.channel_dim - 1
        reduce_dims = tuple(dim for dim in range(x.dim()) if dim != self.channel_dim)
        self.mean = x.mean(dim=reduce_dims, keepdim=False)
        self.std = x.std(dim=reduce_dims, keepdim=False)
        self.std = torch.where(self.std < self.eps, torch.ones_like(self.std), self.std)
        return self

    def _reshape_parameter(self, parameter: Tensor, x: Tensor) -> Tensor:
        """Reshape a fitted parameter for broadcasting on a new tensor.

        Args:
            parameter: Stored channel-wise statistic of shape `(C,)`.
            x: Input tensor to broadcast against.

        Returns:
            Reshaped parameter tensor compatible with `x`.
        """
        if self.channel_offset_from_end is None:
            raise RuntimeError("Scaler must be fitted before broadcasting parameters.")
        channel_dim = x.dim() - self.channel_offset_from_end - 1
        if channel_dim < 0 or channel_dim >= x.dim():
            raise ValueError(
                f"Input tensor with shape {tuple(x.shape)} is incompatible with stored channel layout."
            )
        view_shape = [1] * x.dim()
        view_shape[channel_dim] = parameter.numel()
        return parameter.view(*view_shape).to(x.device)

    def transform(self, x: Tensor) -> Tensor:
        """Apply standardization.

        Args:
            x: Input tensor with the same channel layout as the fitted data.

        Returns:
            Standardized tensor.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fitted before calling transform.")
        mean = self._reshape_parameter(self.mean, x)
        std = self._reshape_parameter(self.std, x)
        return (x - mean) / std

    def inverse_transform(self, x: Tensor) -> Tensor:
        """Restore the original scale.

        Args:
            x: Standardized tensor.

        Returns:
            Tensor in the original data scale.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fitted before calling inverse_transform.")
        mean = self._reshape_parameter(self.mean, x)
        std = self._reshape_parameter(self.std, x)
        return x * std + mean

    def state_dict(self) -> Dict[str, Tensor]:
        """Serialize scaler parameters.

        Returns:
            State dictionary.
        """
        if self.mean is None or self.std is None or self.channel_dim is None:
            raise RuntimeError("Scaler must be fitted before calling state_dict.")
        return {
            "mean": self.mean.cpu(),
            "std": self.std.cpu(),
            "channel_dim": torch.tensor(self.channel_dim, dtype=torch.int64),
            "channel_offset_from_end": torch.tensor(self.channel_offset_from_end, dtype=torch.int64),
        }

    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        """Restore scaler parameters from a state dictionary.

        Args:
            state_dict: Saved scaler state.
        """
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]
        self.channel_dim = int(state_dict["channel_dim"].item())
        self.channel_offset_from_end = int(state_dict["channel_offset_from_end"].item())


class TensorMinMaxScaler:
    """Channel-wise min-max normalization for tensors."""

    def __init__(self, norm_range: str = "unit", eps: float = 1e-7) -> None:
        """Initialize the scaler.

        Args:
            norm_range: Either `"unit"` for `[0, 1]` or `"bipolar"` for `[-1, 1]`.
            eps: Small positive value to avoid division by zero.
        """
        if norm_range not in {"unit", "bipolar"}:
            raise ValueError(f"Unknown normalization range: {norm_range}")

        self.norm_range = norm_range
        self.eps = eps
        self.channel_dim: Optional[int] = None
        self.channel_offset_from_end: Optional[int] = None
        self.data_min: Optional[Tensor] = None
        self.data_max: Optional[Tensor] = None
        self.a, self.b = (0.0, 1.0) if norm_range == "unit" else (-1.0, 1.0)

    def fit(self, x: Tensor, channel_dim: int = -1) -> "TensorMinMaxScaler":
        """Fit channel-wise minima and maxima.

        Args:
            x: Input tensor of shape `(..., C)`.
            channel_dim: Channel dimension index.

        Returns:
            Self instance.
        """
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        if x.numel() == 0:
            raise ValueError("Input tensor cannot be empty.")

        self.channel_dim = channel_dim % x.dim()
        self.channel_offset_from_end = x.dim() - self.channel_dim - 1
        reduce_dims = tuple(dim for dim in range(x.dim()) if dim != self.channel_dim)
        self.data_min = x.amin(dim=reduce_dims, keepdim=False)
        self.data_max = x.amax(dim=reduce_dims, keepdim=False)
        return self

    def _reshape_parameter(self, parameter: Tensor, x: Tensor) -> Tensor:
        """Reshape a fitted parameter for broadcasting on a new tensor.

        Args:
            parameter: Stored channel-wise statistic of shape `(C,)`.
            x: Input tensor to broadcast against.

        Returns:
            Reshaped parameter tensor compatible with `x`.
        """
        if self.channel_offset_from_end is None:
            raise RuntimeError("Scaler must be fitted before broadcasting parameters.")
        channel_dim = x.dim() - self.channel_offset_from_end - 1
        if channel_dim < 0 or channel_dim >= x.dim():
            raise ValueError(
                f"Input tensor with shape {tuple(x.shape)} is incompatible with stored channel layout."
            )
        view_shape = [1] * x.dim()
        view_shape[channel_dim] = parameter.numel()
        return parameter.view(*view_shape).to(x.device)

    def transform(self, x: Tensor) -> Tensor:
        """Apply min-max normalization.

        Args:
            x: Input tensor with the same channel layout as the fitted data.

        Returns:
            Normalized tensor.
        """
        if self.data_min is None or self.data_max is None:
            raise RuntimeError("Scaler must be fitted before calling transform.")
        data_min = self._reshape_parameter(self.data_min, x)
        data_max = self._reshape_parameter(self.data_max, x)
        scale = torch.where(data_max - data_min < self.eps, torch.ones_like(data_max), data_max - data_min)
        return self.a + (x - data_min) * (self.b - self.a) / scale

    def inverse_transform(self, x: Tensor) -> Tensor:
        """Restore the original scale.

        Args:
            x: Normalized tensor.

        Returns:
            Tensor in the original data scale.
        """
        if self.data_min is None or self.data_max is None:
            raise RuntimeError("Scaler must be fitted before calling inverse_transform.")
        data_min = self._reshape_parameter(self.data_min, x)
        data_max = self._reshape_parameter(self.data_max, x)
        scale = torch.where(data_max - data_min < self.eps, torch.ones_like(data_max), data_max - data_min)
        return data_min + (x - self.a) * scale / (self.b - self.a)

    def state_dict(self) -> Dict[str, Tensor]:
        """Serialize scaler parameters.

        Returns:
            State dictionary.
        """
        if self.data_min is None or self.data_max is None or self.channel_dim is None:
            raise RuntimeError("Scaler must be fitted before calling state_dict.")
        return {
            "data_min": self.data_min.cpu(),
            "data_max": self.data_max.cpu(),
            "channel_dim": torch.tensor(self.channel_dim, dtype=torch.int64),
            "channel_offset_from_end": torch.tensor(self.channel_offset_from_end, dtype=torch.int64),
            "range_code": torch.tensor(0 if self.norm_range == "unit" else 1, dtype=torch.int64),
        }

    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        """Restore scaler parameters from a state dictionary.

        Args:
            state_dict: Saved scaler state.
        """
        self.data_min = state_dict["data_min"]
        self.data_max = state_dict["data_max"]
        self.channel_dim = int(state_dict["channel_dim"].item())
        self.channel_offset_from_end = int(state_dict["channel_offset_from_end"].item())
        self.norm_range = "unit" if int(state_dict["range_code"].item()) == 0 else "bipolar"
        self.a, self.b = (0.0, 1.0) if self.norm_range == "unit" else (-1.0, 1.0)
