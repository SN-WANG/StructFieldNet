# Scalers for standardization and normalization
# Author: Shengning Wang

from typing import Dict, Literal, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    from torch import Tensor

    _HAS_TORCH = True
else:
    try:
        import torch
        from torch import Tensor

        _HAS_TORCH = True
    except ImportError:  # pragma: no cover - torch is an explicit dependency
        torch = None
        Tensor = None
        _HAS_TORCH = False


class BaseScaler:
    """Base class for array scalers."""

    def fit(self, x: np.ndarray, channel_dim: int = -1) -> "BaseScaler":
        raise NotImplementedError

    def transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        raise NotImplementedError


class IdentityScalerNP(BaseScaler):
    """No-op NumPy scaler."""

    def fit(self, x: np.ndarray, channel_dim: int = -1) -> "IdentityScalerNP":
        _ = channel_dim
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"identity": np.array(1, dtype=np.int64)}

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        _ = state_dict


def _squeeze_leading_singletons_np(x: np.ndarray, target_ndim: int) -> np.ndarray:
    """Restore the original sample rank after NumPy broadcasting."""
    while x.ndim > target_ndim and x.shape[0] == 1:
        x = np.squeeze(x, axis=0)
    return x


class StandardScalerNP(BaseScaler):
    """Channel-wise NumPy standardization scaler."""

    def __init__(self, eps: float = 1e-7):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.eps = eps
        self.channel_dim: Optional[int] = None

    def fit(self, x: np.ndarray, channel_dim: int = -1) -> "StandardScalerNP":
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a NumPy array.")

        self.channel_dim = channel_dim % x.ndim
        reduce_dims = tuple(dim for dim in range(x.ndim) if dim != self.channel_dim)

        self.mean = np.mean(x, axis=reduce_dims, keepdims=True)
        self.std = np.std(x, axis=reduce_dims, keepdims=True)
        self.std[self.std < self.eps] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")
        transformed = (x - self.mean) / self.std
        return _squeeze_leading_singletons_np(transformed, x.ndim)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")
        restored = x * self.std + self.mean
        return _squeeze_leading_singletons_np(restored, x.ndim)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "mean": self.mean,
            "std": self.std,
            "channel_dim": np.array(self.channel_dim, dtype=np.int64),
        }

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]
        self.channel_dim = int(state_dict["channel_dim"])


class MinMaxScalerNP(BaseScaler):
    """Channel-wise NumPy min-max scaler."""

    def __init__(self, norm_range: Literal["unit", "bipolar"] = "unit", eps: float = 1e-7):
        self.norm_range = norm_range
        self.eps = eps
        self.data_min: Optional[np.ndarray] = None
        self.data_max: Optional[np.ndarray] = None
        self.channel_dim: Optional[int] = None

        if norm_range == "unit":
            self.a, self.b = 0.0, 1.0
        elif norm_range == "bipolar":
            self.a, self.b = -1.0, 1.0
        else:
            raise ValueError("Invalid norm_range. Must be 'unit' or 'bipolar'.")

    def fit(self, x: np.ndarray, channel_dim: int = -1) -> "MinMaxScalerNP":
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a NumPy array.")

        self.channel_dim = channel_dim % x.ndim
        reduce_dims = tuple(dim for dim in range(x.ndim) if dim != self.channel_dim)
        self.data_min = np.min(x, axis=reduce_dims, keepdims=True)
        self.data_max = np.max(x, axis=reduce_dims, keepdims=True)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.data_min is None or self.data_max is None:
            raise RuntimeError("Scaler has not been fitted.")

        scale = self.data_max - self.data_min
        scale[scale < self.eps] = 1.0
        transformed = self.a + (x - self.data_min) * (self.b - self.a) / scale
        return _squeeze_leading_singletons_np(transformed, x.ndim)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.data_min is None or self.data_max is None:
            raise RuntimeError("Scaler has not been fitted.")

        scale = self.data_max - self.data_min
        scale[scale < self.eps] = 1.0
        restored = (x - self.a) * scale / (self.b - self.a) + self.data_min
        return _squeeze_leading_singletons_np(restored, x.ndim)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "data_min": self.data_min,
            "data_max": self.data_max,
            "channel_dim": np.array(self.channel_dim, dtype=np.int64),
            "norm_range": np.array(0 if self.norm_range == "unit" else 1, dtype=np.int64),
        }

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        self.data_min = state_dict["data_min"]
        self.data_max = state_dict["data_max"]
        self.channel_dim = int(state_dict["channel_dim"])
        self.norm_range = "unit" if int(state_dict["norm_range"]) == 0 else "bipolar"
        self.a, self.b = (0.0, 1.0) if self.norm_range == "unit" else (-1.0, 1.0)


if _HAS_TORCH:

    def _squeeze_leading_singletons_tensor(x: Tensor, target_ndim: int) -> Tensor:
        """Restore the original sample rank after tensor broadcasting."""
        while x.ndim > target_ndim and x.shape[0] == 1:
            x = x.squeeze(0)
        return x

    class IdentityScalerTensor:
        """No-op tensor scaler."""

        def fit(self, x: Tensor, channel_dim: int = -1) -> "IdentityScalerTensor":
            _ = channel_dim
            if not isinstance(x, torch.Tensor):
                raise TypeError("Input must be a PyTorch Tensor.")
            return self

        def transform(self, x: Tensor) -> Tensor:
            return x

        def inverse_transform(self, x: Tensor) -> Tensor:
            return x

        def state_dict(self) -> Dict[str, Tensor]:
            return {"identity": torch.tensor(1, dtype=torch.int64)}

        def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
            _ = state_dict


    class StandardScalerTensor:
        """Channel-wise PyTorch standardization scaler."""

        def __init__(self, eps: float = 1e-7):
            self.mean: Optional[Tensor] = None
            self.std: Optional[Tensor] = None
            self.eps = eps
            self.channel_dim: Optional[int] = None

        def fit(self, x: Tensor, channel_dim: int = -1) -> "StandardScalerTensor":
            if not isinstance(x, torch.Tensor):
                raise TypeError("Input must be a PyTorch Tensor.")

            self.channel_dim = channel_dim % x.ndim
            reduce_dims = [dim for dim in range(x.ndim) if dim != self.channel_dim]
            self.mean = torch.mean(x, dim=reduce_dims, keepdim=True)
            self.std = torch.std(x, dim=reduce_dims, keepdim=True)
            self.std = torch.where(self.std < self.eps, torch.ones_like(self.std), self.std)
            return self

        def transform(self, x: Tensor) -> Tensor:
            if self.mean is None or self.std is None:
                raise RuntimeError("Scaler has not been fitted.")

            if self.mean.device != x.device:
                self.mean = self.mean.to(x.device)
                self.std = self.std.to(x.device)
            transformed = (x - self.mean) / self.std
            return _squeeze_leading_singletons_tensor(transformed, x.ndim)

        def inverse_transform(self, x: Tensor) -> Tensor:
            if self.mean is None or self.std is None:
                raise RuntimeError("Scaler has not been fitted.")

            if self.mean.device != x.device:
                self.mean = self.mean.to(x.device)
                self.std = self.std.to(x.device)
            restored = x * self.std + self.mean
            return _squeeze_leading_singletons_tensor(restored, x.ndim)

        def state_dict(self) -> Dict[str, Tensor]:
            return {
                "mean": self.mean,
                "std": self.std,
                "channel_dim": torch.tensor(self.channel_dim, dtype=torch.int64),
            }

        def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
            self.mean = state_dict["mean"]
            self.std = state_dict["std"]
            self.channel_dim = int(state_dict["channel_dim"].item())


    class MinMaxScalerTensor:
        """Channel-wise PyTorch min-max scaler."""

        def __init__(self, norm_range: Literal["unit", "bipolar"] = "unit", eps: float = 1e-7):
            self.norm_range = norm_range
            self.eps = eps
            self.data_min: Optional[Tensor] = None
            self.data_max: Optional[Tensor] = None
            self.channel_dim: Optional[int] = None

            if norm_range == "unit":
                self.a, self.b = 0.0, 1.0
            elif norm_range == "bipolar":
                self.a, self.b = -1.0, 1.0
            else:
                raise ValueError("Invalid norm_range. Must be 'unit' or 'bipolar'.")

        def fit(self, x: Tensor, channel_dim: int = -1) -> "MinMaxScalerTensor":
            if not isinstance(x, torch.Tensor):
                raise TypeError("Input must be a PyTorch Tensor.")

            self.channel_dim = channel_dim % x.ndim
            reduce_dims = [dim for dim in range(x.ndim) if dim != self.channel_dim]

            self.data_min = torch.min(x, dim=reduce_dims[0], keepdim=True)[0]
            self.data_max = torch.max(x, dim=reduce_dims[0], keepdim=True)[0]

            for dim in reduce_dims[1:]:
                self.data_min = torch.min(self.data_min, dim=dim, keepdim=True)[0]
                self.data_max = torch.max(self.data_max, dim=dim, keepdim=True)[0]
            return self

        def transform(self, x: Tensor) -> Tensor:
            if self.data_min is None or self.data_max is None:
                raise RuntimeError("Scaler has not been fitted.")

            if self.data_min.device != x.device:
                self.data_min = self.data_min.to(x.device)
                self.data_max = self.data_max.to(x.device)

            scale = self.data_max - self.data_min
            scale = torch.where(scale < self.eps, torch.ones_like(scale), scale)
            transformed = self.a + (x - self.data_min) * (self.b - self.a) / scale
            return _squeeze_leading_singletons_tensor(transformed, x.ndim)

        def inverse_transform(self, x: Tensor) -> Tensor:
            if self.data_min is None or self.data_max is None:
                raise RuntimeError("Scaler has not been fitted.")

            if self.data_min.device != x.device:
                self.data_min = self.data_min.to(x.device)
                self.data_max = self.data_max.to(x.device)

            scale = self.data_max - self.data_min
            scale = torch.where(scale < self.eps, torch.ones_like(scale), scale)
            restored = (x - self.a) * scale / (self.b - self.a) + self.data_min
            return _squeeze_leading_singletons_tensor(restored, x.ndim)

        def state_dict(self) -> Dict[str, Tensor]:
            return {
                "data_min": self.data_min,
                "data_max": self.data_max,
                "channel_dim": torch.tensor(self.channel_dim, dtype=torch.int64),
                "norm_range": torch.tensor(0 if self.norm_range == "unit" else 1, dtype=torch.int64),
            }

        def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
            self.data_min = state_dict["data_min"]
            self.data_max = state_dict["data_max"]
            self.channel_dim = int(state_dict["channel_dim"].item())
            self.norm_range = "unit" if int(state_dict["norm_range"].item()) == 0 else "bipolar"
            self.a, self.b = (0.0, 1.0) if self.norm_range == "unit" else (-1.0, 1.0)
