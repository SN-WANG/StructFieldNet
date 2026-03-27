"""Shape-preserving tests for dataset scalers."""

import unittest

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from structfieldnet.data.scalers import TensorMinMaxScaler, TensorStandardScaler


class ScalerShapeTest(unittest.TestCase):
    """Verify that fitted scalers preserve single-sample tensor ranks."""

    def test_standard_scaler_preserves_design_shape(self) -> None:
        """Check standardization on batched and unbatched design tensors."""
        train_tensor = torch.randn(8, 25)
        sample_tensor = torch.randn(25)

        scaler = TensorStandardScaler().fit(train_tensor, channel_dim=-1)
        transformed = scaler.transform(sample_tensor)
        restored = scaler.inverse_transform(transformed)

        self.assertEqual(transformed.shape, sample_tensor.shape)
        self.assertEqual(restored.shape, sample_tensor.shape)

    def test_minmax_scaler_preserves_coord_shape(self) -> None:
        """Check min-max normalization on batched and unbatched coordinate tensors."""
        train_tensor = torch.randn(8, 64, 3)
        sample_tensor = torch.randn(64, 3)

        scaler = TensorMinMaxScaler(norm_range="bipolar").fit(train_tensor, channel_dim=-1)
        transformed = scaler.transform(sample_tensor)
        restored = scaler.inverse_transform(transformed)

        self.assertEqual(transformed.shape, sample_tensor.shape)
        self.assertEqual(restored.shape, sample_tensor.shape)


if __name__ == "__main__":
    unittest.main()
