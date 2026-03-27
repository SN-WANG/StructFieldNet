"""Smoke tests for StructFieldNet model components."""

import unittest

import torch

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from structfieldnet.losses.field_loss import StructFieldLoss
from structfieldnet.models.structfieldnet import StructFieldNet


class StructFieldNetSmokeTest(unittest.TestCase):
    """Basic functional checks for model forward and loss evaluation."""

    def test_forward_and_loss(self) -> None:
        """Verify one forward and loss pass on synthetic tensors."""
        batch_size = 2
        num_nodes = 64
        coord_dim = 3
        design_dim = 25

        model = StructFieldNet(
            coord_dim=coord_dim,
            design_dim=design_dim,
            output_dim=1,
            hidden_dim=64,
            branch_hidden_dim=64,
            branch_num_layers=2,
            trunk_hidden_dim=64,
            trunk_num_layers=2,
            fusion_hidden_dim=64,
            fusion_num_layers=1,
            depth=2,
            num_heads=4,
            num_slices=16,
            mlp_ratio=2,
            dropout=0.0,
        )
        criterion = StructFieldLoss()

        coords = torch.randn(batch_size, num_nodes, coord_dim)
        design = torch.randn(batch_size, design_dim)
        target = torch.randn(batch_size, num_nodes, 1)

        pred = model(coords, design)
        loss_dict = criterion(pred, target)

        self.assertEqual(pred.shape, (batch_size, num_nodes, 1))
        self.assertIn("loss", loss_dict)
        self.assertTrue(torch.isfinite(loss_dict["loss"]))


if __name__ == "__main__":
    unittest.main()
