# Trainer for StructFieldNet
# Author: Shengning Wang

from typing import Dict

from torch import Tensor

from training.base_trainer import BaseTrainer


class FieldTrainer(BaseTrainer):
    """
    Trainer for fixed-mesh structural field reconstruction.
    """

    def _compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Compute the supervised reconstruction loss.

        Args:
            batch (Dict[str, Tensor]): Batch dictionary with coords, design, and stress tensors.

        Returns:
            Tensor: Scalar reconstruction loss. ().
        """
        pred = self.model(batch["coords"], batch["design"])
        return self.criterion(pred, batch["stress"])
