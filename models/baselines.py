# Classical baselines for fixed-mesh structural field reconstruction
# Author: Shengning Wang

from __future__ import annotations

import torch
from torch import Tensor


class MeanFieldBaseline:
    """
    Training-mean field baseline.
    """

    name = "mean_field"

    def fit(self, design: Tensor, stress: Tensor) -> "MeanFieldBaseline":
        """
        Fit the mean stress field from training samples.

        Args:
            design (Tensor): Design matrix. (B, C_DESIGN).
            stress (Tensor): Stress fields. (B, N, C_OUT).

        Returns:
            MeanFieldBaseline: Fitted baseline.
        """
        self.mean_field = stress.mean(dim=0)
        return self

    def predict(self, design: Tensor) -> Tensor:
        """
        Predict stress fields for query designs.

        Args:
            design (Tensor): Query design matrix. (B, C_DESIGN).

        Returns:
            Tensor: Predicted stress fields. (B, N, C_OUT).
        """
        return self.mean_field.unsqueeze(0).expand(design.shape[0], -1, -1)


class DesignNearestNeighborBaseline:
    """
    Design-space nearest-neighbor field baseline.
    """

    name = "design_nn"

    def fit(self, design: Tensor, stress: Tensor) -> "DesignNearestNeighborBaseline":
        """
        Store training designs and stress fields.

        Args:
            design (Tensor): Design matrix. (B, C_DESIGN).
            stress (Tensor): Stress fields. (B, N, C_OUT).

        Returns:
            DesignNearestNeighborBaseline: Fitted baseline.
        """
        self.train_design = design
        self.train_stress = stress
        return self

    def predict(self, design: Tensor) -> Tensor:
        """
        Predict stress fields by nearest design vectors.

        Args:
            design (Tensor): Query design matrix. (B, C_DESIGN).

        Returns:
            Tensor: Predicted stress fields. (B, N, C_OUT).
        """
        distances = torch.cdist(design, self.train_design)
        indices = torch.argmin(distances, dim=1)
        return self.train_stress[indices]


class PCALinearBaseline:
    """
    PCA basis plus linear design-to-coefficient baseline.
    """

    name = "pca_linear"

    def __init__(self, num_bases: int) -> None:
        """
        Initialize the PCA-linear baseline.

        Args:
            num_bases (int): Number of retained SVD basis fields.
        """
        self.num_bases = num_bases

    def fit(self, design: Tensor, stress: Tensor) -> "PCALinearBaseline":
        """
        Fit SVD basis fields and a least-squares coefficient map.

        Args:
            design (Tensor): Design matrix. (B, C_DESIGN).
            stress (Tensor): Stress fields. (B, N, C_OUT).

        Returns:
            PCALinearBaseline: Fitted baseline.
        """
        self.num_nodes = stress.shape[1]
        self.output_dim = stress.shape[2]

        flat_stress = stress.reshape(stress.shape[0], -1)
        self.field_mean = flat_stress.mean(dim=0, keepdim=True)
        centered_stress = flat_stress - self.field_mean

        _, _, right_vectors = torch.linalg.svd(centered_stress, full_matrices=False)
        self.basis_vectors = right_vectors[: self.num_bases]
        basis_coeff = centered_stress @ self.basis_vectors.transpose(0, 1)

        bias_column = torch.ones(design.shape[0], 1, dtype=design.dtype, device=design.device)
        design_aug = torch.cat([design, bias_column], dim=1)
        self.coeff_solution = torch.linalg.lstsq(design_aug, basis_coeff).solution
        return self

    def predict(self, design: Tensor) -> Tensor:
        """
        Predict stress fields from query design vectors.

        Args:
            design (Tensor): Query design matrix. (B, C_DESIGN).

        Returns:
            Tensor: Predicted stress fields. (B, N, C_OUT).
        """
        bias_column = torch.ones(design.shape[0], 1, dtype=design.dtype, device=design.device)
        design_aug = torch.cat([design, bias_column], dim=1)
        basis_coeff = design_aug @ self.coeff_solution
        flat_pred = self.field_mean + basis_coeff @ self.basis_vectors
        return flat_pred.reshape(design.shape[0], self.num_nodes, self.output_dim)
