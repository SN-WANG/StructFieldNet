"""Wing full-field stress datasets for StructFieldNet."""

import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from structfieldnet.data.scalers import IdentityScaler, TensorMinMaxScaler, TensorStandardScaler
from structfieldnet.utils.hue_logger import hue, logger


class WingStressDataset(Dataset):
    """Dataset of wing full-field stress reconstruction cases.

    Each sample is a dictionary containing:
        - `coords`: `(N, 3)`
        - `design`: `(M,)`
        - `stress`: `(N, 1)`
    """

    SI_PREFIX_CANDIDATES = (
        1e-12,
        1e-9,
        1e-6,
        1e-3,
        1.0,
        1e3,
        1e6,
        1e9,
        1e12,
    )

    def __init__(
        self,
        case_paths: Sequence[Path],
        verify_fixed_mesh: bool = True,
        harmonize_units: bool = True,
    ) -> None:
        """Load case tensors into memory.

        Args:
            case_paths: Sequence of `.pt` file paths.
            verify_fixed_mesh: Whether to verify that all cases share the same mesh.
            harmonize_units: Whether to harmonize mixed SI-prefix unit scales.
        """
        super().__init__()
        self.case_paths = [Path(path) for path in case_paths]
        self.samples: List[Dict[str, Tensor]] = [self._load_case(path) for path in self.case_paths]

        if not self.samples:
            raise ValueError("Dataset is empty. No case files were provided.")

        if harmonize_units:
            self._harmonize_units()
        if verify_fixed_mesh:
            self._verify_fixed_mesh()

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Return one case sample.

        Args:
            idx: Sample index.

        Returns:
            Sample dictionary.
        """
        sample = self.samples[idx]
        return {
            "coords": sample["coords"],
            "design": sample["design"],
            "stress": sample["stress"],
        }

    def _load_case(self, case_path: Path) -> Dict[str, Tensor]:
        """Load one `.pt` case file.

        Args:
            case_path: Path to the serialized case.

        Returns:
            Loaded and validated sample dictionary.

        Raises:
            FileNotFoundError: If the file does not exist.
            KeyError: If required keys are missing.
            ValueError: If tensor shapes are invalid.
        """
        if not case_path.exists():
            raise FileNotFoundError(f"Case file not found: {case_path}")

        sample = torch.load(case_path, map_location="cpu", weights_only=False)
        required_keys = {"coords", "design", "stress"}
        missing_keys = required_keys.difference(sample.keys())
        if missing_keys:
            raise KeyError(f"Missing keys in {case_path.name}: {sorted(missing_keys)}")

        coords = sample["coords"].to(torch.float32)
        design = sample["design"].to(torch.float32)
        stress = sample["stress"].to(torch.float32)

        if coords.dim() != 2 or coords.shape[-1] != 3:
            raise ValueError(f"coords must have shape (N, 3), got {tuple(coords.shape)}")
        if design.dim() != 1:
            raise ValueError(f"design must have shape (M,), got {tuple(design.shape)}")
        if stress.dim() != 2 or stress.shape[-1] != 1:
            raise ValueError(f"stress must have shape (N, 1), got {tuple(stress.shape)}")
        if coords.shape[0] != stress.shape[0]:
            raise ValueError(
                f"coords and stress must share the same node dimension, got {coords.shape[0]} and {stress.shape[0]}"
            )

        return {"coords": coords, "design": design, "stress": stress}

    def _verify_fixed_mesh(self) -> None:
        """Check that all samples share the same mesh coordinates.

        Raises:
            ValueError: If any case has a different mesh shape or coordinate values.
        """
        reference = self.samples[0]["coords"]
        for idx, sample in enumerate(self.samples[1:], start=1):
            coords = sample["coords"]
            if coords.shape != reference.shape:
                raise ValueError(
                    f"Case {self.case_paths[idx].name} has mesh shape {tuple(coords.shape)}, "
                    f"expected {tuple(reference.shape)}"
                )
            if not torch.allclose(coords, reference, atol=5e-6, rtol=0.0):
                raise ValueError(f"Case {self.case_paths[idx].name} does not share the fixed mesh coordinates.")

    @staticmethod
    def _compute_reference_scale(values: List[float]) -> float:
        """Compute a robust reference magnitude from case-wise maxima.

        Args:
            values: Positive case-wise magnitude estimates.

        Returns:
            Median magnitude used as the reference scale.
        """
        valid_values = [value for value in values if value > 0.0]
        if not valid_values:
            return 1.0
        return float(np.median(np.asarray(valid_values, dtype=np.float64)))

    @classmethod
    def _infer_unit_multiplier(cls, case_scale: float, reference_scale: float) -> float:
        """Infer the best SI-prefix multiplier to match the dataset reference scale.

        Args:
            case_scale: Characteristic magnitude of one case.
            reference_scale: Dataset-level reference magnitude.

        Returns:
            Multiplicative factor selected from SI-prefix candidates.
        """
        if case_scale <= 0.0 or reference_scale <= 0.0:
            return 1.0

        raw_gap = abs(math.log10(case_scale) - math.log10(reference_scale))
        best_multiplier = 1.0
        best_gap = raw_gap

        for multiplier in cls.SI_PREFIX_CANDIDATES:
            corrected_scale = case_scale * multiplier
            gap = abs(math.log10(corrected_scale) - math.log10(reference_scale))
            if gap < best_gap:
                best_gap = gap
                best_multiplier = multiplier

        if raw_gap - best_gap < 1.0:
            return 1.0
        return best_multiplier

    def _harmonize_units(self) -> None:
        """Harmonize mixed SI-prefix scales across cases.

        The ANSYS exports may occasionally mix length units (for example,
        millimeters vs. meters) or stress units (for example, MPa vs. Pa).
        This routine infers a dataset-level reference magnitude and rescales
        individual cases by SI-prefix factors only when the mismatch spans at
        least one full decade in log-space.
        """
        coord_scales = [float(sample["coords"].abs().max()) for sample in self.samples]
        stress_scales = [float(sample["stress"].abs().max()) for sample in self.samples]

        reference_coord_scale = self._compute_reference_scale(coord_scales)
        reference_stress_scale = self._compute_reference_scale(stress_scales)

        coord_adjustments: list[str] = []
        stress_adjustments: list[str] = []
        for case_path, sample, coord_scale, stress_scale in zip(
            self.case_paths,
            self.samples,
            coord_scales,
            stress_scales,
        ):
            coord_multiplier = self._infer_unit_multiplier(coord_scale, reference_coord_scale)
            if coord_multiplier != 1.0:
                sample["coords"] = sample["coords"] * coord_multiplier
                coord_adjustments.append(f"{case_path.stem} x {coord_multiplier:.0e}")

            stress_multiplier = self._infer_unit_multiplier(stress_scale, reference_stress_scale)
            if stress_multiplier != 1.0:
                sample["stress"] = sample["stress"] * stress_multiplier
                stress_adjustments.append(f"{case_path.stem} x {stress_multiplier:.0e}")

        if coord_adjustments:
            logger.warning(
                "harmonized coordinate units for "
                f"{hue.m}{len(coord_adjustments)}{hue.q} case(s): "
                f"{', '.join(coord_adjustments)}"
            )
        if stress_adjustments:
            logger.warning(
                "harmonized stress units for "
                f"{hue.m}{len(stress_adjustments)}{hue.q} case(s): "
                f"{', '.join(stress_adjustments)}"
            )


class ScaledWingStressDataset(Dataset):
    """Dataset wrapper that applies fitted scalers on-the-fly."""

    def __init__(self, dataset: WingStressDataset, scalers: Dict[str, object]) -> None:
        """Initialize the wrapper dataset.

        Args:
            dataset: Underlying raw dataset.
            scalers: Dictionary with `coords`, `design`, and `stress` scalers.
        """
        self.dataset = dataset
        self.scalers = scalers

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Return a normalized sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with normalized tensors.
        """
        sample = self.dataset[idx]
        return {
            "coords": self.scalers["coords"].transform(sample["coords"]),
            "design": self.scalers["design"].transform(sample["design"]),
            "stress": self.scalers["stress"].transform(sample["stress"]),
        }


def discover_case_paths(data_dir: Path) -> List[Path]:
    """Discover case files under the dataset directory.

    Args:
        data_dir: Dataset root containing `dp*.pt`.

    Returns:
        Sorted list of case paths.
    """
    case_paths = sorted(Path(data_dir).glob("dp*.pt"), key=lambda path: int(path.stem[2:]))
    if not case_paths:
        raise FileNotFoundError(f"No dp*.pt files were found in {data_dir}")
    return case_paths


def build_case_splits(
    data_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Path]]:
    """Split case files into train, validation, and test subsets.

    Args:
        data_dir: Dataset root directory.
        train_ratio: Fraction of training cases.
        val_ratio: Fraction of validation cases.
        test_ratio: Fraction of test cases.
        seed: Random seed for deterministic shuffling.

    Returns:
        Dictionary with `train`, `val`, and `test` path lists.

    Raises:
        ValueError: If split ratios are invalid.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.6f}")

    case_paths = discover_case_paths(data_dir)
    rng = np.random.default_rng(seed=seed)
    indices = np.arange(len(case_paths))
    rng.shuffle(indices)
    shuffled = [case_paths[idx] for idx in indices]

    num_total = len(shuffled)
    num_test = int(round(num_total * test_ratio))
    num_val = int(round(num_total * val_ratio))
    num_train = num_total - num_val - num_test

    splits = {
        "train": shuffled[:num_train],
        "val": shuffled[num_train:num_train + num_val],
        "test": shuffled[num_train + num_val:],
    }
    logger.info(
        "dataset split | "
        f"train: {hue.m}{len(splits['train'])}{hue.q} | "
        f"val: {hue.m}{len(splits['val'])}{hue.q} | "
        f"test: {hue.m}{len(splits['test'])}{hue.q}"
    )
    return splits


def fit_dataset_scalers(
    dataset: WingStressDataset,
    coords_norm_range: str = "bipolar",
    normalize_design: bool = True,
    normalize_stress: bool = True,
) -> Dict[str, object]:
    """Fit dataset scalers from the training subset.

    Args:
        dataset: Training dataset.
        coords_norm_range: Coordinate normalization range.
        normalize_design: Whether to standardize design vectors.
        normalize_stress: Whether to standardize stress targets.

    Returns:
        Dictionary of fitted scalers.
    """
    coords_tensor = torch.cat([sample["coords"] for sample in dataset.samples], dim=0)
    design_tensor = torch.stack([sample["design"] for sample in dataset.samples], dim=0)
    stress_tensor = torch.stack([sample["stress"] for sample in dataset.samples], dim=0)

    coords_scaler = TensorMinMaxScaler(norm_range=coords_norm_range).fit(coords_tensor, channel_dim=-1)
    design_scaler = TensorStandardScaler().fit(design_tensor, channel_dim=-1) if normalize_design else IdentityScaler()
    stress_scaler = TensorStandardScaler().fit(stress_tensor, channel_dim=-1) if normalize_stress else IdentityScaler()

    logger.info(
        f"fitted scalers | coords: {hue.m}{coords_norm_range}{hue.q} | "
        f"design: {hue.m}{'standard' if normalize_design else 'identity'}{hue.q} | "
        f"stress: {hue.m}{'standard' if normalize_stress else 'identity'}{hue.q}"
    )
    return {"coords": coords_scaler, "design": design_scaler, "stress": stress_scaler}
