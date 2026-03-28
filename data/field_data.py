# Dataset utilities for StructFieldNet
# Author: Shengning Wang

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils.hue_logger import hue, logger
from utils.scaler import (
    MinMaxScalerTensor,
    StandardScalerTensor,
)


class FieldData(Dataset):
    """Dataset for static structural field reconstruction samples."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        case_names: Sequence[str],
        verify_fixed_mesh: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.case_names = list(case_names)

        self.coords: List[Tensor] = []
        self.designs: List[Tensor] = []
        self.stresses: List[Tensor] = []

        logger.info(f"initializing field dataset with {hue.m}{len(self.case_names)}{hue.q} cases...")
        progress = tqdm(
            self.case_names,
            desc="[FieldData] loading",
            leave=False,
            dynamic_ncols=True,
        )
        for case_name in progress:
            payload = self._load_case(case_name)
            self.coords.append(payload["coords"])
            self.designs.append(payload["design"])
            self.stresses.append(payload["stress"])

        if not self.case_names:
            raise ValueError("No cases were provided to FieldData.")

        if verify_fixed_mesh:
            self._verify_fixed_mesh()

        self.reference_coords = self.coords[0]
        self.coord_dim = int(self.reference_coords.shape[-1])
        self.design_dim = int(self.designs[0].shape[-1])

        logger.info(
            f"{hue.g}dataset initialized.{hue.q} "
            f"cases: {hue.m}{len(self.case_names)}{hue.q}, "
            f"nodes: {hue.m}{self.reference_coords.shape[0]}{hue.q}, "
            f"coord_dim: {hue.m}{self.coord_dim}{hue.q}, "
            f"design_dim: {hue.m}{self.design_dim}{hue.q}"
        )

    def __len__(self) -> int:
        return len(self.case_names)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, str]]:
        return {
            "coords": self.coords[idx],
            "design": self.designs[idx],
            "stress": self.stresses[idx],
            "case_name": self.case_names[idx],
        }

    def _load_case(self, case_name: str) -> Dict[str, Tensor]:
        case_path = self.data_dir / f"{case_name}.pt"
        if not case_path.exists():
            raise FileNotFoundError(f"Case file not found: {case_path}")

        payload = torch.load(case_path, map_location="cpu")
        required_keys = {"coords", "design", "stress"}
        missing_keys = required_keys.difference(payload.keys())
        if missing_keys:
            raise KeyError(f"Missing keys in {case_path.name}: {sorted(missing_keys)}")

        coords = payload["coords"].to(torch.float32)
        design = payload["design"].to(torch.float32)
        stress = payload["stress"].to(torch.float32)

        if coords.dim() != 2:
            raise ValueError(f"coords must have shape (N, C), got {tuple(coords.shape)}")
        if design.dim() != 1:
            raise ValueError(f"design must have shape (M,), got {tuple(design.shape)}")
        if stress.dim() != 2 or stress.shape[-1] != 1:
            raise ValueError(f"stress must have shape (N, 1), got {tuple(stress.shape)}")
        if coords.shape[0] != stress.shape[0]:
            raise ValueError(
                f"coords and stress must share node dimension, got {coords.shape[0]} and {stress.shape[0]}"
            )
        if torch.isnan(coords).any() or torch.isnan(design).any() or torch.isnan(stress).any():
            raise ValueError(f"NaN detected in case {case_name}")

        return {"coords": coords, "design": design, "stress": stress}

    def _verify_fixed_mesh(self, atol: float = 1e-6) -> None:
        reference = self.coords[0]
        for case_name, coords in zip(self.case_names[1:], self.coords[1:]):
            if coords.shape != reference.shape:
                raise ValueError(
                    f"Case {case_name} has mesh shape {tuple(coords.shape)}, expected {tuple(reference.shape)}"
                )
            if not torch.allclose(coords, reference, atol=atol, rtol=0.0):
                raise ValueError(f"Case {case_name} does not share the fixed mesh coordinates.")

    def stack_tensors(self) -> Dict[str, Tensor]:
        """Stack all samples for scaler fitting."""
        return {
            "coords": torch.stack(self.coords, dim=0),
            "design": torch.stack(self.designs, dim=0),
            "stress": torch.stack(self.stresses, dim=0),
        }

    @staticmethod
    def discover_cases(data_dir: Union[str, Path] = "./dataset", prefix: str = "dp") -> List[str]:
        """Discover case names such as ``dp1`` and ``dp200``."""
        data_dir = Path(data_dir)
        case_paths = sorted(
            data_dir.glob(f"{prefix}*.pt"),
            key=lambda path: int(path.stem[len(prefix) :]),
        )
        if not case_paths:
            raise FileNotFoundError(f"No {prefix}*.pt files were found in {data_dir}")
        return [path.stem for path in case_paths]

    @staticmethod
    def build_case_splits(
        data_dir: Union[str, Path] = "./dataset",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Dict[str, List[str]]:
        """Build deterministic train/validation/test splits."""
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.6f}")

        case_names = FieldData.discover_cases(data_dir)
        rng = np.random.default_rng(seed=seed)
        shuffled = list(case_names)
        rng.shuffle(shuffled)

        num_total = len(shuffled)
        num_test = int(round(num_total * test_ratio))
        num_val = int(round(num_total * val_ratio))

        if test_ratio > 0.0 and num_test == 0 and num_total >= 3:
            num_test = 1
        if val_ratio > 0.0 and num_val == 0 and num_total - num_test >= 2:
            num_val = 1

        num_train = num_total - num_val - num_test
        if num_train <= 0:
            overflow = 1 - num_train
            if num_val >= num_test and num_val > 0:
                reduction = min(num_val, overflow)
                num_val -= reduction
                overflow -= reduction
            if overflow > 0 and num_test > 0:
                reduction = min(num_test, overflow)
                num_test -= reduction
                overflow -= reduction
            num_train = num_total - num_val - num_test

        if num_train <= 0:
            raise ValueError("Training split is empty. Adjust the split ratios.")

        return {
            "train": shuffled[:num_train],
            "val": shuffled[num_train : num_train + num_val],
            "test": shuffled[num_train + num_val :],
        }

    @staticmethod
    def save_split_manifest(split_dict: Dict[str, List[str]], output_path: Union[str, Path]) -> None:
        """Save split case names to disk."""
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(split_dict, file, indent=2)

    @staticmethod
    def load_split_manifest(input_path: Union[str, Path]) -> Dict[str, List[str]]:
        """Load split case names from disk."""
        with open(input_path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def spawn(
        data_dir: Union[str, Path] = "./dataset",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        verify_fixed_mesh: bool = True,
        split_manifest: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple["FieldData", "FieldData", "FieldData", Dict[str, List[str]]]:
        """Create train, validation, and test datasets."""
        if split_manifest is None:
            split_manifest = FieldData.build_case_splits(
                data_dir=data_dir,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )

        train_dataset = FieldData(data_dir, split_manifest["train"], verify_fixed_mesh=verify_fixed_mesh)
        val_dataset = FieldData(data_dir, split_manifest["val"], verify_fixed_mesh=verify_fixed_mesh)
        test_dataset = FieldData(data_dir, split_manifest["test"], verify_fixed_mesh=verify_fixed_mesh)
        return train_dataset, val_dataset, test_dataset, split_manifest


class ScaledFieldDataset(Dataset):
    """Dataset wrapper applying fitted scalers on the fly."""

    def __init__(self, dataset: FieldData, scalers: Dict[str, object]) -> None:
        self.dataset = dataset
        self.scalers = scalers

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, str]]:
        sample = self.dataset[idx]
        coord_scaler = self.scalers.get("coord_scaler")
        design_scaler = self.scalers.get("design_scaler")
        stress_scaler = self.scalers.get("stress_scaler")
        return {
            "coords": coord_scaler.transform(sample["coords"]) if coord_scaler is not None else sample["coords"],
            "design": design_scaler.transform(sample["design"]) if design_scaler is not None else sample["design"],
            "stress": stress_scaler.transform(sample["stress"]) if stress_scaler is not None else sample["stress"],
            "case_name": sample["case_name"],
        }


def fit_scalers(
    dataset: FieldData,
    coord_norm_range: str = "bipolar",
    normalize_design: bool = True,
    normalize_stress: bool = True,
    stress_channel_dim: int = 1,
) -> Dict[str, object]:
    """Fit coordinate, design, and stress scalers from the training set."""
    stacked = dataset.stack_tensors()

    scalers: Dict[str, object] = {
        "coord_scaler": MinMaxScalerTensor(norm_range=coord_norm_range).fit(
            stacked["coords"],
            channel_dim=-1,
        )
    }

    if normalize_design:
        scalers["design_scaler"] = StandardScalerTensor().fit(stacked["design"], channel_dim=-1)

    if normalize_stress:
        stress_channel_dim = stress_channel_dim % stacked["stress"].ndim
        scalers["stress_scaler"] = StandardScalerTensor().fit(
            stacked["stress"],
            channel_dim=stress_channel_dim,
        )

    return scalers
