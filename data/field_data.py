# Dataset utilities for fixed-mesh structural field learning
# Author: Shengning Wang

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from utils.hue_logger import hue, logger
from utils.scaler import MinMaxScalerTensor, StandardScalerTensor

SplitManifest = Dict[str, List[str]]
ScalerDict = Dict[str, object]


@dataclass(frozen=True)
class FieldDataMeta:
    """
    Static tensor metadata for one structural field dataset.
    """

    num_nodes: int
    coord_dim: int
    design_dim: int
    output_dim: int


def discover_case_names(data_dir: Union[str, Path], prefix: str = "dp") -> List[str]:
    """
    Discover case names such as dp1 and dp200.

    Args:
        data_dir (Union[str, Path]): Directory containing case files.
        prefix (str): Case-name prefix.

    Returns:
        List[str]: Sorted case names.
    """
    root = Path(data_dir)
    case_paths = sorted(
        root.glob(f"{prefix}*.pt"),
        key=lambda path: int(path.stem[len(prefix):]),
    )
    if not case_paths:
        raise FileNotFoundError(f"No {prefix}*.pt files were found in {root}.")
    return [path.stem for path in case_paths]


def build_case_splits(
    data_dir: Union[str, Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> SplitManifest:
    """
    Build deterministic train, validation, and test splits.

    Args:
        data_dir (Union[str, Path]): Directory containing case files.
        train_ratio (float): Training split ratio.
        val_ratio (float): Validation split ratio.
        test_ratio (float): Test split ratio.
        seed (int): Random seed for shuffling.

    Returns:
        SplitManifest: Case names grouped by split.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.6f}.")

    case_names = discover_case_names(data_dir)
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
        raise ValueError("Training split is empty. Adjust the split ratios.")

    return {
        "train": shuffled[:num_train],
        "val": shuffled[num_train:num_train + num_val],
        "test": shuffled[num_train + num_val:],
    }


def save_split_manifest(split_manifest: SplitManifest, output_path: Union[str, Path]) -> None:
    """
    Save a split manifest as JSON.

    Args:
        split_manifest (SplitManifest): Split manifest to save.
        output_path (Union[str, Path]): Output file path.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(split_manifest, file, indent=2)


def load_split_manifest(input_path: Union[str, Path]) -> SplitManifest:
    """
    Load a split manifest from JSON.

    Args:
        input_path (Union[str, Path]): Input file path.

    Returns:
        SplitManifest: Loaded split manifest.
    """
    with open(input_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _load_case_payload(case_path: Path) -> Dict[str, Tensor]:
    """
    Load one structural case from disk.

    Args:
        case_path (Path): Path to the case file.

    Returns:
        Dict[str, Tensor]: Case payload with coords, design, and stress.
    """
    payload = torch.load(case_path, map_location="cpu")
    required_keys = {"coords", "design", "stress"}
    missing_keys = required_keys.difference(payload.keys())
    if missing_keys:
        raise KeyError(f"Missing keys in {case_path.name}: {sorted(missing_keys)}")

    coords = payload["coords"].to(torch.float32)
    design = payload["design"].to(torch.float32)
    stress = payload["stress"].to(torch.float32)

    if coords.ndim != 2:
        raise ValueError(f"coords must have shape (N, D), got {tuple(coords.shape)}")
    if design.ndim != 1:
        raise ValueError(f"design must have shape (C_DESIGN,), got {tuple(design.shape)}")
    if stress.ndim != 2 or stress.shape[-1] != 1:
        raise ValueError(f"stress must have shape (N, C_OUT), got {tuple(stress.shape)}")
    if coords.shape[0] != stress.shape[0]:
        raise ValueError(f"coords and stress must share node dimension, got {coords.shape[0]} and {stress.shape[0]}")
    if torch.isnan(coords).any() or torch.isnan(design).any() or torch.isnan(stress).any():
        raise ValueError(f"NaN detected in case {case_path.stem}")

    return {"coords": coords, "design": design, "stress": stress}


class FieldData(Dataset):
    """
    In-memory dataset for fixed-mesh structural field reconstruction.
    """

    def __init__(
        self,
        case_names: Sequence[str],
        coords: Tensor,
        designs: Tensor,
        stresses: Tensor,
        data_dir: Union[str, Path] = "./dataset",
    ) -> None:
        """
        Initialize the in-memory dataset.

        Args:
            case_names (Sequence[str]): Case names. (B,).
            coords (Tensor): Shared mesh coordinates. (N, D).
            designs (Tensor): Design vectors. (B, C_DESIGN).
            stresses (Tensor): Target stress fields. (B, N, C_OUT).
            data_dir (Union[str, Path]): Root data directory.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.case_names = list(case_names)
        self.coords = coords.to(torch.float32).contiguous()
        self.designs = designs.to(torch.float32).contiguous()
        self.stresses = stresses.to(torch.float32).contiguous()

        if len(self.case_names) == 0:
            raise ValueError("FieldData cannot be empty.")
        if self.designs.ndim != 2:
            raise ValueError(f"designs must have shape (B, C_DESIGN), got {tuple(self.designs.shape)}")
        if self.stresses.ndim != 3:
            raise ValueError(f"stresses must have shape (B, N, C_OUT), got {tuple(self.stresses.shape)}")
        if self.designs.shape[0] != len(self.case_names):
            raise ValueError("design count does not match case_names.")
        if self.stresses.shape[0] != len(self.case_names):
            raise ValueError("stress count does not match case_names.")
        if self.stresses.shape[1] != self.coords.shape[0]:
            raise ValueError("stress node dimension does not match coords.")

        self.meta = FieldDataMeta(
            num_nodes=int(self.coords.shape[0]),
            coord_dim=int(self.coords.shape[-1]),
            design_dim=int(self.designs.shape[-1]),
            output_dim=int(self.stresses.shape[-1]),
        )
        self.num_nodes = self.meta.num_nodes
        self.coord_dim = self.meta.coord_dim
        self.design_dim = self.meta.design_dim
        self.output_dim = self.meta.output_dim
        self._case_to_index = {case_name: index for index, case_name in enumerate(self.case_names)}

    @classmethod
    def from_directory(
        cls,
        data_dir: Union[str, Path],
        case_names: Optional[Sequence[str]] = None,
        verify_fixed_mesh: bool = True,
    ) -> "FieldData":
        """
        Load cases from disk into one in-memory dataset.

        Args:
            data_dir (Union[str, Path]): Directory containing case files.
            case_names (Optional[Sequence[str]]): Optional subset of case names.
            verify_fixed_mesh (bool): Whether to check shared coordinates.

        Returns:
            FieldData: Loaded dataset.
        """
        root = Path(data_dir)
        if case_names is None:
            case_names = discover_case_names(root)
        case_names = list(case_names)

        logger.info(f"loading field dataset with {hue.m}{len(case_names)}{hue.q} cases...")

        coords_reference: Tensor | None = None
        designs: List[Tensor] = []
        stresses: List[Tensor] = []

        for case_name in tqdm(case_names, desc="[FieldData] loading", leave=False, dynamic_ncols=True):
            case_path = root / f"{case_name}.pt"
            if not case_path.exists():
                raise FileNotFoundError(f"Case file not found: {case_path}")

            payload = _load_case_payload(case_path)
            coords = payload["coords"]

            if coords_reference is None:
                coords_reference = coords
            elif verify_fixed_mesh:
                if coords.shape != coords_reference.shape:
                    raise ValueError(
                        f"Case {case_name} has mesh shape {tuple(coords.shape)}, expected {tuple(coords_reference.shape)}"
                    )
                if not torch.allclose(coords, coords_reference, atol=1e-6, rtol=0.0):
                    raise ValueError(f"Case {case_name} does not share the fixed mesh coordinates.")

            designs.append(payload["design"])
            stresses.append(payload["stress"])

        if coords_reference is None:
            raise RuntimeError("No coordinates were loaded from disk.")

        dataset = cls(
            case_names=case_names,
            coords=coords_reference,
            designs=torch.stack(designs, dim=0),
            stresses=torch.stack(stresses, dim=0),
            data_dir=root,
        )
        logger.info(
            f"{hue.g}dataset initialized.{hue.q} "
            f"cases: {hue.m}{len(dataset)}{hue.q}, "
            f"nodes: {hue.m}{dataset.num_nodes}{hue.q}, "
            f"coord_dim: {hue.m}{dataset.coord_dim}{hue.q}, "
            f"design_dim: {hue.m}{dataset.design_dim}{hue.q}, "
            f"output_dim: {hue.m}{dataset.output_dim}{hue.q}"
        )
        return dataset

    def __len__(self) -> int:
        """
        Return the number of cases.

        Returns:
            int: Number of cases.
        """
        return len(self.case_names)

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str]]:
        """
        Get one case sample.

        Args:
            index (int): Case index.

        Returns:
            Dict[str, Union[Tensor, str]]: Coords. (N, D). Design. (C_DESIGN). Stress. (N, C_OUT).
        """
        return {
            "coords": self.coords,
            "design": self.designs[index],
            "stress": self.stresses[index],
            "case_name": self.case_names[index],
        }

    def subset(self, case_names: Sequence[str]) -> "FieldData":
        """
        Create a subset dataset without reloading from disk.

        Args:
            case_names (Sequence[str]): Case names for the subset.

        Returns:
            FieldData: Subset dataset.
        """
        indices = [self._case_to_index[case_name] for case_name in case_names]
        return FieldData(
            case_names=[self.case_names[index] for index in indices],
            coords=self.coords,
            designs=self.designs[indices],
            stresses=self.stresses[indices],
            data_dir=self.data_dir,
        )

    def split(self, split_manifest: SplitManifest) -> tuple["FieldData", "FieldData", "FieldData"]:
        """
        Split the full dataset with a manifest.

        Args:
            split_manifest (SplitManifest): Split manifest.

        Returns:
            tuple[FieldData, FieldData, FieldData]: Train, validation, and test datasets.
        """
        return (
            self.subset(split_manifest["train"]),
            self.subset(split_manifest["val"]),
            self.subset(split_manifest["test"]),
        )

    def stack_tensors(self) -> Dict[str, Tensor]:
        """
        Return stacked tensors used for scaler fitting.

        Returns:
            Dict[str, Tensor]: Coords. (N, D). Design. (B, C_DESIGN). Stress. (B, N, C_OUT).
        """
        return {
            "coords": self.coords,
            "design": self.designs,
            "stress": self.stresses,
        }


class ScaledFieldDataset(Dataset):
    """
    Pre-scaled dataset wrapper for repeated training and evaluation access.
    """

    def __init__(self, dataset: FieldData, scalers: ScalerDict) -> None:
        """
        Initialize the pre-scaled dataset wrapper.

        Args:
            dataset (FieldData): Raw in-memory dataset.
            scalers (ScalerDict): Fitted coordinate, design, and stress scalers.
        """
        self.case_names = list(dataset.case_names)

        coord_scaler = scalers.get("coord_scaler")
        design_scaler = scalers.get("design_scaler")
        stress_scaler = scalers.get("stress_scaler")

        self.coords = (
            coord_scaler.transform(dataset.coords).to(torch.float32).contiguous()
            if coord_scaler is not None
            else dataset.coords
        )
        self.designs = (
            design_scaler.transform(dataset.designs).to(torch.float32).contiguous()
            if design_scaler is not None
            else dataset.designs
        )
        self.stresses = (
            stress_scaler.transform(dataset.stresses).to(torch.float32).contiguous()
            if stress_scaler is not None
            else dataset.stresses
        )

    def __len__(self) -> int:
        """
        Return the number of cases.

        Returns:
            int: Number of cases.
        """
        return len(self.case_names)

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str]]:
        """
        Get one scaled case sample.

        Args:
            index (int): Case index.

        Returns:
            Dict[str, Union[Tensor, str]]: Scaled coords. (N, D). Scaled design. (C_DESIGN). Scaled stress. (N, C_OUT).
        """
        return {
            "coords": self.coords,
            "design": self.designs[index],
            "stress": self.stresses[index],
            "case_name": self.case_names[index],
        }


def fit_scalers(
    dataset: FieldData,
    coord_norm_range: str = "bipolar",
    normalize_design: bool = True,
    normalize_stress: bool = True,
    stress_channel_dim: int = -1,
) -> ScalerDict:
    """
    Fit coordinate, design, and stress scalers from the training split.

    Args:
        dataset (FieldData): Training dataset.
        coord_norm_range (str): Coordinate normalization range.
        normalize_design (bool): Whether to normalize design vectors.
        normalize_stress (bool): Whether to normalize target fields.
        stress_channel_dim (int): Channel dimension for stress scaling.

    Returns:
        ScalerDict: Fitted scaler dictionary.
    """
    tensors = dataset.stack_tensors()
    scalers: ScalerDict = {
        "coord_scaler": MinMaxScalerTensor(norm_range=coord_norm_range).fit(tensors["coords"], channel_dim=-1)
    }

    if normalize_design:
        scalers["design_scaler"] = StandardScalerTensor().fit(tensors["design"], channel_dim=-1)

    if normalize_stress:
        channel_dim = stress_channel_dim % tensors["stress"].ndim
        scalers["stress_scaler"] = StandardScalerTensor().fit(tensors["stress"], channel_dim=channel_dim)

    return scalers


def build_scaled_loaders(
    train_data: FieldData,
    val_data: FieldData,
    test_data: FieldData,
    scalers: ScalerDict,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build scaled DataLoaders for train, validation, and test splits.

    Args:
        train_data (FieldData): Training dataset.
        val_data (FieldData): Validation dataset.
        test_data (FieldData): Test dataset.
        scalers (ScalerDict): Fitted scaler dictionary.
        batch_size (int): Mini-batch size for train and validation.
        num_workers (int): Number of DataLoader workers.
        pin_memory (bool): Whether to enable pinned-memory transfer.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test loaders.
    """
    use_pin_memory = bool(pin_memory and torch.cuda.is_available())

    train_loader = DataLoader(
        ScaledFieldDataset(train_data, scalers),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        ScaledFieldDataset(val_data, scalers),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        ScaledFieldDataset(test_data, scalers),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
    )
    return train_loader, val_loader, test_loader


def restore_scalers(
    scaler_state_dict: Dict[str, Dict[str, Tensor]],
    coord_norm_range: str,
    normalize_design: bool,
    normalize_stress: bool,
    stress_channel_dim: int,
) -> ScalerDict:
    """
    Restore fitted scalers from a checkpoint payload.

    Args:
        scaler_state_dict (Dict[str, Dict[str, Tensor]]): Serialized scaler states.
        coord_norm_range (str): Coordinate normalization range.
        normalize_design (bool): Whether design normalization is enabled.
        normalize_stress (bool): Whether stress normalization is enabled.
        stress_channel_dim (int): Channel dimension for stress scaling.

    Returns:
        ScalerDict: Restored scaler dictionary.
    """
    scalers: ScalerDict = {}

    coord_scaler = MinMaxScalerTensor(norm_range=coord_norm_range)
    coord_scaler.load_state_dict(scaler_state_dict["coord_scaler"])
    scalers["coord_scaler"] = coord_scaler

    if normalize_design and "design_scaler" in scaler_state_dict:
        design_scaler = StandardScalerTensor()
        design_scaler.load_state_dict(scaler_state_dict["design_scaler"])
        scalers["design_scaler"] = design_scaler

    if normalize_stress and "stress_scaler" in scaler_state_dict:
        stress_scaler = StandardScalerTensor()
        stress_state = scaler_state_dict["stress_scaler"]
        if "channel_dim" not in stress_state:
            stress_scaler.mean = stress_state["mean"]
            stress_scaler.std = stress_state["std"]
            stress_scaler.channel_dim = stress_channel_dim
        else:
            stress_scaler.load_state_dict(stress_state)
        scalers["stress_scaler"] = stress_scaler

    return scalers
