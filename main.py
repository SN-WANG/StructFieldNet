# Main Entry Point for StructFieldNet
# Author: Shengning Wang

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from structfieldnet.data.wing_dataset import (
    WingStressDataset,
    ScaledWingStressDataset,
    build_case_splits,
    fit_dataset_scalers,
)
from structfieldnet.losses.field_loss import StructFieldLoss
from structfieldnet.models.structfieldnet import StructFieldNet
from structfieldnet.trainers.structfield_trainer import StructFieldTrainer
from structfieldnet.utils.config import load_json_config, resolve_project_paths
from structfieldnet.utils.hue_logger import hue, logger
from structfieldnet.utils.seeder import seed_everything


def build_dataloaders(config: Dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any], Dict[str, Any]]:
    """Build dataset splits, scalers, and data loaders.

    Args:
        config: Parsed experiment configuration.

    Returns:
        Tuple containing train, val, and test loaders, fitted scalers, and split metadata.
    """
    data_cfg = config["data"]
    norm_cfg = config["normalization"]
    dataset_dir = Path(config["paths"]["dataset_dir"])

    split_paths = build_case_splits(
        data_dir=dataset_dir,
        train_ratio=float(data_cfg["train_ratio"]),
        val_ratio=float(data_cfg["val_ratio"]),
        test_ratio=float(data_cfg["test_ratio"]),
        seed=int(config["seed"]),
    )

    train_raw = WingStressDataset(
        split_paths["train"],
        verify_fixed_mesh=bool(data_cfg["verify_fixed_mesh"]),
        harmonize_units=bool(data_cfg["harmonize_units"]),
    )
    val_raw = WingStressDataset(
        split_paths["val"],
        verify_fixed_mesh=bool(data_cfg["verify_fixed_mesh"]),
        harmonize_units=bool(data_cfg["harmonize_units"]),
    )
    test_raw = WingStressDataset(
        split_paths["test"],
        verify_fixed_mesh=bool(data_cfg["verify_fixed_mesh"]),
        harmonize_units=bool(data_cfg["harmonize_units"]),
    )

    scalers = fit_dataset_scalers(
        dataset=train_raw,
        coords_norm_range=str(norm_cfg["coords_norm_range"]),
        normalize_design=bool(norm_cfg["normalize_design"]),
        normalize_stress=bool(norm_cfg["normalize_stress"]),
    )

    train_dataset = ScaledWingStressDataset(train_raw, scalers)
    val_dataset = ScaledWingStressDataset(val_raw, scalers)
    test_dataset = ScaledWingStressDataset(test_raw, scalers)

    pin_memory = bool(data_cfg["pin_memory"]) and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, scalers, split_paths


def build_model(config: Dict[str, Any]) -> StructFieldNet:
    """Instantiate StructFieldNet from configuration.

    Args:
        config: Parsed experiment configuration.

    Returns:
        Initialized StructFieldNet model.
    """
    model_cfg = config["model"]
    return StructFieldNet(
        coord_dim=int(model_cfg["coord_dim"]),
        design_dim=int(model_cfg["design_dim"]),
        output_dim=int(model_cfg["output_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        branch_hidden_dim=int(model_cfg["branch_hidden_dim"]),
        branch_num_layers=int(model_cfg["branch_num_layers"]),
        trunk_hidden_dim=int(model_cfg["trunk_hidden_dim"]),
        trunk_num_layers=int(model_cfg["trunk_num_layers"]),
        fusion_hidden_dim=int(model_cfg["fusion_hidden_dim"]),
        fusion_num_layers=int(model_cfg["fusion_num_layers"]),
        depth=int(model_cfg["depth"]),
        num_heads=int(model_cfg["num_heads"]),
        num_slices=int(model_cfg["num_slices"]),
        mlp_ratio=int(model_cfg["mlp_ratio"]),
        dropout=float(model_cfg["dropout"]),
    )


def build_trainer(model: StructFieldNet, config: Dict[str, Any], scalers: Dict[str, Any]) -> StructFieldTrainer:
    """Instantiate the StructFieldNet trainer.

    Args:
        model: StructFieldNet model.
        config: Parsed experiment configuration.
        scalers: Fitted scaler dictionary.

    Returns:
        Configured trainer instance.
    """
    train_cfg = config["training"]
    loss_cfg = config["loss"]

    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(train_cfg["max_epochs"]),
        eta_min=float(train_cfg["scheduler"]["eta_min"]),
    ) if str(train_cfg["scheduler"]["name"]).lower() == "cosine" else None

    criterion = StructFieldLoss(
        global_weight=float(loss_cfg["global_weight"]),
        hotspot_weight=float(loss_cfg["hotspot_weight"]),
        hotspot_percentile=float(loss_cfg["hotspot_percentile"]),
        hotspot_boost=float(loss_cfg["hotspot_boost"]),
    )

    device = str(train_cfg["device"]).lower()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    trainer = StructFieldTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        scalers=scalers,
        output_dir=Path(config["paths"]["output_dir"]),
        max_epochs=int(train_cfg["max_epochs"]),
        patience=int(train_cfg["patience"]),
        gradient_clip_norm=float(train_cfg["gradient_clip_norm"]),
        use_amp=bool(train_cfg["use_amp"]),
        device=device,
    )
    return trainer


def train_pipeline(config: Dict[str, Any]) -> None:
    """Run the full training workflow.

    Args:
        config: Parsed experiment configuration.
    """
    seed_everything(int(config["seed"]))
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, scalers, split_paths = build_dataloaders(config)
    model = build_model(config)

    if bool(config["training"]["compile_model"]) and hasattr(torch, "compile"):
        logger.info("compiling model with torch.compile...")
        model = torch.compile(model)

    num_params = sum(param.numel() for param in model.parameters())
    logger.info(f"model parameters: {hue.m}{num_params}{hue.q}")

    trainer = build_trainer(model, config, scalers)

    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    with (output_dir / "splits.json").open("w", encoding="utf-8") as file:
        json.dump({k: [path.stem for path in v] for k, v in split_paths.items()}, file, indent=2)

    trainer.fit(train_loader, val_loader)
    best_metrics = trainer.evaluate(test_loader, checkpoint_path=output_dir / "best.pt", save_name="test_metrics.json")
    logger.info(
        "test metrics | "
        f"mae: {hue.m}{best_metrics['field_mae']:.4e}{hue.q} | "
        f"rmse: {hue.m}{best_metrics['field_rmse']:.4e}{hue.q} | "
        f"rel_l2: {hue.m}{best_metrics['relative_l2']:.4e}{hue.q}"
    )


def evaluate_pipeline(config: Dict[str, Any]) -> None:
    """Run evaluation from an existing checkpoint.

    Args:
        config: Parsed experiment configuration.
    """
    output_dir = Path(config["paths"]["output_dir"])
    _, _, test_loader, scalers, _ = build_dataloaders(config)
    model = build_model(config)
    trainer = build_trainer(model, config, scalers)
    trainer.evaluate(test_loader, checkpoint_path=output_dir / "best.pt", save_name="test_metrics.json")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="StructFieldNet training and evaluation entry point.")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to the JSON config file.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Execution mode.")
    return parser.parse_args()


def main() -> None:
    """Program entry point."""
    args = parse_args()
    config = resolve_project_paths(load_json_config(Path(args.config)), PROJECT_ROOT)

    if args.mode == "train":
        train_pipeline(config)
    else:
        evaluate_pipeline(config)


if __name__ == "__main__":
    main()
