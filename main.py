# Main script for StructFieldNet probing, training and inference
# Author: Shengning Wang

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import config

from data.field_data import (
    FieldData,
    FieldDataMeta,
    ScalerDict,
    SplitManifest,
    build_case_splits,
    build_scaled_loaders,
    fit_scalers,
    load_split_manifest,
    restore_scalers,
    save_split_manifest,
)
from data.field_metrics import FieldMetrics

from models.fieldnet import StructFieldNet
from training.field_trainer import FieldTrainer

from utils.hue_logger import hue, logger
from utils.seeder import seed_everything


@dataclass
class FieldRuntime:
    """
    Runtime bundle for StructFieldNet data pipelines.
    """

    split_manifest: SplitManifest
    train_data: FieldData
    val_data: FieldData
    test_data: FieldData
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scalers: ScalerDict


def resolve_device(device: str) -> str:
    """
    Resolve the requested computation device.

    Args:
        device (str): Requested device string.

    Returns:
        str: Usable device string.
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable, falling back to CPU.")
        return "cpu"
    return device


def resolve_output_dir(args: Any) -> Path:
    """
    Create and return the output directory.

    Args:
        args (Any): Parsed command-line arguments.

    Returns:
        Path: Output directory.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_split_manifest(
    args: Any,
    output_dir: Path,
    saved_split_manifest: SplitManifest | None = None,
) -> SplitManifest:
    """
    Load a saved split manifest or build one deterministically.

    Args:
        args (Any): Parsed command-line arguments.
        output_dir (Path): Output directory containing split metadata.
        saved_split_manifest (SplitManifest | None): Optional manifest restored from a checkpoint.

    Returns:
        SplitManifest: Train, validation, and test split definition.
    """
    split_path = output_dir / "splits.json"
    if split_path.exists():
        logger.info("loading existing split manifest...")
        return load_split_manifest(split_path)
    if saved_split_manifest is not None:
        logger.info("using split manifest restored from checkpoint...")
        return saved_split_manifest
    return build_case_splits(
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


def load_datasets(args: Any, split_manifest: SplitManifest) -> Tuple[FieldData, FieldData, FieldData]:
    """
    Load the full dataset and split it into train, validation, and test sets.

    Args:
        args (Any): Parsed command-line arguments.
        split_manifest (SplitManifest): Split definition.

    Returns:
        Tuple[FieldData, FieldData, FieldData]: Train, validation, and test datasets.
    """
    dataset = FieldData.from_directory(
        data_dir=args.data_dir,
        verify_fixed_mesh=args.verify_fixed_mesh,
    )
    return dataset.split(split_manifest)


def build_model(
    args: Any | None = None,
    data_meta: FieldDataMeta | None = None,
    model_args: Dict[str, Any] | None = None,
) -> Tuple[StructFieldNet, Dict[str, Any]]:
    """
    Build StructFieldNet and return its constructor arguments.

    Args:
        args (Any | None): Parsed command-line arguments.
        data_meta (FieldDataMeta | None): Dataset metadata used to infer model dimensions.
        model_args (Dict[str, Any] | None): Explicit constructor arguments.

    Returns:
        Tuple[StructFieldNet, Dict[str, Any]]: Model instance and constructor arguments.
    """
    if model_args is None:
        if args is None:
            raise ValueError("args must be provided when model_args is None")

        coord_dim = data_meta.coord_dim if data_meta is not None else args.coord_dim
        design_dim = data_meta.design_dim if data_meta is not None else args.design_dim
        output_dim = data_meta.output_dim if data_meta is not None else args.output_dim
        num_nodes = data_meta.num_nodes if data_meta is not None else None
        if num_nodes is None:
            raise ValueError("num_nodes could not be resolved for StructFieldNet")

        model_args = {
            "num_nodes": num_nodes,
            "coord_dim": coord_dim,
            "design_dim": design_dim,
            "output_dim": output_dim,
            "width": args.width,
            "depth": args.depth,
            "num_slices": args.num_slices,
            "num_heads": args.num_heads,
            "num_bases": args.num_bases,
            "mlp_ratio": args.mlp_ratio,
            "branch_hidden_dim": args.branch_hidden_dim,
            "branch_layers": args.branch_layers,
            "trunk_hidden_dim": args.trunk_hidden_dim,
            "trunk_layers": args.trunk_layers,
            "lifting_hidden_dim": args.lifting_hidden_dim,
            "lifting_layers": args.lifting_layers,
            "dropout": args.dropout,
        }

    return StructFieldNet(**model_args), model_args


def maybe_compile_model(args: Any, model: StructFieldNet) -> StructFieldNet:
    """
    Compile the model when requested.

    Args:
        args (Any): Parsed command-line arguments.
        model (StructFieldNet): Model instance.

    Returns:
        StructFieldNet: Original or compiled model.
    """
    if args.compile_model and hasattr(torch, "compile"):
        logger.info("compiling model with torch.compile...")
        model = torch.compile(model)
    return model


def initialize_model_basis(model: StructFieldNet, train_data: FieldData, scalers: ScalerDict) -> None:
    """
    Warm-start the basis decoder from scaled training tensors.

    Args:
        model (StructFieldNet): Model instance.
        train_data (FieldData): Training dataset.
        scalers (ScalerDict): Fitted scaler dictionary.
    """
    design_scaler = scalers.get("design_scaler")
    stress_scaler = scalers.get("stress_scaler")

    design = design_scaler.transform(train_data.designs) if design_scaler is not None else train_data.designs
    stress = stress_scaler.transform(train_data.stresses) if stress_scaler is not None else train_data.stresses
    model.initialize_basis(design=design, stress=stress)


def build_trainer(
    args: Any,
    model: StructFieldNet,
    params: Dict[str, Any],
    scalers: ScalerDict,
    output_dir: Path,
) -> FieldTrainer:
    """
    Build the StructFieldNet trainer.

    Args:
        args (Any): Parsed command-line arguments.
        model (StructFieldNet): Model instance.
        params (Dict[str, Any]): Checkpoint metadata.
        scalers (ScalerDict): Fitted scaler dictionary.
        output_dir (Path): Output directory.

    Returns:
        FieldTrainer: Configured trainer.
    """
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.eta_min)

    return FieldTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        params=params,
        scalers=scalers,
        output_dir=output_dir,
        max_epochs=args.max_epochs,
        patience=args.patience,
        gradient_clip_norm=args.gradient_clip_norm,
        use_amp=args.use_amp,
        device=resolve_device(args.device),
    )


def data_pipeline(
    args: Any,
    output_dir: Path,
    scalers: ScalerDict | None = None,
    split_manifest: SplitManifest | None = None,
) -> FieldRuntime:
    """
    Build the full data runtime for probe, train, and inference.

    Args:
        args (Any): Parsed command-line arguments.
        output_dir (Path): Output directory.
        scalers (ScalerDict | None): Optional pre-restored scalers.
        split_manifest (SplitManifest | None): Optional pre-restored split manifest.

    Returns:
        FieldRuntime: Runtime bundle with datasets, loaders, scalers, and split metadata.
    """
    logger.info(f"{hue.c}============================== [DATA PIPELINE] START =============================={hue.q}")

    split_manifest = resolve_split_manifest(args, output_dir, saved_split_manifest=split_manifest)
    train_data, val_data, test_data = load_datasets(args, split_manifest)

    if scalers is None:
        scalers = fit_scalers(
            dataset=train_data,
            coord_norm_range=args.coord_norm_range,
            normalize_design=args.normalize_design,
            normalize_stress=args.normalize_stress,
            stress_channel_dim=args.stress_channel_dim,
        )

    train_loader, val_loader, test_loader = build_scaled_loaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        scalers=scalers,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    logger.info(f"{hue.g}=============================== [DATA PIPELINE] END ==============================={hue.q}")
    return FieldRuntime(
        split_manifest=split_manifest,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scalers=scalers,
    )


def resolve_checkpoint_path(output_dir: Path) -> Path:
    """
    Resolve the preferred checkpoint path in one output directory.

    Args:
        output_dir (Path): Output directory.

    Returns:
        Path: Best-available checkpoint path.
    """
    checkpoint_path = output_dir / "best.pt"
    if checkpoint_path.exists():
        return checkpoint_path

    checkpoint_path = output_dir / "ckpt.pt"
    if checkpoint_path.exists():
        return checkpoint_path

    raise FileNotFoundError(f"No checkpoint found in {output_dir}")


def load_checkpoint(output_dir: Path, device: torch.device) -> Dict[str, Any]:
    """
    Load one checkpoint from disk.

    Args:
        output_dir (Path): Output directory.
        device (torch.device): Target device for deserialization.

    Returns:
        Dict[str, Any]: Checkpoint payload.
    """
    checkpoint_path = resolve_checkpoint_path(output_dir)
    logger.info(f"loading checkpoint: {hue.b}{checkpoint_path.name}{hue.q}")
    return torch.load(checkpoint_path, map_location=device)


def normalize_model_state_dict(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Normalize checkpoint state-dict keys for compiled and non-compiled models.

    Args:
        state_dict (Dict[str, Tensor]): Raw checkpoint state dict.

    Returns:
        Dict[str, Tensor]: Normalized state dict.
    """
    if any(key.startswith("_orig_mod.") for key in state_dict):
        return {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    return state_dict


def save_run_config(args: Any, output_dir: Path, split_manifest: SplitManifest) -> None:
    """
    Save the run configuration and split manifest.

    Args:
        args (Any): Parsed command-line arguments.
        output_dir (Path): Output directory.
        split_manifest (SplitManifest): Split definition.
    """
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)
    save_split_manifest(split_manifest, output_dir / "splits.json")


def aggregate_case_metrics(case_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Average case-wise metrics over the full test split.

    Args:
        case_metrics (Dict[str, Dict[str, float]]): Metrics grouped by case name.

    Returns:
        Dict[str, float]: Averaged metrics.
    """
    metric_names = sorted(next(iter(case_metrics.values())).keys())
    return {
        metric_name: float(sum(item[metric_name] for item in case_metrics.values()) / len(case_metrics))
        for metric_name in metric_names
    }


def build_visualizer(args: Any, output_dir: Path):
    """
    Build the optional field visualizer.

    Args:
        args (Any): Parsed command-line arguments.
        output_dir (Path): Output directory.

    Returns:
        Any: Field visualizer or None.
    """
    if not args.render_visualization:
        return None

    from data.field_vis import FieldVis

    return FieldVis(
        output_dir=output_dir,
        off_screen=args.off_screen,
        point_size=args.render_point_size,
        screenshot_scale=args.screenshot_scale,
    )


def train_pipeline(args: Any) -> None:
    """
    Run the training pipeline.

    Args:
        args (Any): Parsed command-line arguments.
    """
    logger.info(f"{hue.c}============================= [TRAIN PIPELINE] START =============================={hue.q}")
    seed_everything(args.seed)
    output_dir = resolve_output_dir(args)
    runtime = data_pipeline(args, output_dir)

    model, model_args = build_model(args=args, data_meta=runtime.train_data.meta)
    initialize_model_basis(model, runtime.train_data, runtime.scalers)
    model = maybe_compile_model(args, model)

    num_params = sum(parameter.numel() for parameter in model.parameters())
    logger.info(f"model has {hue.m}{num_params}{hue.q} parameters")

    params = {
        "model_args": model_args,
        "split_manifest": runtime.split_manifest,
        "data_meta": runtime.train_data.meta.__dict__,
    }
    save_run_config(args, output_dir, runtime.split_manifest)

    trainer = build_trainer(args, model, params, runtime.scalers, output_dir)
    trainer.fit(runtime.train_loader, runtime.val_loader)
    logger.info(f"{hue.g}============================== [TRAIN PIPELINE] END ==============================={hue.q}")


def inference_pipeline(args: Any) -> None:
    """
    Run checkpoint-based inference on the test split.

    Args:
        args (Any): Parsed command-line arguments.
    """
    logger.info(f"{hue.c}============================= [INFER PIPELINE] START =============================={hue.q}")
    output_dir = resolve_output_dir(args)
    device = torch.device(resolve_device(args.device))
    checkpoint = load_checkpoint(output_dir, device)

    checkpoint_params = checkpoint.get("params", {})
    scalers = restore_scalers(
        scaler_state_dict=checkpoint["scaler_state_dict"],
        coord_norm_range=args.coord_norm_range,
        normalize_design=args.normalize_design,
        normalize_stress=args.normalize_stress,
        stress_channel_dim=args.stress_channel_dim,
    )
    runtime = data_pipeline(
        args,
        output_dir=output_dir,
        scalers=scalers,
        split_manifest=checkpoint_params.get("split_manifest"),
    )

    saved_model_args = checkpoint_params.get("model_args")
    model, _ = build_model(args=args, data_meta=runtime.test_data.meta, model_args=saved_model_args)
    model.load_state_dict(normalize_model_state_dict(checkpoint["model_state_dict"]))
    model = maybe_compile_model(args, model)
    model.to(device)
    model.eval()

    metrics = FieldMetrics(hotspot_percentile=args.hotspot_percentile)
    visualizer = build_visualizer(args, output_dir)
    coord_scaler = runtime.scalers.get("coord_scaler")
    stress_scaler = runtime.scalers.get("stress_scaler")

    case_metrics: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for batch in runtime.test_loader:
            case_name = batch["case_name"][0]
            coords_scaled = batch["coords"].to(device)
            design_scaled = batch["design"].to(device)
            target_scaled = batch["stress"].to(device)

            pred_scaled = model(coords_scaled, design_scaled)
            pred = stress_scaler.inverse_transform(pred_scaled).cpu().squeeze(0) if stress_scaler else pred_scaled.cpu().squeeze(0)
            target = stress_scaler.inverse_transform(target_scaled).cpu().squeeze(0) if stress_scaler else target_scaled.cpu().squeeze(0)
            coords = coord_scaler.inverse_transform(coords_scaled).cpu().squeeze(0) if coord_scaler else coords_scaled.cpu().squeeze(0)

            case_metrics[case_name] = metrics.compute(pred, target)
            logger.info(
                f"case {hue.b}{case_name}{hue.q} | "
                f"MSE={hue.m}{case_metrics[case_name]['mse']:.4e}{hue.q}, "
                f"R2={hue.m}{case_metrics[case_name]['r2']:.4f}{hue.q}, "
                f"ACC={hue.m}{case_metrics[case_name]['accuracy']:.2f}%{hue.q}"
            )

            torch.save(pred, output_dir / f"{case_name}_pred.pt")
            if visualizer is not None:
                visualizer.compare_fields(gt=target, pred=pred, coords=coords, case_name=case_name)

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(case_metrics, file, indent=2)

    summary = aggregate_case_metrics(case_metrics)
    with open(output_dir / "test_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    logger.info(
        f"{hue.g}test summary{hue.q} | "
        f"MSE={hue.m}{summary['mse']:.4e}{hue.q}, "
        f"R2={hue.m}{summary['r2']:.4f}{hue.q}, "
        f"ACC={hue.m}{summary['accuracy']:.2f}%{hue.q}"
    )

    if args.render_metric_plots:
        from data.field_plot import plot_metrics_summary, plot_training_curves

        history_path = output_dir / "history.json"
        if history_path.exists():
            plot_training_curves(history_path, output_dir / "training_curve.png")
        plot_metrics_summary(output_dir / "test_metrics.json", output_dir / "metrics_summary.png")

    logger.info(f"{hue.g}============================== [INFER PIPELINE] END ==============================={hue.q}")


def probe_pipeline(args: Any) -> None:
    """
    Run one forward-backward pass to estimate peak GPU memory.

    Args:
        args (Any): Parsed command-line arguments.
    """
    logger.info(f"{hue.c}============================= [PROBE PIPELINE] START =============================={hue.q}")
    device_name = resolve_device(args.device)
    if not device_name.startswith("cuda"):
        logger.warning("No CUDA device detected. Probe is skipped on CPU.")
        logger.info(f"{hue.g}============================== [PROBE PIPELINE] END ==============================={hue.q}")
        return

    seed_everything(args.seed)
    output_dir = resolve_output_dir(args)
    runtime = data_pipeline(args, output_dir)

    batch = next(iter(runtime.train_loader))
    device = torch.device(device_name)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model, _ = build_model(args=args, data_meta=runtime.train_data.meta)
    initialize_model_basis(model, runtime.train_data, runtime.scalers)
    model = maybe_compile_model(args, model)
    model = model.to(device).train()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    coords = batch["coords"].to(device)
    design = batch["design"].to(device)
    target = batch["stress"].to(device)

    optimizer.zero_grad(set_to_none=True)
    pred = model(coords, design)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()

    peak_memory = torch.cuda.max_memory_allocated(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    usage_pct = 100.0 * peak_memory / total_memory

    if usage_pct < 75.0:
        status = f"{hue.g}SAFE{hue.q}"
    elif usage_pct < 92.0:
        status = f"{hue.y}WARNING{hue.q}"
    else:
        status = f"{hue.r}CRITICAL{hue.q}"

    logger.info(
        f"{hue.y}probe config:{hue.q} "
        f"batch={hue.m}{coords.shape[0]}{hue.q}, "
        f"nodes={hue.m}{coords.shape[1]}{hue.q}, "
        f"coord_dim={hue.m}{runtime.train_data.coord_dim}{hue.q}, "
        f"design_dim={hue.m}{runtime.train_data.design_dim}{hue.q}, "
        f"width={hue.m}{args.width}{hue.q}, "
        f"depth={hue.m}{args.depth}{hue.q}, "
        f"slices={hue.m}{args.num_slices}{hue.q}"
    )
    logger.info(
        f"{hue.y}device: {hue.b}{torch.cuda.get_device_name(device)}{hue.q} "
        f"({hue.m}{total_memory / 1e9:.1f}{hue.q} GB)"
    )
    logger.info(
        f"{hue.y}peak usage: {hue.m}{peak_memory / 1e9:.2f}{hue.q} GB "
        f"({hue.m}{usage_pct:.1f}{hue.q} %) -> {status}"
    )
    logger.info(f"{hue.g}============================== [PROBE PIPELINE] END ==============================={hue.q}")


def main() -> None:
    """
    Execute requested StructFieldNet pipelines.
    """
    args = config.get_args()
    phase_map = {
        "probe": probe_pipeline,
        "train": train_pipeline,
        "infer": inference_pipeline,
    }

    for phase in ("probe", "train", "infer"):
        if phase in args.mode:
            logger.info(f"running pipeline: {hue.b}{phase}{hue.q}")
            phase_map[phase](args)


if __name__ == "__main__":
    main()
