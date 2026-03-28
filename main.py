# Main Script for StructFieldNet: Train, Infer, and Probe
# Author: Shengning Wang

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import config
from data.field_data import (
    FieldData,
    ScaledFieldDataset,
    build_case_splits,
    fit_scalers,
    load_split_manifest,
    restore_scalers,
    save_split_manifest,
)
from data.field_metrics import FieldMetrics
from models.structfield_net import StructFieldNet
from training.field_trainer import FieldTrainer
from utils.hue_logger import hue, logger
from utils.seeder import seed_everything


def resolve_device(device: str) -> str:
    """Resolve the requested device."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable, falling back to CPU.")
        return "cpu"
    return device


def resolve_output_dir(args) -> Path:
    """Create and return the output directory."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_split_manifest(args, output_dir: Path) -> Dict[str, list[str]]:
    """Load an existing split manifest or create a new one."""
    split_path = output_dir / "splits.json"
    if split_path.exists():
        logger.info("loading existing split manifest...")
        return load_split_manifest(split_path)

    return build_case_splits(
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


def load_datasets(args, split_manifest: Dict[str, list[str]]) -> Tuple[FieldData, FieldData, FieldData]:
    """Load full dataset and split it into train, val, and test subsets."""
    dataset = FieldData.from_directory(
        data_dir=args.data_dir,
        verify_fixed_mesh=args.verify_fixed_mesh,
    )
    return dataset.split(split_manifest)


def build_loaders(args, train_data: FieldData, val_data: FieldData, test_data: FieldData):
    """Build scaled datasets and dataloaders."""
    scalers = fit_scalers(
        dataset=train_data,
        coord_norm_range=args.coord_norm_range,
        normalize_design=args.normalize_design,
        normalize_stress=args.normalize_stress,
        stress_channel_dim=args.stress_channel_dim,
    )

    pin_memory = bool(args.pin_memory and torch.cuda.is_available())

    train_loader = DataLoader(
        ScaledFieldDataset(train_data, scalers),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        ScaledFieldDataset(val_data, scalers),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        ScaledFieldDataset(test_data, scalers),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, scalers


def build_model(args, num_nodes: int) -> StructFieldNet:
    """Build StructFieldNet."""
    model = StructFieldNet(
        num_nodes=num_nodes,
        coord_dim=args.coord_dim,
        design_dim=args.design_dim,
        output_dim=args.output_dim,
        width=args.width,
        depth=args.depth,
        num_slices=args.num_slices,
        num_heads=args.num_heads,
        num_bases=args.num_bases,
        mlp_ratio=args.mlp_ratio,
        branch_hidden_dim=args.branch_hidden_dim,
        branch_layers=args.branch_layers,
        trunk_hidden_dim=args.trunk_hidden_dim,
        trunk_layers=args.trunk_layers,
        lifting_hidden_dim=args.lifting_hidden_dim,
        lifting_layers=args.lifting_layers,
        dropout=args.dropout,
    )
    return model


def maybe_compile_model(args, model: StructFieldNet) -> StructFieldNet:
    """Compile the model when requested."""
    if args.compile_model and hasattr(torch, "compile"):
        logger.info("compiling model with torch.compile...")
        model = torch.compile(model)
    return model


def initialize_model_basis(model: StructFieldNet, train_data: FieldData, scalers: Dict[str, object]) -> None:
    """Warm-start the basis decoder from scaled training tensors."""
    design_scaler = scalers.get("design_scaler")
    stress_scaler = scalers.get("stress_scaler")

    design = design_scaler.transform(train_data.designs) if design_scaler is not None else train_data.designs
    stress = stress_scaler.transform(train_data.stresses) if stress_scaler is not None else train_data.stresses
    model.initialize_basis(design=design, stress=stress)


def build_trainer(args, model: StructFieldNet, scalers: Dict[str, object], output_dir: Path) -> FieldTrainer:
    """Build the trainer."""
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.eta_min)

    return FieldTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scalers=scalers,
        output_dir=output_dir,
        max_epochs=args.max_epochs,
        patience=args.patience,
        gradient_clip_norm=args.gradient_clip_norm,
        use_amp=args.use_amp,
        device=resolve_device(args.device),
    )


def save_run_config(args, output_dir: Path, split_manifest: Dict[str, list[str]]) -> None:
    """Save configuration and split manifest."""
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)
    save_split_manifest(split_manifest, output_dir / "splits.json")


def aggregate_case_metrics(case_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Average test metrics across all cases."""
    metric_names = sorted(next(iter(case_metrics.values())).keys())
    return {
        metric_name: float(sum(item[metric_name] for item in case_metrics.values()) / len(case_metrics))
        for metric_name in metric_names
    }


def train_pipeline(args) -> None:
    """Run the training pipeline."""
    seed_everything(args.seed)
    output_dir = resolve_output_dir(args)
    split_manifest = resolve_split_manifest(args, output_dir)
    train_data, val_data, test_data = load_datasets(args, split_manifest)
    train_loader, val_loader, _, scalers = build_loaders(args, train_data, val_data, test_data)

    model = build_model(args, num_nodes=train_data.num_nodes)
    initialize_model_basis(model, train_data, scalers)
    model = maybe_compile_model(args, model)
    num_params = sum(parameter.numel() for parameter in model.parameters())
    logger.info(f"model has {hue.m}{num_params}{hue.q} parameters")

    save_run_config(args, output_dir, split_manifest)
    trainer = build_trainer(args, model, scalers, output_dir)
    trainer.fit(train_loader, val_loader)


def inference_pipeline(args) -> None:
    """Run checkpoint-based inference on the test split."""
    output_dir = resolve_output_dir(args)
    checkpoint_path = output_dir / "best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = output_dir / "ckpt.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {output_dir}")

    device = torch.device(resolve_device(args.device))
    logger.info(f"loading checkpoint: {hue.b}{checkpoint_path.name}{hue.q}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    scalers = restore_scalers(
        scaler_state_dict=checkpoint["scaler_state_dict"],
        coord_norm_range=args.coord_norm_range,
        normalize_design=args.normalize_design,
        normalize_stress=args.normalize_stress,
        stress_channel_dim=args.stress_channel_dim,
    )

    split_manifest = resolve_split_manifest(args, output_dir)
    _, _, test_data = load_datasets(args, split_manifest)
    test_loader = DataLoader(ScaledFieldDataset(test_data, scalers), batch_size=1, shuffle=False, num_workers=0)

    model = build_model(args, num_nodes=test_data.num_nodes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = maybe_compile_model(args, model)
    model.to(device)
    model.eval()

    metrics = FieldMetrics(hotspot_percentile=args.hotspot_percentile)
    visualizer = None
    if args.render_visualization:
        from data.field_vis import FieldVis

        visualizer = FieldVis(
            output_dir=output_dir,
            off_screen=args.off_screen,
            point_size=args.render_point_size,
            screenshot_scale=args.screenshot_scale,
        )

    coord_scaler = scalers.get("coord_scaler")
    stress_scaler = scalers.get("stress_scaler")

    case_metrics: Dict[str, Dict[str, float]] = {}

    with torch.no_grad():
        for batch in test_loader:
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


def probe_pipeline(args) -> None:
    """Run one forward-backward pass to estimate peak GPU memory."""
    device_name = resolve_device(args.device)
    if not device_name.startswith("cuda"):
        logger.warning("No CUDA device detected. Probe is skipped on CPU.")
        return

    seed_everything(args.seed)
    output_dir = resolve_output_dir(args)
    split_manifest = resolve_split_manifest(args, output_dir)
    train_data, val_data, test_data = load_datasets(args, split_manifest)
    train_loader, _, _, scalers = build_loaders(args, train_data, val_data, test_data)

    batch = next(iter(train_loader))
    device = torch.device(device_name)

    model = build_model(args, num_nodes=train_data.num_nodes)
    initialize_model_basis(model, train_data, scalers)
    model = maybe_compile_model(args, model)
    model = model.to(device).train()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    coords = batch["coords"].to(device)
    design = batch["design"].to(device)
    target = batch["stress"].to(device)

    criterion = torch.nn.MSELoss()

    torch.cuda.reset_peak_memory_stats(device)
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
        f"width={hue.m}{args.width}{hue.q}, "
        f"depth={hue.m}{args.depth}{hue.q}, "
        f"slices={hue.m}{args.num_slices}{hue.q}"
    )
    logger.info(
        f"peak usage: {hue.m}{peak_memory / 1e9:.2f}{hue.q} GB "
        f"({hue.m}{usage_pct:.1f}{hue.q} %) -> {status}"
    )


if __name__ == "__main__":
    args = config.get_args()

    if "probe" in args.mode:
        logger.info(f"running pipeline: {hue.b}probe{hue.q}")
        probe_pipeline(args)
    if "train" in args.mode:
        logger.info(f"running pipeline: {hue.b}train{hue.q}")
        train_pipeline(args)
    if "infer" in args.mode:
        logger.info(f"running pipeline: {hue.b}infer{hue.q}")
        inference_pipeline(args)
