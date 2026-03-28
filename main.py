# Main script for StructFieldNet: training, inference, and probe
# Author: Shengning Wang

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import config
from data.field_data import FieldData, ScaledFieldDataset, fit_scalers
from models.structfield_net import StructFieldNet
from training.base_criterion import MSECriterion, Metrics
from training.field_trainer import FieldTrainer
from utils.hue_logger import hue, logger
from utils.scaler import IdentityScalerTensor, MinMaxScalerTensor, StandardScalerTensor
from utils.seeder import seed_everything


def _resolve_split_manifest(args, output_dir: Path) -> Dict[str, list[str]]:
    """Load an existing split manifest or create a deterministic new one."""
    manifest_path = output_dir / "splits.json"
    if manifest_path.exists():
        logger.info("loading existing split manifest...")
        return FieldData.load_split_manifest(manifest_path)

    split_manifest = FieldData.build_case_splits(
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    return split_manifest


def _build_datasets(args, split_manifest: Dict[str, list[str]]) -> Tuple[FieldData, FieldData, FieldData]:
    train_raw, val_raw, test_raw, _ = FieldData.spawn(
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        verify_fixed_mesh=args.verify_fixed_mesh,
        split_manifest=split_manifest,
    )
    return train_raw, val_raw, test_raw


def _build_loaders(args, train_raw: FieldData, val_raw: FieldData, test_raw: FieldData):
    scalers = fit_scalers(
        dataset=train_raw,
        coord_norm_range=args.coord_norm_range,
        normalize_design=args.normalize_design,
        normalize_stress=args.normalize_stress,
    )

    train_dataset = ScaledFieldDataset(train_raw, scalers)
    val_dataset = ScaledFieldDataset(val_raw, scalers)
    test_dataset = ScaledFieldDataset(test_raw, scalers)

    pin_memory = bool(args.pin_memory and torch.cuda.is_available())
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, scalers


def _build_model(args) -> StructFieldNet:
    return StructFieldNet(
        coord_dim=args.coord_dim,
        design_dim=args.design_dim,
        output_dim=args.output_dim,
        width=args.width,
        depth=args.depth,
        num_slices=args.num_slices,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        branch_hidden_dim=args.branch_hidden_dim,
        branch_layers=args.branch_layers,
        trunk_hidden_dim=args.trunk_hidden_dim,
        trunk_layers=args.trunk_layers,
        lifting_hidden_dim=args.lifting_hidden_dim,
        lifting_layers=args.lifting_layers,
        dropout=args.dropout,
    )


def _build_trainer(args, model: StructFieldNet, scalers: Dict[str, object], output_dir: Path) -> FieldTrainer:
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs,
        eta_min=args.eta_min,
    )
    criterion = MSECriterion()
    metrics = Metrics(hotspot_percentile=args.hotspot_percentile)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable, falling back to CPU.")
        device = "cpu"

    return FieldTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        metrics=metrics,
        scalers=scalers,
        output_dir=output_dir,
        max_epochs=args.max_epochs,
        patience=args.patience,
        gradient_clip_norm=args.gradient_clip_norm if args.gradient_clip_norm > 0 else None,
        use_amp=args.use_amp,
        device=device,
    )


def _save_run_config(args, output_dir: Path, split_manifest: Dict[str, list[str]]) -> None:
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)
    FieldData.save_split_manifest(split_manifest, output_dir / "splits.json")


def _restore_scalers(checkpoint: Dict[str, object], args) -> Dict[str, object]:
    coord_scaler = MinMaxScalerTensor(norm_range=args.coord_norm_range)
    coord_scaler.load_state_dict(checkpoint["scaler_state_dict"]["coord_scaler"])

    if args.normalize_design:
        design_scaler = StandardScalerTensor()
    else:
        design_scaler = IdentityScalerTensor()
    design_scaler.load_state_dict(checkpoint["scaler_state_dict"]["design_scaler"])

    if args.normalize_stress:
        stress_scaler = StandardScalerTensor()
    else:
        stress_scaler = IdentityScalerTensor()
    stress_scaler.load_state_dict(checkpoint["scaler_state_dict"]["stress_scaler"])

    return {
        "coord_scaler": coord_scaler,
        "design_scaler": design_scaler,
        "stress_scaler": stress_scaler,
    }


def train_pipeline(args) -> None:
    """Run the full training workflow."""
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_manifest = _resolve_split_manifest(args, output_dir)
    train_raw, val_raw, test_raw = _build_datasets(args, split_manifest)
    train_loader, val_loader, _, scalers = _build_loaders(args, train_raw, val_raw, test_raw)

    model = _build_model(args)
    if args.compile_model and hasattr(torch, "compile"):
        logger.info("compiling model with torch.compile...")
        model = torch.compile(model)

    num_params = sum(parameter.numel() for parameter in model.parameters())
    logger.info(f"model has {hue.m}{num_params}{hue.q} parameters")

    _save_run_config(args, output_dir, split_manifest)
    trainer = _build_trainer(args, model, scalers, output_dir)
    trainer.fit(train_loader, val_loader)


def inference_pipeline(args) -> None:
    """Restore the best checkpoint and run case-wise inference."""
    output_dir = Path(args.output_dir)
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")

    best_path = output_dir / "best.pt"
    ckpt_path = output_dir / "ckpt.pt"
    checkpoint_path = best_path if best_path.exists() else ckpt_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {output_dir}")

    logger.info(f"loading checkpoint: {hue.b}{checkpoint_path.name}{hue.q}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    scalers = _restore_scalers(checkpoint, args)

    split_manifest = _resolve_split_manifest(args, output_dir)
    _, _, test_raw = _build_datasets(args, split_manifest)
    test_dataset = ScaledFieldDataset(test_raw, scalers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = _build_model(args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    num_params = sum(parameter.numel() for parameter in model.parameters())
    logger.info(f"model has {hue.m}{num_params}{hue.q} parameters")

    metrics_evaluator = Metrics(hotspot_percentile=args.hotspot_percentile)
    visualizer = None
    if args.render_visualization:
        from data.field_vis import FieldVis

        visualizer = FieldVis(
            output_dir=output_dir,
            mesh_mode=args.mesh_mode,
            off_screen=args.off_screen,
            point_size=args.render_point_size,
            screenshot_scale=args.screenshot_scale,
        )

    case_metrics: Dict[str, Dict[str, float]] = {}

    with torch.no_grad():
        for batch in test_loader:
            case_name = batch["case_name"][0]
            coords_std = batch["coords"].to(device)
            design_std = batch["design"].to(device)
            stress_std = batch["stress"].to(device)

            pred_std = model.predict(coords_std, design_std)
            pred = scalers["stress_scaler"].inverse_transform(pred_std).cpu().squeeze(0)
            target = scalers["stress_scaler"].inverse_transform(stress_std).cpu().squeeze(0)
            coords_raw = scalers["coord_scaler"].inverse_transform(coords_std).cpu().squeeze(0)

            metrics = metrics_evaluator.compute(pred, target)
            case_metrics[case_name] = metrics

            logger.info(
                f"case {hue.b}{case_name}{hue.q} | "
                f"{hue.c}Stress:{hue.q} "
                f"MSE={hue.m}{metrics['mse']:.4e}{hue.q}, "
                f"R2={hue.m}{metrics['r2']:.4f}{hue.q}, "
                f"ACC={hue.m}{metrics['accuracy']:.2f}%{hue.q}"
            )

            torch.save(pred, output_dir / f"{case_name}_pred.pt")
            if visualizer is not None:
                visualizer.compare_fields(
                    gt=target,
                    pred=pred,
                    coords=coords_raw,
                    case_name=case_name,
                )

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(case_metrics, file, indent=2)

    if args.render_metric_plots:
        from data.field_plot import plot_metrics_summary, plot_training_curves

        history_path = output_dir / "history.json"
        if history_path.exists():
            plot_training_curves(history_path, output_dir / "training_curve.png")
        plot_metrics_summary(output_dir / "test_metrics.json", output_dir / "metrics_summary.png")
    logger.info(f"{hue.g}inference completed.{hue.q}")


def probe_pipeline(args) -> None:
    """Run one forward-backward step to estimate peak GPU memory usage."""
    if not torch.cuda.is_available():
        logger.warning("No CUDA device detected. Probe is skipped on CPU.")
        return

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    split_manifest = _resolve_split_manifest(args, output_dir)
    train_raw, val_raw, test_raw = _build_datasets(args, split_manifest)
    train_loader, _, _, scalers = _build_loaders(args, train_raw, val_raw, test_raw)

    batch = next(iter(train_loader))
    coords = batch["coords"].to(args.device)
    design = batch["design"].to(args.device)
    target = batch["stress"].to(args.device)

    model = _build_model(args).to(args.device).train()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = MSECriterion()

    torch.cuda.reset_peak_memory_stats(args.device)

    optimizer.zero_grad(set_to_none=True)
    pred = model(coords, design)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()

    peak_memory = torch.cuda.max_memory_allocated(args.device)
    total_memory = torch.cuda.get_device_properties(args.device).total_memory
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
    arguments = config.get_args()

    if "probe" in arguments.mode:
        probe_pipeline(arguments)
    if "train" in arguments.mode:
        train_pipeline(arguments)
    if "infer" in arguments.mode:
        inference_pipeline(arguments)
