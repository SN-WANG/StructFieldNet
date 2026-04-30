# Main script for StructFieldNet probing, training and inference
# Author: Shengning Wang

import json
from argparse import Namespace
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from data.field_data import (
    FieldData,
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


def train_pipeline(args: Namespace) -> None:
    """
    Run the training pipeline.

    Args:
        args (Namespace): Parsed experiment arguments.
    """
    logger.info(f"{hue.c}============================= [TRAIN PIPELINE] START =============================={hue.q}")
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable, falling back to CPU.")
        device_name = "cpu"

    split_path = output_dir / "splits.json"
    if split_path.exists():
        logger.info("loading existing split manifest...")
        split_manifest = load_split_manifest(split_path)
    else:
        split_manifest = build_case_splits(
            data_dir=args.data_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    dataset = FieldData.from_directory(data_dir=args.data_dir, verify_fixed_mesh=args.verify_fixed_mesh)
    train_data, val_data, test_data = dataset.split(split_manifest)
    scalers = fit_scalers(
        dataset=train_data,
        coord_norm_range=args.coord_norm_range,
        normalize_design=args.normalize_design,
        normalize_stress=args.normalize_stress,
        stress_channel_dim=args.stress_channel_dim,
    )
    train_loader, val_loader, _ = build_scaled_loaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        scalers=scalers,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    model_args = {
        "num_nodes": train_data.meta.num_nodes,
        "coord_dim": train_data.meta.coord_dim,
        "design_dim": train_data.meta.design_dim,
        "output_dim": train_data.meta.output_dim,
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
    model = StructFieldNet(**model_args)

    design_scaler = scalers.get("design_scaler")
    stress_scaler = scalers.get("stress_scaler")
    design = design_scaler.transform(train_data.designs) if design_scaler is not None else train_data.designs
    stress = stress_scaler.transform(train_data.stresses) if stress_scaler is not None else train_data.stresses
    model.initialize_basis(design=design, stress=stress)

    if args.compile_model and hasattr(torch, "compile"):
        logger.info("compiling model with torch.compile...")
        model = torch.compile(model)

    num_params = sum(parameter.numel() for parameter in model.parameters())
    logger.info(f"model has {hue.m}{num_params}{hue.q} parameters")

    params = {
        "model_args": model_args,
        "split_manifest": split_manifest,
        "data_meta": train_data.meta.__dict__,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)
    save_split_manifest(split_manifest, output_dir / "splits.json")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.eta_min)
    trainer = FieldTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        params=params,
        scalers=scalers,
        output_dir=output_dir,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=device_name,
    )
    trainer.fit(train_loader, val_loader)
    logger.info(f"{hue.g}============================== [TRAIN PIPELINE] END ==============================={hue.q}")


def inference_pipeline(args: Namespace) -> None:
    """
    Run checkpoint-based inference on the test split.

    Args:
        args (Namespace): Parsed experiment arguments.
    """
    logger.info(f"{hue.c}============================= [INFER PIPELINE] START =============================={hue.q}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable, falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    checkpoint = None
    checkpoint_path = output_dir / "best.pt"
    if checkpoint_path.exists():
        logger.info(f"loading checkpoint: {hue.b}{checkpoint_path.name}{hue.q}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint_path = output_dir / "ckpt.pt"
        if checkpoint_path.exists():
            logger.info(f"loading checkpoint: {hue.b}{checkpoint_path.name}{hue.q}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in {output_dir}")

    checkpoint_params = checkpoint.get("params", {})
    scalers = restore_scalers(
        scaler_state_dict=checkpoint["scaler_state_dict"],
        coord_norm_range=args.coord_norm_range,
        normalize_design=args.normalize_design,
        normalize_stress=args.normalize_stress,
        stress_channel_dim=args.stress_channel_dim,
    )

    split_path = output_dir / "splits.json"
    if split_path.exists():
        logger.info("loading existing split manifest...")
        split_manifest = load_split_manifest(split_path)
    elif checkpoint_params.get("split_manifest") is not None:
        logger.info("using split manifest restored from checkpoint...")
        split_manifest = checkpoint_params["split_manifest"]
    else:
        split_manifest = build_case_splits(
            data_dir=args.data_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    dataset = FieldData.from_directory(data_dir=args.data_dir, verify_fixed_mesh=args.verify_fixed_mesh)
    train_data, val_data, test_data = dataset.split(split_manifest)
    _, _, test_loader = build_scaled_loaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        scalers=scalers,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    model_args = checkpoint_params.get("model_args")
    if model_args is None:
        model_args = {
            "num_nodes": test_data.meta.num_nodes,
            "coord_dim": test_data.meta.coord_dim,
            "design_dim": test_data.meta.design_dim,
            "output_dim": test_data.meta.output_dim,
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
    model = StructFieldNet(**model_args)

    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    if args.compile_model and hasattr(torch, "compile"):
        logger.info("compiling model with torch.compile...")
        model = torch.compile(model)
    model.to(device)
    model.eval()

    metrics = FieldMetrics(hotspot_percentile=args.hotspot_percentile)
    if args.render_visualization:
        from data.field_vis import FieldVis

        visualizer = FieldVis(
            output_dir=output_dir,
            off_screen=args.off_screen,
            point_size=args.render_point_size,
            screenshot_scale=args.screenshot_scale,
        )
    else:
        visualizer = None

    coord_scaler = scalers.get("coord_scaler")
    stress_scaler = scalers.get("stress_scaler")
    case_metrics = {}
    comparison_paths = []
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

            metric_values = metrics.compute(pred, target)
            case_metrics[case_name] = metric_values
            logger.info(
                f"case {hue.b}{case_name}{hue.q} | "
                f"MSE={hue.m}{metric_values['mse']:.4e}{hue.q}, "
                f"R2={hue.m}{metric_values['r2']:.4f}{hue.q}, "
                f"ACC={hue.m}{metric_values['accuracy']:.2f}%{hue.q}"
            )

            torch.save(pred, output_dir / f"{case_name}_pred.pt")
            if visualizer is not None:
                comparison_path = visualizer.compare_fields(
                    gt=target,
                    pred=pred,
                    coords=coords,
                    case_name=case_name,
                )
                comparison_paths.append(comparison_path)

    if visualizer is not None and args.render_video:
        movie_path = visualizer.save_comparison_movie(
            frame_paths=comparison_paths,
            output_path=output_dir / "inference_comparison_loop.mp4",
            fps=args.video_fps,
        )
        logger.info(f"saved inference animation: {hue.b}{movie_path.name}{hue.q}")

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(case_metrics, file, indent=2)

    metric_names = sorted(next(iter(case_metrics.values())).keys())
    summary = {
        metric_name: float(sum(item[metric_name] for item in case_metrics.values()) / len(case_metrics))
        for metric_name in metric_names
    }
    with open(output_dir / "test_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    if args.render_metric_plots:
        from data.field_plot import plot_metrics_summary, plot_training_curves

        history_path = output_dir / "history.json"
        if history_path.exists():
            plot_training_curves(history_path, output_dir / "training_curve.png")
        plot_metrics_summary(output_dir / "test_metrics.json", output_dir / "metrics_summary.png")

    logger.info(
        f"{hue.g}test summary{hue.q} | "
        f"MSE={hue.m}{summary['mse']:.4e}{hue.q}, "
        f"R2={hue.m}{summary['r2']:.4f}{hue.q}, "
        f"ACC={hue.m}{summary['accuracy']:.2f}%{hue.q}"
    )
    logger.info(f"{hue.g}============================== [INFER PIPELINE] END ==============================={hue.q}")


def probe_pipeline(args: Namespace) -> None:
    """
    Run one forward-backward pass to estimate peak GPU memory.

    Args:
        args (Namespace): Parsed experiment arguments.
    """
    logger.info(f"{hue.c}============================= [PROBE PIPELINE] START =============================={hue.q}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable, falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    if not device_name.startswith("cuda"):
        logger.warning("No CUDA device detected. Probe is skipped on CPU.")
        logger.info(f"{hue.g}============================== [PROBE PIPELINE] END ==============================={hue.q}")
        return

    seed_everything(args.seed)

    split_path = output_dir / "splits.json"
    if split_path.exists():
        logger.info("loading existing split manifest...")
        split_manifest = load_split_manifest(split_path)
    else:
        split_manifest = build_case_splits(
            data_dir=args.data_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    dataset = FieldData.from_directory(data_dir=args.data_dir, verify_fixed_mesh=args.verify_fixed_mesh)
    train_data, val_data, test_data = dataset.split(split_manifest)
    scalers = fit_scalers(
        dataset=train_data,
        coord_norm_range=args.coord_norm_range,
        normalize_design=args.normalize_design,
        normalize_stress=args.normalize_stress,
        stress_channel_dim=args.stress_channel_dim,
    )
    train_loader, _, _ = build_scaled_loaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        scalers=scalers,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    batch = next(iter(train_loader))

    model_args = {
        "num_nodes": train_data.meta.num_nodes,
        "coord_dim": train_data.meta.coord_dim,
        "design_dim": train_data.meta.design_dim,
        "output_dim": train_data.meta.output_dim,
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
    model = StructFieldNet(**model_args)

    design_scaler = scalers.get("design_scaler")
    stress_scaler = scalers.get("stress_scaler")
    design = design_scaler.transform(train_data.designs) if design_scaler is not None else train_data.designs
    stress = stress_scaler.transform(train_data.stresses) if stress_scaler is not None else train_data.stresses
    model.initialize_basis(design=design, stress=stress)

    if args.compile_model and hasattr(torch, "compile"):
        logger.info("compiling model with torch.compile...")
        model = torch.compile(model)
    model = model.to(device).train()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    coords = batch["coords"].to(device)
    design = batch["design"].to(device)
    target = batch["stress"].to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(coords, design), target)
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
        f"coord_dim={hue.m}{train_data.coord_dim}{hue.q}, "
        f"design_dim={hue.m}{train_data.design_dim}{hue.q}, "
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


if __name__ == "__main__":
    args = config.get_args()

    if "probe" in args.mode:
        probe_pipeline(args)

    if "train" in args.mode:
        train_pipeline(args)

    if "infer" in args.mode:
        inference_pipeline(args)
