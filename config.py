# Argument Configuration for StructFieldNet
# Author: Shengning Wang

import argparse
import torch


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for StructFieldNet."""
    parser = argparse.ArgumentParser(
        description="StructFieldNet: Design-conditioned structural field reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add = parser.add_argument

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------
    add("--seed", type=int, default=42, help="Random seed.")
    add("--output_dir", type=str, default="./runs", help="Output directory.")
    add(
        "--mode",
        type=str,
        nargs="+",
        default=["train", "infer", "probe"],
        choices=["train", "infer", "probe"],
        help="Pipelines to run.",
    )
    add(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device.",
    )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    add("--data_dir", type=str, default="./dataset", help="Dataset directory.")
    add("--coord_dim", type=int, default=3, help="Coordinate dimension.")
    add("--design_dim", type=int, default=25, help="Design dimension.")
    add("--output_dim", type=int, default=1, help="Field output dimension.")
    add("--train_ratio", type=float, default=0.70, help="Train split ratio.")
    add("--val_ratio", type=float, default=0.15, help="Validation split ratio.")
    add("--test_ratio", type=float, default=0.15, help="Test split ratio.")
    add("--batch_size", type=int, default=4, help="Mini-batch size.")
    add("--num_workers", type=int, default=0, help="DataLoader workers.")
    add(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable pinned memory.",
    )
    add(
        "--verify_fixed_mesh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check fixed-mesh consistency.",
    )
    add(
        "--coord_norm_range",
        type=str,
        default="bipolar",
        choices=["unit", "bipolar"],
        help="Coordinate normalization range.",
    )
    add(
        "--normalize_design",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize design vectors.",
    )
    add(
        "--normalize_stress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize target fields.",
    )
    add("--stress_channel_dim", type=int, default=-1, help="Channel dimension for stress scaling.")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    add("--model_type", type=str, default="structfieldnet", choices=["structfieldnet"])
    add("--depth", type=int, default=4, help="Number of operator blocks.")
    add("--width", type=int, default=64, help="Hidden width.")
    add("--num_slices", type=int, default=32, help="Number of slice tokens.")
    add("--num_heads", type=int, default=4, help="Number of attention heads.")
    add("--num_bases", type=int, default=32, help="Number of fixed-mesh basis fields.")
    add("--mlp_ratio", type=int, default=4, help="FFN expansion ratio.")
    add("--branch_hidden_dim", type=int, default=64, help="Design encoder width.")
    add("--branch_layers", type=int, default=2, help="Design encoder depth.")
    add("--trunk_hidden_dim", type=int, default=64, help="Coordinate encoder width.")
    add("--trunk_layers", type=int, default=2, help="Coordinate encoder depth.")
    add("--lifting_hidden_dim", type=int, default=64, help="Fusion width.")
    add("--lifting_layers", type=int, default=2, help="Fusion depth.")
    add("--dropout", type=float, default=0.0, help="Dropout.")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    add("--lr", type=float, default=2e-4, help="Learning rate.")
    add("--weight_decay", type=float, default=1e-5, help="Weight decay.")
    add("--max_epochs", type=int, default=40, help="Maximum epochs.")
    add("--patience", type=int, default=6, help="Early stopping patience.")
    add("--eta_min", type=float, default=1e-6, help="Minimum learning rate.")
    add(
        "--gradient_clip_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm.",
    )
    add(
        "--use_amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable AMP on CUDA.",
    )
    add(
        "--compile_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use torch.compile.",
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    add(
        "--hotspot_percentile",
        type=float,
        default=0.95,
        help="Percentile for hotspot overlap.",
    )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    add(
        "--off_screen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use off-screen rendering.",
    )
    add(
        "--render_visualization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render case figures.",
    )
    add(
        "--render_metric_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render metric plots.",
    )
    add("--render_point_size", type=float, default=7.0, help="Point size.")
    add("--screenshot_scale", type=int, default=1, help="Screenshot scale.")

    args = parser.parse_args()

    split_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(split_sum - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + val_ratio + test_ratio must be 1.0, got {split_sum:.6f}")
    if args.patience >= args.max_epochs:
        raise ValueError("patience must be smaller than max_epochs")
    if args.width % args.num_heads != 0:
        raise ValueError("width must be divisible by num_heads")

    return args
