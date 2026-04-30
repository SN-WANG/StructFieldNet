# Argument configuration for StructFieldNet
# Author: Shengning Wang

import argparse

import torch


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments for StructFieldNet workflows.

    Returns:
        argparse.Namespace: Parsed experiment arguments.
    """
    parser = argparse.ArgumentParser(
        description="StructFieldNet: Design-conditioned structural field reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ============================================================
    # 1. General
    # ============================================================

    general = parser.add_argument_group("General")
    general.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    general.add_argument("--data_dir", type=str, default="./dataset", help="Directory containing structural cases.")
    general.add_argument("--output_dir", type=str, default="./runs", help="Directory for checkpoints and outputs.")
    general.add_argument(
        "--mode",
        type=str,
        nargs="+",
        default=["probe", "train", "infer"],
        choices=["probe", "train", "infer"],
        help="Execution phases to run.",
    )
    general.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device.",
    )

    # ============================================================
    # 2. Data
    # ============================================================

    data = parser.add_argument_group("Data")
    data.add_argument("--coord_dim", type=int, default=3, help="Expected coordinate dimension.")
    data.add_argument("--design_dim", type=int, default=25, help="Expected design vector dimension.")
    data.add_argument("--output_dim", type=int, default=1, help="Expected output field dimension.")
    data.add_argument("--train_ratio", type=float, default=0.70, help="Training split ratio.")
    data.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio.")
    data.add_argument("--test_ratio", type=float, default=0.15, help="Test split ratio.")
    data.add_argument("--batch_size", type=int, default=4, help="Mini-batch size.")
    data.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers.")
    data.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable pinned-memory DataLoader transfer.",
    )
    data.add_argument(
        "--verify_fixed_mesh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check whether all cases share one fixed mesh.",
    )
    data.add_argument(
        "--coord_norm_range",
        type=str,
        default="bipolar",
        choices=["unit", "bipolar"],
        help="Coordinate normalization range.",
    )
    data.add_argument(
        "--normalize_design",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize design vectors.",
    )
    data.add_argument(
        "--normalize_stress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize target fields.",
    )
    data.add_argument("--stress_channel_dim", type=int, default=-1, help="Channel dimension for stress scaling.")

    # ============================================================
    # 3. StructFieldNet
    # ============================================================

    model = parser.add_argument_group("StructFieldNet")
    model.add_argument("--model_type", type=str, default="structfieldnet", choices=["structfieldnet"])
    model.add_argument("--depth", type=int, default=4, help="Number of operator blocks.")
    model.add_argument("--width", type=int, default=64, help="Hidden token width.")
    model.add_argument("--num_slices", type=int, default=32, help="Number of slice tokens.")
    model.add_argument("--num_heads", type=int, default=4, help="Number of attention heads.")
    model.add_argument("--num_bases", type=int, default=32, help="Number of fixed-mesh basis fields.")
    model.add_argument("--mlp_ratio", type=int, default=4, help="Feed-forward expansion ratio.")
    model.add_argument("--branch_hidden_dim", type=int, default=64, help="Design encoder hidden width.")
    model.add_argument("--branch_layers", type=int, default=1, help="Number of hidden layers in the design encoder.")
    model.add_argument("--trunk_hidden_dim", type=int, default=64, help="Coordinate encoder hidden width.")
    model.add_argument("--trunk_layers", type=int, default=1, help="Number of hidden layers in the coordinate encoder.")
    model.add_argument("--lifting_hidden_dim", type=int, default=64, help="Fusion MLP hidden width.")
    model.add_argument("--lifting_layers", type=int, default=1, help="Number of hidden layers in the fusion MLP.")
    model.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")

    # ============================================================
    # 4. Trainer
    # ============================================================

    trainer = parser.add_argument_group("Trainer")
    trainer.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate for AdamW.")
    trainer.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay.")
    trainer.add_argument("--max_epochs", type=int, default=40, help="Maximum training epochs.")
    trainer.add_argument("--patience", type=int, default=6, help="Early stopping patience.")
    trainer.add_argument("--eta_min", type=float, default=1e-6, help="Minimum cosine learning rate.")
    trainer.add_argument(
        "--compile_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the model with torch.compile.",
    )

    # ============================================================
    # 5. Evaluation
    # ============================================================

    evaluation = parser.add_argument_group("Evaluation")
    evaluation.add_argument(
        "--hotspot_percentile",
        type=float,
        default=0.95,
        help="Percentile threshold used in hotspot overlap metrics.",
    )

    # ============================================================
    # 6. Visualization
    # ============================================================

    visualization = parser.add_argument_group("Visualization")
    visualization.add_argument(
        "--off_screen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use off-screen rendering.",
    )
    visualization.add_argument(
        "--render_visualization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render per-case field visualizations.",
    )
    visualization.add_argument(
        "--render_metric_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render training and metrics summary plots.",
    )
    visualization.add_argument(
        "--render_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render an MP4 animation from all per-case comparison figures.",
    )
    visualization.add_argument("--video_fps", type=float, default=2.0, help="Frame rate for the MP4 animation.")
    visualization.add_argument("--render_point_size", type=float, default=7.0, help="Point size in visualization.")
    visualization.add_argument("--screenshot_scale", type=int, default=1, help="Screenshot resolution scale.")

    args = parser.parse_args()
    if args.width % args.num_heads != 0:
        raise ValueError("width must be divisible by num_heads")
    return args
