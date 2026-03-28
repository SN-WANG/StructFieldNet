# Argument configuration for StructFieldNet
# Author: Shengning Wang

import argparse

import torch


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for StructFieldNet."""
    parser = argparse.ArgumentParser(
        description="StructFieldNet: Design-conditioned structural field reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    general = parser.add_argument_group("General")
    general.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    general.add_argument(
        "--output_dir",
        type=str,
        default="./runs",
        help="Directory to save checkpoints, logs, metrics, and figures.",
    )
    general.add_argument(
        "--mode",
        type=str,
        nargs="+",
        default=["train", "infer", "probe"],
        choices=["train", "infer", "probe"],
        help="Execution phases to run.",
    )
    general.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device.",
    )

    data = parser.add_argument_group("Data")
    data.add_argument("--data_dir", type=str, default="./dataset", help="Dataset root directory.")
    data.add_argument("--coord_dim", type=int, default=3, help="Coordinate dimension.")
    data.add_argument("--design_dim", type=int, default=25, help="Thickness design vector dimension.")
    data.add_argument("--output_dim", type=int, default=1, help="Output field dimension.")
    data.add_argument("--train_ratio", type=float, default=0.70, help="Training split ratio.")
    data.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio.")
    data.add_argument("--test_ratio", type=float, default=0.15, help="Test split ratio.")
    data.add_argument("--batch_size", type=int, default=2, help="Mini-batch size.")
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
        help="Check that all cases share the same reference mesh.",
    )
    data.add_argument(
        "--coord_norm_range",
        type=str,
        default="bipolar",
        choices=["unit", "bipolar"],
        help="Target normalization range for coordinates.",
    )
    data.add_argument(
        "--normalize_design",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize the thickness design vector.",
    )
    data.add_argument(
        "--normalize_stress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize the scalar stress target.",
    )
    data.add_argument(
        "--stress_channel_dim",
        type=int,
        default=1,
        help=(
            "Channel dimension used for stress standardization. "
            "For stress tensors shaped (num_cases, num_nodes, num_channels), "
            "the default 1 performs node-wise normalization on a fixed mesh."
        ),
    )

    model = parser.add_argument_group("Model")
    model.add_argument(
        "--model_type",
        type=str,
        default="structfieldnet",
        choices=["structfieldnet"],
        help="Neural operator architecture.",
    )
    model.add_argument("--depth", type=int, default=4, help="Number of Physics-Attention blocks.")
    model.add_argument("--width", type=int, default=64, help="Hidden feature dimension.")
    model.add_argument("--num_slices", type=int, default=32, help="Number of slice tokens.")
    model.add_argument("--num_heads", type=int, default=4, help="Number of attention heads.")
    model.add_argument("--mlp_ratio", type=int, default=4, help="Expansion ratio in the feed-forward sublayer.")
    model.add_argument("--branch_hidden_dim", type=int, default=64, help="Hidden dimension of the branch MLP.")
    model.add_argument("--branch_layers", type=int, default=2, help="Number of branch MLP layers.")
    model.add_argument("--trunk_hidden_dim", type=int, default=64, help="Hidden dimension of the trunk MLP.")
    model.add_argument("--trunk_layers", type=int, default=2, help="Number of trunk MLP layers.")
    model.add_argument("--lifting_hidden_dim", type=int, default=64, help="Hidden dimension of the lifting MLP.")
    model.add_argument("--lifting_layers", type=int, default=2, help="Number of lifting MLP layers.")
    model.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")

    optim = parser.add_argument_group("Optimization")
    optim.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate for AdamW.")
    optim.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay coefficient.")
    optim.add_argument("--max_epochs", type=int, default=100, help="Maximum training epochs.")
    optim.add_argument("--patience", type=int, default=80, help="Early-stopping patience.")
    optim.add_argument("--eta_min", type=float, default=1e-6, help="Minimum cosine-annealing learning rate.")
    optim.add_argument(
        "--gradient_clip_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm. Set <= 0 to disable.",
    )
    optim.add_argument(
        "--use_amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic mixed precision on CUDA.",
    )
    optim.add_argument(
        "--compile_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the model with torch.compile when available.",
    )

    evaluation = parser.add_argument_group("Evaluation")
    evaluation.add_argument(
        "--hotspot_percentile",
        type=float,
        default=0.95,
        help="Percentile threshold for hotspot-oriented evaluation.",
    )

    visualization = parser.add_argument_group("Visualization")
    visualization.add_argument(
        "--mesh_mode",
        type=str,
        default="auto",
        choices=["auto", "delaunay", "point_cloud"],
        help="Mesh rendering strategy used by PyVista.",
    )
    visualization.add_argument(
        "--off_screen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use off-screen rendering for headless environments.",
    )
    visualization.add_argument(
        "--render_visualization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate PyVista comparison figures during inference.",
    )
    visualization.add_argument(
        "--render_metric_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate Matplotlib training and metric summary figures.",
    )
    visualization.add_argument(
        "--render_point_size",
        type=float,
        default=8.0,
        help="Point size when point-cloud rendering is used.",
    )
    visualization.add_argument(
        "--screenshot_scale",
        type=int,
        default=1,
        help="Supersampling scale factor for saved screenshots.",
    )

    return parser.parse_args()
