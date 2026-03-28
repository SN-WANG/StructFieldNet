# Plotting helpers for StructFieldNet
# Author: Shengning Wang

import json
import os
from pathlib import Path
import tempfile
from typing import Dict, Union

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "structfieldnet_mpl"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    history_path: Union[str, Path],
    output_path: Union[str, Path],
) -> None:
    """Plot training and validation loss curves from ``history.json``."""
    history_path = Path(history_path)
    if not history_path.exists():
        raise FileNotFoundError(f"history file not found: {history_path}")

    with open(history_path, "r", encoding="utf-8") as file:
        history = json.load(file)

    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item.get("val_loss") for item in history if "val_loss" in item]
    lr = [item["lr"] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2.0)
    if val_loss:
        axes[0].plot(epochs[: len(val_loss)], val_loss, label="Val Loss", color="#d62728", linewidth=2.0)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Curve")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, lr, color="#2ca02c", linewidth=2.0)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_summary(
    metrics_path: Union[str, Path],
    output_path: Union[str, Path],
) -> None:
    """Plot per-case MSE and R2 from ``test_metrics.json``."""
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics_dict: Dict[str, Dict[str, float]] = json.load(file)

    case_names = list(metrics_dict.keys())
    mse = np.asarray([metrics_dict[name]["mse"] for name in case_names], dtype=np.float64)
    r2 = np.asarray([metrics_dict[name]["r2"] for name in case_names], dtype=np.float64)
    accuracy = np.asarray([metrics_dict[name]["accuracy"] for name in case_names], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].bar(case_names, mse, color="#1f77b4")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Per-Case Mean Squared Error")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(case_names, r2, color="#ff7f0e")
    axes[1].set_ylabel("R2")
    axes[1].set_title("Per-Case Coefficient of Determination")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(case_names, accuracy, color="#2ca02c")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_title("Per-Case Accuracy")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].set_xticks(range(len(case_names)))
    axes[2].set_xticklabels(case_names, rotation=90)

    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
