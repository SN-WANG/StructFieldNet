# Reproducibility utilities for machine learning experiments
# Author: Shengning Wang

import random

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is an explicit dependency
    torch = None
    _HAS_TORCH = False

from utils.hue_logger import hue, logger


def seed_everything(seed: int = 42) -> None:
    """Set all available random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"global seed set to {hue.m}{seed}{hue.q}")
