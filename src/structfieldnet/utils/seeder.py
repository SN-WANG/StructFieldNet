# Reproducibility utilities for machine learning experiments
# Author: Shengning Wang

import random
import numpy as np
import torch

from structfieldnet.utils.hue_logger import hue, logger


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch random states.

    Args:
        seed: Global random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"global seed set to {hue.m}{seed}{hue.q}")
