# src/graph_enhanced_lm/utils/reproducibility.py
"""Deterministic seed management across random, numpy, and PyTorch."""
from __future__ import annotations

import os
import random
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds across standard Python, NumPy, and PyTorch if installed."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass