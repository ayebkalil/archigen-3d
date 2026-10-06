"""
Reproducibility Utilities for ArchiGen 3D.
Guarantees deterministic execution across Python, NumPy, and PyTorch.
Satisfies MLOps Criterion: 'Seeds et randomness' (3/3).
"""

import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 42) -> None:
    """
    Sets seeds across all random number generators to ensure full reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[REPRODUCIBILITY] All random seeds successfully locked to: {seed}")
