"""Seeds, set in one place."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, *, deterministic_torch: bool = False) -> None:
    """Set the seeds of random, numpy and torch (if installed).

    ``PYTHONHASHSEED`` here only affects subprocesses: string hashing in the current process is fixed when it
    starts. The code does not depend on the order of ``set`` elements.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rng(seed: int, *salt: int) -> np.random.Generator:
    """Numpy generator derived deterministically from a seed and a salt."""
    return np.random.default_rng([seed, *salt])
