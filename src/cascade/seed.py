"""Gestione centralizzata dei seed."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, *, deterministic_torch: bool = False) -> None:
    """Fissa i seed di random, numpy e (se installato) torch.

    ``PYTHONHASHSEED`` qui vale solo per eventuali sottoprocessi: l'hashing delle stringhe del
    processo corrente è già fissato all'avvio. Il codice non dipende dall'ordine dei ``set``.
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
    """Generatore numpy derivato in modo deterministico da seed e sale."""
    return np.random.default_rng([seed, *salt])
