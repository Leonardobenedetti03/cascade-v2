"""Temperature scaling (Guo et al., 2017) e metriche di calibrazione.

La temperatura T > 0 si stima minimizzando la NLL di ``softmax(logits / T)``
sull'holdout interno della fold; non cambia l'argmax.
"""

from __future__ import annotations

import numpy as np

from src.cascade.fusion import log_softmax


def nll_from_logits(logits: np.ndarray, y: np.ndarray, T: float = 1.0) -> float:
    lp = log_softmax(np.asarray(logits, dtype=np.float64) / T)
    return float(-lp[np.arange(len(y)), y].mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray, bounds=(0.05, 20.0)) -> float:
    from scipy.optimize import minimize_scalar

    if len(y) == 0:
        return 1.0
    res = minimize_scalar(
        lambda lt: nll_from_logits(logits, y, float(np.exp(lt))),
        bounds=(np.log(bounds[0]), np.log(bounds[1])),
        method="bounded",
    )
    return float(np.exp(res.x))


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return np.exp(log_softmax(np.asarray(logits, dtype=np.float64) / T))


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)
