"""Folds for the gate v2: stratified by class and grouped by session, so no session is split across folds."""

from __future__ import annotations

import warnings

import numpy as np


def session_folds(y: np.ndarray, sessions: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold ``[N]`` in ``0..K-1`` from ``StratifiedGroupKFold`` (class = ``y``, group = session).

    A session with takes of several classes goes, as a whole, wherever the stratification puts it. Classes with
    fewer sessions than folds cannot appear in every fold. scikit-learn warns about this; we expect it, so we
    silence the warning."""
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.asarray(y)
    groups = np.asarray(sessions).astype(str)
    n_groups = len(np.unique(groups))
    if n_groups < n_folds:
        raise ValueError(f"{n_groups} sessioni per {n_folds} fold")
    fold = np.full(len(y), -1, dtype=int)
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=int(seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for k, (_, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            fold[te] = k
    if (fold < 0).any():
        raise RuntimeError("take senza fold")
    return fold


def check_grouping(fold: np.ndarray, sessions: np.ndarray) -> None:
    """Raise an error if a session appears in more than one fold."""
    s = np.asarray(sessions).astype(str)
    per = {}
    for f, g in zip(fold, s):
        per.setdefault(g, set()).add(int(f))
    bad = [g for g, fs in per.items() if len(fs) > 1]
    if bad:
        raise RuntimeError(f"sessioni in più fold: {bad[:5]}")
