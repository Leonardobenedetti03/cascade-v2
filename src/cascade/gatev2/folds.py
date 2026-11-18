"""Partizioni del gate v2: stratificate per classe, raggruppate per sessione (nessuna sessione in due fold)."""

from __future__ import annotations

import warnings

import numpy as np


def session_folds(y: np.ndarray, sessions: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold ``[N]`` in ``0..K-1`` con ``StratifiedGroupKFold`` (classe = ``y``, gruppo = sessione).

    Una sessione con take di più classi va dove la stratificazione la mette, intera. Le classi con meno
    sessioni che fold non compaiono in tutte le fold (avviso di scikit-learn, silenziato: è atteso)."""
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
    """Errore se una sessione compare in più di una fold."""
    s = np.asarray(sessions).astype(str)
    per = {}
    for f, g in zip(fold, s):
        per.setdefault(g, set()).add(int(f))
    bad = [g for g, fs in per.items() if len(fs) > 1]
    if bad:
        raise RuntimeError(f"sessioni in più fold: {bad[:5]}")
