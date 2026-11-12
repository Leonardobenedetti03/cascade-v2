"""Fold stratificate per classe sulle take di train (stadio 05).

- ``n_folds`` fold stratificate per classe, una riga per take (quindi raggruppate per
  ``take_uid``: tutte le clip di una take stanno nella stessa fold), seed fisso;
- per ogni fold k, un holdout interno (``inner_holdout_frac``) estratto dalle fold
  di training (≠ k), stratificato, usato per early stopping e temperatura;
- un holdout interno anche per il modello ``full`` (tutte le take di train).

Output: una riga per take con ``take_uid, label_idx, [site,] fold, holdout_f0..holdout_f{K-1},
holdout_full``.

Raggruppamento (M03). ``group_by="take"`` è l'assegnazione stratificata della v1, invariata.
``group_by="site"``: tutte le take di un sito stanno nella stessa fold, così le predizioni fuori
campione simulano lo spostamento verso siti nuovi. La mappa sito -> fold è un dato versionato
(``split.json``, D6) e qui non si ricalcola: basta perdere una take perché un algoritmo di
bilanciamento sposti un sito in un'altra fold, e con lui tutte le predizioni fuori campione.

Riproducibilità: niente ``StratifiedGroupKFold`` (il suo shuffle cambia tra sklearn 1.6
e 1.9: stesse take, fold diverse) né ``np.random.default_rng`` (numpy non garantisce lo
stream di ``Generator`` tra versioni). Si usa ``np.random.RandomState``, il cui stream
è garantito stabile, e un'assegnazione round-robin per classe che bilancia classi e
dimensioni delle fold (scarto massimo 1).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _stratified_subset(labels: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Maschera di un sottoinsieme stratificato (almeno 1 per classe se possibile)."""
    rs = np.random.RandomState(seed)
    mask = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        k = int(round(frac * len(idx)))
        if k == 0 and len(idx) >= 2:
            k = 1
        mask[idx[rs.permutation(len(idx))[:k]]] = True
    return mask


def assign_folds(labels: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold per riga: permutazione per classe, poi round-robin con offset che continua tra
    le classi (così anche le dimensioni totali delle fold differiscono al più di 1)."""
    rs = np.random.RandomState(seed)
    fold = np.full(len(labels), -1, dtype=int)
    offset = 0
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        perm = idx[rs.permutation(len(idx))]
        fold[perm] = (offset + np.arange(len(idx))) % n_folds
        offset = (offset + len(idx)) % n_folds
    return fold


def assign_site_folds(sites: np.ndarray, site_to_fold: Dict[str, int], n_folds: int) -> np.ndarray:
    """Fold di ogni take = fold del suo sito. La mappa è un dato versionato (split.json):
    qui non si ricalcola nulla, si controlla solo che sia completa e coerente."""
    missing = sorted(set(sites) - set(site_to_fold))
    if missing:
        raise ValueError(f"siti di train senza fold in split.json: {missing}")
    fold = np.array([site_to_fold[s] for s in sites], dtype=int)
    if set(fold) != set(range(n_folds)):
        raise ValueError(f"fold usate {sorted(set(fold))}, attese 0..{n_folds - 1}")
    return fold


def make_folds(
    takes: pd.DataFrame,
    n_folds: int = 5,
    inner_holdout_frac: float = 0.1,
    seed: int = 0,
    *,
    group_by: str = "take",
    site_to_fold: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """``takes`` ha colonne ``take_uid, label_idx`` (una riga per take di train) e ``site`` con
    ``group_by="site"``. Se ``site`` è presente finisce anche nell'output."""
    if takes.take_uid.duplicated().any():
        raise ValueError("take_uid duplicati: le fold sono per take")
    if group_by not in ("take", "site"):
        raise ValueError(f"folds.group_by sconosciuto: {group_by!r}")
    if group_by == "site" and "site" not in takes.columns:
        raise ValueError("folds.group_by=site richiede la colonna 'site'")
    cols = ["take_uid", "label_idx"] + (["site"] if "site" in takes.columns else [])
    df = takes[cols].sort_values("take_uid", kind="mergesort").reset_index(drop=True)
    y = df.label_idx.to_numpy()
    if group_by == "take":
        fold = assign_folds(y, n_folds, seed)  # v1, invariato
    else:
        fold = assign_site_folds(df.site.to_numpy(), site_to_fold or {}, n_folds)
    df["fold"] = fold
    # Holdout interno: resta stratificato per classe tra le take delle fold di training.
    for k in range(n_folds):
        cand = np.flatnonzero(fold != k)
        sub = _stratified_subset(y[cand], inner_holdout_frac, seed * 1000 + k + 1)
        col = np.zeros(len(df), dtype=bool)
        col[cand[sub]] = True
        df[f"holdout_f{k}"] = col
    df["holdout_full"] = _stratified_subset(y, inner_holdout_frac, seed * 1000 + 999)
    return df


def fold_names(n_folds: int) -> list:
    return [str(k) for k in range(n_folds)] + ["full"]


def training_masks(folds: pd.DataFrame, fold: str):
    """(train_mask, holdout_mask, eval_mask) sulle righe di ``folds`` per una fold.

    ``eval_mask`` sono le take fuori campione (fold k); vuota per ``full``.
    """
    if fold == "full":
        hold = folds["holdout_full"].to_numpy(bool)
        return ~hold, hold, np.zeros(len(folds), dtype=bool)
    k = int(fold)
    in_k = folds["fold"].to_numpy() == k
    hold = folds[f"holdout_f{k}"].to_numpy(bool)
    return ~in_k & ~hold, hold, in_k
