"""Teste L1 (stadio 30): regressione logistica multinomiale sulle CLS congelate.

Per ogni fold k:
- fit sulle clip di train con fold ≠ k, escluso l'holdout interno; C scelto in CV
  interna (raggruppata per take); feature standardizzate;
- temperatura stimata sull'holdout interno;
- predizioni fuori campione sulla fold k, e su val/test.
Su val e test: media delle probabilità calibrate dei 5 modelli di fold (e media dei
logit, salvata come ``vid_logits``).
"""

from __future__ import annotations

import logging
import warnings
from typing import List

import numpy as np
import pandas as pd

from src.cascade.calibration import apply_temperature, fit_temperature, nll_from_logits

log = logging.getLogger(__name__)


def fit_head(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    Cs: List[float],
    inner_cv: int,
    max_iter: int,
    n_classes: int,
    seed: int = 0,
):
    from sklearn.model_selection import StratifiedGroupKFold

    best_C, best = Cs[0], None
    n_splits = min(inner_cv, int(np.bincount(y, minlength=n_classes)[np.bincount(y, minlength=n_classes) > 0].min()))
    if len(Cs) > 1 and n_splits >= 2:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for C in Cs:
            losses = []
            for tr, te in cv.split(X, y, groups):
                m = _make(C, max_iter).fit(X[tr], y[tr])
                losses.append(nll_from_logits(_logits(m, X[te], n_classes), y[te]))
            score = float(np.mean(losses))
            if best is None or score < best:
                best_C, best = C, score
    return _make(best_C, max_iter).fit(X, y), best_C


def _make(C: float, max_iter: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    class _Quiet:
        def __init__(self, pipe):
            self.pipe = pipe

        def fit(self, X, y):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.pipe.fit(X, y)
            return self

        def __getattr__(self, k):
            return getattr(self.pipe, k)

    return _Quiet(make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter)))


def _logits(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Logit ``[N, C]`` anche se nel training mancava qualche classe (−inf → −50)."""
    raw = model.decision_function(X)
    classes = model.pipe[-1].classes_
    if raw.ndim == 1:  # caso binario
        raw = np.stack([-raw, raw], 1) / 2
    out = np.full((len(X), n_classes), -50.0)
    out[:, classes] = raw
    return out


def run_heads(feats: pd.DataFrame, X: np.ndarray, folds: pd.DataFrame, hcfg, n_classes: int, seed: int = 0):
    """``feats``: clip_id, take_uid, split, label_idx allineati a ``X``.

    Restituisce (oof_train, val, test) come dict di array e un riepilogo per fold.
    """
    fold_of = dict(zip(folds.take_uid, folds.fold))
    y = feats.label_idx.to_numpy().astype(int)
    split = feats.split.to_numpy()
    take = feats.take_uid.to_numpy()
    fold = np.array([fold_of.get(t, -1) for t in take])
    is_train = split == "train"
    if (fold[is_train] < 0).any():
        raise ValueError("clip di train senza fold in folds.csv")
    K = int(folds.fold.max()) + 1
    oof_logits = np.zeros((len(feats), n_classes))
    oof_probs = np.zeros((len(feats), n_classes))
    oof_T = np.full(len(feats), np.nan)
    ev = ~is_train
    ev_logits = np.zeros((K, ev.sum(), n_classes))
    ev_probs = np.zeros((K, ev.sum(), n_classes))
    summary = []
    for k in range(K):
        hold_takes = set(folds.take_uid[folds[f"holdout_f{k}"].to_numpy(bool)])
        hold = is_train & np.isin(take, list(hold_takes))
        fit = is_train & (fold != k) & ~hold
        held = is_train & (fold == k)
        model, C = fit_head(X[fit], y[fit], take[fit], hcfg.Cs, hcfg.inner_cv, hcfg.max_iter, n_classes, seed)
        T = fit_temperature(_logits(model, X[hold], n_classes), y[hold]) if hold.any() else 1.0
        lo = _logits(model, X[held], n_classes)
        oof_logits[held], oof_probs[held], oof_T[held] = lo, apply_temperature(lo, T), T
        le = _logits(model, X[ev], n_classes)
        ev_logits[k], ev_probs[k] = le, apply_temperature(le, T)
        acc = float((lo.argmax(1) == y[held]).mean()) if held.any() else float("nan")
        summary.append(
            {"fold": k, "C": C, "T": T, "n_fit": int(fit.sum()), "n_holdout": int(hold.sum()), "oof_acc": acc}
        )
        log.info("testa lineare fold %d: C=%g T=%.3f acc OOF=%.4f", k, C, T, acc)
    out = {
        "vid_logits": oof_logits,
        "vid_probs_cal": oof_probs,
        "T_vid": oof_T,
    }
    out["vid_logits"][ev] = ev_logits.mean(0)
    out["vid_probs_cal"][ev] = ev_probs.mean(0)
    out["T_vid"][ev] = float(np.mean([s["T"] for s in summary]))
    out["vid_fold"] = np.where(is_train, fold, -1)
    return out, pd.DataFrame(summary)
