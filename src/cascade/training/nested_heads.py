"""Linear heads on the frozen embeddings, refit inside the nested CV (gate v2, N37).

Same protocol as ``scenario_preds/scenario/heads.py::fit_head`` (now ``training/heads.py``; we copied it, as we
did for ``compute_windows`` in the bridge stage 44): StandardScaler + multinomial logistic regression, C chosen
by inner CV grouped by session (log-loss), temperature fitted on the out-of-sample logits of the same CV with
the chosen C. Classes with fewer than two sessions in the fit stay out of the inner CV but are in the final fit.
Classes missing from the fit get a very negative logit.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

log = logging.getLogger(__name__)
MISSING_LOGIT = -50.0


def nll(logits: np.ndarray, y: np.ndarray, T: float = 1.0) -> float:
    lp = log_softmax(logits / T, axis=1)
    return float(-lp[np.arange(len(y)), y].mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return 1.0
    r = minimize_scalar(
        lambda lt: nll(logits, y, float(np.exp(lt))), bounds=(np.log(0.05), np.log(20.0)), method="bounded"
    )
    return float(np.exp(r.x))


def _model(C: float, max_iter: int, class_weight: Optional[str], seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cw = None if class_weight in (None, "none") else class_weight
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=C, max_iter=max_iter, class_weight=cw, random_state=seed)
    )


def _fit(m, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence and missing classes: expected on small folds
        return m.fit(X, y)


def logits_of(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    raw = model.decision_function(X)
    classes = model[-1].classes_
    if raw.ndim == 1:
        raw = np.stack([-raw, raw], 1) / 2
    out = np.full((len(X), n_classes), MISSING_LOGIT)
    out[:, classes] = raw
    return out


@dataclass
class Head:
    model: object
    C: float
    T: float
    n_classes: int

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(logits_of(self.model, X, self.n_classes) / self.T, axis=1)


def fit_head(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_classes: int, hcfg, seed: int) -> Head:
    from sklearn.model_selection import StratifiedGroupKFold

    Cs = list(hcfg.Cs)
    groups = np.asarray(groups).astype(str)
    groups_per_class = np.array([len(np.unique(groups[y == c])) for c in range(n_classes)])
    in_cv = groups_per_class[y] >= 2
    usable = groups_per_class[groups_per_class >= 2]
    n_splits = min(int(hcfg.inner_cv), int(usable.min())) if len(usable) else 1
    best_C, T = Cs[0], 1.0
    if n_splits >= 2:
        Xc, yc, gc = X[in_cv], y[in_cv], groups[in_cv]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cv = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(Xc, yc, gc))
        best, best_oof = None, None
        for C in Cs:
            oof = np.zeros((len(yc), n_classes))
            for tr, te in cv:
                m = _fit(_model(C, hcfg.max_iter, hcfg.class_weight, seed), Xc[tr], yc[tr])
                oof[te] = logits_of(m, Xc[te], n_classes)
            score = nll(oof, yc)
            if best is None or score < best:
                best, best_C, best_oof = score, C, oof
        T = fit_temperature(best_oof, yc)
    else:
        log.warning("testa: CV interna saltata (nessuna classe con 2 sessioni): C=%g, T=1", best_C)
    model = _fit(_model(best_C, hcfg.max_iter, hcfg.class_weight, seed), X, y)
    return Head(model, float(best_C), float(T), n_classes)
