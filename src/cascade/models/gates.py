"""Features and scorers of the gate v2. A higher score means CamFormer is more likely to be wrong.

Features (only from CamFormer and the trajectory):
- ``A``: logit of p_max (clipped), top-1 − top-2 margin, normalised entropy;
- ``B``: reliability of the predicted class (share of CamFormer's correct predictions for that class, smoothed as
  (k + α)/(n + 2α)) and a flag "top-1 and top-2 are both procedural". The reliability of a new row is estimated
  on all the fit rows. For a fit row it is computed OUT OF FOLD (N38): we split the fit rows into K folds by
  session (stratified by predicted class), and each fold gets the reliability computed on the others. So the
  outcome of a row weighs about 1/(takes in the fold) on its own value, as in the gate v1, which left the fold
  out. The version used in the run of 18/11 (job 772601) left out only the row's own session. Inside the training
  set, its value, (correct − correct in the session)/(predicted − predicted in the session), was a decreasing
  function of the session's outcomes, and with one-take sessions (the median) it is exactly leave-one-out. Its
  correlation with the error within a class was 0.6–0.95 on the real data, and LightGBM learned a reversed rule;
- ``C``: camera motion (``motion.MOTION_FEATURES``), imputed with the median of the fit rows.

Scorers:
- ``thr_maxprob``: 1 − p_max, no fit (the reference);
- ``class_rule``: 1 if the predicted class is procedural, 0 otherwise (a per-class rule used as a gate);
- ``logreg_<groups>``: imputation, standardisation and logistic regression WITHOUT class weights (so it gives an
  error probability), C chosen by CV grouped by session (log-loss);
- ``lgbm_<groups>``: LightGBM with small trees and early stopping on a holdout split by session.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np

from src.cascade.data.folds import session_folds
from src.cascade.data.motion import MOTION_FEATURES

log = logging.getLogger(__name__)
EPS = 1e-6
A_FEATURES = ["logit_p_max", "margin", "entropy_norm"]
B_FEATURES = ["reliability", "top2_procedural"]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def group_a(P: np.ndarray) -> np.ndarray:
    p = np.clip(P, 1e-12, 1.0)
    s = np.sort(p, axis=1)
    ent = -(p * np.log(p)).sum(1) / np.log(p.shape[1])
    return np.stack([logit(s[:, -1]), s[:, -1] - s[:, -2], ent], axis=1)


class FeatureBuilder:
    """Build the features. ``fit`` stores the reliability counts of the fit rows and, for the fit rows
    themselves, the out-of-fold reliability (``train_reliability``)."""

    def __init__(self, groups: List[str], procedural: np.ndarray, alpha: float = 1.0, n_folds: int = 5) -> None:
        bad = set(groups) - {"A", "B", "C"}
        if bad:
            raise ValueError(f"gruppi sconosciuti: {sorted(bad)}")
        self.groups, self.proc, self.alpha = list(groups), np.asarray(procedural, bool), float(alpha)
        self.n_folds = int(n_folds)
        self.n_pred = self.n_corr = None
        self.train_reliability: Optional[np.ndarray] = None

    def _rate(self, n: np.ndarray, k: np.ndarray) -> np.ndarray:
        return (k + self.alpha) / (n + 2 * self.alpha)

    def fit(self, P: np.ndarray, y: np.ndarray, sessions: np.ndarray, seed: int = 0) -> FeatureBuilder:
        C = P.shape[1]
        pred, corr = P.argmax(1), (P.argmax(1) == y).astype(float)
        self.n_pred = np.bincount(pred, minlength=C).astype(float)
        self.n_corr = np.bincount(pred, weights=corr, minlength=C)
        self.train_reliability = None
        if "B" in self.groups:
            s = np.asarray(sessions).astype(str)
            k = min(self.n_folds, len(np.unique(s)))
            rel = np.full(len(P), self._rate(np.zeros(1), np.zeros(1))[0])  # a single session: only the prior
            if k >= 2:
                fold = session_folds(pred, s, k, seed)  # stratified by PREDICTED class: no labels used
                for j in range(k):
                    m = fold == j
                    n_out = self.n_pred - np.bincount(pred[m], minlength=C)
                    k_out = self.n_corr - np.bincount(pred[m], weights=corr[m], minlength=C)
                    rel[m] = self._rate(n_out, k_out)[pred[m]]
            else:
                log.warning("affidabilità fuori fold: una sola sessione nel fit, uso il solo prior")
            self.train_reliability = rel
        return self

    def reliability(self, P: np.ndarray) -> np.ndarray:
        """Reliability for NEW rows: counts from all the fit rows."""
        pred = P.argmax(1)
        return self._rate(self.n_pred[pred], self.n_corr[pred])

    def transform(self, P: np.ndarray, M: np.ndarray, training: bool = False) -> np.ndarray:
        """``training=True`` only for the same rows passed to ``fit``: out-of-fold reliability."""
        mats = []
        for g in self.groups:
            if g == "A":
                mats.append(group_a(P))
            elif g == "B":
                if self.n_pred is None:
                    raise RuntimeError("FeatureBuilder.fit prima di transform con il gruppo B")
                top2 = np.argsort(-P, axis=1, kind="stable")[:, :2]
                flag = (self.proc[top2[:, 0]] & self.proc[top2[:, 1]]).astype(float)
                if training:
                    if self.train_reliability is None or len(self.train_reliability) != len(P):
                        raise RuntimeError("transform(training=True) vuole le stesse righe di fit")
                    rel = self.train_reliability
                else:
                    rel = self.reliability(P)
                mats.append(np.stack([rel, flag], 1))
            elif g == "C":
                mats.append(np.asarray(M, float))
        return np.concatenate(mats, 1) if mats else np.zeros((len(P), 0))

    def names(self) -> List[str]:
        out: List[str] = []
        for g in self.groups:
            out += {"A": A_FEATURES, "B": B_FEATURES, "C": list(MOTION_FEATURES)}[g]
        return out


def session_cv(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    """Splits stratified by ``y`` and grouped by session; fewer folds if few groups have positives."""
    from sklearn.model_selection import StratifiedGroupKFold

    g = np.asarray(groups).astype(str)
    n_pos_groups = len(np.unique(g[y == 1]))
    n_neg_groups = len(np.unique(g[y == 0]))
    k = min(int(n_splits), n_pos_groups, n_neg_groups)
    if k < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return list(
            StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=int(seed)).split(np.zeros(len(y)), y, g)
        )


class Scorer:
    name = "scorer"
    groups: List[str] = []
    needs_fit = False

    def __init__(self, procedural: np.ndarray, alpha: float) -> None:
        self.fb = FeatureBuilder(self.groups, procedural, alpha)

    def fit(self, P, M, y, e, sessions, seed: int = 0) -> Scorer:
        return self

    def score(self, P, M) -> np.ndarray:
        raise NotImplementedError

    def importance(self) -> Dict[str, float]:
        return {}


class MaxProbScorer(Scorer):
    name = "thr_maxprob"

    def score(self, P, M) -> np.ndarray:
        return 1.0 - P.max(1)


class ClassRuleScorer(Scorer):
    name = "class_rule"

    def __init__(self, procedural: np.ndarray, alpha: float) -> None:
        super().__init__(procedural, alpha)
        self.proc = np.asarray(procedural, bool)

    def score(self, P, M) -> np.ndarray:
        return self.proc[P.argmax(1)].astype(float)


class LogRegScorer(Scorer):
    needs_fit = True

    def __init__(self, name: str, groups: List[str], procedural, alpha, Cs, inner_cv) -> None:
        self.name, self.groups = name, groups
        super().__init__(procedural, alpha)
        self.Cs, self.inner_cv = list(Cs), int(inner_cv)
        self.model, self.constant = None, None

    def fit(self, P, M, y, e, sessions, seed: int = 0) -> LogRegScorer:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.fb.fit(P, y, sessions, seed=seed)
        X = self.fb.transform(P, M, training=True)
        cv = session_cv(e, sessions, self.inner_cv, seed)
        if cv is None:
            self.constant = float(np.mean(e))
            log.warning("%s: errori in meno di 2 sessioni, score costante %.4f", self.name, self.constant)
            return self
        self.model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegressionCV(Cs=self.Cs, cv=cv, scoring="neg_log_loss", max_iter=5000),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, e)
        return self

    def score(self, P, M) -> np.ndarray:
        if self.model is None:
            return np.full(len(P), self.constant if self.constant is not None else 0.0)
        return self.model.predict_proba(self.fb.transform(P, M))[:, 1]

    def importance(self) -> Dict[str, float]:
        if self.model is None:
            return {}
        return {n: float(c) for n, c in zip(self.fb.names(), self.model[-1].coef_.ravel())}


class LGBMScorer(Scorer):
    needs_fit = True

    def __init__(self, name: str, groups: List[str], procedural, alpha, params: Dict) -> None:
        self.name, self.groups = name, groups
        super().__init__(procedural, alpha)
        p = dict(params)
        self.es_rounds = int(p.pop("early_stopping_rounds", 30))
        self.valid_frac = float(p.pop("valid_frac", 0.2))
        self.params = p
        self.model, self.constant = None, None

    def fit(self, P, M, y, e, sessions, seed: int = 0) -> LGBMScorer:
        import lightgbm as lgb

        self.fb.fit(P, y, sessions, seed=seed)
        X = self.fb.transform(P, M, training=True)
        cv = session_cv(e, sessions, max(2, int(round(1.0 / self.valid_frac))), seed)
        if cv is None or e.sum() < 4:
            self.constant = float(np.mean(e))
            log.warning("%s: troppo pochi errori, score costante %.4f", self.name, self.constant)
            return self
        tr, va = cv[0]
        self.model = lgb.LGBMClassifier(random_state=seed, verbose=-1, **self.params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(
                X[tr],
                e[tr],
                eval_set=[(X[va], e[va])],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(self.es_rounds, verbose=False)],
            )
        return self

    def score(self, P, M) -> np.ndarray:
        if self.model is None:
            return np.full(len(P), self.constant if self.constant is not None else 0.0)
        return self.model.predict_proba(self.fb.transform(P, M))[:, 1]

    def importance(self) -> Dict[str, float]:
        if self.model is None:
            return {}
        gain = self.model.booster_.feature_importance(importance_type="gain")
        return {n: float(g) for n, g in zip(self.fb.names(), gain)}


def _groups(suffix: str) -> List[str]:
    groups = list(suffix)
    if not groups or set(groups) - set("ABC") or groups[0] != "A":
        raise ValueError(f"gruppi non validi {suffix!r}: il primo deve essere A, poi B e/o C")
    return groups


def make_scorer(name: str, gcfg, procedural: np.ndarray) -> Scorer:
    sc = _make_scorer(name, gcfg, procedural)
    sc.fb.n_folds = int(getattr(gcfg, "reliability_folds", 5))  # folds of the out-of-fold reliability (N38)
    return sc


def _make_scorer(name: str, gcfg, procedural: np.ndarray) -> Scorer:
    a = gcfg.reliability_alpha
    if name == "thr_maxprob":
        return MaxProbScorer(procedural, a)
    if name == "class_rule":
        return ClassRuleScorer(procedural, a)
    if name.startswith("logreg_"):
        return LogRegScorer(name, _groups(name[7:]), procedural, a, gcfg.logreg_Cs, gcfg.logreg_inner_cv)
    if name.startswith("lgbm_"):
        params = dict(gcfg.lgbm) or {
            "num_leaves": 15,
            "min_child_samples": 20,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "deterministic": True,
            "force_row_wise": True,
        }
        return LGBMScorer(name, _groups(name[5:]), procedural, a, params)
    raise ValueError(f"scorer sconosciuto: {name!r}")


def check_scorers(names: List[str], gcfg, procedural: np.ndarray) -> None:
    """Fail at once (before the heads) if a scorer does not exist or LightGBM is missing."""
    for n in names:
        make_scorer(n, gcfg, procedural)
        if n.startswith("lgbm_"):
            import lightgbm  # noqa: F401
