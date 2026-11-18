"""Feature e scorer del gate v2. Score più alto = più probabile che CamFormer sbagli.

Feature (solo da CamFormer e dalla traiettoria):
- ``A``: logit di p_max (troncato), margine top-1 − top-2, entropia normalizzata;
- ``B``: affidabilità della classe predetta (quota di predizioni corrette di CamFormer per quella classe, con
  smoothing (k + α)/(n + 2α)) e flag "top-1 e top-2 entrambe procedurali". L'affidabilità si stima sulle righe
  di fit; per una riga di fit esclude la sua sessione (nessuna riga vede la propria etichetta);
- ``C``: moto della camera (``motion.MOTION_FEATURES``), imputato con la mediana del fit.

Scorer:
- ``thr_maxprob``: 1 − p_max, nessun fit (riferimento);
- ``class_rule``: 1 se la classe predetta è procedurale, 0 altrimenti (regola per classe come gate);
- ``logreg_<gruppi>``: imputazione, standardizzazione e logistica SENZA pesi di classe (probabilità di errore),
  C scelto in CV raggruppata per sessione (log-loss);
- ``lgbm_<gruppi>``: LightGBM con alberi piccoli, arresto anticipato su un holdout per sessione.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np

from src.cascade.motion import MOTION_FEATURES

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
    """Costruisce le feature; ``fit`` memorizza i conteggi di affidabilità delle righe di fit."""

    def __init__(self, groups: List[str], procedural: np.ndarray, alpha: float = 1.0) -> None:
        bad = set(groups) - {"A", "B", "C"}
        if bad:
            raise ValueError(f"gruppi sconosciuti: {sorted(bad)}")
        self.groups, self.proc, self.alpha = list(groups), np.asarray(procedural, bool), float(alpha)
        self.n_pred = self.n_corr = None
        self.by_session: Dict[str, tuple] = {}

    def fit(self, P: np.ndarray, y: np.ndarray, sessions: np.ndarray) -> FeatureBuilder:
        C = P.shape[1]
        pred, corr = P.argmax(1), (P.argmax(1) == y).astype(float)
        self.n_pred = np.bincount(pred, minlength=C).astype(float)
        self.n_corr = np.bincount(pred, weights=corr, minlength=C)
        self.by_session = {}
        for s in np.unique(np.asarray(sessions).astype(str)):
            m = np.asarray(sessions).astype(str) == s
            self.by_session[s] = (
                np.bincount(pred[m], minlength=C).astype(float),
                np.bincount(pred[m], weights=corr[m], minlength=C),
            )
        return self

    def reliability(self, P: np.ndarray, sessions: Optional[np.ndarray] = None) -> np.ndarray:
        """Con ``sessions`` (righe di fit) si toglie la propria sessione dai conteggi."""
        pred = P.argmax(1)
        n, k = self.n_pred[pred].copy(), self.n_corr[pred].copy()
        if sessions is not None:
            for i, s in enumerate(np.asarray(sessions).astype(str)):
                if s in self.by_session:
                    sn, sk = self.by_session[s]
                    n[i] -= sn[pred[i]]
                    k[i] -= sk[pred[i]]
        return (k + self.alpha) / (n + 2 * self.alpha)

    def transform(self, P: np.ndarray, M: np.ndarray, own_sessions: Optional[np.ndarray] = None) -> np.ndarray:
        mats = []
        for g in self.groups:
            if g == "A":
                mats.append(group_a(P))
            elif g == "B":
                if self.n_pred is None:
                    raise RuntimeError("FeatureBuilder.fit prima di transform con il gruppo B")
                top2 = np.argsort(-P, axis=1, kind="stable")[:, :2]
                flag = (self.proc[top2[:, 0]] & self.proc[top2[:, 1]]).astype(float)
                mats.append(np.stack([self.reliability(P, own_sessions), flag], 1))
            elif g == "C":
                mats.append(np.asarray(M, float))
        return np.concatenate(mats, 1) if mats else np.zeros((len(P), 0))

    def names(self) -> List[str]:
        out: List[str] = []
        for g in self.groups:
            out += {"A": A_FEATURES, "B": B_FEATURES, "C": list(MOTION_FEATURES)}[g]
        return out


def session_cv(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    """Split stratificati per ``y`` e raggruppati per sessione; meno fold se i gruppi con positivi sono pochi."""
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

        self.fb.fit(P, y, sessions)
        X = self.fb.transform(P, M, own_sessions=sessions)
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

        self.fb.fit(P, y, sessions)
        X = self.fb.transform(P, M, own_sessions=sessions)
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
    """Errore subito (prima delle teste) se uno scorer non esiste o manca LightGBM."""
    for n in names:
        make_scorer(n, gcfg, procedural)
        if n.startswith("lgbm_"):
            import lightgbm  # noqa: F401
