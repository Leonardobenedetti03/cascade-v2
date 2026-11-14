"""Modelli di gate, tutti con interfaccia ``fit(X, y)`` / ``score(X)``.

Score più alto = più conveniente scalare la clip a L1. Famiglie (blueprint, sezione 5.2):

- riferimenti: ``always_L0`` / ``always_L1`` (punti fissi r = 0 e r = 100, ``fixed_r``),
  ``random`` (score casuale, curva mediata su più seed), ``oracle`` (score = c1 − c0, cioè
  prima le clip con y = 1, poi le neutre, per ultime quelle in cui L1 peggiora: ottimo per
  ogni r e ≥ di ogni gate);
- selettivi: ``thr_maxprob`` / ``thr_entropy`` / ``thr_margin``, soglia su una feature di A;
- appresi su y = (1 − c0)·c1: ``logreg_<gruppi>`` (standardizzazione + regressione logistica,
  ``class_weight="balanced"``, C scelto in CV interna, es. ``logreg_ABC``) e
  ``lgbm_<gruppi>`` (LightGBM con alberi piccoli ed early stopping su un holdout interno);
- compare: ``compare_<gruppi>`` (M06), CompareConfidence di Raghu et al. (2019): lo score è
  ``η̂₁(z) − max_k p0_k(z)``, con η̂₁ regressione logistica di c1 senza pesi di classe.

Non si portano ``ridge_*``, ``lgbmreg_*`` e ``bayes_logreg_*`` (M06, nota N16).
"""

from __future__ import annotations

import abc
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.cascade.gate.features import A_FEATURES

log = logging.getLogger(__name__)


class Gate(abc.ABC):
    name = "gate"
    groups: List[str] = []
    stochastic = False
    fixed_r: Optional[int] = None
    needs_truth = False
    target = "hard"  # hard: y = (1 − c0)·c1 | c1: L1 ha ragione (colonna scelta in gate/protocol.py)

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 0, groups: Optional[np.ndarray] = None) -> Gate:
        """``groups`` = take delle righe: le CV interne non separano finestre della stessa take."""
        return self

    @abc.abstractmethod
    def score(self, X: np.ndarray, c0: Optional[np.ndarray] = None, c1: Optional[np.ndarray] = None) -> np.ndarray:
        """Score di escalation per riga (più alto = scala)."""

    def importance(self, names: List[str]) -> Dict[str, float]:
        return {}


class FixedGate(Gate):
    def __init__(self, name: str, r: int) -> None:
        self.name, self.fixed_r = name, r

    def score(self, X, c0=None, c1=None):
        return np.zeros(len(X)) if self.fixed_r == 0 else np.ones(len(X))


class RandomGate(Gate):
    name = "random"
    stochastic = True

    def __init__(self) -> None:
        self.seed = 0

    def fit(self, X, y, seed=0, groups=None):
        self.seed = seed
        return self

    def score(self, X, c0=None, c1=None):
        return np.random.default_rng([self.seed, len(X)]).random(len(X))


class ThresholdGate(Gate):
    groups = ["A"]

    def __init__(self, name: str, feature: str, sign: float) -> None:
        self.name, self.col, self.sign = name, A_FEATURES.index(feature), sign

    def score(self, X, c0=None, c1=None):
        return self.sign * X[:, self.col]


def log_loss_01(y_true, y_prob) -> float:
    """``log_loss`` con le classi {0, 1} esplicite: una fold di validazione con una sola classe
    non ferma il fit. Senza parametro ``labels`` nella firma, perché sklearn >= 1.8 lo aggiunge
    da sé agli scorer che lo accettano (e due ``labels`` danno errore); sklearn 1.6 no."""
    from sklearn.metrics import log_loss

    return float(log_loss(y_true, y_prob, labels=[0, 1]))


class LogRegGate(Gate):
    def __init__(
        self, name: str, groups: List[str], Cs: List[float], inner_cv: int, class_weight: Optional[str] = "balanced"
    ) -> None:
        self.name, self.groups, self.Cs, self.inner_cv = name, groups, Cs, inner_cv
        self.class_weight = class_weight
        self.model = None
        self.constant: Optional[float] = None

    def fit(self, X, y, seed=0, groups=None):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.metrics import make_scorer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if len(np.unique(y)) < 2 or min(class_groups(y, groups)) < 2:
            self.constant = float(np.mean(y))
            log.warning("%s: classi insufficienti per il fit, score costante", self.name)
            return self
        cv = grouped_splits(y, groups, self.inner_cv, seed)
        scorer = make_scorer(log_loss_01, greater_is_better=False, response_method="predict_proba")
        self.model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegressionCV(Cs=self.Cs, cv=cv, class_weight=self.class_weight, scoring=scorer, max_iter=5000),
        )
        with warnings.catch_warnings():  # FutureWarning di sklearn >= 1.8, assenti su 1.6
            warnings.simplefilter("ignore", FutureWarning)
            self.model.fit(X, y)
        return self

    def score(self, X, c0=None, c1=None):
        if self.model is None:
            return np.full(len(X), self.constant if self.constant is not None else 0.0)
        return self.model.predict_proba(X)[:, 1]

    def importance(self, names):
        if self.model is None:
            return {}
        coef = self.model[-1].coef_.ravel()
        return {n: float(c) for n, c in zip(names, coef)}


class CompareGate(LogRegGate):
    """CompareConfidence (Raghu et al., 2019; baseline di Mozannar, Lang et al., 2023),
    reimplementata sul protocollo fuori campione del repo (M06).

    ``score(z) = η̂₁(z) − max_k p0_k(z)``: un modello stima la probabilità che L1 abbia ragione
    (etichetta c1) e si scala dove supera la confidenza calibrata di L0. È lo stimatore plug-in
    del beneficio atteso Δ(z). Senza pesi di classe, perché servono probabilità e non un
    ranking bilanciato. Se c1 ha una sola classe nei dati di fit, η̂₁ è costante e lo score
    ordina le clip come ``thr_maxprob``.
    """

    target = "c1"

    def __init__(self, name: str, groups: List[str], Cs: List[float], inner_cv: int) -> None:
        if not groups or groups[0] != "A":
            raise ValueError("compare_*: il primo gruppo deve essere A (serve p_max)")
        super().__init__(name, groups, Cs, inner_cv, class_weight=None)
        self.p_max_col = A_FEATURES.index("p_max")

    def score(self, X, c0=None, c1=None):
        eta1 = super().score(X)  # costante se c1 ha una sola classe nel fit
        return eta1 - X[:, self.p_max_col]


class LGBMGate(Gate):
    stochastic = True

    def __init__(self, name: str, groups: List[str], params: Dict) -> None:
        self.name, self.groups = name, groups
        p = dict(params)
        self.es_rounds = int(p.pop("early_stopping_rounds", 30))
        self.valid_frac = float(p.pop("valid_frac", 0.2))
        if p.get("num_leaves", 15) > 15 or p.get("min_child_samples", 20) < 20:
            raise ValueError("lgbm: num_leaves <= 15 e min_child_samples >= 20 (alberi piccoli)")
        self.params = p
        self.model = None
        self.constant: Optional[float] = None

    def fit(self, X, y, seed=0, groups=None):
        import lightgbm as lgb

        if len(np.unique(y)) < 2 or np.bincount(y.astype(int)).min() < 4 or min(class_groups(y, groups)) < 2:
            self.constant = float(np.mean(y))
            log.warning("%s: classi insufficienti per il fit, score costante", self.name)
            return self
        n_splits = max(2, int(round(1.0 / self.valid_frac)))
        tr_idx, va_idx = grouped_splits(y, groups, n_splits, seed)[0]
        Xtr, Xva, ytr, yva = X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]
        self.model = lgb.LGBMClassifier(random_state=seed, verbose=-1, **self.params)
        with warnings.catch_warnings():  # eval_set è deprecato solo in lightgbm >= 4.7
            warnings.simplefilter("ignore")
            self.model.fit(
                Xtr,
                ytr,
                eval_set=[(Xva, yva)],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(self.es_rounds, verbose=False)],
            )
        return self

    def score(self, X, c0=None, c1=None):
        if self.model is None:
            return np.full(len(X), self.constant if self.constant is not None else 0.0)
        return self.model.predict_proba(X)[:, 1]

    def importance(self, names):
        if self.model is None:
            return {}
        gain = self.model.booster_.feature_importance(importance_type="gain")
        return {n: float(g) for n, g in zip(names, gain)}


class OracleGate(Gate):
    name = "oracle"
    needs_truth = True

    def score(self, X, c0=None, c1=None):
        if c0 is None or c1 is None:
            raise ValueError("l'oracolo richiede c0 e c1")
        return np.asarray(c1, float) - np.asarray(c0, float)


def class_groups(y: np.ndarray, groups: Optional[np.ndarray]) -> Tuple[int, int]:
    """(gruppi con almeno un positivo, gruppi con almeno un negativo); senza gruppi, le righe."""
    y = np.asarray(y).astype(int)
    if groups is None:
        return int((y == 1).sum()), int((y == 0).sum())
    g = np.asarray(groups).astype(str)
    return len(set(g[y == 1])), len(set(g[y == 0]))


def grouped_splits(y: np.ndarray, groups: Optional[np.ndarray], n_splits: int, seed: int):
    """Split stratificati per y e raggruppati per take, deterministici tra versioni di sklearn.

    Ogni take ha l'etichetta "contiene almeno un positivo"; le take si distribuiscono a giro
    sulle fold, classe per classe, come ``folds.assign_folds`` (RandomState). Le fold sono al
    più quanti i gruppi della classe più rara, così ogni fold di validazione ha entrambe le
    classi (regressione: con ``StratifiedGroupKFold`` e pochi positivi una fold interna poteva
    averne una sola, e ``log_loss`` del gate logistico si fermava).
    """
    from src.cascade.folds import assign_folds

    y = np.asarray(y).astype(int)
    g = np.arange(len(y)).astype(str) if groups is None else np.asarray(groups).astype(str)
    uniq, inv = np.unique(g, return_inverse=True)
    g_pos = np.zeros(len(uniq), dtype=int)
    np.maximum.at(g_pos, inv, y)
    k = min(n_splits, *class_groups(y, g))
    if k < 2:
        raise ValueError(
            f"servono almeno 2 gruppi per classe per la CV interna (gruppi per classe: {class_groups(y, g)})"
        )
    fold = assign_folds(g_pos, k, seed)[inv]
    return [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in range(k)]


def _groups_from_suffix(suffix: str) -> List[str]:
    main, *extra = suffix.split("+")
    groups = list(main)
    for g in groups:
        if g not in "ABC":
            raise ValueError(f"gruppo {g!r} non valido in {suffix!r}")
    return groups + extra


def make_gate(name: str, gcfg) -> Gate:
    if name == "always_L0":
        return FixedGate(name, 0)
    if name == "always_L1":
        return FixedGate(name, 100)
    if name == "random":
        return RandomGate()
    if name == "oracle":
        return OracleGate()
    if name == "thr_maxprob":
        return ThresholdGate(name, "p_max", -1.0)
    if name == "thr_entropy":
        return ThresholdGate(name, "entropy_norm", 1.0)
    if name == "thr_margin":
        return ThresholdGate(name, "margin", -1.0)
    if name.startswith("logreg_"):
        cw = None if gcfg.logreg_class_weight == "none" else gcfg.logreg_class_weight
        return LogRegGate(name, _groups_from_suffix(name[len("logreg_") :]), gcfg.logreg_Cs, gcfg.logreg_inner_cv, cw)
    if name.startswith("compare_"):
        return CompareGate(name, _groups_from_suffix(name[len("compare_") :]), gcfg.compare.Cs, gcfg.compare.inner_cv)
    if name.startswith("lgbm_"):
        return LGBMGate(name, _groups_from_suffix(name[len("lgbm_") :]), gcfg.lgbm)
    raise ValueError(f"gate sconosciuto: {name!r}")
