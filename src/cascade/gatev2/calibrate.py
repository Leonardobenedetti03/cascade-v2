"""Calibratore del gate v2: dallo score al rischio che CamFormer sbagli, r̂(z) = P(c0 = 0 | z).

- ``platt``: r̂ = σ(a + b · logit(s)), due parametri (default: stabile con poche decine di errori);
- ``isotonic``: regressione isotona su s (variante, a gradini).

Il limite prudente ``r_up`` è il quantile 1 − δ di r̂ su repliche bootstrap del fit che ricampionano le
SESSIONI: dove gli errori sono pochi il rischio è incerto e ``r_up`` sale. Il calibratore si stima sempre su
score fuori campione (cross-fitting del livello 2), mai sulle take che valuta.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
from scipy.special import expit

from src.cascade.gatev2.scorers import logit

log = logging.getLogger(__name__)


def _platt_fit(z: np.ndarray, e: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """(a, b) di σ(a + b z) con una logistica appena regolarizzata (C = 100: evita coefficienti infiniti se i
    dati sono separabili). Con una sola classe: b = 0 e a = logit della frequenza con smoothing."""
    from sklearn.linear_model import LogisticRegression

    if w is None:
        w = np.ones(len(e))
    pos = float((w * e).sum())
    if pos == 0 or pos == float(w.sum()):
        p = (pos + 0.5) / (float(w.sum()) + 1.0)
        return float(np.log(p / (1 - p))), 0.0
    m = LogisticRegression(C=100.0, max_iter=1000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(z[:, None], e, sample_weight=w)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


class Calibrator:
    def __init__(self, method: str = "platt", n_boot: int = 200, delta: float = 0.1, seed: int = 0) -> None:
        if method not in ("platt", "isotonic"):
            raise ValueError(f"calibrazione sconosciuta: {method!r} (platt | isotonic)")
        self.method, self.n_boot, self.delta, self.seed = method, int(n_boot), float(delta), int(seed)
        self.params: Tuple[float, float] = (0.0, 0.0)
        self.boot_params: Optional[np.ndarray] = None  # [B, 2] (platt)
        self.iso = None
        self.boot_iso: list = []

    def _fit_one(self, s, e, w=None):
        if self.method == "platt":
            return _platt_fit(logit(s), e, w)
        from sklearn.isotonic import IsotonicRegression

        return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s, e, sample_weight=w)

    def fit(self, s: np.ndarray, e: np.ndarray, sessions: np.ndarray) -> Calibrator:
        s, e = np.asarray(s, float), np.asarray(e, float)
        fitted = self._fit_one(s, e)
        if self.method == "platt":
            self.params = fitted
            if fitted[1] < 0:
                log.warning("calibratore: pendenza negativa (b = %.3f): lo score non ordina gli errori", fitted[1])
        else:
            self.iso = fitted
        if self.n_boot > 0:
            _, inv = np.unique(np.asarray(sessions).astype(str), return_inverse=True)
            G = int(inv.max()) + 1
            rng = np.random.default_rng(self.seed)
            reps = []
            for _ in range(self.n_boot):
                w = np.bincount(rng.integers(0, G, G), minlength=G)[inv].astype(float)
                reps.append(self._fit_one(s, e, w))
            if self.method == "platt":
                self.boot_params = np.array(reps)
            else:
                self.boot_iso = reps
        return self

    def predict(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(r_hat, r_up). Senza bootstrap r_up = r_hat."""
        s = np.asarray(s, float)
        if self.method == "platt":
            a, b = self.params
            z = logit(s)
            r = expit(a + b * z)
            if self.boot_params is None or not len(self.boot_params):
                return r, r.copy()
            R = expit(self.boot_params[:, :1] + self.boot_params[:, 1:] * z[None])
        else:
            r = self.iso.predict(s)
            if not self.boot_iso:
                return r, r.copy()
            R = np.stack([m.predict(s) for m in self.boot_iso])
        up = np.quantile(R, 1.0 - self.delta, axis=0)
        return r, np.maximum(up, r)


def ece(r: np.ndarray, e: np.ndarray, n_bins: int = 10) -> float:
    """Errore di calibrazione atteso su fasce di quantile di r."""
    r, e = np.asarray(r, float), np.asarray(e, float)
    if len(r) == 0:
        return float("nan")
    order = np.argsort(r, kind="stable")
    bins = np.array_split(order, min(n_bins, len(r)))
    return float(sum(len(b) * abs(r[b].mean() - e[b].mean()) for b in bins if len(b)) / len(r))
