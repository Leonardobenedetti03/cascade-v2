"""Calibrator of the gate v2: from the score to the risk that CamFormer is wrong, r̂(z) = P(c0 = 0 | z).

- ``platt``: r̂ = σ(a + b · logit(s)), two parameters (the default: stable with a few dozen errors);
- ``isotonic``: isotonic regression on s (a step-wise variant).

The conservative bound ``r_up`` is the 1 − δ quantile of r̂ over bootstrap refits that resample SESSIONS.
Where errors are few the risk is uncertain, and ``r_up`` goes up. The calibrator is always fitted on
out-of-sample scores (the level-2 cross-fitting), never on the takes it evaluates.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
from scipy.special import expit

from src.cascade.models.gates import logit

log = logging.getLogger(__name__)


def _platt_fit(z: np.ndarray, e: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """(a, b) of σ(a + b z), from a lightly regularised logistic regression (C = 100 avoids infinite coefficients
    when the data are separable). With a single class: b = 0 and a = logit of the smoothed frequency."""
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
        """(r_hat, r_up). Without bootstrap, r_up = r_hat."""
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
    """Expected calibration error over quantile bins of r."""
    r, e = np.asarray(r, float), np.asarray(e, float)
    if len(r) == 0:
        return float("nan")
    order = np.argsort(r, kind="stable")
    bins = np.array_split(order, min(n_bins, len(r)))
    return float(sum(len(b) * abs(r[b].mean() - e[b].mean()) for b in bins if len(b)) / len(r))
