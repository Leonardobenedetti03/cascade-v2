"""Regole di decisione del gate v2. Tutte si fissano su dati di calibrazione (score fuori campione del train
esterno) e si applicano congelate alla fold esterna.

- ``cost`` (β): scala se il rischio (``r_up``, o ``r_hat`` nella variante) è ≥ β. β è il prezzo di una chiamata a
  EgoVLP misurato in errori: 1/β chiamate per ogni errore evitato. La soglia è β stesso, non si sceglie sui dati.
- ``risk`` (α, δ): CamFormer tiene la copertura più grande per cui il limite superiore di Clopper-Pearson
  dell'errore tra le take tenute è ≤ α. Le coperture candidate sono una griglia fissa di G valori
  (1/G, 2/G, ..., 1) e ognuna si verifica al livello 1 − δ/G (Bonferroni): la garanzia vale qualunque
  copertura venga scelta (selective classification con garanzia, Geifman ed El-Yaniv 2017, con la correzione
  per la scelta sulla griglia). Una sequenza fissa dalla copertura più piccola non funziona qui: con poche take
  tenute il limite è largo e la sequenza si fermerebbe subito.
- ``acc_target`` (e_max): la quota minima di escalation con errori lasciati passare ≤ e_max · n (come
  ``within_pts`` della v1, assumendo EgoVLP corretto sulle take scalate). Riferimento, nessuna garanzia.
- ``budget`` (ρ): scala la quota ρ di take con lo score più alto.

Le soglie su s valgono "scala se s > λ"; ``-inf`` = tutto a EgoVLP, ``inf`` = niente.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import beta as beta_dist


def clopper_pearson_upper(k: int, n: int, conf: float) -> float:
    """Limite superiore unilaterale al livello ``conf`` di una proporzione binomiale (k errori su n)."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(conf, k + 1, n - k))


def risk_threshold(s: np.ndarray, e: np.ndarray, alpha: float, delta: float, grid: int) -> Dict[str, float]:
    s, e = np.asarray(s, float), np.asarray(e, int)
    n = len(s)
    out = {"threshold": -np.inf, "coverage_cal": 0.0, "ucb": float("nan"), "errors_kept_cal": 0, "n_kept_cal": 0}
    if n == 0:
        return out
    order = np.sort(s)
    conf = 1.0 - delta / grid
    for g in range(1, grid + 1):
        n_keep = int(np.floor(g / grid * n))
        if n_keep < 1:
            continue
        lam = order[n_keep - 1]
        kept = s <= lam
        k, m = int(e[kept].sum()), int(kept.sum())
        ucb = clopper_pearson_upper(k, m, conf)
        if ucb <= alpha:
            out = {"threshold": float(lam), "coverage_cal": m / n, "ucb": ucb, "errors_kept_cal": k, "n_kept_cal": m}
    if out["n_kept_cal"] == n:
        out["threshold"] = np.inf
    return out


def acc_target_threshold(s: np.ndarray, e: np.ndarray, max_error_rate: float) -> float:
    """Soglia più alta (meno escalation) con errori lasciati passare ≤ max_error_rate · n."""
    s, e = np.asarray(s, float), np.asarray(e, int)
    n = len(s)
    if n == 0:
        return -np.inf
    order = np.argsort(s, kind="stable")
    ss, ee = s[order], e[order]
    cum = np.cumsum(ee)  # errori tenuti se si tengono le prime i+1 take
    best = -np.inf
    for i in range(n):
        if i + 1 < n and ss[i + 1] == ss[i]:
            continue  # a parità di score si tengono tutte o nessuna
        if cum[i] <= max_error_rate * n + 1e-12:
            best = ss[i]
    return float(np.inf if best >= ss[-1] and cum[-1] <= max_error_rate * n + 1e-12 else best)


def budget_threshold(s: np.ndarray, rho: float) -> float:
    s = np.asarray(s, float)
    if rho <= 0 or len(s) == 0:
        return np.inf
    if rho >= 1:
        return -np.inf
    return float(np.quantile(s, 1.0 - rho, method="lower"))


def escalate(rule: str, value: float, s: np.ndarray, r_hat: np.ndarray, r_up: np.ndarray, threshold: float):
    """Decisione per le take di una fold: True = EgoVLP."""
    if rule == "cost":
        return np.asarray(r_up) >= value
    if rule == "cost_hat":
        return np.asarray(r_hat) >= value
    if rule in ("risk", "acc_target", "budget"):
        return np.asarray(s) > threshold
    raise ValueError(f"regola sconosciuta: {rule!r}")
