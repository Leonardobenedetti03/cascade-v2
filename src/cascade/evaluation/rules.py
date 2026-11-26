"""Decision rules of the gate v2. All of them are set on calibration data (out-of-sample scores of the outer
train) and then applied, frozen, to the outer fold.

- ``cost`` (β): escalate if the risk (``r_up``, or ``r_hat`` in the variant) is ≥ β. β is the price of one
  EgoVLP call measured in errors: 1/β calls for each error avoided. The threshold is β itself, it is not
  chosen on the data.
- ``risk`` (α, δ): CamFormer keeps the largest coverage for which the Clopper-Pearson upper bound on the error
  of the kept takes is ≤ α. The candidate coverages are a fixed grid of G values (1/G, 2/G, ..., 1), and each
  one is tested at level 1 − δ/G (Bonferroni), so the guarantee holds whichever coverage is chosen (selective
  classification with a guarantee, Geifman and El-Yaniv 2017, with the correction for choosing on the grid).
  A fixed sequence starting from the smallest coverage does not work here: with few kept takes the bound is
  wide, and the sequence would stop at once.
- ``acc_target`` (e_max): the smallest escalation share whose let-through errors are ≤ e_max · n (like
  ``within_pts`` in v1, assuming EgoVLP is right on the escalated takes). A reference, with no guarantee.
- ``budget`` (ρ): escalate the share ρ of takes with the highest score.

Thresholds on s mean "escalate if s > λ"; ``-inf`` = everything to EgoVLP, ``inf`` = nothing.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import beta as beta_dist


def clopper_pearson_upper(k: int, n: int, conf: float) -> float:
    """One-sided upper bound at level ``conf`` for a binomial proportion (k errors out of n)."""
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
    """Highest threshold (least escalation) whose let-through errors are ≤ max_error_rate · n."""
    s, e = np.asarray(s, float), np.asarray(e, int)
    n = len(s)
    if n == 0:
        return -np.inf
    order = np.argsort(s, kind="stable")
    ss, ee = s[order], e[order]
    cum = np.cumsum(ee)  # errors kept if we keep the first i+1 takes
    best = -np.inf
    for i in range(n):
        if i + 1 < n and ss[i + 1] == ss[i]:
            continue  # takes with the same score are all kept or all escalated
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
    """Decision for the takes of one fold: True = EgoVLP."""
    if rule == "cost":
        return np.asarray(r_up) >= value
    if rule == "cost_hat":
        return np.asarray(r_hat) >= value
    if rule in ("risk", "acc_target", "budget"):
        return np.asarray(s) > threshold
    raise ValueError(f"regola sconosciuta: {rule!r}")
