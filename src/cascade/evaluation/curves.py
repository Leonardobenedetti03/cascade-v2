"""Accuracy–cost curves of the cascade and the metrics derived from them (C12b, M09 and M14).

Decision: we sort the clips by score (descending; ties keep the row order) and escalate the first r%
(r = 0..100). An escalated clip takes the prediction after escalation (with ``fusion.mode = replace``, the one
of L1), the others keep CamFormer's. For each take we average the log-probabilities of its clips and take the
argmax. Cost per take = n_clips · C0 + n_escalated · C1.

All metrics work on arrays ``[..., R]``, so the same code applies to the point curve and to the bootstrap
resamples ``[B, R]``.

New compared with v1:
- balanced accuracy (:func:`balanced_acc`): mean over classes of the per-class recall, using only the classes
  with at least ``min_takes`` takes in the split (:func:`class_membership`);
- the improvement factor I(δ) of Lebovitz et al. (:func:`improvement_factor`);
- thresholds from the budget (:func:`threshold_at_rate`, :func:`threshold_for_budget`), to carry them from val
  to test;
- no forced decisions of level −1 (not carried over, M12), and ``roc_auc`` from scikit-learn.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12


@dataclass
class CurveData:
    r: np.ndarray  # [R] escalation share in %
    correct: np.ndarray  # [R, T] correctness per take (float: mean over seeds)
    cost: np.ndarray  # [R, T] GMACs per take
    take_labels: np.ndarray  # [T]
    member: Optional[np.ndarray] = None  # [C', T] indicator of the takes of each class in the mean

    @property
    def acc(self) -> np.ndarray:
        return self.correct.mean(1)

    @property
    def mean_cost(self) -> np.ndarray:
        return self.cost.mean(1)

    @property
    def bacc(self) -> np.ndarray:
        """Balanced accuracy for each r (NaN if no class enters the mean)."""
        if self.member is None:
            raise ValueError("CurveData senza membership delle classi: costruirla con cascade_curve")
        return balanced_acc(self.correct, self.member)

    @property
    def n_classes(self) -> int:
        """Number of classes in the mean of the balanced accuracy."""
        return 0 if self.member is None else int(self.member.shape[0])


def escalation_order(scores: np.ndarray) -> np.ndarray:
    """Position of each clip in the escalation order (0 = the first to be escalated)."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    pos = np.empty(len(order), dtype=np.int64)
    pos[order] = np.arange(len(order))
    return pos


def _take_index(take_uid: np.ndarray):
    uniq, inv = np.unique(np.asarray(take_uid).astype(str), return_inverse=True)
    return uniq, inv


def evaluate_masks(
    esc: np.ndarray,
    p_cam: np.ndarray,
    p_fused: np.ndarray,
    label_idx: np.ndarray,
    take_uid: np.ndarray,
    gmacs_L0: float,
    gmacs_L1: float,
):
    """Correctness and cost per take for one or more escalation masks ``[R, N]``."""
    esc = np.atleast_2d(esc)
    uniq, inv = _take_index(take_uid)
    T = len(uniq)
    lc, lf = np.log(np.clip(p_cam, EPS, 1)), np.log(np.clip(p_fused, EPS, 1))
    logp = np.where(esc[:, :, None], lf[None], lc[None])  # [R, N, C]
    agg = np.zeros((esc.shape[0], T, p_cam.shape[1]))
    np.add.at(agg, (slice(None), inv), logp)
    n_clips = np.bincount(inv, minlength=T)
    take_label = np.zeros(T, dtype=int)
    take_label[inv] = label_idx
    correct = (agg.argmax(2) == take_label[None]).astype(float)
    n_esc = np.zeros((esc.shape[0], T))
    np.add.at(n_esc, (slice(None), inv), esc.astype(float))
    cost = n_clips[None] * gmacs_L0 + n_esc * gmacs_L1
    return correct, cost, take_label


def cascade_curve(
    scores: np.ndarray,
    p_cam: np.ndarray,
    p_fused: np.ndarray,
    label_idx: np.ndarray,
    take_uid: np.ndarray,
    gmacs_L0: float,
    gmacs_L1: float,
    r_step: int = 1,
    fixed_r: Optional[int] = None,
    min_takes: int = 1,
) -> CurveData:
    n = len(scores)
    r = np.arange(0, 101, r_step)
    if r[-1] != 100:
        r = np.append(r, 100)
    k = np.round(r / 100.0 * n).astype(int)
    if fixed_r is not None:  # always_L0 / always_L1: the same decision for every r
        k = np.full_like(k, int(round(fixed_r / 100.0 * n)))
    pos = escalation_order(scores)
    esc = pos[None, :] < k[:, None]
    correct, cost, take_label = evaluate_masks(esc, p_cam, p_fused, label_idx, take_uid, gmacs_L0, gmacs_L1)
    member, _ = class_membership(take_label, p_cam.shape[1], min_takes)
    return CurveData(r=r, correct=correct, cost=cost, take_labels=take_label, member=member)


def mean_curve(curves) -> CurveData:
    curves = list(curves)
    return CurveData(
        curves[0].r,
        np.mean([c.correct for c in curves], 0),
        np.mean([c.cost for c in curves], 0),
        curves[0].take_labels,
        curves[0].member,
    )


# Balanced accuracy (M09, D5)
def class_membership(take_labels: np.ndarray, n_classes: int, min_takes: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """``(M [C', T], keep [C'])``: each row is the indicator of the takes of one class.

    Classes with fewer than ``min_takes`` takes in the split do not enter the mean (e.g. Bouldering)."""
    take_labels = np.asarray(take_labels).astype(int)
    counts = np.bincount(take_labels, minlength=n_classes)
    keep = np.flatnonzero(counts >= max(int(min_takes), 1))
    return (take_labels[None, :] == keep[:, None]).astype(float), keep


def balanced_acc(correct: np.ndarray, M: np.ndarray, W: Optional[np.ndarray] = None) -> np.ndarray:
    """Mean over classes of the per-class recall (= scikit-learn's ``balanced_accuracy_score``).

    ``correct`` ``[R, T]`` correctness per take; ``M`` ``[C, T]`` (:func:`class_membership`); ``W`` ``[B, T]``
    bootstrap weights per take. Returns ``[R]`` (without W) or ``[B, R]``. A class missing from a resample leaves
    the mean of that resample; with no classes left the result is NaN."""
    single = W is None
    if single:
        W = np.ones((1, correct.shape[1]))
    if M.shape[0] == 0:
        out = np.full((W.shape[0], correct.shape[0]), np.nan)
        return out[0] if single else out
    recalls = []
    for m in M:  # one class at a time: [B, R]
        den = np.einsum("bt,t->b", W, m)  # takes of the class in the resample
        num = np.einsum("bt,rt->br", W * m, correct)
        recalls.append(np.where(den[:, None] > 0, num / np.maximum(den[:, None], EPS), np.nan))
    with warnings.catch_warnings():  # resample without any class: NaN, without a warning
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmean(np.stack(recalls), axis=0)
    return out[0] if single else out


# Metrics on [..., R]
def area_norm(cost: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Area under acc(cost) with the cost normalised to [0, 1] (= mean accuracy)."""
    span = cost[..., -1:] - cost[..., :1]
    x = (cost - cost[..., :1]) / np.where(span > 0, span, 1.0)
    return np.sum(np.diff(x, axis=-1) * (acc[..., 1:] + acc[..., :-1]) / 2.0, axis=-1)


def acc_at_budget(cost: np.ndarray, acc: np.ndarray, budget) -> np.ndarray:
    """Accuracy at the largest r with cost ≤ budget (cost grows with r); NaN if out of reach."""
    budget = np.asarray(budget, dtype=float)[..., None]
    ok = cost <= budget * (1 + 1e-9)
    idx = np.where(ok.any(-1), ok.shape[-1] - 1 - np.argmax(ok[..., ::-1], axis=-1), 0)
    out = np.take_along_axis(acc, idx[..., None], -1)[..., 0]
    return np.where(ok.any(-1), out, np.nan)


def min_cost_within(cost: np.ndarray, acc: np.ndarray, ref_acc, pts: float) -> np.ndarray:
    """Lowest cost with acc ≥ ref_acc − pts (NaN if never reached)."""
    ok = acc >= np.asarray(ref_acc, dtype=float)[..., None] - pts - 1e-12
    c = np.where(ok, cost, np.inf).min(-1)
    return np.where(np.isfinite(c), c, np.nan)


def improvement_factor(cost: np.ndarray, acc: np.ndarray, ref_acc, delta: float) -> np.ndarray:
    """I(δ) = C_L1 / C(r*_δ) (Lebovitz et al., 2023), where r*_δ is the smallest share with acc ≥ ref_acc − δ.

    The numerator is the cost of L1 alone, read from the curve: ``cost[r = 100] − cost[r = 0]``.
    With ``replace``, at r = 100 the cascade has the accuracy of L1, so I(δ) is always defined and is at least
    C_L1 / (C_L0 + C_L1) ≈ 0.9987. Below 1 the cascade saves nothing."""
    c_l1 = cost[..., -1] - cost[..., 0]
    return _ratio(c_l1, min_cost_within(cost, acc, ref_acc, delta))


def average_precision(scores: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if y.min() == y.max():
        return float("nan")
    return float(average_precision_score(y, scores))


def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, scores))


def resample_weights(n_takes: int, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap weights per take ``[B, T]`` (multinomial counts)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_takes, np.full(n_takes, 1.0 / n_takes), size=n_boot).astype(float)


def group_resample_weights(take_groups: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap weights ``[B, T]`` that resample GROUPS (e.g. sessions) with replacement (N37).

    ``take_groups[t]`` is the group of take t, in the take order of the curve (``np.unique`` of ``take_uid``).
    Each drawn group brings all its takes, so the weight of a take is the number of times its group was drawn.
    With one group per take this has the same distribution as :func:`resample_weights`."""
    _, inv = np.unique(np.asarray(take_groups).astype(str), return_inverse=True)
    G = int(inv.max()) + 1
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(G, np.full(G, 1.0 / G), size=n_boot).astype(float)  # [B, G]
    return counts[:, inv]


def boot_curves(c: CurveData, W: np.ndarray):
    """(acc [B, R], cost [B, R]) resampling the takes with the weights ``W``.

    We use ``einsum`` and not ``@``: with numpy 2.0 and Accelerate (macOS arm64), matrix multiplication gives
    spurious warnings ("divide by zero / overflow / invalid value in matmul") on finite numbers."""
    T = c.correct.shape[1]
    return np.einsum("bt,rt->br", W, c.correct) / T, np.einsum("bt,rt->br", W, c.cost) / T


def boot_bacc(c: CurveData, W: np.ndarray) -> np.ndarray:
    """Balanced accuracy ``[B, R]`` on the same resamples as :func:`boot_curves`."""
    if c.member is None:
        raise ValueError("CurveData senza membership delle classi")
    return balanced_acc(c.correct, c.member, W)


def curve_metrics(
    cost: np.ndarray,
    acc: np.ndarray,
    *,
    ref_cost_L1,
    ref_acc_L1,
    budgets,
    within_pts,
    area_random=None,
    area_oracle=None,
    acc_random_b=None,
    acc_oracle_b=None,
    deltas: Sequence[float] = (),
) -> Dict[str, np.ndarray]:
    """Metrics of one curve (or of B resamples). ``acc`` is either the plain or the balanced accuracy: the keys
    are the same, and the caller tells them apart with a prefix."""
    out = {"area": area_norm(cost, acc), "acc_r0": acc[..., 0], "acc_r100": acc[..., -1]}
    for b in budgets:
        out[f"acc_at_{int(round(b * 100))}"] = acc_at_budget(cost, acc, np.asarray(ref_cost_L1) * b)
    mc = min_cost_within(cost, acc, ref_acc_L1, within_pts)
    out["min_cost_within"] = mc
    out["min_cost_within_frac"] = mc / np.asarray(ref_cost_L1)
    for d in deltas:
        out[f"improvement_d{int(round(d * 100))}"] = improvement_factor(cost, acc, ref_acc_L1, d)
    if area_random is not None and area_oracle is not None:
        out["oracle_gain_frac"] = _ratio(out["area"] - area_random, np.asarray(area_oracle) - area_random)
    if acc_random_b is not None and acc_oracle_b is not None:
        for b in budgets:
            key = f"acc_at_{int(round(b * 100))}"
            out[f"oracle_gain_frac_{int(round(b * 100))}"] = _ratio(
                out[key] - acc_random_b[key], acc_oracle_b[key] - acc_random_b[key]
            )
    return out


def _ratio(num, den) -> np.ndarray:
    """``num / den``, NaN where the denominator is ~0 (oracle = random), without warnings."""
    num, den = np.broadcast_arrays(np.asarray(num, dtype=float), np.asarray(den, dtype=float))
    out = np.full(num.shape, np.nan)
    ok = np.abs(den) > 1e-12
    np.divide(num, den, out=out, where=ok)
    return out


# Thresholds and budget (M09, RQ1)
def threshold_at_rate(scores: np.ndarray, rate: float) -> float:
    """Threshold that escalates the share ``rate`` of the clips (escalate if score >= threshold); ``inf`` if it is 0."""
    k = int(np.floor(rate * len(scores) + 1e-9))
    return float("inf") if k == 0 else float(np.sort(scores)[::-1][k - 1])


def rate_for_budget(budget_frac: float, c_l0: float, c_l1: float) -> float:
    """Escalation share whose mean cost is ``budget_frac`` · (C_L0 + C_L1), between 0 and 1."""
    rate = (budget_frac * (c_l0 + c_l1) - c_l0) / c_l1
    return float(np.clip(rate, 0.0, 1.0))


def threshold_for_budget(scores: np.ndarray, budget_frac: float, c_l0: float, c_l1: float) -> float:
    """Threshold with mean cost ≤ ``budget_frac`` · the cost of always_L1 (= C_L0 + C_L1)."""
    return threshold_at_rate(scores, rate_for_budget(budget_frac, c_l0, c_l1))
