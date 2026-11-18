"""Curve accuratezza–costo della cascata e metriche derivate (C12b, M09 e M14).

Decisione: si ordinano le clip per score (decrescente, a parità vince l'ordine di riga) e si
scala il primo r% (r = 0..100). La predizione di una clip scalata è quella dopo l'escalation
(con ``fusion.mode = replace``: quella di L1), altrimenti quella di CamFormer. Per take si media
la log-probabilità delle sue clip e si prende l'argmax. Costo per take = n_clip · C0 + n_scalate · C1.

Tutte le metriche lavorano su array ``[..., R]`` così si applicano uguali alla curva puntuale e ai
ricampionamenti bootstrap ``[B, R]``.

Novità rispetto alla v1:
- accuratezza bilanciata (:func:`balanced_acc`): media sulle classi del richiamo per classe, con
  le sole classi che hanno almeno ``min_takes`` take nello split (:func:`class_membership`);
- fattore di miglioramento I(δ) di Lebovitz et al. (:func:`improvement_factor`);
- soglie dal budget (:func:`threshold_at_rate`, :func:`threshold_for_budget`) per il trasferimento
  val → test;
- niente decisioni forzate del livello −1 (non portato, M12) e ``roc_auc`` da scikit-learn.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12


@dataclass
class CurveData:
    r: np.ndarray  # [R] quota di escalation in %
    correct: np.ndarray  # [R, T] correttezza per take (float: media sui seed)
    cost: np.ndarray  # [R, T] GMACs per take
    take_labels: np.ndarray  # [T]
    member: Optional[np.ndarray] = None  # [C', T] indicatore delle take di ogni classe nella media

    @property
    def acc(self) -> np.ndarray:
        return self.correct.mean(1)

    @property
    def mean_cost(self) -> np.ndarray:
        return self.cost.mean(1)

    @property
    def bacc(self) -> np.ndarray:
        """Accuratezza bilanciata per ogni r (NaN se nessuna classe entra nella media)."""
        if self.member is None:
            raise ValueError("CurveData senza membership delle classi: costruirla con cascade_curve")
        return balanced_acc(self.correct, self.member)

    @property
    def n_classes(self) -> int:
        """Numero di classi nella media dell'accuratezza bilanciata."""
        return 0 if self.member is None else int(self.member.shape[0])


def escalation_order(scores: np.ndarray) -> np.ndarray:
    """Posizione di ogni clip nell'ordine di escalation (0 = prima a essere scalata)."""
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
    """Correttezza e costo per take per una o più maschere di escalation ``[R, N]``."""
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
    if fixed_r is not None:  # always_L0 / always_L1: stessa decisione per ogni r
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


# ------------------------------------------------------------------ accuratezza bilanciata (M09, D5)
def class_membership(take_labels: np.ndarray, n_classes: int, min_takes: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """``(M [C', T], keep [C'])``: riga = indicatore delle take di una classe.

    Le classi con meno di ``min_takes`` take nello split non entrano nella media (es. Bouldering)."""
    take_labels = np.asarray(take_labels).astype(int)
    counts = np.bincount(take_labels, minlength=n_classes)
    keep = np.flatnonzero(counts >= max(int(min_takes), 1))
    return (take_labels[None, :] == keep[:, None]).astype(float), keep


def balanced_acc(correct: np.ndarray, M: np.ndarray, W: Optional[np.ndarray] = None) -> np.ndarray:
    """Media sulle classi del richiamo per classe (= ``balanced_accuracy_score`` di scikit-learn).

    ``correct`` ``[R, T]`` correttezza per take; ``M`` ``[C, T]`` (:func:`class_membership`);
    ``W`` ``[B, T]`` pesi bootstrap per take. Restituisce ``[R]`` (W assente) oppure ``[B, R]``.
    Una classe assente da un ricampionamento esce dalla media di quel ricampionamento; senza
    classi nella media il risultato è NaN."""
    single = W is None
    if single:
        W = np.ones((1, correct.shape[1]))
    if M.shape[0] == 0:
        out = np.full((W.shape[0], correct.shape[0]), np.nan)
        return out[0] if single else out
    recalls = []
    for m in M:  # una classe alla volta: [B, R]
        den = np.einsum("bt,t->b", W, m)  # take della classe nel ricampionamento
        num = np.einsum("bt,rt->br", W * m, correct)
        recalls.append(np.where(den[:, None] > 0, num / np.maximum(den[:, None], EPS), np.nan))
    with warnings.catch_warnings():  # ricampionamento senza nessuna classe: NaN, senza avviso
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmean(np.stack(recalls), axis=0)
    return out[0] if single else out


# ------------------------------------------------------------------ metriche su [..., R]
def area_norm(cost: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Area sotto acc(costo) con costo normalizzato in [0, 1] (= accuracy media)."""
    span = cost[..., -1:] - cost[..., :1]
    x = (cost - cost[..., :1]) / np.where(span > 0, span, 1.0)
    return np.sum(np.diff(x, axis=-1) * (acc[..., 1:] + acc[..., :-1]) / 2.0, axis=-1)


def acc_at_budget(cost: np.ndarray, acc: np.ndarray, budget) -> np.ndarray:
    """Accuracy all'r più grande con costo ≤ budget (costo crescente in r); NaN se irraggiungibile."""
    budget = np.asarray(budget, dtype=float)[..., None]
    ok = cost <= budget * (1 + 1e-9)
    idx = np.where(ok.any(-1), ok.shape[-1] - 1 - np.argmax(ok[..., ::-1], axis=-1), 0)
    out = np.take_along_axis(acc, idx[..., None], -1)[..., 0]
    return np.where(ok.any(-1), out, np.nan)


def min_cost_within(cost: np.ndarray, acc: np.ndarray, ref_acc, pts: float) -> np.ndarray:
    """Costo minimo con acc ≥ ref_acc − pts (NaN se mai raggiunto)."""
    ok = acc >= np.asarray(ref_acc, dtype=float)[..., None] - pts - 1e-12
    c = np.where(ok, cost, np.inf).min(-1)
    return np.where(np.isfinite(c), c, np.nan)


def improvement_factor(cost: np.ndarray, acc: np.ndarray, ref_acc, delta: float) -> np.ndarray:
    """I(δ) = C_L1 / C(r*_δ), con r*_δ la quota minima che dà acc ≥ ref_acc − δ (Lebovitz et al., 2023).

    Il numeratore è il costo di L1 da solo, letto dalla curva: ``cost[r = 100] − cost[r = 0]``.
    Con ``replace`` a r = 100 la cascata ha l'accuratezza di L1, quindi I(δ) è sempre definito e vale
    almeno C_L1 / (C_L0 + C_L1) ≈ 0,9987: sotto 1 la cascata non risparmia nulla."""
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
    """Pesi bootstrap per take ``[B, T]`` (conteggi multinomiali)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_takes, np.full(n_takes, 1.0 / n_takes), size=n_boot).astype(float)


def group_resample_weights(take_groups: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Pesi bootstrap ``[B, T]`` ricampionando i GRUPPI (es. sessioni) con reinserimento (N37).

    ``take_groups[t]`` è il gruppo della take t, nell'ordine delle take della curva (``np.unique`` di
    ``take_uid``). Ogni gruppo estratto porta con sé tutte le sue take: il peso di una take è il numero
    di volte che il suo gruppo è stato estratto. Con un gruppo per take coincide in distribuzione con
    :func:`resample_weights`."""
    _, inv = np.unique(np.asarray(take_groups).astype(str), return_inverse=True)
    G = int(inv.max()) + 1
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(G, np.full(G, 1.0 / G), size=n_boot).astype(float)  # [B, G]
    return counts[:, inv]


def boot_curves(c: CurveData, W: np.ndarray):
    """(acc [B, R], cost [B, R]) ricampionando le take con i pesi ``W``.

    Si usa ``einsum`` e non ``@``: con numpy 2.0 e Accelerate (macOS arm64) la moltiplicazione tra matrici
    emette avvisi spuri («divide by zero / overflow / invalid value in matmul») su numeri finiti."""
    T = c.correct.shape[1]
    return np.einsum("bt,rt->br", W, c.correct) / T, np.einsum("bt,rt->br", W, c.cost) / T


def boot_bacc(c: CurveData, W: np.ndarray) -> np.ndarray:
    """Accuratezza bilanciata ``[B, R]`` sugli stessi ricampionamenti di :func:`boot_curves`."""
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
    """Metriche di una curva (o di B ricampionamenti). ``acc`` è l'accuratezza semplice o la
    bilanciata: le chiavi sono le stesse, e chi chiama le distingue con un prefisso."""
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
    """``num / den``, NaN dove il denominatore è ~0 (oracolo = random), senza avvisi."""
    num, den = np.broadcast_arrays(np.asarray(num, dtype=float), np.asarray(den, dtype=float))
    out = np.full(num.shape, np.nan)
    ok = np.abs(den) > 1e-12
    np.divide(num, den, out=out, where=ok)
    return out


# ------------------------------------------------------------------ soglie e budget (M09, RQ1)
def threshold_at_rate(scores: np.ndarray, rate: float) -> float:
    """Soglia che scala la quota ``rate`` delle clip (si scala se score >= soglia); ``inf`` se è 0."""
    k = int(np.floor(rate * len(scores) + 1e-9))
    return float("inf") if k == 0 else float(np.sort(scores)[::-1][k - 1])


def rate_for_budget(budget_frac: float, c_l0: float, c_l1: float) -> float:
    """Quota di escalation con costo medio = ``budget_frac`` · (C_L0 + C_L1), fra 0 e 1."""
    rate = (budget_frac * (c_l0 + c_l1) - c_l0) / c_l1
    return float(np.clip(rate, 0.0, 1.0))


def threshold_for_budget(scores: np.ndarray, budget_frac: float, c_l0: float, c_l1: float) -> float:
    """Soglia con costo medio ≤ ``budget_frac`` · costo di always_L1 (= C_L0 + C_L1)."""
    return threshold_at_rate(scores, rate_for_budget(budget_frac, c_l0, c_l1))
