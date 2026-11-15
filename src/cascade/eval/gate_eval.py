"""Valutazione dei gate sulla cache (logica dello stadio 50, C12b, M09 e M14).

Per ogni gate e seed: score con il protocollo di ``gate/protocol.py``. Per ogni split valutato
(train fuori campione, val, e test solo con ``final``):

- curva accuratezza–costo e metriche (area, accuratezza a budget, costo minimo entro un punto da
  always_L1, quota del guadagno dell'oracolo), in due versioni: accuratezza semplice e accuratezza
  bilanciata (``bacc_*``, D5), più il fattore di miglioramento I(δ) (``improvement_d0``,
  ``improvement_d1``); IC bootstrap per take (val/test);
- confronti appaiati tra gate (RQ2): differenze sugli stessi ricampionamenti per take, con IC e
  p-value bootstrap a due code;
- AUROC/AUPRC dello score su y, con IC bootstrap per take;
- accuratezza per classe con IC e matrici di confusione (per take) di L0, L1, fusione e cascata al
  punto di lavoro del gate di riferimento;
- calibrazione dello score dei gate probabilistici e numero di positivi y = 1 per fold e classe;
- ``operating_point``: per ogni gate, quota tenuta (coverage), accuratezza di L0 sulle clip tenute e
  di L1 su quelle scalate; per i gate compare anche il punto naturale s > 0;
- ``threshold_transfer`` (RQ1): la soglia scelta su val applicata al test, con lo sforamento di costo;
- ``diag_c1`` (D1, D2): AUROC di η̂₁ contro c1 per compare_A e compare_AC, su train e val, mai sul test;
- ``cells``: masse delle celle A, B, D, N (correttezze di L0 e dopo l'escalation) per split e classe.

Il punto di lavoro (budget ``gate.operating_budget`` × costo di always_L1) si sceglie su val come
soglia sullo score e si applica tale e quale al test. Non si portano dalla v1 la cascata a tre livelli,
l'analisi degli errori e le decisioni forzate del livello −1 (M12).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.cascade.eval.curves import (
    CurveData,
    auroc,
    average_precision,
    balanced_acc,
    boot_bacc,
    boot_curves,
    cascade_curve,
    class_membership,
    curve_metrics,
    escalation_order,
    evaluate_masks,
    mean_curve,
    resample_weights,
    threshold_for_budget,
)
from src.cascade.gate.models import make_gate
from src.cascade.gate.protocol import GateResult, run_gate

log = logging.getLogger(__name__)
REFERENCE_MODELS = ("always_L0", "always_L1", "random", "oracle")
OUTPUT_TABLES = (
    "curves",
    "metrics",
    "auc",
    "operating_point",
    "escalation_by_class",
    "importance",
    "comparisons",
    "per_class",
    "confusion",
    "gate_calibration",
    "positives",
    "curve_bands",
    "threshold_transfer",
    "diag_c1",
    "cells",
)
SERIES = ("acc", "bacc")
COMPARE_METRICS = ("area", "oracle_gain_frac", "bacc_area", "bacc_oracle_gain_frac")
DIAG_SPLITS = ("train", "val")  # D1 e D2 non guardano mai il test
EPS = 1e-12


# --------------------------------------------------------------------------- utilità
def _ci(v: np.ndarray) -> Tuple[float, float, float]:
    """(low, high, frazione di ricampionamenti validi): una metrica può essere indefinita in
    alcuni ricampionamenti (es. budget non raggiungibile); la frazione lo rende visibile."""
    v = np.asarray(v, dtype=float)
    ok = np.isfinite(v)
    if not ok.any():
        return (np.nan, np.nan, 0.0)
    return (float(np.percentile(v[ok], 2.5)), float(np.percentile(v[ok], 97.5)), float(ok.mean()))


def _p_two_sided(d: np.ndarray) -> float:
    """p-value bootstrap a due code per H0: differenza = 0 (percentile, con correzione +1)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if not len(d):
        return float("nan")
    p = 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1))
    return float(min(p, 1.0))


def _nanmean_dicts(items: List[dict]) -> dict:
    """Media chiave per chiave di dizionari numerici (NaN ignorati, nessun avviso)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {k: float(np.nanmean([it[k] for it in items])) for k in items[0]}


def model_seeds(name: str, gcfg) -> List[int]:
    if name == "random":
        return list(range(gcfg.random_seeds))
    if make_gate(name, gcfg).stochastic:
        return list(gcfg.seeds)
    return [int(gcfg.seeds[0]) if gcfg.seeds else 0]


def run_all_gates(table, labels, gcfg) -> Dict[str, List[GateResult]]:
    names = list(dict.fromkeys(list(gcfg.models) + list(REFERENCE_MODELS)))
    out = {}
    for name in names:
        out[name] = [run_gate(table, labels, gcfg, name, seed) for seed in model_seeds(name, gcfg)]
        log.info("gate %s: %d seed", name, len(out[name]))
    return out


def apply_fusion_mode(table, mode: str):
    """Ablation fusione/sostituzione: con ``replace`` le clip scalate usano solo L1 (w = 0)."""
    from src.cascade.cache.schema import add_labels
    from src.cascade.fusion import fuse

    if mode == table.meta.get("fusion_mode", "fuse"):
        return table
    if mode != "replace":
        raise ValueError(f"fusion.mode={mode!r} diverso dalla cache: rifai 35 e 45")
    t = table.take(np.arange(len(table)))
    t.arrays = dict(t.arrays)
    t.arrays["fused_probs"] = fuse(t.arrays["cam_probs_cal"], t.arrays["vid_probs_cal"], 0.0).astype(np.float32)
    t.df["w"] = 0.0
    add_labels(t)
    t.meta = {**table.meta, "fusion_mode": "replace"}
    log.info("ablation: sostituzione al livello video (w = 0), c1 e y ricalcolati")
    return t


@dataclass
class SplitData:
    """Righe di uno split e gli array che servono alla valutazione."""

    name: str
    rows: np.ndarray
    p_cam: np.ndarray
    p_vid: np.ndarray
    p_fused: np.ndarray
    label_idx: np.ndarray
    take_uid: np.ndarray
    y: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    gmacs_L0: float
    gmacs_L1: float
    min_takes: int = 1  # bacc: classi con almeno ``min_takes`` take nello split

    @classmethod
    def build(cls, table, split: str, min_takes: int = 1) -> SplitData:
        rows = np.flatnonzero(table.df.split.to_numpy() == split)
        df = table.df.iloc[rows]
        has = len(rows) > 0
        return cls(
            split,
            rows,
            table.arrays["cam_probs_cal"][rows],
            table.arrays["vid_probs_cal"][rows],
            table.arrays["fused_probs"][rows],
            df.label_idx.to_numpy(),
            df.take_uid.to_numpy(),
            df.y.to_numpy().astype(int),
            df.c0.to_numpy().astype(int),
            df.c1.to_numpy().astype(int),
            float(df.gmacs_L0.iloc[0]) if has else float("nan"),
            float(df.gmacs_L1.iloc[0]) if has else float("nan"),
            int(min_takes),
        )

    @property
    def curve_args(self) -> dict:
        return dict(
            p_cam=self.p_cam,
            p_fused=self.p_fused,
            label_idx=self.label_idx,
            take_uid=self.take_uid,
            gmacs_L0=self.gmacs_L0,
            gmacs_L1=self.gmacs_L1,
            min_takes=self.min_takes,
        )

    def masks_eval(self, esc: np.ndarray):
        return evaluate_masks(
            esc[None], self.p_cam, self.p_fused, self.label_idx, self.take_uid, self.gmacs_L0, self.gmacs_L1
        )

    def point(self, esc: np.ndarray) -> Dict[str, float]:
        """Un punto di lavoro: accuratezza (per take, semplice e bilanciata), costo medio per take,
        quota scalata, quota tenuta (coverage) e, per clip, accuratezza di L0 sulle clip tenute e di
        L1 (dopo l'escalation) su quelle scalate."""
        esc = np.asarray(esc, dtype=bool)
        corr, cost, take_label = self.masks_eval(esc)
        M, _ = class_membership(take_label, self.p_cam.shape[1], self.min_takes)
        keep = ~esc
        return {
            "acc": float(corr.mean()),
            "bacc": float(balanced_acc(corr, M)[0]),
            "cost": float(cost.mean()),
            "escalation_rate": float(esc.mean()),
            "coverage": float(keep.mean()),
            "acc_L0_kept": float(self.c0[keep].mean()) if keep.any() else float("nan"),
            "acc_L1_escalated": float(self.c1[esc].mean()) if esc.any() else float("nan"),
            "n_classes": float(M.shape[0]),
        }


# --------------------------------------------------------------------------- punto di lavoro
def choose_operating_points(table, results, gcfg) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """Su val: per ogni gate e seed la soglia che rispetta ``operating_budget``; per l'oracolo
    (score discreti) la quota r* corrispondente."""
    val = SplitData.build(table, "val", gcfg.bacc_min_takes)
    thresholds: Dict[str, List[float]] = {}
    rates: Dict[str, float] = {}
    if not len(val.rows):
        return thresholds, rates
    _, ref_cost, _ = val.masks_eval(np.ones(len(val.rows), bool))
    budget = gcfg.operating_budget * float(ref_cost.mean())
    for name, res in results.items():
        thresholds[name] = []
        for gr in res:
            sc = gr.scores[val.rows]
            c = cascade_curve(sc, **val.curve_args, r_step=gcfg.r_step, fixed_r=gr.fixed_r)
            ok = c.mean_cost <= budget * (1 + 1e-9)
            r_idx = int(np.flatnonzero(ok)[-1]) if ok.any() else 0
            k = int(round(c.r[r_idx] / 100.0 * len(val.rows)))
            ranked = sc[np.argsort(escalation_order(sc))]
            thresholds[name].append(float(ranked[k - 1]) if k > 0 else float("inf"))
            rates.setdefault(name, c.r[r_idx] / 100.0)
    return thresholds, rates


def escalate_at_operating_point(name: str, gr: GateResult, thr: float, rate: float, sd: SplitData) -> np.ndarray:
    sc = gr.scores[sd.rows]
    if gr.fixed_r is not None:
        return np.full(len(sd.rows), gr.fixed_r == 100)
    if name == "oracle":
        return escalation_order(sc) < int(round(rate * len(sd.rows)))
    return (sc >= thr) if np.isfinite(thr) else np.zeros(len(sd.rows), bool)


def operating_rows(results, thresholds, rates, sd: SplitData, ref_cost_L1: float, labels, gcfg):
    """Punto di lavoro di ogni gate (budget scelto su val) e, per i gate compare, punto naturale s > 0."""
    op_rows, class_rows = [], []

    def row(name, point, thr, budget, escs):
        m = _nanmean_dicts([sd.point(e) for e in escs])
        return {
            "split": sd.name,
            "model": name,
            "point": point,
            "threshold": thr,
            "escalation_rate": m["escalation_rate"],
            "coverage": m["coverage"],
            "acc": m["acc"],
            "bacc": m["bacc"],
            "cost": m["cost"],
            "cost_frac_L1": m["cost"] / ref_cost_L1,
            "budget_frac": budget,
            "acc_L0_kept": m["acc_L0_kept"],
            "acc_L1_escalated": m["acc_L1_escalated"],
            "n_classes": int(m["n_classes"]),
        }

    for name, res in results.items():
        escs = [
            escalate_at_operating_point(name, gr, thr, rates.get(name, 0.0), sd)
            for gr, thr in zip(res, thresholds[name])
        ]
        op_rows.append(row(name, "budget", float(np.mean(thresholds[name])), gcfg.operating_budget, escs))
        if name.startswith("compare_") and res[0].fixed_r is None:  # soglia naturale: L1 più probabile di L0
            op_rows.append(row(name, "s>0", 0.0, float("nan"), [gr.scores[sd.rows] > 0 for gr in res]))
        per_class = [
            [esc[sd.label_idx == c].mean() if (sd.label_idx == c).any() else np.nan for c in range(labels.num_classes)]
            for esc in escs
        ]
        with warnings.catch_warnings():  # classe assente dallo split: NaN, senza avviso
            warnings.simplefilter("ignore", RuntimeWarning)
            per_class_mean = np.nanmean(np.array(per_class), 0)
        for c, v in enumerate(per_class_mean):
            class_rows.append(
                {
                    "split": sd.name,
                    "model": name,
                    "class": labels.names[c],
                    "procedural": bool(labels.procedural_mask[c]),
                    "escalation_rate": float(v),
                }
            )
    return op_rows, class_rows


# --------------------------------------------------------------------------- curve e metriche
def split_curves(results, sd: SplitData, gcfg) -> Tuple[Dict[str, CurveData], Dict[str, List[CurveData]]]:
    mean, per_seed = {}, {}
    for name, res in results.items():
        cs = [cascade_curve(gr.scores[sd.rows], **sd.curve_args, r_step=gcfg.r_step, fixed_r=gr.fixed_r) for gr in res]
        mean[name], per_seed[name] = mean_curve(cs), cs
    return mean, per_seed


def curve_rows(results, per_seed, sd: SplitData, pixels_L1: float) -> List[dict]:
    """Punti delle curve; ``pixels`` = pixel letti medi per take (solo L1 legge pixel)."""
    rows = []
    n_clips = float(len(sd.rows)) / max(len(np.unique(sd.take_uid.astype(str))), 1)
    for name, res in results.items():
        for gr, c in zip(res, per_seed[name]):
            bacc = c.bacc
            rows += [
                {
                    "split": sd.name,
                    "model": name,
                    "seed": gr.seed,
                    "r": int(rv),
                    "cost": float(c.mean_cost[i]),
                    "acc": float(c.acc[i]),
                    "bacc": float(bacc[i]),
                    "pixels": float((c.mean_cost[i] - n_clips * sd.gmacs_L0) / sd.gmacs_L1 * pixels_L1),
                }
                for i, rv in enumerate(c.r)
            ]
    return rows


def _series(c: CurveData, kind: str) -> np.ndarray:
    return c.acc if kind == "acc" else c.bacc


def _tag(d: Dict[str, np.ndarray], kind: str) -> Dict[str, np.ndarray]:
    """Chiavi della versione bilanciata: ``area`` → ``bacc_area``, ``acc_at_50`` → ``bacc_at_50``."""
    if kind == "acc":
        return d
    return {"bacc_" + (k[4:] if k.startswith("acc_") else k): v for k, v in d.items()}


def _kw(l1_cost, l1_acc, gcfg) -> dict:
    return dict(
        ref_cost_L1=l1_cost,
        ref_acc_L1=l1_acc,
        budgets=gcfg.budgets,
        within_pts=gcfg.within_pts,
        deltas=gcfg.improvement_deltas,
    )


def point_metrics(c: CurveData, curves: Dict[str, CurveData], gcfg) -> Dict[str, np.ndarray]:
    """Metriche di una curva (accuratezza semplice e bilanciata), con random e oracolo come riferimento."""
    out: Dict[str, np.ndarray] = {}
    for kind in SERIES:
        kw = _kw(float(curves["always_L1"].mean_cost[-1]), float(_series(curves["always_L1"], kind)[-1]), gcfg)
        ref = {n: curve_metrics(curves[n].mean_cost, _series(curves[n], kind), **kw) for n in ("random", "oracle")}
        m = curve_metrics(
            c.mean_cost,
            _series(c, kind),
            area_random=ref["random"]["area"],
            area_oracle=ref["oracle"]["area"],
            acc_random_b=ref["random"],
            acc_oracle_b=ref["oracle"],
            **kw,
        )
        out.update(_tag(m, kind))
    return out


def bootstrap_metrics(curves: Dict[str, CurveData], W: np.ndarray, gcfg) -> Dict[str, Dict[str, np.ndarray]]:
    """Metriche su B ricampionamenti per take, con gli stessi pesi per tutti i gate (appaiati)."""
    out: Dict[str, Dict[str, np.ndarray]] = {n: {} for n in curves}
    for kind in SERIES:
        boot = {}
        for n, c in curves.items():
            acc_b, cost_b = boot_curves(c, W)
            boot[n] = (acc_b if kind == "acc" else boot_bacc(c, W), cost_b)
        bkw = _kw(boot["always_L1"][1][:, -1], boot["always_L1"][0][:, -1], gcfg)
        ref = {n: curve_metrics(boot[n][1], boot[n][0], **bkw) for n in ("random", "oracle")}
        for n, (ba, bc) in boot.items():
            m = curve_metrics(
                bc,
                ba,
                area_random=ref["random"]["area"],
                area_oracle=ref["oracle"]["area"],
                acc_random_b=ref["random"],
                acc_oracle_b=ref["oracle"],
                **bkw,
            )
            out[n].update(_tag(m, kind))
    return out


def band_rows(curves: Dict[str, CurveData], W: np.ndarray, split: str) -> List[dict]:
    """IC al 95% dell'accuratezza (semplice e bilanciata) per ogni r, per le figure con bande."""
    rows = []
    for name, c in curves.items():
        ba, _ = boot_curves(c, W)
        bb = boot_bacc(c, W)
        lo, hi = np.percentile(ba, 2.5, axis=0), np.percentile(ba, 97.5, axis=0)
        with warnings.catch_warnings():  # r senza nessuna classe valida: NaN, senza avviso
            warnings.simplefilter("ignore", RuntimeWarning)
            blo, bhi = np.nanpercentile(bb, 2.5, axis=0), np.nanpercentile(bb, 97.5, axis=0)
        acc, bacc = c.acc, c.bacc
        rows += [
            {
                "split": split,
                "model": name,
                "r": int(r),
                "cost": float(c.mean_cost[i]),
                "acc": float(acc[i]),
                "acc_lo": float(lo[i]),
                "acc_hi": float(hi[i]),
                "bacc": float(bacc[i]),
                "bacc_lo": float(blo[i]),
                "bacc_hi": float(bhi[i]),
            }
            for i, r in enumerate(c.r)
        ]
    return rows


def metric_rows(results, curves, per_seed, bm, split: str, gcfg) -> List[dict]:
    rows = []
    l1_cost = float(curves["always_L1"].mean_cost[-1])
    for name, c in curves.items():
        if results[name][0].fixed_r is not None:  # punto singolo: solo accuratezza e costo
            rows += [
                {
                    "split": split,
                    "model": name,
                    "metric": k,
                    "value": float(v),
                    "seed_std": 0.0,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "ci_valid": np.nan,
                    "n_classes": c.n_classes,
                }
                for k, v in (
                    ("acc", c.acc[0]),
                    ("bacc", c.bacc[0]),
                    ("cost", c.mean_cost[0]),
                    ("cost_frac", c.mean_cost[0] / l1_cost),
                )
            ]
            continue
        m = point_metrics(c, curves, gcfg)
        seeds = [point_metrics(cs, curves, gcfg) for cs in per_seed[name]]
        for k, v in m.items():
            sv = np.array([float(s[k]) for s in seeds])
            fin = sv[np.isfinite(sv)]
            lo, hi, valid = _ci(bm[name][k]) if bm is not None else (np.nan, np.nan, np.nan)
            rows.append(
                {
                    "split": split,
                    "model": name,
                    "metric": k,
                    "value": float(v),
                    "seed_std": float(fin.std()) if len(fin) > 1 else 0.0,
                    "ci_low": lo,
                    "ci_high": hi,
                    "ci_valid": valid,
                    "n_classes": c.n_classes,
                }
            )
    return rows


def comparison_rows(curves, bm, pairs, split: str, gcfg) -> List[dict]:
    """RQ2: differenze appaiate A − B (stessi ricampionamenti per take), su accuratezza semplice e bilanciata."""
    budgets = [int(round(b * 100)) for b in gcfg.budgets]
    keys = list(COMPARE_METRICS) + [f"acc_at_{b}" for b in budgets] + [f"bacc_at_{b}" for b in budgets]
    rows = []
    for a, b in pairs:
        if a not in curves or b not in curves:
            continue
        ma, mb = point_metrics(curves[a], curves, gcfg), point_metrics(curves[b], curves, gcfg)
        for k in keys:
            d = bm[a][k] - bm[b][k]
            lo, hi, valid = _ci(d)
            rows.append(
                {
                    "split": split,
                    "a": a,
                    "b": b,
                    "metric": k,
                    "diff": float(ma[k] - mb[k]),
                    "ci_low": lo,
                    "ci_high": hi,
                    "ci_valid": valid,
                    "p_two_sided": _p_two_sided(d),
                    "n_boot": int(len(d)),
                }
            )
    return rows


# --------------------------------------------------------------------------- AUROC e calibrazione dello score
def _take_resample_indices(take_uid: np.ndarray, B: int, seed: int) -> List[np.ndarray]:
    uniq, inv = np.unique(np.asarray(take_uid).astype(str), return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))]) for _ in range(B)]


def auc_rows(results, sd: SplitData, gcfg, n_boot: int) -> List[dict]:
    rows = []
    idxs = _take_resample_indices(sd.take_uid, n_boot, gcfg.bootstrap_seed) if n_boot else []
    for name, res in results.items():
        if res[0].fixed_r is not None or name == "oracle":
            continue
        vals = [(auroc(gr.scores[sd.rows], sd.y), average_precision(gr.scores[sd.rows], sd.y)) for gr in res]
        score = np.mean([gr.scores[sd.rows] for gr in res], 0)
        boots = np.array([auroc(score[i], sd.y[i]) for i in idxs]) if idxs else np.array([])
        lo, hi, _ = _ci(boots) if len(boots) else (np.nan, np.nan, np.nan)
        rows.append(
            {
                "split": sd.name,
                "model": name,
                "auroc": float(np.nanmean([v[0] for v in vals])),
                "auprc": float(np.nanmean([v[1] for v in vals])),
                "auroc_std": float(np.nanstd([v[0] for v in vals])),
                "auroc_ci_low": lo,
                "auroc_ci_high": hi,
                "y_rate": float(sd.y.mean()),
                "n_pos": int(sd.y.sum()),
            }
        )
    return rows


def calibration_rows(results, sd: SplitData, n_bins: int = 10) -> List[dict]:
    """Reliability diagram dei gate probabilistici: score medio vs frequenza di y = 1 per
    bin di quantile, ed ECE dello score."""
    rows = []
    for name, res in results.items():
        if not name.startswith(("logreg_", "lgbm_")):
            continue
        s = np.mean([gr.scores[sd.rows] for gr in res], 0)
        if not np.all((s >= 0) & (s <= 1)):
            continue
        edges = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
        b = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, max(len(edges) - 2, 0))
        ece = 0.0
        for k in np.unique(b):
            m = b == k
            ece += m.mean() * abs(sd.y[m].mean() - s[m].mean())
            rows.append(
                {
                    "split": sd.name,
                    "model": name,
                    "bin": int(k),
                    "score_mean": float(s[m].mean()),
                    "y_rate": float(sd.y[m].mean()),
                    "n": int(m.sum()),
                }
            )
        rows.append(
            {
                "split": sd.name,
                "model": name,
                "bin": -1,
                "score_mean": float(s.mean()),
                "y_rate": float(sd.y.mean()),
                "n": int(len(s)),
                "ece": float(ece),
            }
        )
    return rows


def positives_rows(table, labels) -> List[dict]:
    """Positivi y = 1 per split, fold e classe: con pochi positivi il gate è instabile."""
    df = table.df
    rows = []
    for (split, fold, lab), g in df.groupby(["split", "fold", "label_idx"]):
        if split == "test":
            continue
        rows.append(
            {
                "split": split,
                "fold": int(fold),
                "class": labels.names[int(lab)],
                "n": int(len(g)),
                "n_pos": int(g.y.sum()),
                "y_rate": float(g.y.mean()),
            }
        )
    return rows


# --------------------------------------------------------------------------- diagnostiche D1 e D2, celle
def diag_c1_rows(results, sd: SplitData, gcfg, p_max: np.ndarray) -> List[dict]:
    """D1 e D2: AUROC di η̂₁ contro c1, per i gate compare (η̂₁ = score + p_max).

    Con compare_A vicino a 0,5, η̂₁ è di fatto costante: la soglia sulla confidenza è già ottima
    (Proposizione 2). Se compare_AC sale rispetto a compare_A, la traiettoria dice dove L1 riesce."""
    if sd.name not in DIAG_SPLITS:
        raise ValueError(f"diag_c1: split {sd.name!r} non ammesso (solo {DIAG_SPLITS})")
    rows = []
    idxs = _take_resample_indices(sd.take_uid, gcfg.bootstrap, gcfg.bootstrap_seed) if sd.name == "val" else []
    for name, res in results.items():
        if not name.startswith("compare_") or res[0].fixed_r is not None:
            continue
        eta1 = np.mean([gr.scores[sd.rows] for gr in res], 0) + p_max
        boots = np.array([auroc(eta1[i], sd.c1[i]) for i in idxs]) if idxs else np.array([])
        lo, hi, _ = _ci(boots) if len(boots) else (np.nan, np.nan, np.nan)
        rows.append(
            {
                "split": sd.name,
                "model": name,
                "auroc": auroc(eta1, sd.c1),
                "auroc_ci_low": lo,
                "auroc_ci_high": hi,
                "eta1_std": float(eta1.std()),
                "c1_rate": float(sd.c1.mean()),
                "n": int(len(sd.c1)),
                "n_pos": int(sd.c1.sum()),
            }
        )
    return rows


def cells_rows(table, labels, split: str) -> List[dict]:
    """Masse delle celle A (c0 = c1 = 1), B (solo L0), D (solo dopo l'escalation), N (nessuno),
    per classe e in totale. Il guadagno massimo di qualunque gate è P(D); se P(D) è piccola o P(N)
    grande, la cascata ha poco da offrire a prescindere dal gate."""
    df = table.df[table.df.split == split]
    if not len(df):
        return []
    c0, c1 = df.c0.to_numpy().astype(int), df.c1.to_numpy().astype(int)
    lab = df.label_idx.to_numpy()
    rows = []
    for name, m in [("all", np.ones(len(df), bool))] + [
        (labels.names[int(c)], lab == c) for c in np.unique(lab) if (lab == c).any()
    ]:
        a, b = c0[m], c1[m]
        n = int(m.sum())
        cnt = {
            "A": int(((a == 1) & (b == 1)).sum()),
            "B": int(((a == 1) & (b == 0)).sum()),
            "D": int(((a == 0) & (b == 1)).sum()),
            "N": int(((a == 0) & (b == 0)).sum()),
        }
        rows.append(
            {"split": split, "class": name, "n": n}
            | {f"n_{k}": v for k, v in cnt.items()}
            | {f"p_{k}": v / n for k, v in cnt.items()}
        )
    return rows


# --------------------------------------------------------------------------- trasferimento delle soglie (RQ1)
def transfer_for_scores(s_val, s_test, val: SplitData, test: SplitData, budget: float) -> Dict[str, float]:
    """La soglia scelta su val per ``budget`` applicata al test, accanto alla soglia ricalcolata sul test
    per lo stesso budget (solo riferimento: non si usa mai per scegliere). Il costo è quello realizzato."""
    th_val = threshold_for_budget(s_val, budget, val.gmacs_L0, val.gmacs_L1)
    th_test = threshold_for_budget(s_test, budget, test.gmacs_L0, test.gmacs_L1)
    on_val = val.point(np.asarray(s_val) >= th_val)
    moved = test.point(np.asarray(s_test) >= th_val)
    reopt = test.point(np.asarray(s_test) >= th_test)
    nominal = budget * test.point(np.ones(len(s_test), bool))["cost"]  # budget × costo di always_L1
    return {
        "theta_val": th_val,
        "theta_test": th_test,
        "r_val": on_val["escalation_rate"],
        "r_transfer": moved["escalation_rate"],
        "r_reopt": reopt["escalation_rate"],
        "cost_transfer": moved["cost"],
        "cost_reopt": reopt["cost"],
        "cost_over_budget": moved["cost"] / nominal - 1.0,
        "cost_over_budget_reopt": reopt["cost"] / nominal - 1.0,
        "acc_val": on_val["acc"],
        "acc_transfer": moved["acc"],
        "acc_reopt": reopt["acc"],
        "acc_gap": moved["acc"] - reopt["acc"],
        "bacc_transfer": moved["bacc"],
        "bacc_reopt": reopt["bacc"],
        "bacc_gap": moved["bacc"] - reopt["bacc"],
    }


def threshold_transfer_rows(results, val: SplitData, test: SplitData, gcfg) -> List[dict]:
    """Una riga per gate e budget (media sui seed). Salta i gate a r fisso e l'oracolo."""
    rows = []
    for name, res in results.items():
        if res[0].fixed_r is not None or name == "oracle":
            continue
        for b in gcfg.transfer_budgets:
            items = [transfer_for_scores(gr.scores[val.rows], gr.scores[test.rows], val, test, b) for gr in res]
            rows.append(
                {"split": test.name, "model": name, "budget_frac": b, "n_seeds": len(res)} | _nanmean_dicts(items)
            )
    return rows


# --------------------------------------------------------------------------- per classe e confusione
def _take_level(p: np.ndarray, label_idx: np.ndarray, take_uid: np.ndarray):
    uniq, inv = np.unique(np.asarray(take_uid).astype(str), return_inverse=True)
    agg = np.zeros((len(uniq), p.shape[1]))
    np.add.at(agg, inv, np.log(np.clip(p, EPS, 1)))
    lab = np.zeros(len(uniq), dtype=int)
    lab[inv] = label_idx
    return agg.argmax(1), lab


def per_class_rows(systems: Dict[str, np.ndarray], sd: SplitData, labels, n_boot: int, seed: int):
    pc, conf = [], []
    rng = np.random.default_rng(seed)
    for sys_name, p in systems.items():
        pred, lab = _take_level(p, sd.label_idx, sd.take_uid)
        for c in range(labels.num_classes):
            m = lab == c
            if not m.any():
                continue
            corr = (pred[m] == c).astype(float)
            boots = corr[rng.integers(0, len(corr), (n_boot, len(corr)))].mean(1) if n_boot else np.array([])
            lo, hi, _ = _ci(boots) if n_boot else (np.nan, np.nan, np.nan)
            pc.append(
                {
                    "split": sd.name,
                    "system": sys_name,
                    "class": labels.names[c],
                    "procedural": bool(labels.procedural_mask[c]),
                    "acc": float(corr.mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_takes": int(m.sum()),
                }
            )
        cm = np.zeros((labels.num_classes, labels.num_classes), dtype=int)
        np.add.at(cm, (lab, pred), 1)
        conf += [
            {
                "split": sd.name,
                "system": sys_name,
                "true": labels.names[i],
                "pred": labels.names[j],
                "count": int(cm[i, j]),
            }
            for i in range(cm.shape[0])
            for j in range(cm.shape[1])
        ]
    return pc, conf


def reference_gate(results, curves_val: Optional[Dict[str, CurveData]], gcfg) -> Optional[str]:
    """Gate di riferimento per la cascata al punto di lavoro: il primo di ``gate.reference``
    presente, altrimenti il gate appreso con ``bacc_area`` maggiore su val (D5; ``area`` se la
    bilanciata non è definita)."""
    for name in gcfg.reference:
        if name in results:
            return name
    if not curves_val:
        return None
    learned = [n for n in results if n not in REFERENCE_MODELS and results[n][0].fixed_r is None]
    if not learned:
        return None
    areas = {n: point_metrics(curves_val[n], curves_val, gcfg) for n in learned}
    key = "bacc_area" if all(np.isfinite(float(a["bacc_area"])) for a in areas.values()) else "area"
    return max(learned, key=lambda n: float(areas[n][key]))


# --------------------------------------------------------------------------- orchestrazione
def evaluate(table, labels, cfg, *, final: bool) -> Dict[str, object]:
    gcfg = cfg.gate
    table = apply_fusion_mode(table, cfg.fusion.mode)
    results = run_all_gates(table, labels, gcfg)
    thresholds, rates = choose_operating_points(table, results, gcfg)
    out: Dict[str, List[dict]] = {k: [] for k in OUTPUT_TABLES}
    curves_val = None
    a = cfg.egovlp.arch
    pixels_L1 = (
        float(table.df.pixels_L1.iloc[0]) if "pixels_L1" in table.df else float(a.num_frames * a.img_size**2 * 3)
    )
    p_max = table.arrays["cam_probs_cal"].max(1)
    sds: Dict[str, SplitData] = {}
    for s in ["val", "train"] + (["test"] if final else []):  # val per primo: serve al riferimento
        sd = SplitData.build(table, s, gcfg.bacc_min_takes)
        if not len(sd.rows):
            continue
        sds[s] = sd
        curves, per_seed = split_curves(results, sd, gcfg)
        if s == "val":
            curves_val = curves
        bm = None
        if s != "train" and gcfg.bootstrap > 0:
            W = resample_weights(curves["random"].correct.shape[1], gcfg.bootstrap, gcfg.bootstrap_seed)
            bm = bootstrap_metrics(curves, W, gcfg)
            out["comparisons"] += comparison_rows(curves, bm, gcfg.comparisons, s, gcfg)
            out["curve_bands"] += band_rows(curves, W, s)
        out["curves"] += curve_rows(results, per_seed, sd, pixels_L1)
        out["metrics"] += metric_rows(results, curves, per_seed, bm, s, gcfg)
        out["auc"] += auc_rows(results, sd, gcfg, gcfg.bootstrap if s != "train" else 0)
        out["gate_calibration"] += calibration_rows(results, sd)
        out["cells"] += cells_rows(table, labels, s)
        if s in DIAG_SPLITS:
            out["diag_c1"] += diag_c1_rows(results, sd, gcfg, p_max[sd.rows])
        if s in ("val", "test") and thresholds:
            ref_cost_L1 = float(curves["always_L1"].mean_cost[-1])
            op, cl = operating_rows(results, thresholds, rates, sd, ref_cost_L1, labels, gcfg)
            out["operating_point"] += op
            out["escalation_by_class"] += cl
            ref = reference_gate(results, curves_val, gcfg)
            systems = {"L0": sd.p_cam, "L1": sd.p_vid, "fused": sd.p_fused}
            if "half_probs_cal" in table.arrays:
                systems["Lhalf"] = table.arrays["half_probs_cal"][sd.rows]
            if ref is not None:
                esc = escalate_at_operating_point(ref, results[ref][0], thresholds[ref][0], rates.get(ref, 0.0), sd)
                systems[f"cascade[{ref}]"] = np.where(esc[:, None], sd.p_fused, sd.p_cam)
            pc, conf = per_class_rows(systems, sd, labels, gcfg.bootstrap, gcfg.bootstrap_seed)
            out["per_class"] += pc
            out["confusion"] += conf
    if "val" in sds and "test" in sds:  # RQ1: la soglia di val applicata al test (solo con --final)
        out["threshold_transfer"] += threshold_transfer_rows(results, sds["val"], sds["test"], gcfg)
    out["positives"] = positives_rows(table, labels)
    out["importance"] = [
        {"model": n, "seed": gr.seed, "feature": f, "importance": v}
        for n, res in results.items()
        for gr in res
        for f, v in gr.importance.items()
    ]
    score_cols = {f"score__{n}__s{gr.seed}": gr.scores for n, res in results.items() for gr in res}
    scores = pd.concat(
        [table.df[["clip_id", "take_uid", "split", "fold", "label_idx", "c0", "c1", "y"]], pd.DataFrame(score_cols)],
        axis=1,
    )
    if not final:  # senza --final non esce nulla del test, nemmeno gli score
        scores = scores[scores.split != "test"].reset_index(drop=True)
    res_tables = {k: pd.DataFrame(v) for k, v in out.items()}
    res_tables["scores"] = scores
    res_tables["listed_models"] = list(gcfg.models)
    return res_tables


def write_outputs(res: Dict[str, object], out_dir: Path) -> List[Path]:
    from src.cascade.io import write_csv, write_parquet_df

    outs = []
    for key in OUTPUT_TABLES:
        write_csv(res[key], out_dir / f"{key}.csv")
        outs.append(out_dir / f"{key}.csv")
    write_parquet_df(res["scores"], out_dir / "scores.parquet")
    return outs + [out_dir / "scores.parquet"]


def needed_columns(schema_names: List[str], gcfg) -> List[str]:
    """Colonne della cache che servono ai gate (niente CLS/embedding video da 768 float)."""
    drop = {"vid_cls", "vid_emb", "vid_zs_logits"}
    if not any("emb" in make_gate(m, gcfg).groups for m in gcfg.models):
        drop.add("cam_emb")
    return [c for c in schema_names if c not in drop]
