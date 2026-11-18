"""Valutazione del gate v2 sulle decisioni fuori campione della CV annidata.

Ogni take è valutata una volta per repeat, da modelli che non hanno visto la sua sessione. Le quantità per take
si mediano sui repeat (quota dei repeat che scalano, correttezza media); gli intervalli ricampionano le SESSIONI
(take della stessa sessione non sono indipendenti) e i confronti tra scorer sono appaiati (stessi ricampionamenti),
con correzione di Holm sulle coppie pre-registrate.

Sistemi valutati: riferimenti ``always_L0`` (solo le take senza CamFormer vanno a EgoVLP), ``always_L1``,
``oracle`` (scala esattamente dove EgoVLP corregge CamFormer) e ogni combinazione scorer × regola × parametro.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.cascade.eval.curves import group_resample_weights
from src.cascade.gatev2.calibrate import ece
from src.cascade.gatev2.data import ScenarioData, subset_masks
from src.cascade.gatev2.rules import escalate

log = logging.getLogger(__name__)
RULES_ORDER = ("cost", "cost_hat", "risk", "acc_target", "budget")


# --------------------------------------------------------------------------- matrici [R, N]
class Outcomes:
    """Uscite fuori campione per repeat e take, nell'ordine di ``data.takes``."""

    def __init__(self, data: ScenarioData, takes: pd.DataFrame, scorers: List[str]) -> None:
        self.data = data
        self.repeats = sorted(takes.repeat.unique())
        pos = pd.Series(np.arange(len(data)), index=data.takes.take_name.to_numpy())
        R, N = len(self.repeats), len(data)
        self.fold = np.full((R, N), -1, int)
        self.has_l0 = np.zeros((R, N), bool)
        self.pred0 = np.full((R, N), -1, int)
        self.pred1 = np.full((R, N), -1, int)
        self.s, self.rhat, self.rup = {}, {}, {}
        for n in scorers:
            self.s[n], self.rhat[n], self.rup[n] = (np.full((R, N), np.nan) for _ in range(3))
        for i, r in enumerate(self.repeats):
            t = takes[takes.repeat == r]
            j = pos[t.take_name.to_numpy()].to_numpy()
            self.fold[i, j] = t.fold.to_numpy()
            self.has_l0[i, j] = t.has_l0.to_numpy().astype(bool)
            self.pred0[i, j] = t.pred0.to_numpy().astype(int)
            self.pred1[i, j] = t.pred1.to_numpy().astype(int)
            for n in scorers:
                self.s[n][i, j] = t[f"s__{n}"].to_numpy()
                self.rhat[n][i, j] = t[f"rhat__{n}"].to_numpy()
                self.rup[n][i, j] = t[f"rup__{n}"].to_numpy()
        y = data.y[None]
        self.c0 = (self.pred0 == y) & self.has_l0
        self.c1 = self.pred1 == y

    def decisions(self, scorer: str, rule: str, param: float, thresholds: pd.DataFrame) -> np.ndarray:
        th = thresholds[(thresholds.scorer == scorer) & (thresholds.rule == rule) & np.isclose(thresholds.param, param)]
        T = np.full(self.fold.shape, np.nan)
        for i, r in enumerate(self.repeats):
            m = dict(zip(th[th.repeat == r].fold, th[th.repeat == r].threshold))
            T[i] = [m.get(int(f), np.nan) for f in self.fold[i]]
        if np.isnan(T).any():
            raise RuntimeError(f"soglie mancanti per {scorer}/{rule}/{param}")
        esc = np.zeros(self.fold.shape, bool)
        for i in range(len(self.repeats)):
            esc[i] = escalate(rule, param, self.s[scorer][i], self.rhat[scorer][i], self.rup[scorer][i], T[i])
        return esc | ~self.has_l0

    def reference(self, name: str) -> np.ndarray:
        if name == "always_L0":
            return ~self.has_l0
        if name == "always_L1":
            return np.ones_like(self.has_l0)
        if name == "oracle":
            return ~self.has_l0 | (self.c1 & ~self.c0)
        raise ValueError(name)


def systems(out: Outcomes, thresholds: pd.DataFrame, gcfg) -> Dict[Tuple[str, str, float], np.ndarray]:
    sys: Dict[Tuple[str, str, float], np.ndarray] = {}
    for ref in ("always_L0", "always_L1", "oracle"):
        sys[(ref, "fixed", 0.0)] = out.reference(ref)
    if "class_rule" in out.s:  # decisione naturale della regola per classe: scala le procedurali predette
        sys[("class_rule", "natural", 0.0)] = (np.nan_to_num(out.s["class_rule"]) > 0.5) | ~out.has_l0
    keys = thresholds[["scorer", "rule", "param"]].drop_duplicates()
    keys = keys.assign(o=keys.rule.map({r: i for i, r in enumerate(RULES_ORDER)})).sort_values(["scorer", "o", "param"])
    for r in keys.itertuples(index=False):
        sys[(r.scorer, r.rule, float(r.param))] = out.decisions(r.scorer, r.rule, float(r.param), thresholds)
    return sys


# --------------------------------------------------------------------------- metriche puntuali e bootstrap
def _per_take(out: Outcomes, esc: np.ndarray) -> Dict[str, np.ndarray]:
    corr = np.where(esc, out.c1, out.c0)
    return {
        "esc": esc.mean(0),
        "corr": corr.mean(0),
        "err_kept": (~esc & ~out.c0).mean(0),
        "corr_kept": (~esc & out.c0).mean(0),
        "corr_r": corr,
        "esc_r": esc,
    }


def _ci(v: np.ndarray) -> Tuple[float, float]:
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)


def _p_two_sided(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    if not len(d):
        return float("nan")
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def cost_frac(rho: float, gm: Dict[str, float], extra0: float = 0.0, extra1: float = 0.0) -> float:
    c0, c1 = gm["L0"] + extra0, gm["L1"] + extra1
    return float((c0 + rho * c1) / (c0 + c1))


def metric_rows(out: Outcomes, sys, masks, gcfg) -> Tuple[List[dict], Dict[str, np.ndarray]]:
    rows = []
    W = {}
    y = out.data.y
    ref_l1 = _per_take(out, out.reference("always_L1"))
    for sub, m in masks.items():
        if not m.any():
            continue
        W[sub] = group_resample_weights(out.data.sessions[m], gcfg.bootstrap, gcfg.bootstrap_seed)
        w = W[sub]
        classes = [c for c in range(len(out.data.classes)) if (y[m] == c).any()]
        for (name, rule, param), esc in sys.items():
            pt = _per_take(out, esc)
            corr, e, ek = pt["corr"][m], pt["esc"][m], pt["err_kept"][m]
            kept = (1 - e).sum()
            acc_b = w @ corr / w.sum(1)
            rho_b = w @ e / w.sum(1)
            diff_b = acc_b - w @ ref_l1["corr"][m] / w.sum(1)
            sel_b = (w @ ek) / np.maximum(w @ (1 - e), 1e-12)
            acc_rep = pt["corr_r"][:, m].mean(1)
            rho_rep = pt["esc_r"][:, m].mean(1)
            rows.append(
                {
                    "subset": sub,
                    "system": name,
                    "rule": rule,
                    "param": param,
                    "n_takes": int(m.sum()),
                    "n_sessions": int(len(np.unique(out.data.sessions[m]))),
                    "acc": float(corr.mean()),
                    "acc_ci_low": _ci(acc_b)[0],
                    "acc_ci_high": _ci(acc_b)[1],
                    "acc_sd_repeats": float(acc_rep.std()),
                    "bacc": float(np.mean([corr[y[m] == c].mean() for c in classes])),
                    "rho": float(e.mean()),
                    "rho_ci_low": _ci(rho_b)[0],
                    "rho_ci_high": _ci(rho_b)[1],
                    "rho_sd_repeats": float(rho_rep.std()),
                    "camera_off": float(1 - e.mean()),
                    "cost_frac_L1": cost_frac(float(e.mean()), gcfg.gmacs_per_window),
                    "errors_left": float(ek.sum()),
                    "selective_error": float(ek.sum() / kept) if kept > 0 else float("nan"),
                    "selective_error_ci_high": _ci(sel_b)[1],
                    "acc_minus_L1": float(corr.mean() - ref_l1["corr"][m].mean()),
                    "acc_minus_L1_ci_low": _ci(diff_b)[0],
                    "acc_minus_L1_ci_high": _ci(diff_b)[1],
                    "acc_L0_kept": float((pt["corr_kept"][m]).sum() / kept) if kept > 0 else float("nan"),
                }
            )
    return rows, W


# --------------------------------------------------------------------------- curve, AUROC, confronti
def _weighted_auroc(score: np.ndarray, label: np.ndarray, w: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    ok = np.isfinite(score) & (w > 0)
    lab, sc, ww = label[ok], score[ok], w[ok]
    if len(np.unique(lab)) < 2:
        return float("nan")
    return float(roc_auc_score(lab, sc, sample_weight=ww))


def curve_rows(out: Outcomes, masks, gcfg, step: int = 1) -> List[dict]:
    """Accuratezza contro quota scalata, ordinando le take per rischio calibrato (confrontabile tra fold)."""
    rows = []
    grid = np.arange(0, 101, step)
    for sub, m in masks.items():
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        for name in gcfg.scorers:
            accs = np.zeros((len(out.repeats), len(grid)))
            for i in range(len(out.repeats)):
                r = out.rhat[name][i, idx]
                forced = ~out.has_l0[i, idx]
                key = np.where(forced, np.inf, np.nan_to_num(r, nan=np.inf))
                order = np.argsort(-key, kind="stable")
                c0, c1 = out.c0[i, idx][order], out.c1[i, idx][order]
                n = len(idx)
                for gi, g in enumerate(grid):
                    k = max(int(round(g / 100 * n)), int(forced.sum()))
                    accs[i, gi] = (c1[:k].sum() + c0[k:].sum()) / n
            a = accs.mean(0)
            rows += [
                {
                    "subset": sub,
                    "scorer": name,
                    "r": int(g),
                    "acc": float(v),
                    "cost_frac_L1": cost_frac(g / 100, gcfg.gmacs_per_window),
                }
                for g, v in zip(grid, a)
            ]
    return rows


def auc_and_comparisons(out: Outcomes, sys, masks, W, gcfg) -> Tuple[List[dict], List[dict]]:
    auc_rows, comp_rows = [], []
    boots: Dict[Tuple[str, str], np.ndarray] = {}
    for sub, m in masks.items():
        if sub not in W:
            continue
        w = W[sub]
        idx = np.flatnonzero(m)
        for name in gcfg.scorers:
            pts, bs = [], np.zeros((len(out.repeats), w.shape[0]))
            for i in range(len(out.repeats)):
                h = out.has_l0[i, idx]
                sc, lab = out.rhat[name][i, idx], (~out.c0[i, idx]).astype(int)
                pts.append(_weighted_auroc(np.where(h, sc, np.nan), lab, np.ones(len(idx))))
                for b in range(w.shape[0]):
                    bs[i, b] = _weighted_auroc(np.where(h, sc, np.nan), lab, w[b])
            boots[(sub, name)] = np.nanmean(bs, 0)
            lo, hi = _ci(boots[(sub, name)])
            auc_rows.append(
                {
                    "subset": sub,
                    "scorer": name,
                    "auroc": float(np.nanmean(pts)),
                    "auroc_ci_low": lo,
                    "auroc_ci_high": hi,
                    "auroc_sd_repeats": float(np.nanstd(pts)),
                }
            )
        for a, b in gcfg.comparisons:
            if (sub, a) not in boots or (sub, b) not in boots:
                continue
            d = boots[(sub, a)] - boots[(sub, b)]
            comp_rows.append(_comp(sub, a, b, "auroc", np.nanmean(boots[(sub, a)]) - np.nanmean(boots[(sub, b)]), d))
            ea, eb = _decision(sys, a, gcfg), _decision(sys, b, gcfg)
            if ea is not None and eb is not None:
                pa, pb = _per_take(out, ea), _per_take(out, eb)
                for metric, key in (("acc_at_primary", "corr"), ("rho_at_primary", "esc")):
                    d = (w @ pa[key][m] - w @ pb[key][m]) / w.sum(1)
                    comp_rows.append(_comp(sub, a, b, metric, float(pa[key][m].mean() - pb[key][m].mean()), d))
    return auc_rows, holm(comp_rows)


def _decision(sys, name: str, gcfg):
    """Decisione di uno scorer al punto primario; la regola per classe usa la sua decisione naturale."""
    if name == "class_rule":
        return sys.get(("class_rule", "natural", 0.0))
    return sys.get((name, gcfg.primary_rule, float(gcfg.primary_param)))


def _comp(sub, a, b, metric, diff, d) -> dict:
    lo, hi = _ci(d)
    return {
        "subset": sub,
        "a": a,
        "b": b,
        "metric": metric,
        "diff": float(diff),
        "ci_low": lo,
        "ci_high": hi,
        "p_two_sided": _p_two_sided(d),
    }


def holm(rows: List[dict]) -> List[dict]:
    """p di Holm per (sottoinsieme, metrica) sulle coppie pre-registrate."""
    df = pd.DataFrame(rows)
    if df.empty:
        return rows
    df["p_holm"] = np.nan
    for _, g in df.groupby(["subset", "metric"]):
        order = g.p_two_sided.sort_values().index
        m, run = len(order), 0.0
        for j, ix in enumerate(order):
            run = max(run, min(1.0, (m - j) * df.loc[ix, "p_two_sided"]))
            df.loc[ix, "p_holm"] = run
    return df.to_dict("records")


# --------------------------------------------------------------------------- calibrazione, procedurali, costi
def calibration_rows(out: Outcomes, masks, gcfg, n_bins: int = 10) -> List[dict]:
    rows = []
    for sub, m in masks.items():
        if not m.any():
            continue
        for name in gcfg.scorers:
            r = out.rhat[name][:, m].ravel()
            e = (~out.c0[:, m]).ravel().astype(float)
            ok = out.has_l0[:, m].ravel() & np.isfinite(r)
            r, e = r[ok], e[ok]
            order = np.argsort(r, kind="stable")
            for bi, b in enumerate(np.array_split(order, n_bins)):
                if len(b):
                    rows.append(
                        {
                            "subset": sub,
                            "scorer": name,
                            "bin": bi,
                            "r_mean": float(r[b].mean()),
                            "error_rate": float(e[b].mean()),
                            "n": int(len(b)),
                        }
                    )
            rows.append(
                {
                    "subset": sub,
                    "scorer": name,
                    "bin": -1,
                    "r_mean": float(r.mean()),
                    "error_rate": float(e.mean()),
                    "n": int(len(r)),
                    "ece": ece(r, e, n_bins),
                }
            )
    return rows


def procedural_rows(out: Outcomes, sys, masks, gcfg) -> List[dict]:
    """Take che CamFormer predice procedurali: quante restano su CamFormer e quante di queste sono corrette
    (chiamate a EgoVLP risparmiate rispetto alla regola per classe, che le scala tutte)."""
    rows = []
    proc = out.data.procedural
    pred_proc = out.has_l0 & (out.pred0 >= 0) & proc[np.clip(out.pred0, 0, None)]
    for sub, m in masks.items():
        if not m.any():
            continue
        pp = pred_proc[:, m]
        n = pp.sum(1)
        for (name, rule, param), esc in sys.items():
            kept = pp & ~esc[:, m]
            rows.append(
                {
                    "subset": sub,
                    "system": name,
                    "rule": rule,
                    "param": param,
                    "n_pred_procedural": float(n.mean()),
                    "kept_on_L0": float(kept.sum(1).mean()),
                    "kept_correct": float((kept & out.c0[:, m]).sum(1).mean()),
                    "kept_errors": float((kept & ~out.c0[:, m]).sum(1).mean()),
                    "frac_kept": float((kept.sum(1) / np.maximum(n, 1)).mean()),
                }
            )
    return rows


def cost_sensitivity_rows(metrics: pd.DataFrame, gcfg) -> List[dict]:
    rows = []
    for r in metrics.itertuples(index=False):
        for x0 in gcfg.extra_gmacs_L0:
            for x1 in gcfg.extra_gmacs_L1:
                rows.append(
                    {
                        "subset": r.subset,
                        "system": r.system,
                        "rule": r.rule,
                        "param": r.param,
                        "extra_gmacs_L0": x0,
                        "extra_gmacs_L1": x1,
                        "rho": r.rho,
                        "cost_frac_L1": cost_frac(r.rho, gcfg.gmacs_per_window, x0, x1),
                    }
                )
    return rows


def cost_vs_egovlp_rows(metrics: pd.DataFrame, gcfg, n_windows: int) -> List[dict]:
    """Costo della cascata contro "solo EgoVLP su tutte le take" (senza CamFormer davanti), per ogni β della
    regola a costo e per i riferimenti. GMACs per take = finestre × (C_L0 + ρ · C_L1) per la cascata e
    finestre × C_L1 per solo EgoVLP; l'intervallo viene da quello di ρ (bootstrap per sessione)."""
    gm = gcfg.gmacs_per_window
    l0, l1 = n_windows * gm["L0"], n_windows * gm["L1"]
    sel = metrics[metrics.rule.isin(["cost", "cost_hat", "fixed"]) | (metrics.system == "class_rule")]
    rows = []
    for r in sel.itertuples(index=False):
        if r.system == "always_L1":  # "solo EgoVLP": nessun CamFormer davanti
            casc, lo, hi = l1, l1, l1
        else:
            casc, lo, hi = l0 + r.rho * l1, l0 + r.rho_ci_low * l1, l0 + r.rho_ci_high * l1
        rows.append(
            {
                "subset": r.subset,
                "system": r.system,
                "rule": r.rule,
                "param": r.param,
                "primary_beta": bool(r.rule == gcfg.primary_rule and np.isclose(r.param, gcfg.primary_param)),
                "acc": r.acc,
                "acc_minus_egovlp": r.acc_minus_L1,
                "rho": r.rho,
                "gmacs_per_take_cascade": casc,
                "gmacs_per_take_cascade_ci_low": lo,
                "gmacs_per_take_cascade_ci_high": hi,
                "gmacs_per_take_egovlp_only": l1,
                "cost_frac_of_egovlp_only": casc / l1,
                "cost_frac_ci_low": lo / l1,
                "cost_frac_ci_high": hi / l1,
                "saving_frac": 1 - casc / l1,
                "speedup": l1 / casc,
                "gmacs_total_cascade": casc * r.n_takes,
                "gmacs_total_egovlp_only": l1 * r.n_takes,
                "n_takes": r.n_takes,
            }
        )
    return rows


def model_rows(out: Outcomes, masks) -> List[dict]:
    """Accuratezza di CamFormer ed EgoVLP fuori campione per sottoinsieme (media sui repeat)."""
    rows = []
    for sub, m in masks.items():
        if not m.any():
            continue
        h = out.has_l0[:, m]
        rows.append(
            {
                "subset": sub,
                "n_takes": int(m.sum()),
                "acc_L0": float((out.c0[:, m].sum(1) / np.maximum(h.sum(1), 1)).mean()),
                "errors_L0": float((h & ~out.c0[:, m]).sum(1).mean()),
                "acc_L1": float(out.c1[:, m].mean()),
                "errors_L1": float((~out.c1[:, m]).sum(1).mean()),
                "n_no_l0": int((~h[0]).sum()),
            }
        )
    return rows


def evaluate(data: ScenarioData, res: Dict[str, pd.DataFrame], gcfg) -> Dict[str, pd.DataFrame]:
    out = Outcomes(data, res["takes"], list(gcfg.scorers))
    masks = subset_masks(data.takes, list(gcfg.subsets))
    sys = systems(out, res["thresholds"], gcfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mrows, W = metric_rows(out, sys, masks, gcfg)
        metrics = pd.DataFrame(mrows)
        auc, comp = auc_and_comparisons(out, sys, masks, W, gcfg)
        curves = pd.DataFrame(curve_rows(out, masks, gcfg))
        area = curves.groupby(["subset", "scorer"], as_index=False).acc.mean().rename(columns={"acc": "area"})
        tables = {
            "metrics": metrics,
            "models": pd.DataFrame(model_rows(out, masks)),
            "curves": curves,
            "auc": pd.DataFrame(auc).merge(area, on=["subset", "scorer"], how="left"),
            "comparisons": pd.DataFrame(comp),
            "calibration": pd.DataFrame(calibration_rows(out, masks, gcfg)),
            "procedural": pd.DataFrame(procedural_rows(out, sys, masks, gcfg)),
            "cost_sensitivity": pd.DataFrame(cost_sensitivity_rows(metrics, gcfg)),
            "cost_vs_egovlp": pd.DataFrame(cost_vs_egovlp_rows(metrics, gcfg, data.n_windows)),
        }
    return tables


def primary_result(tables: Dict[str, pd.DataFrame], gcfg) -> Dict[str, object]:
    m = tables["metrics"]
    sel = m[
        (m.subset == gcfg.primary_subset)
        & (m.system == gcfg.primary_scorer)
        & (m.rule == gcfg.primary_rule)
        & np.isclose(m.param, gcfg.primary_param)
    ]
    mod = tables["models"]
    cv = tables.get("cost_vs_egovlp")
    cost_primary = {}
    if cv is not None and len(cv):
        c = cv[(cv.subset == gcfg.primary_subset) & (cv.system == gcfg.primary_scorer) & cv.primary_beta]
        cost_primary = c.iloc[0].to_dict() if len(c) else {"missing": True}
    return {
        "cost_vs_egovlp_primary": cost_primary,
        "primary": {
            "subset": gcfg.primary_subset,
            "scorer": gcfg.primary_scorer,
            "rule": gcfg.primary_rule,
            "param": gcfg.primary_param,
            **(sel.iloc[0].to_dict() if len(sel) else {"missing": True}),
        },
        "models": mod.to_dict("records"),
        "l1_always_correct": bool((mod.errors_L1 == 0).all()) if len(mod) else None,
    }
