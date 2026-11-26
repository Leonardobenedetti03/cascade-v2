"""Nested CV of the gate v2 for one outer partition (one "repeat").

For each outer fold k (the takes of the fold are evaluated, the rest is the outer train):

1. **Level 1, heads.** CV by session inside the outer train: the CamFormer head trained on the other inner
   folds predicts the inner fold, which gives out-of-sample predictions ``P_in`` on the whole outer train.
   Then the CamFormer and EgoVLP heads are refit on the whole outer train and predict fold k (with
   ``outer_l0 = inner_mean``, CamFormer uses the mean of the inner models instead of the refit).
2. **Level 2, gate.** Cross-fitting by session on the outer train with ``P_in``: the scorer trained on the
   other gate folds gives out-of-sample scores ``s_cf``. From ``s_cf`` we fit the calibrator and the thresholds
   of the ``risk``, ``acc_target`` and ``budget`` rules. Then the scorer is refit on the whole outer train.
3. **Level 3, application.** Score, risk and decisions on fold k, with everything frozen.

The gate target is e = 1 − c0 (CamFormer is wrong). If EgoVLP is always right, this is the expected benefit
Δ(z) of the optimal rule. EgoVLP's correctness is measured only at level 3, for the evaluation.
Takes without a CamFormer embedding never enter the fits and always go to EgoVLP.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from src.cascade.data.folds import check_grouping, session_folds
from src.cascade.data.gate_inputs import ScenarioData
from src.cascade.evaluation.rules import acc_target_threshold, budget_threshold, risk_threshold
from src.cascade.models.gates import make_scorer
from src.cascade.training.calibration import Calibrator
from src.cascade.training.nested_heads import fit_head

log = logging.getLogger(__name__)


def _seed(*parts: int) -> int:
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0] % (2**31 - 1))


def outer_assignment(data: ScenarioData, gcfg, repeat: int) -> np.ndarray:
    fold = session_folds(data.y, data.sessions, gcfg.outer_folds, _seed(repeat, 1))
    check_grouping(fold, data.sessions)
    return fold


def calibration_rows(data: ScenarioData, tr0: np.ndarray, e_in: np.ndarray, gcfg, repeat: int, k: int):
    """Mask over the rows ``tr0`` of the outer train used for the calibrator and the thresholds (N38), and the
    population actually used.

    It uses only train metadata (``camformer_pretrain``). The decision on the evaluated takes is the usual one,
    score -> risk -> rule, with no metadata."""
    pop = getattr(gcfg, "calibration_population", "all")
    if pop == "all":
        return np.ones(len(tr0), bool), "all"
    if pop != "pretrain_unseen":
        raise ValueError(f"gate_v2.calibration_population sconosciuto: {pop!r} (all | pretrain_unseen)")
    m = data.takes.camformer_pretrain.astype(str).str.lower().to_numpy()[tr0] != "train"
    n_err, n_sess = int(e_in[m].sum()), len(np.unique(data.sessions[tr0][m]))
    if n_err < int(getattr(gcfg, "calibration_min_errors", 10)) or n_sess < 2:
        log.warning(
            "repeat %d fold %d: %d errori in %d sessioni non viste dal pre-addestramento: calibro su tutte",
            repeat,
            k,
            n_err,
            n_sess,
        )
        return np.ones(len(tr0), bool), "all_fallback"
    return m, "pretrain_unseen"


def run_fold(data: ScenarioData, gcfg, repeat: int, k: int, fold: np.ndarray) -> Dict[str, object]:
    """Levels 1–3 for outer fold k. Returns per-take rows, thresholds, the head logs and the importances."""
    y, sess, C = data.y, data.sessions, len(data.classes)
    te = np.flatnonzero(fold == k)
    tr = np.flatnonzero(fold != k)
    tr0 = tr[data.has_l0[tr]]
    te0 = te[data.has_l0[te]]
    heads_log: List[dict] = []

    # Level 1: out-of-sample CamFormer predictions on the outer train
    inner = session_folds(y[tr0], sess[tr0], gcfg.inner_folds, _seed(repeat, k, 2))
    P_in = np.full((len(tr0), C), np.nan)
    inner_models = []
    for j in range(gcfg.inner_folds):
        fit_rows, held = tr0[inner != j], np.flatnonzero(inner == j)
        h = fit_head(data.X_cam[fit_rows], y[fit_rows], sess[fit_rows], C, gcfg.heads, _seed(repeat, k, 3, j))
        P_in[held] = h.predict_proba(data.X_cam[tr0[held]])
        inner_models.append(h)
        heads_log.append(
            {
                "repeat": repeat,
                "fold": k,
                "level": f"inner{j}",
                "model": "camformer",
                "C": h.C,
                "T": h.T,
                "n_fit": len(fit_rows),
            }
        )
    if not np.isfinite(P_in).all():
        raise RuntimeError("predizioni interne mancanti: fold interne incomplete")

    # Outer heads: CamFormer and EgoVLP fitted on the outer train, predictions on fold k
    P_out = np.full((len(te), C), np.nan)
    pos0 = np.isin(te, te0)
    if gcfg.outer_l0 == "refit":
        h0 = fit_head(data.X_cam[tr0], y[tr0], sess[tr0], C, gcfg.heads, _seed(repeat, k, 4))
        heads_log.append(
            {
                "repeat": repeat,
                "fold": k,
                "level": "outer",
                "model": "camformer",
                "C": h0.C,
                "T": h0.T,
                "n_fit": len(tr0),
            }
        )
        if len(te0):
            P_out[pos0] = h0.predict_proba(data.X_cam[te0])
    elif gcfg.outer_l0 == "inner_mean":
        if len(te0):
            P_out[pos0] = np.mean([h.predict_proba(data.X_cam[te0]) for h in inner_models], axis=0)
    else:
        raise ValueError(f"gate_v2.outer_l0 sconosciuto: {gcfg.outer_l0!r} (refit | inner_mean)")
    h1 = fit_head(data.X_vid[tr], y[tr], sess[tr], C, gcfg.heads, _seed(repeat, k, 5))
    heads_log.append(
        {"repeat": repeat, "fold": k, "level": "outer", "model": "egovlp", "C": h1.C, "T": h1.T, "n_fit": len(tr)}
    )
    V_out = h1.predict_proba(data.X_vid[te])

    # Level 2: the gate, cross-fitted on the outer train
    y_in, s_in, M_in = y[tr0], sess[tr0], data.motion[tr0]
    e_in = (P_in.argmax(1) != y_in).astype(int)
    gfold = session_folds(y_in, s_in, gcfg.gate_folds, _seed(repeat, k, 6))
    rows = {
        "take_name": data.takes.take_name.to_numpy()[te],
        "repeat": repeat,
        "fold": k,
        "has_l0": data.has_l0[te],
        "pred0": np.where(pos0, np.nan_to_num(P_out).argmax(1), -1),
        "p_max": np.where(pos0, np.nan_to_num(P_out).max(1), np.nan),
        "pred1": V_out.argmax(1),
    }
    thresholds: List[dict] = []
    importance: List[dict] = []
    cal_m, cal_pop = calibration_rows(data, tr0, e_in, gcfg, repeat, k)
    for name in gcfg.scorers:
        s_cf = np.zeros(len(tr0))
        for g in range(gcfg.gate_folds):
            fit_m, held = gfold != g, gfold == g
            sc = make_scorer(name, gcfg, data.procedural).fit(
                P_in[fit_m], M_in[fit_m], y_in[fit_m], e_in[fit_m], s_in[fit_m], seed=_seed(repeat, k, 7, g)
            )
            s_cf[held] = sc.score(P_in[held], M_in[held])
        s_cal, e_cal = s_cf[cal_m], e_in[cal_m]  # calibrator and thresholds: rows chosen by calibration_population
        cal = Calibrator(gcfg.calibration, gcfg.calibration_boot, gcfg.delta, _seed(repeat, k, 8))
        cal.fit(s_cal, e_cal, s_in[cal_m])
        final = make_scorer(name, gcfg, data.procedural).fit(P_in, M_in, y_in, e_in, s_in, seed=_seed(repeat, k, 9))
        s_te = np.full(len(te), np.nan)
        r_hat = np.full(len(te), np.nan)
        r_up = np.full(len(te), np.nan)
        if len(te0):
            s_te[pos0] = final.score(P_out[pos0], data.motion[te0])
            r_hat[pos0], r_up[pos0] = cal.predict(s_te[pos0])
        rows[f"s__{name}"], rows[f"rhat__{name}"], rows[f"rup__{name}"] = s_te, r_hat, r_up
        base = {
            "repeat": repeat,
            "fold": k,
            "scorer": name,
            "n_cal": len(s_cal),
            "errors_cal": int(e_cal.sum()),
            "cal_population": cal_pop,
        }
        if gcfg.calibration == "platt":
            base.update(platt_a=cal.params[0], platt_b=cal.params[1])
        for b in gcfg.betas:
            thresholds += [{**base, "rule": "cost", "param": b, "threshold": b}]
            thresholds += [{**base, "rule": "cost_hat", "param": b, "threshold": b}]
        for a in gcfg.alphas:
            info = risk_threshold(s_cal, e_cal, a, gcfg.delta, gcfg.risk_grid)
            thresholds += [{**base, "rule": "risk", "param": a, **info}]
        for m in gcfg.max_error_rates:
            thresholds += [
                {**base, "rule": "acc_target", "param": m, "threshold": acc_target_threshold(s_cal, e_cal, m)}
            ]
        for rho in gcfg.budgets:
            thresholds += [{**base, "rule": "budget", "param": rho, "threshold": budget_threshold(s_cal, rho)}]
        importance += [
            {"repeat": repeat, "fold": k, "scorer": name, "feature": f, "importance": v}
            for f, v in final.importance().items()
        ]
    log.info(
        "repeat %d fold %d: %d take valutate, %d nel train esterno (%d errori di CamFormer fuori campione)",
        repeat,
        k,
        len(te),
        len(tr0),
        int(e_in.sum()),
    )
    return {"takes": pd.DataFrame(rows), "thresholds": thresholds, "heads": heads_log, "importance": importance}


def run_all(data: ScenarioData, gcfg, n_jobs: int = 1) -> Dict[str, pd.DataFrame]:
    """All the outer folds of all repeats (in parallel with ``n_jobs``)."""
    from joblib import Parallel, delayed

    jobs = []
    folds = {}
    for r in gcfg.repeats:
        folds[r] = outer_assignment(data, gcfg, r)
        jobs += [(r, k) for k in range(gcfg.outer_folds)]
    res = Parallel(n_jobs=n_jobs)(delayed(run_fold)(data, gcfg, r, k, folds[r]) for r, k in jobs)
    takes = pd.concat([x["takes"] for x in res], ignore_index=True)
    n_rep = takes.groupby("repeat").take_name.nunique()
    if (n_rep != len(data)).any():
        raise RuntimeError(f"take valutate per repeat {n_rep.to_dict()} != {len(data)}")
    return {
        "takes": takes,
        "thresholds": pd.DataFrame([t for x in res for t in x["thresholds"]]),
        "heads": pd.DataFrame([h for x in res for h in x["heads"]]),
        "importance": pd.DataFrame([i for x in res for i in x["importance"]]),
        "outer_folds": pd.DataFrame(
            {
                "take_name": np.tile(data.takes.take_name.to_numpy(), len(folds)),
                "repeat": np.repeat(list(folds), len(data)),
                "fold": np.concatenate([folds[r] for r in folds]),
            }
        ),
    }
