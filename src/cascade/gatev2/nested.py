"""CV annidata del gate v2 per una partizione esterna (un "repeat").

Per ogni fold esterna k (take della fold = valutazione, il resto = train esterno):

1. **Livello 1, teste.** CV per sessione dentro il train esterno: la testa di CamFormer addestrata sulle altre
   fold interne predice la fold interna -> predizioni fuori campione ``P_in`` su tutto il train esterno. Poi le
   teste di CamFormer ed EgoVLP si rifanno su tutto il train esterno e predicono la fold k (``outer_l0 =
   inner_mean``: per CamFormer, media dei modelli interni invece del refit).
2. **Livello 2, gate.** Cross-fitting per sessione sul train esterno con ``P_in``: lo scorer addestrato sulle altre
   fold del gate dà score fuori campione ``s_cf``. Con ``s_cf`` si stimano il calibratore e le soglie delle regole
   ``risk``, ``acc_target`` e ``budget``. Lo scorer si rifà su tutto il train esterno.
3. **Livello 3, applicazione.** Score, rischio e decisioni sulla fold k con tutto congelato.

Il target del gate è e = 1 − c0 (CamFormer sbaglia): con EgoVLP sempre corretto coincide con il beneficio
atteso Δ(z) della regola ottima. La correttezza di EgoVLP si misura solo al livello 3, per valutare.
Le take senza embedding CamFormer non entrano nei fit e vanno sempre a EgoVLP.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from src.cascade.gatev2.calibrate import Calibrator
from src.cascade.gatev2.data import ScenarioData
from src.cascade.gatev2.folds import check_grouping, session_folds
from src.cascade.gatev2.heads import fit_head
from src.cascade.gatev2.rules import acc_target_threshold, budget_threshold, risk_threshold
from src.cascade.gatev2.scorers import make_scorer

log = logging.getLogger(__name__)


def _seed(*parts: int) -> int:
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0] % (2**31 - 1))


def outer_assignment(data: ScenarioData, gcfg, repeat: int) -> np.ndarray:
    fold = session_folds(data.y, data.sessions, gcfg.outer_folds, _seed(repeat, 1))
    check_grouping(fold, data.sessions)
    return fold


def run_fold(data: ScenarioData, gcfg, repeat: int, k: int, fold: np.ndarray) -> Dict[str, object]:
    """Livelli 1–3 per la fold esterna k. Restituisce righe per take, soglie, log delle teste e importanze."""
    y, sess, C = data.y, data.sessions, len(data.classes)
    te = np.flatnonzero(fold == k)
    tr = np.flatnonzero(fold != k)
    tr0 = tr[data.has_l0[tr]]
    te0 = te[data.has_l0[te]]
    heads_log: List[dict] = []

    # ---- livello 1: predizioni fuori campione di CamFormer sul train esterno
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

    # ---- teste esterne: CamFormer ed EgoVLP sul train esterno, predizioni sulla fold k
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

    # ---- livello 2: gate in cross-fitting sul train esterno
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
    for name in gcfg.scorers:
        s_cf = np.zeros(len(tr0))
        for g in range(gcfg.gate_folds):
            fit_m, held = gfold != g, gfold == g
            sc = make_scorer(name, gcfg, data.procedural).fit(
                P_in[fit_m], M_in[fit_m], y_in[fit_m], e_in[fit_m], s_in[fit_m], seed=_seed(repeat, k, 7, g)
            )
            s_cf[held] = sc.score(P_in[held], M_in[held])
        cal = Calibrator(gcfg.calibration, gcfg.calibration_boot, gcfg.delta, _seed(repeat, k, 8)).fit(s_cf, e_in, s_in)
        final = make_scorer(name, gcfg, data.procedural).fit(P_in, M_in, y_in, e_in, s_in, seed=_seed(repeat, k, 9))
        s_te = np.full(len(te), np.nan)
        r_hat = np.full(len(te), np.nan)
        r_up = np.full(len(te), np.nan)
        if len(te0):
            s_te[pos0] = final.score(P_out[pos0], data.motion[te0])
            r_hat[pos0], r_up[pos0] = cal.predict(s_te[pos0])
        rows[f"s__{name}"], rows[f"rhat__{name}"], rows[f"rup__{name}"] = s_te, r_hat, r_up
        base = {"repeat": repeat, "fold": k, "scorer": name, "n_cal": len(s_cf), "errors_cal": int(e_in.sum())}
        if gcfg.calibration == "platt":
            base.update(platt_a=cal.params[0], platt_b=cal.params[1])
        for b in gcfg.betas:
            thresholds += [{**base, "rule": "cost", "param": b, "threshold": b}]
            thresholds += [{**base, "rule": "cost_hat", "param": b, "threshold": b}]
        for a in gcfg.alphas:
            info = risk_threshold(s_cf, e_in, a, gcfg.delta, gcfg.risk_grid)
            thresholds += [{**base, "rule": "risk", "param": a, **info}]
        for m in gcfg.max_error_rates:
            thresholds += [{**base, "rule": "acc_target", "param": m, "threshold": acc_target_threshold(s_cf, e_in, m)}]
        for rho in gcfg.budgets:
            thresholds += [{**base, "rule": "budget", "param": rho, "threshold": budget_threshold(s_cf, rho)}]
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
    """Tutte le fold esterne di tutti i repeat (in parallelo con ``n_jobs``)."""
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
