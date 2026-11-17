"""53 — Analisi del gate su scenario_preds decise prima della run (N31, N32) (CPU).

Scopo, al punto di lavoro di ogni gate (decisioni ``esc__<gate>`` di 50/scores.parquet):
    1. ``op_all.csv``: accuratezza, accuratezza bilanciata e costo **con le take senza L0**
       (``45_cache/.../no_l0.csv``: vanno a EgoVLP e costano L0 + L1, N28);
    2. ``op_no_shared.csv``: lo stesso senza le sessioni che compaiono anche nel train (N31);
    3. ``op_boot.csv``: intervalli bootstrap 95% ricampionando le SESSIONI, anche della differenza
       di accuratezza da always_L1 (take della stessa sessione non sono indipendenti);
    4. regola fissa ``rule_procedural``: scala se la classe prevista da L0 è procedurale (confronto, N31);
    5. ``by_pretrain.csv``: risultati separati per ``camformer_pretrain``;
    6. ``motion_l0.csv`` e ``motion_l0.png``: accuratezza di L0 sul train fuori campione per
       quintili di ``mot_energy`` (soglie dei quintili dal train), in totale e per classe.
    Il test entra solo con ``--final`` (richiede 50 --final). Si rilancia sempre: costa pochi secondi.

Input:
    45_cache/<variante>/cache.parquet, no_l0.csv; 50_gate/<variante>/<tag>/scores.parquet

Output:
    <run_dir>/53_analysis/<variante>/<tag>/*.csv, motion_l0.png, summary.json
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.cascade.cli import open_stage, stage_parser  # noqa: E402
from src.cascade.io import write_json  # noqa: E402
from src.data.cache import read_table  # noqa: E402

STAGE = "53_analysis"
RULE = "rule_procedural"
MOTION_FEATURE = "mot_energy"
N_QUANTILES = 5


def bacc(correct: np.ndarray, label: np.ndarray) -> float:
    rec = [correct[label == c].mean() for c in np.unique(label)]
    return float(np.mean(rec)) if rec else float("nan")


def take_frame(scores: pd.DataFrame, cache, no_l0: pd.DataFrame, procedural: np.ndarray) -> pd.DataFrame:
    """Una riga per take e una colonna ``esc__<gate>`` (quota dei seed che scalano, in [0, 1]).

    Le take senza L0 hanno ``esc = 1`` per ogni gate, ``c0 = 0`` e costo L0 + L1."""
    df = scores.copy()
    pred0 = pd.Series(cache.arrays["cam_probs_cal"].argmax(1), index=cache.df.clip_id.to_numpy())
    df[f"esc__{RULE}"] = procedural[pred0.loc[df.clip_id].to_numpy()].astype(float)
    df["has_l0"] = True
    if len(no_l0):
        extra = pd.DataFrame(
            {
                "clip_id": no_l0.take_name.to_numpy(),
                "take_uid": no_l0.take_uid.astype(str).to_numpy(),
                "split": no_l0.split.to_numpy(),
                "label_idx": no_l0.label_idx.to_numpy(),
                "c0": 0,
                "c1": no_l0.c1.to_numpy(),
                "gmacs_L0": float(df.gmacs_L0.iloc[0]),
                "gmacs_L1": float(df.gmacs_L1.iloc[0]),
                "has_l0": False,
            }
        )
        for c in ("session", "camformer_pretrain"):
            if c in no_l0.columns:
                extra[c] = no_l0[c].to_numpy()
        for c in [c for c in df.columns if c.startswith("esc__")]:
            extra[c] = 1.0
        df = pd.concat([df, extra], ignore_index=True)
    return df


def gates(df: pd.DataFrame) -> List[str]:
    return [c[len("esc__") :] for c in df.columns if c.startswith("esc__")]


def op_rows(d: pd.DataFrame, split: str, subset: str) -> List[dict]:
    """Punto di lavoro per gate: correttezza attesa (1 − e)·c0 + e·c1, costo L0 + e·L1."""
    rows: List[dict] = []
    if d.empty:
        return rows
    y = d.label_idx.to_numpy()
    ref_cost = float((d.gmacs_L0 + d.gmacs_L1).mean())
    for g in gates(d):
        e = d[f"esc__{g}"].to_numpy(float)
        corr = (1 - e) * d.c0.to_numpy() + e * d.c1.to_numpy()
        cost = d.gmacs_L0.to_numpy() + e * d.gmacs_L1.to_numpy()
        rows.append(
            {
                "split": split,
                "subset": subset,
                "model": g,
                "n_takes": int(len(d)),
                "n_no_l0": int((~d.has_l0.astype(bool)).sum()),
                "acc": float(corr.mean()),
                "bacc": bacc(corr, y),
                "cost_frac_L1": float(cost.mean()) / ref_cost,
                "escalation_rate": float(e.mean()),
            }
        )
    return rows


def boot_rows(d: pd.DataFrame, split: str, n_boot: int, seed: int) -> List[dict]:
    """Bootstrap per sessione di accuratezza, costo e accuratezza − accuratezza di always_L1."""
    if d.empty or "session" not in d.columns:
        return []
    sess = d.session.astype(str).to_numpy()
    uniq = np.unique(sess)
    idx_of = {s: np.flatnonzero(sess == s) for s in uniq}
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([idx_of[s] for s in rng.choice(uniq, len(uniq))]) for _ in range(n_boot)]
    c0, c1 = d.c0.to_numpy(float), d.c1.to_numpy(float)
    L0, L1 = d.gmacs_L0.to_numpy(float), d.gmacs_L1.to_numpy(float)
    rows = []
    for g in gates(d):
        e = d[f"esc__{g}"].to_numpy(float)
        corr, cost = (1 - e) * c0 + e * c1, L0 + e * L1
        acc = np.array([corr[i].mean() for i in draws])
        frac = np.array([cost[i].mean() / (L0[i] + L1[i]).mean() for i in draws])
        diff = np.array([corr[i].mean() - c1[i].mean() for i in draws])
        for name, v in (("acc", acc), ("cost_frac_L1", frac), ("acc_minus_always_L1", diff)):
            rows.append(
                {
                    "split": split,
                    "model": g,
                    "metric": name,
                    "lo": float(np.percentile(v, 2.5)),
                    "hi": float(np.percentile(v, 97.5)),
                    "n_sessions": int(len(uniq)),
                }
            )
    return rows


def motion_rows(cache, names) -> pd.DataFrame:
    """Accuratezza di L0 sul train fuori campione per quintili di moto (soglie dal train)."""
    df = cache.df
    x_all = df[MOTION_FEATURE].to_numpy(float)
    tr = (df.split == "train").to_numpy() & np.isfinite(x_all)
    x, c0, lab = x_all[tr], df.c0.to_numpy()[tr], df.label_idx.to_numpy()[tr]
    edges = np.quantile(x, np.linspace(0, 1, N_QUANTILES + 1))
    q = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, N_QUANTILES - 1)
    rows = []
    for k in range(N_QUANTILES):
        m = q == k
        for cls, mc in [("all", m)] + [(name, m & (lab == c)) for c, name in enumerate(names)]:
            rows.append(
                {
                    "class": cls,
                    "quintile": k + 1,
                    "lo": float(edges[k]),
                    "hi": float(edges[k + 1]),
                    "n": int(mc.sum()),
                    "acc_L0": float(c0[mc].mean()) if mc.any() else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def plot_motion(mot: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for cls, g in mot.groupby("class", sort=False):
        g = g[g.n >= 5]
        ax.plot(g.quintile, g.acc_L0, marker="o", lw=2.5 if cls == "all" else 1.0, label=cls)
    ax.set_xlabel(f"quintile di {MOTION_FEATURE} (train, fuori campione)")
    ax.set_ylabel("accuratezza di CamFormer (L0)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary(df: pd.DataFrame, splits: List[str], train_sessions: set, no_l0: pd.DataFrame) -> dict:
    out = {"splits": splits, "motion_feature": MOTION_FEATURE, "n_no_l0": {}}
    out["n_shared_sessions"], out["n_takes_shared"] = {}, {}
    for s in splits:
        d = df[df.split == s]
        out["n_no_l0"][s] = int((no_l0.split == s).sum())
        if train_sessions and "session" in d.columns:
            shared = d.session.astype(str).isin(train_sessions)
            out["n_shared_sessions"][s] = int(d.session[shared].nunique())
            out["n_takes_shared"][s] = int(shared.sum())
    return out


def main(argv: Optional[List[str]] = None) -> None:
    p = stage_parser(__doc__)
    p.add_argument("--tag", type=str, default="main")
    p.add_argument("--final", action="store_true", help="include il test (richiede 50 --final)")
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    ctx = open_stage(args, STAGE, make_stage_dir=False)
    ctx.require("45_cache", parts=[ctx.variant_tag])
    cache_dir = ctx.variant_dir("45_cache")
    gate_dir = ctx.variant_dir("50_gate") / args.tag
    out_dir = ctx.variant_dir(STAGE) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.final and not (gate_dir / "FINAL_DONE").exists():
        raise RuntimeError("--final richiede la valutazione finale dello stadio 50 (FINAL_DONE)")
    cache = read_table(cache_dir / "cache.parquet")
    names = list(ctx.labels.names)
    scores = pd.read_parquet(gate_dir / "scores.parquet")
    if not any(c.startswith("esc__") for c in scores.columns):
        raise RuntimeError(f"{gate_dir}/scores.parquet senza colonne esc__: rifare lo stadio 50 al commit attuale")
    splits = ["val"] + (["test"] if args.final else [])
    scores = scores[scores.split.isin(splits + ["train"])].reset_index(drop=True)
    no_l0 = pd.read_csv(cache_dir / "no_l0.csv", dtype={"take_uid": str})
    no_l0 = no_l0[no_l0.split.isin(splits)]
    if len(no_l0):  # sessione e pretraining delle take senza L0, dall'indice di scenario_preds
        idx = pd.read_csv(Path(cache.meta["scenario_run"]) / "index" / "takes.csv")
        cols = [c for c in ("session", "camformer_pretrain") if c in idx.columns]
        no_l0 = no_l0.merge(idx[["take_name", *cols]], on="take_name", how="left")
    df = take_frame(scores, cache, no_l0, ctx.labels.procedural_mask)
    train_sessions = set(df.session[df.split == "train"].astype(str)) if "session" in df.columns else set()
    op, nosh, boot, pre = [], [], [], []
    for s in splits:
        d = df[df.split == s]
        op += op_rows(d, s, "all")
        if train_sessions:
            nosh += op_rows(d[~d.session.astype(str).isin(train_sessions)], s, "no_shared_sessions")
        boot += boot_rows(d, s, args.n_boot, ctx.cfg.gate.bootstrap_seed)
        if "camformer_pretrain" in d.columns:
            for v, g in d.groupby(d.camformer_pretrain.astype(str)):
                pre += op_rows(g, s, f"camformer_pretrain={v}")
    tables: Dict[str, pd.DataFrame] = {
        "op_all": pd.DataFrame(op),
        "op_no_shared": pd.DataFrame(nosh),
        "op_boot": pd.DataFrame(boot),
        "by_pretrain": pd.DataFrame(pre),
        "motion_l0": motion_rows(cache, names),
    }
    for k, t in tables.items():
        if not args.final and "split" in t.columns and (t.split == "test").any():
            raise RuntimeError(f"{k}: righe del test senza --final")
        t.to_csv(out_dir / f"{k}.csv", index=False)
    plot_motion(tables["motion_l0"], out_dir / "motion_l0.png")
    write_json(out_dir / "summary.json", summary(df, splits, train_sessions, no_l0))
    outs = [out_dir / f"{k}.csv" for k in tables] + [out_dir / "motion_l0.png", out_dir / "summary.json"]
    ctx.finish(outs, part=f"{ctx.variant_tag}.{args.tag}" + (".final" if args.final else ""))
    ctx.log.info("analisi in %s", out_dir)
    for r in tables["op_all"].itertuples():
        ctx.log.info("%s %-16s acc=%.4f bacc=%.4f costo=%.3f", r.split, r.model, r.acc, r.bacc, r.cost_frac_L1)


if __name__ == "__main__":
    main()
