"""52 — Aggregazione su più seed del fine-tuning di CamFormer (CPU).

Scopo:
    Il fine-tuning di CamFormer è lo stadio stocastico principale: lo si ripete con 3 seed
    (--set camformer.seed=0|1|2 sugli stadi 20, 35, 45, 50). Questo stadio legge le tabelle
    dello stadio 50 per ogni seed e riporta, per ogni gate e metrica, media, deviazione
    standard, minimo e massimo tra i seed, accanto alla media degli IC bootstrap per take
    (variabilità del campione) — così si vedono entrambe le fonti di incertezza. Aggrega
    anche threshold_transfer (RQ1) e diag_c1 (D1), quando presenti.

Input:
    <run_dir>/50_gate/seed<S>_hz<F>/<tag>/{metrics,comparisons,threshold_transfer,diag_c1}.csv per S in --seeds

Output:
    <run_dir>/52_seeds/hz<F>/<tag>/metrics_seeds.csv, comparisons_seeds.csv,
    threshold_transfer_seeds.csv, diag_c1_seeds.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.cascade.cli import open_stage, stage_parser  # noqa: E402
from src.cascade.io import write_csv  # noqa: E402

STAGE = "52_seeds"
# (file di 50, chiavi di raggruppamento, colonne da aggregare, file di 52)
TABLES = (
    ("metrics", ["split", "model", "metric"], ["value"], "metrics_seeds"),
    ("comparisons", ["split", "a", "b", "metric"], ["diff"], "comparisons_seeds"),
    (
        "threshold_transfer",
        ["split", "model", "budget_frac"],
        ["cost_over_budget", "acc_transfer", "acc_reopt", "acc_gap", "bacc_gap"],
        "threshold_transfer_seeds",
    ),
    ("diag_c1", ["split", "model"], ["auroc"], "diag_c1_seeds"),
)


def aggregate(frames: List[pd.DataFrame], keys: List[str], values) -> pd.DataFrame:
    """Media, std, min, max e numero di seed per ogni colonna in ``values`` (una colonna, o più)."""
    d = pd.concat(frames, ignore_index=True)
    g = d.groupby(keys)
    values = [values] if isinstance(values, str) else list(values)
    parts = []
    for v in values:
        a = g[v].agg(["mean", "std", "min", "max", "count"]).rename(columns={"count": "n_seeds"})
        parts.append(a if len(values) == 1 else a.add_prefix(f"{v}_"))
    out = pd.concat(parts, axis=1)
    for c in ("ci_low", "ci_high"):
        if c in d:
            out[f"{c}_mean"] = g[c].mean()
    return out.reset_index()


def _read(path: Path) -> Optional[pd.DataFrame]:
    """CSV di 50, o None se manca o è vuoto (threshold_transfer senza --final)."""
    if not path.exists() or path.stat().st_size <= 1:
        return None
    df = pd.read_csv(path)
    return df if len(df) else None


def main(argv: Optional[List[str]] = None) -> None:
    p = stage_parser(__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--tag", type=str, default="main")
    args = p.parse_args(argv)
    ctx = open_stage(args, STAGE, make_stage_dir=False)
    out_dir = ctx.run_dir / STAGE / ctx.traj_tag / args.tag
    found, frames = [], {name: [] for name, _, _, _ in TABLES}
    for sd in args.seeds:
        d = ctx.run_dir / "50_gate" / f"seed{sd}_{ctx.traj_tag}" / args.tag
        if _read(d / "metrics.csv") is None:
            ctx.log.warning("seed %d: %s mancante, escluso", sd, d)
            continue
        found.append(sd)
        for name in frames:
            df = _read(d / f"{name}.csv")
            if df is not None:
                frames[name].append(df.assign(cam_seed=sd))
    if len(found) < 2:
        raise RuntimeError(f"servono almeno 2 seed con lo stadio 50 completo, trovati {found}")
    outs = []
    for name, keys, values, target in TABLES:
        if not frames[name]:
            continue
        outs.append(out_dir / f"{target}.csv")
        write_csv(aggregate(frames[name], keys, values), outs[-1])
    ctx.finish(outs, part=f"{ctx.traj_tag}.{args.tag}", extra={"seeds": found})
    ctx.log.info("aggregati %d seed (%s) in %s", len(found), found, out_dir)


if __name__ == "__main__":
    main()
