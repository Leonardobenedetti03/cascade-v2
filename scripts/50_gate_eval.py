"""50 — Gate e valutazione della cascata (CPU).

Scopo:
    Addestra tutti i gate di gate.models con il protocollo fuori campione (CV di secondo
    livello sulle fold di L0, fit finale sul train, punto di lavoro su val) e calcola curve
    accuratezza–costo, metriche (anche con accuratezza bilanciata e I(δ)), confronti appaiati,
    AUROC/AUPRC, escalation per classe, diagnostiche D1 (AUROC di η̂₁ contro c1) e celle A/B/D/N,
    importanza delle feature e intervalli bootstrap per take.
    Il test si valuta SOLO con --final, che lascia una riga in 50_gate/FINAL_RUNS.log e
    FINAL_DONE nella cartella; senza --final nessun output contiene righe del test (nemmeno
    gli score) e lo stadio si rifiuta di girare se FINAL_DONE esiste già.
    Con --final lo stadio rigira sempre (non consulta il manifest): ogni valutazione finale
    lascia traccia, così verify_run.sh può contare le esecuzioni.

Input:
    <run_dir>/45_cache/<variante>/cache.parquet   (oppure --mock: cache finta realistica)

Output (<run_dir>/50_gate/<variante>/<tag>/, variante = seed<S>_hz<F>):
    curves.csv, metrics.csv, auc.csv, operating_point.csv, escalation_by_class.csv, importance.csv,
    comparisons.csv, per_class.csv, confusion.csv, gate_calibration.csv, positives.csv,
    curve_bands.csv, threshold_transfer.csv, diag_c1.csv, cells.csv, scores.parquet, config.yaml;
    FINAL_DONE con --final (threshold_transfer.csv è vuoto senza --final: serve il test)
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cascade.cache.mock import make_mock_cache  # noqa: E402
from src.cascade.cache.schema import validate_cache  # noqa: E402
from src.cascade.cli import open_stage, stage_parser, utc_now  # noqa: E402
from src.cascade.config import dump_config  # noqa: E402
from src.cascade.eval.gate_eval import OUTPUT_TABLES, evaluate, needed_columns, write_outputs  # noqa: E402
from src.cascade.io import write_text  # noqa: E402
from src.data.cache import parquet_columns, read_table, write_table  # noqa: E402

STAGE = "50_gate"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main(argv: Optional[List[str]] = None) -> None:
    p = stage_parser(__doc__)
    p.add_argument("--mock", action="store_true", help="usa una cache finta invece di 45_cache")
    p.add_argument("--mock-sizes", type=int, nargs=3, default=[1500, 400, 500], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--final", action="store_true", help="valuta anche il test (lascia traccia in FINAL_RUNS.log)")
    p.add_argument("--tag", type=str, default="main", help="sottocartella degli output (ablation)")
    args = p.parse_args(argv)
    ctx = open_stage(args, STAGE, make_stage_dir=False)
    out_dir = ctx.variant_dir(STAGE) / args.tag
    part = f"{ctx.variant_tag}.{args.tag}" + (".final" if args.final else "")
    outs = [out_dir / f"{k}.csv" for k in OUTPUT_TABLES] + [out_dir / "scores.parquet", out_dir / "config.yaml"]
    if args.final:
        outs.append(out_dir / "FINAL_DONE")
    elif (out_dir / "FINAL_DONE").exists():
        raise RuntimeError(
            f"{out_dir / 'FINAL_DONE'} esiste: la valutazione finale è già stata fatta. "
            "Rilanciare senza --final sovrascriverebbe gli output del test; usa un altro --tag."
        )
    if not args.final and ctx.done(outs, part=part):
        return
    if args.mock:
        cache_path = ctx.run_dir / STAGE / "mock_cache.parquet"
        n_tr, n_va, n_te = args.mock_sizes
        write_table(make_mock_cache(ctx.labels, n_tr, n_va, n_te, ctx.cfg.folds.n_folds, ctx.cfg.seed), cache_path)
    else:
        ctx.require("45_cache", parts=[ctx.variant_tag])
        cache_path = ctx.variant_dir("45_cache") / "cache.parquet"
    table = read_table(cache_path, columns=needed_columns(parquet_columns(cache_path), ctx.cfg.gate))
    for w in validate_cache(table, ctx.labels, require_embeddings=False):
        ctx.log.warning("schema: %s", w)
    ctx.log.info("cache %s: %d clip, split %s", cache_path, len(table), table.df.split.value_counts().to_dict())
    res = evaluate(table, ctx.labels, ctx.cfg, final=args.final)
    write_outputs(res, out_dir)
    dump_config(ctx.cfg, out_dir / "config.yaml")
    m = res["metrics"]
    for split in ("val", "test"):
        for key in ("area", "bacc_area"):
            sub = m[(m.split == split) & (m.metric == key)].sort_values("value", ascending=False)
            if len(sub):
                ctx.log.info("%s (%s): %s", key, split, ", ".join(f"{r.model}={r.value:.4f}" for r in sub.itertuples()))
    if args.final:
        line = (
            f"{utc_now()}\tcommit={ctx.commit}\tcache={cache_path}\tcache_sha={_sha(cache_path)}"
            f"\tvariant={ctx.variant_tag}\ttag={args.tag}\tmodels={','.join(ctx.cfg.gate.models)}\n"
        )
        log_path = ctx.run_dir / STAGE / "FINAL_RUNS.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(line)
        write_text(out_dir / "FINAL_DONE", line)
        ctx.log.warning("valutazione FINALE sul test registrata in %s", log_path)
    ctx.finish(outs, part=part)
    ctx.log.info("output in %s", out_dir)


if __name__ == "__main__":
    main()
