"""56 — Gate v2: CV annidata per sessione su tutte le take di scenario_preds (CPU, N37).

Scopo:
    Rifà la valutazione della cascata con un protocollo che non dipende dallo split ufficiale né dal test già
    aperto (N34): tutte le take delle 6 classi si dividono in fold per sessione; per ogni fold esterna le teste
    lineari di CamFormer ed EgoVLP si riaddestrano sulle altre fold, il gate si addestra su predizioni fuori
    campione di CamFormer (CV interna), calibratore e soglie si stimano in cross-fitting, e tutto si applica
    congelato alla fold esterna. Si ripete per ``gate_v2.repeats`` partizioni e si mediano le decisioni per take.

    Il gate vede solo l'uscita di CamFormer e la traiettoria. Label ed EgoVLP servono come target e per contare
    gli errori. Non è una valutazione cieca: val e test ufficiali sono già stati analizzati (N33, N34); il
    metodo e i valori sono fissati nel registro (N37) prima di questa run.

Input:
    --scenario-run DIR: run di scenario_preds con index/takes.csv, features/{camformer,egovlp}.npz,
    configs/heads.json e traj/<take>.npz (per il moto; senza, gruppo C = NaN imputato)

Output (<run_dir>/56_gate_v2/<tag>/):
    takes.csv.gz (score, rischio e predizioni fuori campione per repeat e take), thresholds.csv, heads.csv,
    importance.csv, outer_folds.csv, metrics.csv, models.csv, curves.csv, auc.csv, comparisons.csv,
    calibration.csv, procedural.csv, cost_sensitivity.csv, cost_vs_egovlp.csv, summary.json, config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cascade.cli import open_stage, stage_parser  # noqa: E402
from src.cascade.config import dump_config  # noqa: E402
from src.cascade.gatev2.data import check_subsets, load_scenario_run  # noqa: E402
from src.cascade.gatev2.evaluate import evaluate, primary_result  # noqa: E402
from src.cascade.gatev2.nested import run_all  # noqa: E402
from src.cascade.gatev2.scorers import check_scorers  # noqa: E402
from src.cascade.io import write_json  # noqa: E402

STAGE = "56_gate_v2"
TABLES = (
    "metrics",
    "models",
    "curves",
    "auc",
    "comparisons",
    "calibration",
    "procedural",
    "cost_sensitivity",
    "cost_vs_egovlp",
)
RAW = ("thresholds", "heads", "importance", "outer_folds")


def check_config(g) -> None:
    if g.primary_scorer not in g.scorers:
        raise ValueError(f"gate_v2.primary_scorer {g.primary_scorer!r} non è in gate_v2.scorers")
    rules = {
        "cost": g.betas,
        "cost_hat": g.betas,
        "risk": g.alphas,
        "acc_target": g.max_error_rates,
        "budget": g.budgets,
    }
    if g.primary_rule not in rules or not any(abs(v - g.primary_param) < 1e-12 for v in rules[g.primary_rule]):
        raise ValueError(f"punto primario {g.primary_rule}={g.primary_param} non è nella griglia pre-registrata")
    for a, b in g.comparisons:
        if a not in g.scorers or b not in g.scorers:
            raise ValueError(f"confronto [{a}, {b}] con scorer non elencati in gate_v2.scorers")
    if g.calibration_population not in ("all", "pretrain_unseen"):
        raise ValueError(f"gate_v2.calibration_population sconosciuto: {g.calibration_population!r}")
    if g.reliability_folds < 2:
        raise ValueError("gate_v2.reliability_folds deve essere almeno 2")
    if min(g.outer_folds, g.inner_folds, g.gate_folds) < 2:
        raise ValueError("servono almeno 2 fold per livello")


def main(argv: Optional[List[str]] = None) -> None:
    p = stage_parser(__doc__)
    p.add_argument("--scenario-run", type=Path, required=True, help="run_dir di scenario_preds (feature calcolate)")
    p.add_argument("--tag", type=str, default="main", help="sottocartella degli output")
    p.add_argument("--n-jobs", type=int, default=1, help="fold esterne in parallelo (joblib)")
    p.add_argument("--no-motion", action="store_true", help="non legge le traiettorie (gruppo C = NaN)")
    args = p.parse_args(argv)
    ctx = open_stage(args, STAGE, make_stage_dir=False)
    g = ctx.cfg.gate_v2
    check_config(g)
    out_dir = ctx.run_dir / STAGE / args.tag
    outs = [out_dir / f"{k}.csv" for k in TABLES + RAW] + [
        out_dir / n for n in ("takes.csv.gz", "summary.json", "config.yaml")
    ]
    if ctx.done(outs, part=args.tag):
        return
    labels = ctx.labels
    check_scorers(list(g.scorers), g, labels.procedural_mask)
    scen = args.scenario_run.expanduser().resolve()
    data = load_scenario_run(
        scen, list(labels.names), labels.procedural_mask, int(ctx.cfg.data.vertical_axis), motion=not args.no_motion
    )
    check_subsets(data.takes, list(g.subsets), g.primary_subset)
    ctx.log.warning(
        "gate v2: CV annidata su %d take e %d sessioni, comprese val e test ufficiali (già valutati in N34): "
        "analisi dichiarata in N37, non una valutazione cieca",
        len(data),
        data.takes.session.nunique(),
    )
    res = run_all(data, g, n_jobs=args.n_jobs)
    tables = evaluate(data, res, g)
    out_dir.mkdir(parents=True, exist_ok=True)
    res["takes"].to_csv(out_dir / "takes.csv.gz", index=False)
    for k in RAW:
        res[k].to_csv(out_dir / f"{k}.csv", index=False)
    for k in TABLES:
        tables[k].to_csv(out_dir / f"{k}.csv", index=False)
    summary = primary_result(tables, g)
    summary.update(
        {"counts": data.counts, "n_windows": data.n_windows, "scenario_run": str(scen), "repeats": list(g.repeats)}
    )
    if summary["l1_always_correct"] is False:
        ctx.log.warning("EgoVLP sbaglia su alcune take: il target e = 1 − c0 non coincide più con il beneficio atteso")
    write_json(out_dir / "summary.json", summary)
    dump_config(ctx.cfg, out_dir / "config.yaml")
    ctx.finish(outs, part=args.tag, extra={"scenario_run": str(scen)})
    pr = summary["primary"]
    ctx.log.info(
        "risultato primario (%s, %s, %s=%s): acc %.4f [%.4f, %.4f], quota a EgoVLP %.4f [%.4f, %.4f]",
        pr["subset"],
        pr["scorer"],
        pr["rule"],
        pr["param"],
        pr.get("acc", float("nan")),
        pr.get("acc_ci_low", float("nan")),
        pr.get("acc_ci_high", float("nan")),
        pr.get("rho", float("nan")),
        pr.get("rho_ci_low", float("nan")),
        pr.get("rho_ci_high", float("nan")),
    )
    cp = summary.get("cost_vs_egovlp_primary", {})
    if cp and not cp.get("missing"):
        ctx.log.info(
            "costo al β principale: %.1f GMACs per take contro %.1f di solo EgoVLP (%.1f%%, %.1f volte meno)",
            cp["gmacs_per_take_cascade"],
            cp["gmacs_per_take_egovlp_only"],
            100 * cp["cost_frac_of_egovlp_only"],
            cp["speedup"],
        )
    ctx.log.info("output in %s", out_dir)


if __name__ == "__main__":
    main()
