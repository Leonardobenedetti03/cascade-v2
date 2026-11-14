"""27 — Diagnostiche di sito sulle CLS di L1 (CPU). [M02]

Scopo:
    Quanto pesa il laboratorio (sito) nelle feature di L1 e quanto cala L1 su un sito mai visto.
    Sono i numeri che giustificano lo split per sito (M02) nel report:
    (a) leave-one-site-out sulle CLS, per sito e per classe;
    (b) leggibilità del sito dalle CLS (logistica che predice il sito, CV per take e per sessione);
    (c) baseline «classe più frequente del sito».
    In più la CV della classe per take e per sessione (da check_leak.py della v1).
    Usa solo train e val: lo stadio gira fuori da 50 --final e il test non si guarda.
    Usa tutti gli shard dello stadio 25 (verificati completi).

Input:
    <run_dir>/25_egovlp/features_shard*.parquet, paths.scenario_csv (con il suo hash)

Output (<run_dir>/27_site/):
    loso.csv           site, role, class, n_clips, n_takes, n_scored, n_correct, acc, reported
                       (righe per sito × classe, per sito, per classe e totale; ``ALL``)
    site_probe.json    LOSO complessivo, leggibilità del sito, CV della classe, numero di fit e avvisi
    site_majority.json baseline per sito (in-sample sul train), sito × classe
    Con meno di 2 siti in train+val (dati sintetici) loso.csv ha solo l'intestazione e site_probe.json
    ha la chiave ``skipped``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cascade.cli import open_stage, stage_parser  # noqa: E402
from src.cascade.clips import load_scenario  # noqa: E402
from src.cascade.eval.site_diag import run_site_diagnostics  # noqa: E402
from src.cascade.io import write_csv, write_json  # noqa: E402
from src.cascade.labels import assert_class_order  # noqa: E402
from src.cascade.site_split import check_split_hash  # noqa: E402
from src.data.cache import Table, concat_tables, read_table  # noqa: E402

STAGE = "27_site"


def load_features(ctx) -> Table:
    """Concatena gli shard elencati nei manifest dello stadio 25 (come lo stadio 30)."""
    ms = ctx.require("25_egovlp", sharded=True)
    shards = sorted({o for m in ms for o in m["outputs"] if o.endswith(".parquet")})
    tabs = [read_table(Path(p)) for p in shards]
    for p, t in zip(shards, tabs):
        assert_class_order(t.meta["class_names"], ctx.labels, where=str(p))
    t = concat_tables(tabs)
    t.check_unique()
    return t


def main(argv: Optional[List[str]] = None) -> None:
    p = stage_parser(__doc__)
    args = p.parse_args(argv)
    ctx = open_stage(args, STAGE)
    outs = [ctx.stage_dir / f for f in ("loso.csv", "site_probe.json", "site_majority.json")]
    if ctx.done(outs):
        return
    ctx.require("05_folds")
    cfg = ctx.cfg
    labels = ctx.scenario_labels or ctx.labels
    scenario_csv = Path(ctx.path(cfg.paths.scenario_csv))
    ctx.log.info("split %s: SHA256 %s", scenario_csv.name, check_split_hash(scenario_csv, cfg.paths.split_sha256))
    feats = load_features(ctx)
    takes = load_scenario(scenario_csv, labels)
    sd = cfg.site_diag
    res = run_site_diagnostics(
        feats.df,
        feats.arrays["vid_cls"],
        takes,
        labels.names,
        C=sd.C,
        max_iter=sd.max_iter,
        cv_folds=sd.cv_folds,
        min_class_takes=cfg.gate.bacc_min_takes,
    )
    write_csv(res["loso"], outs[0])
    write_json(outs[1], res["probe"])
    write_json(outs[2], res["majority"])
    probe = res["probe"]
    if "skipped" in probe:
        ctx.log.warning("sonda di sito saltata: %s", probe["skipped"])
    else:
        ctx.log.info(
            "leave-one-site-out: %.4f su %d clip (%d escluse, classe assente dal train); sito leggibile (sessione): %s",
            probe["loso"]["acc"],
            probe["loso"]["n_scored"],
            probe["loso"]["n_excluded_unseen_class"],
            probe["site_readability"]["by_session"].get("acc"),
        )
    ctx.finish(outs)
    ctx.log.info("scritti %s", ctx.stage_dir)


if __name__ == "__main__":
    main()
