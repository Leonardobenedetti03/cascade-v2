"""30 — Teste lineari L1 sulle CLS di EgoVLP (CPU).

Scopo:
    Per ogni fold, LogisticRegression multinomiale sulle CLS a 768 (C scelto in CV
    interna raggruppata per take), temperatura stimata sull'holdout interno.
    Predizioni fuori campione sul train (ogni clip dal modello della sua fold);
    su val e test media delle probabilità calibrate dei 5 modelli di fold.
    Usa tutti gli shard dello stadio 25 (verificati completi).

Input:
    <run_dir>/25_egovlp/features_shard*.parquet,
    <run_dir>/splits/folds.csv

Output (<run_dir>/30_l1_heads/probe/):
    preds.parquet   clip_id, take_uid, split, label_idx, T_vid, vid_fold;
                    vid_logits[8], vid_probs_cal[8] (+ vid_cls, vid_emb, vid_zs_logits passati)
    summary.csv     C, T e accuratezza OOF per fold
    accuracy.json   accuratezza su train OOF e val (il test solo come conteggio)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.cascade.cli import open_stage, stage_parser  # noqa: E402
from src.cascade.heads import run_heads  # noqa: E402
from src.cascade.io import write_csv, write_json  # noqa: E402
from src.cascade.labels import assert_class_order  # noqa: E402
from src.data.cache import Table, concat_tables, read_table, write_table  # noqa: E402

STAGE = "30_l1_heads"


def load_features(ctx) -> Table:
    """Concatena gli shard elencati nei manifest dello stadio 25 (tutti presenti, stessa N)."""
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
    outs = [ctx.l1_dir() / f for f in ("preds.parquet", "summary.csv", "accuracy.json")]
    if ctx.done(outs, part=ctx.l1_tag):
        return
    ctx.require("05_folds")
    folds = pd.read_csv(ctx.run_dir / "splits" / "folds.csv")
    feats = load_features(ctx)
    res, summary = run_heads(
        feats.df, feats.arrays["vid_cls"], folds, ctx.cfg.heads, ctx.labels.num_classes, ctx.cfg.seed
    )
    df = feats.df[["clip_id", "take_uid", "split", "label_idx"]].copy()
    df["T_vid"], df["vid_fold"] = res["T_vid"], res["vid_fold"]
    arrays = {"vid_logits": res["vid_logits"], "vid_probs_cal": res["vid_probs_cal"], **feats.arrays}
    write_table(
        Table(
            df,
            arrays,
            {
                "class_names": list(ctx.labels.names),
                "stage": STAGE,
                "vid_train_source": "oof",
                "l1_source": ctx.cfg.heads.source,
            },
        ),
        outs[0],
    )
    write_csv(summary, outs[1])
    accs = {}
    for s in ("train", "val", "test"):
        m = (df.split == s).to_numpy()
        if m.any():
            accs[s] = {"n": int(m.sum())}
            if s != "test":  # il test non si guarda fuori da 50 --final
                accs[s]["acc"] = float((res["vid_probs_cal"][m].argmax(1) == df.label_idx.to_numpy()[m]).mean())
    write_json(outs[2], accs)
    ctx.finish(outs, part=ctx.l1_tag)
    ctx.log.info("L1: %s", accs)


if __name__ == "__main__":
    main()
