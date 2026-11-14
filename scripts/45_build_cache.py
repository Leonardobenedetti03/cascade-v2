"""45 — Costruzione della cache finale (una riga per clip) (CPU).

Scopo:
    Unisce per clip_id le uscite di L0 (OOF sul train, verificate contro le fold; media
    delle fold su val/test), di L1 (teste lineari), la fusione (w dello stadio 35), i
    costi (stadio 40) e le feature di moto calcolate dalla traiettoria della finestra;
    calcola c0, c1, y e valida lo schema. Con frame_baseline.enabled aggiunge L½ (stadio 28)
    e le fusioni L0+L½ e L0+L½+L1. Si ferma se si perdono più di data.max_missing_frac clip
    utilizzabili rispetto all'indice delle clip.

Input:
    20_camformer/seed<S>_hz<F>/, 30_l1_heads/probe/, 35_fusion/<variante>/w.json,
    40_cost/hz<F>/cost.json, data/clips.csv, splits/folds.csv

Output:
    <run_dir>/45_cache/<variante>/cache.parquet, summary.json (metriche solo su train e val)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.cascade.cache.build import assemble_camformer, join_l0_l1, load_l1, motion_table  # noqa: E402
from src.cascade.cache.checks import check_loss  # noqa: E402
from src.cascade.cache.schema import add_labels, validate_cache  # noqa: E402
from src.cascade.cli import open_stage, stage_parser  # noqa: E402
from src.cascade.fusion import choose_simplex, fuse, fuse_many  # noqa: E402
from src.cascade.io import read_json, write_json  # noqa: E402
from src.data.cache import Table, join_tables, read_table, write_table  # noqa: E402

STAGE = "45_cache"


def add_frame_baseline(ctx, t: Table, cost: dict) -> Table:
    """L½ (stadio 28) e fusioni L0+L½ e L0+L½+L1 con pesi scelti sulle uscite OOF del train."""
    ctx.require("28_frame_baseline")
    half = read_table(ctx.run_dir / "28_frame_baseline" / "preds.parquet")
    if half.meta.get("half_train_source") != "oof":
        raise ValueError("28_frame_baseline: attese predizioni fuori campione sul train")
    keep = half.df.clip_id.isin(set(t.keys())).to_numpy()  # L½ esiste anche per clip senza L0
    half = half.mask(keep)
    missing = sorted(set(t.keys()) - set(half.keys()))
    if missing:
        raise ValueError(f"28_frame_baseline: mancano {len(missing)} clip della cache (es. {missing[:3]})")
    half_df = half.df[["clip_id", "T_half"]]
    t = join_tables(
        t, Table(half_df, {k: half.arrays[k] for k in ("half_logits", "half_probs_cal")}), validate_full=True
    )
    tr = (t.df.split == "train").to_numpy()
    y = t.df.label_idx.to_numpy()[tr]
    pc, ph, pv = t.arrays["cam_probs_cal"], t.arrays["half_probs_cal"], t.arrays["vid_probs_cal"]
    step = ctx.cfg.frame_baseline.w_grid_step
    w01, _, _ = choose_simplex([pc[tr], ph[tr]], y, step)
    w012, _, _ = choose_simplex([pc[tr], ph[tr], pv[tr]], y, step)
    t.arrays["fused01_probs"] = fuse_many([pc, ph], w01).astype(np.float32)
    t.arrays["fused012_probs"] = fuse_many([pc, ph, pv], w012).astype(np.float32)
    t.df["w01"] = w01[0]
    t.df["w012_cam"], t.df["w012_half"], t.df["w012_vid"] = w012
    t.df["gmacs_Lhalf"] = float(cost["gmacs_Lhalf"])
    t.df["pixels_Lhalf"], t.df["pixels_L1"] = int(cost["pixels_Lhalf"]), int(cost["pixels_L1"])
    ctx.log.info("L½: pesi L0+L½ %s, L0+L½+L1 %s", w01, w012)
    return t


def main(argv: Optional[List[str]] = None) -> None:
    p = stage_parser(__doc__)
    args = p.parse_args(argv)
    ctx = open_stage(args, STAGE, make_stage_dir=False)
    cfg = ctx.cfg
    out_dir = ctx.variant_dir(STAGE)
    outs = [out_dir / "cache.parquet", out_dir / "summary.json"]
    part = ctx.variant_tag
    if ctx.done(outs, part=part):
        return
    for st in ("00_inventory", "05_folds"):
        ctx.require(st)
    ctx.require("30_l1_heads", parts=[ctx.l1_tag])
    ctx.require("40_cost", parts=[ctx.traj_tag])
    ctx.require("20_camformer", parts=[f"{ctx.cam_tag}.fold{k}" for k in range(cfg.folds.n_folds)])
    ctx.require("35_fusion", parts=[ctx.variant_tag])
    clips = ctx.read_clips()
    folds = pd.read_csv(ctx.run_dir / "splits" / "folds.csv")
    j = join_l0_l1(
        assemble_camformer(ctx.seed_dir("20_camformer"), cfg, ctx.labels, folds),
        load_l1(ctx.l1_dir(), ctx.labels, folds),
    )
    usable = clips[clips.has_video & clips.has_traj]
    check_loss(len(usable), int(j.keys().size), cfg.data.max_missing_frac, "45: clip con L0 e L1")
    wj = read_json(ctx.variant_dir("35_fusion") / "w.json")
    if wj["mode"] != cfg.fusion.mode:
        raise ValueError(f"w.json calcolato con fusion.mode={wj['mode']}, config={cfg.fusion.mode}: rifai 35")
    cost = read_json(ctx.run_dir / "40_cost" / ctx.traj_tag / "cost.json")
    mot = motion_table(clips[clips.clip_id.isin(j.keys())], cfg, ctx.path(cfg.paths.traj_cache_dir))
    t = join_tables(j, Table(mot), validate_full=True)
    fold_of = dict(zip(folds.take_uid, folds.fold))
    missing_fold = sorted({u for u, s in zip(t.df.take_uid, t.df.split) if s == "train" and u not in fold_of})
    if missing_fold:
        raise ValueError(f"take di train assenti da splits/folds.csv: {missing_fold[:3]} (rifai 05)")
    t.df["fold"] = [int(fold_of[u]) if s == "train" else -1 for u, s in zip(t.df.take_uid, t.df.split)]
    t.arrays["fused_probs"] = fuse(t.arrays["cam_probs_cal"], t.arrays["vid_probs_cal"], float(wj["w"])).astype(
        np.float32
    )
    t.df["w"] = float(wj["w"])
    t.df["gmacs_L0"], t.df["gmacs_L1"] = float(cost["gmacs_L0"]), float(cost["gmacs_L1"])
    add_labels(t)
    if cfg.frame_baseline.enabled:
        t = add_frame_baseline(ctx, t, cost)
    t.meta.update(
        {"stage": STAGE, "window_s": cfg.data.window_s, "fusion_mode": cfg.fusion.mode, "cam_seed": ctx.cam_seed}
    )
    warnings = validate_cache(t, ctx.labels, cam_dim=cfg.camformer.arch.output_dim, vid_dim=cfg.egovlp.arch.embed_dim)
    for w in warnings:
        ctx.log.warning("schema: %s", w)
    write_table(t, outs[0])
    summ = {}
    for s in ("train", "val", "test"):
        m = (t.df.split == s).to_numpy()
        if not m.any():
            continue
        summ[s] = {"n": int(m.sum())}
        if s != "test":  # il test non si guarda fuori da 50 --final
            y = t.df.label_idx.to_numpy()[m]
            summ[s].update(
                {
                    "acc_L0": float(t.df.c0[m].mean()),
                    "acc_fused": float(t.df.c1[m].mean()),
                    "acc_L1": float((t.arrays["vid_probs_cal"][m].argmax(1) == y).mean()),
                    "y_rate": float(t.df.y[m].mean()),
                }
            )
    write_json(
        outs[1],
        {
            "splits": summ,
            "w": wj["w"],
            "warnings": warnings,
            "n_clips_index": int(len(clips)),
            "n_usable": int(len(usable)),
            "n_cache": int(len(t)),
        },
    )
    ctx.finish(outs, part=part)
    ctx.log.info("cache: %d clip -> %s; %s", len(t), outs[0], summ)


if __name__ == "__main__":
    main()
