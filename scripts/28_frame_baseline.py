"""28 — Baseline "un frame" L½: MobileNetV3-Small congelato su 1 frame a 112 px (B1). [GPU/CPU]

Scopo:
    Misurare quanto basta un solo frame a bassa risoluzione: gli 8 scenari sono 8 luoghi
    diversi. Feature dopo l'average pooling (576) del frame frame_baseline.frame_index dei
    16 estratti, poi una testa lineare per fold (C in CV interna raggruppata per take,
    temperatura sull'holdout), predizioni OOF sul train e media dei 5 modelli su val/test.
    Attivo solo con frame_baseline.enabled = true. Pesi da paths.frame_baseline_ckpt.

Input:
    <run_dir>/data/clips.csv, paths.frames_dir/<clip_id>.npy, <run_dir>/splits/folds.csv

Output (<run_dir>/28_frame_baseline/):
    preds.parquet   clip_id, take_uid, split, label_idx, T_half, half_fold;
                    half_logits[8], half_probs_cal[8], half_feat[576]
    summary.csv, accuracy.json (train OOF e val; il test solo come conteggio)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.cascade.cache.checks import check_loss  # noqa: E402
from src.cascade.cli import open_stage, stage_parser  # noqa: E402
from src.cascade.heads import run_heads  # noqa: E402
from src.cascade.io import write_csv, write_json  # noqa: E402

STAGE = "28_frame_baseline"


def main(argv: Optional[List[str]] = None) -> None:
    p = stage_parser(__doc__)
    args = p.parse_args(argv)
    ctx = open_stage(args, STAGE)
    cfg, fb = ctx.cfg, ctx.cfg.frame_baseline
    if not fb.enabled:
        ctx.log.info("frame_baseline.enabled = false: niente da fare")
        return
    outs = [ctx.stage_dir / f for f in ("preds.parquet", "summary.csv", "accuracy.json")]
    if ctx.done(outs):
        return
    for st in ("00_inventory", "05_folds"):
        ctx.require(st)
    ctx.require("10_frames", sharded=True)

    from src.cascade.models.camformer import resolve_device
    from src.cascade.models.frame_baseline import build_frame_model, frame_features, load_frame_weights
    from src.data.cache import Table, write_table

    all_clips = ctx.read_clips()
    frames_dir = ctx.path(cfg.paths.frames_dir)
    clips = all_clips[all_clips.has_video & all_clips.clip_id.map(lambda c: (frames_dir / f"{c}.npy").exists())]
    check_loss(
        int(all_clips.has_video.sum()),
        len(clips),
        cfg.data.max_missing_frac,
        "28: clip con frame",
    )
    model = build_frame_model(fb.arch)
    if fb.pretrained:
        ckpt = ctx.path(cfg.paths.frame_baseline_ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"pesi della baseline un frame mancanti: {ckpt} (o frame_baseline.pretrained=false)"
            )
        load_frame_weights(model, ckpt)
    else:
        ctx.log.warning("frame_baseline.pretrained=false: pesi casuali")
    for prm in model.parameters():
        prm.requires_grad_(False)
    ids = clips.clip_id.tolist()
    feats = frame_features(model, frames_dir, ids, fb.frame_index, fb.size, fb.batch_size, resolve_device(fb.device))
    folds = pd.read_csv(ctx.run_dir / "splits" / "folds.csv")
    df = clips[["clip_id", "take_uid", "split", "label_idx"]].reset_index(drop=True)
    res, summary = run_heads(df, feats, folds, cfg.heads, ctx.labels.num_classes, cfg.seed)
    df["T_half"], df["half_fold"] = res["T_vid"], res["vid_fold"]
    arrays = {
        "half_logits": res["vid_logits"].astype(np.float32),
        "half_probs_cal": res["vid_probs_cal"].astype(np.float32),
        "half_feat": feats.astype(np.float32),
    }
    write_table(
        Table(
            df,
            arrays,
            {"class_names": list(ctx.labels.names), "stage": STAGE, "half_train_source": "oof", "size": fb.size},
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
    ctx.finish(outs)
    ctx.log.info("L½ (1 frame a %d px): %s", fb.size, accs)


if __name__ == "__main__":
    main()
