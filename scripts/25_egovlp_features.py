"""25 — Feature TimeSformer-EgoVLP congelato e logit zero-shot. [GPU, array per shard]

Scopo:
    Per ogni clip con frame estratti prende i 4 frame egovlp.frame_indices dei 16,
    normalizza come EgoVLP e calcola: CLS a 768 (pre-proiezione), embedding proiettato
    (256) e logit zero-shot sulle 8 classi (prompt in egovlp.zero_shot.prompts).
    Backbone congelato: nessun training qui. Verifica che lo stadio 10 sia completo
    (tutti gli shard) e che i frame mancanti non superino data.max_missing_frac.

Input:
    <run_dir>/data/clips.csv, paths.frames_dir/<clip_id>.npy, paths.egovlp_ckpt

Output:
    <run_dir>/25_egovlp/features_shard<i>of<n>.parquet
        clip_id, take_uid, split, label_idx; vid_cls[768], vid_emb[256], vid_zs_logits[8]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.cascade.cache.checks import check_loss  # noqa: E402
from src.cascade.cli import open_stage, shard_items, stage_parser  # noqa: E402

STAGE = "25_egovlp"


def frames_status(run_dir: Path) -> pd.DataFrame:
    """Esito per clip dello stadio 10 (tutti gli shard)."""
    files = sorted((run_dir / "10_frames_manifest").glob("shard*.csv"))
    if not files:
        return pd.DataFrame(columns=["clip_id", "status"])
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def main(argv: Optional[List[str]] = None) -> None:
    p = stage_parser(__doc__, shards=True)
    args = p.parse_args(argv)
    ctx = open_stage(args, STAGE)
    cfg, ecfg = ctx.cfg, ctx.cfg.egovlp
    out = ctx.stage_dir / f"features_shard{ctx.shard_id:03d}of{ctx.num_shards:03d}.parquet"
    part = f"shard{ctx.shard_id:03d}of{ctx.num_shards:03d}"
    if ctx.done([out], part=part):
        return
    ctx.require("00_inventory")
    ctx.require("10_frames", sharded=True)

    import torch
    from torch.utils.data import DataLoader

    from src.cascade.models.camformer import resolve_device
    from src.cascade.models.egovlp import (
        TextEncoder,
        build_video_model,
        load_state,
        load_video_weights,
        normalize,
        zero_shot_logits,
    )
    from src.data.cache import Table, write_table
    from src.data.clips import ClipFramesDataset, collate_frames

    clips = ctx.read_clips()
    clips = clips[clips.has_video].reset_index(drop=True)
    rows = shard_items(list(range(len(clips))), ctx.shard_id, ctx.num_shards)
    mine = clips.iloc[rows]
    ds = ClipFramesDataset(mine, ctx.path(cfg.paths.frames_dir), ecfg.frame_indices)
    st = frames_status(ctx.run_dir).set_index("clip_id").status.reindex(mine.clip_id).fillna("missing")
    dropped = st.str.startswith("out_of_range").to_numpy()  # scartate per politica: attese
    check_loss(int((~dropped).sum()), len(ds), cfg.data.max_missing_frac, "25: frame disponibili")
    model = build_video_model(ecfg.arch)
    sd = None
    if ecfg.pretrained:
        ckpt = ctx.path(cfg.paths.egovlp_ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint EgoVLP mancante: {ckpt} (o egovlp.pretrained=false)")
        sd = load_state(ckpt)
        load_video_weights(model, sd)
    else:
        ctx.log.warning("egovlp.pretrained=false: pesi casuali")
    device = resolve_device(ecfg.device)
    model.eval().to(device)
    for prm in model.parameters():
        prm.requires_grad_(False)

    cls_l, emb_l = [], []
    loader = DataLoader(
        ds,
        batch_size=ecfg.batch_size,
        shuffle=False,
        collate_fn=collate_frames,
        num_workers=ecfg.num_workers,
        pin_memory=ecfg.pin_memory and device.type == "cuda",
    )
    with torch.no_grad():
        for b in loader:
            x = normalize(b["frames"], ecfg.norm_mean, ecfg.norm_std).to(device, non_blocking=True)
            c, e = model(x)
            cls_l.append(c.cpu().numpy())
            emb_l.append(e.cpu().numpy())
    cls = np.concatenate(cls_l) if cls_l else np.zeros((0, ecfg.arch.embed_dim))
    emb = np.concatenate(emb_l) if emb_l else np.zeros((0, ecfg.arch.proj_dim))
    meta_df = clips.set_index("clip_id").loc[ds.clip_ids, ["take_uid", "split", "label_idx"]].reset_index()
    arrays = {"vid_cls": cls.astype(np.float32), "vid_emb": emb.astype(np.float32)}
    zs = ecfg.zero_shot
    if zs.enabled:
        missing = [n for n in ctx.labels.names if n not in zs.prompts]
        if missing:
            raise ValueError(f"egovlp.zero_shot.prompts: mancano {missing}")
        prompts = [zs.prompts[n] for n in ctx.labels.names]  # ordine di labels.yaml
        txt = TextEncoder(zs.text_model, ecfg.arch.proj_dim, sd).encode(prompts)
        arrays["vid_zs_logits"] = zero_shot_logits(emb, txt, zs.temperature).astype(np.float32)
        seen = (meta_df.split != "test").to_numpy()  # il test non si guarda qui
        if seen.any():
            acc = float((arrays["vid_zs_logits"][seen].argmax(1) == meta_df.label_idx.to_numpy()[seen]).mean())
            ctx.log.info("accuratezza zero-shot (shard, train+val): %.4f", acc)
    write_table(
        Table(
            meta_df,
            arrays,
            {"class_names": list(ctx.labels.names), "stage": STAGE, "frame_indices": list(ecfg.frame_indices)},
        ),
        out,
    )
    ctx.finish([out], part=part)
    ctx.log.info("scritte %d clip in %s", len(meta_df), out)


if __name__ == "__main__":
    main()
