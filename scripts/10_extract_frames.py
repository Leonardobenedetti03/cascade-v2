"""10 — Estrazione dei frame delle clip (CPU, job array per shard).

Scopo:
    Per ogni clip con video decodifica frames.n_frames (16) frame equispaziati nella
    finestra (decord, con fallback PyAV), applica Resize(256) lato corto,
    CenterCrop(256), Resize(224) come il transform di test di EgoVLP e salva un
    .npy uint8 [16, 224, 224, 3] per clip (niente JPG singoli).
    Riprendibile: salta i .npy già presenti; --overwrite per rifarli. Le finestre che
    escono dal video seguono data.out_of_range (drop | clamp), con esito nel manifest.

Input:
    <run_dir>/data/clips.csv

Output:
    paths.frames_dir/<clip_id>.npy   (default {data}/frames/<run_name>: persistente, fuori dalla run)
    <run_dir>/10_frames_manifest/shard<i>of<n>.csv (esito per clip)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.cascade.cli import open_stage, shard_items, stage_parser  # noqa: E402
from src.cascade.frames import OutOfRange, extract_clip_frames, save_npy_atomic  # noqa: E402
from src.cascade.io import write_csv  # noqa: E402
from src.cascade.logs import progress  # noqa: E402

STAGE = "10_frames"


def main(argv: Optional[List[str]] = None) -> None:
    p = stage_parser(__doc__, shards=True)
    p.add_argument("--limit", type=int, default=None, help="solo le prime N clip dello shard (debug)")
    args = p.parse_args(argv)
    ctx = open_stage(args, STAGE, make_stage_dir=False)
    ctx.require("00_inventory")
    frames_dir = ctx.path(ctx.cfg.paths.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    man = ctx.run_dir / "10_frames_manifest" / f"shard{ctx.shard_id:03d}of{ctx.num_shards:03d}.csv"
    part = f"shard{ctx.shard_id:03d}of{ctx.num_shards:03d}"
    if args.limit is None and ctx.done([man], part=part):
        return
    clips = ctx.read_clips()
    todo = shard_items(list(clips[clips.has_video].itertuples(index=False)), ctx.shard_id, ctx.num_shards)
    if args.limit:
        todo = todo[: args.limit]
    status = []
    for i, r in enumerate(progress(todo, total=len(todo), desc="frame")):
        out = frames_dir / f"{r.clip_id}.npy"
        if out.exists() and not ctx.overwrite:
            status.append({"clip_id": r.clip_id, "status": "exists"})
            continue
        try:
            arr, clamped = extract_clip_frames(
                Path(r.video_path), r.t_start, r.t_end, ctx.cfg.frames, ctx.cfg.data.out_of_range
            )
            save_npy_atomic(out, arr)
            status.append({"clip_id": r.clip_id, "status": "clamped" if clamped else "ok"})
        except OutOfRange as e:
            ctx.log.warning("clip %s scartata: %s", r.clip_id, e)
            status.append({"clip_id": r.clip_id, "status": f"out_of_range: {e}"})
        except Exception as e:  # noqa: BLE001 - una clip rotta non deve fermare lo shard
            ctx.log.error("clip %s: %s", r.clip_id, e)
            status.append({"clip_id": r.clip_id, "status": f"error: {e}"})
        if (i + 1) % 100 == 0:
            ctx.log.info("%d/%d clip", i + 1, len(todo))
    write_csv(pd.DataFrame(status, columns=["clip_id", "status"]), man)
    st = pd.Series([s["status"].split(":")[0] for s in status], dtype=object).value_counts().to_dict()
    ctx.log.info("shard %d/%d: %s -> %s", ctx.shard_id, ctx.num_shards, st, frames_dir)
    if args.limit is None:
        ctx.finish([man], part=part, extra={"status_counts": st})


if __name__ == "__main__":
    main()
