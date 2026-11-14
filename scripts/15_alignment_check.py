"""15 — Controllo visivo dell'allineamento traiettoria ↔ video.

Scopo:
    Per N take salva un'immagine con una griglia di frame (lungo tutta la take) e,
    sotto, quota e velocità della traiettoria sullo stesso asse temporale (tempo
    video, dopo traj_to_video_time), con le linee verticali agli istanti dei frame.
    Serve a scegliere alignment.mode / offset_s al primo accesso all'HPC e a
    verificare l'orientamento dei frame (frames.rotate_k90).

Input:
    <run_dir>/data/clips.csv (take con video e traiettoria)

Output:
    <run_dir>/15_alignment/<take_uid>.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.cascade.cli import open_stage, stage_parser  # noqa: E402
from src.cascade.eval.style import GRID, SERIES, SURFACE, TEXT2  # noqa: E402
from src.cascade.trajectory import load_take_trajectory  # noqa: E402
from src.cascade.video import decode_frames, frame_indices_for_times, probe_video  # noqa: E402
from src.data.trajectories import clip_meta  # noqa: E402

STAGE = "15_alignment"


def plot_take(row, cfg, out: Path, n_frames: int = 8) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    info = probe_video(Path(row.video_path))
    tr = load_take_trajectory(Path(row.traj_path), cfg, clip_meta(row))
    t_end = min(info.duration_s, row.take_duration)
    times = np.linspace(0, t_end, n_frames + 2)[1:-1]
    frames = decode_frames(
        Path(row.video_path), frame_indices_for_times(times, info.fps, info.n_frames), cfg.frames.decoder
    )
    if cfg.frames.rotate_k90 % 4:
        frames = np.rot90(frames, k=cfg.frames.rotate_k90, axes=(1, 2))
    fig = plt.figure(figsize=(12, 6), dpi=110, facecolor=SURFACE)
    gs = fig.add_gridspec(3, n_frames, height_ratios=[1.2, 1, 1])
    for i, (f, t) in enumerate(zip(frames, times)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(f)
        ax.set_axis_off()
        ax.set_title(f"{t:.1f}s", fontsize=8, color=TEXT2)
    z = tr.poses[:, cfg.data.vertical_axis]
    speed = np.r_[0, np.linalg.norm(np.diff(tr.poses[:, :3], axis=0), axis=1) * cfg.data.traj_fps]
    for j, (y, lab) in enumerate(((z, "quota [m]"), (speed, "velocità [m/s]"))):
        ax = fig.add_subplot(gs[1 + j, :])
        ax.plot(tr.t_video, y, color=SERIES[0], lw=1.5)
        for t in times:
            ax.axvline(t, color=GRID, lw=1)
        ax.axvspan(row.t_start, row.t_end, color=SERIES[1], alpha=0.12, lw=0)
        ax.set_xlim(0, t_end)
        ax.set_ylabel(lab, fontsize=9, color=TEXT2)
        ax.set_facecolor(SURFACE)
        ax.tick_params(labelsize=8, colors=TEXT2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax.set_xlabel("tempo video [s] (fascia = finestra della clip)", fontsize=9, color=TEXT2)
    a = cfg.alignment
    fig.suptitle(f"{row.take_uid} — alignment.mode={a.mode}, offset_s={a.offset_s}", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> None:
    p = stage_parser(__doc__)
    p.add_argument("--n-takes", type=int, default=6)
    p.add_argument("--split", type=str, default="val")
    args = p.parse_args(argv)
    ctx = open_stage(args, STAGE)
    clips = ctx.read_clips()
    ok = clips[clips.has_video & clips.has_traj & (clips.split == args.split)].drop_duplicates("take_uid")
    pick = ok.sample(min(args.n_takes, len(ok)), random_state=ctx.cfg.seed)
    for r in pick.itertuples(index=False):
        out = ctx.stage_dir / f"{r.take_uid}.png"
        if out.exists() and not ctx.overwrite:
            continue
        plot_take(r, ctx.cfg, out)
        ctx.log.info("scritto %s", out)


if __name__ == "__main__":
    main()
