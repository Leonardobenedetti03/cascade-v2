"""41 — Costo end-to-end: tempo per fase di L0 e di L1, a batch 1 (M10). [GPU consigliata]

Scopo:
    I GMACs ignorano ciò che domina in pratica: aprire il video, decodificare, ridimensionare. Per un
    campione di clip stratificato per classe (cost_e2e.n_clips) misura, clip per clip e a batch 1:
        T_L0 = lettura della traiettoria + preprocessing (pose relative, gravità) + forward di CamFormer
        T_L1 = apertura e decodifica del video + preprocessing + forward di TimeSformer
    Il tempo della cascata è T(r) = T_L0 + r · T_L1. Regole: clip di riscaldamento scartate
    (cost_e2e.n_warmup), sincronizzazione della GPU prima e dopo ogni fase che la usa, mediana e 90°
    percentile, ``cost_e2e.passes`` passate sulle stesse clip: si riporta l'ultima (file già nella cache del
    sistema) e la prima si dichiara a parte.
    I modelli hanno l'architettura vera e pesi casuali: il tempo non dipende dai valori dei pesi.

Input:
    <run_dir>/data/clips.csv (stadio 00); facoltativo <run_dir>/40_cost/<hz>/cost.json per la soglia in GMACs

Output (<run_dir>/41_cost_e2e/):
    clip_times.csv   una riga per clip e per passata (colonna ``pass``), tutte le fasi in secondi
    summary.json     mediana e p90 per fase (ultima passata e prima), curva T(r), soglia di pareggio della
                     traiettoria in secondi (e in GMACs se c'è lo stadio 40), ambiente (GPU, CPU, decoder)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.cascade.cli import open_stage, stage_parser  # noqa: E402
from src.cascade.cost_e2e import (  # noqa: E402
    TIME_COLUMNS,
    clip_times,
    pose_breakeven,
    stratified_sample,
    summarize,
    time_curve,
)
from src.cascade.io import read_json, write_csv, write_json  # noqa: E402

STAGE = "41_cost_e2e"
R_GRID = [round(0.1 * i, 1) for i in range(11)]


class L0Timer:
    """L0 a batch 1: lettura della traiettoria, preprocessing delle pose, forward di CamFormer."""

    def __init__(self, cfg, num_classes: int, device) -> None:
        from src.cascade.models.camformer import build_model
        from src.data.trajectories import TrajectoryWindowDataset

        self.cfg, self.device = cfg, device
        self.model = build_model(cfg.camformer.arch, num_classes).eval().to(device)
        # dataset vuoto: serve solo a riusare ``_make`` (stessa finestra e stesso preprocessing dello stadio 20)
        self.ds = TrajectoryWindowDataset(pd.DataFrame(columns=["clip_id", "take_uid", "has_traj"]), cfg)

    def read_window(self, clip) -> dict:
        from src.cascade.trajectory import load_take_trajectory
        from src.data.trajectories import clip_meta

        tr = load_take_trajectory(Path(clip.traj_path), self.cfg, clip_meta(clip))  # senza cache: lettura vera
        half = self.ds.ctx_s / 2
        m = (tr.t_video >= clip.t_center - half) & (tr.t_video < clip.t_center + half)
        return {
            "clip_id": clip.clip_id,
            "label": int(clip.label_idx),
            "t": tr.t_video[m],
            "poses": tr.poses[m],
            "t_center": clip.t_center,
            "half_win": (clip.t_end - clip.t_start) / 2,
        }

    def preprocess(self, item: dict) -> dict:
        from src.cascade.models.camformer import collate

        batch = collate([self.ds._make(item, 0.0)])
        return {k: v.to(self.device) for k, v in batch.items() if hasattr(v, "to")}

    def forward(self, b: dict):
        import torch

        with torch.no_grad():
            return self.model(b["x"], b["pad_mask"], b["lengths"], b["win_start"], b["win_end"])


class L1Timer:
    """L1 a batch 1: apertura e decodifica dei frame usati, preprocessing, forward di TimeSformer."""

    def __init__(self, cfg, device) -> None:
        from src.cascade.models.egovlp import build_video_model

        self.cfg, self.device = cfg, device
        self.model = build_video_model(cfg.egovlp.arch).eval().to(device)
        self.sel = list(cfg.egovlp.frame_indices)

    def decode(self, clip):
        import numpy as np

        from src.cascade.frames import clip_frame_indices
        from src.cascade.video import decode_frames, probe_video

        f = self.cfg.frames
        path = Path(clip.video_path)
        info = probe_video(path)
        idx, _ = clip_frame_indices(info, clip.t_start, clip.t_end, f.n_frames, self.cfg.data.out_of_range)
        return decode_frames(path, [int(i) for i in np.asarray(idx)[self.sel]], f.decoder)  # solo i frame usati

    def preprocess(self, raw):
        from src.cascade.frames import preprocess_frames
        from src.cascade.models.egovlp import normalize

        f, e = self.cfg.frames, self.cfg.egovlp
        arr = preprocess_frames(
            raw,
            f.pre_resize,
            f.crop,
            f.size,
            f.rotate_k90,
            f.pre_resize_mode,
            f.pre_resize_antialias,
            f.final_antialias,
        )
        return normalize(arr[None], e.norm_mean, e.norm_std).to(self.device)

    def forward(self, x):
        import torch

        with torch.no_grad():
            return self.model(x)


def environment(cfg, device) -> dict:
    import torch

    env = {"device": str(device), "torch": torch.__version__, "n_cpu": os.cpu_count(), "decoder": cfg.frames.decoder}
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(device)
    try:
        from src.cascade import video

        env["decord_disponibile"] = bool(video._have_decord())
    except Exception:  # solo informativo
        pass
    return env


def main(argv: Optional[List[str]] = None) -> None:
    p = stage_parser(__doc__)
    args = p.parse_args(argv)
    ctx = open_stage(args, STAGE)
    out_times, out_summary = ctx.stage_dir / "clip_times.csv", ctx.stage_dir / "summary.json"
    outs = [out_times, out_summary]
    if ctx.done(outs):
        return
    ctx.require("00_inventory")

    import torch

    from src.cascade.frames import OutOfRange
    from src.cascade.models.camformer import resolve_device

    cfg, ce = ctx.cfg, ctx.cfg.cost_e2e
    clips = ctx.read_clips()
    clips = clips[clips.has_video & clips.has_traj].reset_index(drop=True)
    sample = stratified_sample(clips, ce.n_clips + ce.n_warmup, ce.seed)
    warm, timed_clips = sample.iloc[: ce.n_warmup], sample.iloc[ce.n_warmup :]
    if timed_clips.empty:
        raise RuntimeError(
            f"41: {len(clips)} clip con video e traiettoria, troppo poche per {ce.n_warmup} di riscaldamento"
        )
    device = resolve_device(cfg.cost.device)
    sync = torch.cuda.synchronize if device.type == "cuda" else None
    l0, l1 = L0Timer(cfg, ctx.labels.num_classes, device), L1Timer(cfg, device)
    for r in warm.itertuples(index=False):  # riscaldamento: modelli, allocatore, decoder
        try:
            clip_times(r, l0, l1, sync)
        except OutOfRange:
            pass
    rows, skipped = [], []
    for ps in range(1, ce.passes + 1):
        for r in timed_clips.itertuples(index=False):
            try:
                row = clip_times(r, l0, l1, sync)
            except OutOfRange:
                if ps == 1:
                    skipped.append(r.clip_id)
                continue
            row["pass"] = ps
            rows.append(row)
    if not rows:
        raise RuntimeError("41: nessuna clip misurata (tutte fuori dal video?)")
    times = pd.DataFrame(rows, columns=[*TIME_COLUMNS, "pass"])
    write_csv(times, out_times)
    last, first = times[times["pass"] == ce.passes], times[times["pass"] == 1]
    rep = summarize(last)
    summary = {
        "n_clips": int(len(last)),
        "n_warmup": int(len(warm)),
        "passes": ce.passes,
        "reported_pass": ce.passes,
        "skipped_out_of_range": skipped,
        "phases_reported": rep,
        "phases_first_pass": summarize(first),
        "weights": "casuali (il tempo non dipende dai valori dei pesi)",
        "env": environment(cfg, device),
    }
    t0, t1 = rep["l0_total_s"]["median_s"], rep["l1_total_s"]["median_s"]
    summary["time_curve"] = time_curve(t0, t1, R_GRID)
    summary["breakeven_pose_s"] = [{"r": r, "pose_s_max": pose_breakeven(r, t1, t0)} for r in R_GRID]
    cost_json = ctx.run_dir / "40_cost" / ctx.traj_tag / "cost.json"
    if cost_json.exists():  # gmacs_L0 comprende già il gate
        c = read_json(cost_json)
        summary["breakeven_pose_gmacs"] = [
            {"r": r, "pose_gmacs_max": pose_breakeven(r, c["gmacs_L1"], c["gmacs_L0"])} for r in R_GRID
        ]
    write_json(out_summary, summary)
    ctx.finish(outs)
    ctx.log.info("tempo mediano per clip: L0 %.4g s, L1 %.4g s (passata %d, %d clip)", t0, t1, ce.passes, len(last))


if __name__ == "__main__":
    main()
