"""Caricamento delle traiettorie Aria e taglio delle finestre.

Sorgenti (``data.traj_source``):
- ``aria_csv``: ``closed_loop_trajectory.csv`` di Ego-Exo4D, take intera (1 kHz),
  colonne ``tracking_timestamp_us`` e ``{t,q}{x,y,z(,w)}_world_device`` — le stesse
  lette da ``seeing_without_pixels/data_prep/step1_extract_segments.py``;
- ``converted_npz``: uscita dello stadio 01 (``src/cascade/traj_convert.py``, M08), un
  ``<take_name>.npz`` con ``t``, ``poses``, ``valid`` e ``fps`` già su griglia uniforme nel
  tempo del video: non si ricampiona e richiede ``alignment.mode: same_origin``;
- ``camformer_npz``: rilascio CamFormer, un ``<take>.npz`` con segmenti attorno alle
  narrazioni (chiave ``<take>_<start>_<end>``, pose (N,7) a passo 50 ms da
  ``start``, vedi ``data_prep/step3_build_longseq.py``). I segmenti vengono cuciti:
  la copertura della take è parziale.

Con ``aria_csv`` e ``camformer_npz`` la traiettoria viene ricampionata su una griglia uniforme a
``data.traj_fps`` (campione più vicino, come lo stride 50 di CamFormer) e portata
nel tempo del video con :func:`src.cascade.alignment.traj_to_video_time`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.cascade.alignment import traj_to_video_time
from src.cascade.io import save_npz

log = logging.getLogger(__name__)

POSE_COLUMNS = [
    "tx_world_device",
    "ty_world_device",
    "tz_world_device",
    "qx_world_device",
    "qy_world_device",
    "qz_world_device",
    "qw_world_device",
]
TIME_COLUMN = "tracking_timestamp_us"
NPZ_STEP_S = 0.05


@dataclass
class TakeTrajectory:
    t_video: np.ndarray  # [N] secondi nel tempo del video, griglia uniforme
    poses: np.ndarray  # [N, 7] tx, ty, tz, qx, qy, qz, qw (mondo)
    valid: np.ndarray  # [N] bool: c'è un campione vicino (gap nei segmenti npz)

    @property
    def fps(self) -> float:
        return 1.0 / float(np.median(np.diff(self.t_video))) if len(self.t_video) > 1 else float("nan")


def load_aria_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """(t_traj [s dal primo campione], pose [N,7], t0_us)."""
    import pandas as pd

    df = pd.read_csv(path, usecols=[TIME_COLUMN] + POSE_COLUMNS)
    df = df.dropna()
    ts = df[TIME_COLUMN].to_numpy(np.float64)
    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    poses = df[POSE_COLUMNS].to_numpy(np.float64)[order]
    return (ts - ts[0]) / 1e6, poses, float(ts[0])


def load_camformer_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Cuce i segmenti di un npz CamFormer: (t_traj, pose) ordinati per tempo."""
    data = np.load(path)
    ts, ps = [], []
    for key in data.files:
        try:
            start = float(key.rsplit("_", 2)[-2])
        except ValueError:
            log.warning("chiave npz non riconosciuta: %s", key)
            continue
        seg = data[key]
        ts.append(start + NPZ_STEP_S * np.arange(len(seg)))
        ps.append(seg)
    if not ts:
        raise ValueError(f"nessun segmento in {path}")
    t = np.concatenate(ts)
    p = np.concatenate(ps)
    order = np.argsort(t, kind="stable")
    return t[order], p[order]


def resample_nearest(
    t: np.ndarray, poses: np.ndarray, fps: float, max_gap_s: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ricampiona su griglia uniforme a ``fps`` col campione più vicino."""
    grid = np.arange(t[0], t[-1] + 1e-9, 1.0 / fps)
    j = np.searchsorted(t, grid)
    j = np.clip(j, 1, len(t) - 1)
    left_closer = (grid - t[j - 1]) <= (t[j] - grid)
    j = np.where(left_closer, j - 1, j)
    gap = np.abs(t[j] - grid)
    valid = gap <= (max_gap_s if max_gap_s is not None else 1.0 / fps)
    out = poses[j].copy()
    q = out[:, 3:7]
    out[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
    return grid, out, valid


def load_converted_npz(path: Path, cfg) -> TakeTrajectory:
    """Traiettoria dello stadio 01 (M08): griglia uniforme già nel tempo del video, ``valid`` esplicito."""
    if cfg.alignment.mode != "same_origin":
        raise ValueError(
            "data.traj_source=converted_npz: il tempo è già quello del video, serve alignment.mode=same_origin"
        )
    with np.load(path) as z:
        t, poses, valid, fps = z["t"], z["poses"], z["valid"].astype(bool), float(z["fps"])
    if abs(fps - cfg.data.traj_fps) > 1e-6:
        raise ValueError(f"{path}: convertita a {fps:g} Hz ma data.traj_fps={cfg.data.traj_fps:g}: rifare lo stadio 01")
    return TakeTrajectory(t_video=traj_to_video_time(t, cfg.alignment), poses=poses, valid=valid)


def load_take_trajectory(path: Path, cfg, meta: Optional[dict] = None) -> TakeTrajectory:
    """Traiettoria della take nel tempo del video. ``meta`` fornisce gli start in µs."""
    meta = meta or {}
    src = cfg.data.traj_source
    if src == "converted_npz":
        return load_converted_npz(path, cfg)
    traj_start = meta.get(cfg.alignment.traj_start_key)
    if src == "aria_csv":
        t, poses, t0_us = load_aria_csv(path)
        traj_start = t0_us  # i timestamp del CSV sono già sull'orologio del device
    elif src == "camformer_npz":
        t, poses = load_camformer_npz(path)
    else:
        raise ValueError(f"data.traj_source sconosciuto: {src!r}")
    grid, rs, valid = resample_nearest(t, poses, cfg.data.traj_fps)
    tv = traj_to_video_time(grid, cfg.alignment, traj_start, meta.get(cfg.alignment.video_start_key))
    return TakeTrajectory(t_video=tv, poses=rs, valid=valid)


def slice_window(traj: TakeTrajectory, t_start: float, t_end: float) -> Tuple[np.ndarray, float]:
    """Pose nella finestra ``[t_start, t_end)`` (tempo video) e frazione coperta."""
    m = (traj.t_video >= t_start - 1e-9) & (traj.t_video < t_end - 1e-9)
    expected = max(int(round((t_end - t_start) * traj.fps)), 1) if np.isfinite(traj.fps) else 1
    coverage = float((m & traj.valid).sum()) / expected
    return traj.poses[m], min(coverage, 1.0)


def cached_take_trajectory(path: Path, cfg, meta: dict, cache_dir: Optional[Path]) -> TakeTrajectory:
    """Come :func:`load_take_trajectory`, con cache npz per take (chiave = sorgente+allineamento)."""
    if cache_dir is None or cfg.data.traj_source == "converted_npz":  # l'npz dello stadio 01 è già la cache
        return load_take_trajectory(path, cfg, meta)
    a = cfg.alignment
    tag = f"{cfg.data.traj_source}_{cfg.data.traj_fps:g}_{a.mode}_{a.offset_s:g}_{a.scale:g}"
    fp = cache_dir / tag / f"{Path(path).parent.name}__{Path(path).stem}__{_meta_tag(meta)}.npz"
    if fp.exists():
        try:
            with np.load(fp) as z:
                return TakeTrajectory(z["t_video"], z["poses"], z["valid"])
        except Exception as e:  # noqa: BLE001 - file troncato da un job interrotto: si ricalcola
            log.warning("cache traiettoria illeggibile (%s), la ricalcolo: %s", e, fp)
    tr = load_take_trajectory(path, cfg, meta)
    # scrittura atomica con nome temporaneo univoco: i job dell'array dello stadio 20
    # possono scrivere la stessa take in parallelo senza corrompersi a vicenda
    save_npz(fp, t_video=tr.t_video, poses=tr.poses, valid=tr.valid)
    return tr


def _meta_tag(meta: dict) -> str:
    """Gli start in µs entrano nella chiave della cache (cambiano l'allineamento)."""
    v = [meta.get(k) for k in sorted(meta)]
    return (
        "na" if not v else "_".join(f"{x:.0f}" if isinstance(x, (int, float)) and np.isfinite(x) else "na" for x in v)
    )
