"""Dataset torch delle finestre di traiettoria per CamFormer.

Per ogni clip si carica la traiettoria della take (cache npz per take), si taglia
il contesto ``[t_center − ctx/2 − jitter, t_center + ctx/2 + jitter]`` (tempo video)
e si ricordano gli indici dei token della finestra centrale ``[t_start, t_end)``.
Il preprocessing (``rel9d_grav``) è quello di CamFormer, applicato per campione
(dipende dal frame centrale del contesto). Le clip con copertura della finestra
inferiore a ``data.min_traj_coverage`` vengono scartate e riportate in ``dropped``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.cascade.models.camformer import preprocess_poses
from src.cascade.trajectory import TakeTrajectory, cached_take_trajectory, slice_window

log = logging.getLogger(__name__)


def clip_meta(row) -> dict:
    out = {}
    for k in ("video_start_us", "traj_start_us"):
        v = getattr(row, k, np.nan)
        if v is not None and np.isfinite(v):
            out[k] = float(v)
    return out


class TrajectoryWindowDataset:
    """Finestre di traiettoria per CamFormer.

    ``prefix_frac < 1`` (variante causale, stadio 22): la finestra si ferma a
    ``t_start + prefix_frac · D`` e dal contesto si tolgono i campioni successivi, così l'input
    non vede mai il futuro (il contesto passato, se ``context_s > D``, resta).
    """

    def __init__(
        self,
        clips: pd.DataFrame,
        cfg,
        *,
        cache_dir: Optional[Path] = None,
        train: bool = False,
        prefix_frac: float = 1.0,
    ) -> None:
        if not 0 < prefix_frac <= 1:
            raise ValueError(f"prefix_frac in (0, 1], trovato {prefix_frac}")
        self.cfg = cfg
        self.train = train
        self.prefix_frac = float(prefix_frac)
        self.fps = cfg.data.traj_fps
        self.ctx_s = cfg.camformer.context_s or cfg.data.window_s
        if self.ctx_s < cfg.data.window_s:
            raise ValueError("camformer.context_s deve essere >= data.window_s")
        self.jitter = cfg.camformer.train.temporal_jitter_s if train else 0.0
        self.encoding = cfg.camformer.arch.pose_encoding
        self.items: List[Dict] = []
        self.dropped: List[Dict] = []
        trajs: Dict[str, TakeTrajectory] = {}
        for r in clips.itertuples(index=False):
            if not r.has_traj:
                self.dropped.append({"clip_id": r.clip_id, "reason": "no_traj"})
                continue
            tr = trajs.get(r.take_uid)
            if tr is None:
                tr = cached_take_trajectory(Path(r.traj_path), cfg, clip_meta(r), cache_dir)
                trajs = {r.take_uid: tr}  # tiene in memoria solo la take corrente
            _, cov = slice_window(tr, r.t_start, r.t_end)
            if cov < cfg.data.min_traj_coverage:
                self.dropped.append({"clip_id": r.clip_id, "reason": f"coverage={cov:.2f}"})
                continue
            half = self.ctx_s / 2 + self.jitter
            m = (tr.t_video >= r.t_center - half) & (tr.t_video < r.t_center + half)
            self.items.append(
                {
                    "clip_id": r.clip_id,
                    "label": int(r.label_idx),
                    "t": tr.t_video[m],
                    "poses": tr.poses[m],
                    "t_center": r.t_center,
                    "half_win": (r.t_end - r.t_start) / 2,
                }
            )
        if self.dropped:
            log.warning("finestre di traiettoria scartate: %d (es. %s)", len(self.dropped), self.dropped[:3])
        self._pre: Dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.items)

    @property
    def clip_ids(self) -> List[str]:
        return [it["clip_id"] for it in self.items]

    @property
    def labels(self) -> np.ndarray:
        return np.array([it["label"] for it in self.items], dtype=int)

    def _make(self, it: dict, shift: float) -> dict:
        c = it["t_center"] + shift
        w_lo, hi = c - it["half_win"], c + self.ctx_s / 2
        w_hi = c + it["half_win"]
        if self.prefix_frac < 1:  # causale: niente campioni dopo il taglio
            w_hi = w_lo + self.prefix_frac * 2 * it["half_win"]
            hi = min(hi, w_hi)
        m = (it["t"] >= c - self.ctx_s / 2) & (it["t"] < hi)
        t, poses = it["t"][m], it["poses"][m]
        if len(poses) < 2 and self.prefix_frac >= 1:
            t, poses = it["t"], it["poses"]
        elif len(poses) < 2:  # prefisso troppo corto: tutto il passato disponibile
            past = it["t"] < hi
            t, poses = (it["t"][past], it["poses"][past]) if past.sum() >= 2 else (it["t"][:2], it["poses"][:2])
        w = (t >= w_lo) & (t < w_hi)
        idx = np.flatnonzero(w)
        ws, we = (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, len(t))
        return {
            "x": preprocess_poses(poses, self.encoding).astype(np.float32),
            "win_start": ws,
            "win_end": we,
            "label": it["label"],
            "clip_id": it["clip_id"],
        }

    def __getitem__(self, i: int) -> dict:
        it = self.items[i]
        if self.jitter > 0:
            return self._make(it, float(np.random.uniform(-self.jitter, self.jitter)))
        if i not in self._pre:
            self._pre[i] = self._make(it, 0.0)
        return self._pre[i]
