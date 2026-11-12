"""Allineamento tra il tempo della traiettoria e il tempo del video.

Unica funzione di conversione: :func:`traj_to_video_time` (e la sua inversa
:func:`video_to_traj_time`, definita a partire dagli stessi parametri).

Convenzioni:
- ``t_traj``: secondi dal primo campione della traiettoria (come ``relative_time``
  in ``seeing_without_pixels/data_prep/step1_extract_segments.py``);
- ``t_video``: secondi dal primo frame del video della take.

Schemi (``alignment.mode``):
- ``same_origin``: le due origini coincidono, ``t_video = scale * t_traj + offset_s``;
- ``device_us``: entrambe le origini sono note sull'orologio del device in µs
  (``traj_start_us``, ``video_start_us``):
  ``t_video = scale * (t_traj + (traj_start_us - video_start_us) / 1e6) + offset_s``.

TODO(HPC): verificare quale schema vale per Ego-Exo4D con ``15_alignment_check.py``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _origin_shift(mode: str, traj_start_us: Optional[float], video_start_us: Optional[float]) -> float:
    if mode == "same_origin":
        return 0.0
    if mode == "device_us":
        if (
            traj_start_us is None
            or video_start_us is None
            or not (np.isfinite(traj_start_us) and np.isfinite(video_start_us))
        ):
            raise ValueError("alignment.mode=device_us richiede traj_start_us e video_start_us")
        return (float(traj_start_us) - float(video_start_us)) / 1e6
    raise ValueError(f"alignment.mode sconosciuto: {mode!r}")


def traj_to_video_time(
    t_traj, cfg, traj_start_us: Optional[float] = None, video_start_us: Optional[float] = None
) -> np.ndarray:
    """Converte tempi della traiettoria in tempi del video. ``cfg`` è un AlignmentCfg."""
    if cfg.scale <= 0:
        raise ValueError("alignment.scale deve essere > 0")
    shift = _origin_shift(cfg.mode, traj_start_us, video_start_us)
    return cfg.scale * (np.asarray(t_traj, dtype=np.float64) + shift) + cfg.offset_s


def video_to_traj_time(
    t_video, cfg, traj_start_us: Optional[float] = None, video_start_us: Optional[float] = None
) -> np.ndarray:
    """Inversa esatta di :func:`traj_to_video_time`."""
    shift = _origin_shift(cfg.mode, traj_start_us, video_start_us)
    return (np.asarray(t_video, dtype=np.float64) - cfg.offset_s) / cfg.scale - shift
