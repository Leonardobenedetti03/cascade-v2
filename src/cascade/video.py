"""Lettura video: probe dei metadati e decodifica di frame a tempi dati.

Decoder: ``decord`` se installato (più veloce sul random access), altrimenti PyAV.
I tempi sono in secondi dall'inizio del video (tempo della take).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    fps: float
    n_frames: int
    width: int
    height: int
    duration_s: float


def probe_video(path: Path) -> VideoInfo:
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate) if s.average_rate else float(s.guessed_rate or 0)
        n = int(s.frames or 0)
        if s.duration is not None and s.time_base is not None:
            dur = float(s.duration * s.time_base)
        elif c.duration is not None:
            dur = c.duration / 1e6
        else:
            dur = n / fps if fps else float("nan")
        if n == 0 and fps:
            n = int(round(dur * fps))
        return VideoInfo(fps=fps, n_frames=n, width=s.width, height=s.height, duration_s=dur)


def frame_indices_for_times(times_s: np.ndarray, fps: float, n_frames: int) -> np.ndarray:
    """Indice del frame che contiene ciascun istante (frame i copre [i/fps, (i+1)/fps))."""
    idx = np.floor(np.asarray(times_s, dtype=np.float64) * fps + 1e-6).astype(np.int64)
    return np.clip(idx, 0, max(n_frames - 1, 0))


def _have_decord() -> bool:
    try:
        import decord  # noqa: F401
    except ImportError:
        return False
    return True


def decode_frames(path: Path, indices: Sequence[int], decoder: str = "auto") -> np.ndarray:
    """Frame RGB uint8 ``[len(indices), H, W, 3]`` agli indici dati (ordine preservato)."""
    indices = [int(i) for i in indices]
    use = decoder
    if decoder == "auto":
        use = "decord" if _have_decord() else "pyav"
    if use == "decord":
        import decord

        vr = decord.VideoReader(str(path), ctx=decord.cpu(0))
        idx = np.clip(indices, 0, len(vr) - 1)
        return vr.get_batch(list(idx)).asnumpy()
    if use != "pyav":
        raise ValueError(f"decoder sconosciuto: {decoder}")
    return _decode_pyav(path, indices)


def _decode_pyav(path: Path, indices: Sequence[int]) -> np.ndarray:
    """Decodifica sequenziale con un seek iniziale: robusto anche con GOP lunghi."""
    import av

    wanted = sorted(set(indices))
    out = {}
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate)
        tb = s.time_base
        start_pts = s.start_time or 0
        first_t = wanted[0] / fps
        c.seek(int(start_pts + max(first_t - 1.0, 0) / tb), stream=s, backward=True, any_frame=False)
        last: Optional[np.ndarray] = None
        last_idx = -1
        it = iter(wanted)
        target = next(it)
        done = False
        for frame in c.decode(s):
            t = float((frame.pts - start_pts) * tb) if frame.pts is not None else (last_idx + 1) / fps
            idx = int(np.floor(t * fps + 1e-3))
            img = None
            while idx >= target:
                img = img if img is not None else frame.to_ndarray(format="rgb24")
                out[target] = img if idx == target or last is None else (last if idx > target else img)
                try:
                    target = next(it)
                except StopIteration:
                    done = True
                    break
            if done:
                break
            last = frame.to_ndarray(format="rgb24") if img is None else img
            last_idx = idx
        if not done:  # indici oltre la fine: ripete l'ultimo frame
            if last is None:
                raise RuntimeError(f"nessun frame decodificato da {path}")
            for t_idx in wanted:
                out.setdefault(t_idx, last)
    return np.stack([out[i] for i in indices])
