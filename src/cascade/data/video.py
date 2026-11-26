"""Frame decoding and preprocessing for EgoVLP (from ``frames.py`` in cascade v2).

Time points: the midpoints of ``n_frames`` equal segments of the window (EgoVLP's ``uniform`` test sampling).
Only ``frame_indices`` (4 of the 16) go to the model, so we decode only those.
EgoVLP's test preprocessing: Resize(256) on the short side (bicubic with antialias, like the ffmpeg resize of
their videos), CenterCrop(256), Resize(224) without antialias (torchvision 0.13).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def frame_times(t_start: float, t_end: float, n: int) -> np.ndarray:
    edges = np.linspace(t_start, t_end, n + 1)
    return (edges[:-1] + edges[1:]) / 2.0


def decode_at(path: str, times: Sequence[float]) -> np.ndarray:
    """RGB uint8 frames ``[len(times), H, W, 3]`` at the given times (nearest frame); decord if present, else PyAV."""
    try:
        import decord

        vr = decord.VideoReader(path, num_threads=1)
        fps = float(vr.get_avg_fps())
        idx = np.clip(np.round(np.asarray(times) * fps).astype(int), 0, len(vr) - 1)
        return vr.get_batch(list(idx)).asnumpy()
    except ImportError:
        return _decode_pyav(path, times)


def _decode_pyav(path: str, times: Sequence[float]) -> np.ndarray:
    import av

    out: List[np.ndarray] = []
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        fps = float(s.average_rate)
        tb = float(s.time_base)
        for t in times:
            target = int(round(t * fps))  # index of the frame we want
            c.seek(int(max(t - 1.0, 0) / tb), stream=s, backward=True, any_frame=False)
            last = None
            for fr in c.decode(s):
                k = int(round(fr.pts * tb * fps)) if fr.pts is not None else None
                last = fr
                if k is not None and k >= target:
                    break
            if last is None:
                raise RuntimeError(f"{path}: nessun frame a t={t:.2f}s")
            out.append(last.to_ndarray(format="rgb24"))
    return np.stack(out)


def preprocess(frames: np.ndarray, pre_resize: int = 256, crop: int = 256, size: int = 224) -> np.ndarray:
    """uint8 ``[T, H, W, 3]`` -> uint8 ``[T, size, size, 3]``."""
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
    h, w = x.shape[-2:]
    nh, nw = (pre_resize, max(1, int(pre_resize * w / h))) if h <= w else (max(1, int(pre_resize * h / w)), pre_resize)
    x = F.interpolate(x, size=(nh, nw), mode="bicubic", align_corners=False, antialias=True)
    top, left = int(round((nh - crop) / 2.0)), int(round((nw - crop) / 2.0))
    x = x[..., top : top + crop, left : left + crop]
    if crop != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=False)
    return (x.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).permute(0, 2, 3, 1).numpy()
