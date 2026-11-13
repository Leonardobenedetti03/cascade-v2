"""Estrazione dei frame di una clip (stadio 10).

Per ogni clip: ``frames.n_frames`` istanti equispaziati nella finestra (punti medi di
n segmenti uguali, come il campionamento ``uniform`` di test di EgoVLP in
``base/base_dataset.py::sample_frames``), decodificati con decord o PyAV.

Preprocessing come quello di EgoVLP:

- i video di EgoVLP sono pre-ridimensionati a 256 px di altezza con ffmpeg
  (``utils/video_resize.py``: ``scale=...:256``, filtro bicubico antialiasato): qui
  ``Resize(256)`` sul lato corto con ``frames.pre_resize_mode`` e antialias;
- poi il transform di test (``data_loader/transforms.py::init_video_transform_dict['test']``):
  ``CenterCrop(256)`` → ``Resize(224)``. EgoVLP usa torchvision 0.13.1, dove ``Resize`` su
  tensori non applica antialias: ``frames.final_antialias = false``.

Il risultato si salva in uint8 ``[T, 224, 224, 3]``, un ``.npy`` per clip; la
normalizzazione ImageNet si applica al momento dell'inferenza (stadio 25).
Finestre oltre la fine del video: ``data.out_of_range`` (``drop`` scarta la clip,
``clamp`` limita gli istanti all'ultimo frame e lo registra).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.cascade.video import decode_frames, frame_indices_for_times, probe_video


def frame_times(t_start: float, t_end: float, n: int) -> np.ndarray:
    """Punti medi di ``n`` segmenti uguali di ``[t_start, t_end]``."""
    if n < 1 or t_end <= t_start:
        raise ValueError("n >= 1 e t_end > t_start")
    edges = np.linspace(t_start, t_end, n + 1)
    return (edges[:-1] + edges[1:]) / 2.0


def _resize_short(x, size: int, mode: str = "bilinear", antialias: bool = True):
    import torch.nn.functional as F

    h, w = x.shape[-2:]
    if h <= w:
        nh, nw = size, max(1, int(size * w / h))  # come torchvision: il lato lungo si tronca
    else:
        nh, nw = max(1, int(size * h / w)), size
    if (nh, nw) == (h, w):
        return x
    return F.interpolate(x, size=(nh, nw), mode=mode, align_corners=False, antialias=antialias)


def _center_crop(x, size: int):
    h, w = x.shape[-2:]
    top, left = int(round((h - size) / 2.0)), int(round((w - size) / 2.0))
    return x[..., top : top + size, left : left + size]


def preprocess_frames(
    frames: np.ndarray,
    pre_resize: int = 256,
    crop: int = 256,
    size: int = 224,
    rotate_k90: int = 0,
    pre_resize_mode: str = "bicubic",
    pre_resize_antialias: bool = True,
    final_antialias: bool = False,
) -> np.ndarray:
    """uint8 [T, H, W, 3] → uint8 [T, size, size, 3] (pipeline di test EgoVLP)."""
    import torch
    import torch.nn.functional as F

    if rotate_k90 % 4:
        frames = np.rot90(frames, k=rotate_k90, axes=(1, 2))
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
    x = _center_crop(_resize_short(x, pre_resize, pre_resize_mode, pre_resize_antialias), crop)
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=final_antialias)
    return (x.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).permute(0, 2, 3, 1).numpy()


class OutOfRange(ValueError):
    """La finestra della clip esce dal video e ``data.out_of_range = drop``."""


def clip_frame_indices(info, t_start: float, t_end: float, n_frames: int, policy: str = "drop"):
    """Indici dei frame della finestra; (indici, clamped). Tolleranza di un frame alla fine."""
    times = frame_times(t_start, t_end, n_frames)
    tol = 1.0 / info.fps if info.fps else 0.0
    outside = (times < -tol) | (times > info.duration_s + tol)
    if outside.any():
        if policy == "drop":
            raise OutOfRange(f"finestra [{t_start:.2f}, {t_end:.2f}] s oltre il video ({info.duration_s:.2f} s)")
        times = np.clip(times, 0.0, max(info.duration_s - 1e-3, 0.0))
    return frame_indices_for_times(times, info.fps, info.n_frames), bool(outside.any())


def extract_clip_frames(video_path: Path, t_start: float, t_end: float, fcfg, policy: str = "drop"):
    """(frame uint8 [T, size, size, 3], clamped)."""
    info = probe_video(video_path)
    idx, clamped = clip_frame_indices(info, t_start, t_end, fcfg.n_frames, policy)
    raw = decode_frames(video_path, idx, fcfg.decoder)
    out = preprocess_frames(
        raw,
        fcfg.pre_resize,
        fcfg.crop,
        fcfg.size,
        fcfg.rotate_k90,
        fcfg.pre_resize_mode,
        fcfg.pre_resize_antialias,
        fcfg.final_antialias,
    )
    return out, clamped


def save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    from src.cascade.io import save_npy

    save_npy(path, arr)
