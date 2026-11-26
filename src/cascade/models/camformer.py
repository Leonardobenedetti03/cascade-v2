"""Frozen CamFormer: one 512-d embedding per window (stage ``feat-camformer``).

Encoder and preprocessing come from the official repository (``camformer.repo``), imported by path:
``models/cm_encoder.py::CameraPoseSeqEncoder`` with ``pooling='no'`` and
``utils/dataset_utils.py::transform_camera_pose``, as in ``tasks/pretrain_longseq.py``.
The input is the context ``context_s`` centred on the window (poses relative to its central sample,
6D + translation, plus gravity in the camera frame). We then average the tokens of the window only.
The weights are loaded the way their code loads ``init_ckpt`` (``model.`` prefix removed, full encoder).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional

import numpy as np

from src.cascade.data.poses import Trajectory

log = logging.getLogger(__name__)
POSE_ENCODINGS = {"abs7d": 0, "rel9d": 2, "rel9d_grav": 11}
_MODS: Dict[str, ModuleType] = {}


def load_repo_module(repo: Path, rel: str) -> ModuleType:
    key = f"{repo}:{rel}"
    if key in _MODS:
        return _MODS[key]
    p = Path(repo) / rel
    if not p.is_file():
        raise FileNotFoundError(f"{p} mancante: clonare github.com/google-deepmind/seeing_without_pixels in camformer.repo")
    name = "swp_" + rel.replace("/", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODS[key] = mod
    return mod


def build_encoder(ccfg: dict, repo: Path):
    enc = load_repo_module(repo, "models/cm_encoder.py")
    return enc.CameraPoseSeqEncoder(
        d_model=ccfg["d_model"],
        encode_pose=POSE_ENCODINGS[ccfg["pose_encoding"]],
        output_dim=ccfg["output_dim"],
        nhead=ccfg["nhead"],
        num_layers=ccfg["num_layers"],
        dim_feedforward=ccfg["dim_feedforward"],
        dropout=ccfg["dropout"],
        pooling="no",
    )


def load_weights(model, ckpt: Path) -> None:
    import torch

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    sd = {k[len("model.") :] if k.startswith("model.") else k: v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    if res.missing_keys:
        raise RuntimeError(
            f"{ckpt}: mancano {len(res.missing_keys)} pesi dell'encoder (es. {res.missing_keys[:5]}): "
            "pose_encoding coerente con il checkpoint? (egoexo4d_dur8 = rel9d_grav)"
        )
    if res.unexpected_keys:
        log.info("%s: %d chiavi non dell'encoder ignorate (es. %s)", ckpt, len(res.unexpected_keys), res.unexpected_keys[:3])


def window_input(traj: Trajectory, t_start: float, t_end: float, context_s: float, min_coverage: float, transform, enc_id: int) -> Optional[dict]:
    """Model input for one window, or None if the window is not covered well enough.

    Context = ``[c - ctx/2, c + ctx/2)``, with c the centre of the window (never shorter than the window).
    The pooled tokens are the ones in ``[t_start, t_end)``.
    """
    c = 0.5 * (t_start + t_end)
    half = 0.5 * max(context_s, t_end - t_start)
    in_win = (traj.t >= t_start) & (traj.t < t_end)
    n_expected = max(1, int(round((t_end - t_start) * (1.0 / np.median(np.diff(traj.t))))))
    coverage = float((in_win & traj.valid).sum()) / n_expected
    if coverage < min_coverage:
        return None
    m = (traj.t >= c - half) & (traj.t < c + half)
    t, poses = traj.t[m], traj.poses[m]
    if len(t) < 2:
        return None
    idx = np.flatnonzero((t >= t_start) & (t < t_end))
    x = transform(np.asarray(poses, np.float64), enc_id).astype(np.float32)
    return {"x": x, "win_start": int(idx[0]), "win_end": int(idx[-1]) + 1, "coverage": coverage}


def embed(model, items: List[dict], batch_size: int, device) -> np.ndarray:
    """Embeddings ``[len(items), D]``: mean of the window tokens (like ``encode_motion``)."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            b = items[i : i + batch_size]
            seqs = [torch.from_numpy(it["x"]) for it in b]
            lengths = torch.tensor([len(s) for s in seqs])
            x = pad_sequence(seqs, batch_first=True).to(device)
            pad = (torch.arange(x.shape[1])[None, :] >= lengths[:, None]).to(device)
            feats = model(x, pad, lengths.to(device))  # [B, T, D], gravity token already removed
            for j, it in enumerate(b):
                out.append(feats[j, it["win_start"] : it["win_end"]].mean(0).cpu().numpy())
    return np.stack(out) if out else np.zeros((0, 0), np.float32)
