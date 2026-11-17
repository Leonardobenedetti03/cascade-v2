"""CamFormer congelato: embedding a 512 per finestra (stadio ``feat-camformer``).

Encoder e preprocessing vengono dal repository ufficiale (``camformer.repo``), importati per percorso:
``models/cm_encoder.py::CameraPoseSeqEncoder`` con ``pooling='no'`` e
``utils/dataset_utils.py::transform_camera_pose``, come ``tasks/pretrain_longseq.py``:
in ingresso il contesto ``context_s`` centrato sulla finestra (pose relative al suo campione centrale,
6D + traslazione, più la gravità nel frame della camera), poi media dei token della sola finestra.
I pesi si caricano come ``init_ckpt`` del loro codice (prefisso ``model.`` tolto, encoder completo).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional

import numpy as np

from scenario.poses import Trajectory

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
    """Input del modello per una finestra, o None se la copertura della finestra è insufficiente.

    Contesto = ``[c - ctx/2, c + ctx/2)`` con c centro della finestra (mai più corto della finestra);
    i token di pooling sono quelli in ``[t_start, t_end)``.
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
    """Embedding ``[len(items), D]``: media dei token della finestra (come ``encode_motion``)."""
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
            feats = model(x, pad, lengths.to(device))  # [B, T, D], token di gravità già tolto
            for j, it in enumerate(b):
                out.append(feats[j, it["win_start"] : it["win_end"]].mean(0).cpu().numpy())
    return np.stack(out) if out else np.zeros((0, 0), np.float32)
