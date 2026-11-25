"""EgoVLP congelato: CLS a 768 per finestra (stadio ``feat-egovlp``).

Ramo video di ``EgoVLP/model/model.py::FrozenInTime``: SpaceTimeTransformer (copia in ``vendor/``,
identica a quella di cascade v2) con head/pre_logits/fc = Identity; i nomi coincidono con quelli del
checkpoint, che si carica dopo aver tolto ``module.``. Ingresso ``[B, T, 3, H, W]`` normalizzato ImageNet.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

log = logging.getLogger(__name__)


def build_model(ecfg: dict):
    import torch.nn as nn

    from src.cascade.models.vendor.egovlp_video_transformer import SpaceTimeTransformer

    class EgoVLPVideo(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            vm = SpaceTimeTransformer(
                img_size=ecfg["img_size"],
                patch_size=ecfg["patch_size"],
                embed_dim=ecfg["embed_dim"],
                depth=ecfg["depth"],
                num_heads=ecfg["num_heads"],
                num_frames=ecfg["num_frames"],
                time_init="zeros",
                attention_style="frozen-in-time",
            )
            vm.head = nn.Identity()
            vm.pre_logits = nn.Identity()
            vm.fc = nn.Identity()
            self.video_model = vm
            self.vid_proj = nn.Sequential(nn.Linear(ecfg["embed_dim"], ecfg["proj_dim"]))

        def forward(self, x):
            return self.video_model(x)  # CLS [B, 768], prima della proiezione

    return EgoVLPVideo()


def _tolerant_pickle():
    """``pickle`` che sostituisce le classi non importabili (es. ``parse_config.ConfigParser`` salvato
    nel checkpoint EgoVLP) con un segnaposto: serve solo lo ``state_dict``."""
    import pickle
    import types

    class _Missing:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            self.__dict__["_state"] = state

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, AttributeError):
                return _Missing

    mod = types.ModuleType("tolerant_pickle")
    mod.__dict__.update({k: getattr(pickle, k) for k in dir(pickle) if not k.startswith("__")})
    mod.Unpickler = _Unpickler
    return mod


def load_weights(model, ckpt: Path) -> None:
    import torch

    ck = torch.load(ckpt, map_location="cpu", weights_only=False, pickle_module=_tolerant_pickle())
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    sd = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}
    own = model.state_dict()
    key = "video_model.temporal_embed"
    if key in sd and key in own and sd[key].shape != own[key].shape:  # come _inflate_positional_embeds ('zeros')
        loaded, cur = sd[key], own[key]
        new = torch.zeros_like(cur)
        n = min(loaded.shape[1], cur.shape[1])
        new[:, :n] = loaded[:, :n]
        sd[key] = new
    sub = {k: v for k, v in sd.items() if k.startswith(("video_model.", "vid_proj."))}
    res = model.load_state_dict(sub, strict=False)
    if res.missing_keys:
        raise RuntimeError(f"{ckpt}: mancano {len(res.missing_keys)} chiavi del modello video (es. {res.missing_keys[:5]})")
    log.info("EgoVLP: %d chiavi video caricate, %d altre ignorate", len(sub), len(sd) - len(sub))


def normalize(frames_uint8: np.ndarray, mean: Sequence[float], std: Sequence[float]):
    """uint8 ``[B, T, H, W, 3]`` -> float ``[B, T, 3, H, W]`` normalizzato."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(frames_uint8)).float().div_(255.0).permute(0, 1, 4, 2, 3)
    m = torch.tensor(mean).view(1, 1, 3, 1, 1)
    s = torch.tensor(std).view(1, 1, 3, 1, 1)
    return (x - m) / s


class WindowFrames:
    """Dataset torch: una voce per (take, finestra), frame già preprocessati ``[4, 224, 224, 3]``."""

    def __init__(self, items: List[Dict], ecfg: dict) -> None:
        self.items = items
        self.ecfg = ecfg

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict:
        from src.cascade.data.video import decode_at, frame_times, preprocess

        it, e = self.items[i], self.ecfg
        times = frame_times(it["t_start"], it["t_end"], e["n_frames"])[list(e["frame_indices"])]
        try:
            raw = decode_at(it["video_path"], times)
            fr = preprocess(raw, e["pre_resize"], e["crop"], e["size"])
            ok = True
        except Exception as ex:  # noqa: BLE001 - una clip illeggibile non ferma lo stadio
            log.warning("%s: decodifica fallita (%s)", it["video_path"], ex)
            fr = np.zeros((len(e["frame_indices"]), e["size"], e["size"], 3), np.uint8)
            ok = False
        return {"frames": fr, "ok": ok, "i": i}


def collate(batch):
    return {
        "frames": np.stack([b["frames"] for b in batch]),
        "ok": np.array([b["ok"] for b in batch]),
        "i": np.array([b["i"] for b in batch]),
    }
