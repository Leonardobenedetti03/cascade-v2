"""L0: CamFormer for scenario classification.

Encoder and preprocessing are imported from the submodule ``third_party/seeing_without_pixels``, by path and
without touching ``sys.path``, so their generic ``models``/``utils`` modules do not clash with other packages:

- ``models/cm_encoder.py::CameraPoseSeqEncoder`` with ``pooling='no'`` and ``output_dim=512``, as in
  ``tasks/pretrain_longseq.py``;
- ``utils/dataset_utils.py::transform_camera_pose`` (poses relative to the central frame, 9D, + gravity for
  ``rel9d_grav``).

Contextual encoding: the input is the context (``camformer.context_s``), and the mean pool runs only over the
tokens of the central window ``[win_start, win_end)`` (like ``encode_motion`` in the longseq task). A linear
head with C classes sits on top. Fine-tuning is end-to-end. The authors did not release their classification
script, so we wrote it here.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Dict, Tuple

import numpy as np

from src import REPO_ROOT

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*nested tensors is in prototype.*")
SWP_ROOT = REPO_ROOT / "third_party" / "seeing_without_pixels"
POSE_ENCODINGS = {"abs7d": 0, "rel9d": 2, "rel9d_grav": 11}
INPUT_DIM = {0: 7, 2: 9, 11: 12}
_MODULES: Dict[str, ModuleType] = {}


def load_swp_module(rel_path: str) -> ModuleType:
    """Import a file of the CamFormer submodule under a unique name."""
    if rel_path in _MODULES:
        return _MODULES[rel_path]
    path = SWP_ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(f"{path} mancante: git submodule update --init third_party/seeing_without_pixels")
    name = "swp_" + rel_path.replace("/", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODULES[rel_path] = mod
    return mod


def camformer_class_names() -> list:
    """``class_names`` of ``tasks/pretrain_longseq.py`` (the order of the ``cls_model`` head), read with ``ast``
    without importing the module (which needs clip and lightning)."""
    import ast

    tree = ast.parse((SWP_ROOT / "tasks" / "pretrain_longseq.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "class_names" for t in node.targets):
            return [ast.literal_eval(e) for e in node.value.elts]
    raise RuntimeError("class_names non trovato in tasks/pretrain_longseq.py")


def check_head_class_order(labels) -> None:
    """The checkpoint head uses CamFormer's class order, which must match labels.yaml."""
    theirs = camformer_class_names()
    if list(labels.csv_tasks) != theirs:
        raise ValueError(
            f"ordine classi di labels.yaml {list(labels.csv_tasks)} diverso da quello della testa "
            f"di CamFormer {theirs}: init_head_from_ckpt mischierebbe le classi"
        )


def encode_pose_id(name: str) -> int:
    try:
        return POSE_ENCODINGS[name]
    except KeyError:
        raise ValueError(f"pose_encoding sconosciuto: {name!r} ({list(POSE_ENCODINGS)})") from None


def preprocess_poses(poses: np.ndarray, pose_encoding: str = "rel9d_grav") -> np.ndarray:
    """Absolute poses [N, 7] → CamFormer input, with the original function."""
    du = load_swp_module("utils/dataset_utils.py")
    return du.transform_camera_pose(np.asarray(poses, dtype=np.float64), encode_pose_id(pose_encoding))


def build_model(arch, num_classes: int):
    import torch.nn as nn

    enc_mod = load_swp_module("models/cm_encoder.py")

    class CamFormerClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = enc_mod.CameraPoseSeqEncoder(
                d_model=arch.d_model,
                encode_pose=encode_pose_id(arch.pose_encoding),
                output_dim=arch.output_dim,
                nhead=arch.nhead,
                num_layers=arch.num_layers,
                dim_feedforward=arch.dim_feedforward,
                dropout=arch.dropout,
                pooling="no",
            )
            self.head = nn.Linear(arch.output_dim, num_classes)

        def forward(self, x, pad_mask, lengths, win_start, win_end) -> Tuple:
            import torch

            feats = self.encoder(x, pad_mask, lengths)  # [B, T, D] without gravity
            T = feats.shape[1]
            idx = torch.arange(T, device=feats.device)[None, :]
            m = ((idx >= win_start[:, None]) & (idx < win_end[:, None])).float()
            emb = (feats * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1.0)
            return self.head(emb), emb

    return CamFormerClassifier()


def load_pretrained(model, ckpt_path: Path, *, init_head: bool = False) -> Dict[str, list]:
    """Load the encoder weights from a CamFormer checkpoint (``model.`` prefix removed).

    As in ``tasks/pretrain_longseq.py``: unexpected keys are ignored (and logged), missing encoder keys are an
    error (usually a wrong ``pose_encoding``). With ``init_head`` it uses ``cls_model.{weight,bias}`` if they are
    there (same class order).
    """
    import torch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    enc_sd = {k[len("model.") :] if k.startswith("model.") else k: v for k, v in sd.items()}
    res = model.encoder.load_state_dict(enc_sd, strict=False)
    if res.missing_keys:
        raise RuntimeError(
            f"checkpoint {ckpt_path}: mancano {len(res.missing_keys)} pesi dell'encoder "
            f"(es. {res.missing_keys[:5]}); pose_encoding coerente col checkpoint?"
        )
    unexpected = [k for k in res.unexpected_keys]
    if unexpected:
        log.info(
            "checkpoint %s: %d chiavi non dell'encoder ignorate (es. %s)", ckpt_path, len(unexpected), unexpected[:3]
        )
    head_loaded = False
    if init_head and "cls_model.weight" in sd:
        w, b = sd["cls_model.weight"], sd["cls_model.bias"]
        if tuple(w.shape) == tuple(model.head.weight.shape):
            model.head.weight.data.copy_(w)
            model.head.bias.data.copy_(b)
            head_loaded = True
        else:
            log.warning("cls_model nel checkpoint con shape %s, testa nuova", tuple(w.shape))
    return {"missing": res.missing_keys, "unexpected": unexpected, "head_loaded": [head_loaded]}


def collate(batch):
    """Padding like ``loader.longseq_pose_text_collate`` (pad_mask is True on the padding)."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    seqs = [torch.as_tensor(b["x"], dtype=torch.float32) for b in batch]
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    x = pad_sequence(seqs, batch_first=True)
    pad_mask = torch.arange(x.shape[1])[None, :] >= lengths[:, None]
    return {
        "x": x,
        "pad_mask": pad_mask,
        "lengths": lengths,
        "win_start": torch.tensor([b["win_start"] for b in batch], dtype=torch.long),
        "win_end": torch.tensor([b["win_end"] for b in batch], dtype=torch.long),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "clip_id": [b["clip_id"] for b in batch],
    }


def resolve_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
