"""L0: CamFormer per la classificazione dello scenario.

Encoder e preprocessing vengono importati dal submodule
``third_party/seeing_without_pixels`` (per percorso, senza toccare ``sys.path``,
così i loro moduli generici ``models``/``utils`` non collidono con altri pacchetti):

- ``models/cm_encoder.py::CameraPoseSeqEncoder`` con ``pooling='no'`` e
  ``output_dim=512``, come in ``tasks/pretrain_longseq.py``;
- ``utils/dataset_utils.py::transform_camera_pose`` (pose relative al frame centrale,
  9D, + gravità per ``rel9d_grav``).

Encoding contestualizzato: in ingresso c'è il contesto (``camformer.context_s``),
il mean-pool si fa solo sui token della finestra centrale ``[win_start, win_end)``
(come ``encode_motion`` del task longseq); sopra c'è una testa lineare a C classi.
Il fine-tuning è end-to-end (script di classificazione non pubblico: scritto qui).
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
    """Importa un file del submodule CamFormer con un nome univoco."""
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
    """``class_names`` di ``tasks/pretrain_longseq.py`` (ordine della testa ``cls_model``),
    letto con ``ast`` senza importare il modulo (che richiede clip e lightning)."""
    import ast

    tree = ast.parse((SWP_ROOT / "tasks" / "pretrain_longseq.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "class_names" for t in node.targets):
            return [ast.literal_eval(e) for e in node.value.elts]
    raise RuntimeError("class_names non trovato in tasks/pretrain_longseq.py")


def check_head_class_order(labels) -> None:
    """La testa del checkpoint usa l'ordine di CamFormer: deve coincidere con labels.yaml."""
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
    """Pose assolute [N, 7] → input di CamFormer, con la funzione originale."""
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

            feats = self.encoder(x, pad_mask, lengths)  # [B, T, D] senza gravità
            T = feats.shape[1]
            idx = torch.arange(T, device=feats.device)[None, :]
            m = ((idx >= win_start[:, None]) & (idx < win_end[:, None])).float()
            emb = (feats * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1.0)
            return self.head(emb), emb

    return CamFormerClassifier()


def load_pretrained(model, ckpt_path: Path, *, init_head: bool = False) -> Dict[str, list]:
    """Carica i pesi dell'encoder da un checkpoint CamFormer (prefisso ``model.`` rimosso).

    Come ``tasks/pretrain_longseq.py``: chiavi inattese ignorate (loggate), chiavi
    mancanti dell'encoder → errore (di solito ``pose_encoding`` sbagliato).
    Con ``init_head`` usa ``cls_model.{weight,bias}`` se presenti (stesso ordine classi).
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
    """Padding come ``loader.longseq_pose_text_collate`` (pad_mask True sul padding)."""
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
