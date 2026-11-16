"""L1: TimeSformer (SpaceTimeTransformer) con pesi EgoVLP, backbone congelato.

Struttura come ``EgoVLP/model/model.py::FrozenInTime`` ristretta al ramo video:
``video_model`` (ViT-B/16, CLS a 768, ``head``/``pre_logits``/``fc`` = Identity) e
``vid_proj = Sequential(Linear(768, 256))``. I nomi dei moduli coincidono con quelli
del checkpoint, quindi lo ``state_dict`` si carica dopo aver tolto ``module.``.

Input e normalizzazione (da ``data_loader/transforms.py`` e ``model.forward``):
frame float in [0, 1], normalizzati con media/std ImageNet per canale, tensore
``[B, T, C, H, W]`` (T = 4 frame a 224 px nello zero-shot / pretraining).

Zero-shot: testo DistilBERT, ``last_hidden_state[:, 0]`` → ``txt_proj = ReLU, Linear(768, 256)``
(``compute_text``); logit = similarità coseno video–testo / temperatura.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

log = logging.getLogger(__name__)


def build_video_model(arch):
    import torch.nn as nn

    from src.cascade.models.vendor.egovlp_video_transformer import SpaceTimeTransformer

    class EgoVLPVideo(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            vm = SpaceTimeTransformer(
                img_size=arch.img_size,
                patch_size=arch.patch_size,
                embed_dim=arch.embed_dim,
                depth=arch.depth,
                num_heads=arch.num_heads,
                num_frames=arch.num_frames,
                time_init="zeros",
                attention_style="frozen-in-time",
            )
            vm.head = nn.Identity()
            vm.pre_logits = nn.Identity()
            vm.fc = nn.Identity()
            self.video_model = vm
            self.vid_proj = nn.Sequential(nn.Linear(arch.embed_dim, arch.proj_dim))

        def forward(self, x):
            cls = self.video_model(x)  # [B, 768] pre-proiezione
            return cls, self.vid_proj(cls)

    return EgoVLPVideo()


def _strip_module(sd: Dict) -> Dict:
    return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}


def _tolerant_pickle():
    """Modulo ``pickle`` il cui Unpickler sostituisce le classi non importabili con un segnaposto.

    Il checkpoint EgoVLP_PT_BEST salva anche la config di addestramento come oggetto
    ``parse_config.ConfigParser`` del repository EgoVLP: senza quel modulo nel path ``torch.load``
    fallisce, anche se a noi serve solo ``state_dict``. Il segnaposto non viene mai usato.
    """
    import pickle
    import types

    class _Missing:
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            self.__dict__["_state"] = state

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, AttributeError):
                log.info("checkpoint: classe %s.%s non importabile, sostituita da un segnaposto", module, name)
                return _Missing

    mod = types.ModuleType("tolerant_pickle")
    mod.__dict__.update({k: getattr(pickle, k) for k in dir(pickle) if not k.startswith("__")})
    mod.Unpickler = _Unpickler
    return mod


def load_state(ckpt_path: Path) -> Dict:
    import torch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False, pickle_module=_tolerant_pickle())
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    return _strip_module(sd)


def load_video_weights(model, sd: Dict) -> Dict[str, List[str]]:
    """Carica i pesi video (strict=False) con log delle chiavi; errore se manca una chiave video.

    ``temporal_embed`` con un numero diverso di frame viene troncato o completato con
    zeri, come ``FrozenInTime._inflate_positional_embeds`` (``load_temporal_fix='zeros'``).
    """
    import torch

    own = model.state_dict()
    sd = dict(sd)
    key = "video_model.temporal_embed"
    if key in sd and key in own and sd[key].shape != own[key].shape:
        loaded, cur = sd[key], own[key]
        n_cur = cur.shape[1]
        if loaded.shape[1] >= n_cur:
            sd[key] = loaded[:, :n_cur]
        else:
            new = torch.zeros_like(cur)
            new[:, : loaded.shape[1]] = loaded
            sd[key] = new
        log.info("temporal_embed adattato da %d a %d frame", loaded.shape[1], n_cur)
    sub = {k: v for k, v in sd.items() if k.startswith(("video_model.", "vid_proj."))}
    res = model.load_state_dict(sub, strict=False)
    other = sorted(set(sd) - set(sub))
    log.info(
        "pesi EgoVLP: %d chiavi video caricate, %d chiavi non video ignorate (es. %s)", len(sub), len(other), other[:3]
    )
    if res.unexpected_keys:
        log.warning("chiavi inattese: %s", res.unexpected_keys[:10])
    if res.missing_keys:
        raise RuntimeError(
            f"pesi EgoVLP: mancano {len(res.missing_keys)} chiavi del modello video " f"(es. {res.missing_keys[:5]})"
        )
    return {"missing": res.missing_keys, "unexpected": res.unexpected_keys, "ignored": other}


def normalize(frames_uint8: np.ndarray, mean: Sequence[float], std: Sequence[float]):
    """uint8 [B, T, H, W, 3] → float [B, T, 3, H, W] normalizzato (come NormalizeVideo)."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(frames_uint8)).float() / 255.0
    x = x.permute(0, 1, 4, 2, 3)
    m = torch.tensor(mean, dtype=x.dtype).view(1, 1, 3, 1, 1)
    s = torch.tensor(std, dtype=x.dtype).view(1, 1, 3, 1, 1)
    return (x - m) / s


class TextEncoder:
    """Encoder di testo per lo zero-shot.

    ``text_model = distilbert-base-uncased``: DistilBERT + ``txt_proj`` dal checkpoint
    (richiede ``transformers`` e il download del tokenizer/modello).
    ``text_model = hashing``: proiezione casuale deterministica dei token (solo
    ``--synthetic``: niente rete, niente pesi).
    """

    def __init__(self, text_model: str, proj_dim: int, sd: Dict = None) -> None:
        self.kind = text_model
        self.proj_dim = proj_dim
        if text_model == "hashing":
            return
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(text_model)
        self.model = AutoModel.from_pretrained(text_model).eval()
        hidden = self.model.config.hidden_size
        self.txt_proj = nn.Sequential(nn.ReLU(), nn.Linear(hidden, proj_dim))
        if sd is None:
            raise ValueError("lo zero-shot reale richiede i pesi del checkpoint EgoVLP")
        tm = {k[len("text_model.") :]: v for k, v in sd.items() if k.startswith("text_model.")}
        tp = {k[len("txt_proj.") :]: v for k, v in sd.items() if k.startswith("txt_proj.")}
        r1 = self.model.load_state_dict(tm, strict=False)
        r2 = self.txt_proj.load_state_dict(tp, strict=True)
        log.info("testo: chiavi mancanti %d, inattese %d", len(r1.missing_keys), len(r1.unexpected_keys))
        del r2

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.kind == "hashing":
            out = []
            for t in texts:
                v = np.zeros(self.proj_dim)
                for w in t.lower().split():
                    h = int(hashlib.sha256(w.encode()).hexdigest(), 16)
                    v += np.random.default_rng(h % (2**32)).standard_normal(self.proj_dim)
                out.append(v)
            return np.stack(out)
        import torch

        with torch.no_grad():
            tok = self.tok(texts, return_tensors="pt", padding=True, truncation=True)
            h = self.model(**tok).last_hidden_state[:, 0, :]
            return self.txt_proj(h).numpy()


def zero_shot_logits(vid_emb: np.ndarray, txt_emb: np.ndarray, temperature: float) -> np.ndarray:
    a = vid_emb / np.maximum(np.linalg.norm(vid_emb, axis=1, keepdims=True), 1e-8)
    b = txt_emb / np.maximum(np.linalg.norm(txt_emb, axis=1, keepdims=True), 1e-8)
    return (a @ b.T) / temperature
