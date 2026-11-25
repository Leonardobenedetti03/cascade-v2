"""L½: classificatore su un solo frame a bassa risoluzione (baseline "un frame", B1).

Gli 8 scenari sono 8 luoghi diversi: un solo frame piccolo può bastare. Per misurarlo:
MobileNetV3-Small di torchvision, backbone congelato, feature dopo l'average pooling
(576 dimensioni), testa lineare per fold (stesso protocollo delle teste L1, stadio 28).

- Input: il frame ``frame_baseline.frame_index`` dei 16 estratti dallo stadio 10 (già
  ``CenterCrop`` + ``Resize(224)``), ridimensionato a ``frame_baseline.size`` (112) con
  antialias, normalizzazione ImageNet (quella dei pesi torchvision).
- Pesi: da file locale (``paths.frame_baseline_ckpt``, state_dict di torchvision
  ``mobilenet_v3_small``); nessun download dentro la pipeline (TODO(HPC): scaricarli sul
  login node, vedi HPC_CHECKLIST). Con ``pretrained: false`` pesi casuali (solo sintetico).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_frame_model(arch: str = "mobilenet_v3_small"):
    """Backbone con uscita = feature dopo l'average pooling ([B, 576] per MobileNetV3-Small)."""
    import torch.nn as nn

    try:
        from torchvision import models
    except ImportError as e:  # pragma: no cover - dipende dall'ambiente
        raise ImportError("frame_baseline richiede torchvision (env/requirements.txt)") from e
    if arch != "mobilenet_v3_small":
        raise ValueError(f"frame_baseline.arch non supportata: {arch!r}")
    net = models.mobilenet_v3_small(weights=None)

    class FrameFeatures(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = net.features
            self.avgpool = net.avgpool
            self.dim = net.classifier[0].in_features

        def forward(self, x):
            return self.avgpool(self.features(x)).flatten(1)

    return FrameFeatures()


def load_frame_weights(model, ckpt: Path) -> None:
    """Carica uno state_dict torchvision di mobilenet_v3_small (chiavi ``features.*``);
    le chiavi del classificatore ImageNet vengono ignorate; chiavi del backbone mancanti → errore."""
    import torch

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}
    res = model.load_state_dict({k: v for k, v in sd.items() if k.startswith("features.")}, strict=False)
    if res.missing_keys:
        raise RuntimeError(f"{ckpt}: mancano {len(res.missing_keys)} pesi del backbone (es. {res.missing_keys[:3]})")


def preprocess_single(frames_uint8: np.ndarray, size: int):
    """uint8 [B, H, W, 3] → float [B, 3, size, size] normalizzato ImageNet."""
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.ascontiguousarray(frames_uint8)).permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    m = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - m) / s


def pixels_read(size: int, n_frames: int = 1) -> int:
    """Pixel in ingresso al modello (× 3 canali): la metrica "pixel letti"."""
    return int(n_frames * size * size * 3)


def frame_features(
    model, frames_dir: Path, clip_ids, frame_index: int, size: int, batch_size: int, device, num_workers: int = 0
) -> Optional[np.ndarray]:
    """Feature [N, D] per le clip (stesso ordine di ``clip_ids``)."""
    import torch

    model.eval().to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(clip_ids), batch_size):
            batch = np.stack(
                [
                    np.load(Path(frames_dir) / f"{c}.npy", mmap_mode="r")[frame_index]
                    for c in clip_ids[i : i + batch_size]
                ]
            )
            out.append(model(preprocess_single(batch, size).to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, getattr(model, "dim", 0)))
