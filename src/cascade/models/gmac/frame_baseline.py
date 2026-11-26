"""L½: a classifier on a single low-resolution frame (the "one frame" baseline, B1).

The 8 scenarios happen in 8 different kinds of places, so one small frame might be enough. To measure this we
use torchvision's MobileNetV3-Small, frozen, with the features after average pooling (576 dimensions) and a
linear head per fold (same protocol as the L1 heads, stage 28).

- Input: frame ``frame_baseline.frame_index`` of the 16 extracted by stage 10 (already ``CenterCrop`` +
  ``Resize(224)``), resized to ``frame_baseline.size`` (112) with antialias, ImageNet normalisation (the one
  of the torchvision weights).
- Weights: from a local file (``paths.frame_baseline_ckpt``, torchvision ``mobilenet_v3_small``
  state_dict). Nothing is downloaded inside the pipeline: we fetch the file on the login node beforehand.
  With ``pretrained: false`` the weights are random (synthetic runs only).
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
    """Backbone whose output is the features after average pooling ([B, 576] for MobileNetV3-Small)."""
    import torch.nn as nn

    try:
        from torchvision import models
    except ImportError as e:  # pragma: no cover - depends on the environment
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
    """Load a torchvision state_dict of mobilenet_v3_small (keys ``features.*``). The keys of the ImageNet
    classifier are ignored; missing backbone keys are an error."""
    import torch

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}
    res = model.load_state_dict({k: v for k, v in sd.items() if k.startswith("features.")}, strict=False)
    if res.missing_keys:
        raise RuntimeError(f"{ckpt}: mancano {len(res.missing_keys)} pesi del backbone (es. {res.missing_keys[:3]})")


def preprocess_single(frames_uint8: np.ndarray, size: int):
    """uint8 [B, H, W, 3] → float [B, 3, size, size], ImageNet-normalised."""
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.ascontiguousarray(frames_uint8)).permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    m = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - m) / s


def pixels_read(size: int, n_frames: int = 1) -> int:
    """Pixels fed to the model (× 3 channels): the "pixels read" metric."""
    return int(n_frames * size * size * 3)


def frame_features(
    model, frames_dir: Path, clip_ids, frame_index: int, size: int, batch_size: int, device, num_workers: int = 0
) -> Optional[np.ndarray]:
    """Features [N, D] for the clips (same order as ``clip_ids``)."""
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
