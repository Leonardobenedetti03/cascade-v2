"""Schema della cache finale (stadio 45), una riga per clip.

Colonne scalari obbligatorie:
    clip_id, take_uid, split, fold, label_idx, c0, c1, y,
    gmacs_L0, gmacs_L1, T_cam, T_vid, w, e le feature di moto ``mot_*``.
Colonne vettoriali obbligatorie (``fixed_size_list<float32>``):
    cam_logits[C], cam_probs_cal[C], cam_emb[D0], vid_logits[C], vid_probs_cal[C],
    vid_cls[D1], fused_probs[C].
Opzionali: vid_emb[P], vid_zs_logits[C]; con la baseline un frame (B1): half_logits[C],
    half_probs_cal[C], fused01_probs[C] (L0+L½), fused012_probs[C] (L0+L½+L1), gmacs_Lhalf,
    pixels_Lhalf, pixels_L1, w01, w012_*.

Metadati: ``class_names`` (ordine di labels.yaml) e ``cam_train_source = "oof"``
(le righe di train hanno uscite di L0 fuori campione: requisito del gate).

Con ``fusion.mode = replace`` (default, M05) ``w`` = 0 e ``fused_probs`` coincide con L1.
Convenzioni: ``fold`` ∈ [0, K) per train e −1 per val/test; c0 = CamFormer corretto,
c1 = L1 fuso corretto, y = (1 − c0) · c1.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.cascade.labels import Labels
from src.cascade.motion import MOTION_FEATURES
from src.data.cache import Table

SPLITS = ("train", "val", "test")
SCALARS_REQUIRED = [
    "clip_id",
    "take_uid",
    "split",
    "fold",
    "label_idx",
    "c0",
    "c1",
    "y",
    "gmacs_L0",
    "gmacs_L1",
    "T_cam",
    "T_vid",
    "w",
] + MOTION_FEATURES
PROB_VECTORS = ["cam_probs_cal", "vid_probs_cal", "fused_probs"]
CLASS_VECTORS = ["cam_logits", "vid_logits"] + PROB_VECTORS
EMB_VECTORS = ["cam_emb", "vid_cls"]
OPTIONAL_VECTORS = [
    "vid_emb",
    "vid_zs_logits",
    "half_logits",
    "half_probs_cal",
    "fused01_probs",
    "fused012_probs",
]
OPTIONAL_SCALARS = [
    "gmacs_Lhalf",
    "pixels_Lhalf",
    "pixels_L1",
    "w01",
    "w012_cam",
    "w012_half",
    "w012_vid",
]
OPTIONAL_PROB_VECTORS = ["half_probs_cal", "fused01_probs", "fused012_probs"]
NAN_ALLOWED = set(MOTION_FEATURES)


class SchemaError(ValueError):
    pass


def validate_cache(
    t: Table,
    labels: Labels,
    *,
    cam_dim: Optional[int] = None,
    vid_dim: Optional[int] = None,
    prob_tol: float = 1e-3,
    require_embeddings: bool = True,
) -> List[str]:
    """Valida la cache. Solleva :class:`SchemaError` con tutti i problemi trovati.

    ``require_embeddings=False`` per le letture a colonne (stadio 50), che non caricano
    ``cam_emb``/``vid_cls``. Restituisce gli avvisi non bloccanti (es. NaN nel moto).
    """
    errors: List[str] = []
    warnings: List[str] = []
    C = labels.num_classes
    df = t.df
    for c in SCALARS_REQUIRED:
        if c not in df.columns:
            errors.append(f"colonna scalare mancante: {c}")
    for c in CLASS_VECTORS + (EMB_VECTORS if require_embeddings else []):
        if c not in t.arrays:
            errors.append(f"colonna vettoriale mancante: {c}")
    if errors:
        raise SchemaError("; ".join(errors))

    if t.meta.get("class_names") != list(labels.names):
        errors.append(f"class_names nei metadati {t.meta.get('class_names')} != labels.yaml")
    if t.meta.get("cam_train_source") != "oof":
        errors.append("metadato cam_train_source != 'oof': il gate richiede L0 fuori campione sul train")
    if df.clip_id.duplicated().any():
        errors.append("clip_id duplicati")
    bad_split = set(df.split) - set(SPLITS)
    if bad_split:
        errors.append(f"split non validi: {bad_split}")
    fold = df.fold.to_numpy()
    is_train = (df.split == "train").to_numpy()
    if (fold[is_train] < 0).any():
        errors.append("righe di train senza fold")
    if (fold[~is_train] != -1).any():
        errors.append("righe val/test con fold != -1")
    y_true = df.label_idx.to_numpy()
    if ((y_true < 0) | (y_true >= C)).any():
        errors.append("label_idx fuori range")
    for c in CLASS_VECTORS:
        if t.arrays[c].shape[1] != C:
            errors.append(f"{c}: {t.arrays[c].shape[1]} colonne, attese {C}")
    for c, d in (("cam_emb", cam_dim), ("vid_cls", vid_dim)):
        if d is not None and c in t.arrays and t.arrays[c].shape[1] != d:
            errors.append(f"{c}: dimensione {t.arrays[c].shape[1]}, attesa {d}")
    for c in PROB_VECTORS + [v for v in OPTIONAL_PROB_VECTORS if v in t.arrays]:
        p = t.arrays[c]
        if (p < -1e-6).any() or np.abs(p.sum(1) - 1).max() > prob_tol:
            errors.append(f"{c}: righe non sono distribuzioni di probabilità")
    for c in list(t.arrays):
        if np.isnan(t.arrays[c]).any():
            errors.append(f"NaN in {c}")
    for c in SCALARS_REQUIRED:
        if c in ("clip_id", "take_uid", "split"):
            continue
        n_nan = int(df[c].isna().sum())
        if n_nan:
            (warnings if c in NAN_ALLOWED else errors).append(f"{n_nan} NaN in {c}")
    if not errors:
        c0 = (t.arrays["cam_probs_cal"].argmax(1) == y_true).astype(int)
        c1 = (t.arrays["fused_probs"].argmax(1) == y_true).astype(int)
        if not np.array_equal(c0, df.c0.to_numpy().astype(int)):
            errors.append("c0 incoerente con cam_probs_cal")
        if not np.array_equal(c1, df.c1.to_numpy().astype(int)):
            errors.append("c1 incoerente con fused_probs")
        if not np.array_equal(df.y.to_numpy().astype(int), (1 - c0) * c1):
            errors.append("y != (1 - c0) * c1")
    if errors:
        raise SchemaError("; ".join(errors))
    return warnings


def add_labels(t: Table) -> None:
    """Calcola c0, c1, y dalle probabilità (in place)."""
    y_true = t.df.label_idx.to_numpy()
    c0 = (t.arrays["cam_probs_cal"].argmax(1) == y_true).astype(int)
    c1 = (t.arrays["fused_probs"].argmax(1) == y_true).astype(int)
    t.df["c0"], t.df["c1"], t.df["y"] = c0, c1, (1 - c0) * c1
