"""Cache finta realistica per sviluppare e testare il gate senza dati.

Proprietà (default):
- CamFormer ~95% sulle classi fisiche, 55–70% sulle procedurali, con gli errori
  delle procedurali concentrati sulle altre procedurali;
- L1 ~95% ovunque, errori casuali; la fusione è ``replace`` (M05): ``w`` = 0 e
  ``fused_probs`` coincide con L1;
- probabilità non perfettamente calibrate (logit con temperatura sbagliata);
- feature di moto correlate con la classe (le procedurali si sovrappongono);
- embedding con struttura di classe; fold stratificate sul train, ``fold = -1`` altrove.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.cascade.cache.schema import add_labels
from src.cascade.fusion import choose_w, fuse, log_softmax
from src.cascade.labels import Labels
from src.cascade.motion import MOTION_FEATURES
from src.data.cache import Table

# Frequenze approssimate delle classi in scenario_cls.csv (ordine di labels.yaml).
CLASS_PRIOR = np.array([909, 728, 276, 249, 676, 1401, 396, 359], dtype=float)


def _softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(z))


def _stage_logits(
    y: np.ndarray, acc: np.ndarray, confuse: Optional[np.ndarray], rng, C: int, sharp: float
) -> np.ndarray:
    """Logit con accuratezza per classe ``acc[y]`` e confusioni guidate da ``confuse``."""
    n = len(y)
    correct = rng.random(n) < acc[y]
    pred = y.copy()
    for i in np.flatnonzero(~correct):
        cand = np.arange(C)[np.arange(C) != y[i]]
        p = np.ones(len(cand))
        if confuse is not None:
            p = p + 20.0 * confuse[y[i], cand]
        pred[i] = rng.choice(cand, p=p / p.sum())
    z = 0.5 * rng.standard_normal((n, C))
    margin = np.where(correct, rng.normal(3.2, 1.0, n), rng.normal(2.0, 0.6, n))
    z[np.arange(n), pred] += np.clip(margin, 0.6, None)
    z[np.arange(n), y] += np.where(correct, 0.0, rng.uniform(0.2, 0.9, n))
    return sharp * z


def make_mock_cache(
    labels: Labels,
    n_train: int = 1500,
    n_val: int = 400,
    n_test: int = 500,
    n_folds: int = 5,
    seed: int = 0,
    cam_dim: int = 512,
    vid_dim: int = 768,
    acc_cam_physical: float = 0.95,
    acc_cam_procedural=(0.55, 0.70),
    acc_vid: float = 0.95,
    gmacs_L0: float = 0.02,
    gmacs_L1: float = 90.0,
    clips_per_take: int = 1,
) -> Table:
    rng = np.random.default_rng(seed)
    C = labels.num_classes
    proc = labels.procedural_mask
    prior = CLASS_PRIOR[:C] / CLASS_PRIOR[:C].sum()
    acc_cam = np.where(proc, 0.0, acc_cam_physical)
    acc_cam[proc] = rng.uniform(*acc_cam_procedural, proc.sum())
    confuse = np.outer(proc, proc).astype(float)
    np.fill_diagonal(confuse, 0)

    rows = []
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        n_takes = max(n // clips_per_take, 1)
        ys = rng.choice(C, size=n_takes, p=prior)
        for j, yt in enumerate(ys):
            for w in range(clips_per_take):
                rows.append(
                    {
                        "take_uid": f"mock_{split}_{j:05d}",
                        "clip_id": f"mock_{split}_{j:05d}__w{w:02d}",
                        "split": split,
                        "label_idx": int(yt),
                    }
                )
    df = pd.DataFrame(rows)
    y = df.label_idx.to_numpy()
    n = len(df)

    # fold stratificate per take sul train
    fold = np.full(n, -1)
    tr = (df.split == "train").to_numpy()
    takes = df[tr].drop_duplicates("take_uid")
    take_fold = {}
    for c in range(C):
        uids = takes[takes.label_idx == c].take_uid.to_numpy()
        rng.shuffle(uids)
        for i, u in enumerate(uids):
            take_fold[u] = i % n_folds
    fold[tr] = df[tr].take_uid.map(take_fold).to_numpy()
    df["fold"] = fold

    cam_logits = _stage_logits(y, acc_cam, confuse, rng, C, sharp=1.0)
    vid_logits = _stage_logits(y, np.full(C, acc_vid), None, rng, C, sharp=1.2)
    T_cam, T_vid = 1.4, 1.2
    cam_probs = _softmax(cam_logits / T_cam * 1.1)  # calibrazione imperfetta
    vid_probs = _softmax(vid_logits / T_vid * 0.9)

    emb_c = rng.standard_normal((C, cam_dim))
    emb_v = rng.standard_normal((C, vid_dim))
    cam_emb = 0.5 * emb_c[y] + rng.standard_normal((n, cam_dim))
    vid_cls = 0.5 * emb_v[y] + rng.standard_normal((n, vid_dim))

    # moto: media per classe, procedurali vicine tra loro
    base = rng.uniform(0.5, 3.0, (C, len(MOTION_FEATURES)))
    base[proc] = base[proc].mean(0) + 0.1 * rng.standard_normal((proc.sum(), len(MOTION_FEATURES)))
    motion = base[y] * np.exp(0.25 * rng.standard_normal((n, len(MOTION_FEATURES))))
    motion[:, MOTION_FEATURES.index("mot_zfreq_peak")] = np.clip(
        motion[:, MOTION_FEATURES.index("mot_zfreq_peak")] / 3.0, 0, 1
    )
    for k, name in enumerate(MOTION_FEATURES):
        df[name] = motion[:, k]

    w, _ = choose_w(cam_probs[tr], vid_probs[tr], y[tr])  # default replace: w = 0
    fused = fuse(cam_probs, vid_probs, w)
    df["gmacs_L0"], df["gmacs_L1"] = gmacs_L0, gmacs_L1
    df["T_cam"], df["T_vid"], df["w"] = T_cam, T_vid, w
    t = Table(
        df,
        {
            "cam_logits": cam_logits.astype(np.float32),
            "cam_probs_cal": cam_probs.astype(np.float32),
            "cam_emb": cam_emb.astype(np.float32),
            "vid_logits": vid_logits.astype(np.float32),
            "vid_probs_cal": vid_probs.astype(np.float32),
            "vid_cls": vid_cls.astype(np.float32),
            "fused_probs": fused.astype(np.float32),
        },
        {"class_names": list(labels.names), "cam_train_source": "oof", "source": "mock", "seed": seed},
    )
    add_labels(t)
    return t
