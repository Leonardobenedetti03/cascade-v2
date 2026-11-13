"""Fusione L0 + L1 al livello video (stadio 35).

``log p_fusa = w · log p_cam + (1 − w) · log p_vid`` (poi rinormalizzata).
``w`` si sceglie su una griglia in [0, 1] con le uscite fuori campione del train:
massima accuracy, a parità minima NLL. ``mode = replace`` (ablation) fissa ``w = 0``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

EPS = 1e-12


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def fuse(p_cam: np.ndarray, p_vid: np.ndarray, w: float) -> np.ndarray:
    if not 0.0 <= w <= 1.0:
        raise ValueError("w deve stare in [0, 1]")
    lp = w * np.log(np.clip(p_cam, EPS, 1.0)) + (1.0 - w) * np.log(np.clip(p_vid, EPS, 1.0))
    return np.exp(log_softmax(lp))


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], EPS, 1.0))))


def choose_w(
    p_cam: np.ndarray, p_vid: np.ndarray, y: np.ndarray, step: float = 0.05, mode: str = "fuse"
) -> Tuple[float, List[Dict[str, float]]]:
    """Restituisce (w scelto, griglia con accuracy e NLL per ogni w)."""
    if mode == "replace":
        p = fuse(p_cam, p_vid, 0.0)
        return 0.0, [{"w": 0.0, "acc": float((p.argmax(1) == y).mean()), "nll": nll(p, y)}]
    if mode != "fuse":
        raise ValueError(f"fusion.mode sconosciuto: {mode!r}")
    n = int(round(1.0 / step))
    grid = []
    for w in np.linspace(0.0, 1.0, n + 1):
        p = fuse(p_cam, p_vid, float(w))
        grid.append({"w": float(w), "acc": float((p.argmax(1) == y).mean()), "nll": nll(p, y)})
    best = max(grid, key=lambda g: (round(g["acc"], 12), -g["nll"]))
    return best["w"], grid


# --------------------------------------------------------------------------- fusione appresa (B5)
LEARNED_FEATURES = ("cam_logprobs", "vid_cls", "cam_emb")


def fusion_features(table, kinds) -> np.ndarray:
    """Feature della fusione appresa per ogni riga della tabella L0 ⋈ L1.

    ``cam_logprobs``: log delle probabilità calibrate OOF di CamFormer (8). Sono confrontabili
    tra fold (stesse classi), a differenza di ``cam_emb``: ogni fold ha il suo modello
    fine-tuned e quindi il suo spazio di embedding (opzione possibile, ma va saputo).
    ``vid_cls``: CLS di EgoVLP congelato (768), stesso spazio per tutte le clip.
    """
    mats = []
    for k in kinds:
        if k not in LEARNED_FEATURES:
            raise ValueError(f"fusion.learned_features: {k!r} non valido ({list(LEARNED_FEATURES)})")
        if k == "cam_logprobs":
            mats.append(np.log(np.clip(table.arrays["cam_probs_cal"], EPS, 1.0)))
        else:
            mats.append(table.arrays[k])
    return np.concatenate(mats, axis=1).astype(np.float64)


def learned_fusion(table, folds, hcfg, n_classes: int, kinds, seed: int = 0):
    """Fusione appresa fuori campione: per la fold k logistica sulle fold ≠ k (senza holdout),
    temperatura sull'holdout, predizione della fold k; val/test = media dei K modelli.
    Restituisce (probabilità [N, C], riepilogo per fold)."""
    from src.cascade.heads import run_heads

    res, summary = run_heads(table.df, fusion_features(table, kinds), folds, hcfg, n_classes, seed)
    return res["vid_probs_cal"].astype(np.float64), summary


def compare_fusions(table, w: float, learned=None, n_boot: int = 1000, seed: int = 0):
    """Accuratezza e NLL (per take dove serve) di L0, L1, fusione log-lineare, sostituzione e
    fusione appresa su train OOF e val, con IC appaiato della differenza rispetto a L1."""
    rows = []
    y_all = table.df.label_idx.to_numpy()
    systems = {
        "L0": table.arrays["cam_probs_cal"],
        "L1 (sostituzione)": table.arrays["vid_probs_cal"],
        f"log-lineare (w={w:.2f})": fuse(table.arrays["cam_probs_cal"], table.arrays["vid_probs_cal"], w),
    }
    if learned is not None:
        systems["appresa"] = learned
    rng = np.random.default_rng(seed)
    for split in ("train", "val"):
        m = (table.df.split == split).to_numpy()
        if not m.any():
            continue
        y = y_all[m]
        corr_l1 = (systems["L1 (sostituzione)"][m].argmax(1) == y).astype(float)
        idx = rng.integers(0, m.sum(), (n_boot, m.sum())) if n_boot else None
        for name, p in systems.items():
            corr = (p[m].argmax(1) == y).astype(float)
            row = {"split": split, "system": name, "acc": float(corr.mean()), "nll": nll(p[m], y), "n": int(m.sum())}
            if idx is not None and name != "L1 (sostituzione)":
                d = corr[idx].mean(1) - corr_l1[idx].mean(1)
                row.update(
                    {
                        "diff_vs_L1": float(corr.mean() - corr_l1.mean()),
                        "ci_low": float(np.percentile(d, 2.5)),
                        "ci_high": float(np.percentile(d, 97.5)),
                    }
                )
            rows.append(row)
    import pandas as pd

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- fusioni con L½ (B1)
def fuse_many(probs, weights) -> np.ndarray:
    """Fusione log-lineare di più livelli: softmax(Σ_i w_i · log p_i), con Σ w_i = 1."""
    lp = sum(w * np.log(np.clip(p, EPS, 1.0)) for p, w in zip(probs, weights))
    return np.exp(log_softmax(lp))


def simplex_grid(k: int, step: float):
    """Punti del simplesso a ``k`` componenti con passo ``step`` (k = 2 o 3)."""
    n = int(round(1.0 / step))
    if k == 2:
        return [(i / n, (n - i) / n) for i in range(n + 1)]
    if k == 3:
        return [(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)]
    raise ValueError("simplex_grid: k = 2 o 3")


def choose_simplex(probs, y: np.ndarray, step: float = 0.1):
    """Pesi sul simplesso che massimizzano l'accuracy, a parità minima NLL, sulle uscite OOF
    del train. Restituisce (pesi, accuracy, NLL)."""
    best = None
    for w in simplex_grid(len(probs), step):
        p = fuse_many(probs, w)
        key = (round(float((p.argmax(1) == y).mean()), 12), -nll(p, y))
        if best is None or key > best[0]:
            best = (key, tuple(float(x) for x in w))
    return best[1], best[0][0], -best[0][1]
