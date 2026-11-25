"""Dati del gate v2 da una run completa di scenario_preds (fino a ``feat-camformer`` e ``feat-egovlp``).

Si usano le embedding congelate, non le predizioni: le teste lineari si rifanno dentro la CV annidata.
Servono ``index/takes.csv``, ``features/camformer.npz``, ``features/egovlp.npz``, ``configs/heads.json`` e,
per il moto, ``traj/<take>.npz``. Il file ``sealed/`` non si legge: le etichette del test vengono
dall'indice, che le contiene comunque (il test ufficiale del 17/11 è già stato valutato, N34).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.cascade.data.motion import MOTION_FEATURES
from src.cascade.data.scenario_io import compute_windows, load_npz, take_motion

log = logging.getLogger(__name__)
REQUIRED = ("take_name", "label", "session", "split", "duration_s")


@dataclass
class ScenarioData:
    """Una riga per take nelle classi della testa, con embedding di EgoVLP."""

    takes: pd.DataFrame  # take_name, take_uid, label, label_idx, session, site, split, camformer_pretrain
    X_cam: np.ndarray  # [N, D0], righe NaN = take senza embedding CamFormer (vanno a EgoVLP)
    X_vid: np.ndarray  # [N, D1]
    motion: np.ndarray  # [N, len(MOTION_FEATURES)], NaN se manca la traiettoria
    classes: List[str]
    procedural: np.ndarray  # [C] bool
    n_windows: int
    counts: Dict[str, int] = field(default_factory=dict)

    @property
    def y(self) -> np.ndarray:
        return self.takes.label_idx.to_numpy().astype(int)

    @property
    def sessions(self) -> np.ndarray:
        return self.takes.session.astype(str).to_numpy()

    @property
    def has_l0(self) -> np.ndarray:
        return np.isfinite(self.X_cam).all(1)

    def __len__(self) -> int:
        return len(self.takes)


def load_scenario_run(
    scen_dir: Path, class_names: List[str], procedural: np.ndarray, vertical_axis: int = 2, motion: bool = True
) -> ScenarioData:
    scen_dir = Path(scen_dir)
    hcfg = json.loads((scen_dir / "configs" / "heads.json").read_text())
    if list(hcfg["labels"]["classes"]) != list(class_names):
        raise ValueError(f"classi di scenario_preds {hcfg['labels']['classes']} != labels.yaml {list(class_names)}")
    idx = pd.read_csv(scen_dir / "index" / "takes.csv", dtype={"take_uid": str})
    missing = [c for c in REQUIRED if c not in idx.columns]
    if missing:
        raise ValueError(f"index/takes.csv senza le colonne {missing}")
    if "take_uid" not in idx.columns:
        idx["take_uid"] = idx.take_name
    for col, default in (("site", "unknown"), ("camformer_pretrain", "unknown")):
        if col not in idx.columns:
            idx[col] = default
    counts = {"n_index": int(len(idx))}
    in_head = idx.label.isin(class_names).to_numpy()
    counts["n_out_of_head"] = int((~in_head).sum())
    idx = idx[in_head].reset_index(drop=True)
    cam = load_npz(scen_dir / "features" / "camformer.npz")
    vid = load_npz(scen_dir / "features" / "egovlp.npz")
    d0 = len(next(iter(cam.values())))
    d1 = len(next(iter(vid.values())))
    X0 = np.stack([np.asarray(cam.get(n, np.full(d0, np.nan)), float) for n in idx.take_name])
    X1 = np.stack([np.asarray(vid.get(n, np.full(d1, np.nan)), float) for n in idx.take_name])
    has1 = np.isfinite(X1).all(1)
    counts["n_no_l1"] = int((~has1).sum())
    idx, X0, X1 = idx[has1].reset_index(drop=True), X0[has1], X1[has1]
    counts["n_no_l0"] = int((~np.isfinite(X0).all(1)).sum())
    idx["label_idx"] = idx.label.map({n: i for i, n in enumerate(class_names)}).astype(int)
    w = hcfg["window"]
    n_win = int(w["n_windows"])
    if motion:
        fps = float(hcfg.get("poses", {}).get("fps", 20.0))
        rows = [
            take_motion(
                scen_dir / "traj" / f"{r.take_name}.npz",
                compute_windows(float(r.duration_s), float(w["length_s"]), n_win, w["placement"]),
                fps,
                vertical_axis,
            )
            for r in idx.itertuples(index=False)
        ]
        M = pd.DataFrame(rows)[MOTION_FEATURES].to_numpy(float)
    else:
        M = np.full((len(idx), len(MOTION_FEATURES)), np.nan)
    counts["n_takes"] = int(len(idx))
    counts["n_sessions"] = int(idx.session.nunique())
    counts["n_no_motion"] = int((~np.isfinite(M).all(1)).sum())
    log.info("scenario_preds %s: %s", scen_dir, counts)
    keep = ["take_name", "take_uid", "label", "label_idx", "session", "site", "split", "camformer_pretrain"]
    return ScenarioData(idx[keep].copy(), X0, X1, M, list(class_names), np.asarray(procedural, bool), n_win, counts)


def subset_masks(takes: pd.DataFrame, names: List[str]) -> Dict[str, np.ndarray]:
    """Sottoinsiemi della valutazione (definiti senza label né EgoVLP).

    ``pretrain_unseen``: take che il pre-addestramento di CamFormer non ha usato per i gradienti
    (``camformer_pretrain`` diverso da ``train``); ``official_eval``: val + test ufficiali (285 take in
    official_v2); ``official_<split>``: un solo split ufficiale."""
    pre = takes.camformer_pretrain.astype(str).str.lower().to_numpy()
    split = takes.split.astype(str).str.lower().to_numpy()
    out: Dict[str, np.ndarray] = {}
    for n in names:
        if n == "all":
            m = np.ones(len(takes), bool)
        elif n == "pretrain_unseen":
            m = pre != "train"
        elif n == "pretrain_seen":
            m = pre == "train"
        elif n == "official_eval":
            m = split != "train"
        elif n.startswith("official_"):
            m = split == n[len("official_") :]
        else:
            raise ValueError(f"sottoinsieme sconosciuto: {n!r}")
        out[n] = m
    return out


def check_subsets(takes: pd.DataFrame, names: List[str], primary: Optional[str]) -> None:
    masks = subset_masks(takes, names)
    if primary is not None and primary not in masks:
        raise ValueError(f"gate_v2.primary_subset {primary!r} non è in gate_v2.subsets {names}")
    empty = [n for n, m in masks.items() if not m.any()]
    if empty:
        log.warning("sottoinsiemi vuoti: %s", empty)
