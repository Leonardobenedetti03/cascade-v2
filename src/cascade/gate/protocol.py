"""Protocollo di addestramento e scoring dei gate.

1. Controllo: i dati di fit sono le righe di train con uscite OOF di L0
   (:func:`src.cascade.gate.check_fit_split`).
2. Score fuori campione sul train: CV di secondo livello sulle stesse fold di L0
   (per la fold h: fit sulle fold ≠ h, score sulla fold h; l'affidabilità B esclude
   la fold della riga e la fold h).
3. Fit finale sul train intero; score su val e test.

L'etichetta di fit dipende dal gate (``gate.target``): ``hard`` è y = (1 − c0)·c1, ``c1`` è
l'etichetta del CompareGate (M06). Restituisce gli score per ogni riga della cache (la
valutazione è in ``eval/curves.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from src.cascade.gate import check_fit_split
from src.cascade.gate.features import FeatureBuilder
from src.cascade.gate.models import make_gate

log = logging.getLogger(__name__)

TARGET_COLUMN = {"hard": "y", "c1": "c1"}  # gate.target → colonna della tabella


def cv_groups(df, unit: str) -> np.ndarray:
    """Gruppi della CV interna dei gate (N37): ``session`` se richiesto e presente, altrimenti la take.

    Con una riga per take (cache dello stadio 44) i gruppi per take non raggruppano nulla: take della
    stessa sessione finivano sia nel fit sia nella validazione interna (scelta di C ottimista)."""
    if unit == "session":
        if "session" in df.columns:
            return df.session.astype(str).to_numpy()
        log.warning("gate.inner_cv_groups = session ma la cache non ha la colonna session: gruppi per take")
    elif unit != "take":
        raise ValueError(f"gate.inner_cv_groups sconosciuto: {unit!r} (take | session)")
    return df.take_uid.astype(str).to_numpy()


@dataclass
class GateResult:
    name: str
    seed: int
    scores: np.ndarray  # [n_righe della cache]
    importance: Dict[str, float] = field(default_factory=dict)
    fixed_r: object = None


def run_gate(table, labels, gcfg, name: str, seed: int = 0) -> GateResult:
    check_fit_split("train", table.meta.get("cam_train_source", "unknown"))
    gate = make_gate(name, gcfg)
    df = table.df
    split = df.split.to_numpy()
    fold = df.fold.to_numpy().astype(int)
    y = df[TARGET_COLUMN[gate.target]].to_numpy().astype(int)
    take = cv_groups(df, getattr(gcfg, "inner_cv_groups", "take"))
    c0, c1 = df.c0.to_numpy(), df.c1.to_numpy()
    train = np.flatnonzero(split == "train")
    other = np.flatnonzero(split != "train")
    scores = np.zeros(len(df))
    if gate.fixed_r is not None or gate.needs_truth:
        scores[:] = gate.score(np.zeros((len(df), 0)), c0, c1)
        return GateResult(name, seed, scores, fixed_r=gate.fixed_r)

    groups = gate.groups or ["A"]
    fb = FeatureBuilder(table, labels, groups, alpha=gcfg.reliability_alpha, emb_pca_dim=gcfg.emb_pca_dim)

    def X_of(rows, held_out=None):
        return fb.build(rows, held_out=held_out)

    n_folds = int(fold[train].max()) + 1
    for h in range(n_folds):
        fit_rows = train[fold[train] != h]
        held = train[fold[train] == h]
        fb.fit_pca(fit_rows, seed)
        g = make_gate(name, gcfg).fit(X_of(fit_rows, h), y[fit_rows], seed=seed, groups=take[fit_rows])
        scores[held] = g.score(X_of(held, h))
    fb.fit_pca(train, seed)
    gate.fit(X_of(train), y[train], seed=seed, groups=take[train])
    if len(other):
        scores[other] = gate.score(X_of(other))
    return GateResult(name, seed, scores, gate.importance(fb.names()))
