"""Feature del gate: solo da L0 (CamFormer) e dalla traiettoria, mai da L1.

Gruppi:
- ``A``: probabilità massima, margine top-1 − top-2, entropia normalizzata (/ log C),
  da probabilità calibrate;
- ``B``: affidabilità della classe predetta (precisione fuori campione di CamFormer
  per quella classe, calcolata escludendo le fold indicate) e flag
  "top-1 e top-2 entrambe procedurali";
- ``C``: feature di moto ``mot_*`` (``src/cascade/motion.py``);
- opzionali: ``onehot`` (classe predetta), ``emb`` (PCA di ``cam_emb``, fittata solo
  sulle righe di fit).

Non si portano dalla v1 i gruppi ``mc`` e ``take`` né l'affidabilità Beta-Binomiale
(M06, nota N16 del registro): l'affidabilità di B è sempre quella empirica.

Leakage di B: l'affidabilità di una clip non usa mai le etichette della sua fold, e
dentro la CV di secondo livello esclude anche la fold tenuta fuori
(vedi :func:`reliability`, argomento ``exclude``). Con le fold per sito (M03) significa
«precisione di L0 misurata sugli altri siti».
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.cascade.motion import MOTION_FEATURES

A_FEATURES = ["p_max", "margin", "entropy_norm"]
B_FEATURES = ["reliability", "top2_procedural"]
KNOWN_GROUPS = ("A", "B", "C", "onehot", "emb")


def group_a(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    s = np.sort(p, axis=1)
    ent = -(p * np.log(p)).sum(1) / np.log(p.shape[1])
    return np.stack([s[:, -1], s[:, -1] - s[:, -2], ent], axis=1)


def top2_procedural(probs: np.ndarray, procedural_mask: np.ndarray) -> np.ndarray:
    top2 = np.argsort(-probs, axis=1, kind="stable")[:, :2]
    return (procedural_mask[top2[:, 0]] & procedural_mask[top2[:, 1]]).astype(float)


def reliability_counts(
    pred: np.ndarray, correct: np.ndarray, fold: np.ndarray, n_folds: int, n_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Conteggi per fold e classe predetta: (n_pred[K, C], n_correct[K, C])."""
    n_pred = np.zeros((n_folds, n_classes))
    n_corr = np.zeros((n_folds, n_classes))
    np.add.at(n_pred, (fold, pred), 1.0)
    np.add.at(n_corr, (fold, pred), correct.astype(float))
    return n_pred, n_corr


def reliability(
    pred_rows: np.ndarray, exclude: np.ndarray, n_pred: np.ndarray, n_corr: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    """Precisione fuori campione della classe predetta, con smoothing Beta(α, α).

    Args:
        pred_rows: classe predetta da L0 per ogni riga da descrivere.
        exclude: maschera ``[n_rows, K]``: True = la fold k non va usata per quella riga.
        n_pred, n_corr: conteggi per fold dalle righe OOF di train.
    """
    keep = (~exclude).astype(float)
    n = (keep @ n_pred)[np.arange(len(pred_rows)), pred_rows]
    k = (keep @ n_corr)[np.arange(len(pred_rows)), pred_rows]
    return (k + alpha) / (n + 2.0 * alpha)


def parse_groups(spec: Sequence[str]) -> List[str]:
    groups = list(spec)
    bad = set(groups) - set(KNOWN_GROUPS)
    if bad:
        raise ValueError(f"gruppi di feature sconosciuti: {bad}")
    return groups


class FeatureBuilder:
    """Costruisce la matrice delle feature per un insieme di righe della cache.

    ``fit_rows`` sono le righe su cui si stima ciò che va stimato (PCA per ``emb``);
    le righe OOF di train (``fold >= 0``) forniscono i conteggi di affidabilità.
    """

    def __init__(
        self,
        table,
        labels,
        groups: Sequence[str],
        *,
        alpha: float = 1.0,
        emb_pca_dim: int = 16,
    ) -> None:
        self.t = table
        self.labels = labels
        self.groups = parse_groups(groups)
        self.alpha = alpha
        self.emb_pca_dim = emb_pca_dim
        probs = table.arrays["cam_probs_cal"]
        self.probs = probs
        self.pred = probs.argmax(1)
        y = table.df.label_idx.to_numpy()
        self.fold = table.df.fold.to_numpy().astype(int)
        self.n_folds = int(self.fold.max()) + 1 if (self.fold >= 0).any() else 0
        train = self.fold >= 0
        self.n_pred, self.n_corr = reliability_counts(
            self.pred[train], self.pred[train] == y[train], self.fold[train], self.n_folds, labels.num_classes
        )
        self._pca = None

    def names(self) -> List[str]:
        out: List[str] = []
        for g in self.groups:
            if g == "A":
                out += A_FEATURES
            elif g == "B":
                out += B_FEATURES
            elif g == "C":
                out += MOTION_FEATURES
            elif g == "onehot":
                out += [f"pred_is_{n}" for n in self.labels.names]
            elif g == "emb":
                out += [f"emb_pc{i}" for i in range(self.emb_pca_dim)]
        return out

    def exclusion(self, rows: np.ndarray, held_out: Optional[int]) -> np.ndarray:
        """Fold escluse per l'affidabilità: la fold della riga e la fold tenuta fuori."""
        ex = np.zeros((len(rows), self.n_folds), dtype=bool)
        own = self.fold[rows]
        has = own >= 0
        ex[np.flatnonzero(has), own[has]] = True
        if held_out is not None:
            ex[:, held_out] = True
        return ex

    def fit_pca(self, rows: np.ndarray, seed: int = 0) -> None:
        if "emb" in self.groups:
            from sklearn.decomposition import PCA

            k = min(self.emb_pca_dim, len(rows), self.t.arrays["cam_emb"].shape[1])
            self._pca = PCA(n_components=k, random_state=seed).fit(self.t.arrays["cam_emb"][rows])

    def build(self, rows: np.ndarray, held_out: Optional[int] = None) -> np.ndarray:
        rows = np.asarray(rows)
        mats = []
        for g in self.groups:
            if g == "A":
                mats.append(group_a(self.probs[rows]))
            elif g == "B":
                flag = top2_procedural(self.probs[rows], self.labels.procedural_mask)
                rel = reliability(self.pred[rows], self.exclusion(rows, held_out), self.n_pred, self.n_corr, self.alpha)
                mats.append(np.stack([rel, flag], 1))
            elif g == "C":
                mats.append(self.t.df.loc[rows, MOTION_FEATURES].to_numpy(dtype=float))
            elif g == "onehot":
                mats.append(np.eye(self.labels.num_classes)[self.pred[rows]])
            elif g == "emb":
                if self._pca is None:
                    raise RuntimeError("fit_pca va chiamato prima di build con il gruppo emb")
                z = self._pca.transform(self.t.arrays["cam_emb"][rows])
                pad = np.zeros((len(rows), self.emb_pca_dim))
                pad[:, : z.shape[1]] = z
                mats.append(pad)
        return np.concatenate(mats, axis=1) if mats else np.zeros((len(rows), 0))
