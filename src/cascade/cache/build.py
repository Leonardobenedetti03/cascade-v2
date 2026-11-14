"""Assemblaggio delle uscite degli stadi nella cache finale (stadi 35 e 45).

L0 (CamFormer, stadio 20, ``20_camformer/seed<S>_hz<F>/``):
- righe di train: ``fold<k>/preds_oof.parquet`` per k = 0..K−1. Si verifica la provenienza
  fuori campione: metadato ``source = oof`` e ``fold = k``, e ogni take della tabella deve
  appartenere alla fold k secondo ``splits/folds.csv``;
- righe di val/test: con ``camformer.eval_source = fold_ensemble`` media delle
  probabilità calibrate (e dei logit) dei K modelli di fold; ``cam_emb`` dal modello
  ``full`` se presente, altrimenti dalla fold 0 (embedding di modelli diversi non sono
  confrontabili, quindi non si mediano); con ``full`` tutto dal modello full.
L1 (stadio 30): ``30_l1_heads/probe/preds.parquet``, con la stessa verifica OOF (``vid_fold``).
Il join tra stadi è sempre per ``clip_id``.

``check_loss`` e ``ProvenanceError`` vivono in ``cache/checks.py`` (nota N11) e qui sono
riesportati perché i test e gli script li importano anche da questo modulo.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.cascade.cache.checks import ProvenanceError, check_loss  # noqa: F401 - riesportati
from src.cascade.labels import assert_class_order
from src.data.cache import Table, concat_tables, join_tables, read_table

log = logging.getLogger(__name__)


def _read(path: Path, labels) -> Table:
    if not path.exists():
        raise FileNotFoundError(f"{path} mancante")
    t = read_table(path)
    assert_class_order(t.meta.get("class_names", []), labels, where=str(path))
    return t


def _check_oof(t: Table, k: int, fold_of: Optional[Dict[str, int]], where: str) -> None:
    if t.meta.get("source") != "oof" or str(t.meta.get("fold")) != str(k):
        raise ProvenanceError(
            f"{where}: metadati source={t.meta.get('source')!r} fold={t.meta.get('fold')!r}, attesi 'oof' e {k}"
        )
    if (t.df.split != "train").any():
        raise ProvenanceError(f"{where}: righe non di train in preds_oof")
    if fold_of is not None:
        bad = sorted({u for u in t.df.take_uid if fold_of.get(u) != k})
        if bad:
            raise ProvenanceError(f"{where}: take che in splits/folds.csv non sono della fold {k}: {bad[:3]}")


def assemble_camformer(cam_dir: Path, cfg, labels, folds: Optional[pd.DataFrame] = None) -> Table:
    """``cam_dir`` = ``<run_dir>/20_camformer/seed<S>_hz<F>``."""
    K = cfg.folds.n_folds
    missing = [k for k in range(K) if not (cam_dir / f"fold{k}" / "summary.json").exists()]
    if missing:
        raise FileNotFoundError(f"stadio 20 incompleto: mancano le fold {missing} in {cam_dir}")
    fold_of = dict(zip(folds.take_uid, folds.fold)) if folds is not None else None
    oofs = []
    for k in range(K):
        t = _read(cam_dir / f"fold{k}" / "preds_oof.parquet", labels)
        _check_oof(t, k, fold_of, f"{cam_dir}/fold{k}/preds_oof.parquet")
        oofs.append(t)
    oof = concat_tables(oofs)
    oof.check_unique()
    keep = ["clip_id", "take_uid", "split", "label_idx", "T_cam"]
    parts = [Table(oof.df[keep].copy(), dict(oof.arrays))]
    for split in ("val", "test"):
        if cfg.camformer.eval_source == "full":
            t = _read(cam_dir / "full" / f"preds_{split}.parquet", labels)
            parts.append(Table(t.df[keep].copy(), dict(t.arrays)))
            continue
        tabs = [_read(cam_dir / f"fold{k}" / f"preds_{split}.parquet", labels).sort_by() for k in range(K)]
        ids = tabs[0].keys()
        for t in tabs[1:]:
            if not np.array_equal(t.keys(), ids):
                raise ValueError(f"le fold hanno clip diverse in preds_{split}")
        for t in tabs:
            if (t.df.split != split).any():
                raise ProvenanceError(f"preds_{split}: righe di altri split")
        emb_src = cam_dir / "full" / f"preds_{split}.parquet"
        emb_tab = _read(emb_src, labels).sort_by() if emb_src.exists() else tabs[0]
        if not np.array_equal(emb_tab.keys(), ids):
            raise ValueError(f"{emb_src}: clip diverse dalle fold")
        arrays = {
            "cam_logits": np.mean([t.arrays["cam_logits"] for t in tabs], 0),
            "cam_probs_cal": np.mean([t.arrays["cam_probs_cal"] for t in tabs], 0),
            "cam_emb": emb_tab.arrays["cam_emb"],
        }
        df = tabs[0].df[[c for c in keep if c != "T_cam"]].copy()
        df["T_cam"] = float(np.mean([t.df.T_cam.iloc[0] for t in tabs if len(t)])) if len(df) else np.nan
        parts.append(Table(df[keep], arrays))
    out = concat_tables(parts)
    out.check_unique()
    out.meta = {
        "class_names": list(labels.names),
        "cam_train_source": "oof",
        "cam_eval_source": cfg.camformer.eval_source,
    }
    return out


def load_l1(l1_dir: Path, labels, folds: Optional[pd.DataFrame] = None) -> Table:
    """``l1_dir`` = ``<run_dir>/30_l1_heads/probe``."""
    t = _read(l1_dir / "preds.parquet", labels)
    if t.meta.get("vid_train_source") != "oof":
        raise ProvenanceError("le predizioni L1 sul train devono essere fuori campione")
    if folds is not None:
        fold_of = dict(zip(folds.take_uid, folds.fold))
        tr = t.df.split == "train"
        exp = t.df.take_uid[tr].map(fold_of)
        if exp.isna().any() or (exp.to_numpy() != t.df.vid_fold[tr].to_numpy()).any():
            raise ProvenanceError("30_l1_heads: vid_fold non coincide con splits/folds.csv")
    return t


def join_l0_l1(cam: Table, vid: Table) -> Table:
    """Join per clip_id; logga le clip presenti in un solo livello."""
    only_cam = sorted(set(cam.keys()) - set(vid.keys()))
    only_vid = sorted(set(vid.keys()) - set(cam.keys()))
    if only_cam or only_vid:
        log.warning(
            "clip solo in L0: %d (es. %s); solo in L1: %d (es. %s) — escluse",
            len(only_cam),
            only_cam[:3],
            len(only_vid),
            only_vid[:3],
        )
    vid_df = vid.df.drop(columns=["take_uid", "split", "label_idx"])
    j = join_tables(cam, Table(vid_df, vid.arrays, vid.meta))
    j.meta = {**cam.meta, "vid_train_source": vid.meta.get("vid_train_source")}
    return j


def motion_table(clips: pd.DataFrame, cfg, cache_dir=None) -> pd.DataFrame:
    """Feature di moto per clip dalla traiettoria della finestra."""
    from src.cascade.motion import MOTION_FEATURES, motion_features
    from src.cascade.trajectory import cached_take_trajectory, slice_window
    from src.data.trajectories import clip_meta

    rows: List[Dict] = []
    cur, tr = None, None
    for r in clips.itertuples(index=False):
        feats = {k: np.nan for k in MOTION_FEATURES}
        if r.has_traj:
            if r.take_uid != cur:
                tr = cached_take_trajectory(Path(r.traj_path), cfg, clip_meta(r), cache_dir)
                cur = r.take_uid
            poses, cov = slice_window(tr, r.t_start, r.t_end)
            if cov >= cfg.data.min_traj_coverage:
                feats = motion_features(poses, cfg.data.traj_fps, cfg.data.vertical_axis)
        rows.append({"clip_id": r.clip_id, **feats})
    return pd.DataFrame(rows)
