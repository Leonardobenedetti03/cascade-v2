"""Indice delle take (stadio ``index``): una riga per take con etichetta, split, fold e percorsi.

Sorgente: gli ``split_manifest.csv`` delle due metà del subset. Le colonne comuni sono ``take_uid``,
``take_name``, ``split``, ``scenario``, ``camera_pose_file``; nella metà procedurale ``scenario`` è il
compito specifico (es. "Cooking an Omelet") e va ricondotto allo scenario (:func:`scenario_of`).
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

HEALTH = {"Covid-19 Rapid Antigen Test", "First Aid - CPR"}
BIKE = {"Fix a Flat Tire - Replace a Bike Tube", "Remove a Wheel", "Install a Wheel", "Clean and Lubricate the Chain"}
DIRECT = {
    "Basketball": "Basketball",
    "Dance": "Dance",
    "Soccer": "Soccer",
    "Music": "Music",
    "Rock Climbing": "Bouldering",
    "Bouldering": "Bouldering",
    "Health": "Health",
    "Cooking": "Cooking",
    "Bike Repair": "Bike repair",
}


def scenario_of(raw: str) -> str:
    """Scenario Ego-Exo4D (8 classi) dal valore della colonna ``scenario`` del manifest."""
    raw = str(raw).strip()
    if raw in DIRECT:
        return DIRECT[raw]
    if raw in HEALTH:
        return "Health"
    if raw in BIKE:
        return "Bike repair"
    if raw.startswith(("Cooking ", "Making ")):
        return "Cooking"
    raise ValueError(f"scenario non riconosciuto: {raw!r} (aggiornare scenario/index.py)")


def site_of(take_name: str) -> str:
    """Sito = prefisso del nome della take (cmu_, iiith_, uniandes_, ...)."""
    return take_name.split("_", 1)[0].lower()


def session_of(take_name: str) -> str:
    """Sessione = nome della take senza l'ultimo suffisso: take della stessa sessione restano insieme."""
    return take_name.rsplit("_", 1)[0]


def read_manifests(roots: Sequence[str]) -> pd.DataFrame:
    frames = []
    for root in roots:
        mf = Path(root) / "split_manifest.csv"
        if not mf.is_file():
            raise FileNotFoundError(f"manca {mf}")
        df = pd.read_csv(mf, dtype=str)
        need = {"take_uid", "take_name", "split", "scenario", "camera_pose_file"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"{mf}: colonne mancanti {sorted(missing)}")
        df = df[sorted(need)].copy()
        df["root"] = str(root)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    dup = out.take_name.duplicated(keep="first")
    if dup.any():
        log.warning("%d take in più radici: vale la prima", int(dup.sum()))
        out = out[~dup]
    return out.reset_index(drop=True)


def find_video(root: str, take_name: str, video_glob: str) -> Optional[str]:
    found = sorted(glob.glob(str(Path(root) / video_glob.format(take_name=take_name))))
    return found[0] if found else None


def probe_video(path: str):
    """(numero di frame, fps) dall'header del video con PyAV."""
    import av

    with av.open(path) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate) if s.average_rate else float("nan")
        n = int(s.frames or 0)
        if n <= 0 and s.duration is not None and s.time_base is not None:
            n = int(round(float(s.duration * s.time_base) * fps))
    return n, fps


def pretrain_takes(repo: Optional[Path]) -> Dict[str, str]:
    """take_name -> 'train' / 'val' se la take compare nei dati di pretraining di CamFormer."""
    out: Dict[str, str] = {}
    if repo is None:
        return out
    for split in ("val", "train"):  # train vince: è quello che conta per il leakage
        f = Path(repo) / "data_files" / "egoexo4d" / f"{split}_v2.csv"
        if not f.is_file():
            log.warning("non trovo %s: colonna camformer_pretrain vuota", f)
            continue
        ids = pd.read_csv(f, usecols=["save_id"]).save_id.str.rsplit("_", n=2).str[0]
        out.update({t: split for t in ids.unique()})
    return out


def cut_official_val(val: pd.DataFrame, vcfg: dict) -> np.ndarray:
    """Divide la val ufficiale in ``val`` e ``test``: per sessione, stratificato per classe.

    ``mode``: ``undecided`` (tutto ``val``; lo stadio heads si rifiuta di girare), ``split``
    (quota ``test_frac`` al test), ``all_test`` (tutta la val ufficiale diventa test; il gate si
    calibra sulle predizioni fuori campione del train), ``all_val``. Con ``split``, per ogni classe
    (classe di maggioranza della sessione) le sessioni si mescolano con ``seed`` e si aggiungono al
    test se avvicinano la quota; una classe con almeno 2 sessioni ne ha almeno una per parte.
    """
    mode = vcfg["mode"]
    if mode in ("undecided", "all_val"):
        return np.full(len(val), "val", dtype=object)
    if mode == "all_test":
        return np.full(len(val), "test", dtype=object)
    if mode != "split":
        raise ValueError(f"split.official_val.mode sconosciuto: {mode!r}")
    frac = float(vcfg["test_frac"])
    if not 0.0 < frac < 1.0:
        raise ValueError("split.official_val.test_frac deve stare in (0, 1)")
    rng = np.random.default_rng(vcfg["seed"])
    sess = val.groupby("session").label.agg(lambda x: x.value_counts().index[0]).to_frame("label")
    sess["n"] = val.groupby("session").size()
    to_test = set()
    for _, g in sess.groupby("label"):
        order = g.index.to_numpy()[rng.permutation(len(g))]
        target = frac * g.n.sum()
        got, chosen = 0, []
        for s_ in order:
            n = g.n[s_]
            if abs(got + n - target) < abs(got - target):
                chosen.append(s_)
                got += n
        if len(g) >= 2:
            if not chosen:
                chosen = [order[0]]
            elif len(chosen) == len(g):
                chosen = chosen[:-1]
        to_test.update(chosen)
    return np.where(val.session.isin(to_test), "test", "val").astype(object)


def assign_split_and_folds(df: pd.DataFrame, scfg: dict) -> pd.DataFrame:
    """Colonne ``split`` (train / val / test), ``fold_s<seed>`` per ogni seed e ``fold`` (= primo seed).

    Le fold valgono 0..K-1 sul train e -1 altrove. Con ``mode: site`` le fold sono i siti, uguali per tutti i seed.
    """
    df = df.copy()
    seeds = list(scfg["fold_seeds"])
    if not seeds:
        raise ValueError("split.fold_seeds vuoto")
    if scfg["mode"] == "site":
        val, test = set(scfg["val_sites"]), set(scfg["test_sites"])
        df["split"] = np.where(df.site.isin(test), "test", np.where(df.site.isin(val), "val", "train"))
        s2f = scfg["site_to_fold"]
        tr = (df.split == "train").to_numpy()
        missing = sorted(set(df.site[tr]) - set(s2f))
        if missing:
            raise ValueError(f"siti di train senza fold in split.site_to_fold: {missing}")
        f = np.where(tr, df.site.map(s2f).fillna(-1), -1).astype(int)
        folds = {s: f for s in seeds}
    elif scfg["mode"] == "official":
        df["split"] = df.split_official.str.lower()
        bad = sorted(set(df.split) - {"train", "val"})
        if bad:
            raise ValueError(f"valori di split inattesi nei manifest: {bad} (attesi train e val)")
        is_val = (df.split == "val").to_numpy()
        df.loc[is_val, "split"] = cut_official_val(df[is_val], scfg["official_val"])
        tr = (df.split == "train").to_numpy()
        folds = {}
        for s in seeds:
            f = np.full(len(df), -1, dtype=int)
            f[tr] = official_folds(df[tr], scfg["n_folds"], s)
            folds[s] = f
    else:
        raise ValueError(f"split.mode sconosciuto: {scfg['mode']!r}")
    for s in seeds:
        df[f"fold_s{s}"] = folds[s]
    df["fold"] = folds[seeds[0]]
    return df


def official_folds(train: pd.DataFrame, n_folds: int, seed: int) -> np.ndarray:
    """StratifiedGroupKFold per sessione sulle take con classe nella testa; le altre a rotazione."""
    from sklearn.model_selection import StratifiedGroupKFold

    fold = np.full(len(train), -1, dtype=int)
    head = train.in_head.to_numpy()
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    sub = train[head]
    for k, (_, te) in enumerate(cv.split(sub, sub.label, sub.session)):
        fold[np.flatnonzero(head)[te]] = k
    rest = np.flatnonzero(~head)
    fold[rest] = (np.arange(len(rest)) + seed) % n_folds
    return fold


def build_index(cfg: dict, camformer_repo: Optional[Path]) -> pd.DataFrame:
    classes: List[str] = list(cfg["labels"]["classes"])
    raw = read_manifests(cfg["data"]["roots"])
    df = pd.DataFrame(
        {
            "take_uid": raw.take_uid,
            "take_name": raw.take_name,
            "root": raw.root,
            "split_official": raw.split,
            "scenario_raw": raw.scenario,
            "pose_json_path": [str(Path(r) / p) for r, p in zip(raw.root, raw.camera_pose_file)],
        }
    )
    df["label"] = df.scenario_raw.map(scenario_of)
    df["in_head"] = df.label.isin(classes)
    df["label_idx"] = df.label.map({c: i for i, c in enumerate(classes)}).fillna(-1).astype(int)
    df["site"] = df.take_name.map(site_of)
    df["session"] = df.take_name.map(session_of)
    df = assign_split_and_folds(df, cfg["split"])

    videos, n_frames, fps = [], [], []
    for r in df.itertuples(index=False):
        v = find_video(r.root, r.take_name, cfg["data"]["video_glob"])
        videos.append(v)
        if v is None:
            n_frames.append(0)
            fps.append(np.nan)
            continue
        try:
            n, f = probe_video(v)
        except Exception as e:  # noqa: BLE001 - un video rotto non ferma l'indice
            log.warning("%s: video illeggibile (%s)", r.take_name, e)
            n, f = 0, np.nan
        n_frames.append(n)
        fps.append(f)
    df["video_path"] = videos
    df["video_frames"] = n_frames
    df["video_fps"] = fps
    df["duration_s"] = np.where(df.video_frames > 0, df.video_frames / df.video_fps, np.nan)
    df["aria_csv_path"] = [cfg["data"]["aria_csv_pattern"].format(take_name=t) for t in df.take_name]
    df["aria_csv_exists"] = [Path(p).is_file() for p in df.aria_csv_path]
    pre = pretrain_takes(camformer_repo)
    df["camformer_pretrain"] = df.take_name.map(pre).fillna("none")
    return df


def summary(df: pd.DataFrame, window_s: float) -> dict:
    """Conteggi per decidere e controllare (non sono risultati): take e sessioni per split e classe,
    composizione della val ufficiale, sessioni condivise tra train e val/test."""

    def per_class(d: pd.DataFrame) -> dict:
        return {c: {"takes": int(len(x)), "sessions": int(x.session.nunique())} for c, x in d.groupby("label")}

    shared = sorted(set(df.session[df.split == "train"]) & set(df.session[df.split != "train"]))
    off_val = df[df.split_official.str.lower() == "val"]
    return {
        "n_takes": int(len(df)),
        "by_split": {s: per_class(g) for s, g in df.groupby("split")},
        "official_val_composition": per_class(off_val),
        "sessions_train_and_eval": {
            "n": len(shared),
            "takes_eval": int(((df.split != "train") & df.session.isin(shared)).sum()),
            "sessions": shared,
        },
        "folds_first_seed": {int(k): int(v) for k, v in df[df.split == "train"].groupby("fold").size().items()},
        "no_video": int(df.video_path.isna().sum()),
        "short_takes_lt_window": int((df.duration_s < window_s).sum()),
        "aria_csv_missing": int((~df.aria_csv_exists).sum()),
        "camformer_pretrain_by_split": pd.crosstab(df.split, df.camformer_pretrain).to_dict(),
    }
