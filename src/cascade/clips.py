"""Finestre temporali per take e indice delle clip (stadio 00).

Una clip è una finestra di durata D (``data.window_s``) nel tempo del video della
take. Con ``n_windows = 1`` è centrata sul centro della take; con più finestre i
centri sono spaziati di ``window_stride_s`` attorno al centro (``window_placement =
center``) o distribuiti su tutta la take (``uniform``). Ogni clip ha un
``clip_id`` distinto e il ``take_uid`` della take; l'aggregazione per take avviene
più avanti (media delle log-probabilità delle sue clip).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CLIP_COLUMNS = [
    "clip_id",
    "take_uid",
    "split",
    "label",
    "label_idx",
    "t_center",
    "t_start",
    "t_end",
    "video_path",
    "traj_path",
    "has_video",
    "has_traj",
]


@dataclass(frozen=True)
class Window:
    t_center: float
    t_start: float
    t_end: float


def compute_windows(
    duration_s: float,
    window_s: float,
    n_windows: int = 1,
    stride_s: Optional[float] = None,
    placement: str = "center",
) -> List[Window]:
    """Fino a ``n_windows`` finestre di durata ``window_s`` dentro ``[0, duration_s]``.

    ``placement``:
    - ``center``: centri spaziati di ``stride_s`` (default ``window_s``) attorno al centro;
    - ``uniform``: centri in ``(i + 0,5) · durata / K``, cioè la take divisa in K parti.
    Con K = 1 le due coincidono (finestra centrata). Se la take è più corta della finestra,
    la finestra è la take intera. Le finestre che sporgono vengono traslate dentro la take
    mantenendo la durata; quelle che dopo la traslazione coincidono contano una volta sola
    (regressione: prima una take corta dava K clip identiche, pagate K volte).
    """
    if duration_s <= 0 or not np.isfinite(duration_s):
        raise ValueError(f"durata non valida: {duration_s}")
    if window_s <= 0 or n_windows < 1:
        raise ValueError("window_s > 0 e n_windows >= 1")
    if placement == "uniform":
        centers = [(i + 0.5) * duration_s / n_windows for i in range(n_windows)]
    elif placement == "center":
        stride = window_s if stride_s is None else stride_s
        mid = duration_s / 2.0
        centers = [mid + (i - (n_windows - 1) / 2.0) * stride for i in range(n_windows)]
    else:
        raise ValueError(f"placement {placement!r}: center | uniform")
    out: List[Window] = []
    for c in centers:
        if window_s >= duration_s:
            w = Window(duration_s / 2.0, 0.0, duration_s)
        else:
            start = min(max(c - window_s / 2.0, 0.0), duration_s - window_s)
            w = Window(start + window_s / 2.0, start, start + window_s)
        if not out or abs(w.t_start - out[-1].t_start) > 1e-6:  # i centri sono ordinati
            out.append(w)
    return out


def make_clip_id(take_uid: str, idx: int) -> str:
    return f"{take_uid}__w{idx:02d}"


def load_scenario(csv_path: Path, labels) -> pd.DataFrame:
    """CSV etichette/split di CamFormer → ``take_uid, split, label, label_idx``.

    ``take_uid`` è il ``take_name`` del CSV. ``label`` è il nome della classe di
    labels.yaml; la colonna ``label`` originale del CSV viene ignorata.
    """
    df = pd.read_csv(csv_path)
    need = {"take_name", "split", "task"}
    if not need <= set(df.columns):
        raise ValueError(f"{csv_path}: colonne attese {need}, trovate {list(df.columns)}")
    if df.take_name.duplicated().any():
        raise ValueError(f"{csv_path}: take_name duplicati")
    unknown = sorted(set(df.task) - set(labels.csv_tasks))
    if unknown:
        raise ValueError(f"task non presenti in labels.yaml: {unknown}")
    idx = df.task.map(labels.task_to_index()).astype(int)
    return pd.DataFrame(
        {
            "take_uid": df.take_name.astype(str).to_numpy(),
            "split": df.split.astype(str).to_numpy(),
            "label": [labels.names[i] for i in idx],
            "label_idx": idx.to_numpy(),
        }
    )


def load_takes_metadata(path: Path) -> Dict[str, dict]:
    """``takes.json`` → dict take_name → record. Accetta lista o dict con chiave ``takes``."""
    with open(path) as f:
        raw = json.load(f)
    items = raw["takes"] if isinstance(raw, dict) and "takes" in raw else raw
    if isinstance(items, dict):
        items = list(items.values())
    return {str(t["take_name"]): t for t in items if "take_name" in t}


def resolve_take_paths(ctx, take_name: str) -> Tuple[Path, Path]:
    p = ctx.cfg.paths
    takes_root = ctx.path(p.takes_root)
    video = ctx.path(p.video_pattern, takes_root=takes_root, take_name=take_name)
    if not video.exists() and p.video_glob:
        pattern = p.video_glob.format(
            takes_root=takes_root, take_name=take_name, data=ctx.data_root, repo="", run=ctx.run_dir
        )
        base = Path(pattern).parent
        matches = sorted(base.glob(Path(pattern).name)) if base.exists() else []
        if matches:
            video = matches[0]
    if ctx.cfg.data.traj_source == "converted_npz":
        # M08: l'uscita dello stadio 01 sta in paths.traj_dir, un <take_name>.npz per take.
        # paths.traj_pattern resta il CSV di Aria (sorgente aria_csv) e qui non va usato.
        traj = ctx.path(p.traj_dir) / f"{take_name}.npz"
    else:
        traj = ctx.path(p.traj_pattern, takes_root=takes_root, take_name=take_name)
    return video, traj


def build_inventory(ctx) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Costruisce l'indice delle clip, il report delle mancanti e un riepilogo."""
    cfg = ctx.cfg
    labels = ctx.scenario_labels or ctx.labels  # l'inventario è sempre per scenario (anche col task keystep)
    takes = load_scenario(ctx.path(cfg.paths.scenario_csv), labels)
    meta_path = ctx.path(cfg.paths.takes_json)
    meta: Dict[str, dict] = {}
    if meta_path.exists():
        meta = load_takes_metadata(meta_path)
    elif cfg.alignment.mode == "device_us":
        # gli start in µs stanno solo in takes.json: senza, ogni take sarebbe no_timesync
        raise FileNotFoundError(f"metadati delle take non trovati: {meta_path} (servono con alignment.mode=device_us)")
    else:
        # la durata ripiega sul video (_take_duration), come nella v1 sul cluster (N5)
        log.warning("metadati delle take non trovati: %s (durata dal video)", meta_path)

    rows, missing = [], []
    need_sync = cfg.alignment.mode == "device_us"
    for t in takes.itertuples(index=False):
        video, traj = resolve_take_paths(ctx, t.take_uid)
        has_video, has_traj = video.exists(), traj.exists()
        m = meta.get(t.take_uid, {})
        problems = []
        if need_sync and has_traj and not _finite(m.get(cfg.alignment.video_start_key)):
            # senza lo start del video la traiettoria non è allineabile: inutilizzabile
            problems.append("no_timesync")
            has_traj = False
        duration = _take_duration(
            cfg.data.duration_source, m, video if has_video else None, traj if has_traj else None, ctx
        )
        if not has_video:
            problems.append("no_video")
        if not has_traj:
            problems.append("no_traj")
        if duration is None:
            problems.append("no_duration")
        elif duration < cfg.data.window_s:
            problems.append("short_take" if cfg.data.short_take == "keep" else "short_take_dropped")
        if problems:
            missing.append(
                {
                    "take_uid": t.take_uid,
                    "split": t.split,
                    "label": t.label,
                    "problems": ";".join(problems),
                    "video_path": str(video),
                    "traj_path": str(traj),
                }
            )
        if duration is None or (duration < cfg.data.window_s and cfg.data.short_take == "drop"):
            continue
        for i, w in enumerate(
            compute_windows(
                duration, cfg.data.window_s, cfg.data.n_windows, cfg.data.window_stride_s, cfg.data.window_placement
            )
        ):
            rows.append(
                {
                    "clip_id": make_clip_id(t.take_uid, i),
                    "take_uid": t.take_uid,
                    "split": t.split,
                    "label": t.label,
                    "label_idx": int(t.label_idx),
                    "t_center": w.t_center,
                    "t_start": w.t_start,
                    "t_end": w.t_end,
                    "video_path": str(video),
                    "traj_path": str(traj),
                    "has_video": has_video,
                    "has_traj": has_traj,
                    "take_duration": duration,
                    "ego_take_uid": m.get("take_uid", ""),
                    "video_start_us": m.get(cfg.alignment.video_start_key, np.nan),
                    "traj_start_us": m.get(cfg.alignment.traj_start_key, np.nan),
                }
            )
    clips = pd.DataFrame(rows)
    miss = pd.DataFrame(missing, columns=["take_uid", "split", "label", "problems", "video_path", "traj_path"])
    summary = _summary(takes, clips, miss, cfg.data.expected_split_counts)
    return clips, miss, summary


def _take_duration(source: str, meta: dict, video: Optional[Path], traj: Optional[Path], ctx) -> Optional[float]:
    order = {
        "takes_json": ["takes_json", "video", "trajectory"],
        "video": ["video", "takes_json", "trajectory"],
        "trajectory": ["trajectory", "takes_json", "video"],
    }[source]
    for s in order:
        if s == "takes_json" and meta.get("duration_sec") is not None:
            return float(meta["duration_sec"])
        if s == "video" and video is not None:
            from src.cascade.video import probe_video

            try:
                return float(probe_video(video).duration_s)
            except Exception as e:  # noqa: BLE001
                log.warning("probe fallito per %s: %s", video, e)
        if s == "trajectory" and traj is not None:
            from src.cascade.trajectory import load_take_trajectory

            tr = load_take_trajectory(traj, ctx.cfg, meta)
            # le finestre vivono nel tempo del video (origine 0): conta la fine, non l'ampiezza
            return float(tr.t_video[-1])
    return None


def _finite(v) -> bool:
    try:
        return v is not None and bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def _summary(takes: pd.DataFrame, clips: pd.DataFrame, miss: pd.DataFrame, expected: Dict[str, int]) -> dict:
    counts = takes.split.value_counts().to_dict()
    checks = {}
    for split, n in (expected or {}).items():
        found = int(counts.get(split, 0))
        checks[split] = {"expected": int(n), "found": found, "ok": found == int(n)}
        if found != int(n):
            log.warning("sanity check split %s: attese %d take, trovate %d", split, n, found)
    usable = clips[clips.has_video & clips.has_traj] if len(clips) else clips
    return {
        "takes_per_split": {k: int(v) for k, v in counts.items()},
        "clips": int(len(clips)),
        "clips_usable": int(len(usable)),
        "clips_usable_per_split": usable.split.value_counts().to_dict() if len(usable) else {},
        "takes_with_problems": int(len(miss)),
        "problems": miss.problems.str.split(";").explode().value_counts().to_dict() if len(miss) else {},
        "expected_split_counts": checks,
    }


def inspect_layout(ctx, max_entries: int = 12) -> str:
    """Descrizione testuale del layout dei dati, per il primo avvio sull'HPC."""
    lines = [f"data_root = {ctx.data_root}"]

    def tree(p: Path, depth: int, prefix: str = "") -> None:
        if depth < 0 or not p.is_dir():
            return
        entries = sorted(p.iterdir())
        for e in entries[:max_entries]:
            lines.append(f"{prefix}{e.name}{'/' if e.is_dir() else ''}")
            tree(e, depth - 1, prefix + "    ")
        if len(entries) > max_entries:
            lines.append(f"{prefix}... ({len(entries) - max_entries} altri)")

    tree(ctx.data_root, 2)
    cfg = ctx.cfg
    scen = ctx.path(cfg.paths.scenario_csv)
    lines.append(f"\nscenario_csv: {scen} (esiste: {scen.exists()})")
    meta_path = ctx.path(cfg.paths.takes_json)
    lines.append(f"takes_json: {meta_path} (esiste: {meta_path.exists()})")
    if meta_path.exists():
        meta = load_takes_metadata(meta_path)
        first = next(iter(meta.values()), {})
        lines.append(f"  {len(meta)} take; chiavi del primo record: {sorted(first)[:40]}")
    if scen.exists():
        name = str(pd.read_csv(scen).take_name.iloc[0])
        takes_root = ctx.path(cfg.paths.takes_root)
        tdir = takes_root / name
        lines.append(f"\ntake d'esempio: {tdir} (esiste: {tdir.exists()})")
        if tdir.exists():
            for f in sorted(tdir.rglob("*"))[:60]:
                lines.append(f"    {f.relative_to(tdir)}")
        video, traj = resolve_take_paths(ctx, name)
        lines.append(f"video atteso: {video} (esiste: {video.exists()})")
        if video.exists():
            from src.cascade.video import probe_video

            lines.append(f"  {probe_video(video)}")
        lines.append(f"traiettoria attesa: {traj} (esiste: {traj.exists()})")
        if traj.exists() and traj.suffix == ".npz":
            lines.append(f"  npz di {traj.stat().st_size} byte")
        elif traj.exists():
            with open(traj) as f:
                lines.extend("  " + next(f).rstrip()[:300] for _ in range(3))
    return "\n".join(lines)
