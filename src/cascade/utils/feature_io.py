"""Atomic writes (temporary file + rename) and completion markers for the stages.

An output exists only if it is complete. The marker ``_done/<stage>.json`` says so: it is written last, with
the hash of the config sections the stage used. When we run again, a stage is skipped if the marker and the
outputs are there and the hash matches. If the hash is different the stage stops (you need ``--overwrite``),
so outputs of different configs never get mixed.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd


@contextmanager
def atomic_path(dest: Path) -> Iterator[Path]:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=dest.suffix)
    os.close(fd)
    try:
        yield Path(tmp)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_csv(df: pd.DataFrame, dest: Path) -> None:
    with atomic_path(dest) as tmp:
        df.to_csv(tmp, index=False)


def write_json(obj, dest: Path) -> None:
    with atomic_path(dest) as tmp:
        Path(tmp).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def save_npz(dest: Path, **arrays) -> None:
    with atomic_path(dest) as tmp:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)


class StageConflict(RuntimeError):
    pass


def is_done(run_dir: Path, stage: str, key: str, outputs: Iterable[Path], overwrite: bool) -> bool:
    marker = Path(run_dir) / "_done" / f"{stage}.json"
    if overwrite or not marker.exists():
        return False
    info = json.loads(marker.read_text())
    if info.get("key") != key:
        raise StageConflict(
            f"{stage}: già eseguito con un'altra config (hash {info.get('key')} invece di {key}). "
            "Usa --overwrite per rifarlo, o un altro run_dir."
        )
    return all(Path(p).exists() for p in outputs)


def mark_done(run_dir: Path, stage: str, key: str, extra: dict = None) -> None:
    write_json({"stage": stage, "key": key, **(extra or {})}, Path(run_dir) / "_done" / f"{stage}.json")


def require(run_dir: Path, stage: str) -> dict:
    marker = Path(run_dir) / "_done" / f"{stage}.json"
    if not marker.exists():
        raise RuntimeError(f"manca lo stadio {stage!r}: lancialo prima (python -m scenario {stage})")
    return json.loads(marker.read_text())
