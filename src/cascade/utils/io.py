"""Scritture atomiche: file temporaneo con nome univoco nella stessa cartella, poi rename.

Un processo interrotto a metà scrittura lascia al massimo un ``.tmp`` orfano, mai un
output troncato con il nome definitivo. Il nome temporaneo è univoco per processo,
quindi anche job concorrenti (array SLURM) che scrivono lo stesso file non si
corrompono a vicenda: vince l'ultimo rename, e ogni rename porta un file completo.
Il suffisso finale viene preservato (``np.save``/``np.savez``/``savefig`` lo usano per
decidere il formato).
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@contextlib.contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Restituisce un percorso temporaneo; al termine senza errori lo rinomina in ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp{path.suffix}")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_text(path: Path, text: str) -> None:
    with atomic_path(path) as tmp:
        tmp.write_text(text)


def write_json(path: Path, obj: Any) -> None:
    with atomic_path(path) as tmp:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2, default=json_default)


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def json_default(o: object) -> object:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"non serializzabile in JSON: {type(o)}")


def write_csv(df, path: Path, **kwargs: Any) -> None:
    kwargs.setdefault("index", False)
    with atomic_path(path) as tmp:
        df.to_csv(tmp, **kwargs)


def write_parquet_df(df, path: Path) -> None:
    with atomic_path(path) as tmp:
        df.to_parquet(tmp, index=False)


def save_npy(path: Path, arr: np.ndarray) -> None:
    with atomic_path(path) as tmp:
        np.save(tmp, arr)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    with atomic_path(path) as tmp:
        np.savez(tmp, **arrays)


def save_fig(fig, path: Path, **kwargs: Any) -> None:
    with atomic_path(path) as tmp:
        fig.savefig(tmp, **kwargs)


def torch_save(obj: Any, path: Path) -> None:
    import torch

    with atomic_path(path) as tmp:
        torch.save(obj, tmp)
