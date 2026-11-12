"""Logging su file (nella run_dir) e su console."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(log_file: Optional[Path] = None, level: str = "INFO") -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(FMT))
    root.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(FMT))
        root.addHandler(fh)
    for noisy in ("matplotlib", "PIL", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("cascade")


def progress(iterable, total=None, desc: str = ""):
    """Barra di avanzamento solo se stderr è un terminale (mai nei file di log né in SLURM)."""
    if not sys.stderr.isatty():
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)
