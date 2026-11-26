"""Path resolution: one root, ``DL4CV_ROOT``, with optional overrides.

- data: ``$CASCADE_DATA_ROOT`` if set, otherwise ``$DL4CV_ROOT/data``;
- runs: ``$CASCADE_RUN_ROOT`` if set, otherwise ``$DL4CV_ROOT/runs``.

In the YAML files, paths are templates with the placeholders ``{repo}``, ``{data}``, ``{run}`` (and any other
field passed to :func:`resolve`), so there are no absolute paths in the code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from src import REPO_ROOT, dl4cv_root

DATA_ENV = "CASCADE_DATA_ROOT"
RUN_ENV = "CASCADE_RUN_ROOT"


def _env_dir(var: str) -> Optional[Path]:
    value = os.environ.get(var)
    if not value:
        return None
    return Path(value).expanduser()


def data_root() -> Path:
    """Data root."""
    override = _env_dir(DATA_ENV)
    return override if override is not None else dl4cv_root() / "data"


def run_root() -> Path:
    """Runs root."""
    override = _env_dir(RUN_ENV)
    return override if override is not None else dl4cv_root() / "runs"


def run_dir(run_name: str) -> Path:
    """``<run_root>/<run_name>``; the name cannot contain path separators."""
    if not run_name or "/" in run_name or run_name in {".", ".."}:
        raise ValueError(f"run_name non valido: {run_name!r}")
    return run_root() / run_name


def resolve(template: str, *, data: Optional[Path] = None, run: Optional[Path] = None, **fields: object) -> Path:
    """Expand a path template.

    ``{repo}`` is the repository root, ``{data}`` the data root (or ``data``), ``{run}`` the run_dir.
    A template without placeholders is relative to the data root.
    """
    values = {"repo": str(REPO_ROOT), **{k: str(v) for k, v in fields.items()}}
    if "{data}" in template or not any(t in template for t in ("{repo}", "{run}")):
        values["data"] = str(data if data is not None else data_root())
    if "{run}" in template:
        if run is None:
            raise ValueError(f"il template {template!r} richiede la run_dir")
        values["run"] = str(run)
    expanded = template.format(**values)
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = Path(values["data"]) / path
    return path
