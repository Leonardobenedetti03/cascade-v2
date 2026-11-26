"""Configuration: YAML plus ``--set dotted.key=value`` overrides, with environment variables expanded.

Rules: an override key that does not exist in the YAML is an error (a typo must not turn into a silent
default). An undefined ``${VAR}`` stays in the text, and it becomes an error only when that value is used
as a path (:func:`path`). This way the stages that do not use it still run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(obj: Any) -> Any:
    if isinstance(obj, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return obj


def apply_override(cfg: Dict[str, Any], item: str) -> None:
    if "=" not in item:
        raise ValueError(f"--set vuole chiave=valore, trovato {item!r}")
    key, raw = item.split("=", 1)
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        if not isinstance(node, dict) or p not in node:
            raise KeyError(f"--set: chiave sconosciuta {key!r}")
        node = node[p]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise KeyError(f"--set: chiave sconosciuta {key!r}")
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(path: Path, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    for item in overrides:
        apply_override(cfg, item)
    return _expand(cfg)


def path(value: str) -> Path:
    """Path from a config value, with a clear error if an undefined variable is left in it."""
    m = _ENV.search(str(value))
    if m:
        raise EnvironmentError(f"variabile d'ambiente {m.group(1)} non definita (usata in {value!r}): vedi hpc/env.sh")
    return Path(value).expanduser()


def section_hash(cfg: Dict[str, Any], keys: Iterable[str]) -> str:
    """Hash of the config sections that determine the output of a stage (used to resume)."""
    sub = {k: cfg.get(k) for k in sorted(keys)}
    return hashlib.sha256(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()[:16]


def snapshot(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(cfg)
