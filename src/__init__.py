"""Pacchetto principale della cascata CamFormer -> TimeSformer-EgoVLP.

Qui c'è solo la radice dei percorsi: dati e run vivono sotto la directory indicata
dalla variabile d'ambiente ``DL4CV_ROOT`` (con gli override di ``src/cascade/paths.py``)
e non sono mai hardcoded nel codice. Le fold sono versionate nel repo, in ``splits/``.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "DL4CV_ROOT"

#: Radice del repository (serve solo per file versionati: configs/, splits/).
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
SPLITS_DIR = REPO_ROOT / "splits"


def dl4cv_root() -> Path:
    """Restituisce ``$DL4CV_ROOT`` come ``Path``.

    Raises:
        RuntimeError: se la variabile non è impostata o non punta a una directory.
    """
    value = os.environ.get(ENV_VAR)
    if not value:
        raise RuntimeError(
            f"La variabile d'ambiente {ENV_VAR} non è impostata. " f"Esempio: export {ENV_VAR}=/scratch/$USER/dl4cv"
        )
    root = Path(value).expanduser()
    if not root.is_dir():
        raise RuntimeError(f"{ENV_VAR}={value!r} non è una directory esistente.")
    return root
