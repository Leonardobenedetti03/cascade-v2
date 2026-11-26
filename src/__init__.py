"""Root package of the project.

It only knows where things are. Data and runs live under the folder in the environment variable
``DL4CV_ROOT`` (with the overrides in ``src/cascade/utils/paths.py``), so no data path is hardcoded.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "DL4CV_ROOT"

#: Repository root, used only for files kept in git (configs/; splits/ was for the first pipeline).
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
SPLITS_DIR = REPO_ROOT / "splits"


def dl4cv_root() -> Path:
    """Return ``$DL4CV_ROOT`` as a ``Path``.

    Raises:
        RuntimeError: if the variable is not set or does not point to a folder.
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
