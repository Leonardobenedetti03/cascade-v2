"""Reading the outputs of scenario_preds. Shared by the old bridge stage 44 and by the gate v2 (stage 56, N37).

These functions used to live in the script ``44_scenario_bridge.py``. We moved them here unchanged, so that
the gate v2 computes motion exactly like the cache of the gate v1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.cascade.data.motion import MOTION_FEATURES, motion_features


def compute_windows(duration_s: float, length_s: float, n_windows: int, placement: str) -> List[Tuple[float, float]]:
    """Same as ``data/windows.py::compute_windows`` (once ``scenario/windows.py``), so the windows match."""
    if not np.isfinite(duration_s) or duration_s <= 0:
        return []
    if duration_s <= length_s:
        return [(0.0, float(duration_s))]
    if placement == "uniform":
        centers = [(i + 0.5) * duration_s / n_windows for i in range(n_windows)]
    elif placement == "center":
        mid = duration_s / 2.0
        centers = [mid + (i - (n_windows - 1) / 2.0) * length_s for i in range(n_windows)]
    else:
        raise ValueError(f"window.placement sconosciuto: {placement!r}")
    out = []
    for c in centers:
        start = min(max(c - length_s / 2.0, 0.0), duration_s - length_s)
        w = (round(start, 6), round(start + length_s, 6))
        if w not in out:
            out.append(w)
    return out


def take_motion(traj_path: Path, windows, fps: float, vertical_axis: int) -> dict:
    """Motion features averaged over the windows; NaN if the trajectory is missing (the gate imputes them)."""
    if not traj_path.exists() or not windows:
        return {k: float("nan") for k in MOTION_FEATURES}
    with np.load(traj_path) as z:
        t, poses, valid = z["t"], z["poses"], z["valid"].astype(bool)
    feats = []
    for t0, t1 in windows:
        sel = (t >= t0) & (t < t1) & valid
        feats.append(motion_features(poses[sel], fps, vertical_axis))
    with np.errstate(all="ignore"):
        return {k: float(np.nanmean([f[k] for f in feats])) if feats else float("nan") for k in MOTION_FEATURES}


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    """``take_name -> embedding`` from the ``features/<model>.npz`` file of scenario_preds."""
    with np.load(path) as z:
        return {str(n): x for n, x in zip(z["take_name"], z["X"])}
