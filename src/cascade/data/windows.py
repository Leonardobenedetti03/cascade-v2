"""Time windows of a take (video time, in seconds)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def compute_windows(duration_s: float, length_s: float, n_windows: int = 1, placement: str = "center") -> List[Tuple[float, float]]:
    """Up to ``n_windows`` windows ``(t_start, t_end)`` of length ``length_s`` inside ``[0, duration_s]``.

    ``center``: centres ``length_s`` apart around the centre of the take. ``uniform``: centres at the midpoints
    of K equal parts. A take shorter than the window gives one window, the whole take.
    Windows are shifted to stay inside the take, and duplicates are removed.
    """
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
