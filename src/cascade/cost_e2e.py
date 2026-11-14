"""Costo end-to-end (stadio 41, M10): tempo per fase di L0 e L1 e formula di pareggio della traiettoria.

Funzioni pure, senza torch: la misura vera (modelli, video, traiettorie) sta in ``scripts/41_cost_e2e.py``.

    T_L0 = t lettura traiettoria + t preprocessing + t forward
    T_L1 = t apertura e decodifica + t preprocessing + t forward
    T(r) = T_L0 + r · T_L1,   fattore di miglioramento in tempo = T_L1 / T(r)

Se stimare la traiettoria costa ``C_pose`` per clip, la cascata conviene rispetto a usare sempre L1 quando
``C_pose < (1 − r) · C_L1 − C_L0 − C_g`` (eq. 6 del blueprint); lo stesso vale in tempo.
"""

from __future__ import annotations

import time
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

L0_PHASES = ("l0_read_s", "l0_pre_s", "l0_fwd_s")
L1_PHASES = ("l1_decode_s", "l1_pre_s", "l1_fwd_s")
TIME_COLUMNS = ["clip_id", *L0_PHASES, *L1_PHASES, "l0_total_s", "l1_total_s"]


def timed(fn, *args, sync=None):
    """(risultato, secondi). ``sync`` = ``torch.cuda.synchronize`` sulle fasi che usano la GPU."""
    if sync:
        sync()
    t0 = time.perf_counter()
    out = fn(*args)
    if sync:
        sync()
    return out, time.perf_counter() - t0


def clip_times(clip, l0, l1, sync=None) -> Dict:
    """Una riga della tabella: stesse fasi per i due livelli, batch di una clip."""
    row = {"clip_id": clip.clip_id}
    item, row["l0_read_s"] = timed(l0.read_window, clip)  # file della traiettoria
    x, row["l0_pre_s"] = timed(l0.preprocess, item, sync=sync)  # pose relative + gravità, batch
    _, row["l0_fwd_s"] = timed(l0.forward, x, sync=sync)
    raw, row["l1_decode_s"] = timed(l1.decode, clip)  # apertura del video + frame
    v, row["l1_pre_s"] = timed(l1.preprocess, raw, sync=sync)  # ridimensiona, ritaglia, normalizza
    _, row["l1_fwd_s"] = timed(l1.forward, v, sync=sync)
    row["l0_total_s"] = sum(row[k] for k in L0_PHASES)
    row["l1_total_s"] = sum(row[k] for k in L1_PHASES)
    return row


def summarize(times: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Mediana e 90° percentile per ogni fase (non la media: i tempi sul nodo condiviso hanno code lunghe)."""
    if len(times) == 0:
        return {}
    out = {}
    for c in TIME_COLUMNS[1:]:
        v = times[c].to_numpy(float)
        out[c] = {"median_s": float(np.median(v)), "p90_s": float(np.percentile(v, 90)), "n": int(len(v))}
    return out


def cascade_time(t_l0: float, t_l1: float, r: float) -> float:
    """T(r) = T_L0 + r · T_L1, con r = frazione di clip che scalano a L1."""
    if not 0.0 <= r <= 1.0:
        raise ValueError("r deve stare in [0, 1]")
    return t_l0 + r * t_l1


def time_speedup(t_l0: float, t_l1: float, r: float) -> float:
    """Fattore di miglioramento in tempo rispetto a usare sempre L1: T_L1 / T(r)."""
    return t_l1 / cascade_time(t_l0, t_l1, r)


def time_curve(t_l0: float, t_l1: float, rs: Sequence[float]) -> List[Dict[str, float]]:
    return [
        {"r": float(r), "t_cascade_s": cascade_time(t_l0, t_l1, r), "speedup": time_speedup(t_l0, t_l1, r)} for r in rs
    ]


def pose_breakeven(r: float, c_l1: float, c_l0: float, c_gate: float = 0.0) -> float:
    """Costo massimo per stimare la traiettoria di una clip: (1 − r) · C_L1 − C_L0 − C_g (GMACs o secondi)."""
    if not 0.0 <= r <= 1.0:
        raise ValueError("r deve stare in [0, 1]")
    return (1.0 - r) * c_l1 - c_l0 - c_gate


def stratified_sample(clips: pd.DataFrame, n: int, seed: int = 0, by: str = "label_idx") -> pd.DataFrame:
    """Fino a ``n`` clip, il più possibile bilanciate per classe (a turno, una per classe); deterministico."""
    rng = np.random.default_rng(seed)
    if n >= len(clips):
        return clips.iloc[rng.permutation(len(clips))].reset_index(drop=True)
    labels = clips[by].to_numpy()
    pools = {k: list(rng.permutation(np.flatnonzero(labels == k))) for k in sorted(set(labels))}
    chosen: List[int] = []
    while len(chosen) < n and any(pools.values()):
        for k in pools:
            if pools[k] and len(chosen) < n:
                chosen.append(int(pools[k].pop()))
    return clips.iloc[chosen].reset_index(drop=True)
