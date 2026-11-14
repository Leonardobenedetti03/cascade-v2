"""Feature di moto (gruppo C del gate), in numpy, dalla traiettoria della finestra.

Input: pose ``[N, 7]`` (tx, ty, tz, qx, qy, qz, qw nel mondo) campionate a ``fps``.

Feature:
- ``mot_speed_mean``, ``mot_speed_std``: velocità di traslazione [m/s];
- ``mot_angvel_mean``, ``mot_angvel_std``: velocità angolare [rad/s];
- ``mot_jerk_mean``: norma media della derivata terza della posizione [m/s³];
- ``mot_vert_range``: escursione verticale robusta (p95 − p5) [m];
- ``mot_energy``: energia totale del moto, ``mean(v²) + mean(ω²)``;
- ``mot_zfreq_dom``: frequenza dominante della quota (FFT dopo detrend lineare) [Hz];
- ``mot_zfreq_peak``: potenza del picco / potenza totale (DC esclusa), in [0, 1].

Con meno di ``MIN_SAMPLES`` campioni le feature sono NaN (il gate le imputa).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

MOTION_FEATURES = [
    "mot_speed_mean",
    "mot_speed_std",
    "mot_angvel_mean",
    "mot_angvel_std",
    "mot_jerk_mean",
    "mot_vert_range",
    "mot_energy",
    "mot_zfreq_dom",
    "mot_zfreq_peak",
]
MIN_SAMPLES = 8


def _angular_speed(quats: np.ndarray, fps: float) -> np.ndarray:
    q = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    dots = np.abs(np.sum(q[1:] * q[:-1], axis=1))
    return 2.0 * np.arccos(np.clip(dots, -1.0, 1.0)) * fps


def dominant_frequency(x: np.ndarray, fps: float):
    """(frequenza dominante [Hz], forza relativa del picco) dopo detrend lineare."""
    n = len(x)
    t = np.arange(n)
    coef = np.polyfit(t, x, 1)
    xd = x - np.polyval(coef, t)
    xd = xd * np.hanning(n)
    power = np.abs(np.fft.rfft(xd)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    power, freqs = power[1:], freqs[1:]
    total = power.sum()
    if total <= 0:
        return 0.0, 0.0
    k = int(np.argmax(power))
    # interpolazione parabolica del picco per una frequenza sub-bin
    if 0 < k < len(power) - 1:
        a, b, c = np.log(power[k - 1] + 1e-30), np.log(power[k] + 1e-30), np.log(power[k + 1] + 1e-30)
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    df = freqs[1] - freqs[0] if len(freqs) > 1 else fps / n
    return float(freqs[k] + delta * df), float(power[k] / total)


def motion_features(poses: np.ndarray, fps: float, vertical_axis: int = 2) -> Dict[str, float]:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"attese pose [N, 7], trovate {poses.shape}")
    if len(poses) < MIN_SAMPLES:
        return {k: float("nan") for k in MOTION_FEATURES}
    pos = poses[:, :3]
    vel = np.diff(pos, axis=0) * fps
    speed = np.linalg.norm(vel, axis=1)
    ang = _angular_speed(poses[:, 3:], fps)
    jerk = np.linalg.norm(np.diff(pos, n=3, axis=0), axis=1) * fps**3
    z = pos[:, vertical_axis]
    f_dom, peak = dominant_frequency(z, fps)
    return {
        "mot_speed_mean": float(speed.mean()),
        "mot_speed_std": float(speed.std()),
        "mot_angvel_mean": float(ang.mean()),
        "mot_angvel_std": float(ang.std()),
        "mot_jerk_mean": float(jerk.mean()),
        "mot_vert_range": float(np.percentile(z, 95) - np.percentile(z, 5)),
        "mot_energy": float(np.mean(speed**2) + np.mean(ang**2)),
        "mot_zfreq_dom": f_dom,
        "mot_zfreq_peak": peak,
    }
