"""Dataset sintetico con lo stesso layout di quello reale, per ``--synthetic``.

Genera sotto ``<run_dir>/synthetic_data``:

- ``scenario_cls.csv`` (colonne ``take_name, split, task, label`` come CamFormer,
  con la colonna ``label`` in un ordine diverso da labels.yaml, come nel CSV vero);
- ``takes.json`` (``take_uid, take_name, duration_sec, video_start_us, traj_start_us``);
- ``takes/<take>/closed_loop_trajectory.csv`` in formato Aria
  (``tracking_timestamp_us, t{x,y,z}_world_device, q{x,y,z,w}_world_device``);
- ``takes/<take>/aria01_214-1.mp4``: video piccolo con colore per classe e una barra
  la cui altezza segue la quota della traiettoria (serve a 15_alignment_check).

La traiettoria parte ``traj_offset_s`` prima del video (orologio device in µs), così
la modalità di allineamento ``device_us`` viene esercitata davvero. Le classi
procedurali hanno parametri di moto sovrapposti: CamFormer le confonde.
Alcune take sono senza video o senza traiettoria, per il report delle mancanti;
una quota ``video_confusion`` di video ha il colore di un'altra classe (L1 non perfetto).
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.cascade.seed import rng as make_rng

log = logging.getLogger(__name__)

TRAJ_COLUMNS = [
    "tracking_timestamp_us",
    "tx_world_device",
    "ty_world_device",
    "tz_world_device",
    "qx_world_device",
    "qy_world_device",
    "qz_world_device",
    "qw_world_device",
]

# Parametri di moto per classe (nome di labels.yaml):
# freq. di oscillazione verticale [Hz], ampiezza [m], velocità [m/s], yaw [rad/s], salita [m/s]
MOTION_PARAMS: Dict[str, Tuple[float, float, float, float, float]] = {
    "Basketball": (2.6, 0.05, 2.0, 0.8, 0.0),
    "Dance": (1.8, 0.08, 0.6, 1.5, 0.0),
    "Music": (0.4, 0.01, 0.02, 0.1, 0.0),
    "Soccer": (3.0, 0.06, 3.0, 0.6, 0.0),
    "Bouldering": (0.7, 0.03, 0.1, 0.2, 0.3),
    "Cooking": (1.0, 0.015, 0.15, 0.30, 0.0),
    "Health": (1.1, 0.015, 0.12, 0.28, 0.0),
    "Bike repair": (0.9, 0.016, 0.14, 0.32, 0.0),
}

CLASS_COLORS = np.array(
    [
        [220, 60, 60],
        [60, 200, 60],
        [60, 60, 220],
        [220, 220, 60],
        [220, 60, 220],
        [60, 220, 220],
        [240, 150, 40],
        [120, 70, 200],
    ],
    dtype=np.float64,
)

# Ordine della colonna `label` del CSV vero (diverso da labels.yaml).
CSV_LABEL_ORDER = ["Health", "Dance", "Basketball", "Rock Climbing", "Cooking", "Soccer", "Music", "Bike Repair"]


def synthetic_trajectory(
    cls_name: str, duration_s: float, hz: float, rng: np.random.Generator, t_offset_s: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Traiettoria (t [s], pose [N,7] tx,ty,tz,qx,qy,qz,qw) per una classe."""
    from scipy.spatial.transform import Rotation

    f, a, v, yaw_rate, climb = MOTION_PARAMS[cls_name]
    procedural = cls_name in {"Cooking", "Health", "Bike repair"}
    jitter = 0.35 if procedural else 0.12
    f *= 1 + jitter * rng.standard_normal()
    a *= abs(1 + jitter * rng.standard_normal())
    v *= abs(1 + jitter * rng.standard_normal())
    t = np.arange(0.0, duration_s, 1.0 / hz) - t_offset_s
    phase = rng.uniform(0, 2 * np.pi)
    heading = rng.uniform(0, 2 * np.pi) + 0.3 * np.sin(2 * np.pi * 0.1 * t)
    x = np.cumsum(v * np.cos(heading)) / hz
    y = np.cumsum(v * np.sin(heading)) / hz
    z = 1.6 + a * np.sin(2 * np.pi * f * t + phase) + climb * np.clip(t, 0, None)
    pos = np.stack([x, y, z], 1) + 0.002 * rng.standard_normal((len(t), 3))
    yaw = np.cumsum(np.full(len(t), yaw_rate) * np.sign(np.sin(0.2 * t + phase))) / hz
    pitch = -0.4 + 0.15 * np.sin(2 * np.pi * f * t + phase + 0.5)
    quat = Rotation.from_euler("zyx", np.stack([yaw, pitch, np.zeros_like(t)], 1)).as_quat()
    return t + t_offset_s, np.concatenate([pos, quat], 1)


def write_video(path, frames: np.ndarray, fps: float) -> None:
    """Scrive frame uint8 [T,H,W,3] in mp4 con PyAV."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        try:
            stream = container.add_stream("libx264", rate=int(round(fps)))
        except Exception:  # noqa: BLE001 - build di ffmpeg senza x264
            stream = container.add_stream("mpeg4", rate=int(round(fps)))
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        for fr in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def synthetic_video(
    label_idx: int, z_of_t, duration_s: float, fps: float, size: int, rng: np.random.Generator
) -> np.ndarray:
    n = int(round(duration_s * fps))
    t = (np.arange(n) + 0.5) / fps
    base = CLASS_COLORS[label_idx]
    frames = np.empty((n, size, size, 3), dtype=np.uint8)
    z = z_of_t(t)
    zn = (z - z.min()) / (np.ptp(z) + 1e-6)
    for i in range(n):
        img = np.tile(base, (size, size, 1)) + 25 * rng.standard_normal((size, size, 3))
        row = int((1 - zn[i]) * (size - 6))
        img[row : row + 6, :, :] = 255
        frames[i] = np.clip(img, 0, 255).astype(np.uint8)
    return frames


def ensure_synthetic_dataset(ctx) -> None:
    """Genera il dataset sintetico se manca (idempotente, marker ``.complete``)."""
    root = ctx.data_root
    scfg = ctx.cfg.synthetic
    spec = json.dumps({"synthetic": ctx.cfg.to_dict()["synthetic"]}, sort_keys=True)
    marker = root / ".complete"
    if marker.exists() and marker.read_text() == spec:
        return
    log.info("genero il dataset sintetico in %s", root)
    root.mkdir(parents=True, exist_ok=True)
    labels = ctx.scenario_labels or ctx.labels  # i dati sintetici sono sempre sugli 8 scenari
    rng = make_rng(scfg.seed)
    rows, takes = [], []
    for split, n_per in scfg.takes_per_class.items():
        for c, name in enumerate(labels.names):
            for j in range(n_per):
                take_name = f"syn_{name.lower().replace(' ', '')}_{split}_{j:03d}"
                task = labels.csv_tasks[c]
                rows.append(
                    {"take_name": take_name, "split": split, "task": task, "label": CSV_LABEL_ORDER.index(task)}
                )
    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df.to_csv(root / "scenario_cls.csv", index=False)

    takes_root = root / "takes"
    no_video = set(df[df.split == "test"].take_name.iloc[: scfg.missing_video])
    no_traj = set(df[df.split == "val"].take_name.iloc[: scfg.missing_traj])
    for i, r in enumerate(df.itertuples()):
        trng = make_rng(scfg.seed, i + 1)
        c = labels.index_of_task(r.task)
        name = labels.names[c]
        duration = float(trng.uniform(*scfg.duration_s))
        video_start_us = int(1_000_000_000 + i * 1_000_000_000)
        traj_start_us = int(video_start_us - scfg.traj_offset_s * 1e6)
        t, poses = synthetic_trajectory(name, duration + scfg.traj_offset_s + 0.5, scfg.traj_hz, trng)
        tdir = takes_root / r.take_name
        tdir.mkdir(parents=True, exist_ok=True)
        if r.take_name not in no_traj:
            traj = pd.DataFrame(poses, columns=TRAJ_COLUMNS[1:])
            traj.insert(0, "tracking_timestamp_us", (traj_start_us + t * 1e6).astype(np.int64))
            traj.to_csv(tdir / "closed_loop_trajectory.csv", index=False)
        if r.take_name not in no_video:
            # quota in tempo video: t_traj = t_video + offset
            def z_of_t(tv, _t=t, _z=poses[:, 2]):
                return np.interp(tv + scfg.traj_offset_s, _t, _z)

            vc = c  # una quota di video "difficili" ha l'aspetto di un'altra classe
            if trng.random() < scfg.video_confusion:
                vc = int(trng.choice([k for k in range(labels.num_classes) if k != c]))
            frames = synthetic_video(vc, z_of_t, duration, scfg.video_fps, scfg.video_size, trng)
            write_video(tdir / "aria01_214-1.mp4", frames, scfg.video_fps)
        takes.append(
            {
                "take_uid": f"uid-{i:05d}",
                "take_name": r.take_name,
                "duration_sec": duration,
                "video_start_us": video_start_us,
                "traj_start_us": traj_start_us,
            }
        )
    with open(root / "takes.json", "w") as f:
        json.dump(takes, f)
    marker.write_text(spec)
    log.info("dataset sintetico: %d take (%s)", len(df), df.split.value_counts().to_dict())
