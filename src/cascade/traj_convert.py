"""Conversione delle pose per frame di Ego-Exo4D in traiettorie uniformi a 20 Hz (stadio 01, M08).

Porta in ``src/`` la logica di ``hpc/make_traj_data.py`` della v1, che girava fuori dal repository.

Sorgente (identica alla v1): ogni radice del dataset contiene ``split_manifest.csv`` con le colonne
``take_name`` e ``camera_pose_file``; il file indicato (``annotations/ego_pose/*/camera_pose/<take>.json``) ha una
chiave per camera (``aria01``, ...) con ``camera_extrinsics``: ``{indice di frame: matrice 3x4 mondo -> camera}``
a circa 30 Hz. Il tempo non è scritto nel file: il frame ``i`` vale ``i / data.pose_src_fps`` secondi dall'inizio del
video, quindi la traiettoria nasce già nel tempo del video e ``alignment.mode`` deve essere ``same_origin``.

Che cosa fa, per ogni take:
1. sceglie la camera: quella del video che si usa davvero (``aria01_214-1.mp4`` -> ``aria01``); se manca nel JSON e
   c'è una sola camera ``aria*``, quella; altrimenti la take è scartata (``cam_ambiguous``);
2. porta le pose da mondo -> camera a camera -> mondo (centro ``C = -R^T t``, quaternione di ``R^T``, ordine
   ``x, y, z, w``);
3. ricampiona a ``data.traj_fps`` con interpolazione lineare delle posizioni e slerp delle rotazioni; tra due frame
   sorgente distanti più di ``data.pose_max_gap_frames`` la traiettoria non c'è (``valid = False``).

Differenze dalla v1 (registro M08): il risultato è un ``.npz`` con la griglia uniforme completa e la maschera
``valid``, non un CSV con righe mancanti; i campioni prima della prima posa sorgente (v1: copia della prima posa,
contata come valida) sono marcati non validi; i quaternioni hanno segno continuo da un campione al successivo
(stessa rotazione, nessun salto di segno).
"""

from __future__ import annotations

import csv
import glob
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.cascade.io import save_npz

log = logging.getLogger(__name__)

POSE_MANIFEST = "split_manifest.csv"
OK = "ok"
STATUS_FIELDS = [
    "take_name",
    "root",
    "status",
    "cam",
    "n_src",
    "first_idx",
    "last_idx",
    "gaps",
    "dropped",
    "lead_fill",
    "n_out",
    "dur_s",
    "z_mean",
    "z_std",
    "path_m",
]
SLERP_LINEAR = 0.9995  # sopra questo prodotto scalare lo slerp si riduce all'interpolazione lineare


class ConversionError(Exception):
    """Take non convertibile: ``status`` è il codice scritto in status.csv."""

    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


@dataclass
class Converted:
    t: np.ndarray  # [N] secondi, tempo del video: t[k] = k / fps
    poses: np.ndarray  # [N, 7] tx, ty, tz, qx, qy, qz, qw (camera -> mondo)
    valid: np.ndarray  # [N] bool: False nei buchi lunghi e prima della prima posa
    stats: Dict[str, Any]


@dataclass(frozen=True)
class PoseSource:
    root: Path
    pose_file: str  # relativo alla radice


# --------------------------------------------------------------------------- geometria
def quat_from_rotmat(R: np.ndarray) -> np.ndarray:
    """Quaternioni ``x, y, z, w`` normalizzati da matrici di rotazione ``[N, 3, 3]`` (stessi rami della v1)."""
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    c0 = tr > 0
    c1 = ~c0 & (m00 > m11) & (m00 > m22)
    c2 = ~c0 & ~c1 & (m11 > m22)
    with np.errstate(invalid="ignore", divide="ignore"):
        s0 = np.sqrt(np.maximum(tr + 1.0, 0.0)) * 2
        s1 = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 0.0)) * 2
        s2 = np.sqrt(np.maximum(1.0 + m11 - m00 - m22, 0.0)) * 2
        s3 = np.sqrt(np.maximum(1.0 + m22 - m00 - m11, 0.0)) * 2
        b0 = np.stack([(m21 - m12) / s0, (m02 - m20) / s0, (m10 - m01) / s0, 0.25 * s0], axis=1)
        b1 = np.stack([0.25 * s1, (m01 + m10) / s1, (m02 + m20) / s1, (m21 - m12) / s1], axis=1)
        b2 = np.stack([(m01 + m10) / s2, 0.25 * s2, (m12 + m21) / s2, (m02 - m20) / s2], axis=1)
        b3 = np.stack([(m02 + m20) / s3, (m12 + m21) / s3, 0.25 * s3, (m10 - m01) / s3], axis=1)
    q = np.where(c0[:, None], b0, np.where(c1[:, None], b1, np.where(c2[:, None], b2, b3)))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def camera_to_world(E: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Da estrinseche mondo -> camera ``[N, 3, 4]`` a (centro camera ``[N, 3]``, quaternione ``[N, 4]``)."""
    R = E[:, :, :3]
    t = E[:, :, 3]
    Rwc = np.transpose(R, (0, 2, 1))
    centre = -np.einsum("nij,nj->ni", Rwc, t)
    return centre, quat_from_rotmat(Rwc)


def slerp(q0: np.ndarray, q1: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Slerp tra quaternioni ``[M, 4]`` con pesi ``a [M]``; prende il cammino corto, lineare per angoli piccoli."""
    d = np.sum(q0 * q1, axis=1)
    q1 = np.where((d < 0)[:, None], -q1, q1)
    d = np.abs(d)
    theta = np.arccos(np.minimum(1.0, d))
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.sin(theta)
        w0 = np.where(d > SLERP_LINEAR, 1.0 - a, np.sin((1.0 - a) * theta) / s)
        w1 = np.where(d > SLERP_LINEAR, a, np.sin(a * theta) / s)
    q = w0[:, None] * q0 + w1[:, None] * q1
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def align_quat_signs(q: np.ndarray) -> np.ndarray:
    """Rende continuo il segno dei quaternioni (``q`` e ``-q`` sono la stessa rotazione): prodotto scalare >= 0."""
    if len(q) < 2:
        return q
    flips = np.concatenate([[0], (np.sum(q[1:] * q[:-1], axis=1) < 0).astype(np.int64)])
    sign = np.where(np.cumsum(flips) % 2 == 1, -1.0, 1.0)
    return q * sign[:, None]


# --------------------------------------------------------------------------- una take
def convert_extrinsics(
    ext: Any, *, src_fps: float = 30.0, out_fps: float = 20.0, max_gap_frames: int = 15
) -> Converted:
    """Estrinseche per frame (``{indice: matrice 3x4}``) -> traiettoria su griglia uniforme a ``out_fps``.

    Riga ``k`` = istante ``k / out_fps`` = frame sorgente ``k * src_fps / out_fps``. Le righe tra due frame sorgente
    distanti più di ``max_gap_frames`` hanno ``valid = False`` e la posa del campione valido più vicino; così anche
    quelle prima della prima posa sorgente (copia della prima posa, come nella v1).
    """
    if not isinstance(ext, dict) or len(ext) < 2:
        raise ConversionError("ext_not_per_frame")
    try:
        frames = sorted((int(k), k) for k in ext)
    except (TypeError, ValueError):
        raise ConversionError("bad_keys") from None
    idxs = np.array([f for f, _ in frames], dtype=np.int64)
    if len(np.unique(idxs)) != len(idxs):
        raise ConversionError("bad_keys")
    try:
        E = np.array([ext[k] for _, k in frames], dtype=np.float64)
    except (TypeError, ValueError):
        raise ConversionError("bad_matrix") from None
    if E.ndim != 3 or E.shape[1] < 3 or E.shape[2] < 4 or not np.isfinite(E[:, :3, :4]).all():
        raise ConversionError("bad_matrix")
    centre, quat = camera_to_world(E[:, :3, :4])

    step = src_fps / out_fps
    n_out = int(idxs[-1] / step) + 1
    if n_out < 2:
        raise ConversionError("too_short")
    f = np.arange(n_out) * step
    j = np.searchsorted(idxs, f, side="right") - 1  # ultimo frame sorgente <= f
    lead = j < 0  # prima della prima posa
    j0 = np.clip(j, 0, len(idxs) - 1)
    j1 = np.minimum(j0 + 1, len(idxs) - 1)
    exact = ~lead & ((idxs[j0] == f) | (j0 + 1 >= len(idxs)))
    between = ~lead & ~exact
    span = idxs[j1] - idxs[j0]
    dropped = between & (span > max_gap_frames)
    interp = between & ~dropped

    out_c = centre[j0].copy()  # lead ed exact: posa del frame j0 (per lead, j0 = 0)
    out_q = quat[j0].copy()
    if interp.any():
        k = np.flatnonzero(interp)
        a = (f[k] - idxs[j0[k]]) / (idxs[j1[k]] - idxs[j0[k]])
        out_c[k] = centre[j0[k]] + a[:, None] * (centre[j1[k]] - centre[j0[k]])
        out_q[k] = slerp(quat[j0[k]], quat[j1[k]], a)
    valid = ~dropped & ~lead
    if not valid.any():
        raise ConversionError("all_gaps")
    drop_k = np.flatnonzero(dropped)
    if drop_k.size:  # posa del campione valido più vicino (a pari distanza, il precedente)
        ok_k = np.flatnonzero(valid)
        pos = np.searchsorted(ok_k, drop_k)
        left = ok_k[np.clip(pos - 1, 0, len(ok_k) - 1)]
        right = ok_k[np.clip(pos, 0, len(ok_k) - 1)]
        nearest = np.where((drop_k - left) <= (right - drop_k), left, right)
        out_c[drop_k] = out_c[nearest]
        out_q[drop_k] = out_q[nearest]
    out_q = align_quat_signs(out_q)

    cv = out_c[valid]
    path_m = float(np.linalg.norm(np.diff(cv, axis=0), axis=1).sum()) if len(cv) > 1 else 0.0
    stats = {
        "n_src": int(len(idxs)),
        "first_idx": int(idxs[0]),
        "last_idx": int(idxs[-1]),
        "gaps": int((interp & (span > 1)).sum()),  # righe interpolate attraverso un buco corto
        "dropped": int(dropped.sum()),
        "lead_fill": int(lead.sum()),
        "n_out": int(n_out),
        "dur_s": round(n_out / out_fps, 3),
        "z_mean": round(float(cv[:, 2].mean()), 3),
        "z_std": round(float(cv[:, 2].std()), 3),
        "path_m": round(path_m, 1),
    }
    return Converted(
        t=np.arange(n_out) / out_fps,
        poses=np.concatenate([out_c, out_q], axis=1),
        valid=valid,
        stats=stats,
    )


def pick_camera(pose_json: Dict[str, Any], want: Optional[str]) -> Tuple[Optional[str], str]:
    """(camera, esito): ``want`` se presente nel JSON, altrimenti l'unica camera ``aria*``, altrimenti ambiguo."""
    if want is not None and want in pose_json:
        return want, OK
    arias = [k for k in pose_json if str(k).startswith("aria")]
    if len(arias) == 1:
        return arias[0], OK
    return None, "cam_ambiguous:" + ",".join(arias)


# --------------------------------------------------------------------------- sorgenti e file
def load_pose_index(roots: Sequence[Path]) -> Dict[str, PoseSource]:
    """``take_name -> (radice, file di pose)`` dai ``split_manifest.csv`` delle radici; vince la prima radice."""
    index: Dict[str, PoseSource] = {}
    duplicates = 0
    for root in roots:
        mf = Path(root) / POSE_MANIFEST
        if not mf.is_file():
            raise FileNotFoundError(f"manca {mf}: ogni radice di pose deve contenere {POSE_MANIFEST}")
        with open(mf, newline="") as fh:
            reader = csv.DictReader(fh)
            missing = {"take_name", "camera_pose_file"} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{mf}: colonne mancanti {sorted(missing)}")
            for row in reader:
                take = row["take_name"]
                if take in index:
                    duplicates += 1
                    continue
                index[take] = PoseSource(Path(root), row["camera_pose_file"])
    if duplicates:
        log.warning("%d take presenti in più radici: vale la prima", duplicates)
    return index


def find_video(root: Path, take: str, video_pattern: str, video_glob: str) -> Optional[Path]:
    """Video della take sotto una radice, come nella v1: prima il nome esatto, poi il glob (primo in ordine).

    I due modelli possono usare solo ``{takes_root}`` (= ``<radice>/takes``) e ``{take_name}``.
    """
    fields = {"takes_root": str(Path(root) / "takes"), "take_name": take}
    p = Path(video_pattern.format(**fields))
    if p.exists():
        return p
    found = sorted(glob.glob(video_glob.format(**fields)))
    return Path(found[0]) if found else None


def convert_take(
    take: str,
    index: Dict[str, PoseSource],
    *,
    video_pattern: str,
    video_glob: str,
    src_fps: float,
    out_fps: float,
    max_gap_frames: int,
) -> Tuple[Optional[Converted], Dict[str, Any]]:
    """Converte una take. Restituisce (traiettoria o None, riga di status.csv)."""
    row: Dict[str, Any] = {k: "" for k in STATUS_FIELDS}
    row["take_name"] = take
    src = index.get(take)
    if src is None:
        row["status"] = "no_manifest_row"
        return None, row
    row["root"] = str(src.root)
    video = find_video(src.root, take, video_pattern, video_glob)
    if video is None:
        row["status"] = "no_video"
        return None, row
    pose_path = src.root / src.pose_file
    if not pose_path.is_file():
        row["status"] = "no_pose_file"
        return None, row
    try:
        with open(pose_path) as fh:
            pose_json = json.load(fh)
    except (OSError, ValueError):
        row["status"] = "bad_json"
        return None, row
    if not isinstance(pose_json, dict):
        row["status"] = "bad_json"
        return None, row
    cam, status = pick_camera(pose_json, video.name.split("_")[0])
    if cam is None:
        row["status"] = status
        return None, row
    row["cam"] = cam
    cam_data = pose_json[cam]
    ext = cam_data.get("camera_extrinsics") if isinstance(cam_data, dict) else None
    try:
        conv = convert_extrinsics(ext, src_fps=src_fps, out_fps=out_fps, max_gap_frames=max_gap_frames)
    except ConversionError as e:
        row["status"] = e.status
        return None, row
    row.update(conv.stats)
    row["status"] = OK
    return conv, row


def write_npz(path: Path, conv: Converted, *, out_fps: float, src_fps: float) -> None:
    """Scrive la traiettoria (atomico, via ``io.save_npz``). Chiavi lette da ``trajectory.load_converted_npz``."""
    save_npz(
        path,
        t=conv.t,
        poses=conv.poses,
        valid=conv.valid,
        fps=np.float64(out_fps),
        src_fps=np.float64(src_fps),
    )


def summarize(rows: List[Dict[str, Any]], n_takes: int) -> Dict[str, Any]:
    """Riepilogo per summary.json e per il controllo ``data.max_missing_frac``."""
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    ok = [r for r in rows if r["status"] == OK]
    return {
        "n_takes": n_takes,
        "n_ok": len(ok),
        "missing_frac": (n_takes - len(ok)) / n_takes if n_takes else 0.0,
        "status_counts": dict(sorted(counts.items())),
        "takes_with_lead_fill": sum(1 for r in ok if r["lead_fill"] > 0),
        "takes_with_gaps": sum(1 for r in ok if r["gaps"] > 0 or r["dropped"] > 0),
    }
