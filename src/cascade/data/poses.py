"""Camera trajectories for CamFormer (stage ``traj``), and a diagnostic for the reference frame.

There are two sources. Both are resampled on a uniform grid at ``poses.fps`` (20 Hz) in video time.

- ``aria_csv``: ``closed_loop_trajectory.csv`` (MPS, 1 kHz, device -> world pose, world with gravity
  along -z). This is the source CamFormer was pre-trained on (``data_prep/step1_extract_segments.py``):
  time = seconds from the first sample, downsampled to the nearest sample. The authors used the same time
  convention to align the narrations, so we use it for our windows too.
- ``pose_json``: the per-frame ``camera_extrinsics`` (world -> RGB camera, ~30 Hz) in the JSON files
  ``annotations/ego_pose/*/camera_pose``. Same logic as ``traj_convert.py`` in cascade v2: inversion,
  linear interpolation + slerp, long gaps marked invalid. Here the pose is the RGB camera's, not the
  device's, so ``json_cam_to_device`` and ``json_world_rotation`` bring it back to the pre-training
  convention (estimate them with ``check-frame``).

Pose format: ``[N, 7] = tx, ty, tz, qx, qy, qz, qw`` (device -> world), as in CamFormer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

log = logging.getLogger(__name__)

CSV_TIME = "tracking_timestamp_us"
CSV_POSE = [
    "tx_world_device",
    "ty_world_device",
    "tz_world_device",
    "qx_world_device",
    "qy_world_device",
    "qz_world_device",
    "qw_world_device",
]
SLERP_LINEAR = 0.9995


class PoseError(Exception):
    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


@dataclass
class Trajectory:
    t: np.ndarray  # [N] seconds, uniform grid
    poses: np.ndarray  # [N, 7]
    valid: np.ndarray  # [N] bool
    source: str
    n_missing_src: int = 0  # source samples missing between the first and the last one (inner gaps)
    first_src_frame: int = 0  # pose_json: index of the first frame with a pose (> 0 = missing start)

    def save(self, dest: Path) -> None:
        from src.cascade.utils.feature_io import save_npz

        save_npz(dest, t=self.t, poses=self.poses, valid=self.valid, source=np.array(self.source),
                 n_missing_src=np.array(self.n_missing_src), first_src_frame=np.array(self.first_src_frame))

    @classmethod
    def load(cls, path: Path) -> "Trajectory":
        with np.load(path) as z:
            extra = {k: int(z[k]) for k in ("n_missing_src", "first_src_frame") if k in z.files}
            return cls(z["t"], z["poses"], z["valid"].astype(bool), str(z["source"]), **extra)


# Geometry
def slerp(q0: np.ndarray, q1: np.ndarray, a: np.ndarray) -> np.ndarray:
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
    """Keep the sign of the quaternions continuous (q and -q are the same rotation)."""
    if len(q) < 2:
        return q
    flips = np.concatenate([[0], (np.sum(q[1:] * q[:-1], axis=1) < 0).astype(np.int64)])
    return q * np.where(np.cumsum(flips) % 2 == 1, -1.0, 1.0)[:, None]


def extrinsics_to_c2w(E: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """World -> camera extrinsics ``[N, 3, 4]`` -> (centre ``[N, 3]``, camera -> world quaternion ``[N, 4]``)."""
    R, t = E[:, :, :3], E[:, :, 3]
    Rwc = np.transpose(R, (0, 2, 1))
    centre = -np.einsum("nij,nj->ni", Rwc, t)
    return centre, Rotation.from_matrix(Rwc).as_quat()


def apply_frame_correction(poses: np.ndarray, cam_to_device: np.ndarray, world_rot: np.ndarray) -> np.ndarray:
    """R_w'd = W @ R_wc @ X, centre' = W @ centre (we ignore the camera-device lever arm)."""
    X, W = np.asarray(cam_to_device, float), np.asarray(world_rot, float)
    if np.allclose(X, np.eye(3)) and np.allclose(W, np.eye(3)):
        return poses
    for M, name in ((X, "json_cam_to_device"), (W, "json_world_rotation")):
        if not np.allclose(M @ M.T, np.eye(3), atol=1e-4) or np.linalg.det(M) < 0:
            raise ValueError(f"poses.{name} non è una rotazione")
    R = Rotation.from_quat(poses[:, 3:]).as_matrix()
    R = np.einsum("ij,njk,kl->nil", W, R, X)
    out = poses.copy()
    out[:, :3] = poses[:, :3] @ W.T
    out[:, 3:] = align_quat_signs(Rotation.from_matrix(R).as_quat())
    return out


# Sources
def from_aria_csv(path: Path, fps: float) -> Trajectory:
    import pandas as pd

    if not Path(path).is_file():
        raise PoseError("no_aria_csv")
    df = pd.read_csv(path, usecols=[CSV_TIME] + CSV_POSE).dropna()
    if len(df) < 2:
        raise PoseError("too_short")
    ts = df[CSV_TIME].to_numpy(np.float64)
    order = np.argsort(ts, kind="stable")
    t = (ts[order] - ts[order][0]) / 1e6
    p = df[CSV_POSE].to_numpy(np.float64)[order]
    grid = np.arange(0.0, t[-1] + 1e-9, 1.0 / fps)
    j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    j = np.where((grid - t[j - 1]) <= (t[j] - grid), j - 1, j)
    valid = np.abs(t[j] - grid) <= 1.0 / fps
    out = p[j].copy()
    out[:, 3:] /= np.linalg.norm(out[:, 3:], axis=1, keepdims=True)
    out[:, 3:] = align_quat_signs(out[:, 3:])
    return Trajectory(grid, out, valid, "aria_csv", n_missing_src=int((~valid).sum()))


def pick_camera(pose_json: Dict, video_path: Optional[str]) -> str:
    want = Path(video_path).name.split("_")[0] if video_path else None
    if want and want in pose_json:
        return want
    arias = [k for k in pose_json if str(k).startswith("aria")]
    if len(arias) == 1:
        return arias[0]
    raise PoseError("cam_ambiguous")


def from_pose_json(path: Path, video_path: Optional[str], pcfg: dict) -> Trajectory:
    if not Path(path).is_file():
        raise PoseError("no_pose_json")
    with open(path) as fh:
        js = json.load(fh)
    cam = js[pick_camera(js, video_path)]
    ext = cam.get("camera_extrinsics") if isinstance(cam, dict) else None
    if not isinstance(ext, dict) or len(ext) < 2:
        raise PoseError("ext_not_per_frame")
    keys = sorted(ext, key=int)
    idxs = np.array([int(k) for k in keys])
    E = np.array([ext[k] for k in keys], dtype=np.float64)[:, :3, :4]
    centre, quat = extrinsics_to_c2w(E)
    quat = align_quat_signs(quat)

    step = pcfg["json_src_fps"] / pcfg["fps"]
    n_out = int(idxs[-1] / step) + 1
    f = np.arange(n_out) * step
    j = np.searchsorted(idxs, f, side="right") - 1
    lead = j < 0
    j0 = np.clip(j, 0, len(idxs) - 1)
    j1 = np.minimum(j0 + 1, len(idxs) - 1)
    exact = ~lead & ((idxs[j0] == f) | (j0 + 1 >= len(idxs)))
    between = ~lead & ~exact
    dropped = between & ((idxs[j1] - idxs[j0]) > pcfg["json_max_gap_frames"])
    interp = between & ~dropped
    out_c, out_q = centre[j0].copy(), quat[j0].copy()
    if interp.any():
        k = np.flatnonzero(interp)
        a = (f[k] - idxs[j0[k]]) / (idxs[j1[k]] - idxs[j0[k]])
        out_c[k] = centre[j0[k]] + a[:, None] * (centre[j1[k]] - centre[j0[k]])
        out_q[k] = slerp(quat[j0[k]], quat[j1[k]], a)
    poses = np.concatenate([out_c, align_quat_signs(out_q)], axis=1)
    poses = apply_frame_correction(poses, pcfg["json_cam_to_device"], pcfg["json_world_rotation"])
    n_missing = int(idxs[-1] - idxs[0] + 1 - len(idxs))
    return Trajectory(np.arange(n_out) / pcfg["fps"], poses, ~dropped & ~lead, "pose_json",
                      n_missing_src=n_missing, first_src_frame=int(idxs[0]))


def load_take(row, pcfg: dict) -> Trajectory:
    """Trajectory of one take, following ``poses.source`` (and ``fallback_to_json``)."""
    if pcfg["source"] == "aria_csv":
        try:
            return from_aria_csv(Path(row.aria_csv_path), pcfg["fps"])
        except PoseError as e:
            if not (pcfg["fallback_to_json"] and e.status == "no_aria_csv"):
                raise
    elif pcfg["source"] != "pose_json":
        raise ValueError(f"poses.source sconosciuto: {pcfg['source']!r}")
    return from_pose_json(Path(row.pose_json_path), row.video_path, pcfg)


# Diagnostics
def _kabsch(a: np.ndarray, b: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """Rotation R that minimises sum w ||a_i - R b_i||^2."""
    w = np.ones(len(a)) if w is None else w
    H = (b * w[:, None]).T @ a
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def estimate_frame(json_traj: Trajectory, csv_traj: Trajectory, lag_s: float = 0.5, max_shift_s: float = 3.0) -> dict:
    """Estimate ``json_cam_to_device`` (X) and ``json_world_rotation`` (W) from a take that has both sources.

    Model: R_wd(t) = W @ R_wc(t + s) @ X. We find the time shift s by correlating the angular velocities,
    X from the relative rotations over ``lag_s`` (a hand-eye problem: rotvec(A) = X rotvec(B), solved with Kabsch),
    and W as the mean of R_wd X^T R_wc^T, projected on SO(3). The residual in degrees tells whether the model holds.
    """
    fps = 1.0 / float(np.median(np.diff(json_traj.t)))
    Rc = Rotation.from_quat(json_traj.poses[:, 3:])
    Rd = Rotation.from_quat(csv_traj.poses[:, 3:])

    def ang_speed(R: Rotation) -> np.ndarray:
        return np.r_[0.0, (R[:-1].inv() * R[1:]).magnitude()]

    sc, sd = ang_speed(Rc), ang_speed(Rd)
    n = min(len(sc), len(sd))
    best, best_shift = -np.inf, 0
    for sh in range(-int(max_shift_s * fps), int(max_shift_s * fps) + 1):
        a = sc[max(0, sh) : n + min(0, sh)]
        b = sd[max(0, -sh) : n - max(0, sh)]
        m = min(len(a), len(b))
        if m < 50:
            continue
        c = np.corrcoef(a[:m], b[:m])[0, 1]
        if c > best:
            best, best_shift = c, sh
    ic = np.arange(n)
    idc, idd = ic + max(0, best_shift), ic + max(0, -best_shift)
    keep = (idc < len(Rc)) & (idd < len(Rd))
    idc, idd = idc[keep], idd[keep]
    ok = json_traj.valid[idc] & csv_traj.valid[idd]
    idc, idd = idc[ok], idd[ok]
    k = max(1, int(round(lag_s * fps)))
    A = (Rc[idc[:-k]].inv() * Rc[idc[k:]]).as_rotvec()
    B = (Rd[idd[:-k]].inv() * Rd[idd[k:]]).as_rotvec()
    w = np.linalg.norm(B, axis=1)
    sel = w > 0.05
    if sel.sum() < 20:
        raise ValueError("troppo poco movimento per la stima")
    X = _kabsch(A[sel], B[sel], w[sel])
    Ms = Rd[idd].as_matrix() @ X.T @ np.transpose(Rc[idc].as_matrix(), (0, 2, 1))
    U, _, Vt = np.linalg.svd(Ms.mean(0))
    W = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    pred = Rotation.from_matrix(W @ Rc[idc].as_matrix() @ X)
    resid = np.degrees((pred.inv() * Rd[idd]).magnitude())
    return {
        "shift_s": best_shift / fps,
        "shift_corr": float(best),
        "json_cam_to_device": np.round(X, 4).tolist(),
        "json_world_rotation": np.round(W, 4).tolist(),
        "residual_deg_median": float(np.median(resid)),
        "residual_deg_p90": float(np.percentile(resid, 90)),
        "n_samples": int(len(idc)),
    }


def average_rotations(mats) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.mean(np.asarray(mats, float), axis=0))
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


# Which way is up in the JSON poses
OPENCV_TO_ARIA = np.diag([-1.0, -1.0, 1.0])  # paper (supp. 2.3): OpenCV (x right, y down) -> Aria (x left, y up)


def roll_candidates() -> Dict[int, np.ndarray]:
    """Candidate X (camera -> device): the paper's conversion composed with a roll of 0/90/180/270°
    around the optical axis (the Aria RGB camera is mounted rotated with respect to the device)."""
    return {k: Rotation.from_euler("z", 90 * k, degrees=True).as_matrix() @ OPENCV_TO_ARIA for k in range(4)}


def up_statistics(traj: Trajectory) -> dict:
    """On a trajectory WITHOUT correction (camera frame of the JSON files): mean direction of world up (+z)
    in the camera frame, and mean tilt of the optical axis (degrees, < 0 = pointing down).

    People wearing the glasses mostly stand upright, so the mean tells which camera axis points up.
    """
    R = Rotation.from_quat(traj.poses[traj.valid, 3:]).as_matrix()  # camera -> world
    if len(R) < 10:
        raise ValueError("troppi pochi campioni validi")
    up_c = R[:, 2, :]  # z row of R_wc = world e_z in the camera frame (R_wc^T e_z)
    mean = up_c.mean(0)
    pitch = np.degrees(np.arcsin(np.clip(R[:, 2, 2], -1, 1)))  # z component (world) of the optical axis
    return {
        "up_x": float(mean[0]), "up_y": float(mean[1]), "up_z": float(mean[2]),
        "up_consistency": float(np.linalg.norm(mean)),  # 1 = always the same direction
        "pitch_deg_mean": float(pitch.mean()),
        "pitch_deg_median": float(np.median(pitch)),
    }


def suggest_cam_to_device(up_mean: np.ndarray) -> dict:
    """Pick the roll whose device y axis (up) is best aligned with the observed mean up direction.

    Returns the suggested X, the alignment (cosine) and the residual roll in degrees with respect to the chosen
    candidate (in the image plane; a large residual means the camera is not mounted at a multiple of 90°)."""
    u = np.asarray(up_mean, float)
    u_img = u.copy()
    u_img[2] = 0.0
    u_img /= max(np.linalg.norm(u_img), 1e-9)
    scores = {k: float(X[:, 1] @ u_img) for k, X in roll_candidates().items()}
    k = max(scores, key=scores.get)
    resid = float(np.degrees(np.arccos(np.clip(scores[k], -1, 1))))
    return {"roll_k": k, "roll_deg": 90 * k, "json_cam_to_device": roll_candidates()[k].round(6).tolist(),
            "alignment": scores[k], "residual_roll_deg": resid, "scores": scores}
