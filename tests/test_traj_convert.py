"""Stadio 01 (M08): conversione delle pose per frame in traiettorie a 20 Hz.

L'oracolo ``_v1_*`` è il codice di ``hpc/make_traj_data.py`` della v1 (Python puro, un campione alla volta):
la versione vettorializzata deve darne le stesse pose, tranne le tre differenze volute del registro M08.
"""

import bisect
import csv
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.cascade.traj_convert import (
    OK,
    STATUS_FIELDS,
    ConversionError,
    PoseSource,
    align_quat_signs,
    convert_extrinsics,
    convert_take,
    find_video,
    load_pose_index,
    pick_camera,
    quat_from_rotmat,
    summarize,
    write_npz,
)
from src.cascade.trajectory import cached_take_trajectory, load_take_trajectory, slice_window

VIDEO_PATTERN = "{takes_root}/{take_name}/frame_aligned_videos/downscaled/448/aria01_214-1.mp4"
VIDEO_GLOB = "{takes_root}/{take_name}/frame_aligned_videos/downscaled/448/*214-1*.mp4"
PARAMS = dict(video_pattern=VIDEO_PATTERN, video_glob=VIDEO_GLOB, src_fps=30.0, out_fps=20.0, max_gap_frames=15)


# --------------------------------------------------------------------------- oracolo: codice v1
def _v1_quat_from_R(R):
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    n = math.sqrt(x * x + y * y + z * z + w * w)
    return (x / n, y / n, z / n, w / n)


def _v1_pose_from_E(E):
    R = [E[0][:3], E[1][:3], E[2][:3]]
    t = [E[0][3], E[1][3], E[2][3]]
    Rwc = [[R[j][i] for j in range(3)] for i in range(3)]
    C = [-sum(R[j][i] * t[j] for j in range(3)) for i in range(3)]
    return C, _v1_quat_from_R(Rwc)


def _v1_slerp(q0, q1, a):
    d = sum(x * y for x, y in zip(q0, q1))
    if d < 0:
        q1 = tuple(-x for x in q1)
        d = -d
    if d > 0.9995:
        q = [x + a * (y - x) for x, y in zip(q0, q1)]
    else:
        th = math.acos(min(1.0, d))
        s = math.sin(th)
        w0, w1 = math.sin((1 - a) * th) / s, math.sin(a * th) / s
        q = [w0 * x + w1 * y for x, y in zip(q0, q1)]
    n = math.sqrt(sum(x * x for x in q))
    return tuple(x / n for x in q)


def _v1_convert(ext, src_fps=30.0, out_fps=20.0, max_gap=15):
    """Il ciclo di ``process`` della v1, senza I/O: (k, centro, quaternione) delle righe scritte e n. di buchi."""
    idxs = sorted(int(k) for k in ext)
    P = [_v1_pose_from_E(ext[str(i)]) for i in idxs]
    step = src_fps / out_fps
    n_out = int(idxs[-1] / step) + 1
    rows, gaps = [], 0
    for k in range(n_out):
        f = k * step
        j = bisect.bisect_right(idxs, f) - 1
        if j < 0:
            C, q = P[0]
        elif idxs[j] == f or j + 1 >= len(idxs):
            C, q = P[j]
        else:
            i0, i1 = idxs[j], idxs[j + 1]
            if i1 - i0 > max_gap:
                continue
            a = (f - i0) / (i1 - i0)
            (C0, q0), (C1, q1) = P[j], P[j + 1]
            C = [x + a * (y - x) for x, y in zip(C0, C1)]
            q = _v1_slerp(q0, q1, a)
            if i1 - i0 > 1:
                gaps += 1
        rows.append((k, C, q))
    return rows, gaps, n_out


# --------------------------------------------------------------------------- dati di prova
def _extrinsics_from_poses(frames, centres, rots):
    """{indice: matrice 3x4 mondo -> camera} da centro camera e rotazione camera -> mondo."""
    out = {}
    for i, c, r in zip(frames, centres, rots):
        R = r.as_matrix().T
        out[str(i)] = np.hstack([R, (-R @ c)[:, None]]).tolist()
    return out


def _walk_extrinsics(frames, seed):
    """Moto casuale con rotazioni grandi (copre tutti i rami del quaternione)."""
    rng = np.random.default_rng(seed)
    n = len(frames)
    centres = np.cumsum(rng.normal(scale=0.02, size=(n, 3)), axis=0)
    steps = Rotation.from_rotvec(rng.normal(scale=0.08, size=(n, 3)))
    rot = Rotation.from_quat(rng.normal(size=4))
    rots = []
    for i in range(n):
        rot = rot * steps[i]
        rots.append(rot)
    return _extrinsics_from_poses(frames, centres, rots)


def _uniform_extrinsics(frames, v=(0.3, 0.0, 0.1), omega=0.5):
    """Velocità costante ``v`` m/s e rotazione costante ``omega`` rad/s attorno a z, a 30 Hz."""
    t = np.asarray(frames, dtype=float) / 30.0
    centres = np.outer(t, v)
    rots = [Rotation.from_rotvec([0.0, 0.0, omega * x]) for x in t]
    return _extrinsics_from_poses(frames, centres, rots)


def _same_rotation(q, ref):
    return abs(float(np.dot(q, ref))) > 1 - 1e-9


# --------------------------------------------------------------------------- geometria
def test_quat_from_rotmat_all_branches():
    mats = [Rotation.from_quat(np.random.default_rng(1).normal(size=(200, 4))).as_matrix()]
    for axis in range(3):  # rotazioni di 180°: tr = -1, un ramo diverso per asse
        v = np.zeros(3)
        v[axis] = np.pi
        mats.append(Rotation.from_rotvec(v).as_matrix()[None])
    mats.append(np.eye(3)[None])
    R = np.concatenate(mats)
    q = quat_from_rotmat(R)
    ref = Rotation.from_matrix(R).as_quat()
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)
    assert all(_same_rotation(a, b) for a, b in zip(q, ref))


def test_matches_v1_reference():
    frames = [i for i in range(7, 900) if not (100 <= i < 106) and not (300 <= i < 333)]
    ext = _walk_extrinsics(frames, seed=3)
    conv = convert_extrinsics(ext)
    rows, gaps, n_out = _v1_convert(ext)
    lead = {k for k, _, _ in rows if k * 1.5 < 7}
    written = {k for k, _, _ in rows}
    dropped = set(range(n_out)) - written
    assert conv.stats["n_out"] == n_out and len(conv.t) == n_out
    assert dropped and lead  # il caso prova davvero buchi lunghi e righe prima della prima posa
    for k, C, q in rows:
        assert np.allclose(conv.poses[k, :3], C, atol=1e-9), k
        assert _same_rotation(conv.poses[k, 3:], q), k
        assert bool(conv.valid[k]) == (k not in lead), k
    assert not conv.valid[sorted(dropped)].any()
    assert conv.stats["dropped"] == len(dropped)
    assert conv.stats["lead_fill"] == len(lead)
    assert conv.stats["gaps"] == gaps


def test_known_motion():
    ext = _uniform_extrinsics(range(0, 901))
    conv = convert_extrinsics(ext)
    assert conv.valid.all() and conv.stats["lead_fill"] == 0 and conv.stats["dropped"] == 0
    assert np.allclose(conv.t, np.arange(len(conv.t)) / 20.0)
    assert np.allclose(conv.poses[:, :3], np.outer(conv.t, (0.3, 0.0, 0.1)), atol=1e-9)  # posizione lineare
    R = Rotation.from_quat(conv.poses[:, 3:])
    truth = Rotation.from_rotvec(np.outer(conv.t * 0.5, (0.0, 0.0, 1.0)))
    assert (R.inv() * truth).magnitude().max() < 1e-6  # camera -> mondo, non l'inversa
    assert np.allclose((R[:-1].inv() * R[1:]).magnitude() * 20, 0.5, atol=1e-4)  # velocità angolare
    q = conv.poses[:, 3:]  # 22 rad di rotazione: attraversa 2π, dove il quaternione grezzo cambia segno
    assert (np.sum(q[1:] * q[:-1], axis=1) > 0).all()
    speed = np.linalg.norm(np.diff(conv.poses[:, :3], axis=0), axis=1) * 20
    assert np.allclose(speed, math.hypot(0.3, 0.1), atol=1e-9)
    assert conv.stats["path_m"] == pytest.approx(math.hypot(0.3, 0.1) * conv.t[-1], abs=0.06)


def test_gaps_lead_and_fill():
    frames = [i for i in range(10, 400) if not (200 <= i < 206) and not (300 <= i < 340)]
    conv = convert_extrinsics(_uniform_extrinsics(frames))
    st = conv.stats
    assert (st["lead_fill"], st["dropped"], st["gaps"]) == (7, 27, 5)  # righe 0-6; 200-226; interpolate sul buco corto
    bad = list(range(0, 7)) + list(range(200, 227))
    assert set(np.flatnonzero(~conv.valid)) == set(bad)
    assert np.allclose(conv.poses[200, :3], conv.poses[199, :3])  # buco: posa del campione valido più vicino
    assert np.allclose(conv.poses[226, :3], conv.poses[227, :3])
    assert np.allclose(conv.poses[:7, :3], conv.poses[0, :3])  # prima della prima posa: copia della prima


def test_gap_limit_is_strict():
    """Un buco di 15 frame sorgente (0,5 s) si interpola, uno di 16 no: come ``i1 - i0 > MAX_GAP_FR`` della v1."""
    frames = [
        i for i in range(0, 120) if not (40 <= i < 55) and not (80 <= i < 94)
    ]  # buchi tra 39-55 (16) e 79-94 (15)
    conv = convert_extrinsics(_uniform_extrinsics(frames))
    assert (conv.stats["dropped"], conv.stats["gaps"]) == (10, 10)
    assert set(np.flatnonzero(~conv.valid)) == set(range(27, 37))  # f = 40,5 ... 54
    assert convert_extrinsics(_uniform_extrinsics(frames), max_gap_frames=16).stats["dropped"] == 0


def test_align_quat_signs():
    rng = np.random.default_rng(0)
    rots = Rotation.from_rotvec(np.cumsum(rng.normal(scale=0.05, size=(300, 3)), axis=0))
    q = rots.as_quat()
    flipped = q * np.where(rng.random(300) < 0.5, -1.0, 1.0)[:, None]
    aligned = align_quat_signs(flipped)
    assert (np.sum(aligned[1:] * aligned[:-1], axis=1) > 0).all()
    assert all(_same_rotation(a, b) for a, b in zip(aligned, q))


@pytest.mark.parametrize(
    "ext, status",
    [
        (None, "ext_not_per_frame"),
        ({"0": [[1, 0, 0, 0]] * 3}, "ext_not_per_frame"),  # una sola posa
        ({"a": [[1, 0, 0, 0]] * 3, "b": [[1, 0, 0, 0]] * 3}, "bad_keys"),
        ({"0": [[1, 0, 0, 0]] * 3, "1": [[1, 0, 0]] * 3}, "bad_matrix"),
        ({"0": [[1, 0, 0, 0]] * 3, "1": [[1, 0, 0, float("nan")]] * 3}, "bad_matrix"),
        ({"0": np.eye(4)[:3].tolist(), "1": np.eye(4)[:3].tolist()}, "too_short"),
        ({"5": np.eye(4)[:3].tolist(), "100": np.eye(4)[:3].tolist()}, "all_gaps"),  # nessuna riga cade su un frame
    ],
)
def test_error_statuses(ext, status):
    with pytest.raises(ConversionError) as e:
        convert_extrinsics(ext)
    assert e.value.status == status


def test_4x4_matrices_are_accepted():
    ext3 = _uniform_extrinsics(range(0, 91))
    ext4 = {k: v + [[0.0, 0.0, 0.0, 1.0]] for k, v in ext3.items()}
    assert np.allclose(convert_extrinsics(ext3).poses, convert_extrinsics(ext4).poses)


# --------------------------------------------------------------------------- file della v1
def _write_manifest(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "split_manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["take_name", "split", "camera_pose_file"])
        for take, rel in rows:
            w.writerow([take, "train", rel])


def _write_take(root, take, data, video="aria01_214-1.mp4"):
    rel = f"annotations/ego_pose/train/camera_pose/{take}.json"
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    if video:
        v = root / "takes" / take / "frame_aligned_videos" / "downscaled" / "448" / video
        v.parent.mkdir(parents=True, exist_ok=True)
        v.write_bytes(b"")
    return rel


def test_pick_camera():
    assert pick_camera({"aria01": {}, "aria02": {}}, "aria02") == ("aria02", OK)
    assert pick_camera({"aria01": {}, "gopro": {}}, "aria05") == ("aria01", OK)  # una sola camera aria
    cam, status = pick_camera({"aria01": {}, "aria02": {}}, "aria05")
    assert cam is None and status == "cam_ambiguous:aria01,aria02"
    assert pick_camera({"gopro": {}}, None)[0] is None


def test_pose_index_first_root_wins_and_errors(tmp_path):
    r1, r2 = tmp_path / "r1", tmp_path / "r2"
    _write_manifest(r1, [("t1", "a.json"), ("t2", "b.json")])
    _write_manifest(r2, [("t2", "c.json"), ("t3", "d.json")])
    idx = load_pose_index([r1, r2])
    assert idx["t2"] == PoseSource(r1, "b.json") and idx["t3"].root == r2 and len(idx) == 3
    with pytest.raises(FileNotFoundError):
        load_pose_index([tmp_path / "manca"])
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "split_manifest.csv").write_text("take,file\nx,y\n")
    with pytest.raises(ValueError, match="colonne mancanti"):
        load_pose_index([bad])


def test_find_video_exact_then_glob(tmp_path):
    _write_take(tmp_path, "exact", {}, video="aria01_214-1.mp4")
    _write_take(tmp_path, "glob", {}, video="aria02_214-1.mp4")
    _write_take(tmp_path, "none", {}, video=None)
    assert find_video(tmp_path, "exact", VIDEO_PATTERN, VIDEO_GLOB).name == "aria01_214-1.mp4"
    assert find_video(tmp_path, "glob", VIDEO_PATTERN, VIDEO_GLOB).name == "aria02_214-1.mp4"
    assert find_video(tmp_path, "none", VIDEO_PATTERN, VIDEO_GLOB) is None


def test_convert_take_statuses(tmp_path):
    ext = _uniform_extrinsics(range(0, 300))
    rows = [
        ("ok", _write_take(tmp_path, "ok", {"aria01": {"camera_extrinsics": ext}})),
        ("glob", _write_take(tmp_path, "glob", {"aria02": {"camera_extrinsics": ext}}, video="aria02_214-1.mp4")),
        ("novideo", _write_take(tmp_path, "novideo", {"aria01": {"camera_extrinsics": ext}}, video=None)),
        ("ambig", _write_take(tmp_path, "ambig", {"aria01": {}, "aria02": {}}, video="aria03_214-1.mp4")),
        ("noext", _write_take(tmp_path, "noext", {"aria01": {"camera_extrinsics": [1, 2]}})),
        ("nofile", _write_take(tmp_path, "nofile", {})),
    ]
    (tmp_path / rows[-1][1]).unlink()  # riga nel manifest e video presente, ma il file di pose manca
    rows.append(("badjson", _write_take(tmp_path, "badjson", {})))
    (tmp_path / rows[-1][1]).write_text("{non json")
    _write_manifest(tmp_path, rows)
    index = load_pose_index([tmp_path])
    status = {}
    for take in [r[0] for r in rows] + ["absent"]:
        conv, row = convert_take(take, index, **PARAMS)
        status[take] = row["status"]
        assert (conv is not None) == (row["status"] == OK)
        assert set(row) == set(STATUS_FIELDS)
    assert status == {
        "ok": OK,
        "glob": OK,
        "novideo": "no_video",
        "ambig": "cam_ambiguous:aria01,aria02",
        "noext": "ext_not_per_frame",
        "nofile": "no_pose_file",
        "badjson": "bad_json",
        "absent": "no_manifest_row",
    }
    conv, row = convert_take("glob", index, **PARAMS)
    assert row["cam"] == "aria02" and row["n_out"] == len(conv.t)
    summary = summarize([convert_take(t, index, **PARAMS)[1] for t in status], n_takes=len(status))
    assert summary["n_ok"] == 2 and summary["missing_frac"] == pytest.approx(6 / 8)
    assert summary["status_counts"]["no_video"] == 1


# --------------------------------------------------------------------------- sorgente converted_npz
def _cfg(mode="same_origin", fps=20.0):
    align = SimpleNamespace(
        mode=mode, offset_s=0.0, scale=1.0, video_start_key="video_start_us", traj_start_key="traj_start_us"
    )
    return SimpleNamespace(data=SimpleNamespace(traj_source="converted_npz", traj_fps=fps), alignment=align)


def test_converted_npz_roundtrip(tmp_path):
    frames = [i for i in range(0, 900) if not (300 <= i < 340)]
    conv = convert_extrinsics(_uniform_extrinsics(frames))
    path = tmp_path / "take.npz"
    write_npz(path, conv, out_fps=20.0, src_fps=30.0)
    tr = load_take_trajectory(path, _cfg())
    assert np.array_equal(tr.t_video, conv.t) and np.array_equal(tr.poses, conv.poses)
    assert np.array_equal(tr.valid, conv.valid) and tr.fps == pytest.approx(20.0)
    full, cov_full = slice_window(tr, 14.0, 22.0)  # dopo il buco (righe 200-226, cioè 10,0-11,3 s)
    assert len(full) == 160 and cov_full == pytest.approx(1.0)
    _, cov_gap = slice_window(tr, 8.0, 16.0)  # attraversa il buco: la copertura lo vede (133/160)
    assert cov_gap == pytest.approx(133 / 160)
    cache = tmp_path / "cache"
    cached_take_trajectory(path, _cfg(), {}, cache)
    assert not cache.exists()  # già su disco: nessuna copia nella cache delle traiettorie


def test_converted_npz_refuses_inconsistent_config(tmp_path):
    path = tmp_path / "take.npz"
    write_npz(path, convert_extrinsics(_uniform_extrinsics(range(0, 90))), out_fps=20.0, src_fps=30.0)
    with pytest.raises(ValueError, match="traj_fps"):
        load_take_trajectory(path, _cfg(fps=10.0))
    with pytest.raises(ValueError, match="same_origin"):
        load_take_trajectory(path, _cfg(mode="device_us"))


def test_config_accepts_converted_npz():
    config = pytest.importorskip("src.cascade.config")
    cfg = config.load_config(overrides=["data.traj_source=converted_npz", "paths.pose_roots=[/a, /b]"])
    assert cfg.data.traj_source == "converted_npz" and cfg.paths.pose_roots == ["/a", "/b"]
    assert cfg.data.pose_src_fps == 30.0 and cfg.data.pose_max_gap_frames == 15
