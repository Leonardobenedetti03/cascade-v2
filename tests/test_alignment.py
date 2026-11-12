import numpy as np
import pandas as pd
import pytest

from src.cascade.alignment import traj_to_video_time, video_to_traj_time
from src.cascade.config import load_config
from src.cascade.trajectory import (
    POSE_COLUMNS,
    TIME_COLUMN,
    load_camformer_npz,
    load_take_trajectory,
    resample_nearest,
    slice_window,
)


def _align(**kw):
    return load_config(overrides=[f"alignment.{k}={v}" for k, v in kw.items()]).alignment


def test_same_origin_and_offset():
    t = np.array([0.0, 1.0, 2.5])
    np.testing.assert_allclose(traj_to_video_time(t, _align()), t)
    np.testing.assert_allclose(traj_to_video_time(t, _align(offset_s=0.3)), t + 0.3)
    np.testing.assert_allclose(traj_to_video_time(t, _align(scale=1.001)), t * 1.001)


def test_device_us_schema():
    a = _align(mode="device_us")
    # la traiettoria parte 0.4 s prima del video: t_traj = 0.4 corrisponde a t_video = 0
    out = traj_to_video_time(np.array([0.0, 0.4, 1.4]), a, traj_start_us=999_600_000, video_start_us=1_000_000_000)
    np.testing.assert_allclose(out, [-0.4, 0.0, 1.0], atol=1e-12)
    with pytest.raises(ValueError):
        traj_to_video_time(np.array([0.0]), a)


@pytest.mark.parametrize("kw", [{}, {"offset_s": -0.2, "scale": 0.999}, {"mode": "device_us", "offset_s": 0.05}])
def test_inverse_roundtrip(kw):
    a = _align(**kw)
    t = np.linspace(-1, 30, 50)
    extra = dict(traj_start_us=5e6, video_start_us=4.2e6) if a.mode == "device_us" else {}
    np.testing.assert_allclose(video_to_traj_time(traj_to_video_time(t, a, **extra), a, **extra), t, atol=1e-9)


def _write_aria_csv(path, start_us, hz, dur, offset_rows=0):
    n = int(dur * hz)
    t_us = start_us + (np.arange(n) * 1e6 / hz).astype(np.int64)
    df = pd.DataFrame(np.zeros((n, 7)), columns=POSE_COLUMNS)
    df["tz_world_device"] = np.arange(n) / hz  # quota = tempo traiettoria
    df["qw_world_device"] = 1.0
    df.insert(0, TIME_COLUMN, t_us)
    df.sample(frac=1, random_state=0).to_csv(path, index=False)  # righe non ordinate
    return n


def test_aria_csv_resampled_and_aligned(tmp_path):
    cfg = load_config(overrides=["alignment.mode=device_us", "data.traj_fps=20"])
    p = tmp_path / "closed_loop_trajectory.csv"
    _write_aria_csv(p, start_us=10_000_000, hz=1000, dur=6.0)
    tr = load_take_trajectory(p, cfg, {"video_start_us": 10_500_000})
    assert tr.fps == pytest.approx(20.0)
    # t_video = t_traj - 0.5; la quota codifica t_traj
    np.testing.assert_allclose(tr.poses[:, 2], tr.t_video + 0.5, atol=1e-3)
    poses, cov = slice_window(tr, 1.0, 3.0)
    assert len(poses) == 40 and cov == pytest.approx(1.0)
    np.testing.assert_allclose(poses[0, 2], 1.5, atol=1e-3)
    _, cov_out = slice_window(tr, 4.0, 8.0)
    assert cov_out < 0.5


def test_resample_flags_gaps():
    t = np.concatenate([np.arange(0, 1, 0.05), np.arange(3, 4, 0.05)])
    p = np.tile([0, 0, 0, 0, 0, 0, 1.0], (len(t), 1))
    grid, _, valid = resample_nearest(t, p, 20.0)
    assert not valid[(grid > 1.2) & (grid < 2.8)].any() and valid[grid < 0.9].all()


def test_camformer_npz_stitching(tmp_path):
    seg = np.tile([0, 0, 0, 0, 0, 0, 1.0], (40, 1))
    np.savez(tmp_path / "take.npz", **{"take_0.00_2.00": seg, "take_5.00_7.00": seg})
    t, p = load_camformer_npz(tmp_path / "take.npz")
    assert t[0] == 0.0 and t[-1] == pytest.approx(5.0 + 39 * 0.05) and len(p) == 80
