import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.cascade.data.motion import MOTION_FEATURES, dominant_frequency, motion_features


def _poses(n=200, fps=20.0, f=1.5, amp=0.05, vx=0.0, yaw_rate=0.0):
    t = np.arange(n) / fps
    p = np.zeros((n, 7))
    p[:, 0] = vx * t
    p[:, 2] = 1.6 + amp * np.sin(2 * np.pi * f * t) + 0.01 * t  # con trend lineare
    p[:, 3:] = Rotation.from_euler("z", (yaw_rate * t)[:, None]).as_quat()
    return p


@pytest.mark.parametrize("f", [0.6, 1.5, 2.7, 4.0])
def test_sinusoid_dominant_frequency(f):
    m = motion_features(_poses(f=f), 20.0)
    assert m["mot_zfreq_dom"] == pytest.approx(f, abs=0.05)
    assert m["mot_zfreq_peak"] > 0.5


def test_speed_and_angular_velocity():
    m = motion_features(_poses(amp=0.0, vx=1.2, yaw_rate=0.5), 20.0)
    assert m["mot_speed_mean"] == pytest.approx(np.hypot(1.2, 0.01), rel=1e-3)
    assert m["mot_angvel_mean"] == pytest.approx(0.5, rel=1e-3)
    assert m["mot_speed_std"] < 1e-6 and m["mot_jerk_mean"] < 1e-6
    assert m["mot_energy"] == pytest.approx(m["mot_speed_mean"] ** 2 + 0.25, rel=1e-3)


def test_vertical_range():
    m = motion_features(_poses(amp=0.1, f=1.0, n=400), 20.0)
    assert 0.17 < m["mot_vert_range"] < 0.3


def test_short_window_is_nan():
    m = motion_features(_poses(n=5), 20.0)
    assert set(m) == set(MOTION_FEATURES) and all(np.isnan(v) for v in m.values())


def test_flat_signal():
    assert dominant_frequency(np.zeros(50), 20.0) == (0.0, 0.0)
