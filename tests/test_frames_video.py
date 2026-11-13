import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("av")

from src.cascade.frames import frame_times, preprocess_frames  # noqa: E402
from src.cascade.synthetic import write_video  # noqa: E402
from src.cascade.video import decode_frames, frame_indices_for_times, probe_video  # noqa: E402


def test_frame_times_are_segment_midpoints():
    t = frame_times(2.0, 10.0, 16)
    assert len(t) == 16 and t[0] == pytest.approx(2.25) and t[-1] == pytest.approx(9.75)
    np.testing.assert_allclose(np.diff(t), 0.5)
    assert np.mean(t) == pytest.approx(6.0)
    with pytest.raises(ValueError):
        frame_times(3.0, 3.0, 4)


def test_frame_indices_for_times():
    idx = frame_indices_for_times(np.array([0.0, 0.049, 0.05, 0.99, 5.0]), fps=20, n_frames=30)
    np.testing.assert_array_equal(idx, [0, 0, 1, 19, 29])


def test_preprocess_center_crop_and_size():
    fr = np.zeros((2, 100, 200, 3), np.uint8)
    fr[:, :, 50:150] = 200  # colonna centrale chiara, bordi neri
    out = preprocess_frames(fr, pre_resize=64, crop=64, size=32)
    assert out.shape == (2, 32, 32, 3) and out.dtype == np.uint8
    assert out[:, :, 4:28].mean() > 190  # il crop centrale è nella parte chiara


def test_preprocess_rotation():
    fr = np.zeros((1, 40, 40, 3), np.uint8)
    fr[:, :10] = 255  # banda in alto
    out = preprocess_frames(fr, 40, 40, 40, rotate_k90=1)
    assert out[0, :, :8].mean() > 200  # dopo 90° antiorari la banda è a sinistra


def test_decode_exact_indices(tmp_path):
    n = 60
    frames = np.zeros((n, 32, 32, 3), np.uint8)
    for i in range(n):
        frames[i] = (i * 4) % 250  # luminosità = indice del frame
    p = tmp_path / "v.mp4"
    write_video(p, frames, fps=20)
    info = probe_video(p)
    assert info.fps == pytest.approx(20) and abs(info.n_frames - n) <= 1
    want = [3, 17, 17, 40, 59, 5]
    got = decode_frames(p, want, "pyav")
    assert got.shape == (6, 32, 32, 3)
    np.testing.assert_allclose(got.mean(axis=(1, 2, 3)), [(i * 4) % 250 for i in want], atol=6)
