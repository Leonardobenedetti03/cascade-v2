import pandas as pd
import pytest

from src.cascade.clips import build_inventory, compute_windows, make_clip_id


def test_single_window_centered():
    (w,) = compute_windows(20.0, 8.0)
    assert (w.t_start, w.t_center, w.t_end) == (6.0, 10.0, 14.0)


def test_short_take_uses_whole_take():
    (w,) = compute_windows(5.0, 8.0)
    assert (w.t_start, w.t_end) == (0.0, 5.0)


def test_multi_window_shifted_inside():
    ws = compute_windows(20.0, 8.0, n_windows=3, stride_s=8.0)
    assert [w.t_center for w in ws] == [4.0, 10.0, 16.0]
    ws = compute_windows(20.0, 8.0, n_windows=3, stride_s=10.0)
    assert ws[0].t_start == 0.0 and ws[-1].t_end == 20.0
    assert all(abs((w.t_end - w.t_start) - 8.0) < 1e-12 for w in ws)


def test_invalid_windows():
    with pytest.raises(ValueError):
        compute_windows(0.0, 8.0)
    with pytest.raises(ValueError):
        compute_windows(10.0, 8.0, n_windows=0)


def test_clip_id():
    assert make_clip_id("take_a", 3) == "take_a__w03"


def test_inventory_on_synthetic(syn_ctx):
    clips, missing, summary = build_inventory(syn_ctx)
    assert clips.clip_id.is_unique
    assert {"clip_id", "take_uid", "split", "label_idx", "t_start", "t_end", "has_video", "has_traj"} <= set(clips)
    assert set(missing.problems) == {"no_video", "no_traj"}
    assert summary["clips_usable"] == len(clips) - 2
    lab = syn_ctx.labels
    for r in clips.itertuples():
        assert lab.names[r.label_idx] == r.label
        assert abs((r.t_end - r.t_start) - syn_ctx.cfg.data.window_s) < 1e-9
        assert abs(r.t_center - r.take_duration / 2) < 1e-9


def test_inventory_multi_window(syn_root):
    from tests.conftest import make_ctx

    ctx = make_ctx(syn_root, extra=["data.n_windows=2", "data.window_s=2.0"])
    clips, _, _ = build_inventory(ctx)
    per_take = clips.groupby("take_uid").clip_id.nunique()
    assert (per_take == 2).all()
    assert isinstance(clips, pd.DataFrame)
