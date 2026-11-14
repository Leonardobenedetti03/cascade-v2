"""Costo end-to-end (src/cascade/cost_e2e.py): formula di pareggio e schema della tabella dei tempi."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.cascade.cost_e2e import (
    L0_PHASES,
    L1_PHASES,
    TIME_COLUMNS,
    cascade_time,
    clip_times,
    pose_breakeven,
    stratified_sample,
    summarize,
    time_curve,
    time_speedup,
    timed,
)


def test_pose_breakeven_matches_the_blueprint():
    c_l1, c_l0 = 92.31, 0.1216
    assert round(pose_breakeven(0.5, c_l1, c_l0), 1) == 46.0
    assert round(pose_breakeven(0.7, c_l1, c_l0), 1) == 27.6
    assert pose_breakeven(0.5, c_l1, c_l0, c_gate=1.0) == pytest.approx(pose_breakeven(0.5, c_l1, c_l0) - 1.0)
    assert pose_breakeven(1.0, c_l1, c_l0) < 0  # se si scala sempre, la traiettoria non conviene mai
    with pytest.raises(ValueError):
        pose_breakeven(1.5, c_l1, c_l0)


def test_timed_syncs_before_and_after():
    calls = []
    out, dt = timed(lambda a, b: calls.append("fn") or a + b, 2, 3, sync=lambda: calls.append("sync"))
    assert out == 5 and dt >= 0.0
    assert calls == ["sync", "fn", "sync"]
    assert timed(lambda: 1)[0] == 1  # senza sync


def _dummy(offset):
    return SimpleNamespace(
        read_window=lambda c: ("w", c.clip_id),
        decode=lambda c: ("raw", c.clip_id),
        preprocess=lambda x: x,
        forward=lambda x: offset,
    )


def test_clip_times_schema_and_totals():
    row = clip_times(SimpleNamespace(clip_id="c1"), _dummy(0), _dummy(1), sync=None)
    assert list(row) == [c for c in TIME_COLUMNS if c != "clip_id"] or set(row) == set(TIME_COLUMNS)
    assert row["clip_id"] == "c1"
    assert row["l0_total_s"] == pytest.approx(sum(row[k] for k in L0_PHASES))
    assert row["l1_total_s"] == pytest.approx(sum(row[k] for k in L1_PHASES))
    assert all(row[k] >= 0 for k in TIME_COLUMNS[1:])


def test_summarize_median_and_p90():
    df = pd.DataFrame({c: np.arange(1.0, 11.0) for c in TIME_COLUMNS[1:]})
    s = summarize(df)
    assert set(s) == set(TIME_COLUMNS[1:])
    assert s["l0_total_s"]["median_s"] == pytest.approx(5.5)
    assert s["l0_total_s"]["p90_s"] == pytest.approx(9.1)
    assert s["l0_total_s"]["n"] == 10
    assert summarize(df.iloc[:0]) == {}


def test_time_curve_and_speedup():
    t0, t1 = 0.01, 0.5
    assert cascade_time(t0, t1, 0.0) == pytest.approx(t0)
    assert time_speedup(t0, t1, 0.0) == pytest.approx(t1 / t0)  # nessuna scalata: solo L0
    assert time_speedup(t0, t1, 1.0) < 1.0  # scalano tutte: si paga anche L0
    curve = time_curve(t0, t1, [0.0, 0.5, 1.0])
    assert [c["r"] for c in curve] == [0.0, 0.5, 1.0]
    assert curve[0]["speedup"] > curve[1]["speedup"] > curve[2]["speedup"]
    with pytest.raises(ValueError):
        cascade_time(t0, t1, -0.1)


def test_stratified_sample_is_balanced_and_deterministic():
    df = pd.DataFrame({"clip_id": [f"c{i}" for i in range(60)], "label_idx": [i % 3 for i in range(60)]})
    a, b = stratified_sample(df, 9, seed=1), stratified_sample(df, 9, seed=1)
    assert list(a.clip_id) == list(b.clip_id)
    assert a.clip_id.is_unique and len(a) == 9
    assert a.label_idx.value_counts().tolist() == [3, 3, 3]
    assert list(stratified_sample(df, 9, seed=2).clip_id) != list(a.clip_id)
    assert len(stratified_sample(df, 500, seed=0)) == 60  # più del disponibile: tutte
    skew = pd.DataFrame({"clip_id": list("abcdef"), "label_idx": [0, 0, 0, 0, 1, 2]})
    assert sorted(stratified_sample(skew, 5, seed=0).label_idx.tolist()) == [0, 0, 0, 1, 2]
