"""Stadio 41 sui dati sintetici: tabella dei tempi per clip e riepilogo con mediana e p90."""

import importlib.util
import json

import pandas as pd
import pytest

from src import REPO_ROOT
from src.cascade.cost_e2e import TIME_COLUMNS

pytest.importorskip("torch")
pytest.importorskip("einops")
pytest.importorskip("av")


def _run(script, argv):
    spec = importlib.util.spec_from_file_location(script.replace(".py", ""), REPO_ROOT / "scripts" / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(argv)


def test_cost_e2e_on_synthetic(syn_root, monkeypatch):
    from tests.conftest import SMALL

    monkeypatch.setenv("DL4CV_ROOT", str(syn_root))
    base = ["--synthetic", "--run-name", "t41"]
    for o in SMALL + ["data.max_missing_frac=0.1", "cost_e2e.n_clips=6", "cost_e2e.n_warmup=1"]:
        base += ["--set", o]
    _run("01_convert_trajectories.py", base)
    _run("00_inventory.py", base)
    _run("41_cost_e2e.py", base)
    out = syn_root / "runs" / "t41" / "41_cost_e2e"
    times = pd.read_csv(out / "clip_times.csv")
    assert set(TIME_COLUMNS) <= set(times.columns)
    assert set(times["pass"]) == {1, 2}  # due passate: la seconda è quella riportata
    assert (times[["l0_total_s", "l1_total_s"]] > 0).all().all()
    s = json.loads((out / "summary.json").read_text())
    assert s["reported_pass"] == 2 and s["n_clips"] > 0
    assert set(s["phases_reported"]) == set(TIME_COLUMNS[1:])
    assert len(s["time_curve"]) == 11 and len(s["breakeven_pose_s"]) == 11
    assert s["time_curve"][0]["speedup"] > s["time_curve"][-1]["speedup"]
