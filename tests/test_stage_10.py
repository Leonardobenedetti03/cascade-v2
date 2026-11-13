"""Stadio 10 sui dati sintetici: i frame vanno nella cartella persistente {data}/frames/<run_name>."""

import importlib.util

import numpy as np
import pytest

from src import REPO_ROOT

pytest.importorskip("av")


def _run(script, argv):
    spec = importlib.util.spec_from_file_location(script.replace(".py", ""), REPO_ROOT / "scripts" / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(argv)


def test_frames_go_to_the_persistent_dir(syn_root, monkeypatch):
    from tests.conftest import SMALL

    monkeypatch.setenv("DL4CV_ROOT", str(syn_root))
    base = ["--synthetic", "--run-name", "t10"]
    for o in SMALL + ["data.max_missing_frac=0.1"]:  # 2 take perse su 72 = 2,8% > 2%
        base += ["--set", o]
    _run("01_convert_trajectories.py", base)
    _run("00_inventory.py", base)
    _run("10_extract_frames.py", base)
    frames = syn_root / "runs" / "t10" / "synthetic_data" / "frames" / "t10"
    files = sorted(frames.glob("*.npy"))
    assert files, f"nessun frame in {frames}"
    arr = np.load(files[0])
    assert arr.ndim == 4 and arr.shape[-1] == 3 and arr.dtype == np.uint8
    assert not (syn_root / "runs" / "t10" / "10_frames").exists()  # non più dentro la run
