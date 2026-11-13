"""M08: lo stadio 01 e l'inventario (00) con la sorgente ``converted_npz``, sui dati sintetici.

Lancia gli script veri come sottoprocessi (come ``run_synthetic_pipeline.sh``): verifica che 00 pretenda lo stadio 01,
che il percorso ``{run}/01_traj/{take_name}.npz`` si risolva e che le take utilizzabili siano quelle attese.
"""

import os
import subprocess
import sys

import pandas as pd

from src import REPO_ROOT

# 8 classi x 3 split x 2 take = 48 take; il generatore ne lascia 1 senza video (test) e 1 senza traiettoria (val)
SETS = ["synthetic.takes_per_class={train: 2, val: 2, test: 2}", "data.max_missing_frac=0.1"]
CONVERTED = [
    "data.traj_source=converted_npz",
    "alignment.mode=same_origin",
    'paths.traj_pattern="{run}/01_traj/{take_name}.npz"',
]


def _run(script, root, run_name):
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / script), "--synthetic", "--run-name", run_name]
    for s in SETS + CONVERTED:
        cmd += ["--set", s]
    env = {**os.environ, "DL4CV_ROOT": str(root)}
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)


def test_inventory_requires_stage_01(tmp_path):
    r = _run("00_inventory.py", tmp_path, "early")
    assert r.returncode != 0, "00 deve fermarsi se manca lo stadio 01"
    assert not (tmp_path / "runs" / "early" / "data" / "clips.csv").exists()


def test_converted_inventory_matches_expected_takes(tmp_path):
    first = _run("01_convert_trajectories.py", tmp_path, "conv")
    assert first.returncode == 0, first.stderr[-2000:]
    second = _run("00_inventory.py", tmp_path, "conv")
    assert second.returncode == 0, second.stderr[-2000:]
    run = tmp_path / "runs" / "conv"
    status = pd.read_csv(run / "01_traj" / "status.csv")
    assert dict(status["status"].value_counts()) == {"ok": 46, "no_manifest_row": 1, "no_video": 1}
    clips = pd.read_csv(run / "data" / "clips.csv")
    usable = clips[clips.has_video & clips.has_traj]
    assert len(clips) == 48 and len(usable) == 46  # come con aria_csv: 2 take escluse dal generatore
    assert all((run / "01_traj" / f"{t}.npz").exists() for t in usable.take_uid)
