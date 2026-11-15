"""La config vera della run (``configs/hpc/site_v2.yaml``) sui dati sintetici, 01 → 00 → 05 → 20.

La pipeline sintetica usa ``aria_csv`` e non passa mai per ``converted_npz``: senza questo test
un errore nel collegamento tra lo stadio 01 e l'inventario si scopre solo sul cluster (B1).
Restano fuori le fold per sito: i sintetici hanno un solo sito, quindi qui ``group_by`` è ``take``
(le fold per sito sono coperte da ``test_site_folds`` sulle fold versionate).
"""

import importlib.util
import json

import pandas as pd
import pytest

from src import REPO_ROOT

pytest.importorskip("torch")

SITE_V2 = str(REPO_ROOT / "configs" / "hpc" / "site_v2.yaml")
# Solo ciò che i sintetici non possono avere uguale alla run vera.
SYNTHETIC_ONLY = [
    "alignment.mode=same_origin",
    "data.max_missing_frac=0.1",
    "folds.group_by=take",
    'paths.pose_roots=["{data}"]',  # site_v2.yaml ha le radici del cluster (N23)
]


def _run(script, argv):
    spec = importlib.util.spec_from_file_location(script.replace(".py", ""), REPO_ROOT / "scripts" / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(argv)


def test_site_v2_config_runs_from_conversion_to_camformer(syn_root, monkeypatch):
    from tests.conftest import SMALL

    monkeypatch.setenv("DL4CV_ROOT", str(syn_root))
    base = ["--synthetic", "--run-name", "site_chain", "--config", SITE_V2]
    for o in SMALL + SYNTHETIC_ONLY + ["camformer.train.epochs=2"]:
        base += ["--set", o]
    run = syn_root / "runs" / "site_chain"

    _run("01_convert_trajectories.py", base)
    _run("00_inventory.py", base)
    clips = pd.read_csv(run / "data" / "clips.csv")
    usable = clips[clips.has_video & clips.has_traj]
    assert len(usable) >= 0.8 * len(clips), "l'inventario non trova le traiettorie dello stadio 01"
    assert all(str(p).endswith(".npz") and "01_traj" in str(p) for p in usable.traj_path)

    _run("05_folds.py", base)
    _run("20_camformer_finetune.py", base + ["--fold", "0"])
    s = json.loads((run / "20_camformer" / "seed0_hz20" / "fold0" / "summary.json").read_text())
    assert s["preds"]["oof"]["n"] > 0 and s["T"] > 0
