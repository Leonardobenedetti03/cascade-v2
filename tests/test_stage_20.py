import importlib.util
import json

import pytest

from src import REPO_ROOT
from src.data.cache import read_table

pytest.importorskip("torch")


def _run(script, argv):
    spec = importlib.util.spec_from_file_location(script.replace(".py", ""), REPO_ROOT / "scripts" / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(argv)


def test_finetune_fold_on_synthetic(syn_root, monkeypatch):
    from tests.conftest import SMALL

    monkeypatch.setenv("DL4CV_ROOT", str(syn_root))
    base = ["--synthetic", "--run-name", "t"]
    for o in SMALL + ["camformer.train.epochs=2", "data.max_missing_frac=0.1"]:
        base += ["--set", o]
    _run("01_convert_trajectories.py", base)
    _run("00_inventory.py", base)
    _run("05_folds.py", base)
    _run("20_camformer_finetune.py", base + ["--fold", "1"])
    out = syn_root / "runs" / "t" / "20_camformer" / "seed0_hz20" / "fold1"
    s = json.loads((out / "summary.json").read_text())
    assert s["T"] > 0 and s["preds"]["oof"]["n"] > 0
    assert not (out / "ckpt_last.pt").exists()  # la ripresa sparisce a stadio completo
    oof = read_table(out / "preds_oof.parquet")
    import pandas as pd

    folds = pd.read_csv(syn_root / "runs" / "t" / "splits" / "folds.csv")
    fold_of = dict(zip(folds.take_uid, folds.fold))
    assert {fold_of[t] for t in oof.df.take_uid} == {1}  # solo clip della fold 1
    assert oof.arrays["cam_probs_cal"].shape[1] == 6  # N28
    assert oof.meta["class_names"][0] == "Basketball"
    val = read_table(out / "preds_val.parquet")
    assert set(val.df.split) == {"val"}
    # idempotente: il secondo lancio salta
    mtime = (out / "summary.json").stat().st_mtime
    _run("20_camformer_finetune.py", base + ["--fold", "1"])
    assert (out / "summary.json").stat().st_mtime == mtime
