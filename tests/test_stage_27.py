"""Stadio 27 (diagnostiche di sito) sul dataset sintetico: i tre file, il test escluso, la ripresa.

I take_uid sintetici hanno tutti il prefisso ``syn``: con un solo sito la sonda si salta e lo stadio
scrive comunque i tre file. La logica vera (più siti) è provata in ``test_site_diag.py``."""

import importlib.util
import json

import pandas as pd
import pytest

from src import REPO_ROOT
from src.cascade.eval.site_diag import LOSO_COLUMNS
from src.cascade.manifest import SPLIT_DEPENDENT_STAGES, STAGE_KEYS

pytest.importorskip("torch")
pytest.importorskip("einops")
pytest.importorskip("av")
pytest.importorskip("pyarrow")
pytest.importorskip("sklearn")


def _run(script, argv):
    spec = importlib.util.spec_from_file_location(script.replace(".py", ""), REPO_ROOT / "scripts" / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(argv)


def test_stage_27_keys_track_split_and_diag_config():
    assert "27_site" in SPLIT_DEPENDENT_STAGES
    assert "site_diag" in STAGE_KEYS["27_site"] and "gate.bacc_min_takes" in STAGE_KEYS["27_site"]


def test_stage_27_on_synthetic(syn_root, monkeypatch):
    from tests.conftest import SMALL

    monkeypatch.setenv("DL4CV_ROOT", str(syn_root))
    monkeypatch.delenv("CASCADE_DATA_ROOT", raising=False)
    monkeypatch.delenv("CASCADE_RUN_ROOT", raising=False)
    from src.cascade.paths import run_dir

    base = ["--synthetic", "--run-name", "t27"]
    for o in SMALL + ["data.max_missing_frac=0.1"]:
        base += ["--set", o]
    _run("01_convert_trajectories.py", base)
    _run("00_inventory.py", base)
    _run("05_folds.py", base)
    _run("10_extract_frames.py", base + ["--num-shards", "2", "--shard-id", "0"])
    _run("10_extract_frames.py", base + ["--num-shards", "2", "--shard-id", "1"])
    _run("25_egovlp_features.py", base)
    _run("27_site_diagnostics.py", base)

    out = run_dir("t27") / "27_site"
    loso = pd.read_csv(out / "loso.csv")
    assert list(loso.columns) == LOSO_COLUMNS and loso.empty  # un solo sito: niente leave-one-site-out
    probe = json.loads((out / "site_probe.json").read_text())
    assert probe["n_sites"] == 1 and "skipped" in probe and probe["scope"].startswith("train+val")
    maj = json.loads((out / "site_majority.json").read_text())
    assert 0 < maj["accuracy_train_insample"] <= 1
    assert {r["site"] for r in maj["per_site"]} == {"syn"}
    assert all("test" not in r["role"] for r in maj["per_site"])  # il test non entra
    assert (run_dir("t27") / "_manifest" / "27_site.json").exists()
    _run("27_site_diagnostics.py", base)  # secondo giro: salta senza errori
