"""Controlli dei percorsi di dati e pesi (``src/cascade/hpc_check.py``) e config della run ``site_v2``."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import pytest

from src import REPO_ROOT
from src.cascade.config import load_config
from src.cascade.hpc_check import FAIL, OK, WARN, main, path_checks

SITE_V2 = REPO_ROOT / "configs" / "hpc" / "site_v2.yaml"
# paths.pose_roots è ancora TODO(HPC) in base.yaml: nei test punta a una radice finta
POSES = ['paths.pose_roots=["{data}/poses"]']


def _cfg(*extra: str):
    return load_config([SITE_V2], POSES + list(extra))


def _take_names() -> list:
    with open(REPO_ROOT / "splits" / "site_v1" / "scenario_site.csv", newline="") as f:
        return [row["take_name"] for row in csv.DictReader(f)]


@pytest.fixture
def data_tree(tmp_path: Path, monkeypatch) -> Path:
    data = tmp_path / "data"
    monkeypatch.setenv("CASCADE_DATA_ROOT", str(data))
    monkeypatch.setenv("CASCADE_RUN_ROOT", str(tmp_path / "runs"))
    (data / "egoexo4d").mkdir(parents=True)
    (data / "egoexo4d" / "takes.json").write_text("[]")
    for name in _take_names():
        (data / "egoexo4d" / "takes" / name).mkdir(parents=True)
    for rel in (
        "checkpoints/egovlp/egovlp.pth",
        "checkpoints/torchvision/mobilenet_v3_small-047dcff4.pth",
        "poses/split_manifest.csv",
    ):
        f = data / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
    return data


def _status(checks) -> Dict[str, str]:
    return {c.name: c.status for c in checks}


def test_site_v2_config_loads_and_pins_the_run():
    cfg = load_config([SITE_V2])
    assert cfg.run_name == "site_v2"
    assert cfg.data.traj_source == "converted_npz"
    assert cfg.frame_baseline.enabled and not cfg.egovlp.zero_shot.enabled
    assert cfg.folds.group_by == "site" and not cfg.camformer.pretrained and cfg.fusion.mode == "replace"
    assert dict(cfg.data.expected_split_counts) == {"train": 1138, "val": 68, "test": 67}
    sums = (REPO_ROOT / "splits" / "site_v1" / "SHA256SUMS").read_text().splitlines()
    sha = next(line.split()[0] for line in sums if line.endswith("scenario_site.csv"))
    assert cfg.paths.split_sha256 == sha


def test_site_v2_overrides_only_what_differs_from_base():
    base, run = load_config().to_dict(), load_config([SITE_V2]).to_dict()
    changed = sorted(
        f"{sec}.{key}"
        for sec, vals in run.items()
        if isinstance(vals, dict)
        for key, val in vals.items()
        if base[sec][key] != val
    )
    assert changed == ["data.traj_source", "egovlp.zero_shot", "frame_baseline.enabled"]
    assert run["run_name"] == "site_v2"


def test_all_present(data_tree):
    checks = path_checks(_cfg(), "site_v2")
    assert all(c.status == OK for c in checks), [c for c in checks if c.status != OK]
    names = {c.name for c in checks}
    assert {"takes.json", "pesi EgoVLP", "pesi MobileNetV3-Small", "hash dello split", "take sul disco"} <= names
    assert "pesi CamFormer" not in names  # camformer.pretrained è falso (M04)


def test_missing_weights_fail(data_tree):
    (data_tree / "checkpoints" / "egovlp" / "egovlp.pth").unlink()
    st = _status(path_checks(_cfg(), "site_v2"))
    assert st["pesi EgoVLP"] == FAIL and st["pesi MobileNetV3-Small"] == OK


def test_camformer_weights_only_when_pretrained(data_tree):
    cfg = _cfg("camformer.pretrained=true")
    st = _status(path_checks(cfg, "site_v2"))
    assert st["pesi CamFormer"] == FAIL


def test_split_hash_mismatch_fails(data_tree):
    cfg = _cfg("paths.split_sha256=" + "0" * 64)
    assert _status(path_checks(cfg, "site_v2"))["hash dello split"] == FAIL
    cfg = _cfg("paths.split_sha256=null")
    assert _status(path_checks(cfg, "site_v2"))["hash dello split"] == WARN


def test_takes_partial_warns_none_fails(data_tree):
    takes = data_tree / "egoexo4d" / "takes"
    names = _take_names()
    for name in names[1:]:
        (takes / name).rmdir()
    assert _status(path_checks(_cfg(), "site_v2"))["take sul disco"] == WARN
    (takes / names[0]).rmdir()
    assert _status(path_checks(_cfg(), "site_v2"))["take sul disco"] == FAIL


def test_main_exit_codes(data_tree, capsys):
    args = ["--config", str(SITE_V2), "--set", POSES[0]]
    assert main(args) == 0
    assert "pesi EgoVLP" in capsys.readouterr().out
    (data_tree / "egoexo4d" / "takes.json").unlink()
    assert main(args) == 0  # takes.json mancante: solo un avviso (N5, durata dal video)
    assert "attenzione takes.json" in capsys.readouterr().out
    (data_tree / "checkpoints" / "egovlp" / "egovlp.pth").unlink()
    assert main(args) == 1
    assert "FALLITO" in capsys.readouterr().out


def test_takes_json_required_only_with_device_us(data_tree):
    (data_tree / "egoexo4d" / "takes.json").unlink()
    assert _status(path_checks(_cfg(), "site_v2"))["takes.json"] == WARN
    assert _status(path_checks(_cfg("alignment.mode=device_us"), "site_v2"))["takes.json"] == FAIL


def test_pose_roots_empty_or_without_manifest_fail(data_tree):
    """B3: con traj_source=converted_npz le radici delle pose sono indispensabili (stadio 01)."""
    st = _status(path_checks(load_config([SITE_V2]), "site_v2"))  # pose_roots vuoto, come in base.yaml
    assert st["radici delle pose"] == FAIL
    (data_tree / "poses" / "split_manifest.csv").unlink()
    assert _status(path_checks(_cfg(), "site_v2"))["radice delle pose 0"] == FAIL
    st = _status(path_checks(_cfg('paths.pose_roots=["{data}/poses", "{data}/altrove"]'), "site_v2"))
    assert st["radice delle pose 1"] == FAIL


def test_pose_roots_ignored_with_aria_csv(data_tree):
    names = {c.name for c in path_checks(load_config([SITE_V2], ["data.traj_source=aria_csv"]), "site_v2")}
    assert not any(n.startswith("radic") for n in names)
