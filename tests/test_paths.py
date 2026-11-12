import pytest

from src import dl4cv_root
from src.cascade import paths


def test_missing_root_is_explicit_error(monkeypatch):
    for v in ("DL4CV_ROOT", paths.DATA_ENV, paths.RUN_ENV):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(RuntimeError, match="DL4CV_ROOT"):
        dl4cv_root()
    with pytest.raises(RuntimeError):
        paths.data_root()


def test_roots_and_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("DL4CV_ROOT", str(tmp_path))
    monkeypatch.delenv(paths.DATA_ENV, raising=False)
    monkeypatch.delenv(paths.RUN_ENV, raising=False)
    assert paths.data_root() == tmp_path / "data"
    assert paths.run_root() == tmp_path / "runs"
    monkeypatch.setenv(paths.DATA_ENV, str(tmp_path / "d2"))
    monkeypatch.setenv(paths.RUN_ENV, str(tmp_path / "r2"))
    assert paths.data_root() == tmp_path / "d2"
    assert paths.run_dir("x") == tmp_path / "r2" / "x"
    with pytest.raises(ValueError):
        paths.run_dir("../x")


def test_resolve_templates(tmp_path):
    d, r = tmp_path / "data", tmp_path / "run"
    assert paths.resolve("{data}/a.csv", data=d) == d / "a.csv"
    assert paths.resolve("b/c.csv", data=d) == d / "b" / "c.csv"
    assert paths.resolve("{run}/x", data=d, run=r) == r / "x"
    assert paths.resolve("{takes_root}/{take_name}.mp4", data=d, takes_root=d / "t", take_name="k") == d / "t" / "k.mp4"
    with pytest.raises(ValueError):
        paths.resolve("{run}/x", data=d)
