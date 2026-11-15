"""M07: un cambio di fold o di split dopo lo stadio 20 deve invalidare gli stadi che ne dipendono."""

import pytest

from src.cascade.config import load_config
from src.cascade.manifest import (
    CLIP_DEPENDENT_STAGES,
    CLIP_KEYS,
    SPLIT_DEPENDENT_STAGES,
    SPLIT_KEYS,
    STAGE_KEYS,
    ConfigMismatch,
    is_done,
    require_stage,
    write_manifest,
)

BY_SITE = ["folds.group_by=site", "folds.site_map=splits/site_v1/split.json"]


@pytest.mark.parametrize("stage", SPLIT_DEPENDENT_STAGES)
def test_dependent_stages_track_split_and_folds(stage):
    for key in SPLIT_KEYS:
        assert key in STAGE_KEYS[stage]


def _stage_20_done(tmp_path, cfg):
    out = tmp_path / "preds.parquet"
    out.write_text("x")
    write_manifest(tmp_path, "20_camformer", cfg.to_dict(), [out], part="fold0")
    return out


def test_changed_folds_invalidate_stage_20(tmp_path):
    out = _stage_20_done(tmp_path, load_config())
    assert is_done(tmp_path, "20_camformer", load_config().to_dict(), [out], part="fold0")
    by_site = load_config(overrides=BY_SITE).to_dict()
    with pytest.raises(ConfigMismatch):  # lo stadio 20 non si salta in silenzio
        is_done(tmp_path, "20_camformer", by_site, [out], part="fold0")
    with pytest.raises(ConfigMismatch):  # e uno stadio a valle (es. 45) non lo accetta come a monte
        require_stage(tmp_path, "20_camformer", by_site, parts=["fold0"])


def test_changed_split_hash_invalidates_stage_20(tmp_path):
    out = _stage_20_done(tmp_path, load_config(overrides=["paths.split_sha256=aaaa"]))
    other = load_config(overrides=["paths.split_sha256=bbbb"]).to_dict()
    with pytest.raises(ConfigMismatch):
        is_done(tmp_path, "20_camformer", other, [out], part="fold0")


def test_stage_25_does_not_depend_on_folds(tmp_path):
    out = tmp_path / "features.parquet"
    out.write_text("x")
    write_manifest(tmp_path, "25_egovlp", load_config().to_dict(), [out])
    assert is_done(tmp_path, "25_egovlp", load_config(overrides=BY_SITE).to_dict(), [out])


@pytest.mark.parametrize("stage", CLIP_DEPENDENT_STAGES)
def test_dependent_stages_track_clip_definition(stage):
    """I2: le chiavi dello stadio 00 (finestre, durata, sorgente delle traiettorie, hash) arrivano a valle."""
    assert set(CLIP_KEYS) <= set(STAGE_KEYS[stage])
    assert "paths.split_sha256" in CLIP_KEYS and "data.traj_source" in CLIP_KEYS


def test_changed_clip_definition_invalidates_stage_20(tmp_path):
    out = _stage_20_done(tmp_path, load_config())
    other = load_config(overrides=["data.duration_source=video"]).to_dict()
    with pytest.raises(ConfigMismatch):
        is_done(tmp_path, "20_camformer", other, [out], part="fold0")


def test_stage_41_has_its_own_keys(tmp_path):
    out = tmp_path / "summary.json"
    out.write_text("x")
    write_manifest(tmp_path, "41_cost_e2e", load_config().to_dict(), [out])
    with pytest.raises(ConfigMismatch):
        is_done(tmp_path, "41_cost_e2e", load_config(overrides=["cost_e2e.n_clips=3"]).to_dict(), [out])
