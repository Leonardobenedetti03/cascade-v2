import pytest

from src.cascade.utils.config import config_from_dict, deep_merge, load_config


def test_defaults_load():
    cfg = load_config()
    assert cfg.data.window_s == 8.0
    assert cfg.camformer.arch.output_dim == 512
    assert cfg.egovlp.arch.embed_dim == 768
    assert "oracle" in cfg.gate.models


def test_synthetic_and_overrides():
    cfg = load_config(synthetic=True, overrides=["data.window_s=16", "gate.models=[oracle]"])
    assert cfg.run_name == "synthetic"
    assert cfg.data.window_s == 16
    assert cfg.gate.models == ["oracle"]
    assert cfg.data.expected_split_counts is None


def test_unknown_key_is_error():
    with pytest.raises(KeyError):
        config_from_dict({"data": {"window_sec": 8}})
    with pytest.raises(KeyError):
        load_config(overrides=["gate.modles=[x]"])


def test_bad_override_syntax():
    with pytest.raises(ValueError):
        load_config(overrides=["data.window_s"])


def test_deep_merge():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}


def test_removed_sections_are_unknown():
    """M12: streaming, keystep, task, bayes, level_minus1 and the L1 fine-tuning no longer exist."""
    for key in ("streaming", "task", "keystep"):
        with pytest.raises(KeyError):
            config_from_dict({key: {}})
    with pytest.raises(KeyError):
        config_from_dict({"gate": {"bayes": {}}})
    with pytest.raises(KeyError):
        config_from_dict({"gate": {"level_minus1": {}}})
    with pytest.raises(KeyError):
        config_from_dict({"egovlp": {"finetune": {}}})


def test_site_folds_need_site_map():
    with pytest.raises(ValueError):
        config_from_dict({"folds": {"group_by": "site"}})
    cfg = config_from_dict({"folds": {"group_by": "site", "site_map": "splits/site_v1/split.json"}})
    assert cfg.folds.group_by == "site"
    with pytest.raises(ValueError):
        config_from_dict({"folds": {"group_by": "random"}})


def test_new_sections_have_defaults():
    cfg = load_config()
    assert cfg.gate.compare.inner_cv == 3 and 1.0 in cfg.gate.compare.Cs
    assert cfg.cost_e2e.n_clips == 200 and cfg.cost_e2e.passes == 2
    assert cfg.site_diag.C == 0.1 and cfg.site_diag.cv_folds == 5
    assert (
        cfg.folds.group_by == "site"
        and cfg.paths.split_sha256 == "c9f4e3083f57b099ef9693a5fa57b2f8666bddb6f4377834a7d015e2c7e0344d"
    )
    assert load_config(synthetic=True).folds.group_by == "take"


def test_removed_gate_fields_are_unknown():
    """M06: beta_*, prior_confusion, reliability and ridge_alphas no longer exist (note N16)."""
    for key in ("beta_n0", "beta_q", "prior_confusion", "reliability", "ridge_alphas"):
        with pytest.raises(KeyError):
            config_from_dict({"gate": {key: 1}})
