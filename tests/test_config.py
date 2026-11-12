import pytest

from src.cascade.config import config_from_dict, deep_merge, load_config


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
