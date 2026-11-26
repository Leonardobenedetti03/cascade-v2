import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")

from src.cascade.models.gmac.egovlp import (  # noqa: E402
    TextEncoder,
    build_video_model,
    load_video_weights,
    normalize,
    zero_shot_logits,
)
from src.cascade.utils.config import load_config  # noqa: E402


def _arch(**kw):
    a = load_config(synthetic=True).egovlp.arch
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_forward_shapes():
    m = build_video_model(_arch()).eval()
    cls, emb = m(torch.rand(2, 4, 3, 32, 32))
    assert cls.shape == (2, 32) and emb.shape == (2, 16)


def test_load_weights_module_prefix_and_text_ignored():
    src = build_video_model(_arch())
    sd = {"module." + k: v.clone() for k, v in src.state_dict().items()}
    sd["module.text_model.embeddings.word_embeddings.weight"] = torch.zeros(3, 3)
    sd["module.txt_proj.1.weight"] = torch.zeros(16, 768)
    from src.cascade.models.gmac.egovlp import _strip_module

    dst = build_video_model(_arch())
    info = load_video_weights(dst, _strip_module(sd))
    assert "text_model.embeddings.word_embeddings.weight" in info["ignored"]
    for k, v in src.state_dict().items():
        torch.testing.assert_close(dst.state_dict()[k], v)


def test_missing_video_key_is_error():
    sd = {k: v for k, v in build_video_model(_arch()).state_dict().items() if not k.startswith("vid_proj")}
    with pytest.raises(RuntimeError, match="mancano"):
        load_video_weights(build_video_model(_arch()), sd)


def test_temporal_embed_inflation_with_zeros():
    src = build_video_model(_arch(num_frames=2))
    sd = dict(src.state_dict())
    sd["video_model.temporal_embed"] = torch.ones_like(sd["video_model.temporal_embed"])
    dst = build_video_model(_arch(num_frames=4))
    load_video_weights(dst, sd)
    te = dst.state_dict()["video_model.temporal_embed"]
    assert te.shape[1] == 4 and torch.all(te[:, :2] == 1) and torch.all(te[:, 2:] == 0)


def test_normalize_layout_and_values():
    fr = np.full((1, 4, 8, 8, 3), 255, np.uint8)
    fr[..., 1] = 0
    x = normalize(fr, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    assert tuple(x.shape) == (1, 4, 3, 8, 8)
    assert float(x[0, 0, 0, 0, 0]) == pytest.approx((1 - 0.485) / 0.229, rel=1e-5)
    assert float(x[0, 0, 1, 0, 0]) == pytest.approx(-0.456 / 0.224, rel=1e-5)


def test_zero_shot_follows_prompt_order():
    enc = TextEncoder("hashing", 16)
    prompts = ["#C C plays soccer", "#C C cooks food", "#C C repairs a bicycle"]
    t = enc.encode(prompts)
    logits = zero_shot_logits(t[[2, 0, 1]], t, temperature=0.05)
    np.testing.assert_array_equal(logits.argmax(1), [2, 0, 1])
    np.testing.assert_allclose(enc.encode(prompts), t)  # deterministic


def _real_ckpt():
    root = os.environ.get("DL4CV_ROOT")
    return Path(root) / "data" / "checkpoints" / "egovlp" / "egovlp.pth" if root else None


@pytest.mark.skipif(not (_real_ckpt() and _real_ckpt().exists()), reason="checkpoint EgoVLP reale assente")
def test_real_checkpoint_loads():
    from src.cascade.models.gmac.egovlp import load_state

    load_video_weights(build_video_model(load_config().egovlp.arch), load_state(_real_ckpt()))


def test_load_state_ignores_classes_missing_from_the_path(tmp_path, monkeypatch):
    """The real checkpoint contains a ``parse_config.ConfigParser`` from the EgoVLP repository (an error we hit on
    the cluster on 16/11): ``load_state`` must return the ``state_dict`` even without that module."""
    import sys
    import types

    from src.cascade.models.gmac.egovlp import load_state

    fake = types.ModuleType("parse_config")

    class ConfigParser:
        def __init__(self):
            self.config = {"arch": "x"}

    ConfigParser.__module__, ConfigParser.__qualname__ = "parse_config", "ConfigParser"
    fake.ConfigParser = ConfigParser
    monkeypatch.setitem(sys.modules, "parse_config", fake)
    ck = {"state_dict": {"module.video_model.w": torch.ones(2)}, "config": ConfigParser(), "epoch": 3}
    torch.save(ck, tmp_path / "ck.pth")
    monkeypatch.delitem(sys.modules, "parse_config")
    with pytest.raises(ModuleNotFoundError):
        torch.load(tmp_path / "ck.pth", map_location="cpu", weights_only=False)
    sd = load_state(tmp_path / "ck.pth")
    assert list(sd) == ["video_model.w"] and torch.equal(sd["video_model.w"], torch.ones(2))
