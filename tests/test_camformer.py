import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.cascade.config import load_config  # noqa: E402
from src.cascade.models.camformer import (  # noqa: E402
    build_model,
    collate,
    load_pretrained,
    load_swp_module,
    preprocess_poses,
)


def _rand_poses(n, seed=0):
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed)
    q = Rotation.from_euler("zyx", rng.normal(0, 0.3, (n, 3))).as_quat()
    return np.concatenate([np.cumsum(rng.normal(0, 0.01, (n, 3)), 0), q], 1)


def _tiny_arch():
    return load_config(synthetic=True).camformer.arch


def test_preprocess_matches_original():
    du = load_swp_module("utils/dataset_utils.py")
    p = _rand_poses(30)
    out = preprocess_poses(p, "rel9d_grav")
    np.testing.assert_allclose(out, du.transform_camera_pose(p, 11))
    assert out.shape == (30, 12)
    np.testing.assert_allclose(out[15, :3], 0, atol=1e-9)  # relativa al frame centrale


def _batch(n=3, T=40):
    items = []
    for i in range(n):
        x = preprocess_poses(_rand_poses(T - i, seed=i), "rel9d_grav").astype(np.float32)
        items.append({"x": x, "win_start": 5, "win_end": 25, "label": i % 8, "clip_id": f"c{i}"})
    return collate(items)


def test_forward_shapes_and_window_pooling():
    torch.manual_seed(0)
    arch = _tiny_arch()
    model = build_model(arch, 8).eval()
    b = _batch()
    logits, emb = model(b["x"], b["pad_mask"], b["lengths"], b["win_start"], b["win_end"])
    assert logits.shape == (3, 8) and emb.shape == (3, arch.output_dim)
    feats = model.encoder(b["x"], b["pad_mask"], b["lengths"])
    torch.testing.assert_close(emb, feats[:, 5:25].mean(1))


def test_load_pretrained_prefix_and_missing(tmp_path):
    arch = _tiny_arch()
    src = build_model(arch, 8)
    sd = {"model." + k: v for k, v in src.encoder.state_dict().items()}
    sd["cls_model.weight"] = torch.randn(8, arch.output_dim)
    sd["cls_model.bias"] = torch.randn(8)
    torch.save({"state_dict": sd}, tmp_path / "ck.pt")
    dst = build_model(arch, 8)
    info = load_pretrained(dst, tmp_path / "ck.pt", init_head=True)
    assert info["head_loaded"] == [True]
    for k, v in src.encoder.state_dict().items():
        torch.testing.assert_close(dst.encoder.state_dict()[k], v)
    torch.testing.assert_close(dst.head.weight, sd["cls_model.weight"])
    bad = {k: v for k, v in sd.items() if "in_proj_gravity" not in k}
    torch.save({"state_dict": bad}, tmp_path / "bad.pt")
    with pytest.raises(RuntimeError, match="mancano"):
        load_pretrained(build_model(arch, 8), tmp_path / "bad.pt")


def _real_ckpt():
    root = os.environ.get("DL4CV_ROOT")
    return Path(root) / "data" / "checkpoints" / "camformer" / "egoexo4d_dur8.pt" if root else None


@pytest.mark.skipif(not (_real_ckpt() and _real_ckpt().exists()), reason="checkpoint CamFormer reale assente")
def test_real_checkpoint_loads():
    model = build_model(load_config().camformer.arch, 8)
    load_pretrained(model, _real_ckpt())


def _one_step(freeze: bool):
    from src.cascade.finetune import make_optimizer

    cfg = load_config(
        synthetic=True, overrides=["camformer.pretrained=true", f"camformer.freeze_encoder={str(freeze).lower()}"]
    )
    torch.manual_seed(0)
    model = build_model(cfg.camformer.arch, 8)
    opt = make_optimizer(model, cfg)
    enc0 = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    head0 = {k: v.clone() for k, v in model.head.state_dict().items()}
    model.train()
    if freeze:
        model.encoder.eval()
    b = _batch()
    logits, _ = model(b["x"], b["pad_mask"], b["lengths"], b["win_start"], b["win_end"])
    torch.nn.functional.cross_entropy(logits, b["label"]).backward()
    opt.step()
    enc_moved = any(not torch.equal(enc0[k], v) for k, v in model.encoder.state_dict().items())
    head_moved = any(not torch.equal(head0[k], v) for k, v in model.head.state_dict().items())
    return model, opt, enc_moved, head_moved


def test_freeze_encoder_trains_only_the_head():
    """M17: con freeze_encoder l'encoder del checkpoint resta identico e impara solo la testa."""
    model, opt, enc_moved, head_moved = _one_step(freeze=True)
    assert head_moved and not enc_moved
    assert len(opt.param_groups) == 1
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


def test_without_freeze_the_encoder_is_finetuned():
    model, opt, enc_moved, head_moved = _one_step(freeze=False)
    assert head_moved and enc_moved and len(opt.param_groups) == 2


def test_freeze_needs_pretrained_weights():
    with pytest.raises(Exception, match="freeze_encoder"):
        load_config(synthetic=True, overrides=["camformer.freeze_encoder=true"])
    cfg = load_config()  # configs/camformer.yaml: la run usa il checkpoint congelato (M17)
    assert cfg.camformer.pretrained and cfg.camformer.freeze_encoder
