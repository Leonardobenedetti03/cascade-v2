"""Fusione L0 + L1 (src/cascade/fusion.py): replace di default (M05), fusione log-lineare come ablation."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.cascade.fusion import choose_simplex, choose_w, compare_fusions, fuse, fuse_many, simplex_grid


def _p(rows):
    a = np.asarray(rows, float)
    return a / a.sum(1, keepdims=True)


def test_fuse_extremes_and_normalization():
    pc, pv = _p([[0.7, 0.2, 0.1]]), _p([[0.1, 0.1, 0.8]])
    np.testing.assert_allclose(fuse(pc, pv, 1.0), pc, atol=1e-9)
    np.testing.assert_allclose(fuse(pc, pv, 0.0), pv, atol=1e-9)
    f = fuse(pc, pv, 0.5)
    assert f.sum() == pytest.approx(1.0)
    expected = np.sqrt(pc * pv) / np.sqrt(pc * pv).sum()
    np.testing.assert_allclose(f, expected, atol=1e-9)
    with pytest.raises(ValueError):
        fuse(pc, pv, 1.5)


def test_choose_w_max_acc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 300)
    pv = np.full((300, 4), 0.1)
    pv[np.arange(300), y] = 0.7  # L1 sempre corretto
    pc = rng.dirichlet(np.ones(4), 300)  # L0 rumore
    w, grid = choose_w(pc, _p(pv), y, step=0.1, mode="fuse")
    best = max(g["acc"] for g in grid)
    chosen = next(g for g in grid if g["w"] == w)
    assert chosen["acc"] == best
    assert w < 0.5


def test_choose_w_tie_breaks_on_nll():
    y = np.array([0, 1])
    pc = _p([[0.9, 0.1], [0.1, 0.9]])
    pv = _p([[0.6, 0.4], [0.4, 0.6]])
    w, grid = choose_w(pc, pv, y, step=0.25, mode="fuse")
    assert all(g["acc"] == 1.0 for g in grid)
    assert w == 1.0  # a parità di accuracy vince la NLL più bassa (L0 più sicuro qui)


def test_replace_is_the_default_mode():
    pc, pv, y = _p([[1, 2]]), _p([[2, 1]]), np.array([0])
    w, grid = choose_w(pc, pv, y)
    assert w == 0.0 and len(grid) == 1  # L1 sostituisce L0
    assert grid[0]["acc"] == 1.0  # la classe 0 è quella di L1
    assert choose_w(pc, pv, y, mode="replace") == (w, grid)


def test_unknown_mode_is_rejected():
    for mode in ("learned", "boh"):
        with pytest.raises(ValueError, match="fusion.mode"):
            choose_w(_p([[1, 2]]), _p([[2, 1]]), np.array([0]), mode=mode)


def test_compare_fusions_has_no_learned_system():
    rng = np.random.default_rng(1)
    n, c = 40, 3
    y = rng.integers(0, c, n)
    tab = SimpleNamespace(
        df=pd.DataFrame({"label_idx": y, "split": ["train"] * 25 + ["val"] * 15}),
        arrays={"cam_probs_cal": rng.dirichlet(np.ones(c), n), "vid_probs_cal": rng.dirichlet(np.ones(c), n)},
    )
    comp = compare_fusions(tab, 0.3, n_boot=50, seed=0)
    assert set(comp.system) == {"L0", "L1 (sostituzione)", "log-lineare (w=0.30)"}
    assert set(comp.split) == {"train", "val"}
    assert comp.loc[comp.system == "L1 (sostituzione)", "diff_vs_L1"].isna().all()  # L1 contro se stesso: nessun IC


def test_simplex_grid_and_choose_simplex():
    pts = simplex_grid(3, 0.5)
    assert len(pts) == 6 and all(sum(p) == pytest.approx(1.0) for p in pts)
    with pytest.raises(ValueError):
        simplex_grid(4, 0.5)
    y = np.array([0, 1, 2, 0])
    good = np.full((4, 3), 0.1)
    good[np.arange(4), y] = 0.8
    bad = np.full((4, 3), 1 / 3)
    wts, acc, _ = choose_simplex([good, bad, bad], y, step=0.5)
    assert acc == 1.0 and wts[0] > 0
    np.testing.assert_allclose(fuse_many([good, bad], (1.0, 0.0)), good / good.sum(1, keepdims=True), atol=1e-9)


def test_config_fusion_defaults_and_no_learned_mode():
    from src.cascade.config import load_config

    cfg = load_config()
    assert cfg.fusion.mode == "replace"
    assert not hasattr(cfg.fusion, "learned_features") and not hasattr(cfg.fusion, "compare_learned")
    assert load_config(overrides=["fusion.mode=fuse"]).fusion.mode == "fuse"
    with pytest.raises(ValueError, match="fusion.mode"):
        load_config(overrides=["fusion.mode=learned"])
