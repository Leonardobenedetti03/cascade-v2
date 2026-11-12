import numpy as np
import pandas as pd

from src.cascade.folds import make_folds, training_masks


def _takes(n_per_class=30, n_classes=8):
    rows = [{"take_uid": f"t{c}_{i}", "label_idx": c} for c in range(n_classes) for i in range(n_per_class)]
    return pd.DataFrame(rows)


def test_folds_stratified_and_disjoint():
    f = make_folds(_takes(), 5, 0.1, seed=0)
    assert f.take_uid.is_unique
    assert set(f.fold) == set(range(5))
    ct = pd.crosstab(f.label_idx, f.fold)
    assert (ct.max(axis=1) - ct.min(axis=1)).max() <= 1


def test_inner_holdout_inside_training_folds():
    f = make_folds(_takes(), 5, 0.1, seed=0)
    for k in range(5):
        hold = f[f"holdout_f{k}"]
        assert not (hold & (f.fold == k)).any()
        frac = hold.sum() / (f.fold != k).sum()
        assert 0.07 < frac < 0.13
        tr, ho, ev = training_masks(f, str(k))
        assert not (tr & ho).any() and not (tr & ev).any() and not (ho & ev).any()
        assert (tr | ho | ev).all()
    tr, ho, ev = training_masks(f, "full")
    assert not ev.any() and (tr ^ ho).all()


def test_folds_deterministic():
    a = make_folds(_takes(), 5, 0.1, seed=3)
    b = make_folds(_takes().sample(frac=1, random_state=1), 5, 0.1, seed=3)
    pd.testing.assert_frame_equal(a, b)
    c = make_folds(_takes(), 5, 0.1, seed=4)
    assert not np.array_equal(a.fold, c.fold)
