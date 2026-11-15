"""I1: una classe con una sola take nel fit (Bouldering nelle fold 1-4 di site_v1) non deve
spegnere la scelta di C né sporcare la perdita della CV interna."""

import numpy as np
import pandas as pd
import pytest

from src import REPO_ROOT
from src.cascade.calibration import apply_temperature
from src.cascade.heads import _logits, fit_head

CS = [0.001, 0.01, 0.1, 1.0, 10.0]
N_CLASSES = 8
RARE = 5  # Bouldering


def _data(counts, d=16, seed=0):
    """Feature gaussiane con una media per classe; una clip per take."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(N_CLASSES), counts)
    X = rng.standard_normal((len(y), d)) + 1.5 * rng.standard_normal((N_CLASSES, d))[y]
    return X, y, np.array([f"take{i:04d}" for i in range(len(y))])


def _fold_counts(k):
    """Take per classe nel fit della fold ``k``, dalle fold versionate."""
    f = pd.read_csv(REPO_ROOT / "splits" / "site_v1" / "folds.csv")
    fit = (f.fold != k).to_numpy() & ~f[f"holdout_f{k}"].to_numpy(bool)
    return np.bincount(f.label_idx[fit], minlength=N_CLASSES)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_versioned_folds_have_a_single_take_class(k):
    assert _fold_counts(k)[RARE] == 1  # il caso che questo file protegge esiste davvero


def test_single_take_class_does_not_change_the_chosen_C():
    X, y, g = _data([40, 40, 0, 12, 20, 1, 9, 9])
    keep = y != RARE
    _, C_with = fit_head(X, y, g, CS, 3, 500, N_CLASSES)
    _, C_without = fit_head(X[keep], y[keep], g[keep], CS, 3, 500, N_CLASSES)
    assert C_with == C_without


def test_columns_stay_aligned_and_the_rare_class_is_learned():
    X, y, g = _data([40, 40, 0, 12, 20, 1, 9, 9])
    model, _ = fit_head(X, y, g, CS, 3, 500, N_CLASSES)
    assert RARE in model.pipe[-1].classes_  # il fit finale vede la take rara
    lo = _logits(model, X, N_CLASSES)
    assert lo.shape == (len(X), N_CLASSES)
    assert (lo[:, 2] == -50.0).all()  # Music: assente, colonna al suo posto
    p = apply_temperature(lo, 1.0)
    assert np.allclose(p.sum(1), 1.0) and p[:, 2].max() < 1e-12
    assert (lo.argmax(1) == y).mean() > 0.9  # le colonne corrispondono alle classi giuste


def test_no_class_with_two_takes_falls_back_with_a_warning(caplog):
    X, y, g = _data([1, 1, 0, 1, 1, 1, 1, 1])
    with caplog.at_level("WARNING"):
        _, C = fit_head(X, y, g, CS, 3, 200, N_CLASSES)
    assert C == CS[0] and "CV interna di C saltata" in caplog.text
