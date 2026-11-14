"""CompareGate (C11, M06): score = η̂₁(z) − p_max, con η̂₁ stimata sull'etichetta c1.

Test sintetico del blueprint (riadatta il ``synthetic_data`` di Mozannar, Lang et al.): un L1
che riesce dove una feature di moto è alta. Lavora sulla matrice delle feature, senza cache.
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.cascade.gate.models import CompareGate, LogRegGate, ThresholdGate, make_gate

CS = [0.01, 0.1, 1.0, 10.0]


def _data(n: int = 1200, seed: int = 0):
    """A = (p_max, margin, entropia) e C = tre colonne di moto; c1 = 1 dove la velocità è alta."""
    rng = np.random.default_rng(seed)
    p_max = rng.uniform(0.3, 1.0, n)
    A = np.stack([p_max, p_max - rng.uniform(0.0, 0.3, n), rng.uniform(0.0, 1.0, n)], axis=1)
    speed = rng.normal(size=n)
    C = np.column_stack([speed, rng.normal(size=(n, 2))])
    c1 = ((speed > 0) ^ (rng.random(n) < 0.05)).astype(int)
    take = np.repeat(np.arange(n // 2), 2).astype(str)
    return A, C, speed, c1, take


def test_first_group_must_be_A():
    with pytest.raises(ValueError, match="primo gruppo"):
        CompareGate("compare_C", ["C"], CS, 3)
    with pytest.raises(ValueError, match="primo gruppo"):
        CompareGate("compare_", [], CS, 3)


def test_no_class_weight_and_target():
    g = CompareGate("compare_A", ["A"], CS, 3)
    assert g.class_weight is None and g.target == "c1" and g.groups == ["A"]
    assert isinstance(g, LogRegGate)


def test_score_is_eta1_minus_pmax():
    A, _, _, c1, take = _data()
    g = CompareGate("compare_A", ["A"], CS, 3).fit(A, c1, seed=0, groups=take)
    eta1 = LogRegGate.score(g, A)
    assert ((eta1 >= 0) & (eta1 <= 1)).all()
    np.testing.assert_allclose(g.score(A), eta1 - A[:, 0])


def test_constant_eta1_orders_like_threshold():
    """Se c1 ha una sola classe nei dati di fit, lo score ordina le clip come thr_maxprob."""
    A, _, _, _, take = _data()
    g = CompareGate("compare_A", ["A"], CS, 3).fit(A, np.ones(len(A), dtype=int), seed=0, groups=take)
    s = g.score(A)
    np.testing.assert_allclose(s, 1.0 - A[:, 0])
    thr = ThresholdGate("thr_maxprob", "p_max", -1.0).score(A)
    np.testing.assert_array_equal(np.argsort(s, kind="stable"), np.argsort(thr, kind="stable"))


def test_trajectory_group_recovers_c1_where_confidence_cannot():
    """η̂₁ da sola confidenza non vede dove L1 riesce; con il moto sì. Tra compare_A e
    compare_AC cambia solo il gruppo C."""
    A, C, speed, c1, take = _data(n=1600, seed=1)
    tr, te = np.arange(1200), np.arange(1200, 1600)
    gA = CompareGate("compare_A", ["A"], CS, 3).fit(A[tr], c1[tr], seed=0, groups=take[tr])
    X = np.hstack([A, C])
    gAC = CompareGate("compare_AC", ["A", "C"], CS, 3).fit(X[tr], c1[tr], seed=0, groups=take[tr])
    auc_a = roc_auc_score(c1[te], LogRegGate.score(gA, A[te]))
    auc_ac = roc_auc_score(c1[te], LogRegGate.score(gAC, X[te]))
    assert auc_a < 0.6 < 0.9 < auc_ac, (auc_a, auc_ac)


def test_natural_threshold_scales_where_l1_is_more_likely_right():
    """La soglia «naturale» è s > 0: si scala dove η̂₁ supera la confidenza di L0."""
    A, C, speed, c1, take = _data(n=1600, seed=2)
    X = np.hstack([A, C])
    tr, te = np.arange(1200), np.arange(1200, 1600)
    g = CompareGate("compare_AC", ["A", "C"], CS, 3).fit(X[tr], c1[tr], seed=0, groups=take[tr])
    s = g.score(X[te])
    fast, slow = speed[te] > 0.5, speed[te] < -0.5
    assert (s[fast] > 0).mean() > 0.8
    assert (s[slow] > 0).mean() < 0.05


def test_make_gate_builds_compare_from_name():
    from src.cascade.config import load_config

    gcfg = load_config().gate
    g = make_gate("compare_AC", gcfg)
    assert isinstance(g, CompareGate) and g.Cs == gcfg.compare.Cs and g.inner_cv == gcfg.compare.inner_cv
    with pytest.raises(ValueError, match="primo gruppo"):
        make_gate("compare_C", gcfg)
