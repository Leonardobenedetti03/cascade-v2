"""Curve della cascata e metriche (C12b, M09 e M14): ordinamento, costi, accuratezza bilanciata,
I(δ), soglie dal budget. Solo numpy e scikit-learn: nessun test richiede torch."""

import numpy as np
import pytest
from sklearn.metrics import recall_score

from src.cascade.evaluation.curves import (
    acc_at_budget,
    area_norm,
    balanced_acc,
    boot_bacc,
    boot_curves,
    cascade_curve,
    class_membership,
    curve_metrics,
    escalation_order,
    improvement_factor,
    min_cost_within,
    rate_for_budget,
    resample_weights,
    threshold_at_rate,
    threshold_for_budget,
)

C0, C1 = 0.12, 92.0
N_CLASSES = 4


def _world(n_takes=120, per_take=3, seed=0):
    """Take con 4 classi (la 3 è rara: 2 take), L0 al 55%, L1 al 90%, una clip su due per take."""
    rng = np.random.default_rng(seed)
    lab_take = rng.integers(0, 3, n_takes)
    lab_take[:2] = 3
    lab = np.repeat(lab_take, per_take)
    take = np.repeat(np.arange(n_takes), per_take).astype(str)
    N = len(lab)

    def probs(acc):
        p = rng.dirichlet(np.ones(N_CLASSES), N)
        hit = rng.random(N) < acc
        p[np.flatnonzero(hit), lab[hit]] += 3
        return p / p.sum(1, keepdims=True)

    p0, p1 = probs(0.55), probs(0.9)
    c0, c1 = (p0.argmax(1) == lab), (p1.argmax(1) == lab)
    oracle = c1.astype(float) - c0 + 0.01 * rng.random(N)
    return dict(p0=p0, p1=p1, lab=lab, take=take, oracle=oracle, rng=rng)


@pytest.fixture(scope="module")
def w():
    return _world()


def _curve(w, scores, **kw):
    return cascade_curve(scores, w["p0"], w["p1"], w["lab"], w["take"], C0, C1, **kw)


def test_order_is_descending_and_stable():
    pos = escalation_order(np.array([0.1, 0.9, 0.9, 0.5]))
    np.testing.assert_array_equal(pos, [3, 0, 1, 2])  # a parità vince l'ordine di riga


def test_endpoints_are_always_l0_and_always_l1(w):
    c = _curve(w, w["oracle"])
    assert c.r[0] == 0 and c.r[-1] == 100
    n = len(w["lab"])
    # r = 0: nessuna escalation, costo = n_clip · C0; r = 100: tutte le clip scalate
    np.testing.assert_allclose(c.mean_cost[0], n / 120 * C0)
    np.testing.assert_allclose(c.mean_cost[-1], n / 120 * (C0 + C1))
    fixed0 = _curve(w, np.zeros(n), fixed_r=0)
    fixed1 = _curve(w, np.zeros(n), fixed_r=100)
    assert np.allclose(fixed0.acc, c.acc[0]) and np.allclose(fixed1.acc, c.acc[-1])


def test_cost_is_monotone_and_oracle_beats_random(w):
    c = _curve(w, w["oracle"])
    assert np.all(np.diff(c.mean_cost) >= -1e-12)
    rnd = _curve(w, w["rng"].random(len(w["lab"])))
    assert area_norm(c.mean_cost, c.acc) > area_norm(rnd.mean_cost, rnd.acc)


def test_bacc_matches_sklearn_on_kept_classes(w):
    """A r = 0 e r = 100 la bacc è il richiamo medio (= balanced_accuracy_score) sulle classi con >= 5 take."""
    c = _curve(w, w["oracle"], min_takes=5)
    assert c.n_classes == 3  # la classe 3 ha 2 take e resta fuori
    for ri, p in ((0, w["p0"]), (100, w["p1"])):
        uniq, inv = np.unique(w["take"], return_inverse=True)
        agg = np.zeros((len(uniq), N_CLASSES))
        np.add.at(agg, inv, np.log(np.clip(p, 1e-12, 1)))
        lt = np.zeros(len(uniq), int)
        lt[inv] = w["lab"]
        keep = np.bincount(lt, minlength=N_CLASSES) >= 5
        m = keep[lt]
        ref = recall_score(lt[m], agg.argmax(1)[m], labels=np.flatnonzero(keep), average="macro")  # = bacc
        assert c.bacc[ri] == pytest.approx(ref)


def test_class_membership_threshold():
    M, keep = class_membership(np.array([0, 0, 0, 1, 1, 2]), 3, min_takes=2)
    assert list(keep) == [0, 1] and M.shape == (2, 6)
    M0, keep0 = class_membership(np.array([0, 1]), 3, min_takes=5)
    assert M0.shape[0] == 0
    assert np.isnan(balanced_acc(np.ones((4, 2)), M0)).all()


def test_bootstrap_bacc_matches_point_with_unit_weights(w):
    c = _curve(w, w["oracle"], min_takes=5)
    ones = np.ones((1, c.correct.shape[1]))
    np.testing.assert_allclose(boot_bacc(c, ones)[0], c.bacc)
    np.testing.assert_allclose(boot_curves(c, ones)[0][0], c.acc)
    W = resample_weights(c.correct.shape[1], 40, seed=1)
    assert boot_bacc(c, W).shape == (40, len(c.r)) and np.isfinite(boot_bacc(c, W)).mean() > 0.95


def test_improvement_factor_definition_and_lower_bound(w):
    c = _curve(w, w["oracle"])
    cost, acc = c.mean_cost, c.acc
    c_l1 = cost[-1] - cost[0]
    i0 = improvement_factor(cost, acc, acc[-1], 0.0)
    assert i0 == pytest.approx(c_l1 / min_cost_within(cost, acc, acc[-1], 0.0))
    # mai sotto C_L1 / (C_L0 + C_L1): a r = 100 la cascata ha l'accuratezza di L1
    floor = C1 / (C0 + C1)
    assert improvement_factor(cost, acc, acc[-1], 0.0) >= floor - 1e-9
    assert improvement_factor(cost, acc, acc[-1], 0.01) >= i0 - 1e-9  # più tolleranza, più risparmio


def test_improvement_factor_floor_when_gate_is_useless():
    r = np.arange(0, 101, 1.0)
    cost = C0 + r / 100 * C1
    acc = np.linspace(0.5, 0.9, 101)  # l'accuratezza di L1 si raggiunge solo a r = 100
    got = improvement_factor(cost, acc, acc[-1], 0.0)
    assert got == pytest.approx(C1 / (C0 + C1))


def test_ratio_is_nan_when_oracle_equals_random():
    out = curve_metrics(
        np.linspace(0, 1, 5),
        np.full(5, 0.5),
        ref_cost_L1=1.0,
        ref_acc_L1=0.5,
        budgets=[0.5],
        within_pts=0.01,
        area_random=0.5,
        area_oracle=0.5,
    )
    assert np.isnan(out["oracle_gain_frac"])


def test_acc_at_budget_unreachable_is_nan():
    cost = np.array([[1.0, 2.0, 3.0]])
    assert np.isnan(acc_at_budget(cost, np.array([[0.5, 0.6, 0.7]]), 0.5)).all()
    assert acc_at_budget(cost, np.array([[0.5, 0.6, 0.7]]), 2.5)[0] == 0.6


def test_threshold_at_rate_hits_the_requested_share():
    s = np.random.default_rng(0).random(1000)
    for rate in (0.0, 0.1, 0.3, 1.0):
        th = threshold_at_rate(s, rate)
        assert (s >= th).mean() == pytest.approx(rate, abs=1e-9)


def test_budget_to_rate_uses_cost_and_clips():
    # C_L0 + r·C_L1 <= b·(C_L0 + C_L1): con b = 0.5 la quota è ~49,9%, non 50%
    assert rate_for_budget(0.5, C0, C1) == pytest.approx((0.5 * (C0 + C1) - C0) / C1)
    assert rate_for_budget(0.5, C0, C1) < 0.5
    assert rate_for_budget(0.0, C0, C1) == 0.0 and rate_for_budget(1.0, C0, C1) == 1.0
    s = np.random.default_rng(1).random(500)
    th = threshold_for_budget(s, 0.5, C0, C1)
    assert C0 + (s >= th).mean() * C1 <= 0.5 * (C0 + C1) + 1e-9


def test_bootstrap_matches_loop_and_raises_no_warnings(w):
    """boot_curves e balanced_acc coincidono con il calcolo esplicito e non emettono avvisi numerici
    (con ``@`` numpy 2.0 + Accelerate su macOS ne emetteva di spuri)."""
    import warnings

    c = _curve(w, w["oracle"], min_takes=5)
    W = resample_weights(c.correct.shape[1], 8, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        acc_b, cost_b = boot_curves(c, W)
        bb = boot_bacc(c, W)
    for b in range(W.shape[0]):
        np.testing.assert_allclose(acc_b[b], (W[b][:, None] * c.correct.T).sum(0) / c.correct.shape[1])
        np.testing.assert_allclose(cost_b[b], (W[b][:, None] * c.cost.T).sum(0) / c.cost.shape[1])
        rec = [(W[b] * m * c.correct).sum(1) / (W[b] * m).sum() for m in c.member if (W[b] * m).sum() > 0]
        np.testing.assert_allclose(bb[b], np.mean(rec, axis=0))
