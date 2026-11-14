import math

import numpy as np
import pytest

from src.cascade.cost import cascade_gmacs, macs_to_gmacs, max_escalations, relative_cost


def test_macs_to_gmacs():
    assert macs_to_gmacs(0) == 0.0
    assert macs_to_gmacs(2.5e9) == pytest.approx(2.5)
    with pytest.raises(ValueError):
        macs_to_gmacs(-1)


def test_cascade_gmacs_endpoints():
    n = 8
    assert cascade_gmacs(np.zeros(n, bool), 1.0, 100.0, gate_gmacs=0.01) == pytest.approx(1.01)
    assert cascade_gmacs(np.ones(n, bool), 1.0, 100.0, gate_gmacs=0.01) == pytest.approx(101.01)


def test_cascade_gmacs_fraction():
    esc = [True, False, False, True]  # 50%
    assert cascade_gmacs(esc, 2.0, 10.0) == pytest.approx(2.0 + 0.5 * 10.0)
    assert cascade_gmacs(np.array([1, 0, 0, 0]), 2.0, 10.0) == pytest.approx(4.5)


@pytest.mark.parametrize("bad", [[], [[True]], [0, 2]])
def test_cascade_gmacs_invalid_mask(bad):
    with pytest.raises(ValueError):
        cascade_gmacs(bad, 1.0, 1.0)


def test_cascade_gmacs_negative_cost():
    with pytest.raises(ValueError):
        cascade_gmacs([True], -1.0, 1.0)
    with pytest.raises(ValueError):
        cascade_gmacs([True], 1.0, math.nan)


def test_max_escalations_basic():
    # base 1, stadio 2 costa 10: budget 4 -> 3 GMACs extra medi -> 30% di 10 campioni.
    assert max_escalations(10, 4.0, 1.0, 10.0) == 3
    assert max_escalations(10, 4.9, 1.0, 10.0) == 3
    assert max_escalations(10, 1.0, 1.0, 10.0) == 0
    assert max_escalations(10, 1000.0, 1.0, 10.0) == 10


def test_max_escalations_float_boundary():
    # 10 * (1.3 - 1.0) / 1.0 = 2.9999999999999996 in float: deve dare 3.
    assert max_escalations(10, 1.3, 1.0, 1.0) == 3


def test_max_escalations_respects_budget():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 50))
        c1, c2, g = rng.uniform(0, 5), rng.uniform(0.1, 50), rng.uniform(0, 0.5)
        budget = rng.uniform(c1 + g, c1 + g + c2)
        k = max_escalations(n, budget, c1, c2, g)
        esc = np.zeros(n, bool)
        esc[:k] = True
        assert cascade_gmacs(esc, c1, c2, g) <= budget + 1e-9
        if k < n:
            esc[k] = True
            assert cascade_gmacs(esc, c1, c2, g) > budget


def test_max_escalations_budget_below_stage1():
    with pytest.raises(ValueError):
        max_escalations(10, 0.5, 1.0, 10.0)
    with pytest.raises(ValueError):
        max_escalations(10, 1.0, 1.0, 10.0, gate_gmacs=0.1)


def test_max_escalations_free_stage2():
    assert max_escalations(7, 1.0, 1.0, 0.0) == 7


def test_relative_cost():
    assert relative_cost(25.0, 100.0) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        relative_cost(1.0, 0.0)
