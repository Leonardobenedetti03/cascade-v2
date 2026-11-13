import numpy as np

from src.cascade.calibration import apply_temperature, ece, fit_temperature, nll_from_logits


def _miscalibrated(n=3000, C=8, seed=0, overconf=3.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, C, n)
    z = rng.standard_normal((n, C))
    z[np.arange(n), y] += 1.5  # logit "veri" (T = 1)
    return z * overconf, y  # sovra-confidenti


def test_temperature_reduces_nll():
    logits, y = _miscalibrated()
    T = fit_temperature(logits, y)
    assert T > 1.5  # sovra-confidenti → T > 1
    assert nll_from_logits(logits, y, T) < nll_from_logits(logits, y) - 0.1
    assert ece(apply_temperature(logits, T), y) < ece(apply_temperature(logits, 1.0), y)


def test_temperature_keeps_argmax():
    logits, _ = _miscalibrated(n=100)
    np.testing.assert_array_equal(apply_temperature(logits, 2.5).argmax(1), logits.argmax(1))


def test_underconfident_gets_T_below_one():
    logits, y = _miscalibrated(overconf=0.4)
    assert fit_temperature(logits, y) < 1.0
