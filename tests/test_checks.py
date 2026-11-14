"""check_loss e ProvenanceError (src/cascade/cache/checks.py)."""

import logging

import pytest

from src.cascade.cache.checks import ProvenanceError, check_loss


def test_no_loss_returns_zero():
    assert check_loss(10, 10, 0.0, "x") == 0.0
    assert check_loss(0, 0, 0.0, "x") == 0.0  # niente di atteso: nessuna perdita


def test_small_loss_is_returned_and_warned(caplog):
    with caplog.at_level(logging.WARNING):
        lost = check_loss(100, 98, 0.05, "stadio")
    assert lost == pytest.approx(0.02)
    assert "perse 2 clip su 100" in caplog.text


def test_loss_above_threshold_raises():
    with pytest.raises(RuntimeError, match="max_missing_frac"):
        check_loss(72, 70, 0.02, "stadio")  # 2,8% > 2%: il caso dei test di stadio con SMALL


def test_threshold_is_inclusive():
    assert check_loss(100, 98, 0.02, "stadio") == pytest.approx(0.02)


def test_provenance_error_is_a_value_error():
    assert issubclass(ProvenanceError, ValueError)
