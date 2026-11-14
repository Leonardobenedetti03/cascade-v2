"""Controlli di provenienza e di perdita di clip, condivisi dagli stadi 25, 28, 30 e dall'assemblaggio (stadio 45)."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class ProvenanceError(ValueError):
    """Le predizioni sul train non sono (verificabilmente) fuori campione."""


def check_loss(n_expected: int, n_found: int, max_frac: float, what: str) -> float:
    """Frazione di clip perse; errore se supera ``data.max_missing_frac`` (fail fast)."""
    lost = 1.0 - (n_found / n_expected) if n_expected else 0.0
    if lost > max_frac + 1e-12:
        raise RuntimeError(
            f"{what}: perse {n_expected - n_found} clip su {n_expected} ({lost:.1%} > "
            f"data.max_missing_frac={max_frac:.1%}); controlla i log degli stadi a monte"
        )
    if lost > 0:
        log.warning("%s: perse %d clip su %d (%.2f%%)", what, n_expected - n_found, n_expected, 100 * lost)
    return lost
