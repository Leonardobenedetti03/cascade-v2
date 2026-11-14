"""Palette condivisa (src/cascade/eval/style.py)."""

import re

from src.cascade.eval import style


def test_palette_values_are_hex_colors():
    colors = [style.SURFACE, style.TEXT2, style.GRID, *style.SERIES]
    assert all(re.fullmatch(r"#[0-9a-f]{6}", c) for c in colors)
    assert len(style.SERIES) == 4
