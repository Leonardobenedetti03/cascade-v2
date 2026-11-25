import numpy as np
import pytest

from src import REPO_ROOT
from src.cascade.data.labels import assert_class_order, assert_num_classes, load_labels

REAL_CSV = REPO_ROOT / "third_party/seeing_without_pixels/data_files/egoexo4d/scenario_cls.csv"


def test_labels_order_and_kinds():
    lab = load_labels()
    assert lab.num_classes == 6  # N28: Music e Bouldering fuori
    assert lab.names == ("Basketball", "Dance", "Soccer", "Cooking", "Health", "Bike repair")
    procedural = {n for n, p in zip(lab.names, lab.procedural_mask) if p}
    assert procedural == {"Cooking", "Health", "Bike repair"}


def test_assert_class_order():
    lab = load_labels()
    assert_class_order(list(lab.names), lab)
    with pytest.raises(ValueError):
        assert_class_order(list(lab.names[::-1]), lab)
    with pytest.raises(ValueError):
        assert_num_classes(np.zeros((3, 7)), lab, where="x")
