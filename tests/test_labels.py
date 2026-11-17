import numpy as np
import pandas as pd
import pytest

from src import REPO_ROOT
from src.cascade.clips import load_scenario
from src.cascade.labels import assert_class_order, assert_num_classes, load_labels

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


@pytest.mark.skip(
    reason="N28: labels.yaml ha 6 classi, il CSV di CamFormer anche Music e Rock Climbing; "
    "le take ora vengono da scenario_preds (stadio 44), lo stadio 00 non si usa"
)
def test_real_scenario_csv_mapping_by_name():
    lab = load_labels()
    df = load_scenario(REAL_CSV, lab)
    raw = pd.read_csv(REAL_CSV)
    assert len(df) == len(raw)
    # la colonna `label` del CSV usa un altro ordine: la mappatura deve passare dal nome
    assert (df.label_idx.to_numpy() != raw.label.to_numpy()).any()
    for task, idx in zip(raw.task, df.label_idx):
        assert lab.csv_tasks[idx] == task
    counts = df.split.value_counts().to_dict()
    assert counts == {"train": 3049, "test": 1109, "val": 836}


def test_unknown_task_rejected(tmp_path):
    p = tmp_path / "s.csv"
    pd.DataFrame({"take_name": ["a"], "split": ["train"], "task": ["Juggling"], "label": [0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Juggling"):
        load_scenario(p, load_labels())
