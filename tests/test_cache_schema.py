import numpy as np
import pandas as pd
import pytest

from src.cascade.cache.mock import make_mock_cache
from src.cascade.cache.schema import SchemaError, add_labels, validate_cache
from src.cascade.labels import load_labels
from src.data.cache import Table, read_table, write_table

LAB = load_labels()


@pytest.fixture(scope="module")
def mock():
    return make_mock_cache(LAB, 600, 150, 200, seed=1, cam_dim=32, vid_dim=48)


def test_mock_is_valid_and_roundtrips(mock, tmp_path):
    assert validate_cache(mock, LAB, cam_dim=32, vid_dim=48) == []
    write_table(mock, tmp_path / "c.parquet")
    back = read_table(tmp_path / "c.parquet")
    validate_cache(back, LAB)
    assert list(back.df.clip_id) == list(mock.df.clip_id)


def test_mock_properties():
    t = make_mock_cache(LAB, 3000, 100, 100, seed=0, cam_dim=8, vid_dim=8)
    df = t.df
    proc = LAB.procedural_mask[df.label_idx]
    assert df.c0[~proc].mean() > 0.88
    assert 0.5 < df.c0[proc].mean() < 0.75
    vid_acc = (t.arrays["vid_probs_cal"].argmax(1) == df.label_idx).mean()
    assert 0.92 < vid_acc < 0.98
    wrong_proc = proc & (df.c0 == 0)
    pred = t.arrays["cam_probs_cal"].argmax(1)
    assert LAB.procedural_mask[pred[wrong_proc]].mean() > 0.8  # confusione tra procedurali
    # calibrazione imperfetta: confidenza media != accuratezza
    assert abs(t.arrays["cam_probs_cal"].max(1).mean() - df.c0.mean()) > 0.01
    # moto correlato con la classe
    means = df.groupby("label_idx").mot_energy.mean()
    assert means.max() / means.min() > 1.5


def test_mock_fusion_is_replace(mock):
    """M05: con la fusione replace w = 0 e c1 è la correttezza di L1."""
    assert (mock.df.w == 0.0).all()
    l1_correct = (mock.arrays["vid_probs_cal"].argmax(1) == mock.df.label_idx.to_numpy()).astype(int)
    assert np.array_equal(mock.df.c1.to_numpy(), l1_correct)
    np.testing.assert_allclose(mock.arrays["fused_probs"], mock.arrays["vid_probs_cal"], atol=1e-5)


def test_label_y():
    """c0, c1 e y = (1 − c0)·c1 su quattro casi: (1,1), (0,1), (1,0), (0,0)."""
    cam_pred, fused_pred = [0, 0, 2, 0], [0, 1, 0, 0]
    t = Table(
        pd.DataFrame({"clip_id": list("abcd"), "label_idx": [0, 1, 2, 3]}),
        {"cam_probs_cal": np.eye(4)[cam_pred], "fused_probs": np.eye(4)[fused_pred]},
    )
    add_labels(t)
    assert t.df.c0.tolist() == [1, 0, 1, 0]
    assert t.df.c1.tolist() == [1, 1, 0, 0]
    assert t.df.y.tolist() == [0, 1, 0, 0]  # y = 1 solo dove L0 sbaglia e L1 riesce


@pytest.mark.parametrize(
    "breaker,msg",
    [
        (lambda t: t.df.__setitem__("c0", 1 - t.df.c0), "c0"),
        (lambda t: t.df.__setitem__("y", 1 - t.df.y), "y"),
        (lambda t: t.df.loc.__setitem__((t.df.split == "val", "fold"), 0), "fold"),
        (lambda t: t.arrays["cam_probs_cal"].__setitem__((0, 0), np.nan), "NaN"),
        (lambda t: t.df.__setitem__("w", np.nan), "NaN in w"),
        (lambda t: t.meta.__setitem__("class_names", list(LAB.names[::-1])), "class_names"),
        (lambda t: t.meta.__setitem__("cam_train_source", "in_sample"), "oof"),
        (lambda t: t.df.drop(columns=["gmacs_L1"], inplace=True), "gmacs_L1"),
    ],
)
def test_schema_rejects(mock, breaker, msg):
    t = mock.take(np.arange(len(mock)))
    t.arrays = {k: v.copy() for k, v in t.arrays.items()}
    t.meta = dict(mock.meta)
    breaker(t)
    with pytest.raises(SchemaError, match=msg):
        validate_cache(t, LAB)


def test_motion_nan_is_warning(mock):
    t = mock.take(np.arange(len(mock)))
    t.df.loc[0, "mot_energy"] = np.nan
    assert any("mot_energy" in w for w in validate_cache(t, LAB))
