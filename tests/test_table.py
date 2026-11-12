import numpy as np
import pandas as pd
import pytest

from src.data.cache import Table, concat_tables, join_tables, read_table, write_table


def _t(ids, k=3, val=0.0):
    n = len(ids)
    return Table(
        pd.DataFrame({"clip_id": ids, "x": np.arange(n)}),
        {"v": np.full((n, k), val, dtype=np.float32) + np.arange(n)[:, None]},
        {"class_names": ["a"]},
    )


def test_roundtrip(tmp_path):
    t = _t(["b", "a", "c"])
    write_table(t, tmp_path / "t.parquet")
    r = read_table(tmp_path / "t.parquet")
    assert list(r.df.clip_id) == ["b", "a", "c"]
    np.testing.assert_array_equal(r.arrays["v"], t.arrays["v"])
    assert r.meta == {"class_names": ["a"]}


def test_duplicate_key_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_table(_t(["a", "a"]), tmp_path / "t.parquet")


def test_join_is_by_key_not_position():
    left = _t(["a", "b", "c"])
    right = Table(pd.DataFrame({"clip_id": ["c", "a", "b"], "y": [30, 10, 20]}), {"w": np.array([[3.0], [1.0], [2.0]])})
    j = join_tables(left, right, validate_full=True)
    assert list(j.df.y) == [10, 20, 30]
    np.testing.assert_array_equal(j.arrays["w"][:, 0], [1, 2, 3])
    assert j.df.y.dtype.kind == "i"


def test_join_incomplete_and_conflicts():
    left, right = _t(["a", "b"]), Table(pd.DataFrame({"clip_id": ["a"], "y": [1]}))
    assert len(join_tables(left, right)) == 1
    with pytest.raises(ValueError, match="incompleto"):
        join_tables(left, right, validate_full=True)
    conflict = Table(pd.DataFrame({"clip_id": ["a", "b"], "x": [5, 6]}))
    with pytest.raises(ValueError, match="valori diversi"):
        join_tables(left, conflict)


def test_concat():
    c = concat_tables([_t(["a"]), _t(["b", "c"])])
    assert len(c) == 3 and c.arrays["v"].shape == (3, 3)
