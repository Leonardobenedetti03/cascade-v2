"""M03: fold per sito. La mappa sito -> fold è un dato (split.json): si legge, non si ricalcola."""

import numpy as np
import pandas as pd
import pytest

from src.cascade.folds import make_folds

SITE2FOLD = {
    "uniandes": 0,
    "unc": 1,
    "sfu": 2,
    "cmu": 3,
    "iiith": 3,
    "minnesota": 3,
    "indiana": 4,
    "upenn": 4,
    "utokyo": 4,
}
SIZES = {
    "uniandes": 495,
    "unc": 244,
    "sfu": 130,
    "cmu": 40,
    "iiith": 42,
    "minnesota": 40,
    "indiana": 50,
    "upenn": 60,
    "utokyo": 37,
}


def _takes():
    rs = np.random.RandomState(0)
    rows = [(f"{s}_t{i}", int(rs.randint(0, 6)), s) for s, n in SIZES.items() for i in range(n)]
    return pd.DataFrame(rows, columns=["take_uid", "label_idx", "site"])


def _by_site(takes=None, **kw):
    return make_folds(_takes() if takes is None else takes, 5, 0.1, 0, group_by="site", site_to_fold=SITE2FOLD, **kw)


def test_expected_fold_sizes_and_one_site_per_fold():
    f = _by_site()
    assert len(f) == 1138 and f.groupby("fold").size().tolist() == [495, 244, 130, 122, 147]
    assert (f.groupby("site").fold.nunique() == 1).all()


def test_inner_holdout_never_in_the_held_out_fold():
    f = _by_site()
    for k in range(5):
        assert not (f[f"holdout_f{k}"] & (f.fold == k)).any()
        assert 0.07 < f.loc[f.fold != k, f"holdout_f{k}"].mean() < 0.13


def test_incomplete_or_inconsistent_map_is_an_error():
    with pytest.raises(ValueError, match="senza fold"):
        make_folds(_takes(), 5, 0.1, 0, group_by="site", site_to_fold={"uniandes": 0})
    no_fold4 = {s: min(k, 3) for s, k in SITE2FOLD.items()}
    with pytest.raises(ValueError, match="fold usate"):
        make_folds(_takes(), 5, 0.1, 0, group_by="site", site_to_fold=no_fold4)


def test_site_mode_needs_site_column_and_known_group_by():
    with pytest.raises(ValueError, match="colonna 'site'"):
        make_folds(_takes().drop(columns="site"), 5, 0.1, 0, group_by="site", site_to_fold=SITE2FOLD)
    with pytest.raises(ValueError, match="sconosciuto"):
        make_folds(_takes(), 5, 0.1, 0, group_by="random")


def test_take_mode_is_unchanged_by_the_site_column():
    with_site = make_folds(_takes(), 5, 0.1, 0, group_by="take")
    without = make_folds(_takes().drop(columns="site"), 5, 0.1, 0)
    assert with_site.groupby("fold").size().max() - with_site.groupby("fold").size().min() <= 1
    pd.testing.assert_frame_equal(with_site.drop(columns="site"), without)


def test_map_is_the_only_input_for_the_site_assignment():
    """Stessa mappa, take rimescolate e una take in meno: nessun sito cambia fold."""
    a = _by_site()
    t = _takes().sample(frac=1, random_state=1).iloc[1:]
    b = _by_site(t)
    assert a.set_index("take_uid").loc[b.take_uid, "fold"].tolist() == b.fold.tolist()
