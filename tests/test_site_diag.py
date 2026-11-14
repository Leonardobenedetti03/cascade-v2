"""Diagnostiche di sito (stadio 27, M02): funzioni pure su array costruiti a mano."""

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from src.cascade.config import load_config  # noqa: E402
from src.cascade.eval.site_diag import (  # noqa: E402
    ALL,
    LOSO_COLUMNS,
    loso_table,
    majority_summary,
    roles_from_takes,
    run_site_diagnostics,
    session_of,
)
from src.cascade.site_split import site_majority_accuracy  # noqa: E402

NAMES = ["A", "B", "C", "D"]
KW = dict(C=0.1, max_iter=500, cv_folds=5, min_class_takes=5)


def _world(n_sites=8, per_site=12, class_from_site=True, seed=0, extra_class_site0=0, test_clips=6):
    """Sito = vettore proprio + rumore; classe = parità del sito (``class_from_site``) o casuale.

    Tre take per sessione. L'ultimo sito è val; ``test_clips`` clip di un sito «tst» sono di test
    (non devono comparire da nessuna parte). ``extra_class_site0`` clip di classe 3 solo nel sito 0."""
    rng = np.random.default_rng(seed)
    rows, X = [], []
    dim = n_sites + 1
    for s in range(n_sites):
        for j in range(per_site):
            cls = (s % 2) if class_from_site else int(rng.integers(0, 2))
            vec = np.zeros(dim)
            vec[s] = 6.0
            X.append(vec + rng.standard_normal(dim))
            rows.append(
                {
                    "clip_id": f"c{s}_{j}",
                    "take_uid": f"site{s}_scene{j // 3}_{j % 3}",
                    "split": "val" if s == n_sites - 1 else "train",
                    "label_idx": cls,
                }
            )
    for j in range(extra_class_site0):
        vec = np.zeros(dim)
        vec[0] = 6.0
        X.append(vec + rng.standard_normal(dim))
        rows.append({"clip_id": f"x{j}", "take_uid": f"site0_extra_{j}", "split": "train", "label_idx": 3})
    for j in range(test_clips):
        X.append(rng.standard_normal(dim) + 3.0)
        rows.append({"clip_id": f"t{j}", "take_uid": f"tst_scene{j}_0", "split": "test", "label_idx": j % 2})
    feats = pd.DataFrame(rows)
    takes = feats[["take_uid", "split", "label_idx"]].copy()
    takes["label"] = [NAMES[i] for i in takes.label_idx]
    return feats, np.asarray(X), takes


def test_session_of():
    assert session_of("uniandes_dance_022_7") == "uniandes_dance_022"
    assert session_of("solo") == "solo"


def test_site_is_readable_and_class_is_a_site_shortcut():
    feats, X, takes = _world()
    res = run_site_diagnostics(feats, X, takes, NAMES, **KW)
    p = res["probe"]
    assert p["n_sites"] == 8 and "skipped" not in p
    sr = p["site_readability"]
    assert sr["by_session"]["acc"] > 0.9 and sr["chance_majority"] < 0.2  # il sito si legge
    # con i siti mescolati la classe si legge (scorciatoia di sito); su un sito mai visto no
    assert p["label_cv"]["by_take"]["acc"] > 0.9
    assert p["loso"]["acc"] < p["label_cv"]["by_take"]["acc"] - 0.2


def test_test_split_never_appears():
    feats, X, takes = _world()
    res = run_site_diagnostics(feats, X, takes, NAMES, **KW)
    loso, p, m = res["loso"], res["probe"], res["majority"]
    assert "tst" not in set(loso.site) and "test" not in " ".join(loso.role)
    assert p["n_clips"] == len(feats) - 6 and p["n_sites"] == 8
    assert "tst" not in {r["site"] for r in m["per_site"]} and "tst" not in m["crosstab"]
    assert all("test" not in r["role"] for r in m["per_site"])


def test_loso_scores_only_classes_seen_in_training():
    feats, X, takes = _world(extra_class_site0=4)
    loso = run_site_diagnostics(feats, X, takes, NAMES, **KW)["loso"]
    assert list(loso.columns) == LOSO_COLUMNS
    s0 = loso[(loso.site == "site0") & (loso["class"] == ALL)].iloc[0]
    assert s0.n_clips - s0.n_scored == 4  # le 4 clip di classe D, assente dagli altri siti
    d = loso[(loso.site == ALL) & (loso["class"] == "D")].iloc[0]
    assert d.n_clips == 4 and d.n_scored == 0 and np.isnan(d.acc)
    tot = loso[(loso.site == ALL) & (loso["class"] == ALL)].iloc[0]
    assert tot.n_correct <= tot.n_scored and tot.n_scored == tot.n_clips - 4


def test_reported_follows_min_class_takes():
    feats, X, takes = _world(extra_class_site0=4)
    loso = run_site_diagnostics(feats, X, takes, NAMES, **KW)["loso"]
    cls = loso[(loso.site == ALL) & (loso["class"] != ALL)].set_index("class")
    assert bool(cls.loc["A", "reported"]) and bool(cls.loc["B", "reported"])
    assert not bool(cls.loc["D", "reported"])  # 4 take < 5, come Bouldering in M14


def test_single_site_is_skipped():
    feats, X, takes = _world(n_sites=1, per_site=9, test_clips=0)
    res = run_site_diagnostics(feats, X, takes, NAMES, **KW)
    assert res["probe"]["n_sites"] == 1 and "skipped" in res["probe"]
    assert list(res["loso"].columns) == LOSO_COLUMNS and res["loso"].empty
    assert res["majority"]["n_sites"] == 1  # la baseline per sito c'è comunque


def test_convergence_warnings_are_counted():
    feats, X, takes = _world()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_site_diagnostics(feats, X, takes, NAMES, **{**KW, "max_iter": 1})
    f = res["probe"]["fits"]
    assert f["fits"] > 0 and f["convergence_warnings"] > 0


def test_majority_summary():
    roles = pd.DataFrame(
        {
            "take_uid": ["a_1", "a_2", "a_3", "b_1", "b_2", "v_1", "v_2", "t_1"],
            "label_idx": [0, 0, 1, 2, 2, 1, 3, 0],
            "role": ["train", "train", "train", "train", "train", "val", "val", "test"],
        }
    )
    roles["site"] = roles.take_uid.str.split("_").str[0]
    m = majority_summary(roles, NAMES)
    assert m["accuracy_train_insample"] == pytest.approx(4 / 5) == site_majority_accuracy(roles)
    by = {r["site"]: r for r in m["per_site"]}
    assert set(by) == {"a", "b", "v"}  # il sito di test non compare
    assert by["a"]["majority_class"] == "A" and by["a"]["majority_share"] == pytest.approx(2 / 3)
    assert by["v"]["majority_class"] == "B"  # pareggio: la classe con indice minore
    assert m["n_sites_single_class"] == 1 and m["crosstab"]["b"] == {"C": 2}
    assert m["n_train_takes"] == 5


def test_roles_from_takes():
    takes = pd.DataFrame({"take_uid": ["x_1", "y_2"], "split": ["train", "val"], "label_idx": [1, 2]})
    r = roles_from_takes(takes)
    assert list(r.columns) == ["take_uid", "label_idx", "role", "site"]
    assert list(r.site) == ["x", "y"] and list(r.role) == ["train", "val"]


def test_loso_table_shape():
    df = pd.DataFrame(
        {
            "take_uid": ["a_1", "a_2", "b_1", "b_2"],
            "site": ["a", "a", "b", "b"],
            "role": ["train", "train", "val", "val"],
            "label_idx": [0, 1, 0, 1],
            "pred": [0, 0, 0, 1],
            "scored": [True, True, True, True],
        }
    )
    t = loso_table(df, NAMES, 1)
    assert len(t) == 2 * 3 + 3  # per sito: ALL + 2 classi; poi totale ALL + 2 classi
    tot = t[(t.site == ALL) & (t["class"] == ALL)].iloc[0]
    assert tot.n_correct == 3 and tot.acc == pytest.approx(0.75)


def test_site_diag_config_defaults_and_validation():
    cfg = load_config()
    assert cfg.site_diag.C == 0.1 and cfg.site_diag.max_iter == 500 and cfg.site_diag.cv_folds == 5
    with pytest.raises(ValueError, match="site_diag"):
        load_config(overrides=["site_diag.cv_folds=1"])
