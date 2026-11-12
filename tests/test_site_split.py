"""Test di src/cascade/site_split.py.

La fixture riproduce la matrice sito x classe reale del subset (1273 take, train+val ufficiali
insieme), così i test verificano le scelte di split sul caso vero.
"""

import numpy as np
import pandas as pd
import pytest

from src.cascade.folds import make_folds, training_masks
from src.cascade.site_split import (
    check_coverage,
    check_split_hash,
    file_sha256,
    fold_diagnostics,
    plan_roles,
    plan_site_to_fold,
    roles_from_scenario,
    site_majority_accuracy,
    site_of,
)

NAMES = ["Basketball", "Dance", "Music", "Soccer", "Cooking", "Bouldering", "Health", "Bike repair"]
IDX = {"Basketball": 0, "Dance": 1, "Soccer": 3, "Cooking": 4, "Bouldering": 5, "Health": 6, "Bike": 7}

# take per (sito, classe), train+val del subset
MATRIX = {
    "cmu": {"Bike": 10},
    "fair": {"Bike": 21, "Cooking": 3},
    "georgiatech": {"Bike": 39, "Cooking": 7, "Health": 21},
    "iiith": {"Cooking": 49, "Soccer": 62},
    "indiana": {"Bike": 11, "Cooking": 9},
    "minnesota": {"Cooking": 1},
    "nus": {"Cooking": 8, "Health": 36},
    "sfu": {"Basketball": 102, "Cooking": 18, "Health": 10},
    "unc": {"Basketball": 244},
    "uniandes": {"Basketball": 122, "Bouldering": 2, "Cooking": 9, "Dance": 362},
    "upenn": {"Cooking": 18, "Dance": 90},
    "utokyo": {"Cooking": 1, "Health": 10, "Soccer": 8},
}


@pytest.fixture(scope="module")
def takes():
    rows = []
    for site, classes in MATRIX.items():
        for cls, n in classes.items():
            rows += [{"take_uid": f"{site}_{cls.lower()}_{i:03d}", "label_idx": IDX[cls]} for i in range(n)]
    df = pd.DataFrame(rows)
    assert len(df) == 1273
    return df


def test_site_of():
    assert site_of("uniandes_dance_022_7") == "uniandes"
    assert site_of("upenn_0706_Dance_5_6") == "upenn"
    with pytest.raises(ValueError):
        site_of("_senza_sito")


def test_plan_roles_and_errors(takes):
    r = plan_roles(takes, ["sfu"], ["georgiatech"])
    assert set(r.role) == {"train", "val", "test"}
    assert set(r[r.role == "val"].site) == {"sfu"} and set(r[r.role == "test"].site) == {"georgiatech"}
    assert len(r[r.role == "test"]) == 67 and len(r[r.role == "val"]) == 130
    with pytest.raises(ValueError, match="sia in val sia in test"):
        plan_roles(takes, ["sfu"], ["sfu"])
    with pytest.raises(ValueError, match="non presenti"):
        plan_roles(takes, ["atlantide"], ["sfu"])


def test_coverage_accepts_chosen_split(takes):
    problems, cov = check_coverage(plan_roles(takes, ["sfu"], ["georgiatech"]), NAMES)
    assert problems == []
    assert cov.set_index("classe").loc["Dance", "train_siti"] == 2


def test_coverage_rejects_first_proposal(takes):
    """Test su upenn, nus, utokyo, indiana, sfu: Dance, Health e Soccer restano con un solo sito di train."""
    r = plan_roles(takes, [], ["upenn", "nus", "utokyo", "indiana", "sfu"])
    problems, _ = check_coverage(r, NAMES)
    flagged = {p.split(":")[0] for p in problems}
    # Soccer ha 8 take nel test (= soglia di valutazione) e un solo sito di train: segnalata anche lei
    assert {"Dance", "Health", "Soccer"} <= flagged


def test_assign_site_folds_properties():
    sizes = {"a": 100, "b": 60, "c": 40, "d": 30, "e": 20, "f": 10}
    classes = {"a": {0}, "b": {0, 1}, "c": {1}, "d": {1, 2}, "e": {2}, "f": {0}}
    f1 = plan_site_to_fold(sizes, classes, 3)
    assert f1 == plan_site_to_fold(sizes, classes, 3)  # deterministico
    assert set(f1) == set(sizes) and set(f1.values()) == {0, 1, 2}
    # nessuna fold contiene tutti i siti di una classe presente in 2+ siti
    for c in (0, 1, 2):
        sites_c = {s for s, cs in classes.items() if c in cs}
        for k in range(3):
            assert not sites_c <= {s for s, v in f1.items() if v == k}


def test_assign_site_folds_errors():
    with pytest.raises(ValueError, match="almeno tanti siti"):
        plan_site_to_fold({"a": 1, "b": 1}, {"a": {0}, "b": {0}}, 5)
    with pytest.raises(ValueError):
        plan_site_to_fold({"a": 1, "b": 1}, {"a": {0}, "b": {0}}, 1)


def _site_folds(train, seed=0):
    sizes = train.site.value_counts().to_dict()
    classes = {s: set(g) for s, g in train.label_idx.groupby(train.site)}
    s2f = plan_site_to_fold(sizes, classes, 5)
    return make_folds(train, 5, 0.1, seed=seed, group_by="site", site_to_fold=s2f)


def test_site_folds_on_real_matrix(takes):
    train = plan_roles(takes, ["sfu"], ["georgiatech"]).query("role == 'train'")
    assert len(train) == 1076
    folds = _site_folds(train)
    ref = make_folds(train, 5, 0.1, seed=0)  # per take: stesse colonne
    assert list(folds.columns) == list(ref.columns)
    assert len(folds) == len(train) and not folds.take_uid.duplicated().any()
    # un sito sta in una sola fold
    site = folds.take_uid.map(site_of)
    assert (folds.groupby(site).fold.nunique() == 1).all()
    assert set(folds.fold) == {0, 1, 2, 3, 4}
    for k in range(5):
        train_m, hold_m, eval_m = training_masks(folds, str(k))
        assert not (train_m & hold_m).any() and not (train_m & eval_m).any() and not (hold_m & eval_m).any()
        # i siti fuori campione non compaiono mai in training ne' nell'holdout
        assert not set(site[eval_m]) & set(site[train_m | hold_m])
        assert not folds[f"holdout_f{k}"][eval_m].any()  # l'holdout viene dalle fold di training
    # le classi con piu' di un sito sono viste da ogni modello di fold, tranne Bouldering (1 sito, 2 take)
    diag = fold_diagnostics(folds, NAMES)
    unseen = set(",".join(diag.classi_senza_train).replace("-", "").split(",")) - {""}
    assert unseen <= {"Bouldering"}


def test_site_folds_deterministic(takes):
    train = plan_roles(takes, ["sfu"], ["georgiatech"]).query("role == 'train'")
    a = _site_folds(train)
    b = _site_folds(train.sample(frac=1, random_state=3))  # ordine di input diverso
    pd.testing.assert_frame_equal(a, b)


BLUEPRINT_MAP = {
    "uniandes": 0, "unc": 1, "sfu": 2, "cmu": 3, "iiith": 3, "minnesota": 3, "indiana": 4, "upenn": 4, "utokyo": 4,
}  # fmt: skip


def test_algorithm_reproduces_the_versioned_site_map(takes):
    """site_v1 (val fair + nus, test georgiatech): la mappa di D6 e le dimensioni delle fold del blueprint."""
    train = plan_roles(takes, ["fair", "nus"], ["georgiatech"]).query("role == 'train'")
    assert len(train) == 1138
    sizes = train.site.value_counts().to_dict()
    classes = {s: set(g) for s, g in train.label_idx.groupby(train.site)}
    assert plan_site_to_fold(sizes, classes, 5) == BLUEPRINT_MAP
    folds = _site_folds(train)
    assert folds.groupby("fold").size().tolist() == [495, 244, 130, 122, 147]


def test_roles_from_scenario_csv(takes):
    r = plan_roles(takes, ["fair", "nus"], ["georgiatech"])
    df = pd.DataFrame({"take_name": r.take_uid, "split": r.role, "task": "x", "label": r.label_idx})
    back = roles_from_scenario(df)
    assert (back.role.to_numpy() == r.role.to_numpy()).all() and (back.site.to_numpy() == r.site.to_numpy()).all()
    assert (back.label_idx.to_numpy() == r.label_idx.to_numpy()).all()
    with pytest.raises(ValueError, match="non ammessi"):
        roles_from_scenario(df.assign(split="boh"))
    with pytest.raises(ValueError, match="colonne attese"):
        roles_from_scenario(df.drop(columns="label"))


def test_split_hash(tmp_path):
    f = tmp_path / "scenario_site.csv"
    f.write_text("take_name,split\na_1,train\n")
    h = file_sha256(f)
    assert len(h) == 64 and check_split_hash(f, h) == h and check_split_hash(f, None) == h
    with pytest.raises(ValueError, match="modificato"):
        check_split_hash(f, "0" * 64)


def test_site_majority_accuracy(takes):
    r = plan_roles(takes, [], [])
    acc = site_majority_accuracy(r)
    assert 0.7 < acc < 0.85  # scenario e laboratorio quasi sovrapposti
    assert np.isclose(site_majority_accuracy(r.assign(role="test")), 0.0)  # niente train: nessun hit
