"""Gate (C11, M06): feature, protocollo fuori campione e famiglie di gate.

La cache è finta e costruita qui (``make_table``): ``cache/mock.py`` arriva con C12.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.cascade.config import load_config
from src.cascade.gate import check_fit_split
from src.cascade.gate.features import FeatureBuilder, group_a, reliability, reliability_counts
from src.cascade.gate.models import CompareGate, LogRegGate, make_gate
from src.cascade.gate.protocol import TARGET_COLUMN, run_gate
from src.cascade.labels import load_labels
from src.cascade.motion import MOTION_FEATURES
from src.data.cache import Table

LAB = load_labels()
K = LAB.num_classes
N_FOLDS = 5


def make_table(
    n_train: int = 400,
    n_val: int = 150,
    n_test: int = 150,
    seed: int = 0,
    motion_effect: float = 2.5,
    source: str = "oof",
    c1_const=None,
) -> Table:
    """Cache finta: L0 sbaglia di più dove la confidenza è bassa; L1 ha ragione dove la
    velocità media (``mot_speed_mean``) è alta, indipendentemente da L0. Due finestre per take,
    fold per take sul train e -1 altrove."""
    rng = np.random.default_rng(seed)
    n = n_train + n_val + n_test
    split = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
    take = np.arange(n) // 2
    fold = np.where(split == "train", take % N_FOLDS, -1)
    label = rng.integers(0, K, n)
    difficulty = rng.random(n)
    logits = rng.normal(0.0, 1.0, (n, K))
    logits[np.arange(n), label] += 4.0 * (1.0 - difficulty) - 0.5
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    c0 = (probs.argmax(1) == label).astype(int)
    speed = rng.normal(size=n)
    c1 = (rng.random(n) < 1.0 / (1.0 + np.exp(-(motion_effect * speed + 1.0)))).astype(int)
    if c1_const is not None:
        c1 = np.full(n, c1_const)
    df = pd.DataFrame(
        {
            "clip_id": [f"clip{i:05d}" for i in range(n)],
            "take_uid": [f"take{t:04d}" for t in take],
            "split": split,
            "fold": fold,
            "label_idx": label,
            "y": (1 - c0) * c1,
            "c0": c0,
            "c1": c1,
        }
    )
    for j, name in enumerate(MOTION_FEATURES):
        df[name] = speed if j == 0 else rng.normal(size=n)
    arrays = {
        "cam_probs_cal": probs.astype(np.float32),
        "cam_emb": rng.normal(size=(n, 16)).astype(np.float32),
    }
    return Table(df, arrays, {"cam_train_source": source})


@pytest.fixture(scope="module")
def table():
    return make_table(seed=2)


def test_check_fit_split():
    check_fit_split("train", "oof")
    with pytest.raises(ValueError):
        check_fit_split("calib", "oof")
    with pytest.raises(ValueError, match="fuori campione"):
        check_fit_split("train", "in_sample")


def test_run_gate_refuses_in_sample():
    with pytest.raises(ValueError):
        run_gate(make_table(source="in_sample"), LAB, load_config().gate, "logreg_A")


def test_group_a():
    p = np.array([[0.7, 0.2, 0.1], [1 / 3, 1 / 3, 1 / 3]])
    a = group_a(p)
    np.testing.assert_allclose(a[:, 0], [0.7, 1 / 3])
    np.testing.assert_allclose(a[:, 1], [0.5, 0.0], atol=1e-12)
    assert a[1, 2] == pytest.approx(1.0) and a[0, 2] < 1


def test_reliability_excludes_folds():
    pred = np.array([0, 0, 0, 0])
    corr = np.array([1, 1, 0, 0])
    fold = np.array([0, 0, 1, 1])
    n_pred, n_corr = reliability_counts(pred, corr, fold, 2, 2)
    r = reliability(np.array([0, 0]), np.array([[True, False], [False, True]]), n_pred, n_corr, alpha=0.0)
    np.testing.assert_allclose(r, [0.0, 1.0])  # fold 0 vede solo fold 1 e viceversa


def _rel_feature(t, rows, held_out=None):
    return FeatureBuilder(t, LAB, ["B"]).build(rows, held_out=held_out)[:, 0]


def test_feature_b_no_leakage(table):
    """Cambiare le etichette della fold f non deve cambiare l'affidabilità delle clip di f."""
    t = table
    f = 2
    rows_f = np.flatnonzero(t.df.fold.to_numpy() == f)
    before = _rel_feature(t, rows_f)
    t2 = make_table(seed=2)  # stessa tabella, poi si cambiano le etichette di una fold
    t2.df.loc[rows_f, "label_idx"] = (t2.df.loc[rows_f, "label_idx"] + 1) % K
    np.testing.assert_array_equal(before, _rel_feature(t2, rows_f))
    # nella CV di secondo livello, le righe di training (fold g != h) non vedono la fold h
    h, g = 0, 3
    rows_g = np.flatnonzero(t.df.fold.to_numpy() == g)
    rows_h = np.flatnonzero(t.df.fold.to_numpy() == h)
    b = _rel_feature(t, rows_g, held_out=h)
    t3 = make_table(seed=2)
    t3.df.loc[rows_h, "label_idx"] = (t3.df.loc[rows_h, "label_idx"] + 1) % K
    np.testing.assert_array_equal(b, _rel_feature(t3, rows_g, held_out=h))
    # controllo positivo: senza esclusione la feature cambierebbe
    assert not np.array_equal(_rel_feature(t, rows_g), _rel_feature(t2, rows_g))


def test_feature_groups_shapes(table):
    fb = FeatureBuilder(table, LAB, ["A", "B", "C", "onehot", "emb"], emb_pca_dim=4)
    rows = np.arange(50)
    fb.fit_pca(rows)
    X = fb.build(rows)
    assert X.shape == (50, len(fb.names()))
    assert len(fb.names()) == 3 + 2 + 9 + K + 4


def test_removed_feature_groups_and_options():
    """M06: i gruppi mc e take e l'affidabilità Beta-Binomiale non si portano (N16)."""
    for g in ("mc", "take"):
        with pytest.raises(ValueError, match="sconosciuti"):
            FeatureBuilder(make_table(), LAB, [g])
    with pytest.raises(TypeError):
        FeatureBuilder(make_table(), LAB, ["B"], reliability_mode="beta")


@pytest.mark.parametrize(
    "name",
    [
        "random",
        "thr_maxprob",
        "thr_entropy",
        "thr_margin",
        "logreg_A",
        "logreg_AC",
        "logreg_ABC",
        "lgbm_ABC",
        "compare_A",
        "compare_AC",
        "oracle",
        "always_L0",
        "always_L1",
        "logreg_AB+onehot",
    ],
)
def test_all_gates_score_everything(table, name):
    res = run_gate(table, LAB, load_config().gate, name, seed=0)
    assert res.scores.shape == (len(table),) and np.isfinite(res.scores).all()


def test_removed_gates_are_unknown():
    gcfg = load_config().gate
    for name in ("ridge_ABC", "lgbmreg_ABC", "bayes_logreg_ABC"):
        with pytest.raises(ValueError, match="sconosciuto"):
            make_gate(name, gcfg)


def test_target_column_follows_gate_target():
    gcfg = load_config().gate
    assert TARGET_COLUMN == {"hard": "y", "c1": "c1"}
    assert make_gate("logreg_A", gcfg).target == "hard" and make_gate("lgbm_A", gcfg).target == "hard"
    g = make_gate("compare_AC", gcfg)
    assert isinstance(g, CompareGate) and g.target == "c1" and g.groups == ["A", "C"]
    assert g.class_weight is None  # servono probabilità, non un ranking bilanciato
    lr = make_gate("logreg_A", gcfg)
    assert isinstance(lr, LogRegGate) and lr.class_weight == "balanced"


def test_learned_gate_beats_random(table):
    y = table.df.y.to_numpy()
    val = (table.df.split == "val").to_numpy()
    s = run_gate(table, LAB, load_config().gate, "logreg_ABC").scores
    assert roc_auc_score(y[val], s[val]) > 0.65


def test_compare_uses_c1_and_trajectory_helps():
    """compare_AC contro compare_A isola l'effetto della traiettoria (gruppo C). Qui L1 ha
    ragione dove la velocità è alta: la traiettoria deve alzare l'AUROC su y."""
    t = make_table(n_train=600, n_val=300, n_test=100, seed=5)
    y = t.df.y.to_numpy()
    val = (t.df.split == "val").to_numpy()
    gcfg = load_config().gate
    auc = {n: roc_auc_score(y[val], run_gate(t, LAB, gcfg, n).scores[val]) for n in ("compare_A", "compare_AC")}
    assert auc["compare_AC"] > auc["compare_A"] + 0.03, auc


def test_compare_with_constant_c1_orders_like_threshold():
    """Se c1 ha una sola classe nei dati di fit, η̂₁ è costante e lo score ordina come thr_maxprob."""
    t = make_table(c1_const=1, seed=3)
    gcfg = load_config().gate
    s = run_gate(t, LAB, gcfg, "compare_A").scores
    p_max = t.arrays["cam_probs_cal"].max(1)
    np.testing.assert_allclose(s, 1.0 - p_max, atol=1e-6)
    thr = run_gate(t, LAB, gcfg, "thr_maxprob").scores
    np.testing.assert_array_equal(np.argsort(s, kind="stable"), np.argsort(thr, kind="stable"))


def test_lgbm_small_trees_enforced():
    gcfg = load_config(overrides=["gate.lgbm.num_leaves=63"]).gate
    with pytest.raises(ValueError):
        make_gate("lgbm_ABC", gcfg)


def test_default_models_and_comparisons_are_consistent():
    """Tutti i gate della config si costruiscono e i confronti di RQ2 usano solo gate della lista."""
    gcfg = load_config().gate
    for name in gcfg.models:
        make_gate(name, gcfg)
    models = set(gcfg.models)
    assert {"compare_A", "compare_AC", "thr_maxprob", "oracle"} <= models
    assert all(a in models and b in models for a, b in gcfg.comparisons)
    assert ["compare_AC", "compare_A"] in gcfg.comparisons and ["compare_A", "thr_maxprob"] in gcfg.comparisons
