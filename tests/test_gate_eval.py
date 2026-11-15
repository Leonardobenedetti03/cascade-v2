"""Valutazione dei gate (C12b, M09 e M14) sulla cache finta: nessun test del set di test fuori da
``final``, accuratezza bilanciata, I(δ), trasferimento delle soglie, diagnostiche D1 e celle."""

import numpy as np
import pandas as pd
import pytest

from src.cascade.cache.mock import make_mock_cache
from src.cascade.config import load_config
from src.cascade.eval.gate_eval import (
    OUTPUT_TABLES,
    SplitData,
    cells_rows,
    diag_c1_rows,
    evaluate,
    needed_columns,
    transfer_for_scores,
)
from src.cascade.gate.protocol import GateResult
from src.cascade.labels import load_labels

MODELS = "[thr_maxprob,logreg_AC,compare_A,compare_AC]"
OVERRIDES = [
    f"gate.models={MODELS}",
    "gate.bootstrap=25",
    "gate.random_seeds=2",
    "gate.seeds=[0]",
    "gate.comparisons=[[compare_A,thr_maxprob],[compare_AC,compare_A]]",
]
NON_FIXED = ["thr_maxprob", "logreg_AC", "compare_A", "compare_AC", "random", "oracle"]


@pytest.fixture(scope="module")
def labels():
    return load_labels()


@pytest.fixture(scope="module")
def cfg():
    return load_config(overrides=OVERRIDES)


@pytest.fixture(scope="module")
def table(labels):
    return make_mock_cache(labels, 600, 200, 200, 5, 3, cam_dim=8, vid_dim=8)


@pytest.fixture(scope="module")
def res_nofinal(table, labels, cfg):
    return evaluate(table, labels, cfg, final=False)


@pytest.fixture(scope="module")
def res_final(table, labels, cfg):
    return evaluate(table, labels, cfg, final=True)


def test_without_final_nothing_from_test(res_nofinal):
    for key in OUTPUT_TABLES:
        d = res_nofinal[key]
        if "split" in d:
            assert "test" not in set(d["split"]), key
    assert "test" not in set(res_nofinal["scores"].split)
    assert len(res_nofinal["threshold_transfer"]) == 0  # serve il test


def test_final_has_test_rows(res_final):
    assert {"train", "val", "test"} <= set(res_final["metrics"].split)
    assert "test" in set(res_final["scores"].split)
    assert len(res_final["threshold_transfer"]) > 0


def test_metrics_have_balanced_and_improvement(res_final, table):
    m = res_final["metrics"]
    names = set(m.metric)
    assert {"area", "bacc_area", "bacc_oracle_gain_frac", "improvement_d0", "improvement_d1", "bacc_at_50"} <= names
    floor = float(table.df.gmacs_L1.iloc[0] / (table.df.gmacs_L0.iloc[0] + table.df.gmacs_L1.iloc[0]))
    imp = m[m.metric.isin(["improvement_d0", "improvement_d1"]) & m.model.isin(NON_FIXED)]
    assert len(imp) and (imp.value >= floor - 1e-9).all()
    d0 = m[m.metric == "improvement_d0"].set_index(["split", "model"]).value
    d1 = m[m.metric == "improvement_d1"].set_index(["split", "model"]).value
    assert (d1 >= d0 - 1e-9).all()  # più tolleranza, più risparmio


def test_oracle_beats_everything_on_both_series(res_final):
    m = res_final["metrics"]
    for split in ("val", "test"):
        for key in ("area", "bacc_area"):
            v = m[(m.split == split) & (m.metric == key)].set_index("model").value
            assert v["oracle"] >= v.drop("oracle").max() - 1e-9, (split, key)


def test_operating_point_respects_budget_on_val(res_final):
    op = res_final["operating_point"]
    v = op[(op.split == "val") & (op.point == "budget") & op.model.isin(NON_FIXED)]
    assert len(v) and (v.cost_frac_L1 <= 0.5 + 1e-6).all()
    nat = op[(op.point == "s>0")]
    assert set(nat.model) == {"compare_A", "compare_AC"}  # solo i gate compare hanno una soglia naturale


def test_comparisons_include_balanced_metrics(res_final):
    c = res_final["comparisons"]
    assert {"area", "bacc_area", "bacc_at_25"} <= set(c.metric)
    assert c.p_two_sided.between(0, 1).all()


def test_transfer_rows_skip_fixed_and_oracle(res_final):
    t = res_final["threshold_transfer"]
    assert set(t.split) == {"test"}
    assert set(t.model) == {"thr_maxprob", "logreg_AC", "compare_A", "compare_AC", "random"}
    assert set(t.budget_frac) == {0.25, 0.5}
    assert np.isfinite(t.cost_over_budget).all() and (t.cost_over_budget > -1).all()


def test_transfer_on_identical_scores_has_no_gap(table):
    """Se val e test coincidono, la soglia trasferita è quella ricalcolata: nessun divario, nessuno sforamento."""
    sd = SplitData.build(table, "val", 5)
    s = np.random.default_rng(0).random(len(sd.rows))
    out = transfer_for_scores(s, s, sd, sd, 0.5)
    assert out["theta_val"] == out["theta_test"]
    assert out["acc_gap"] == 0 and out["bacc_gap"] == 0
    assert out["cost_over_budget"] <= 1e-9  # mai sopra il budget a parità di punteggi
    assert out["cost_over_budget"] == out["cost_over_budget_reopt"]


def test_diag_c1_only_train_and_val(res_final, table, cfg):
    d = res_final["diag_c1"]
    assert set(d.split) == {"train", "val"} and set(d.model) == {"compare_A", "compare_AC"}
    assert d.auroc.between(0, 1).all()
    sd = SplitData.build(table, "test", 5)
    with pytest.raises(ValueError, match="non ammesso"):
        diag_c1_rows(
            {"compare_A": [GateResult("compare_A", 0, np.zeros(len(table)))]}, sd, cfg.gate, np.zeros(len(sd.rows))
        )


def test_cells_partition_the_mass(res_final, table):
    cells = res_final["cells"]
    tot = cells[cells["class"] == "all"].set_index("split")
    np.testing.assert_allclose(tot[["p_A", "p_B", "p_D", "p_N"]].sum(axis=1), 1.0)
    for split in ("train", "val", "test"):
        df = table.df[table.df.split == split]
        assert tot.loc[split, "p_A"] + tot.loc[split, "p_B"] == pytest.approx(df.c0.mean())
        assert tot.loc[split, "p_A"] + tot.loc[split, "p_D"] == pytest.approx(df.c1.mean())
        assert tot.loc[split, "p_D"] == pytest.approx(df.y.mean())  # y = (1 − c0)·c1 è la cella D
    assert cells_rows(table, load_labels(), "inesistente") == []


def test_positives_exclude_test(res_final):
    assert "test" not in set(res_final["positives"].split)


def test_bacc_undefined_is_nan_not_error(table, labels):
    cfg = load_config(overrides=OVERRIDES + ["gate.bacc_min_takes=100000", "gate.bootstrap=10"])
    res = evaluate(table, labels, cfg, final=False)
    m = res["metrics"]
    assert m[m.metric == "bacc_area"].value.isna().all()
    assert m[m.metric == "area"].value.notna().all()
    assert (m.n_classes == 0).all()


def test_needed_columns_drop_heavy_vectors(cfg):
    cols = ["clip_id", "cam_emb", "vid_cls", "vid_emb", "cam_probs_cal"]
    assert needed_columns(cols, cfg.gate) == ["clip_id", "cam_probs_cal"]
    emb = load_config(overrides=OVERRIDES + ["gate.models=[logreg_A+emb]"])
    assert "cam_emb" in needed_columns(cols, emb.gate)


def test_scores_frame_is_aligned_with_table(res_final, table):
    s = res_final["scores"]
    assert len(s) == len(table) and (s.clip_id.to_numpy() == table.df.clip_id.to_numpy()).all()
    assert pd.api.types.is_numeric_dtype(s["score__oracle__s0"])


def test_lhalf_is_evaluated_when_in_the_cache(labels, cfg):
    """I4: con la baseline a un frame nella cache, lo stadio 50 riporta anche L½ (riga L½ di T1)."""
    t = make_mock_cache(labels, 200, 80, 80, 5, 3, cam_dim=8, vid_dim=8)
    base = pd.DataFrame(evaluate(t, labels, cfg, final=True)["per_class"])
    assert "Lhalf" not in set(base.system)
    t.arrays["half_probs_cal"] = t.arrays["vid_probs_cal"].copy()
    pc = pd.DataFrame(evaluate(t, labels, cfg, final=True)["per_class"])
    assert {"Lhalf", "L1"} <= set(pc.system) and "test" in set(pc[pc.system == "Lhalf"].split)
    a = pc[pc.system == "Lhalf"].set_index(["split", "class"]).acc
    b = pc[pc.system == "L1"].set_index(["split", "class"]).acc
    pd.testing.assert_series_equal(a, b, check_names=False)  # stesse probabilità, stesse accuratezze
