import json

import numpy as np
import pandas as pd
import pytest

from src.cascade.cache.build import assemble_camformer, join_l0_l1, load_l1
from src.cascade.cache.checks import ProvenanceError
from src.cascade.config import load_config
from src.cascade.labels import load_labels
from src.data.cache import Table, write_table

LAB = load_labels()


def _preds(ids, split, val, fold, take=None, source=None):
    n = len(ids)
    p = np.full((n, 8), 0.02)
    p[:, int(val)] = 0.86
    df = pd.DataFrame(
        {
            "clip_id": ids,
            "take_uid": take or ids,
            "split": split,
            "label_idx": 0,
            "T_cam": 1.0 + fold,
            "cam_fold": str(fold),
        }
    )
    src = source or ("oof" if split == "train" else "eval")
    return Table(
        df,
        {"cam_logits": np.log(p), "cam_probs_cal": p, "cam_emb": np.full((n, 4), float(fold))},
        {"class_names": list(LAB.names), "source": src, "fold": str(fold)},
    )


def _write_run(tmp_path, K=2, with_full=False):
    base = tmp_path / "20_camformer"
    for k in range(K):
        d = base / f"fold{k}"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({"fold": k}))
        write_table(_preds([f"tr{k}"], "train", k, k), d / "preds_oof.parquet")
        write_table(_preds(["v1", "v0"], "val", k, k), d / "preds_val.parquet")
        write_table(_preds(["t0"], "test", k, k), d / "preds_test.parquet")
    if with_full:
        d = base / "full"
        d.mkdir()
        write_table(_preds(["v0", "v1"], "val", 3, 9), d / "preds_val.parquet")
        write_table(_preds(["t0"], "test", 3, 9), d / "preds_test.parquet")
    return tmp_path


def test_fold_ensemble_average_and_oof(tmp_path):
    cfg = load_config(overrides=["folds.n_folds=2"])
    t = assemble_camformer(_write_run(tmp_path) / "20_camformer", cfg, LAB)
    t = t.sort_by()
    assert set(t.keys()) == {"tr0", "tr1", "v0", "v1", "t0"}
    assert t.meta["cam_train_source"] == "oof"
    v0 = list(t.keys()).index("v0")
    np.testing.assert_allclose(t.arrays["cam_probs_cal"][v0, :2], [(0.86 + 0.02) / 2] * 2)
    np.testing.assert_allclose(t.arrays["cam_emb"][v0], 0.0)  # niente full: embedding della fold 0
    tr1 = list(t.keys()).index("tr1")
    assert t.arrays["cam_probs_cal"][tr1].argmax() == 1  # OOF dal proprio modello


def test_full_embeddings_and_missing_fold(tmp_path):
    cfg = load_config(overrides=["folds.n_folds=2"])
    t = assemble_camformer(_write_run(tmp_path, with_full=True) / "20_camformer", cfg, LAB).sort_by()
    np.testing.assert_allclose(t.arrays["cam_emb"][list(t.keys()).index("v1")], 9.0)
    with pytest.raises(FileNotFoundError, match="fold"):
        assemble_camformer(tmp_path / "20_camformer", load_config(overrides=["folds.n_folds=3"]), LAB)


def test_eval_source_full_uses_full_model(tmp_path):
    cfg = load_config(overrides=["folds.n_folds=2", "camformer.eval_source=full"])
    t = assemble_camformer(_write_run(tmp_path, with_full=True) / "20_camformer", cfg, LAB).sort_by()
    assert t.meta["cam_eval_source"] == "full"
    v0 = list(t.keys()).index("v0")
    assert t.arrays["cam_probs_cal"][v0].argmax() == 3  # dal modello full, non dalla media delle fold
    tr1 = list(t.keys()).index("tr1")
    assert t.arrays["cam_probs_cal"][tr1].argmax() == 1  # il train resta OOF


def test_oof_provenance_is_verified(tmp_path):
    """Regressione audit A-I4: il train deve arrivare da preds_oof della fold giusta."""
    from src.cascade.cache.build import ProvenanceError as FromBuild

    assert FromBuild is ProvenanceError  # riesportato da checks.py (nota N11)
    cfg = load_config(overrides=["folds.n_folds=2"])
    base = _write_run(tmp_path) / "20_camformer"
    folds = pd.DataFrame({"take_uid": ["tr0", "tr1"], "fold": [0, 1]})
    assemble_camformer(base, cfg, LAB, folds)  # coerente: ok
    wrong = pd.DataFrame({"take_uid": ["tr0", "tr1"], "fold": [1, 0]})
    with pytest.raises(ProvenanceError, match="fold"):
        assemble_camformer(base, cfg, LAB, wrong)
    write_table(_preds(["tr0"], "train", 0, 0, source="eval"), base / "fold0" / "preds_oof.parquet")
    with pytest.raises(ProvenanceError, match="source"):
        assemble_camformer(base, cfg, LAB, folds)


def test_load_l1_checks_provenance(tmp_path):
    df = pd.DataFrame(
        {
            "clip_id": ["a", "b"],
            "take_uid": ["ta", "tb"],
            "split": ["train", "val"],
            "label_idx": 0,
            "T_vid": 1.0,
            "vid_fold": [0, -1],
        }
    )
    arrays = {"vid_probs_cal": np.full((2, 8), 0.125)}
    meta = {"class_names": list(LAB.names), "vid_train_source": "oof"}
    write_table(Table(df, arrays, meta), tmp_path / "preds.parquet")
    ok = pd.DataFrame({"take_uid": ["ta"], "fold": [0]})
    assert len(load_l1(tmp_path, LAB, ok)) == 2
    with pytest.raises(ProvenanceError, match="vid_fold"):
        load_l1(tmp_path, LAB, pd.DataFrame({"take_uid": ["ta"], "fold": [1]}))
    write_table(Table(df, arrays, {**meta, "vid_train_source": "in_sample"}), tmp_path / "preds.parquet")
    with pytest.raises(ProvenanceError, match="fuori campione"):
        load_l1(tmp_path, LAB)


def test_join_l0_l1_by_clip_id():
    cam = _preds(["a", "b", "c"], "train", 0, 0)
    cam.meta["cam_train_source"] = "oof"
    vid = Table(
        pd.DataFrame(
            {"clip_id": ["c", "a"], "take_uid": ["c", "a"], "split": "train", "label_idx": 0, "T_vid": [2.0, 3.0]}
        ),
        {"vid_probs_cal": np.array([[1.0] + [0] * 7, [0, 1.0] + [0] * 6])},
        {"vid_train_source": "oof"},
    )
    j = join_l0_l1(cam, vid).sort_by()
    assert list(j.keys()) == ["a", "c"]
    np.testing.assert_allclose(j.df.T_vid, [3.0, 2.0])
    assert j.arrays["vid_probs_cal"][0].argmax() == 1
