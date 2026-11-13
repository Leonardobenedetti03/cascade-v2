import numpy as np
import pandas as pd

from src.cascade.config import load_config
from src.cascade.folds import make_folds
from src.cascade.heads import run_heads


def test_heads_oof_and_fold_average():
    rng = np.random.default_rng(0)
    C, rows = 4, []
    for split, n in (("train", 30), ("val", 6), ("test", 6)):
        for c in range(C):
            for i in range(n):
                rows.append(
                    {"clip_id": f"{split}{c}_{i}", "take_uid": f"{split}{c}_{i}", "split": split, "label_idx": c}
                )
    feats = pd.DataFrame(rows)
    centers = rng.normal(0, 3, (C, 10))
    X = centers[feats.label_idx] + rng.normal(0, 1, (len(feats), 10))
    folds = make_folds(feats[feats.split == "train"][["take_uid", "label_idx"]], 5, 0.1, 0)
    hcfg = load_config(overrides=["heads.Cs=[0.1, 1.0]"]).heads
    res, summary = run_heads(feats, X, folds, hcfg, C)
    assert len(summary) == 5 and (summary.oof_acc > 0.8).all()
    tr = (feats.split == "train").to_numpy()
    fold_T = dict(zip(summary.fold, summary["T"]))
    fold_of = dict(zip(folds.take_uid, folds.fold))
    np.testing.assert_allclose(res["T_vid"][tr], [fold_T[fold_of[t]] for t in feats.take_uid[tr]])
    np.testing.assert_allclose(res["vid_probs_cal"].sum(1), 1, atol=1e-6)
    assert (res["vid_fold"][~tr] == -1).all()
    assert (res["vid_probs_cal"][~tr].argmax(1) == feats.label_idx[~tr]).mean() > 0.8
