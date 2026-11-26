"""Linear heads on the frozen embeddings (stage ``heads``), with the same protocol for both models.

One row = one take (embedding averaged over its windows). For each train fold k:
- fit on the train takes with fold != k and a class in the head (Music and Bouldering are left out):
  StandardScaler + multinomial LogisticRegression, C chosen by inner CV grouped by session (log-loss), and a
  temperature fitted on the out-of-sample logits of the inner CV with the chosen C;
- predict the train takes of fold k (out of sample, Bouldering included) and the val/test takes.
For val and test we average the calibrated probabilities of the K fold models. This way every take gets a
prediction from a model that has not seen it. With more seeds (different fold partitions) we repeat everything
and average the probabilities: on train they stay out of sample, on val and test they become the mean of
K x seeds models.

Note: with ``class_weight: balanced`` the probabilities include the weight correction (rare classes look more
likely than with the real frequencies). The temperature fixes the scale, not this shift.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

log = logging.getLogger(__name__)
MISSING_LOGIT = -50.0


def nll(logits: np.ndarray, y: np.ndarray, T: float = 1.0) -> float:
    lp = log_softmax(logits / T, axis=1)
    return float(-lp[np.arange(len(y)), y].mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return 1.0
    r = minimize_scalar(lambda lt: nll(logits, y, float(np.exp(lt))), bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(r.x))


def _model(C: float, max_iter: int, class_weight: Optional[str], seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter, class_weight=class_weight, random_state=seed))


def _fit(m, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return m.fit(X, y)


def logits_of(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Logits ``[N, C]`` even if some class was missing from the fit (very negative logit)."""
    raw = model.decision_function(X)
    classes = model[-1].classes_
    if raw.ndim == 1:
        raw = np.stack([-raw, raw], 1) / 2
    out = np.full((len(X), n_classes), MISSING_LOGIT)
    out[:, classes] = raw
    return out


def fit_head(X, y, groups, n_classes: int, hcfg: dict, seed: int):
    """(model, C, T). Classes with fewer than 2 groups stay out of the inner CV but are in the final fit."""
    from sklearn.model_selection import StratifiedGroupKFold

    Cs = list(hcfg["Cs"])
    groups_per_class = np.array([len(np.unique(groups[y == c])) for c in range(n_classes)])
    in_cv = groups_per_class[y] >= 2
    usable = groups_per_class[groups_per_class >= 2]
    n_splits = min(hcfg["inner_cv"], int(usable.min())) if len(usable) else 1
    best_C, T = Cs[0], 1.0
    if n_splits >= 2:
        Xc, yc, gc = X[in_cv], y[in_cv], groups[in_cv]
        cv = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(Xc, yc, gc))
        best, best_oof = None, None
        for C in Cs:
            oof = np.zeros((len(yc), n_classes))
            for tr, te in cv:
                m = _fit(_model(C, hcfg["max_iter"], hcfg["class_weight"], seed), Xc[tr], yc[tr])
                oof[te] = logits_of(m, Xc[te], n_classes)
            score = nll(oof, yc)
            if best is None or score < best:
                best, best_C, best_oof = score, C, oof
        T = fit_temperature(best_oof, yc)
    else:
        log.warning("CV interna saltata (nessuna classe con 2 gruppi): C=%g, T=1", best_C)
    model = _fit(_model(best_C, hcfg["max_iter"], hcfg["class_weight"], seed), X, y)
    return model, best_C, T


def cross_fit(X: np.ndarray, takes: pd.DataFrame, n_classes: int, hcfg: dict, seed: int, fold_col: str = "fold") -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Calibrated probabilities ``[N, C]`` for all takes with an embedding; NaN where it is missing."""
    y = takes.label_idx.to_numpy()
    fold = takes[fold_col].to_numpy()
    split = takes.split.to_numpy()
    groups = takes.session.to_numpy()
    has_x = np.isfinite(X).all(1)
    is_train = split == "train"
    fit_pool = is_train & (y >= 0) & has_x
    ev = ~is_train & has_x
    probs = np.full((len(takes), n_classes), np.nan)
    source = np.full(len(takes), "", dtype=object)
    ens: List[np.ndarray] = []
    rows = []
    for k in sorted(set(fold[is_train]) - {-1}):
        fit = fit_pool & (fold != k)
        missing = sorted(set(range(n_classes)) - set(y[fit]))
        model, C, T = fit_head(X[fit], y[fit], groups[fit], n_classes, hcfg, seed)
        held = is_train & (fold == k) & has_x
        if held.any():
            probs[held] = softmax(logits_of(model, X[held], n_classes) / T, axis=1)
            source[held] = "oof"
        if ev.any():
            ens.append(softmax(logits_of(model, X[ev], n_classes) / T, axis=1))
        hy = y[held]
        acc = float((probs[held].argmax(1)[hy >= 0] == hy[hy >= 0]).mean()) if (hy >= 0).any() else float("nan")
        rows.append({"fold": int(k), "C": C, "T": T, "n_fit": int(fit.sum()), "n_oof": int(held.sum()), "oof_acc": acc, "classes_missing_in_fit": missing})
        log.info("fold %d: C=%g T=%.3f n_fit=%d acc OOF=%.4f mancanti=%s", k, C, T, fit.sum(), acc, missing)
    if ens:
        probs[ev] = np.mean(ens, axis=0)
        source[ev] = "fold_ensemble"
    return {"probs": probs, "source": source}, pd.DataFrame(rows)


def metrics(df: pd.DataFrame, classes: List[str], split: str, min_takes: int) -> Optional[dict]:
    """Accuracy, balanced accuracy (classes with >= min_takes takes) and recall per class."""
    from sklearn.metrics import confusion_matrix

    labelled = df[(df.split == split) & (df.label_idx >= 0)]
    d = labelled[labelled.pred.notna()]
    if d.empty:
        return None
    y, p = d.label_idx.to_numpy(), d.pred_idx.to_numpy().astype(int)
    counts = np.bincount(y, minlength=len(classes))
    recall = {classes[c]: float((p[y == c] == c).mean()) for c in range(len(classes)) if counts[c] > 0}
    kept = [classes[c] for c in range(len(classes)) if counts[c] >= min_takes]
    return {
        "n": int(len(d)),
        "n_no_embedding": int(len(labelled) - len(d)),  # labelled takes with no prediction (not in these metrics)
        "acc": float((y == p).mean()),
        "bacc": float(np.mean([recall[c] for c in kept])) if kept else float("nan"),
        "bacc_classes": kept,
        "recall": recall,
        "support": {classes[c]: int(counts[c]) for c in range(len(classes)) if counts[c] > 0},
        "confusion": confusion_matrix(y, p, labels=list(range(len(classes)))).tolist(),
    }


def cross_fit_seeds(X: np.ndarray, takes: pd.DataFrame, n_classes: int, hcfg: dict, seeds: List[int]):
    """:func:`cross_fit` for each seed (the ``fold_s<seed>`` columns of the index), with averaged probabilities.

    It also returns ``seed_sd``: the standard deviation across seeds of the probability of the predicted class,
    to see how much the prediction depends on the fold partition.
    """
    runs, tabs, source = [], [], None
    for s in seeds:
        res, tab = cross_fit(X, takes, n_classes, hcfg, s, f"fold_s{s}")
        runs.append(res["probs"])
        tabs.append(tab.assign(seed=s))
        source = res["source"]
    stack = np.stack(runs)  # [S, N, C]
    P = stack.mean(0)
    sd = np.full(len(P), np.nan)
    ok = np.isfinite(P).all(1)
    if ok.any():
        k = P[ok].argmax(1)
        sd[ok] = stack[:, ok, :][:, np.arange(ok.sum()), k].std(0)
    return {"probs": P, "source": source, "seed_sd": sd}, pd.concat(tabs, ignore_index=True)


def _bacc(y: np.ndarray, p: np.ndarray, keep: List[int]) -> float:
    rec = [(p[y == c] == c).mean() for c in keep if (y == c).any()]
    return float(np.mean(rec)) if rec else float("nan")


def bacc_ci(df: pd.DataFrame, classes: List[str], split: str, min_takes: int, n_boot: int, level: float, seed: int = 0) -> Optional[Tuple[float, float]]:
    """Bootstrap interval of the balanced accuracy, resampling SESSIONS (takes of the same session are not
    independent). Classes as in :func:`metrics`."""
    d = df[(df.split == split) & (df.label_idx >= 0) & df.pred.notna()]
    if d.empty:
        return None
    y, p = d.label_idx.to_numpy(), d.pred_idx.to_numpy().astype(int)
    counts = np.bincount(y, minlength=len(classes))
    keep = [c for c in range(len(classes)) if counts[c] >= min_takes]
    sess = d.session.to_numpy()
    uniq = np.unique(sess)
    rows = {s_: np.flatnonzero(sess == s_) for s_ in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[s_] for s_ in rng.choice(uniq, len(uniq), replace=True)])
        vals.append(_bacc(y[idx], p[idx], keep))
    a = (1.0 - level) / 2.0
    return float(np.nanpercentile(vals, 100 * a)), float(np.nanpercentile(vals, 100 * (1 - a)))


def preregistered_check(df: pd.DataFrame, classes: List[str], ccfg: dict, hcfg: dict) -> dict:
    """Check written in the decision log before the run, on one model (usually CamFormer = L0):

    1. on split ``ccfg['split']`` (out-of-sample train, or val if there is one): balanced accuracy
       >= ``min_bacc``; with ``ci_above_chance: true`` the lower end of the bootstrap interval must also be
       above chance (chance = 1 / number of evaluated classes);
    2. on the out-of-sample train: recall >= ``min_recall`` for every head class present in train
       (``min_recall: null`` = recall > 0);
    3. with ``require_all_classes: true``: every head class has at least one evaluated take both in the split of
       the check and in train (so a class cannot silently drop out of the balanced accuracy).
    The test split is never used here.
    """
    split = ccfg["split"]
    if split == "test":
        raise ValueError("check.split = test non è ammesso: il test resta sigillato")
    if ccfg["min_bacc"] is None:
        raise ValueError("check.min_bacc non fissato: scriverlo nel registro e nella config PRIMA della run")
    m = metrics(df, classes, split, hcfg["min_takes_metric"])
    if m is None:
        raise ValueError(f"nessuna take valutabile nello split {split!r} (con official_val.mode = all_test usare check.split = train)")
    ci = bacc_ci(df, classes, split, hcfg["min_takes_metric"], hcfg["n_boot"], hcfg["ci_level"])
    chance = 1.0 / len(m["bacc_classes"])
    tr = metrics(df, classes, "train", 1)
    min_rec = ccfg.get("min_recall")
    recall = tr["recall"] if tr else {}
    low = sorted(c for c, r in recall.items() if (r < min_rec if min_rec is not None else r == 0.0))
    zero = sorted(c for c, r in recall.items() if r == 0.0)
    absent = sorted(set(classes) - set(tr["support"] if tr else {}))
    ci_ok = ci is not None and ci[0] > chance
    c1 = bool(m["bacc"] >= ccfg["min_bacc"] and (ci_ok or not ccfg.get("ci_above_chance", False)))
    c2 = not low
    missing_split = sorted(set(classes) - set(m["bacc_classes"]))
    require_all = bool(ccfg.get("require_all_classes", False))
    c3 = not (require_all and (missing_split or absent))
    return {
        "model": ccfg["model"],
        "split": split,
        "bacc": m["bacc"],
        "bacc_ci": ci,
        "ci_level": hcfg["ci_level"],
        "chance": chance,
        "min_bacc": ccfg["min_bacc"],
        "bacc_classes": m["bacc_classes"],
        "cond1_bacc_ok": c1,
        "train_oof_recall": tr["recall"] if tr else {},
        "ci_above_chance_required": bool(ccfg.get("ci_above_chance", False)),
        "ci_above_chance": bool(ci_ok),
        "min_recall": min_rec,
        "classes_below_min_recall": low,
        "classes_zero_recall": zero,
        "classes_absent_in_train": absent,
        "cond2_recall_ok": c2,
        "require_all_classes": require_all,
        "classes_missing_in_split": missing_split,
        "cond3_all_classes_ok": c3,
        "n_no_embedding": m["n_no_embedding"],
        "passed": bool(c1 and c2 and c3),
    }
