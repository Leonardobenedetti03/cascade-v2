"""Diagnostiche di sito sulle CLS di L1 (stadio 27, M02).

Perché esiste. Nei dati disponibili il laboratorio (sito) e lo scenario sono quasi la stessa cosa
(vedi ``site_split.py``). Con fold per take L1 sembra quasi perfetto, ma su un sito mai visto
scende molto. Queste diagnostiche misurano quanto: sono i numeri che giustificano M02 nel report.

Tre analisi (blueprint 12.6), tutte su **train e val**: il test non entra mai, perché lo stadio
gira fuori da ``50 --final``.

(a) ``loso_table``: leave-one-site-out. Una logistica sulle CLS addestrata su tutti i siti meno uno
    e valutata su quello, per sito e per classe. Si valutano solo le clip la cui classe compare nel
    train di quel giro (come ``check_site.py`` della v1): le altre misurerebbero l'assenza della
    classe, non lo spostamento di dominio.
(b) ``site_readability``: quanto il sito si legge dalle feature, cioè una logistica che predice il
    sito dalle CLS con CV raggruppata (per take e per sessione), contro il caso.
(c) ``majority_summary``: baseline «classe più frequente del sito» (``site_majority_accuracy``).

In più, da ``check_leak.py`` della v1: ``grouped_cv_accuracy`` applicata alla classe, con CV per take
e per sessione. Con i siti mescolati tra train e test la classe si legge quasi sempre; il salto
rispetto al leave-one-site-out è la misura della scorciatoia di sito.

Le funzioni sono pure (niente I/O); scikit-learn si importa solo dentro i fit.

Limiti dichiarati:
- il sito è il prefisso del ``take_uid``; la sessione è il ``take_uid`` senza l'ultimo suffisso;
- gli avvisi di convergenza della logistica non si nascondono: si contano in ``fits`` del JSON;
- la baseline per sito (c) è in-sample, come ``site_majority_accuracy``: è un riferimento, non un
  modello.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from src.cascade.site_split import site_majority_accuracy, site_of

ALL = "ALL"
LOSO_COLUMNS = ["site", "role", "class", "n_clips", "n_takes", "n_scored", "n_correct", "acc", "reported"]
SCOPE = "train+val (il test non è usato)"


def session_of(take_uid: str) -> str:
    """Sessione di registrazione: il ``take_uid`` senza l'ultimo suffisso (``uniandes_dance_022_7`` ->
    ``uniandes_dance_022``). Senza ``_`` la sessione coincide con la take."""
    head, sep, _tail = str(take_uid).rpartition("_")
    return head if sep else str(take_uid)


def roles_from_takes(takes: pd.DataFrame) -> pd.DataFrame:
    """Output di ``clips.load_scenario`` (``take_uid, split, label_idx``) nel formato di ``plan_roles``
    (``take_uid, label_idx, role, site``): serve a ``site_majority_accuracy``."""
    out = pd.DataFrame(
        {
            "take_uid": takes.take_uid.to_numpy(),
            "label_idx": takes.label_idx.to_numpy().astype(int),
            "role": takes.split.to_numpy(),
        }
    )
    out["site"] = out.take_uid.map(site_of)
    return out


def _fit_predict(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, C: float, max_iter: int, counter: Dict[str, int]
):
    """StandardScaler + LogisticRegression(C, max_iter). Con una sola classe nel train predice quella.

    Gli avvisi emessi dal fit sono contati (quelli di convergenza) e poi riemessi tali e quali."""
    classes = np.unique(y_tr)
    if len(classes) < 2:
        return np.full(len(X_te), classes[0])
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_tr, y_tr)
    for w in caught:
        if issubclass(w.category, ConvergenceWarning):
            counter["convergence_warnings"] += 1
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    counter["fits"] += 1
    return model.predict(X_te)


def loso_predict(X: np.ndarray, y: np.ndarray, sites: np.ndarray, C: float, max_iter: int, counter: Dict[str, int]):
    """Per ogni sito: fit sugli altri, predizione su quel sito. Restituisce ``(pred, scored)``;
    ``scored`` è vero per le clip la cui classe è presente nel train di quel giro."""
    sites = np.asarray(sites)
    pred = np.full(len(y), -1, dtype=int)
    scored = np.zeros(len(y), dtype=bool)
    for s in sorted(set(sites.tolist())):
        te = sites == s
        tr = ~te
        pred[te] = _fit_predict(X[tr], y[tr], X[te], C, max_iter, counter)
        scored[te] = np.isin(y[te], np.unique(y[tr]))
    return pred, scored


def _row(sub: pd.DataFrame, site: str, role: str, cname: str, reported: bool) -> Dict[str, Any]:
    sc = sub[sub["scored"]]
    n_scored = int(len(sc))
    n_correct = int((sc["pred"] == sc["label_idx"]).sum())
    return {
        "site": site,
        "role": role,
        "class": cname,
        "n_clips": int(len(sub)),
        "n_takes": int(sub["take_uid"].nunique()),
        "n_scored": n_scored,
        "n_correct": n_correct,
        "acc": n_correct / n_scored if n_scored else float("nan"),
        "reported": bool(reported),
    }


def loso_table(df: pd.DataFrame, class_names: Sequence[str], min_class_takes: int) -> pd.DataFrame:
    """Tabella lunga del leave-one-site-out: una riga per (sito, classe), per sito (classe ``ALL``),
    per classe (sito ``ALL``) e il totale (``ALL``, ``ALL``).

    ``df`` ha ``take_uid, site, role, label_idx, pred, scored``. ``reported`` è falso per le classi
    con meno di ``min_class_takes`` take in tutto (stessa regola di M14: Bouldering resta fuori)."""
    takes_per_class = df.groupby("label_idx")["take_uid"].nunique()

    def reported(c: int) -> bool:
        return bool(takes_per_class.get(c, 0) >= min_class_takes)

    rows = []
    for site in sorted(df["site"].unique()):
        sub = df[df["site"] == site]
        role = ",".join(sorted(set(sub["role"])))
        rows.append(_row(sub, site, role, ALL, True))
        for c in sorted(sub["label_idx"].unique()):
            rows.append(_row(sub[sub["label_idx"] == c], site, role, class_names[int(c)], reported(c)))
    rows.append(_row(df, ALL, "train+val", ALL, True))
    for c in sorted(df["label_idx"].unique()):
        rows.append(_row(df[df["label_idx"] == c], ALL, "train+val", class_names[int(c)], reported(c)))
    return pd.DataFrame(rows, columns=LOSO_COLUMNS)


def grouped_cv_accuracy(
    X: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    C: float,
    max_iter: int,
    counter: Dict[str, int],
) -> Dict[str, Any]:
    """Accuratezza fuori campione di una logistica sul ``target`` (intero) con GroupKFold sui ``groups``."""
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(groups)
    n_groups = len(set(groups.tolist()))
    k = min(n_splits, n_groups)
    if k < 2:
        return {"skipped": "meno di 2 gruppi", "n_groups": int(n_groups)}
    pred = np.empty(len(target), dtype=target.dtype)
    for tr, te in GroupKFold(n_splits=k).split(X, target, groups):
        pred[te] = _fit_predict(X[tr], target[tr], X[te], C, max_iter, counter)
    return {"acc": float((pred == target).mean()), "n_splits": int(k), "n_groups": int(n_groups)}


def site_readability(
    X: np.ndarray,
    sites: np.ndarray,
    take_groups: np.ndarray,
    session_groups: np.ndarray,
    n_splits: int,
    C: float,
    max_iter: int,
    counter: Dict[str, int],
) -> Dict[str, Any]:
    """Quanto il sito si legge dalle CLS: accuratezza di una logistica che predice il sito, con CV per
    take e per sessione, contro il caso (sito più grande e uniforme)."""
    codes, uniques = pd.factorize(np.asarray(sites))
    codes = np.asarray(codes)
    counts = np.bincount(codes)
    return {
        "n_sites": int(len(uniques)),
        "chance_majority": float(counts.max() / len(codes)),
        "chance_uniform": float(1.0 / len(uniques)),
        "by_take": grouped_cv_accuracy(X, codes, take_groups, n_splits, C, max_iter, counter),
        "by_session": grouped_cv_accuracy(X, codes, session_groups, n_splits, C, max_iter, counter),
    }


def majority_summary(roles: pd.DataFrame, class_names: Sequence[str]) -> Dict[str, Any]:
    """Baseline «classe più frequente del sito» (in-sample sul train) e tabella sito × classe.

    ``roles`` ha ``take_uid, label_idx, role, site``. Le take di test non compaiono: né i loro siti né
    i conteggi. L'accuratezza è quella di ``site_majority_accuracy`` (sulle take di train)."""
    seen = roles[roles["role"] != "test"]
    per_site = []
    crosstab: Dict[str, Dict[str, int]] = {}
    for site, g in seen.groupby("site"):
        counts = g["label_idx"].value_counts()
        top = int(counts[counts == counts.max()].index.min())
        per_site.append(
            {
                "site": site,
                "role": ",".join(sorted(set(g["role"]))),
                "n_takes": int(len(g)),
                "n_classes": int(len(counts)),
                "majority_class": class_names[top],
                "majority_share": float(counts.max() / len(g)),
            }
        )
        crosstab[site] = {class_names[int(c)]: int(n) for c, n in sorted(counts.items())}
    return {
        "scope": SCOPE,
        "accuracy_train_insample": float(site_majority_accuracy(roles)),
        "n_train_takes": int((roles["role"] == "train").sum()),
        "n_sites": len(per_site),
        "n_sites_single_class": sum(1 for r in per_site if r["n_classes"] == 1),
        "per_site": per_site,
        "crosstab": crosstab,
    }


def run_site_diagnostics(
    feats: pd.DataFrame,
    X: np.ndarray,
    takes: pd.DataFrame,
    class_names: Sequence[str],
    *,
    C: float,
    max_iter: int,
    cv_folds: int,
    min_class_takes: int,
) -> Dict[str, Any]:
    """Le tre analisi più il controllo di ``check_leak``. Restituisce ``loso`` (DataFrame), ``probe`` e
    ``majority`` (dict).

    ``feats``: una riga per clip con ``clip_id, take_uid, split, label_idx``; ``X``: [N, D] CLS di L1;
    ``takes``: output di ``clips.load_scenario`` (tutte le take dello split, per la baseline)."""
    counter = {"fits": 0, "convergence_warnings": 0}
    keep = (feats["split"] != "test").to_numpy()
    df = feats.loc[keep, ["clip_id", "take_uid", "split", "label_idx"]].reset_index(drop=True)
    Xs = np.asarray(X)[keep]
    df["site"] = df["take_uid"].map(site_of)
    df["role"] = df["split"]
    y = df["label_idx"].to_numpy().astype(int)
    sites = df["site"].to_numpy()
    take_g = df["take_uid"].to_numpy()
    sess_g = df["take_uid"].map(session_of).to_numpy()
    n_sites = len(set(sites.tolist()))
    majority = majority_summary(roles_from_takes(takes), class_names)
    probe: Dict[str, Any] = {
        "scope": SCOPE,
        "model": {"features": "vid_cls", "scaler": "StandardScaler", "C": C, "max_iter": max_iter},
        "cv_folds": cv_folds,
        "n_clips": int(len(df)),
        "n_takes": int(df["take_uid"].nunique()),
        "n_sessions": int(len(set(sess_g.tolist()))),
        "n_sites": n_sites,
        "fits": counter,
    }
    if n_sites < 2:
        probe["skipped"] = f"{n_sites} sito in train+val: servono almeno 2 siti"
        return {"loso": pd.DataFrame(columns=LOSO_COLUMNS), "probe": probe, "majority": majority}
    pred, scored = loso_predict(Xs, y, sites, C, max_iter, counter)
    df["pred"], df["scored"] = pred, scored
    loso = loso_table(df, class_names, min_class_takes)
    tot = loso[(loso["site"] == ALL) & (loso["class"] == ALL)].iloc[0]
    probe["loso"] = {
        "acc": float(tot["acc"]),
        "n_scored": int(tot["n_scored"]),
        "n_excluded_unseen_class": int(tot["n_clips"] - tot["n_scored"]),
    }
    probe["site_readability"] = site_readability(Xs, sites, take_g, sess_g, cv_folds, C, max_iter, counter)
    probe["label_cv"] = {
        "by_take": grouped_cv_accuracy(Xs, y, take_g, cv_folds, C, max_iter, counter),
        "by_session": grouped_cv_accuracy(Xs, y, sess_g, cv_folds, C, max_iter, counter),
    }
    return {"loso": loso, "probe": probe, "majority": majority}
