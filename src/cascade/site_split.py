"""Split per sito (laboratorio) delle take di Ego-Exo4D.

Perché esiste. Nei dati disponibili scenario e laboratorio sono quasi la stessa cosa
(Dance solo a uniandes e upenn, Bouldering a uniandes, Basketball a sfu/unc/uniandes...).
Con fold per take lo stesso laboratorio finisce in train e in val: L1 sembra quasi perfetto
(99%) ma sui siti mai visti scende molto (leave-one-site-out 62%). Qui il sito è l'unità di
divisione:

- ``plan_roles``: assegna ogni take a ``train``, ``val`` o ``test`` in base al sito;
- ``check_coverage``: controlla che ogni classe valutata nei siti tenuti fuori abbia almeno
  ``min_train_sites`` siti nel train (altrimenti la valutazione misura l'assenza di quella
  classe nel train e non lo shift di dominio);
- ``plan_site_to_fold``: calcola UNA volta la mappa sito -> fold (la scrive ``tools/make_site_split.py``
  in ``split.json``); le fold vere le fa ``folds.make_folds(group_by="site")`` leggendo quella mappa.

Il sito è il prefisso del ``take_uid`` (``uniandes_dance_022_7`` -> ``uniandes``).

Limiti dichiarati:
- l'holdout interno (early stopping e temperatura) resta stratificato per take, come in
  ``folds.py``: la temperatura è stimata su siti già visti e quindi sottostima lo shift;
- il sito più grande (uniandes) occupa da solo una fold intera, quindi le fold non sono
  bilanciate per dimensione;
- una classe con un solo sito nel train non ha esempi nel train della fold che contiene quel
  sito: lo si segnala nel report, non è un errore.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

ROLES = ("train", "val", "test")


def site_of(take_uid: str) -> str:
    """Sito di una take: il prefisso del ``take_uid`` prima del primo ``_``."""
    site = str(take_uid).split("_")[0]
    if not site:
        raise ValueError(f"take_uid senza prefisso di sito: {take_uid!r}")
    return site


def plan_roles(takes: pd.DataFrame, val_sites: Iterable[str], test_sites: Iterable[str]) -> pd.DataFrame:
    """Copia di ``takes`` con le colonne ``site`` e ``role`` (train / val / test).

    ``takes`` ha almeno ``take_uid`` e ``label_idx``. I siti non elencati vanno in ``train``.
    Siti sconosciuti o presenti sia in val sia in test sono un errore.
    """
    val, test = set(val_sites), set(test_sites)
    if val & test:
        raise ValueError(f"siti sia in val sia in test: {sorted(val & test)}")
    sites = takes.take_uid.map(site_of)
    unknown = sorted((val | test) - set(sites))
    if unknown:
        raise ValueError(f"siti non presenti nelle take: {unknown} (presenti: {sorted(set(sites))})")
    out = takes.copy()
    out["site"] = sites.to_numpy()
    out["role"] = np.where(sites.isin(test), "test", np.where(sites.isin(val), "val", "train"))
    return out


def check_coverage(
    roles: pd.DataFrame,
    class_names: Sequence[str],
    min_eval_takes: int = 8,
    min_train_sites: int = 2,
    min_takes_per_site: int = 5,
) -> Tuple[List[str], pd.DataFrame]:
    """Controlla la copertura delle classi. Restituisce (problemi, tabella per classe).

    Una classe è "valutata" in val o in test se ci ha almeno ``min_eval_takes`` take. Per
    ogni classe valutata servono almeno ``min_train_sites`` siti di train con almeno
    ``min_takes_per_site`` take della classe.
    """
    problems: List[str] = []
    rows = []
    train = roles[roles.role == "train"]
    for c in sorted(roles.label_idx.unique()):
        per_site = train[train.label_idx == c].groupby("site").size()
        n_sites = int((per_site >= min_takes_per_site).sum())
        row = {"classe": class_names[int(c)], "train_take": int(per_site.sum()), "train_siti": n_sites}
        for role in ("val", "test"):
            n = int(((roles.role == role) & (roles.label_idx == c)).sum())
            row[f"{role}_take"] = n
            if n >= min_eval_takes and n_sites < min_train_sites:
                problems.append(
                    f"{class_names[int(c)]}: {n} take in {role} ma solo {n_sites} siti di train "
                    f"con almeno {min_takes_per_site} take (servono {min_train_sites})"
                )
        rows.append(row)
    return problems, pd.DataFrame(rows)


def plan_site_to_fold(site_sizes: Dict[str, int], site_classes: Dict[str, Set[int]], n_folds: int) -> Dict[str, int]:
    """Assegna ogni sito a una fold (greedy, deterministico).

    I siti si prendono dal più grande; ognuno va nella fold col minor numero di take, tra
    quelle valide. Una fold non è valida se, con quel sito, conterrebbe tutti i siti di una
    classe presente in almeno due siti: il modello di quella fold non vedrebbe mai la classe.
    """
    if n_folds < 2:
        raise ValueError("servono almeno 2 fold")
    if len(site_sizes) < n_folds:
        raise ValueError(f"{len(site_sizes)} siti di train per {n_folds} fold: servono almeno tanti siti quante fold")
    class_sites: Dict[int, Set[str]] = {}
    for site, classes in site_classes.items():
        for c in classes:
            class_sites.setdefault(c, set()).add(site)
    multi = [sites for sites in class_sites.values() if len(sites) >= 2]
    fold_sites: List[Set[str]] = [set() for _ in range(n_folds)]
    load = [0] * n_folds
    out: Dict[str, int] = {}
    for site in sorted(site_sizes, key=lambda s: (-site_sizes[s], s)):
        valid = [k for k in range(n_folds) if not any(ss <= (fold_sites[k] | {site}) for ss in multi)]
        if not valid:
            raise ValueError(f"nessuna fold valida per il sito {site!r}")
        k = min(valid, key=lambda j: (load[j], j))
        fold_sites[k].add(site)
        load[k] += int(site_sizes[site])
        out[site] = k
    if any(not fs for fs in fold_sites):
        raise ValueError("una fold è rimasta senza siti: riduci n_folds")
    return out


def fold_diagnostics(folds: pd.DataFrame, class_names: Sequence[str]) -> pd.DataFrame:
    """Per fold: siti, take fuori campione e classi che la fold predice senza averle viste."""
    from .folds import training_masks

    n_folds = int(folds.fold.max()) + 1
    sites = folds.take_uid.map(site_of)
    rows = []
    for k in range(n_folds):
        train_m, _hold, eval_m = training_masks(folds, str(k))
        seen = set(folds.label_idx[train_m])
        unseen = sorted(set(folds.label_idx[eval_m]) - seen)
        rows.append(
            {
                "fold": k,
                "siti": ",".join(sorted(set(sites[eval_m]))),
                "take": int(eval_m.sum()),
                "classi_senza_train": ",".join(class_names[int(c)] for c in unseen) or "-",
            }
        )
    return pd.DataFrame(rows)


def site_majority_accuracy(roles: pd.DataFrame) -> float:
    """Accuratezza (in-sample, sul train) di "predici la classe più frequente del sito".

    Misura quanto scenario e laboratorio sono sovrapposti: è un riferimento per il report,
    non un modello.
    """
    train = roles[roles.role == "train"]
    hit = 0
    for _site, g in train.groupby("site"):
        hit += int(g.label_idx.value_counts().iloc[0])
    return hit / max(len(train), 1)


def roles_from_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """Ruoli da uno ``scenario_site.csv`` già fissato (colonne ``take_name, split, label``):
    stesso formato di ``plan_roles`` (``take_uid, label_idx, role, site``)."""
    need = {"take_name", "split", "label"}
    if not need <= set(df.columns):
        raise ValueError(f"scenario csv: colonne attese {sorted(need)}, trovate {list(df.columns)}")
    bad = sorted(set(df.split) - set(ROLES))
    if bad:
        raise ValueError(f"scenario csv: split non ammessi {bad} (ammessi: {list(ROLES)})")
    out = pd.DataFrame(
        {"take_uid": df.take_name.to_numpy(), "label_idx": df.label.astype(int).to_numpy(), "role": df.split.to_numpy()}
    )
    out["site"] = out.take_uid.map(site_of)
    return out


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_split_hash(path, expected: Optional[str]) -> str:
    """SHA256 del file di split; ``ValueError`` se ``expected`` è dato e non coincide."""
    got = file_sha256(path)
    if expected and got != expected:
        raise ValueError(f"{path}: SHA256 {got} diverso da quello in config ({expected}): lo split è stato modificato")
    return got
