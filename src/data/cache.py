"""Tabelle parquet con colonne vettoriali e join per chiave.

Formato su disco (cache finale e predizioni intermedie degli stadi): un file parquet
con una riga per clip. Le colonne scalari sono colonne parquet normali; i vettori
(``cam_logits[8]``, ``cam_emb[512]``, ...) sono colonne ``fixed_size_list<float32>``.
Scalari e vettori stanno nella stessa riga dello stesso file, quindi non possono
disallinearsi; i metadati (ordine delle classi, stadio, ...) stanno nello schema.

In memoria una :class:`Table` è un ``DataFrame`` di scalari più un dict di matrici
``[N, K]`` allineate per riga. Il join tra tabelle è **sempre per chiave**
(default ``clip_id``), mai per posizione.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

META_KEY = b"cascade"
KEY = "clip_id"


@dataclass
class Table:
    df: pd.DataFrame
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.df = self.df.reset_index(drop=True)
        n = len(self.df)
        for name, arr in self.arrays.items():
            if arr.ndim != 2 or arr.shape[0] != n:
                raise ValueError(f"array {name!r}: shape {arr.shape}, attese {n} righe 2D")
            if name in self.df.columns:
                raise ValueError(f"{name!r} è sia colonna scalare sia vettoriale")

    def __len__(self) -> int:
        return len(self.df)

    @property
    def columns(self) -> List[str]:
        return list(self.df.columns) + list(self.arrays)

    def take(self, idx: np.ndarray) -> Table:
        idx = np.asarray(idx)
        return Table(
            self.df.iloc[idx].reset_index(drop=True), {k: v[idx] for k, v in self.arrays.items()}, dict(self.meta)
        )

    def mask(self, m: np.ndarray) -> Table:
        return self.take(np.flatnonzero(np.asarray(m, dtype=bool)))

    def sort_by(self, key: str = KEY) -> Table:
        order = np.argsort(self.df[key].astype(str).to_numpy(), kind="stable")
        return self.take(order)

    def keys(self, key: str = KEY) -> np.ndarray:
        return self.df[key].astype(str).to_numpy()

    def check_unique(self, key: str = KEY) -> None:
        dup = self.df[key][self.df[key].duplicated()]
        if len(dup):
            raise ValueError(f"chiave {key!r} duplicata, es. {dup.iloc[:3].tolist()}")


def write_table(table: Table, path: Path, *, key: Optional[str] = KEY) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if key is not None:
        table.check_unique(key)
    pa_tab = pa.Table.from_pandas(table.df, preserve_index=False)
    for name, arr in table.arrays.items():
        a = np.ascontiguousarray(arr, dtype=np.float32)
        flat = pa.array(a.reshape(-1), type=pa.float32())
        pa_tab = pa_tab.append_column(name, pa.FixedSizeListArray.from_arrays(flat, a.shape[1]))
    meta = dict(pa_tab.schema.metadata or {})
    meta[META_KEY] = json.dumps(table.meta).encode()
    pa_tab = pa_tab.replace_schema_metadata(meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa_tab, tmp)
    tmp.replace(path)


def read_table(path: Path, columns: Optional[Sequence[str]] = None) -> Table:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pa_tab = pq.read_table(path, columns=list(columns) if columns is not None else None)
    arrays: Dict[str, np.ndarray] = {}
    scalar_cols = []
    for f in pa_tab.schema:
        if pa.types.is_fixed_size_list(f.type):
            col = pa_tab.column(f.name).combine_chunks()
            values = col.flatten().to_numpy(zero_copy_only=False)
            arrays[f.name] = values.reshape(len(col), f.type.list_size).astype(np.float32, copy=False)
        else:
            scalar_cols.append(f.name)
    df = pa_tab.select(scalar_cols).to_pandas()
    for c in df.columns:  # stringhe come object, uguale con pandas 2 e 3
        if pd.api.types.is_string_dtype(df[c]) and not pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].astype(object)
    raw = (pa_tab.schema.metadata or {}).get(META_KEY)
    meta = json.loads(raw.decode()) if raw else {}
    return Table(df, arrays, meta)


def parquet_columns(path: Path) -> List[str]:
    """Nomi delle colonne senza leggere i dati (per letture a colonne)."""
    import pyarrow.parquet as pq

    return list(pq.read_schema(path).names)


def join_tables(left: Table, right: Table, *, key: str = KEY, how: str = "inner", validate_full: bool = False) -> Table:
    """Join per ``key``. Con ``validate_full`` le chiavi dei due lati devono coincidere.

    Le colonne in comune (oltre alla chiave) devono avere valori identici, altrimenti
    è un errore: si evita di scegliere in silenzio una delle due versioni.
    """
    if how not in {"inner", "left"}:
        raise ValueError("how deve essere 'inner' o 'left'")
    left.check_unique(key)
    right.check_unique(key)
    lk, rk = left.keys(key), right.keys(key)
    if validate_full and set(lk) != set(rk):
        only_l, only_r = sorted(set(lk) - set(rk)), sorted(set(rk) - set(lk))
        raise ValueError(
            f"join incompleto su {key}: solo a sinistra {only_l[:5]} ({len(only_l)}), "
            f"solo a destra {only_r[:5]} ({len(only_r)})"
        )
    r_pos = {k: i for i, k in enumerate(rk)}
    if how == "inner":
        l_idx = np.array([i for i, k in enumerate(lk) if k in r_pos], dtype=int)
    else:
        l_idx = np.arange(len(lk))
    r_idx = np.array([r_pos.get(lk[i], -1) for i in l_idx], dtype=int)
    if how == "left" and (r_idx < 0).any() and right.arrays:
        raise ValueError("left join con righe mancanti a destra e colonne vettoriali: non supportato")
    ldf = left.df.iloc[l_idx].reset_index(drop=True)
    rdf = right.df.iloc[np.where(r_idx >= 0, r_idx, 0)].reset_index(drop=True)
    if (r_idx < 0).any():
        rdf.loc[r_idx < 0, :] = np.nan
    common = [c for c in rdf.columns if c in ldf.columns and c != key]
    for c in common:
        a, b = ldf[c].to_numpy(), rdf[c].to_numpy()
        ok = r_idx < 0
        try:
            same = ok | (a == b) | (pd.isna(a) & pd.isna(b))
        except TypeError:
            same = ok | (a.astype(str) == b.astype(str))
        if not np.all(same):
            raise ValueError(f"colonna {c!r} presente in entrambe le tabelle con valori diversi")
    df = pd.concat([ldf, rdf.drop(columns=[key] + common)], axis=1)
    clash = set(left.arrays) & set(right.arrays)
    if clash:
        raise ValueError(f"colonne vettoriali in entrambe le tabelle: {sorted(clash)}")
    arrays = {k: v[l_idx] for k, v in left.arrays.items()}
    arrays.update({k: v[r_idx] for k, v in right.arrays.items()})
    meta = {**right.meta, **left.meta}
    return Table(df, arrays, meta)


def concat_tables(tables: Iterable[Table]) -> Table:
    tables = list(tables)
    if not tables:
        raise ValueError("nessuna tabella")
    names = set(tables[0].arrays)
    for t in tables[1:]:
        if set(t.arrays) != names:
            raise ValueError("tabelle con colonne vettoriali diverse")
    df = pd.concat([t.df for t in tables], ignore_index=True)
    arrays = {k: np.concatenate([t.arrays[k] for t in tables]) for k in names}
    return Table(df, arrays, dict(tables[0].meta))
