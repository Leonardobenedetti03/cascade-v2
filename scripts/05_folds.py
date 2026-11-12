"""05 — Fold del train, per take o per sito (``folds.group_by``).

Scopo:
    ``n_folds`` fold del train, un holdout interno (``inner_holdout_frac``) estratto dalle fold di
    training per early stopping e temperatura, più un holdout per il modello ``full``.
    - ``group_by: take``: fold stratificate per classe, una riga per take (v1, run ``id_v2``);
    - ``group_by: site``: tutte le take di un sito stanno nella stessa fold (run ``site_v2``). La mappa
      sito -> fold si LEGGE da ``folds.site_map`` (``split.json`` versionato, D6), non si ricalcola.

Input:
    paths.scenario_csv (+ folds.site_map con group_by=site)

Output:
    <run_dir>/splits/folds.csv
    folds.csv versionato accanto a scenario_csv: se esiste deve coincidere con quello appena calcolato,
    altrimenti lo stadio si ferma (le fold non cambiano in silenzio tra macchine o versioni);
    --update-repo-folds per sostituirlo. Non con --synthetic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.cascade.cli import open_stage, stage_parser  # noqa: E402
from src.cascade.clips import load_scenario  # noqa: E402
from src.cascade.folds import make_folds  # noqa: E402
from src.cascade.io import write_csv  # noqa: E402
from src.cascade.site_split import site_of  # noqa: E402

STAGE = "05_folds"


def main(argv: Optional[List[str]] = None) -> None:
    p = stage_parser(__doc__)
    p.add_argument("--no-export", action="store_true", help="non confrontare/scrivere il folds.csv versionato")
    p.add_argument("--update-repo-folds", action="store_true", help="sostituisci il folds.csv versionato")
    args = p.parse_args(argv)
    ctx = open_stage(args, STAGE, make_stage_dir=False)
    out = ctx.run_dir / "splits" / "folds.csv"
    if ctx.done([out]):
        return
    fc = ctx.cfg.folds
    scenario_csv = ctx.path(ctx.cfg.paths.scenario_csv)
    takes = load_scenario(scenario_csv, ctx.scenario_labels or ctx.labels)
    train = takes[takes.split == "train"]
    if train.empty:
        raise ValueError(f"nessuna take di train in {scenario_csv}")
    kwargs = {}
    if fc.group_by == "site":
        meta = json.loads(Path(ctx.path(fc.site_map)).read_text())
        if int(meta["n_folds"]) != fc.n_folds:
            raise ValueError(f"folds.n_folds={fc.n_folds} ma {fc.site_map} ha n_folds={meta['n_folds']}")
        kwargs = {"group_by": "site", "site_to_fold": {k: int(v) for k, v in meta["site_to_fold"].items()}}
        train = train.assign(site=train.take_uid.map(site_of))
    folds = make_folds(train, fc.n_folds, fc.inner_holdout_frac, fc.seed, **kwargs)
    if not (args.synthetic or args.no_export):
        repo = Path(scenario_csv).parent / "folds.csv"
        if repo.exists() and not args.update_repo_folds:
            old = pd.read_csv(repo)
            same = old.shape == folds.shape and (old.astype(str).to_numpy() == folds.astype(str).to_numpy()).all()
            if not same:
                raise RuntimeError(
                    f"{repo} differisce dalle fold calcolate con questa config (folds={fc}): "
                    "usa le stesse impostazioni, oppure --update-repo-folds se il cambio è voluto"
                )
            ctx.log.info("fold identiche a %s", repo)
        else:
            write_csv(folds, repo)
            ctx.log.info("esportato %s", repo)
    write_csv(folds, out)
    ctx.log.info("fold (%s): %s", fc.group_by, folds.fold.value_counts().sort_index().to_dict())
    clips_csv = ctx.run_dir / "data" / "clips.csv"
    if clips_csv.exists():
        clips = pd.read_csv(clips_csv)
        usable = set(clips[clips.has_video & clips.has_traj].take_uid)
        lacking = folds[~folds.take_uid.isin(usable)]
        ctx.log.info(
            "take di train senza video o traiettoria: %d (per fold %s)",
            len(lacking),
            lacking.fold.value_counts().sort_index().to_dict(),
        )
    ctx.finish([out])
    ctx.log.info("scritto %s", out)


if __name__ == "__main__":
    main()
