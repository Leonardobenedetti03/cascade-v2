"""Genera lo split per sito versionato (splits/site_v1/): una volta, non dentro le run.

Due modi:
  --clips-csv <run>/data/clips.csv --val-sites ... --test-sites ...
      crea scenario_site.csv dalle take disponibili (come ``06_site_split.py`` della v1);
  --from-scenario-csv splits/site_v1/scenario_site.csv
      parte dallo split già fissato: scenario_site.csv non si riscrive e il suo hash resta quello.

In entrambi i casi scrive, accanto a scenario_site.csv:
  split.json      parametri, conteggi, hash e la mappa sito -> fold (D6: un dato, non un algoritmo)
  folds.csv       le fold del train (stesso formato dello stadio 05, con la colonna ``site``)
  report.md       siti x classe per ruolo, copertura, fold del train
  SHA256SUMS      scenario_site.csv per primo (hpc/ ne usa la prima riga come hash dello split)

Uso (split già fissato, da rifare identico):
    python tools/make_site_split.py --from-scenario-csv splits/site_v1/scenario_site.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import REPO_ROOT  # noqa: E402
from src.cascade.folds import make_folds  # noqa: E402
from src.cascade.io import write_csv, write_json, write_text  # noqa: E402
from src.cascade.labels import load_labels  # noqa: E402
from src.cascade.site_split import (  # noqa: E402
    check_coverage,
    file_sha256,
    fold_diagnostics,
    plan_roles,
    plan_site_to_fold,
    roles_from_scenario,
    site_majority_accuracy,
)

SUMS = ["scenario_site.csv", "split.json", "folds.csv", "report.md"]  # scenario_site.csv per primo


def _block(title: str, df: pd.DataFrame) -> str:
    return f"## {title}\n\n```\n{df.to_string()}\n```\n"


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--clips-csv", type=Path)
    src.add_argument("--from-scenario-csv", type=Path)
    p.add_argument("--val-sites", nargs="+")
    p.add_argument("--test-sites", nargs="+")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "splits" / "site_v1")
    p.add_argument("--labels", type=Path, default=REPO_ROOT / "configs" / "labels.yaml")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--inner-holdout-frac", type=float, default=0.1)
    p.add_argument("--min-eval-takes", type=int, default=8)
    p.add_argument("--min-train-sites", type=int, default=2)
    p.add_argument("--min-takes-per-site", type=int, default=5)
    p.add_argument("--allow-problems", action="store_true", help="scrivi i file anche se la copertura ha problemi")
    args = p.parse_args(argv)

    labels = load_labels(args.labels)
    if args.from_scenario_csv:
        out = args.from_scenario_csv.resolve().parent
        roles = roles_from_scenario(pd.read_csv(args.from_scenario_csv))
        val_sites = sorted(set(roles[roles.role == "val"].site))
        test_sites = sorted(set(roles[roles.role == "test"].site))
    else:
        if not args.val_sites or not args.test_sites:
            raise SystemExit("--clips-csv richiede --val-sites e --test-sites")
        out = args.out
        clips = pd.read_csv(args.clips_csv)
        need = {"take_uid", "label_idx", "has_video", "has_traj"}
        if not need <= set(clips.columns):
            raise SystemExit(f"{args.clips_csv}: colonne attese {sorted(need)}, trovate {list(clips.columns)}")
        usable = clips[clips.has_video.astype(bool) & clips.has_traj.astype(bool)]
        takes = usable.drop_duplicates("take_uid")[["take_uid", "label_idx"]].reset_index(drop=True)
        takes["label_idx"] = takes.label_idx.astype(int)
        print(f"take utilizzabili: {len(takes)} (clip: {len(clips)})")
        roles = plan_roles(takes, args.val_sites, args.test_sites)
        val_sites, test_sites = sorted(args.val_sites), sorted(args.test_sites)

    problems, cov = check_coverage(
        roles, labels.names, args.min_eval_takes, args.min_train_sites, args.min_takes_per_site
    )
    for msg in problems:
        print("PROBLEMA:", msg)
    if problems and not args.allow_problems:
        raise SystemExit("copertura insufficiente: cambia i siti oppure usa --allow-problems")

    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "scenario_site.csv"
    if args.clips_csv:  # con --from-scenario-csv il CSV non si tocca: l'hash resta quello fissato
        csv_out = pd.DataFrame(
            {
                "take_name": roles.take_uid.to_numpy(),
                "split": roles.role.to_numpy(),
                "task": [labels.csv_tasks[i] for i in roles.label_idx],
                "label": roles.label_idx.to_numpy(),
            }
        ).sort_values("take_name", kind="mergesort")
        write_csv(csv_out, csv_path)
    digest = file_sha256(csv_path)

    # la mappa sito -> fold si calcola qui, una volta, e finisce in split.json (D6)
    train = roles[roles.role == "train"]
    sizes = train.site.value_counts().to_dict()
    classes = {s: set(g) for s, g in train.label_idx.groupby(train.site)}
    site_to_fold = plan_site_to_fold(sizes, classes, args.n_folds)
    folds = make_folds(train, args.n_folds, args.inner_holdout_frac, seed=0, group_by="site", site_to_fold=site_to_fold)
    diag = fold_diagnostics(folds, labels.names)
    write_csv(folds, out / "folds.csv")

    names = {i: n for i, n in enumerate(labels.names)}
    by_role = {
        role: roles[roles.role == role].assign(classe=lambda d: d.label_idx.map(names))
        for role in ("train", "val", "test")
    }
    parts = ["# Split per sito (site_v1)\n"]
    parts.append(
        f"- val: {val_sites} - test: {test_sites}\n"
        f"- take: train {len(by_role['train'])}, val {len(by_role['val'])}, test {len(by_role['test'])}\n"
        f"- criteri: classe valutata se >= {args.min_eval_takes} take in val/test; servono >= "
        f"{args.min_train_sites} siti di train con >= {args.min_takes_per_site} take della classe\n"
        f"- accuratezza di 'classe piu' frequente del sito' sul train (in-sample): "
        f"{site_majority_accuracy(roles):.3f}\n"
        f"- problemi di copertura: {problems or 'nessuno'}\n"
        f"- mappa sito -> fold (split.json): {dict(sorted(site_to_fold.items(), key=lambda kv: (kv[1], kv[0])))}\n"
    )
    for role, d in by_role.items():
        if len(d):
            parts.append(_block(f"{role}: sito x classe", pd.crosstab(d.site, d.classe, margins=True)))
    parts.append(_block("copertura per classe", cov.set_index("classe")))
    parts.append(_block("fold del train (per sito)", diag.set_index("fold")))
    write_text(out / "report.md", "\n".join(parts))
    write_json(
        out / "split.json",
        {
            "val_sites": val_sites,
            "test_sites": test_sites,
            "n_folds": args.n_folds,
            "inner_holdout_frac": args.inner_holdout_frac,
            "min_eval_takes": args.min_eval_takes,
            "min_train_sites": args.min_train_sites,
            "min_takes_per_site": args.min_takes_per_site,
            "n_take": {r: int(len(d)) for r, d in by_role.items()},
            "scenario_site_sha256": digest,
            "site_to_fold": dict(sorted(site_to_fold.items(), key=lambda kv: (kv[1], kv[0]))),
            "problems": problems,
        },
    )
    # SHA256SUMS per ultimo: scenario_site.csv per primo (hpc/ ne usa la prima riga come hash dello split)
    write_text(out / "SHA256SUMS", "".join(f"{file_sha256(out / n)}  {n}\n" for n in SUMS))
    print(f"scritti split.json, folds.csv, report.md e SHA256SUMS in {out}")
    print(json.dumps({r: int(len(d)) for r, d in by_role.items()}))


if __name__ == "__main__":
    main()
