"""00 — Inventario delle take e indice delle clip.

Scopo:
    Legge etichette e split ufficiali (scenario_cls.csv di CamFormer) e i metadati
    delle take, verifica per ogni take la presenza di video ego e traiettoria e
    calcola le finestre (durata D = data.window_s, centrate sul centro della take).

Input:
    paths.scenario_csv (SHA256 controllato contro paths.split_sha256), paths.takes_json,
    video e traiettorie sotto paths.takes_root

Output (nella run_dir):
    data/clips.csv          clip_id, take_uid, split, label, label_idx, t_center,
                            t_start, t_end, video_path, traj_path, has_video, has_traj, ...
    data/missing_takes.csv  take con video/traiettoria/durata mancanti o troppo corte
    data/inventory_summary.json

Con --inspect stampa la struttura delle directory e qualche file d'esempio
(primo avvio sull'HPC) e non scrive nulla.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cascade import paths  # noqa: E402
from src.cascade.cli import open_stage, stage_parser  # noqa: E402
from src.cascade.clips import build_inventory, inspect_layout  # noqa: E402
from src.cascade.io import write_csv, write_json  # noqa: E402
from src.cascade.site_split import check_split_hash  # noqa: E402

STAGE = "00_inventory"


def main(argv: Optional[List[str]] = None) -> None:
    p = stage_parser(__doc__)
    p.add_argument("--inspect", action="store_true", help="stampa il layout dei dati ed esce")
    args = p.parse_args(argv)
    ctx = open_stage(args, STAGE, make_stage_dir=False)
    if args.inspect:
        print(inspect_layout(ctx))
        return
    scenario_csv = paths.resolve(ctx.cfg.paths.scenario_csv, data=ctx.data_root, run=ctx.run_dir)
    split_sha = check_split_hash(scenario_csv, ctx.cfg.paths.split_sha256)  # ValueError se lo split è cambiato
    ctx.log.info("split %s: SHA256 %s", scenario_csv.name, split_sha)
    out_dir = ctx.run_dir / "data"
    outs = [out_dir / "clips.csv", out_dir / "missing_takes.csv", out_dir / "inventory_summary.json"]
    if ctx.done(outs):
        return
    clips, missing, summary = build_inventory(ctx)
    write_csv(clips, outs[0])
    write_csv(missing, outs[1])
    write_json(outs[2], summary)
    ctx.finish(outs)
    ctx.log.info(
        "clip: %d (utilizzabili %d), take con problemi: %d",
        summary["clips"],
        summary["clips_usable"],
        summary["takes_with_problems"],
    )
    ctx.log.info("scritto %s", outs[0])


if __name__ == "__main__":
    main()
