"""01 — Conversione delle pose per frame in traiettorie uniformi a 20 Hz (M08).

Scopo:
    Porta nel repository la conversione che nella v1 girava fuori (hpc/make_traj_data.py): per ogni take
    dello split legge le pose della camera RGB di Aria (matrici mondo -> camera, circa 30 Hz, un file JSON
    per take) e scrive una traiettoria camera -> mondo su griglia uniforme a data.traj_fps, nel tempo del
    video. Dettagli e differenze dalla v1: src/cascade/traj_convert.py.

    Non dipende da altri stadi: l'elenco delle take viene da paths.scenario_csv (SHA256 controllato contro
    paths.split_sha256) e le pose dalle radici paths.pose_roots. Va lanciato prima dell'inventario (00), che
    cerca le traiettorie convertite.

Input:
    paths.scenario_csv, paths.pose_roots (ognuna con split_manifest.csv, annotations/ego_pose/...,
    takes/<take>/frame_aligned_videos/... per scegliere la camera del video usato)

Output (nella run_dir):
    <paths.traj_dir>/<take_name>.npz   t [N] s, poses [N, 7] (tx ty tz qx qy qz qw), valid [N], fps, src_fps
    01_traj/status.csv                 una riga per take: esito, camera, buchi, righe senza posa, durata
    01_traj/summary.json               conteggi per esito e frazione di take perse

Si ferma, senza scrivere il manifest, se le take perse superano data.max_missing_frac.
Con --take o --limit converte solo quelle take, stampa una riga per take e non scrive il manifest (prove).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cascade import paths  # noqa: E402
from src.cascade.cli import open_stage, stage_parser  # noqa: E402
from src.cascade.io import write_csv, write_json  # noqa: E402
from src.cascade.site_split import check_split_hash  # noqa: E402
from src.cascade.traj_convert import (  # noqa: E402
    OK,
    STATUS_FIELDS,
    convert_take,
    load_pose_index,
    summarize,
    write_npz,
)

STAGE = "01_traj"


def _npz_complete(out_dir: Path, status_csv: Path) -> bool:
    """Tutte le traiettorie delle take convertite (status ok) sono ancora su disco?"""
    import pandas as pd

    st = pd.read_csv(status_csv)
    return all((out_dir / f"{take}.npz").exists() for take in st.loc[st["status"] == OK, "take_name"])


def main(argv: Optional[List[str]] = None) -> None:
    import pandas as pd

    p = stage_parser(__doc__)
    p.add_argument("--take", action="append", default=[], help="solo questa take (ripetibile; prova, niente manifest)")
    p.add_argument("--limit", type=int, default=0, help="solo le prime N take (prova, niente manifest)")
    args = p.parse_args(argv)
    ctx = open_stage(args, STAGE)
    cfg = ctx.cfg
    if ctx.synthetic:
        raise SystemExit("lo stadio 01 non gira su dati sintetici (le pose finte arrivano in C07b)")
    debug = bool(args.take or args.limit)

    def resolve(template: str) -> Path:
        return paths.resolve(template, data=ctx.data_root, run=ctx.run_dir)

    scenario_csv = resolve(cfg.paths.scenario_csv)
    ctx.log.info("split %s: SHA256 %s", scenario_csv.name, check_split_hash(scenario_csv, cfg.paths.split_sha256))
    if not cfg.paths.pose_roots:
        raise SystemExit("paths.pose_roots è vuoto: indicare le radici del dataset con split_manifest.csv (TODO(HPC))")
    out_dir = resolve(cfg.paths.traj_dir)
    status_csv, summary_json = ctx.stage_dir / "status.csv", ctx.stage_dir / "summary.json"
    outs = [status_csv, summary_json]
    if not debug and ctx.done(outs):
        if _npz_complete(out_dir, status_csv):
            return
        ctx.log.warning("mancano alcuni .npz in %s: rifaccio la conversione", out_dir)

    takes = sorted(set(pd.read_csv(scenario_csv, usecols=["take_name"])["take_name"]))
    if args.take:
        takes = [t for t in takes if t in set(args.take)]
    if args.limit:
        takes = takes[: args.limit]
    index = load_pose_index([resolve(r) for r in cfg.paths.pose_roots])
    ctx.log.info(
        "take da convertire: %d (pose trovate in %d radici per %d take)",
        len(takes),
        len(cfg.paths.pose_roots),
        len(index),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        video_pattern=cfg.paths.video_pattern,
        video_glob=cfg.paths.video_glob,
        src_fps=cfg.data.pose_src_fps,
        out_fps=cfg.data.traj_fps,
        max_gap_frames=cfg.data.pose_max_gap_frames,
    )
    rows = []
    for i, take in enumerate(takes, 1):
        conv, row = convert_take(take, index, **kwargs)
        if conv is not None:
            write_npz(out_dir / f"{take}.npz", conv, out_fps=cfg.data.traj_fps, src_fps=cfg.data.pose_src_fps)
        rows.append(row)
        if debug:
            ctx.log.info(
                "%-32s %-12s cam=%s n_out=%s dur=%ss gaps=%s dropped=%s lead=%s",
                take,
                row["status"],
                row["cam"],
                row["n_out"],
                row["dur_s"],
                row["gaps"],
                row["dropped"],
                row["lead_fill"],
            )
        elif i % 100 == 0:
            ctx.log.info("%d/%d", i, len(takes))
    summary = summarize(rows, len(takes))
    summary.update(
        traj_fps=cfg.data.traj_fps,
        pose_src_fps=cfg.data.pose_src_fps,
        pose_max_gap_frames=cfg.data.pose_max_gap_frames,
        pose_roots=list(cfg.paths.pose_roots),
    )
    ctx.log.info("esiti: %s", summary["status_counts"])
    ctx.log.info(
        "take con righe prima della prima posa: %d | con buchi: %d",
        summary["takes_with_lead_fill"],
        summary["takes_with_gaps"],
    )
    if debug:
        return
    write_csv(pd.DataFrame(rows, columns=STATUS_FIELDS), status_csv)
    write_json(summary_json, summary)
    if summary["missing_frac"] > cfg.data.max_missing_frac:
        raise SystemExit(
            f"take perse {summary['missing_frac']:.1%} > data.max_missing_frac={cfg.data.max_missing_frac:.1%}: "
            f"vedere {status_csv}"
        )
    ctx.finish(outs, extra={"status_counts": summary["status_counts"]})
    ctx.log.info("scritto %s", status_csv)


if __name__ == "__main__":
    main()
