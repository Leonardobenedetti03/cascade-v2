"""35 — Fusione L0 + L1 al livello video (CPU).

Scopo:
    - fusion.mode = replace (default, M05): w = 0, cioè L1 sostituisce L0;
    - fusion.mode = fuse (ablation): log p_fusa = w · log p_cam + (1 − w) · log p_vid, con w
      scelto su una griglia in [0, 1] (passo fusion.w_grid_step) usando SOLO le uscite fuori
      campione del train (OOF di CamFormer e delle teste L1): massima accuracy, a parità
      minima NLL.
    In ogni caso scrive il confronto tra L0, L1 da solo e fusione log-lineare su train OOF e
    val, con IC della differenza rispetto a L1: la fusione va dichiarata solo se migliora
    davvero L1.

Input:
    <run_dir>/20_camformer/<cam>/fold*/preds_oof.parquet, <run_dir>/30_l1_heads/probe/preds.parquet

Output (<run_dir>/35_fusion/<variante>/, variante = seed<S>_hz<F>):
    w.json (w, modo, griglia accuratezza/NLL), fusion_comparison.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.cascade.cache.build import assemble_camformer, join_l0_l1, load_l1  # noqa: E402
from src.cascade.cli import open_stage, stage_parser  # noqa: E402
from src.cascade.fusion import choose_w, compare_fusions  # noqa: E402
from src.cascade.io import write_csv, write_json  # noqa: E402

STAGE = "35_fusion"


def main(argv: Optional[List[str]] = None) -> None:
    p = stage_parser(__doc__)
    args = p.parse_args(argv)
    ctx = open_stage(args, STAGE, make_stage_dir=False)
    cfg = ctx.cfg
    out_dir = ctx.variant_dir(STAGE)
    outs = [out_dir / "w.json", out_dir / "fusion_comparison.csv"]
    part = ctx.variant_tag
    if ctx.done(outs, part=part):
        return
    ctx.require("20_camformer", parts=[f"{ctx.cam_tag}.fold{k}" for k in range(cfg.folds.n_folds)])
    ctx.require("30_l1_heads", parts=[ctx.l1_tag])
    folds = pd.read_csv(ctx.run_dir / "splits" / "folds.csv")
    j = join_l0_l1(
        assemble_camformer(ctx.seed_dir("20_camformer"), cfg, ctx.labels, folds),
        load_l1(ctx.l1_dir(), ctx.labels, folds),
    )
    tr = (j.df.split == "train").to_numpy()
    y = j.df.label_idx.to_numpy()[tr]
    w, grid = choose_w(
        j.arrays["cam_probs_cal"][tr],
        j.arrays["vid_probs_cal"][tr],
        y,
        cfg.fusion.w_grid_step,
        cfg.fusion.mode,
    )
    best = [g for g in grid if g["w"] == w][0]
    comp = compare_fusions(j, w, n_boot=cfg.gate.bootstrap, seed=cfg.gate.bootstrap_seed)
    write_csv(comp, out_dir / "fusion_comparison.csv")
    write_json(
        out_dir / "w.json",
        {
            "w": w,
            "mode": cfg.fusion.mode,
            "n_train_oof": int(tr.sum()),
            "grid": grid,
            "acc_at_w": best["acc"],
            "nll_at_w": best["nll"],
        },
    )
    ctx.finish(outs, part=part)
    ctx.log.info("w = %.2f (accuratezza OOF train %.4f, NLL %.4f) su %d clip", w, best["acc"], best["nll"], tr.sum())
    ctx.log.info("confronto delle fusioni (train OOF e val):\n%s", comp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
