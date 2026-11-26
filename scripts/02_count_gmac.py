"""Stage 40: cost per take of the two levels, as GMACs (fvcore) and latency. [GPU or CPU]

What it does:
    GMACs from fvcore (plus a handle for SDPA attention), batch 1, a single view, the same input as at
    inference, next to the torch counter and the analytic formula. CamFormer runs on the context
    (camformer.context_s, or the window) at data.traj_fps plus the gravity token; TimeSformer-EgoVLP on
    4 frames at 224 px (and cost.timesformer_frames, e.g. 8 and 16). Mean latency on the available device
    (GPU if there is one).
    Expected values, as a sanity check (cost.expected): CamFormer ~0.12 GMACs (8 s at 20 Hz),
    TimeSformer-B 4×224 ~90 GMACs. A gap larger than 3× is reported in the log.

Input:
    the configuration (architectures); no data

Output:
    <run_dir>/40_cost/hz<F>/cost.json   (CamFormer's cost depends on the trajectory rate)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cascade.utils.cli import open_stage, stage_parser  # noqa: E402
from src.cascade.utils.io import write_json  # noqa: E402

STAGE = "40_cost"


def main(argv: Optional[List[str]] = None) -> None:
    p = stage_parser(__doc__)
    args = p.parse_args(argv)
    ctx = open_stage(args, STAGE)
    out = ctx.stage_dir / ctx.traj_tag / "cost.json"
    if ctx.done([out], part=ctx.traj_tag):
        return
    import copy

    import torch

    from src.cascade.evaluation.cost import camformer_macs, measure_gmacs, measure_latency_ms, timesformer_macs
    from src.cascade.models.gmac.camformer import INPUT_DIM, build_model, encode_pose_id, resolve_device
    from src.cascade.models.gmac.egovlp import build_video_model

    cfg = ctx.cfg
    dev = resolve_device(cfg.cost.device)
    res = {"device": str(dev)}
    # CamFormer
    ctx_s = cfg.camformer.context_s or cfg.data.window_s
    T = int(round(ctx_s * cfg.data.traj_fps))
    cam = build_model(cfg.camformer.arch, ctx.labels.num_classes).to(dev)
    x = torch.zeros(1, T, INPUT_DIM[encode_pose_id(cfg.camformer.arch.pose_encoding)], device=dev)
    inputs = (
        x,
        torch.zeros(1, T, dtype=torch.bool, device=dev),
        torch.tensor([T], device=dev),
        torch.tensor([0], device=dev),
        torch.tensor([T], device=dev),
    )
    g = measure_gmacs(
        cam, inputs, cfg.cost.counter, analytic_macs=camformer_macs(T, cfg.camformer.arch, ctx.labels.num_classes)
    )
    res["camformer"] = {**g, "tokens": T, "context_s": ctx_s, **measure_latency_ms(cam, inputs, cfg.cost.latency_reps)}
    # TimeSformer
    res["timesformer"] = {}
    a = cfg.egovlp.arch
    frames = sorted(set([a.num_frames] + list(cfg.cost.timesformer_frames)))
    for nf in frames:
        arch = copy.deepcopy(a)
        arch.num_frames = nf
        m = build_video_model(arch).to(dev)
        xi = (torch.zeros(1, nf, 3, a.img_size, a.img_size, device=dev),)
        res["timesformer"][str(nf)] = {
            **measure_gmacs(m, xi, cfg.cost.counter, analytic_macs=timesformer_macs(arch, nf)),
            **measure_latency_ms(m, xi, max(1, cfg.cost.latency_reps // 4)),
        }
        del m
    # One-frame baseline (B1)
    if cfg.frame_baseline.enabled:
        from src.cascade.models.gmac.frame_baseline import build_frame_model, pixels_read

        fb = cfg.frame_baseline
        fm = build_frame_model(fb.arch).to(dev)
        xf = (torch.zeros(1, 3, fb.size, fb.size, device=dev),)
        res["frame_baseline"] = {
            **measure_gmacs(fm, xf, cfg.cost.counter),
            **measure_latency_ms(fm, xf, cfg.cost.latency_reps),
            "size": fb.size,
        }
        res["gmacs_Lhalf"] = res["frame_baseline"]["gmacs"]
        res["pixels_Lhalf"] = pixels_read(fb.size, 1)
    res["pixels_L0"] = 0
    res["pixels_L1"] = int(a.num_frames * a.img_size * a.img_size * 3)
    res["gmacs_L0"] = res["camformer"]["gmacs"] + cfg.cost.gate_gmacs
    res["gmacs_L1"] = res["timesformer"][str(a.num_frames)]["gmacs"]
    res["gate_gmacs"] = cfg.cost.gate_gmacs
    if cfg.cost.override_gmacs:
        res["measured_gmacs_L0"], res["measured_gmacs_L1"] = res["gmacs_L0"], res["gmacs_L1"]
        res["gmacs_L0"] = float(cfg.cost.override_gmacs["L0"]) + cfg.cost.gate_gmacs
        res["gmacs_L1"] = float(cfg.cost.override_gmacs["L1"])
        if "Lhalf" in cfg.cost.override_gmacs and "gmacs_Lhalf" in res:
            res["measured_gmacs_Lhalf"] = res["gmacs_Lhalf"]
            res["gmacs_Lhalf"] = float(cfg.cost.override_gmacs["Lhalf"])
        ctx.log.warning(
            "cost.override_gmacs attivo: L0=%.4g, L1=%.4g (misurati salvati a parte)", res["gmacs_L0"], res["gmacs_L1"]
        )
    for key, val in (("camformer", res["camformer"]["gmacs"]), (f"timesformer_{a.num_frames}f", res["gmacs_L1"])):
        exp = (cfg.cost.expected or {}).get(key)
        if exp:
            ratio = val / exp
            res.setdefault("sanity", {})[key] = {"measured": val, "expected": exp, "ratio": ratio}
            (ctx.log.warning if not 1 / 3 <= ratio <= 3 else ctx.log.info)(
                "sanity %s: %.4g GMACs misurati vs ~%.4g attesi (x%.2f)", key, val, exp, ratio
            )
    write_json(out, res)
    ctx.finish([out], part=ctx.traj_tag)
    ctx.log.info("GMACs: L0=%.4g, L1=%.4g (%s)", res["gmacs_L0"], res["gmacs_L1"], res["camformer"]["method"])


if __name__ == "__main__":
    main()
