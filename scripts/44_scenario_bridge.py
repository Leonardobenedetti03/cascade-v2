"""44 — Ponte da scenario_preds alla cache del gate (CPU). Sostituisce lo stadio 45 (N28).

Scopo:
    Costruisce ``45_cache/<variante>/cache.parquet`` (stesso schema, stesso manifest ``45_cache``,
    quindi lo stadio 50 non cambia) dalle uscite di una run di scenario_preds: L0 = CamFormer
    congelato + testa lineare, L1 = EgoVLP congelato + testa lineare, una riga per take.
    - probabilità: ``cam_p_*`` e ``vid_p_*`` di predictions.csv (già calibrate con la temperatura
      della testa); logit = log delle probabilità, ``T_cam = T_vid = 1``;
    - fusione ``replace`` (M05): ``fused_probs`` = L1, ``w = 0``;
    - train: probabilità fuori campione (media dei seed di scenario_preds), ``fold`` = ``fold_s0``;
      val/test: media dei modelli di fold di tutti i seed, ``fold = -1``;
    - embedding: ``features/camformer.npz`` e ``features/egovlp.npz``;
    - moto ``mot_*``: dalle traiettorie ``traj/<take>.npz`` (solo campioni validi) nelle stesse
      finestre usate per le embedding, mediate sulle finestre;
    - costi: stadio 40 × numero di finestre per take (L0 e L1 girano su ogni finestra).
    Take senza embedding CamFormer (audit punto 3): fuori dalla cache, elencate in ``no_l0.csv``;
    nel risultato finale vanno a EgoVLP e si contano come costo L0 + L1 (stadio 52 / report).
    Take senza L1 o fuori dalle 6 classi (Bouldering): fuori, contate nel summary.
    Il test entra solo con ``--include-test`` (da ``sealed/predictions.csv``), al passo finale.

Input:
    --scenario-run DIR (run di scenario_preds completa fino a heads), 40_cost/hz<F>/cost.json

Output:
    <run_dir>/45_cache/<variante>/cache.parquet, summary.json, no_l0.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.cascade.cache.schema import add_labels, validate_cache  # noqa: E402
from src.cascade.cli import open_stage, stage_parser  # noqa: E402
from src.cascade.io import read_json, write_json  # noqa: E402
from src.cascade.scenario_io import compute_windows, load_npz, take_motion  # noqa: E402,F401
from src.data.cache import Table, write_table  # noqa: E402

STAGE = "45_cache"  # stesso manifest dello stadio 45: lo stadio 50 lo richiede invariato
EPS = 1e-12


def build(scen_dir: Path, ctx, include_test: bool, gmacs: Tuple[float, float]):
    cfg, labels = ctx.cfg, ctx.labels
    scfg = json.loads((scen_dir / "configs" / "heads.json").read_text())
    if list(scfg["labels"]["classes"]) != list(labels.names):
        raise ValueError(f"classi di scenario_preds {scfg['labels']['classes']} != labels.yaml {list(labels.names)}")
    if cfg.fusion.mode != "replace":
        raise ValueError("lo stadio 44 supporta solo fusion.mode = replace (M05)")
    pred = pd.read_csv(scen_dir / "predictions.csv", dtype={"take_uid": str})
    if (pred.split == "test").any():
        raise RuntimeError("predictions.csv contiene righe del test: il test sta solo in sealed/")
    if include_test:
        test = pd.read_csv(scen_dir / "sealed" / "predictions.csv", dtype={"take_uid": str})
        pred = pd.concat([pred, test[test.split == "test"]], ignore_index=True)
    idx = pd.read_csv(scen_dir / "index" / "takes.csv", dtype={"take_uid": str})[["take_name", "duration_s"]]
    pred = pred.merge(idx, on="take_name", how="left", validate="one_to_one")
    names = list(labels.names)
    cam_cols, vid_cols = [f"cam_p_{n}" for n in names], [f"vid_p_{n}" for n in names]
    in_head = pred.label.isin(names).to_numpy()
    P0, P1 = pred[cam_cols].to_numpy(float), pred[vid_cols].to_numpy(float)
    has0, has1 = np.isfinite(P0).all(1), np.isfinite(P1).all(1)
    counts = {
        "n_scenario": int(len(pred)),
        "n_out_of_head": int((~in_head).sum()),
        "n_no_l1": int((in_head & ~has1).sum()),
    }
    no_l0 = pred[in_head & has1 & ~has0].copy()
    no_l0["label_idx"] = no_l0.label.map({n: i for i, n in enumerate(names)}).astype(int)
    no_l0["c1"] = (no_l0[vid_cols].to_numpy(float).argmax(1) == no_l0.label_idx.to_numpy()).astype(int)
    no_l0 = no_l0[["take_name", "take_uid", "split", "label", "label_idx", "c1"]]
    keep = in_head & has0 & has1
    d = pred[keep].reset_index(drop=True)
    P0, P1 = P0[keep], P1[keep]
    P0, P1 = P0 / P0.sum(1, keepdims=True), P1 / P1.sum(1, keepdims=True)
    label_idx = d.label.map({n: i for i, n in enumerate(names)}).astype(int).to_numpy()
    K = cfg.folds.n_folds
    fold = np.where(d.split == "train", d.fold_s0.fillna(-1).astype(int), -1)
    if ((d.split == "train").to_numpy() & ((fold < 0) | (fold >= K))).any():
        raise ValueError(f"righe di train senza fold_s0 in [0, {K}): controllare split.n_folds di scenario_preds")
    cam_emb, vid_emb = load_npz(scen_dir / "features" / "camformer.npz"), load_npz(scen_dir / "features" / "egovlp.npz")
    E0 = np.stack([cam_emb[n] for n in d.take_name]).astype(np.float32)
    E1 = np.stack([vid_emb[n] for n in d.take_name]).astype(np.float32)
    w = scfg["window"]
    n_win = int(w["n_windows"])
    mot = [
        take_motion(
            scen_dir / "traj" / f"{r.take_name}.npz",
            compute_windows(float(r.duration_s), float(w["length_s"]), n_win, w["placement"]),
            float(scfg["poses"]["fps"]),
            int(cfg.data.vertical_axis),
        )
        for r in d.itertuples(index=False)
    ]
    df = pd.DataFrame(
        {
            "clip_id": d.take_name.astype(str),
            "take_uid": d.take_uid.astype(str),
            "split": d.split.astype(str),
            "fold": fold.astype(int),
            "label_idx": label_idx,
            "T_cam": 1.0,
            "T_vid": 1.0,
            "w": 0.0,
            "gmacs_L0": gmacs[0] * n_win,
            "gmacs_L1": gmacs[1] * n_win,
        }
    )
    for col in ("session", "site", "camformer_pretrain"):
        if col in d.columns:
            df[col] = d[col].to_numpy()
    df = pd.concat([df, pd.DataFrame(mot)], axis=1)
    arrays = {
        "cam_probs_cal": P0.astype(np.float32),
        "cam_logits": np.log(np.clip(P0, EPS, 1.0)).astype(np.float32),
        "vid_probs_cal": P1.astype(np.float32),
        "vid_logits": np.log(np.clip(P1, EPS, 1.0)).astype(np.float32),
        "fused_probs": P1.astype(np.float32),
        "cam_emb": E0,
        "vid_cls": E1,
    }
    t = Table(df, arrays, {"class_names": names, "cam_train_source": "oof"})
    add_labels(t)
    t.meta.update(
        {
            "stage": "44_scenario_bridge",
            "source": "scenario_preds",
            "scenario_run": str(scen_dir),
            "n_windows": n_win,
            "json_cam_to_device": json.dumps(scfg["poses"]["json_cam_to_device"]),
            "fusion_mode": "replace",
            "cam_seed": ctx.cam_seed,
        }
    )
    return t, no_l0, counts, n_win


def main(argv: Optional[List[str]] = None) -> None:
    p = stage_parser(__doc__)
    p.add_argument("--scenario-run", type=Path, required=True, help="run_dir di scenario_preds (fino a heads)")
    p.add_argument("--include-test", action="store_true", help="aggiunge il test da sealed/ (solo al passo finale)")
    args = p.parse_args(argv)
    ctx = open_stage(args, STAGE, make_stage_dir=False)
    out_dir = ctx.variant_dir(STAGE)
    outs = [out_dir / "cache.parquet", out_dir / "summary.json", out_dir / "no_l0.csv"]
    part = ctx.variant_tag
    if ctx.done(outs, part=part):
        return
    ctx.require("40_cost", parts=[ctx.traj_tag])
    cost = read_json(ctx.run_dir / "40_cost" / ctx.traj_tag / "cost.json")
    scen = args.scenario_run.expanduser().resolve()
    t, no_l0, counts, n_win = build(scen, ctx, args.include_test, (float(cost["gmacs_L0"]), float(cost["gmacs_L1"])))
    warnings = validate_cache(t, ctx.labels, cam_dim=t.arrays["cam_emb"].shape[1], vid_dim=t.arrays["vid_cls"].shape[1])
    for w in warnings:
        ctx.log.warning("schema: %s", w)
    write_table(t, outs[0])
    no_l0_out = no_l0 if args.include_test else no_l0[no_l0.split != "test"]
    no_l0_out.to_csv(outs[2], index=False)
    summ = {}
    for s in ("train", "val", "test"):
        m = (t.df.split == s).to_numpy()
        n_no = int((no_l0.split == s).sum())
        if not m.any() and not n_no:
            continue
        summ[s] = {"n": int(m.sum()), "n_no_l0": n_no}
        if s != "test":  # il test non si guarda fuori da 50 --final
            y = t.df.label_idx.to_numpy()[m]
            summ[s].update(
                {
                    "acc_L0": float(t.df.c0[m].mean()),
                    "acc_L1": float((t.arrays["vid_probs_cal"][m].argmax(1) == y).mean()),
                    "y_rate": float(t.df.y[m].mean()),
                }
            )
    write_json(
        outs[1],
        {"splits": summ, "counts": counts, "n_windows": n_win, "scenario_run": str(scen), "warnings": warnings},
    )
    ctx.finish(outs, part=part, extra={"scenario_run": str(scen), "include_test": bool(args.include_test)})
    ctx.log.info("cache da scenario_preds: %d take -> %s; %s; %s", len(t), outs[0], summ, counts)


if __name__ == "__main__":
    main()
