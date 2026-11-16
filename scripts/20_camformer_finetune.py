"""20 — Fine-tuning di CamFormer (L0) per una fold. [GPU]

Scopo:
    Carica il checkpoint pre-addestrato (paths.camformer_ckpt, egoexo4d_dur8.pt),
    aggiunge la testa lineare e, con camformer.freeze_encoder (M17, default), allena
    solo la testa con l'encoder congelato; senza, fine-tuning end-to-end. Early stopping
    e temperature scaling sull'holdout interno (escluso dal training). Encoder e preprocessing sono quelli del submodule CamFormer.
    Il seed del fine-tuning è camformer.seed (default: seed globale); seed diversi
    scrivono in cartelle diverse, per gli stadi stocastici ripetuti (3 seed).

Input:
    <run_dir>/data/clips.csv, <run_dir>/splits/folds.csv, traiettorie delle take

Output (<run_dir>/20_camformer/seed<S>_hz<F>/fold<k>/ oppure full/):
    model.pt                      checkpoint (encoder + testa + T)
    preds_oof.parquet             clip della fold k, fuori campione (non per full)
    preds_val.parquet, preds_test.parquet
        colonne: clip_id, take_uid, split, label_idx, T_cam, cam_fold;
        vettori cam_logits[8], cam_probs_cal[8], cam_emb[512] (+ MC dropout se attivo)
    roles.csv                     ruolo di ogni clip (train / holdout / oof / val / test)
    history.csv, summary.json     (metriche solo su holdout, OOF e val: mai sul test)
    ckpt_last.pt                  (temporaneo) stato dell'ultima epoca: un job interrotto riprende da qui;
                                  sparisce quando lo stadio scrive il manifest

Job array SLURM: --fold $SLURM_ARRAY_TASK_ID con 0..4, e "full" a parte.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.cascade.cli import open_stage, stage_parser  # noqa: E402
from src.cascade.folds import fold_names, training_masks  # noqa: E402
from src.cascade.io import torch_save, write_csv, write_json  # noqa: E402
from src.cascade.manifest import STAGE_KEYS, config_subset  # noqa: E402
from src.cascade.seed import set_seed  # noqa: E402

STAGE = "20_camformer"


def _ckpt_key(ctx, fold: str, seed: int) -> str:
    """Identità di un checkpoint di ripresa: chiavi di config dello stadio + fold + seed."""
    subset = {"cfg": config_subset(ctx.cfg_dict, STAGE_KEYS[STAGE]), "fold": fold, "seed": seed}
    return hashlib.sha256(json.dumps(subset, sort_keys=True, default=str).encode()).hexdigest()[:16]


def role_datasets(ctx, fold: str, *, prefix_frac: float = 1.0, roles: Optional[List[str]] = None) -> tuple:
    """Dataset per ruolo (train / holdout / oof / val / test) della fold, e l'indice delle clip.

    ``roles`` limita i ruoli costruiti; ``prefix_frac < 1`` tronca le finestre (stadio 22).
    """
    from src.data.trajectories import TrajectoryWindowDataset

    clips = ctx.read_clips()
    folds = pd.read_csv(ctx.run_dir / "splits" / "folds.csv")
    tr_m, ho_m, ev_m = training_masks(folds, fold)
    takes = {r: set(folds.take_uid[m]) for r, m in (("train", tr_m), ("holdout", ho_m), ("oof", ev_m))}
    is_train = clips.split == "train"
    masks = {r: clips.take_uid.isin(t) & is_train for r, t in takes.items()}
    masks.update({"val": clips.split == "val", "test": clips.split == "test"})
    cache_dir = ctx.path(ctx.cfg.paths.traj_cache_dir)
    sets = {
        r: TrajectoryWindowDataset(
            clips[m].reset_index(drop=True), ctx.cfg, cache_dir=cache_dir, train=r == "train", prefix_frac=prefix_frac
        )
        for r, m in masks.items()
        if roles is None or r in roles
    }
    return sets, clips


def load_model(ctx):
    """CamFormer con pesi pre-addestrati (o casuali); restituisce (modello, info sul caricamento)."""
    from src.cascade.models.camformer import build_model, check_head_class_order, load_pretrained

    cfg = ctx.cfg
    model = build_model(cfg.camformer.arch, ctx.labels.num_classes)
    if not cfg.camformer.pretrained:
        ctx.log.warning("camformer.pretrained=false: pesi casuali")
        return model, {}
    ckpt = ctx.path(cfg.paths.camformer_ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint CamFormer mancante: {ckpt} (o camformer.pretrained=false)")
    if cfg.camformer.init_head_from_ckpt:
        check_head_class_order(ctx.labels)
    ctx.log.info("pesi pre-addestrati da %s", ckpt)
    return model, load_pretrained(model, ckpt, init_head=cfg.camformer.init_head_from_ckpt)


def write_predictions(
    ctx, model, sets, clips, fold: str, T: float, device, out, mc_passes: Optional[int] = None
) -> dict:
    """preds_{oof,val,test}.parquet; accuratezza solo per OOF e val (mai per il test)."""
    from src.cascade.finetune import predict, preds_table
    from src.data.cache import write_table

    mc = ctx.cfg.camformer.mc_dropout_passes if mc_passes is None else mc_passes
    report = {}
    for role in ("oof", "val", "test"):
        if role == "oof" and fold == "full":
            continue
        d = sets[role]
        pred = predict(model, d, device, ctx.cfg.camformer.batch_size_eval, mc)
        t = preds_table(d, pred, T, clips, ctx.labels, fold=fold, source="oof" if role == "oof" else "eval")
        write_table(t, out / f"preds_{role}.parquet")
        report[role] = {"n": len(t)}
        if role != "test" and len(t):  # il test non si guarda fuori da 50 --final
            report[role]["acc"] = float((t.arrays["cam_probs_cal"].argmax(1) == t.df.label_idx.to_numpy()).mean())
    return report


def main(argv: Optional[List[str]] = None) -> None:
    p = stage_parser(__doc__)
    p.add_argument("--fold", type=str, required=True, help="0..K-1 oppure full")
    args = p.parse_args(argv)
    ctx = open_stage(args, STAGE, make_stage_dir=False)
    cfg = ctx.cfg
    if args.fold not in fold_names(cfg.folds.n_folds):
        raise ValueError(f"--fold deve essere in {fold_names(cfg.folds.n_folds)}")
    ctx.require("00_inventory")
    ctx.require("05_folds")
    seed = ctx.cam_seed
    set_seed(seed)
    name = "full" if args.fold == "full" else f"fold{args.fold}"
    out, part = ctx.seed_dir(STAGE) / name, f"{ctx.cam_tag}.{name}"
    files = ["model.pt", "preds_val.parquet", "preds_test.parquet", "roles.csv", "history.csv", "summary.json"]
    outputs = [out / f for f in files] + ([out / "preds_oof.parquet"] if args.fold != "full" else [])
    if ctx.done(outputs, part=part):
        return
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt_last.pt"  # ripresa per epoca: un job ucciso riparte dall'ultima epoca salvata
    ckpt_key = _ckpt_key(ctx, args.fold, seed)

    from src.cascade.finetune import train_fold
    from src.cascade.models.camformer import resolve_device

    sets, clips = role_datasets(ctx, args.fold)
    ctx.log.info("fold %s (seed %d): %s clip", args.fold, seed, {r: len(d) for r, d in sets.items()})
    device = resolve_device(cfg.camformer.device)
    model, loaded = load_model(ctx)
    fit = train_fold(model, sets["train"], sets["holdout"], cfg, device, seed, ckpt_path=ckpt_path, ckpt_key=ckpt_key)
    ctx.log.info(
        "temperatura T=%.3f (NLL holdout %.4f -> %.4f)", fit["T"], fit["holdout_nll_raw"], fit["holdout_nll_cal"]
    )
    torch_save(
        {
            "state_dict": model.state_dict(),
            "T": fit["T"],
            "fold": args.fold,
            "seed": seed,
            "class_names": list(ctx.labels.names),
            "arch": cfg.to_dict()["camformer"]["arch"],
        },
        out / "model.pt",
    )
    write_csv(fit["history"], out / "history.csv")
    write_csv(pd.concat([pd.DataFrame({"clip_id": d.clip_ids, "role": r}) for r, d in sets.items()]), out / "roles.csv")
    report = write_predictions(ctx, model, sets, clips, args.fold, fit["T"], device, out)
    summary = {
        "fold": args.fold,
        "seed": seed,
        "T": fit["T"],
        "holdout_nll_raw": fit["holdout_nll_raw"],
        "holdout_nll_cal": fit["holdout_nll_cal"],
        "preds": report,
        "dropped": [x for d in sets.values() for x in d.dropped],
        "pretrained": cfg.camformer.pretrained,
        "freeze_encoder": cfg.camformer.freeze_encoder,
        "head_from_ckpt": bool(loaded.get("head_loaded", [False])[0]),
    }
    write_json(out / "summary.json", summary)
    ctx.finish(outputs, part=part)
    ckpt_path.unlink(missing_ok=True)  # a manifest scritto la ripresa non serve più
    ctx.log.info("predizioni: %s", report)


if __name__ == "__main__":
    main()
