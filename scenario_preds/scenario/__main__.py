"""Punto d'ingresso unico:  python -m scenario <stadio> --config configs/default.yaml [--set k=v] [--overwrite]

Stadi, nell'ordine:
  index           indice delle take: etichette, split, fold, video, durata          (CPU)
  traj            traiettorie a 20 Hz, una per take                                  (CPU)
  check-up        (diagnostica) verso dell'alto nei JSON: rollio della camera RGB e
                  segno dell'asse z del mondo; suggerisce poses.json_cam_to_device (CPU)
  check-frame     (diagnostica) stima la rotazione JSON -> device da pose vere del device (CPU)
  feat-camformer  embedding CamFormer congelato per take                             (GPU, va anche su CPU)
  feat-egovlp     CLS EgoVLP congelato per take                                       (GPU)
  heads           teste a 6 classi, predizioni per take, metriche, controllo pre-registrato (CPU)
                  --eval-test: metriche e predizioni del test (una volta sola, alla fine)
                  --skip-check: salta il controllo (solo per analisi secondarie)
  select          sceglie sulla val variante delle pose e numero di finestre tra run candidate (CPU)
                  --candidates DIR [DIR ...]: run fatte con heads --models camformer --skip-check

Uscita con codice 3 se il controllo pre-registrato fallisce: i job a valle (gate) non partono.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scenario import config as C
from scenario import io

# Sezioni di config da cui dipende ogni stadio. Traiettorie e feature non dipendono da split ed
# etichette: cambiare il taglio val/test o i seed rifà solo index e heads, non il lavoro su GPU.
STAGE_KEYS = {
    "index": ["data", "labels", "split"],
    "traj": ["data", "poses"],
    "feat-camformer": ["data", "poses", "window", "camformer"],
    "feat-egovlp": ["data", "window", "egovlp"],
    "heads": ["data", "labels", "split", "poses", "window", "camformer", "egovlp", "heads", "check"],
}
CHECK_FAILED = 3
MODELS = ("camformer", "egovlp")


def setup_logging(run_dir: Path, stage: str) -> logging.Logger:
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "logs" / f"{stage}.log")):
        h.setFormatter(fmt)
        root.addHandler(h)
    return logging.getLogger(f"scenario.{stage}")


def resolve_device(name: str):
    import torch

    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def read_index(run_dir: Path) -> pd.DataFrame:
    io.require(run_dir, "index")
    return pd.read_csv(run_dir / "index" / "takes.csv", dtype={"take_uid": str})


def check_loss(n_lost: int, n_total: int, max_frac: float, what: str) -> None:
    if n_total and n_lost / n_total > max_frac:
        raise RuntimeError(f"{what}: perse {n_lost}/{n_total} take (> {max_frac:.0%}). Vedi lo status dello stadio.")


def take_windows(row, wcfg: dict):
    from scenario.windows import compute_windows

    return compute_windows(float(row.duration_s), wcfg["length_s"], wcfg["n_windows"], wcfg["placement"])


# ----------------------------------------------------------------------------- stadi
def stage_index(cfg, run_dir, args, log):
    from scenario.index import build_index, summary

    out = run_dir / "index" / "takes.csv"
    key = C.section_hash(cfg, STAGE_KEYS["index"])
    if io.is_done(run_dir, "index", key, [out], args.overwrite):
        log.info("index già fatto")
        return
    repo = None
    try:
        repo = C.path(cfg["camformer"]["repo"])
    except EnvironmentError:
        log.warning("CAMFORMER_REPO non definito: colonna camformer_pretrain vuota")
    df = build_index(cfg, repo)
    exp = cfg["data"]["expected_takes"]
    if exp is not None and len(df) != int(exp):
        raise RuntimeError(f"index: {len(df)} take invece di {exp} (data.expected_takes)")
    io.write_csv(df, out)
    s = summary(df, cfg["window"]["length_s"])
    io.write_json(s, run_dir / "index" / "summary.json")
    check_loss(int(df.video_path.isna().sum()), len(df), cfg["data"]["max_missing_frac"], "index: video")
    off = df.split_official.str.lower() == "val"
    log.info("val ufficiale (prima del taglio), take e sessioni per classe:\n%s",
             df[off].groupby("label").agg(takes=("take_name", "size"), sessions=("session", "nunique")))
    log.info("indice: %d take\n%s", len(df), pd.crosstab(df.label, df.split, margins=True))
    log.info("sessioni sia nel train sia in val/test: %d (%d take di val/test)",
             s["sessions_train_and_eval"]["n"], s["sessions_train_and_eval"]["takes_eval"])
    log.info("take viste dal pretraining di CamFormer:\n%s", pd.crosstab(df.split, df.camformer_pretrain))
    if cfg["split"]["mode"] == "official" and cfg["split"]["official_val"]["mode"] == "undecided":
        log.warning("taglio val/test non deciso (split.official_val.mode = undecided): heads non partirà")
    io.mark_done(run_dir, "index", key)


def stage_traj(cfg, run_dir, args, log):
    from scenario.poses import PoseError, Trajectory, load_take

    takes = read_index(run_dir)
    tdir = run_dir / "traj"
    status_path = tdir / "status.csv"
    key = C.section_hash(cfg, STAGE_KEYS["traj"])
    if io.is_done(run_dir, "traj", key, [status_path], args.overwrite):
        log.info("traj già fatto")
        return
    rows = []
    for i, r in enumerate(takes.itertuples(index=False)):
        dest = tdir / f"{r.take_name}.npz"
        row = {"take_name": r.take_name, "status": "ok", "source": "", "n": 0, "valid_frac": np.nan, "dur_s": np.nan,
               "n_missing_src": np.nan, "first_src_frame": np.nan}
        try:
            if dest.exists() and not args.overwrite:
                tr = Trajectory.load(dest)
            else:
                tr = load_take(r, cfg["poses"])
                tr.save(dest)
            row.update(source=tr.source, n=len(tr.t), valid_frac=float(tr.valid.mean()), dur_s=float(tr.t[-1]),
                       n_missing_src=tr.n_missing_src, first_src_frame=tr.first_src_frame)
        except PoseError as e:
            row["status"] = e.status
        except Exception as e:  # noqa: BLE001
            log.warning("%s: %s", r.take_name, e)
            row["status"] = f"error:{type(e).__name__}"
        rows.append(row)
        if (i + 1) % 100 == 0:
            log.info("traiettorie: %d/%d", i + 1, len(takes))
    st = pd.DataFrame(rows)
    st["has_gap"] = st.n_missing_src > 0
    io.write_csv(st, status_path)
    log.info("esiti:\n%s", st.status.value_counts())
    log.info("take con buchi interni nella sorgente delle pose: %d (atteso dal report: %s)",
             int(st.has_gap.sum()), cfg["data"]["expected_gap_takes"])
    log.info("take con le pose che iniziano dopo il frame 0: %d", int((st.first_src_frame > 0).sum()))
    lost = int((st.status != "ok").sum())
    if lost and cfg["poses"]["source"] == "aria_csv" and (st.status == "no_aria_csv").any():
        log.warning("%d take senza closed_loop_trajectory.csv: --set poses.fallback_to_json=true per usare i JSON", int((st.status == "no_aria_csv").sum()))
    check_loss(lost, len(st), cfg["data"]["max_missing_frac"], "traj")
    io.mark_done(run_dir, "traj", key)


def stage_check_up(cfg, run_dir, args, log):
    """Senza pose del device: dove punta l'alto del mondo nel frame della camera dei JSON (vedi poses.up_statistics)."""
    from scenario.poses import from_pose_json, suggest_cam_to_device, up_statistics

    takes = read_index(run_dir)
    cand = takes[takes.video_path.notna()].sample(frac=1.0, random_state=0)
    if args.n_check > 0:
        cand = cand.groupby("label").head(args.n_check)
    pc = dict(cfg["poses"], json_cam_to_device=np.eye(3).tolist(), json_world_rotation=np.eye(3).tolist())
    rows = []
    for r in cand.itertuples(index=False):
        try:
            rows.append({"take_name": r.take_name, "label": r.label, **up_statistics(from_pose_json(Path(r.pose_json_path), r.video_path, pc))})
        except Exception as e:  # noqa: BLE001
            log.warning("%s: %s", r.take_name, e)
    if not rows:
        raise RuntimeError("nessuna take leggibile")
    df = pd.DataFrame(rows)
    up = df[["up_x", "up_y", "up_z"]].to_numpy()
    per_take = [suggest_cam_to_device(u)["roll_k"] for u in up]
    df["roll_k"] = per_take
    sug = suggest_cam_to_device(up.mean(0))
    agree = float(np.mean(np.array(per_take) == sug["roll_k"]))
    pitch = float(df.pitch_deg_median.median())
    warnings_ = []
    if agree < 0.9:
        warnings_.append(f"il rollio suggerito vale solo per il {agree:.0%} delle take: guardare per_take.csv")
    if pitch > 10:
        warnings_.append("in media la camera guarda IN ALTO: probabile asse z del mondo verso il basso (gravità e token di gravità sbagliati)")
    if sug["residual_roll_deg"] > 20:
        warnings_.append(f"rollio residuo di {sug['residual_roll_deg']:.0f}°: il montaggio non è a multipli di 90°, valutare check-frame")
    out = {
        "n_takes": int(len(df)),
        "mean_up_in_camera": np.round(up.mean(0), 4).tolist(),
        "pitch_deg_median": pitch,
        "pitch_deg_median_by_label": df.groupby("label").pitch_deg_median.median().round(1).to_dict(),
        "takes_agreeing_with_roll": agree,
        "current_json_cam_to_device": cfg["poses"]["json_cam_to_device"],
        "suggested": sug,
        "matches_current": bool(np.allclose(sug["json_cam_to_device"], cfg["poses"]["json_cam_to_device"], atol=1e-6)),
        "warnings": warnings_,
    }
    io.write_csv(df, run_dir / "check_up" / "per_take.csv")
    io.write_json(out, run_dir / "check_up" / "summary.json")
    log.info("check-up: alto medio nel frame camera %s, inclinazione mediana %.1f°, rollio suggerito %d° (accordo %.0f%%, residuo %.1f°)",
             out["mean_up_in_camera"], pitch, sug["roll_deg"], 100 * agree, sug["residual_roll_deg"])
    log.info("X suggerita %s; coincide con la config: %s", sug["json_cam_to_device"], out["matches_current"])
    for w in warnings_:
        log.warning(w)


def stage_check_frame(cfg, run_dir, args, log):
    """Per le take con CSV e JSON stima X e W (vedi poses.estimate_frame) e ne fa la media."""
    from scenario.poses import Trajectory, average_rotations, estimate_frame, from_aria_csv, from_pose_json

    takes = read_index(run_dir)
    cand = takes[takes.aria_csv_exists & takes.video_path.notna()]
    cand = cand.sample(frac=1.0, random_state=0).groupby("label").head(args.n_check)
    pc = dict(cfg["poses"], json_cam_to_device=np.eye(3).tolist(), json_world_rotation=np.eye(3).tolist())
    rows = []
    for r in cand.itertuples(index=False):
        try:
            res = estimate_frame(from_pose_json(Path(r.pose_json_path), r.video_path, pc), from_aria_csv(Path(r.aria_csv_path), pc["fps"]))
            rows.append({"take_name": r.take_name, "label": r.label, **res})
        except Exception as e:  # noqa: BLE001
            log.warning("%s: %s", r.take_name, e)
    if not rows:
        raise RuntimeError("nessuna take con entrambe le sorgenti utilizzabile")
    df = pd.DataFrame(rows)
    good = df[df.residual_deg_median < 5.0]
    X = average_rotations([np.array(x) for x in good.json_cam_to_device]) if len(good) else None
    W = average_rotations([np.array(w) for w in good.json_world_rotation]) if len(good) else None
    out = {
        "n_takes": int(len(df)),
        "n_good": int(len(good)),
        "shift_s_median": float(df.shift_s.median()),
        "residual_deg_median": float(df.residual_deg_median.median()),
        "json_cam_to_device": None if X is None else np.round(X, 4).tolist(),
        "json_world_rotation": None if W is None else np.round(W, 4).tolist(),
        "spread_deg_X": None if X is None else float(np.median([np.degrees(np.arccos(np.clip((np.trace(X.T @ np.array(x)) - 1) / 2, -1, 1))) for x in good.json_cam_to_device])),
    }
    io.write_csv(df.assign(json_cam_to_device=df.json_cam_to_device.map(str), json_world_rotation=df.json_world_rotation.map(str)), run_dir / "check_frame" / "per_take.csv")
    io.write_json(out, run_dir / "check_frame" / "summary.json")
    log.info("check-frame: %s", out)
    log.info("Se spread e residuo sono piccoli, copiare le due matrici in poses.json_cam_to_device / json_world_rotation.")


def stage_feat_camformer(cfg, run_dir, args, log):
    import torch

    from scenario import camformer as cf
    from scenario.poses import Trajectory

    ccfg = cfg["camformer"]
    out = run_dir / "features" / "camformer.npz"
    key = C.section_hash(cfg, STAGE_KEYS["feat-camformer"])
    if io.is_done(run_dir, "feat-camformer", key, [out], args.overwrite):
        log.info("feat-camformer già fatto")
        return
    io.require(run_dir, "traj")
    takes = read_index(run_dir)
    repo, ckpt = C.path(ccfg["repo"]), C.path(ccfg["ckpt"])
    du = cf.load_repo_module(repo, "utils/dataset_utils.py")
    enc_id = cf.POSE_ENCODINGS[ccfg["pose_encoding"]]
    model = cf.build_encoder(ccfg, repo)
    cf.load_weights(model, ckpt)
    device = resolve_device(ccfg["device"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    items, owner, status = [], [], []
    for i, r in enumerate(takes.itertuples(index=False)):
        f = run_dir / "traj" / f"{r.take_name}.npz"
        if not f.exists():
            status.append("no_traj")
            continue
        tr = Trajectory.load(f)
        dur = float(r.duration_s) if np.isfinite(r.duration_s) else float(tr.t[-1])
        wins = take_windows(r._replace(duration_s=dur), cfg["window"])
        got = 0
        for t0, t1 in wins:
            it = cf.window_input(tr, t0, min(t1, float(tr.t[-1]) + 1e-6), ccfg["context_s"], ccfg["min_coverage"], du.transform_camera_pose, enc_id)
            if it is not None:
                items.append(it)
                owner.append(i)
                got += 1
        status.append("ok" if got else "low_coverage")
    t_start = time.time()
    emb = cf.embed(model, items, ccfg["batch_size"], device)
    log.info("CamFormer: %d finestre in %.1fs", len(items), time.time() - t_start)
    X = np.full((len(takes), ccfg["output_dim"]), np.nan, np.float32)
    n_win = np.zeros(len(takes), int)
    owner = np.asarray(owner)
    for i in np.unique(owner):
        X[i] = emb[owner == i].mean(0)
        n_win[i] = int((owner == i).sum())
    st = pd.Series(status)
    log.info("esiti:\n%s", st.value_counts())
    # take senza embedding: restano nelle predizioni senza cam_pred e nel gate vanno a EgoVLP (registro, audit punto 3)
    check_loss(int((st != "ok").sum()), len(st), ccfg.get("max_missing_frac", cfg["data"]["max_missing_frac"]), "feat-camformer")
    io.save_npz(out, take_name=takes.take_name.to_numpy().astype(str), X=X, n_windows=n_win, status=st.to_numpy().astype(str))
    io.mark_done(run_dir, "feat-camformer", key)


def stage_feat_egovlp(cfg, run_dir, args, log):
    import torch
    from torch.utils.data import DataLoader

    from scenario import egovlp as ev

    ecfg = cfg["egovlp"]
    out = run_dir / "features" / "egovlp.npz"
    key = C.section_hash(cfg, STAGE_KEYS["feat-egovlp"])
    if io.is_done(run_dir, "feat-egovlp", key, [out], args.overwrite):
        log.info("feat-egovlp già fatto")
        return
    takes = read_index(run_dir)
    items, owner = [], []
    for i, r in enumerate(takes.itertuples(index=False)):
        if not isinstance(r.video_path, str) or not np.isfinite(r.duration_s):
            continue
        for t0, t1 in take_windows(r, cfg["window"]):
            items.append({"video_path": r.video_path, "t_start": t0, "t_end": t1})
            owner.append(i)
    model = ev.build_model(ecfg)
    ev.load_weights(model, C.path(ecfg["ckpt"]))
    device = resolve_device(ecfg["device"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    loader = DataLoader(ev.WindowFrames(items, ecfg), batch_size=ecfg["batch_size"], num_workers=ecfg["num_workers"], collate_fn=ev.collate, shuffle=False)
    emb = np.full((len(items), ecfg["embed_dim"]), np.nan, np.float32)
    t_start, done = time.time(), 0
    with torch.no_grad():
        for b in loader:
            x = ev.normalize(b["frames"], ecfg["norm_mean"], ecfg["norm_std"]).to(device, non_blocking=True)
            cls = model(x).float().cpu().numpy()
            cls[~b["ok"]] = np.nan
            emb[b["i"]] = cls
            done += len(b["i"])
            if done % (20 * ecfg["batch_size"]) < ecfg["batch_size"]:
                log.info("EgoVLP: %d/%d finestre (%.0fs)", done, len(items), time.time() - t_start)
    X = np.full((len(takes), ecfg["embed_dim"]), np.nan, np.float32)
    n_win = np.zeros(len(takes), int)
    owner = np.asarray(owner)
    for i in np.unique(owner):
        e = emb[owner == i]
        e = e[np.isfinite(e).all(1)]
        if len(e):
            X[i] = e.mean(0)
            n_win[i] = len(e)
    status = np.where(n_win > 0, "ok", "no_frames")
    check_loss(int((status != "ok").sum()), len(takes), cfg["data"]["max_missing_frac"], "feat-egovlp")
    io.save_npz(out, take_name=takes.take_name.to_numpy().astype(str), X=X, n_windows=n_win, status=status.astype(str))
    io.mark_done(run_dir, "feat-egovlp", key)


def stage_heads(cfg, run_dir, args, log):
    from scenario.heads import bacc_ci, cross_fit_seeds, metrics, preregistered_check

    takes = read_index(run_dir)
    classes = list(cfg["labels"]["classes"])
    hcfg, ccfg = cfg["heads"], cfg["check"]
    seeds = list(cfg["split"]["fold_seeds"])
    if cfg["split"]["mode"] == "official" and cfg["split"]["official_val"]["mode"] == "undecided":
        raise RuntimeError("split.official_val.mode = undecided: decidere il taglio val/test (registro) e rifare index --overwrite")
    flags = {"models": sorted(args.models), "eval_test": bool(args.eval_test), "skip_check": bool(args.skip_check)}
    key = C.section_hash(dict(cfg, _args=flags), STAGE_KEYS["heads"] + ["_args"])
    check_path = run_dir / "check.json"
    if io.is_done(run_dir, "heads", key, [run_dir / "predictions.csv"], args.overwrite):
        log.info("heads già fatto")
        if not args.skip_check and check_path.exists() and not io_json(check_path)["passed"]:
            log.error("controllo pre-registrato FALLITO (vedi %s): ci si ferma", check_path)
            sys.exit(CHECK_FAILED)
        return
    fold_cols = [f"fold_s{s}" for s in seeds]
    joined = takes[["take_uid", "take_name", "site", "session", "split", "fold", *fold_cols, "label", "label_idx", "in_head", "camformer_pretrain"]].copy()
    all_metrics, per_model = {}, {}
    for m in args.models:
        f = run_dir / "features" / f"{m}.npz"
        io.require(run_dir, f"feat-{m}")
        z = np.load(f)
        if list(z["take_name"]) != takes.take_name.tolist():
            raise RuntimeError(f"{f}: ordine delle take diverso dall'indice: rifare feat-{m}")
        res, fold_tab = cross_fit_seeds(z["X"].astype(np.float64), takes, len(classes), hcfg, seeds)
        P = res["probs"]
        has = np.isfinite(P).all(1)
        pred_idx = np.where(has, np.nan_to_num(P).argmax(1), -1)
        df = joined.copy()
        df["pred_source"] = res["source"]
        df["pred_idx"] = np.where(has, pred_idx, np.nan)
        df["pred"] = [classes[k] if k >= 0 else None for k in pred_idx]
        df["conf"] = np.where(has, np.nan_to_num(P).max(1), np.nan)
        df["conf_seed_sd"] = res["seed_sd"]
        df["correct"] = np.where(has & (df.label_idx >= 0), pred_idx == df.label_idx, np.nan)
        for c, name in enumerate(classes):
            df[f"p_{name}"] = P[:, c]
        df["note"] = np.where(~df.in_head, "classe fuori dalla testa", np.where(~has, "nessuna embedding", ""))
        per_model[m] = df
        outd = run_dir / "heads" / m
        write_split_files(df, outd / "predictions.csv", args.eval_test)
        io.write_csv(fold_tab, outd / "folds.csv")
        splits = ["train", "val"] + (["test"] if args.eval_test else [])
        mets = {}
        for s in splits:
            mt = metrics(df, classes, s, hcfg["min_takes_metric"])
            if mt is not None:
                mt["bacc_ci"] = bacc_ci(df, classes, s, hcfg["min_takes_metric"], hcfg["n_boot"], hcfg["ci_level"])
            mets[s] = mt
        mets["note"] = f"train = fuori campione (OOF), media su {len(seeds)} seed; val/test = media dei modelli di fold di tutti i seed"
        io.write_json(mets, outd / "metrics.json")
        all_metrics[m] = {s: (None if v is None else {k: v[k] for k in ("n", "acc", "bacc", "bacc_ci")}) for s, v in mets.items() if s != "note"}
        prefix = {"camformer": "cam", "egovlp": "vid"}[m]
        for col in ("pred", "conf", "conf_seed_sd", "pred_source"):
            joined[f"{prefix}_{col}"] = df[col]
        for name in classes:
            joined[f"{prefix}_p_{name}"] = df[f"p_{name}"]
        log.info("%s: %s", m, all_metrics[m])
    write_split_files(joined, run_dir / "predictions.csv", args.eval_test)
    io.write_json({"classes": classes, "excluded_from_head": sorted(set(takes.label) - set(classes)), "fold_seeds": seeds}, run_dir / "classes.json")
    io.write_json(all_metrics, run_dir / "metrics_summary.json")
    passed = True
    if not args.skip_check:
        if ccfg["model"] not in per_model:
            raise RuntimeError(f"check.model = {ccfg['model']} non è tra i modelli valutati {args.models}")
        res = preregistered_check(per_model[ccfg["model"]], classes, ccfg, hcfg)
        io.write_json(res, check_path)
        passed = res["passed"]
        log.info("controllo pre-registrato su %s (%s): bacc=%.3f (soglia %.2f) CI=%s caso=%.3f; richiamo sotto %s: %s; "
                 "classi assenti dal train: %s; classi senza take nello split: %s; take senza embedding: %d -> %s",
                 res["model"], res["split"], res["bacc"], res["min_bacc"], res["bacc_ci"], res["chance"], res["min_recall"],
                 res["classes_below_min_recall"] or "nessuna", res["classes_absent_in_train"] or "nessuna",
                 res["classes_missing_in_split"] or "nessuna", res["n_no_embedding"],
                 "SUPERATO" if passed else "FALLITO")
    io.mark_done(run_dir, "heads", key, flags)
    if not passed:
        log.error("controllo pre-registrato FALLITO: ci si ferma qui (vedi %s)", check_path)
        sys.exit(CHECK_FAILED)


IDENTITY = np.eye(3)
PAPER_FLIP = np.diag([-1.0, -1.0, 1.0])


def pose_variant(m) -> str:
    m = np.asarray(m, float)
    if np.allclose(m, IDENTITY):
        return "identita"
    if np.allclose(m, PAPER_FLIP):
        return "diag(-1,-1,1)"
    return "altra"


def stage_select(cfg, run_dir, args, log):
    """Sceglie sulla VAL la variante delle pose e il numero di finestre (registro, audit punto 1).

    Ogni candidata è una run completa fino a ``heads --models camformer --skip-check`` con la stessa config
    salvo ``poses.json_cam_to_device`` e ``window.n_windows``. Regola pre-registrata: massima accuratezza
    bilanciata di CamFormer sulla val; a parità meno finestre, poi l'identità. Legge solo la val: una
    candidata con metriche del test (--eval-test) è un errore. EgoVLP usa lo stesso numero di finestre.
    """
    import json

    if not args.candidates:
        raise ValueError("select: servono --candidates DIR [DIR ...]")
    rows, ref_idx, ref_key = [], None, None
    fixed = ["data", "labels", "split", "camformer", "heads", "check"]
    for c in map(Path, args.candidates):
        io.require(c, "heads")
        m = json.loads((c / "heads" / "camformer" / "metrics.json").read_text())
        if "test" in m:
            raise RuntimeError(f"{c}: metriche del test presenti (--eval-test): la candidata non è ammessa")
        val = m.get("val")
        if val is None:
            raise RuntimeError(f"{c}: nessuna metrica di val (serve split.official_val.mode = split)")
        ccfg = json.loads((c / "configs" / "heads.json").read_text())
        rest = dict({k: ccfg[k] for k in fixed},
                    poses={k: v for k, v in ccfg["poses"].items() if k != "json_cam_to_device"},
                    window={k: v for k, v in ccfg["window"].items() if k != "n_windows"})
        key = C.section_hash(rest, list(rest))
        idx = pd.read_csv(c / "index" / "takes.csv", usecols=["take_name", "split"]).sort_values("take_name")
        idx_key = hash(tuple(map(tuple, idx.to_numpy())))
        if ref_key is None:
            ref_key, ref_idx = key, idx_key
        elif key != ref_key or idx_key != ref_idx:
            raise RuntimeError(f"{c}: config o divisione diverse dalla prima candidata (cambiano solo pose e finestre)")
        X = ccfg["poses"]["json_cam_to_device"]
        rows.append({"run_dir": str(c), "variant": pose_variant(X), "json_cam_to_device": X,
                     "n_windows": int(ccfg["window"]["n_windows"]), "val_bacc": float(val["bacc"]),
                     "val_bacc_ci": val.get("bacc_ci"), "val_n": int(val["n"]),
                     "val_n_no_embedding": val.get("n_no_embedding")})
    rank = sorted(rows, key=lambda r: (-round(r["val_bacc"], 9), r["n_windows"], r["variant"] != "identita"))
    best = rank[0]
    sets = [f"poses.json_cam_to_device={json.dumps(best['json_cam_to_device'], separators=(',', ':'))}", f"window.n_windows={best['n_windows']}"]
    out = {"rule": "max bacc CamFormer sulla val; a parità meno finestre, poi identità", "chosen": best,
           "set_for_final_run": sets, "candidates": rank}
    io.write_json(out, run_dir / "select" / "selection.json")
    io.write_csv(pd.DataFrame(rank).drop(columns=["json_cam_to_device"]), run_dir / "select" / "candidates.csv")
    for r in rank:
        log.info("%-14s finestre=%d  bacc val=%.4f  CI=%s  (%s)", r["variant"], r["n_windows"], r["val_bacc"], r["val_bacc_ci"], r["run_dir"])
    log.info("scelta: %s, %d finestre. Run finale con:  %s", best["variant"], best["n_windows"], "  ".join(f"--set '{x}'" for x in sets))


def io_json(path: Path) -> dict:
    import json

    return json.loads(Path(path).read_text())


def write_split_files(df: pd.DataFrame, dest: Path, eval_test: bool) -> None:
    """Train e val in ``dest``; il test in ``sealed/`` accanto, solo con --eval-test (test sigillato)."""
    io.write_csv(df[df.split != "test"], dest)
    sealed = dest.parent / "sealed" / dest.name
    if eval_test:
        io.write_csv(df[df.split == "test"], sealed)


STAGES = {
    "index": stage_index,
    "traj": stage_traj,
    "check-up": stage_check_up,
    "check-frame": stage_check_frame,
    "feat-camformer": stage_feat_camformer,
    "feat-egovlp": stage_feat_egovlp,
    "heads": stage_heads,
    "select": stage_select,
}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m scenario", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=list(STAGES))
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="CHIAVE=VALORE")
    p.add_argument("--overwrite", action="store_true", help="rifà lo stadio anche se già completo")
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS, help="heads: modelli da valutare")
    p.add_argument("--eval-test", action="store_true", help="heads: metriche e predizioni del test (una volta, alla fine)")
    p.add_argument("--skip-check", action="store_true", help="heads: salta il controllo pre-registrato (analisi secondarie)")
    p.add_argument("--candidates", nargs="+", default=[], help="select: run_dir delle run candidate")
    p.add_argument("--n-check", type=int, default=20, help="check-up / check-frame: take per classe (0 = tutte, solo check-up)")
    args = p.parse_args(argv)
    cfg = C.load_config(Path(args.config), args.overrides)
    run_dir = C.path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(run_dir, args.stage)
    io.write_json(cfg, run_dir / "configs" / f"{args.stage}.json")
    log.info("stadio %s, run_dir %s", args.stage, run_dir)
    STAGES[args.stage](cfg, run_dir, args, log)
    log.info("fine %s", args.stage)


if __name__ == "__main__":
    main()
