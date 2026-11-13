"""Fine-tuning end-to-end di CamFormer per una fold (logica dello stadio 20).

Per la fold k (o ``full``):
- training sulle take di train con fold ≠ k, escluso l'holdout interno della fold;
- early stopping e temperature scaling sull'holdout interno;
- predizioni fuori campione sulla fold k (``preds_oof``), e su val e test.
Ottimizzatore AdamW (lr della testa × ``head_lr_mult``), tutto da ``camformer.train``.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.cascade.calibration import apply_temperature, fit_temperature, nll_from_logits
from src.cascade.io import torch_save
from src.cascade.models.camformer import collate

log = logging.getLogger(__name__)


def _loader(
    ds,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    *,
    pin_memory: bool = False,
    prefetch_factor=None,
    persistent_workers: bool = False,
):
    """DataLoader con shuffle seminato. torch >= 1.9 risemina numpy in ogni worker, quindi
    il jitter temporale (np.random in ``__getitem__``) non si ripete tra worker."""
    import torch
    from torch.utils.data import DataLoader

    g = torch.Generator()
    g.manual_seed(seed)
    kw = {}
    if num_workers > 0:
        kw["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kw["prefetch_factor"] = prefetch_factor
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=num_workers,
        generator=g,
        drop_last=False,
        pin_memory=pin_memory and torch.cuda.is_available(),
        **kw,
    )


def mc_dropout_mode(model) -> None:
    """Modalità MC dropout per il classificatore CamFormer.

    Non basta mettere in train i soli ``nn.Dropout``: in eval ``nn.TransformerEncoderLayer``
    usa il fast path (che ignora i dropout) e il dropout dell'attenzione dipende da
    ``layer.training``. Quindi tutto in train, tranne il flag del modulo
    ``CameraPoseSeqEncoder``, che in train sostituirebbe a caso il token di gravità con
    quello appreso (non è incertezza del modello, è un'augmentation del pretraining).
    """
    model.train()
    if hasattr(model, "encoder"):
        model.encoder.training = False  # solo il flag di questo modulo, non dei figli


def _to(batch, device):
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def predict(model, ds, device, batch_size: int, mc_passes: int = 0) -> Dict[str, np.ndarray]:
    import torch

    if len(ds) == 0:
        return {"logits": np.zeros((0, model.head.out_features)), "emb": np.zeros((0, model.head.in_features))}
    model.eval()
    logits, embs = [], []
    with torch.no_grad():
        for b in _loader(ds, batch_size, False, 0, 0):
            b = _to(b, device)
            lo, e = model(b["x"], b["pad_mask"], b["lengths"], b["win_start"], b["win_end"])
            logits.append(lo.cpu().numpy())
            embs.append(e.cpu().numpy())
    out = {"logits": np.concatenate(logits), "emb": np.concatenate(embs)}
    if mc_passes > 0:
        mc_dropout_mode(model)
        passes = []
        with torch.no_grad():
            for _ in range(mc_passes):
                pl = []
                for b in _loader(ds, batch_size, False, 0, 0):
                    b = _to(b, device)
                    pl.append(model(b["x"], b["pad_mask"], b["lengths"], b["win_start"], b["win_end"])[0].cpu().numpy())
                passes.append(np.concatenate(pl))
        model.eval()
        out["mc_logits"] = np.stack(passes)
    return out


def save_epoch(path: Path, epoch: int, model, opt, best, best_state, bad: int, history, key: str) -> None:
    """Stato di fine epoca per riprendere un job interrotto (scrittura atomica, ``io.torch_save``)."""
    torch_save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "best": best,
            "best_state": best_state,
            "bad": bad,
            "history": history,
            "key": key,
        },
        path,
    )


def load_epoch(path: Path, model, opt, key: str):
    """(epoca da cui ripartire, best, best_state, bad, history).

    ``key`` identifica la config dello stadio, la fold e il seed: un checkpoint nato da un'altra
    config, o non leggibile, si ignora e l'addestramento riparte dall'epoca 0.
    """
    import torch

    fresh = (0, None, None, 0, [])
    if not path.exists():
        return fresh
    try:
        st = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001 - un checkpoint rovinato non deve fermare lo stadio
        log.warning("checkpoint di ripresa %s non leggibile (%s): riparto da zero", path, e)
        return fresh
    if st.get("key") != key:
        log.warning("checkpoint di ripresa %s di un'altra config/fold/seed: ignorato", path)
        return fresh
    model.load_state_dict(st["model"])
    opt.load_state_dict(st["opt"])
    return st["epoch"] + 1, st["best"], st["best_state"], st["bad"], st["history"]


def train_fold(
    model,
    train_ds,
    hold_ds,
    cfg,
    device,
    seed: int,
    *,
    ckpt_path: Optional[Path] = None,
    ckpt_key: str = "",
) -> Dict[str, object]:
    """Addestra una fold. Con ``ckpt_path`` salva lo stato a fine epoca e, se trova un
    checkpoint con la stessa ``ckpt_key``, riparte dall'epoca successiva all'ultima salvata."""
    import torch
    import torch.nn.functional as F

    tc = cfg.camformer.train
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    enc_params = [p for p in model.parameters() if id(p) not in head_ids]
    if tc.optimizer.lower() != "adamw":
        raise ValueError("camformer.train.optimizer: solo adamw nell'MVP")
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": tc.lr}, {"params": head_params, "lr": tc.lr * tc.head_lr_mult}],
        weight_decay=tc.weight_decay,
    )
    model.to(device)
    best, best_state, bad, history = None, None, 0, []
    start = 0
    if ckpt_path is not None:
        start, best, best_state, bad, history = load_epoch(ckpt_path, model, opt, ckpt_key)
        if start:
            log.info("ripresa dall'epoca %d da %s", start, ckpt_path)
            if bad >= tc.patience:  # l'early stopping era già scattato prima dell'interruzione
                start = tc.epochs
    for epoch in range(start, tc.epochs):
        model.train()
        tot, n = 0.0, 0
        for b in _loader(
            train_ds,
            tc.batch_size,
            True,
            tc.num_workers,
            seed + epoch,
            pin_memory=tc.pin_memory,
            prefetch_factor=tc.prefetch_factor,
            persistent_workers=tc.persistent_workers,
        ):
            b = _to(b, device)
            logits, _ = model(b["x"], b["pad_mask"], b["lengths"], b["win_start"], b["win_end"])
            loss = F.cross_entropy(logits, b["label"])
            opt.zero_grad()
            loss.backward()
            if tc.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            tot += float(loss.detach()) * len(b["label"])
            n += len(b["label"])
        hp = predict(model, hold_ds, device, cfg.camformer.batch_size_eval)
        y = hold_ds.labels
        h_loss = nll_from_logits(hp["logits"], y) if len(y) else float("nan")
        h_acc = float((hp["logits"].argmax(1) == y).mean()) if len(y) else float("nan")
        history.append({"epoch": epoch, "train_loss": tot / max(n, 1), "holdout_loss": h_loss, "holdout_acc": h_acc})
        log.info("epoca %d: train_loss=%.4f holdout_loss=%.4f holdout_acc=%.4f", epoch, tot / max(n, 1), h_loss, h_acc)
        score = -h_loss if tc.monitor == "val_loss" else h_acc
        stop = False
        if best is None or score > best:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            stop = bad >= tc.patience
        if ckpt_path is not None:  # dopo l'aggiornamento di best e bad: la ripresa riparte da qui
            save_epoch(ckpt_path, epoch, model, opt, best, best_state, bad, history, ckpt_key)
        if stop:
            log.info("early stopping all'epoca %d", epoch)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    hp = predict(model, hold_ds, device, cfg.camformer.batch_size_eval)
    T = fit_temperature(hp["logits"], hold_ds.labels) if len(hold_ds) else 1.0
    return {
        "history": pd.DataFrame(history),
        "T": T,
        "holdout_nll_raw": nll_from_logits(hp["logits"], hold_ds.labels) if len(hold_ds) else float("nan"),
        "holdout_nll_cal": nll_from_logits(hp["logits"], hold_ds.labels, T) if len(hold_ds) else float("nan"),
    }


def preds_table(ds, pred: Dict[str, np.ndarray], T: float, clips: pd.DataFrame, labels, *, fold: str, source: str):
    """Tabella delle predizioni nel formato comune (join per clip_id)."""
    from src.data.cache import Table

    meta_cols = clips.set_index("clip_id").loc[ds.clip_ids, ["take_uid", "split", "label_idx"]].reset_index()
    arrays = {
        "cam_logits": pred["logits"].astype(np.float32),
        "cam_probs_cal": apply_temperature(pred["logits"], T).astype(np.float32),
        "cam_emb": pred["emb"].astype(np.float32),
    }
    df = meta_cols.copy()
    df["T_cam"] = T
    df["cam_fold"] = fold
    if "mc_logits" in pred:
        mc = np.stack([apply_temperature(lg, T) for lg in pred["mc_logits"]])
        arrays["cam_mc_probs_mean"] = mc.mean(0).astype(np.float32)
        arrays["cam_mc_probs_std"] = mc.std(0).astype(np.float32)
        pm = mc.mean(0)
        ent = -(pm * np.log(np.clip(pm, 1e-12, 1))).sum(1)  # incertezza totale
        exp_ent = -(mc * np.log(np.clip(mc, 1e-12, 1))).sum(2).mean(0)  # aleatoria attesa
        df["cam_mc_entropy"] = ent
        df["cam_mc_mi"] = np.clip(ent - exp_ent, 0, None)  # epistemica (BALD)
    return Table(
        df, arrays, {"class_names": list(labels.names), "stage": "20_camformer", "fold": fold, "source": source}
    )
