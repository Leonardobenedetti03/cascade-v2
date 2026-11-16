"""Ripresa per epoca dello stadio 20 (D7): ckpt_last.pt, chiave di config, early stopping già scattato."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.cascade.config import load_config  # noqa: E402
from src.cascade.finetune import load_epoch, train_fold  # noqa: E402


class _Toy:
    """Dataset minuscolo nel formato di TrajectoryWindowDataset: serie di 4 canali, 3 classi."""

    def __init__(self, n: int, seed: int) -> None:
        rng = np.random.RandomState(seed)
        self.labels = rng.randint(0, 3, size=n)
        self.items = [
            {
                "x": (rng.randn(6 + i % 3, 4) + lab).astype(np.float32),
                "win_start": 0,
                "win_end": 6,
                "label": int(lab),
                "clip_id": f"c{seed}_{i}",
            }
            for i, lab in enumerate(self.labels)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


class _ToyModel(torch.nn.Module):
    """Stessa interfaccia del classificatore CamFormer: forward(x, pad_mask, lengths, win_start, win_end)."""

    def __init__(self) -> None:
        super().__init__()
        self.enc = torch.nn.Linear(4, 8)
        self.head = torch.nn.Linear(8, 3)

    def forward(self, x, pad_mask, lengths, win_start, win_end):
        keep = (~pad_mask).float()[..., None]
        emb = torch.tanh(self.enc(x)) * keep
        emb = emb.sum(1) / keep.sum(1).clamp(min=1.0)
        return self.head(emb), emb


def _cfg(**train):
    over = ["camformer.train.num_workers=0", "camformer.train.pin_memory=false", "camformer.train.batch_size=4"]
    over += ["camformer.freeze_encoder=false"]  # il modello giocattolo non ha un encoder separato (M17)
    over += [f"camformer.train.{k}={v}" for k, v in train.items()]
    return load_config(overrides=over)


def _fit(cfg, path, key):
    torch.manual_seed(0)
    model = _ToyModel()
    out = train_fold(model, _Toy(24, 1), _Toy(12, 2), cfg, torch.device("cpu"), 0, ckpt_path=path, ckpt_key=key)
    return model, out


def test_resume_continues_from_the_last_epoch(tmp_path):
    path = tmp_path / "ckpt_last.pt"
    _, first = _fit(_cfg(epochs=2, patience=50), path, "k")
    assert len(first["history"]) == 2 and path.exists()
    _, second = _fit(_cfg(epochs=4, patience=50), path, "k")
    h1, h2 = first["history"], second["history"]
    assert list(h2.epoch) == [0, 1, 2, 3]
    cols = ["train_loss", "holdout_loss"]  # le prime due epoche vengono dal checkpoint, non sono riaddestrate
    assert np.allclose(h2.loc[:1, cols].to_numpy(), h1[cols].to_numpy())
    assert torch.load(path, map_location="cpu", weights_only=False)["epoch"] == 3


def test_checkpoint_of_another_config_is_ignored(tmp_path):
    path = tmp_path / "ckpt_last.pt"
    _fit(_cfg(epochs=2, patience=50), path, "a")
    torch.manual_seed(0)
    model = _ToyModel()
    opt = torch.optim.AdamW(model.parameters())
    assert load_epoch(path, model, opt, "b") == (0, None, None, 0, [])
    assert load_epoch(tmp_path / "manca.pt", model, opt, "a")[0] == 0
    path.write_bytes(b"non e' un checkpoint")
    assert load_epoch(path, model, opt, "a")[0] == 0  # file rovinato: si riparte, senza errore


def test_early_stopped_run_is_not_extended(tmp_path):
    path = tmp_path / "ckpt_last.pt"
    cfg = _cfg(epochs=6, patience=1, lr=1e-12, weight_decay=0.0)  # la config esige lr > 0
    _, first = _fit(cfg, path, "k")
    n = len(first["history"])
    assert (
        n == 2
    )  # epoca 0 fissa il best, epoca 1 non migliora (lr ~ 0, pesi invariati in float32) e scatta la patience
    _, second = _fit(cfg, path, "k")
    assert len(second["history"]) == n  # nessuna epoca in più dopo la ripresa
