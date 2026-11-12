"""Dataset torch delle clip video già estratte (frame ``.npy`` dello stadio 10).

Ogni elemento: ``{"clip_id", "frames": uint8 [k, H, W, 3], "label"}`` con i k frame
scelti da ``egovlp.frame_indices`` tra i 16 salvati. La normalizzazione è in
``src/cascade/models/egovlp.normalize`` (fatta per batch).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd


class ClipFramesDataset:
    def __init__(self, clips: pd.DataFrame, frames_dir: Path, frame_indices: Sequence[int]) -> None:
        self.frames_dir = Path(frames_dir)
        self.idx = list(frame_indices)
        self.rows = []
        self.missing: List[str] = []
        for r in clips.itertuples(index=False):
            p = self.frames_dir / f"{r.clip_id}.npy"
            if p.exists():
                self.rows.append((r.clip_id, int(r.label_idx), p))
            else:
                self.missing.append(r.clip_id)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def clip_ids(self) -> List[str]:
        return [r[0] for r in self.rows]

    def __getitem__(self, i: int) -> dict:
        cid, lab, p = self.rows[i]
        arr = np.load(p, mmap_mode="r")
        if max(self.idx) >= len(arr):
            raise IndexError(f"{p}: {len(arr)} frame, richiesti indici {self.idx}")
        return {"clip_id": cid, "frames": np.asarray(arr[self.idx]), "label": lab}


def collate_frames(batch):
    return {
        "clip_id": [b["clip_id"] for b in batch],
        "frames": np.stack([b["frames"] for b in batch]),
        "label": np.array([b["label"] for b in batch]),
    }
