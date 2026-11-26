"""Class order. The only source is ``configs/labels.yaml``.

Every file with logits or probabilities stores the class names (in the parquet metadata), and
:func:`assert_class_order` checks them against ``labels.yaml`` when the file is loaded. We always map
the CSV by task name, never by its ``label`` column, which uses a different order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import yaml

from src import CONFIGS_DIR

LABELS_FILE = CONFIGS_DIR / "labels.yaml"
PROCEDURAL = "procedural"
PHYSICAL = "physical"


@dataclass(frozen=True)
class Labels:
    names: tuple
    csv_tasks: tuple
    kinds: tuple

    @property
    def num_classes(self) -> int:
        return len(self.names)

    @property
    def procedural_mask(self) -> np.ndarray:
        return np.array([k == PROCEDURAL for k in self.kinds], dtype=bool)

    def index_of_task(self, task: str) -> int:
        try:
            return self.csv_tasks.index(task)
        except ValueError:
            raise KeyError(f"task {task!r} non presente in labels.yaml ({list(self.csv_tasks)})") from None

    def task_to_index(self) -> Dict[str, int]:
        return {t: i for i, t in enumerate(self.csv_tasks)}


def load_labels(path: Path = LABELS_FILE) -> Labels:
    with open(path) as f:
        raw = yaml.safe_load(f)
    classes = raw["classes"]
    names = tuple(c["name"] for c in classes)
    tasks = tuple(c.get("csv_task", c["name"]) for c in classes)
    kinds = tuple(c["kind"] for c in classes)
    if len(set(names)) != len(names) or len(set(tasks)) != len(tasks):
        raise ValueError("labels.yaml: nomi o csv_task duplicati")
    bad = {k for k in kinds} - {PROCEDURAL, PHYSICAL}
    if bad:
        raise ValueError(f"labels.yaml: kind non validi {bad}")
    return Labels(names=names, csv_tasks=tasks, kinds=kinds)


def assert_class_order(found: Sequence[str], labels: Labels, *, where: str = "") -> None:
    """Raise an error if the saved class order is not the one in labels.yaml."""
    found = tuple(found)
    if found != labels.names:
        raise ValueError(
            f"ordine delle classi incoerente{' in ' + where if where else ''}: "
            f"trovato {list(found)}, atteso {list(labels.names)}"
        )


def assert_num_classes(arr: np.ndarray, labels: Labels, *, where: str = "") -> None:
    if arr.ndim != 2 or arr.shape[1] != labels.num_classes:
        raise ValueError(f"{where}: attesa shape [N, {labels.num_classes}], trovata {arr.shape}")
