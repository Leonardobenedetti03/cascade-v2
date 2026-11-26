"""Common arguments and the context of one pipeline stage.

Every script does::

    parser = stage_parser(__doc__, shards=True)
    args = parser.parse_args(argv)
    ctx = open_stage(args, "10_frames")

``ctx`` holds the config, the run_dir, the stage folder, the labels and the logger.
With ``--synthetic`` the data root becomes ``<run_dir>/synthetic_data``, and the fake dataset is
generated the first time it is needed (generating it twice does nothing).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from src import REPO_ROOT
from src.cascade.data.labels import Labels, load_labels
from src.cascade.utils import manifest, paths
from src.cascade.utils.config import Config, dump_config, load_config
from src.cascade.utils.io import json_default, read_json, write_json  # noqa: F401 - re-exported
from src.cascade.utils.logs import setup_logging
from src.cascade.utils.seed import set_seed


def stage_parser(doc: Optional[str], *, shards: bool = False) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--config",
        action="append",
        default=[],
        type=Path,
        help="YAML aggiuntivo (ripetibile), es. configs/ablations/window_4s.yaml",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="CHIAVE=VALORE",
        help="override puntuale, es. --set data.window_s=16",
    )
    p.add_argument("--run-name", type=str, default=None, help="default: run_name della config")
    p.add_argument("--synthetic", action="store_true", help="usa dati sintetici generati al volo")
    p.add_argument("--overwrite", action="store_true", help="rifà gli output già presenti")
    p.add_argument("--log-level", type=str, default="INFO")
    if shards:
        p.add_argument("--shard-id", type=int, default=0)
        p.add_argument("--num-shards", type=int, default=1)
    return p


@dataclass
class StageContext:
    name: str
    cfg: Config
    run_dir: Path
    stage_dir: Path
    data_root: Path
    labels: Labels
    log: logging.Logger
    synthetic: bool
    overwrite: bool
    shard_id: int = 0
    num_shards: int = 1
    commit: str = "unknown"
    cfg_dict: dict = field(default_factory=dict)
    scenario_labels: Optional[Labels] = None  # the 8 scenarios: inventory (00) and folds (05)

    def path(self, template: str, **fields: object) -> Path:
        fields.setdefault("run_name", self.run_dir.name)  # {run_name}: persistent frames_dir
        return paths.resolve(template, data=self.data_root, run=self.run_dir, **fields)

    @property
    def cam_seed(self) -> int:
        """CamFormer fine-tuning seed. It identifies the outputs of 20, 35, 45 and 50."""
        s = self.cfg.camformer.seed
        return self.cfg.seed if s is None else int(s)

    @property
    def traj_tag(self) -> str:
        """Trajectory rate (ablation B4). It identifies the outputs of 40."""
        return f"hz{self.cfg.data.traj_fps:g}"

    @property
    def cam_tag(self) -> str:
        """L0 variant = fine-tuning seed + trajectory rate. It identifies the outputs of 20, 35, 45 and 50
        (3 seeds and the frame-rate ablation in the same run)."""
        return f"seed{self.cam_seed}_{self.traj_tag}"

    @property
    def l1_tag(self) -> str:
        """Source of L1: always ``probe`` (25 + logistic regression); we do not carry over the fine-tuning (M12)."""
        return self.cfg.heads.source

    @property
    def variant_tag(self) -> str:
        """Identity of a cascade, i.e. the L0 variant. It identifies the outputs of 35, 45 and 50."""
        return self.cam_tag

    def seed_dir(self, stage: str) -> Path:
        """Folder of the L0 variant (stage 20)."""
        return self.run_dir / stage / self.cam_tag

    def variant_dir(self, stage: str) -> Path:
        """Folder of the full variant (stages 35, 45, 50)."""
        return self.run_dir / stage / self.variant_tag

    def l1_dir(self) -> Path:
        return self.run_dir / "30_l1_heads" / self.l1_tag

    @property
    def clips_path(self) -> Path:
        """Clip table (scenario): ``data/clips.csv``, written by stage 00."""
        return self.run_dir / "data" / "clips.csv"

    def read_clips(self):
        """Clips of the task, after checking the stage that produces them."""
        import pandas as pd

        self.require("00_inventory")
        return pd.read_csv(self.clips_path)

    def done(self, outputs: Iterable[Path], part: Optional[str] = None) -> bool:
        """True (and a log line) if the stage is already complete with this config; error if the config differs."""
        outs = list(outputs)
        if self.overwrite:
            return False
        if manifest.is_done(self.run_dir, self.name, self.cfg_dict, outs, part=part):
            self.log.info(
                "già completato con questa config, salto (--overwrite per rifarlo): %s%s",
                self.name,
                f" [{part}]" if part else "",
            )
            return True
        return False

    def finish(self, outputs: Iterable[Path], part: Optional[str] = None, extra: Optional[dict] = None) -> None:
        """Write the stage manifest. Call it last, after all the outputs."""
        manifest.write_manifest(
            self.run_dir, self.name, self.cfg_dict, outputs, part=part, commit=self.commit, extra=extra
        )

    def require(
        self, stage: str, *, parts: Optional[Sequence[str]] = None, sharded: bool = False, ignore: Sequence[str] = ()
    ) -> List[dict]:
        """Check that an upstream stage is complete and matches the current config; stop otherwise."""
        return manifest.require_stage(self.run_dir, stage, self.cfg_dict, parts=parts, sharded=sharded, ignore=ignore)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - git may be missing on the node
        return "unknown"


def open_stage(args: argparse.Namespace, name: str, *, make_stage_dir: bool = True) -> StageContext:
    cfg = load_config(args.config, args.overrides, synthetic=args.synthetic)
    run_name = args.run_name or cfg.run_name
    rdir = paths.run_dir(run_name)
    rdir.mkdir(parents=True, exist_ok=True)
    shard_id = getattr(args, "shard_id", 0)
    num_shards = getattr(args, "num_shards", 1)
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"shard-id {shard_id} fuori da [0, {num_shards})")
    suffix = f".shard{shard_id:03d}of{num_shards:03d}" if num_shards > 1 else ""
    log = setup_logging(rdir / "logs" / f"{name}{suffix}.log", args.log_level)
    stage_dir = rdir / name
    if make_stage_dir:
        stage_dir.mkdir(parents=True, exist_ok=True)
    data_root = rdir / "synthetic_data" if args.synthetic else paths.data_root()
    scenario_labels = load_labels()
    labels = scenario_labels  # in v2 the task is always the scenario (M12)
    set_seed(cfg.seed)
    dump_config(cfg, rdir / "configs" / f"{name}{suffix}.yaml")
    commit = git_commit()
    log.info(
        "stadio %s | run_dir=%s | data_root=%s | synthetic=%s | commit=%s",
        name,
        rdir,
        data_root,
        args.synthetic,
        commit,
    )
    ctx = StageContext(
        name=name,
        cfg=cfg,
        run_dir=rdir,
        stage_dir=stage_dir,
        data_root=data_root,
        labels=labels,
        log=log,
        synthetic=args.synthetic,
        overwrite=args.overwrite,
        shard_id=shard_id,
        num_shards=num_shards,
        commit=commit,
        cfg_dict=cfg.to_dict(),
        scenario_labels=scenario_labels,
    )
    if args.synthetic:
        from src.cascade.utils.synthetic import ensure_synthetic_dataset

        ensure_synthetic_dataset(ctx)
    return ctx


def shard_items(items: Sequence, shard_id: int, num_shards: int) -> List:
    """Deterministic round-robin split: item i goes to shard i % num_shards."""
    return [x for i, x in enumerate(items) if i % num_shards == shard_id]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
