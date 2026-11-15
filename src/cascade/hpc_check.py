"""Controlli sui percorsi di dati e pesi di una run (usato da ``hpc/check_cluster.sh``).

Non carica torch né i dati: espande i template di ``paths.*`` come fanno gli stadi e verifica che
file e cartelle ci siano. Si lancia sul login node prima della prima run::

    python -m src.cascade.hpc_check --config configs/hpc/site_v2.yaml

Esce con 1 se manca qualcosa di indispensabile.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence

from src.cascade import paths
from src.cascade.config import Config, load_config
from src.cascade.traj_convert import POSE_MANIFEST

OK, WARN, FAIL = "ok", "warn", "fail"
_LABEL = {OK: "ok        ", WARN: "attenzione", FAIL: "FALLITO   "}


class Check(NamedTuple):
    name: str
    status: str  # ok | warn | fail
    detail: str


def _file(name: str, path: Path) -> Check:
    if path.is_file():
        return Check(name, OK, str(path))
    return Check(name, FAIL, f"manca il file {path}")


def _scenario_checks(cfg: Config, scenario: Path, takes_root: Path) -> List[Check]:
    out: List[Check] = []
    digest = hashlib.sha256(scenario.read_bytes()).hexdigest()
    expected = cfg.paths.split_sha256
    if not expected:
        out.append(Check("hash dello split", WARN, f"split_sha256 non impostato (SHA256 del file: {digest})"))
    elif digest == expected:
        out.append(Check("hash dello split", OK, digest[:16]))
    else:
        out.append(Check("hash dello split", FAIL, f"{digest[:16]} nel file, {expected[:16]} in config"))
    with open(scenario, newline="") as f:
        names = [row["take_name"] for row in csv.DictReader(f)]
    present = sum((takes_root / n).is_dir() for n in names) if takes_root.is_dir() else 0
    detail = f"{present}/{len(names)} take trovate in {takes_root}"
    if present == len(names):
        out.append(Check("take sul disco", OK, detail))
    elif present:
        out.append(Check("take sul disco", WARN, detail + " (lo stadio 00 elenca le mancanti in missing_takes.csv)"))
    else:
        out.append(Check("take sul disco", FAIL, detail))
    return out


def _pose_root_checks(roots: List[Path]) -> List[Check]:
    """Radici delle pose per lo stadio 01 (M08): ognuna con ``split_manifest.csv`` (B3)."""
    if not roots:
        return [Check("radici delle pose", FAIL, "paths.pose_roots è vuoto (TODO(HPC)): lo stadio 01 si fermerebbe")]
    out = []
    for i, root in enumerate(roots):
        mf = root / POSE_MANIFEST
        out.append(
            _file(f"radice delle pose {i}", mf)
            if root.is_dir()
            else Check(f"radice delle pose {i}", FAIL, f"manca la cartella {root}")
        )
    return out


def path_checks(cfg: Config, run_name: str) -> List[Check]:
    """Verifica i file indispensabili per ``run_name`` con la config ``cfg``."""
    data, run = paths.data_root(), paths.run_dir(run_name)

    def res(template: str) -> Path:
        return paths.resolve(template, data=data, run=run, run_name=run_name)

    p = cfg.paths
    tj = res(p.takes_json)
    if tj.is_file():
        out = [Check("takes.json", OK, str(tj))]
    elif cfg.alignment.mode == "device_us":
        out = [Check("takes.json", FAIL, f"manca {tj}: serve con alignment.mode=device_us")]
    else:
        out = [Check("takes.json", WARN, f"manca {tj}: lo stadio 00 prende la durata dal video (N5)")]
    takes_root = res(p.takes_root)
    out.append(Check("cartella delle take", OK if takes_root.is_dir() else FAIL, str(takes_root)))
    if cfg.egovlp.pretrained:
        out.append(_file("pesi EgoVLP", res(p.egovlp_ckpt)))
    if cfg.camformer.pretrained:
        out.append(_file("pesi CamFormer", res(p.camformer_ckpt)))
    if cfg.frame_baseline.enabled and cfg.frame_baseline.pretrained:
        out.append(_file("pesi MobileNetV3-Small", res(p.frame_baseline_ckpt)))
    if cfg.data.traj_source == "converted_npz":
        out += _pose_root_checks([res(r) for r in p.pose_roots])
    scenario = res(p.scenario_csv)
    out.append(_file("split per sito (CSV)", scenario))
    if scenario.is_file():
        out += _scenario_checks(cfg, scenario, takes_root)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", default=[], type=Path, help="YAML aggiuntivo (ripetibile)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="CHIAVE=VALORE")
    ap.add_argument("--run-name", type=str, default=None, help="default: run_name della config")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    run_name = args.run_name or cfg.run_name
    checks = path_checks(cfg, run_name)
    for c in checks:
        print(f"  {_LABEL[c.status]} {c.name}: {c.detail}")
    return 1 if any(c.status == FAIL for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
