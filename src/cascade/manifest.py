"""Manifest degli stadi: marcatore di completamento e controllo di coerenza della config.

Ogni stadio, quando ha scritto tutti i suoi output, scrive per ultimo
``<run_dir>/_manifest/<stadio>[.<parte>].json`` con: config effettiva, chiavi di
config che determinano l'output (``STAGE_KEYS``), elenco degli output, commit,
versioni delle librerie, comando lanciato, seed. La ``parte`` distingue gli shard
(``shard003of020``) e le fold (``fold2``).

Regole:
- uno stadio si salta solo se manifest **e** output esistono e la config attuale
  coincide con quella del manifest sulle chiavi dello stadio; se differisce è un
  errore (non si mescolano output di config diverse nella stessa run_dir);
- output presenti senza manifest = esecuzione interrotta: si rifà;
- uno stadio a valle verifica che gli stadi a monte siano completi (tutti gli shard,
  tutte le fold) e coerenti con la config attuale sulle loro chiavi.
"""

from __future__ import annotations

import logging
import platform
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from src.cascade.io import read_json, write_json

log = logging.getLogger(__name__)

MANIFEST_DIR = "_manifest"

# Chiavi di config (prefissi puntati) che determinano l'output di ciascuno stadio.
STAGE_KEYS: Dict[str, List[str]] = {
    "00_inventory": [
        "paths.scenario_csv",
        "paths.takes_json",
        "paths.takes_root",
        "paths.video_pattern",
        "paths.video_glob",
        "paths.traj_pattern",
        "paths.traj_dir",
        "data.traj_source",
        "data.window_s",
        "data.n_windows",
        "data.window_stride_s",
        "data.window_placement",
        "data.duration_source",
        "data.short_take",
        "alignment.mode",
        "alignment.video_start_key",
        "alignment.traj_start_key",
    ],
    "05_folds": ["paths.scenario_csv", "paths.split_sha256", "folds"],
    "10_frames": ["frames", "data.out_of_range"],
    "01_traj": [
        "paths.scenario_csv",
        "paths.split_sha256",
        "paths.pose_roots",
        "paths.video_pattern",
        "paths.video_glob",
        "paths.traj_dir",
        "data.traj_fps",
        "data.pose_src_fps",
        "data.pose_max_gap_frames",
    ],
    "20_camformer": [
        "seed",
        "camformer.seed",
        "data.traj_source",
        "data.traj_fps",
        "data.min_traj_coverage",
        "data.out_of_range",
        "alignment",
        "camformer",
        "paths.camformer_ckpt",
    ],
    "25_egovlp": [
        "egovlp.arch",
        "egovlp.frame_indices",
        "egovlp.norm_mean",
        "egovlp.norm_std",
        "egovlp.pretrained",
        "egovlp.zero_shot",
        "paths.egovlp_ckpt",
    ],
    "28_frame_baseline": ["frame_baseline", "paths.frame_baseline_ckpt"],
    "30_l1_heads": ["seed", "heads"],
    "35_fusion": ["fusion", "camformer.eval_source", "gate.bootstrap", "gate.bootstrap_seed"],
    "40_cost": [
        "cost",
        "camformer.arch",
        "camformer.context_s",
        "egovlp.arch",
        "data.window_s",
        "data.traj_fps",
        "frame_baseline",
    ],
    "45_cache": [
        "data.vertical_axis",
        "data.max_missing_frac",
        "data.window_s",
        "data.min_traj_coverage",
        "data.traj_fps",
        "camformer.eval_source",
        "fusion",
        "frame_baseline",
    ],
    "27_site": ["site_diag", "gate.bacc_min_takes"],
    "50_gate": ["gate", "fusion.mode"],
    "52_seeds": ["gate", "fusion.mode"],
}

# [M07] Un cambio di split o di fold deve invalidare tutto ciò che ne dipende. Senza queste chiavi
# is_done() considera "già fatto" uno stadio a valle calcolato con fold vecchie e lo salta.
SPLIT_KEYS = ["paths.scenario_csv", "paths.split_sha256", "folds"]
SPLIT_DEPENDENT_STAGES = (
    "20_camformer",
    "28_frame_baseline",
    "30_l1_heads",
    "35_fusion",
    "45_cache",
    "50_gate",
    "27_site",
)
for _stage in SPLIT_DEPENDENT_STAGES:
    STAGE_KEYS[_stage] = STAGE_KEYS[_stage] + [k for k in SPLIT_KEYS if k not in STAGE_KEYS[_stage]]

# [I2] Stesso difetto per la definizione delle clip: rifare lo stadio 00 con un'altra finestra, durata o sorgente
# delle traiettorie deve invalidare frame, feature, predizioni e cache. Lo stadio 00 prende anche l'hash dello
# split (un CSV cambiato a parità di nome cambia clips.csv) e il 41 ha finalmente le sue chiavi.
STAGE_KEYS["00_inventory"].append("paths.split_sha256")
STAGE_KEYS["41_cost_e2e"] = [
    "cost_e2e",
    "cost.device",
    "frames",
    "camformer.arch",
    "camformer.context_s",
    "egovlp.arch",
    "egovlp.frame_indices",
    "data.traj_fps",
]
CLIP_KEYS = list(STAGE_KEYS["00_inventory"])
CLIP_DEPENDENT_STAGES = (
    "10_frames",
    "20_camformer",
    "25_egovlp",
    "27_site",
    "28_frame_baseline",
    "30_l1_heads",
    "35_fusion",
    "41_cost_e2e",
    "45_cache",
    "50_gate",
)
for _stage in CLIP_DEPENDENT_STAGES:
    STAGE_KEYS[_stage] = STAGE_KEYS[_stage] + [k for k in CLIP_KEYS if k not in STAGE_KEYS[_stage]]


class ConfigMismatch(RuntimeError):
    """Output esistenti prodotti con una config diversa da quella attuale."""


def _get(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def config_subset(cfg_dict: dict, keys: Sequence[str]) -> Dict[str, object]:
    return {k: _get(cfg_dict, k) for k in keys}


def diff_subsets(old: dict, new: dict) -> List[str]:
    return [f"{k}: {old.get(k)!r} -> {new.get(k)!r}" for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]


def manifest_path(run_dir: Path, stage: str, part: Optional[str] = None) -> Path:
    name = stage if part is None else f"{stage}.{part}"
    return run_dir / MANIFEST_DIR / f"{name}.json"


def library_versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "pyarrow", "sklearn", "lightgbm", "torch", "scipy", "av", "fvcore"):
        m = sys.modules.get(mod)
        if m is not None:
            out[mod] = getattr(m, "__version__", "?")
    return out


def write_manifest(
    run_dir: Path,
    stage: str,
    cfg_dict: dict,
    outputs: Iterable[Path],
    *,
    part: Optional[str] = None,
    commit: str = "unknown",
    extra: Optional[dict] = None,
) -> Path:
    keys = STAGE_KEYS.get(stage, [])
    outs = [str(p) for p in outputs]
    missing = [o for o in outs if not Path(o).exists()]
    if missing:
        raise RuntimeError(f"{stage}: manifest richiesto ma mancano output {missing[:3]}")
    info = {
        "stage": stage,
        "part": part,
        "commit": commit,
        "argv": sys.argv,
        "keys": config_subset(cfg_dict, keys),
        "config": cfg_dict,
        "outputs": outs,
        "versions": library_versions(),
        "seed": cfg_dict.get("seed"),
    }
    if extra:
        info.update(extra)
    p = manifest_path(run_dir, stage, part)
    write_json(p, info)
    return p


def read_manifest(run_dir: Path, stage: str, part: Optional[str] = None) -> Optional[dict]:
    p = manifest_path(run_dir, stage, part)
    return read_json(p) if p.exists() else None


def check_keys(stage: str, manifest: dict, cfg_dict: dict, *, ignore: Sequence[str] = ()) -> None:
    keys = [k for k in STAGE_KEYS.get(stage, []) if not any(k == i or k.startswith(i + ".") for i in ignore)]
    old = {k: v for k, v in manifest.get("keys", {}).items() if k in keys}
    new = config_subset(cfg_dict, keys)
    diff = diff_subsets(old, new)
    if diff:
        raise ConfigMismatch(
            f"lo stadio {stage} (parte {manifest.get('part')}) è stato eseguito con una config diversa: "
            + "; ".join(diff[:6])
            + ". Usa la stessa config, oppure un --run-name diverso, oppure rifai lo stadio con --overwrite."
        )


def is_done(run_dir: Path, stage: str, cfg_dict: dict, outputs: Iterable[Path], *, part: Optional[str] = None) -> bool:
    """True se lo stadio (o la parte) è completo e coerente con la config attuale."""
    m = read_manifest(run_dir, stage, part)
    outs = list(outputs)
    if m is None:
        if outs and any(Path(o).exists() for o in outs):
            log.warning(
                "%s%s: output presenti ma senza manifest (esecuzione interrotta?): li rifaccio",
                stage,
                f" [{part}]" if part else "",
            )
        return False
    if not all(Path(o).exists() for o in outs):
        return False
    check_keys(stage, m, cfg_dict)
    return True


_SHARD = re.compile(r"\.shard(\d{3})of(\d{3})\.json$")


def require_stage(
    run_dir: Path,
    stage: str,
    cfg_dict: dict,
    *,
    parts: Optional[Sequence[str]] = None,
    sharded: bool = False,
    ignore: Sequence[str] = (),
) -> List[dict]:
    """Verifica che uno stadio a monte sia completo e coerente; restituisce i manifest.

    ``parts``: parti attese (es. le fold); ``sharded``: tutte le parti ``shardXXXofNNN``
    con lo stesso NNN devono esserci.
    """
    mdir = run_dir / MANIFEST_DIR
    if sharded:
        found = sorted(mdir.glob(f"{stage}.shard*of*.json")) if mdir.exists() else []
        single = manifest_path(run_dir, stage)
        if not found and single.exists():
            ms = [read_json(single)]
        else:
            if not found:
                raise FileNotFoundError(f"stadio {stage} non eseguito (nessun manifest in {mdir})")
            nums = {}
            for f in found:
                g = _SHARD.search(f.name)
                nums.setdefault(int(g.group(2)), set()).add(int(g.group(1)))
            if len(nums) > 1:
                raise RuntimeError(
                    f"stadio {stage}: manifest con numeri di shard diversi {sorted(nums)}: "
                    "pulisci _manifest/ o rifai lo stadio con un solo --num-shards"
                )
            ((n, ids),) = nums.items()
            missing = sorted(set(range(n)) - ids)
            if missing:
                raise RuntimeError(f"stadio {stage}: mancano gli shard {missing} su {n} (job falliti?)")
            ms = [read_json(f) for f in found]
    elif parts is not None:
        ms = []
        for p in parts:
            m = read_manifest(run_dir, stage, p)
            if m is None:
                raise FileNotFoundError(f"stadio {stage}: parte {p} non completata (manca il manifest)")
            ms.append(m)
    else:
        m = read_manifest(run_dir, stage)
        if m is None:
            raise FileNotFoundError(f"stadio {stage} non completato (manca {manifest_path(run_dir, stage)})")
        ms = [m]
    for m in ms:
        check_keys(stage, m, cfg_dict, ignore=ignore)
    return ms


def build_run_manifest(run_dir: Path) -> dict:
    """Riepilogo della run da tutti i manifest degli stadi: per ogni stadio e parte, commit,
    comando, seed, versioni delle librerie, chiavi di config e output. Va in
    ``<run_dir>/run_manifest.json`` (scritto dallo stadio 55 e da ``60_make_report.sh``)."""
    mdir = Path(run_dir) / MANIFEST_DIR
    stages: Dict[str, dict] = {}
    commits, versions = set(), {}
    for f in sorted(mdir.glob("*.json")) if mdir.exists() else []:
        m = read_json(f)
        commits.add(m.get("commit"))
        versions = m.get("versions", versions)
        stages.setdefault(m["stage"], {})[m.get("part") or "-"] = {
            "commit": m.get("commit"),
            "argv": m.get("argv"),
            "seed": m.get("seed"),
            "keys": m.get("keys"),
            "outputs": m.get("outputs"),
            "versions": m.get("versions"),
        }
    return {
        "run_dir": str(run_dir),
        "commits": sorted(c for c in commits if c),
        "stages": stages,
        "latest_versions": versions,
        "mixed_commits": len({c for c in commits if c}) > 1,
    }


def write_run_manifest(run_dir: Path) -> Path:
    p = Path(run_dir) / "run_manifest.json"
    write_json(p, build_run_manifest(run_dir))
    return p


if __name__ == "__main__":  # python -m src.cascade.manifest <run_dir>
    print(write_run_manifest(Path(sys.argv[1])))
