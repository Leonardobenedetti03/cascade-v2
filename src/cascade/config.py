"""Configurazione: YAML in ``configs/`` → dataclass tipizzate.

Ordine di merge (l'ultimo vince):
``base.yaml`` → ``camformer.yaml`` → ``egovlp.yaml`` → ``gate.yaml`` →
``synthetic.yaml`` (solo con ``--synthetic``) → file passati con ``--config`` →
override ``--set sezione.chiave=valore`` (valore parsato come YAML).

Le chiavi sconosciute sono un errore: un refuso nel YAML non deve passare in silenzio.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, get_args, get_origin, get_type_hints

import yaml

from src import CONFIGS_DIR

BASE_FILES = ("base.yaml", "camformer.yaml", "egovlp.yaml", "gate.yaml")
SYNTHETIC_FILE = "synthetic.yaml"


# --------------------------------------------------------------------------- sezioni
@dataclass
class PathsCfg:
    scenario_csv: str = "{repo}/third_party/seeing_without_pixels/data_files/egoexo4d/scenario_cls.csv"
    takes_json: str = "{data}/egoexo4d/takes.json"
    takes_root: str = "{data}/egoexo4d/takes"
    video_pattern: str = "{takes_root}/{take_name}/frame_aligned_videos/downscaled/448/aria01_214-1.mp4"
    video_glob: str = "{takes_root}/{take_name}/frame_aligned_videos/downscaled/448/*214-1*.mp4"
    traj_pattern: str = "{takes_root}/{take_name}/trajectory/closed_loop_trajectory.csv"
    camformer_npz_dir: str = "{data}/camformer/egoexo4d_pretrain"
    camformer_ckpt: str = "{data}/checkpoints/camformer/egoexo4d_dur8.pt"
    egovlp_ckpt: str = "{data}/checkpoints/egovlp/egovlp.pth"
    frame_baseline_ckpt: str = "{data}/checkpoints/torchvision/mobilenet_v3_small-047dcff4.pth"
    frames_dir: str = "{run}/10_frames"
    traj_cache_dir: str = "{run}/traj_cache"


@dataclass
class DataCfg:
    window_s: float = 8.0
    n_windows: int = 1
    window_stride_s: Optional[float] = None
    window_placement: str = "center"  # center | uniform (K finestre su tutta la take)
    duration_source: str = "takes_json"  # takes_json | video | trajectory
    traj_source: str = "aria_csv"  # aria_csv | camformer_npz
    traj_fps: float = 20.0
    min_traj_coverage: float = 0.9
    vertical_axis: int = 2  # asse "su" nel mondo Aria (z)
    out_of_range: str = "drop"  # drop | clamp: finestra oltre video/traiettoria
    short_take: str = "keep"  # keep | drop: take più corte della finestra
    max_missing_frac: float = 0.02  # clip perse tra stadi oltre cui fermarsi
    expected_split_counts: Optional[Dict[str, int]] = field(default_factory=dict)


@dataclass
class AlignmentCfg:
    mode: str = "same_origin"  # same_origin | device_us
    offset_s: float = 0.0
    scale: float = 1.0
    video_start_key: str = "video_start_us"
    traj_start_key: str = "traj_start_us"


@dataclass
class FramesCfg:
    n_frames: int = 16
    pre_resize: int = 256
    crop: int = 256
    size: int = 224
    pre_resize_mode: str = "bicubic"  # come lo scale di ffmpeg usato da EgoVLP
    pre_resize_antialias: bool = True
    final_antialias: bool = False  # torchvision 0.13 (EgoVLP): niente antialias sui tensori
    rotate_k90: int = 0
    decoder: str = "auto"  # auto | decord | pyav


@dataclass
class FoldsCfg:
    n_folds: int = 5
    inner_holdout_frac: float = 0.1
    seed: int = 0


@dataclass
class CamformerArch:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    output_dim: int = 512
    pose_encoding: str = "rel9d_grav"


@dataclass
class CamformerTrain:
    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-3
    head_lr_mult: float = 10.0
    epochs: int = 30
    batch_size: int = 64
    patience: int = 5
    monitor: str = "val_loss"  # val_loss | val_acc
    temporal_jitter_s: float = 0.0
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: Optional[int] = 2
    persistent_workers: bool = False
    grad_clip: float = 1.0


@dataclass
class CamformerCfg:
    arch: CamformerArch = field(default_factory=CamformerArch)
    train: CamformerTrain = field(default_factory=CamformerTrain)
    context_s: Optional[float] = None  # None = uguale alla finestra
    pretrained: bool = True
    init_head_from_ckpt: bool = False
    eval_source: str = "fold_ensemble"  # fold_ensemble | full
    seed: Optional[int] = None  # seed del fine-tuning (None = seed globale)
    mc_dropout_passes: int = 0
    batch_size_eval: int = 256
    device: str = "auto"


@dataclass
class EgovlpArch:
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    num_frames: int = 4
    proj_dim: int = 256


@dataclass
class ZeroShotCfg:
    enabled: bool = True
    text_model: str = "distilbert-base-uncased"
    temperature: float = 0.05
    prompts: Dict[str, str] = field(default_factory=dict)


@dataclass
class EgovlpFinetuneCfg:
    """Fine-tuning end-to-end di TimeSformer (fase D): opzione, mai di default."""

    num_frames: int = 16
    frame_indices: Optional[List[int]] = None  # None = tutti i frame estratti (16)
    lr_backbone: float = 3e-5  # quello del fine-tuning di EgoVLP
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 20
    warmup_epochs: float = 1.0
    batch_size: int = 4
    grad_accum: int = 1
    amp: bool = True
    grad_checkpointing: bool = False
    freeze_blocks: int = 0
    label_smoothing: float = 0.0
    augment: bool = True
    patience: int = 3


@dataclass
class EgovlpCfg:
    arch: EgovlpArch = field(default_factory=EgovlpArch)
    zero_shot: ZeroShotCfg = field(default_factory=ZeroShotCfg)
    finetune: EgovlpFinetuneCfg = field(default_factory=EgovlpFinetuneCfg)
    frame_indices: List[int] = field(default_factory=lambda: [2, 6, 9, 13])
    norm_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    norm_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    pretrained: bool = True
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    device: str = "auto"


@dataclass
class FrameBaselineCfg:
    enabled: bool = False
    arch: str = "mobilenet_v3_small"
    frame_index: int = 8  # frame dei 16 estratti (centro della finestra)
    size: int = 112
    pretrained: bool = True
    batch_size: int = 64
    device: str = "auto"
    w_grid_step: float = 0.1  # griglia per le fusioni con L½ (stadio 45)
    r_step: int = 5  # griglia (r1, r2) della cascata a tre livelli


@dataclass
class TaskCfg:
    """Task di classificazione: scenario (principale) o keystep (secondario, fase E4)."""

    name: str = "scenario"  # scenario | keystep
    labels: str = "{repo}/configs/labels.yaml"  # classi del task (00 e 05 usano sempre gli scenari)


@dataclass
class KeystepCfg:
    """Task keystep (fase E4): solo struttura. TODO(HPC): segmenti e classi dalle annotazioni."""

    segments_csv: str = "{data}/keystep/segments.csv"  # CSV normalizzato, vedi src/cascade/keystep.py
    window: str = "center"  # center (finestra D centrata nel segmento) | segment (finestra = segmento)
    min_segment_s: float = 1.0


@dataclass
class StreamingCfg:
    """Variante causale (fase E3): L0 su prefissi crescenti della finestra (stadi 22 e 57)."""

    enabled: bool = False
    prefixes: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])  # frazioni di D; 1 = cache
    recalibrate: bool = True  # temperatura per prefisso sull'holdout della fold
    tau_stop: List[float] = field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    gate: str = "thr_maxprob"  # score dell'escalation a fine finestra: thr_maxprob o un gate dello stadio 50
    r_step: int = 5
    within_pts: float = 1.0  # punto di lavoro: latenza minima entro questi punti di accuratezza


@dataclass
class HeadsCfg:
    source: str = "probe"  # probe (25 + logistica) | finetune (stadio 26)
    Cs: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    inner_cv: int = 3
    max_iter: int = 2000


@dataclass
class FusionCfg:
    mode: str = "fuse"  # fuse | replace | learned
    w_grid_step: float = 0.05
    learned_features: List[str] = field(default_factory=lambda: ["cam_logprobs", "vid_cls"])
    compare_learned: bool = True  # calcola comunque la fusione appresa per il confronto


@dataclass
class CostCfg:
    timesformer_frames: List[int] = field(default_factory=lambda: [4, 8, 16])
    latency_reps: int = 20
    gate_gmacs: float = 0.0
    counter: str = "fvcore"  # fvcore | torch
    override_gmacs: Optional[Dict[str, float]] = None  # {L0: ..., L1: ...}: solo per --synthetic
    device: str = "auto"
    expected: Optional[Dict[str, float]] = field(default_factory=dict)


@dataclass
class LevelMinus1Cfg:
    enabled: bool = False
    feature: str = "mot_energy"
    threshold_quantile: float = 0.05
    action: str = "escalate"  # escalate | accept


@dataclass
class BayesCfg:
    inference: str = "svi"  # svi | nuts
    steps: int = 2000  # SVI
    lr: float = 0.01
    num_samples: int = 300
    num_warmup: int = 300  # NUTS
    prior_scale: float = 1.0  # deviazione standard del prior su β
    score: str = "mean"  # mean | quantile (score "pessimistico")
    quantile: float = 0.9


@dataclass
class GateCfg:
    reliability: str = "empirical"  # empirical | beta
    reliability_alpha: float = 1.0
    beta_n0: float = 20.0  # pseudo-osservazioni del prior Beta
    beta_q: float = 0.1  # quantile della posterior (10° percentile)
    prior_confusion: Optional[str] = "{repo}/configs/priors/camformer_confusion.csv"
    bayes: BayesCfg = field(default_factory=BayesCfg)
    emb_pca_dim: int = 16
    models: List[str] = field(default_factory=list)
    logreg_Cs: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    logreg_inner_cv: int = 3
    logreg_class_weight: str = "balanced"  # balanced | none
    ridge_alphas: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0, 100.0])
    lgbm: Dict[str, Any] = field(default_factory=dict)
    random_seeds: int = 10
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    r_step: int = 1
    budgets: List[float] = field(default_factory=lambda: [0.25, 0.5])
    within_pts: float = 0.01
    operating_budget: float = 0.5
    bootstrap: int = 1000
    bootstrap_seed: int = 0
    comparisons: List[List[str]] = field(default_factory=list)  # coppie [A, B] per A − B (RQ2)
    reference: List[str] = field(default_factory=list)  # gate per la cascata per classe
    level_minus1: LevelMinus1Cfg = field(default_factory=LevelMinus1Cfg)


@dataclass
class SyntheticCfg:
    takes_per_class: Dict[str, int] = field(default_factory=lambda: {"train": 12, "val": 4, "test": 4})
    duration_s: List[float] = field(default_factory=lambda: [12.0, 20.0])
    traj_hz: float = 100.0
    video_fps: float = 10.0
    video_size: int = 64
    traj_offset_s: float = 0.4
    missing_video: int = 1
    missing_traj: int = 1
    video_confusion: float = 0.08
    seed: int = 0


@dataclass
class Config:
    run_name: str = "mvp_D8"
    seed: int = 0
    paths: PathsCfg = field(default_factory=PathsCfg)
    data: DataCfg = field(default_factory=DataCfg)
    alignment: AlignmentCfg = field(default_factory=AlignmentCfg)
    frames: FramesCfg = field(default_factory=FramesCfg)
    folds: FoldsCfg = field(default_factory=FoldsCfg)
    camformer: CamformerCfg = field(default_factory=CamformerCfg)
    egovlp: EgovlpCfg = field(default_factory=EgovlpCfg)
    frame_baseline: FrameBaselineCfg = field(default_factory=FrameBaselineCfg)
    streaming: StreamingCfg = field(default_factory=StreamingCfg)
    task: TaskCfg = field(default_factory=TaskCfg)
    keystep: KeystepCfg = field(default_factory=KeystepCfg)
    heads: HeadsCfg = field(default_factory=HeadsCfg)
    fusion: FusionCfg = field(default_factory=FusionCfg)
    cost: CostCfg = field(default_factory=CostCfg)
    gate: GateCfg = field(default_factory=GateCfg)
    synthetic: SyntheticCfg = field(default_factory=SyntheticCfg)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- merge
def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ricorsivo: i dict si fondono, il resto si sovrascrive."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_set(item: str) -> Dict[str, Any]:
    if "=" not in item:
        raise ValueError(f"override non valido (atteso chiave=valore): {item!r}")
    key, raw = item.split("=", 1)
    value = yaml.safe_load(raw)
    node: Dict[str, Any] = {}
    cur = node
    parts = key.strip().split(".")
    for p in parts[:-1]:
        cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
    return node


def _coerce(value: Any, typ: Any, where: str) -> Any:
    """Converte ``value`` al tipo dichiarato o solleva ``TypeError`` con il percorso.

    Serve soprattutto per i float scritti come ``1e-4``: PyYAML (YAML 1.1) li legge come
    stringhe, sia nei file sia in ``--set``.
    """
    if typ is Any:
        return value
    origin = get_origin(typ)
    if origin is Union:
        args = [a for a in get_args(typ) if a is not type(None)]
        if value is None:
            return None
        return _coerce(value, args[0], where)
    if origin in (list, List):
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{where}: attesa una lista, trovato {value!r}")
        (inner,) = get_args(typ) or (Any,)
        return [_coerce(v, inner, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin in (dict, Dict):
        if not isinstance(value, dict):
            raise TypeError(f"{where}: atteso un dict, trovato {value!r}")
        _, inner = get_args(typ) or (Any, Any)
        return {k: _coerce(v, inner, f"{where}.{k}") for k, v in value.items()}
    if typ is bool:
        if isinstance(value, bool):
            return value
        raise TypeError(f"{where}: atteso true/false, trovato {value!r}")
    if typ is int:
        if isinstance(value, bool):
            raise TypeError(f"{where}: atteso un intero, trovato {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
        raise TypeError(f"{where}: atteso un intero, trovato {value!r}")
    if typ is float:
        if isinstance(value, bool):
            raise TypeError(f"{where}: atteso un numero, trovato {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        raise TypeError(f"{where}: atteso un numero, trovato {value!r}")
    if typ is str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        raise TypeError(f"{where}: attesa una stringa, trovato {value!r}")
    return value


def _build(cls: type, data: Dict[str, Any], where: str) -> Any:
    if not isinstance(data, dict):
        raise TypeError(f"{where or 'config'}: atteso un dict, trovato {type(data).__name__}")
    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"chiavi sconosciute in {where or 'config'}: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        typ = hints[name]
        if is_dataclass(typ):
            kwargs[name] = _build(typ, value or {}, f"{where}{name}.")
        else:
            kwargs[name] = _coerce(value, typ, f"{where}{name}")
    return cls(**kwargs)


def _one_of(errors: List[str], where: str, value: Any, allowed: Sequence[Any]) -> None:
    if value not in allowed:
        errors.append(f"{where}={value!r} non ammesso (valori: {list(allowed)})")


def validate_config(cfg: Config) -> None:
    """Controlli di dominio: raccoglie tutti gli errori e li riporta insieme (fail fast)."""
    e: List[str] = []
    d, a, fr, fo, cam, ev, g = cfg.data, cfg.alignment, cfg.frames, cfg.folds, cfg.camformer, cfg.egovlp, cfg.gate
    _one_of(e, "data.duration_source", d.duration_source, ["takes_json", "video", "trajectory"])
    _one_of(e, "data.traj_source", d.traj_source, ["aria_csv", "camformer_npz"])
    _one_of(e, "data.out_of_range", d.out_of_range, ["drop", "clamp"])
    _one_of(e, "data.short_take", d.short_take, ["keep", "drop"])
    _one_of(e, "data.vertical_axis", d.vertical_axis, [0, 1, 2])
    if d.window_s <= 0 or d.n_windows < 1 or d.traj_fps <= 0:
        e.append("data.window_s > 0, data.n_windows >= 1, data.traj_fps > 0")
    _one_of(e, "data.window_placement", d.window_placement, ["center", "uniform"])
    if d.window_stride_s is not None and (d.window_stride_s <= 0 or d.window_placement != "center"):
        e.append("data.window_stride_s > 0, e solo con data.window_placement = center")
    if not 0 < d.min_traj_coverage <= 1 or not 0 <= d.max_missing_frac <= 1:
        e.append("data.min_traj_coverage in (0, 1] e data.max_missing_frac in [0, 1]")
    _one_of(e, "alignment.mode", a.mode, ["same_origin", "device_us"])
    if a.scale <= 0:
        e.append("alignment.scale > 0")
    _one_of(e, "frames.decoder", fr.decoder, ["auto", "decord", "pyav"])
    _one_of(e, "frames.pre_resize_mode", fr.pre_resize_mode, ["bilinear", "bicubic"])
    if not (fr.size <= fr.crop <= fr.pre_resize) or fr.n_frames < 1:
        e.append("frames: size <= crop <= pre_resize e n_frames >= 1")
    if fo.n_folds < 2 or not 0 < fo.inner_holdout_frac < 0.5:
        e.append("folds.n_folds >= 2 e folds.inner_holdout_frac in (0, 0.5)")
    _one_of(e, "camformer.arch.pose_encoding", cam.arch.pose_encoding, ["abs7d", "rel9d", "rel9d_grav"])
    _one_of(e, "camformer.eval_source", cam.eval_source, ["fold_ensemble", "full"])
    _one_of(e, "camformer.train.monitor", cam.train.monitor, ["val_loss", "val_acc"])
    _one_of(e, "camformer.train.optimizer", cam.train.optimizer.lower(), ["adamw"])
    if cam.context_s is not None and cam.context_s < d.window_s:
        e.append("camformer.context_s deve essere >= data.window_s (o null)")
    if cam.mc_dropout_passes < 0 or cam.train.lr <= 0:
        e.append("camformer.mc_dropout_passes >= 0 e camformer.train.lr > 0")
    if len(ev.frame_indices) != ev.arch.num_frames:
        e.append(
            f"egovlp.frame_indices ha {len(ev.frame_indices)} elementi, egovlp.arch.num_frames=" f"{ev.arch.num_frames}"
        )
    if any(i < 0 or i >= fr.n_frames for i in ev.frame_indices):
        e.append(f"egovlp.frame_indices {ev.frame_indices} fuori da [0, frames.n_frames={fr.n_frames})")
    if len(ev.norm_mean) != 3 or len(ev.norm_std) != 3:
        e.append("egovlp.norm_mean e norm_std: 3 valori")
    fb = cfg.frame_baseline
    if not 0 <= fb.frame_index < fr.n_frames or fb.size < 16 or not 1 <= fb.r_step <= 100:
        e.append("frame_baseline: frame_index in [0, n_frames), size >= 16, r_step in [1, 100]")
    _one_of(e, "fusion.mode", cfg.fusion.mode, ["fuse", "replace", "learned"])
    _one_of(e, "heads.source", cfg.heads.source, ["probe", "finetune"])
    ft = ev.finetune
    if ft.frame_indices is not None and (
        len(ft.frame_indices) != ft.num_frames or any(i < 0 or i >= fr.n_frames for i in ft.frame_indices)
    ):
        e.append("egovlp.finetune.frame_indices: num_frames indici in [0, frames.n_frames)")
    if ft.frame_indices is None and ft.num_frames != fr.n_frames:
        e.append("egovlp.finetune.num_frames != frames.n_frames: specificare finetune.frame_indices")
    if not 0 < cfg.fusion.w_grid_step <= 1:
        e.append("fusion.w_grid_step in (0, 1]")
    _one_of(e, "cost.counter", cfg.cost.counter, ["fvcore", "torch", "analytic"])
    _one_of(e, "gate.reliability", g.reliability, ["empirical", "beta"])
    _one_of(e, "gate.logreg_class_weight", g.logreg_class_weight, ["balanced", "none"])
    _one_of(e, "gate.bayes.inference", g.bayes.inference, ["svi", "nuts"])
    _one_of(e, "gate.bayes.score", g.bayes.score, ["mean", "quantile"])
    if g.beta_n0 <= 0 or not 0 < g.beta_q < 1 or not 0 < g.bayes.quantile < 1:
        e.append("gate.beta_n0 > 0, gate.beta_q e gate.bayes.quantile in (0, 1)")
    _one_of(e, "gate.level_minus1.action", g.level_minus1.action, ["escalate", "accept"])
    if not all(0 < b <= 1 for b in g.budgets) or not 0 < g.operating_budget <= 1:
        e.append("gate.budgets e gate.operating_budget in (0, 1]")
    if not 1 <= g.r_step <= 100:
        e.append("gate.r_step in [1, 100]")
    if any(len(p) != 2 for p in g.comparisons):
        e.append("gate.comparisons: coppie [A, B]")
    _one_of(e, "task.name", cfg.task.name, ["scenario", "keystep"])
    _one_of(e, "keystep.window", cfg.keystep.window, ["center", "segment"])
    if cfg.keystep.min_segment_s <= 0:
        e.append("keystep.min_segment_s > 0")
    st = cfg.streaming
    pf = list(st.prefixes)
    if not pf or pf != sorted(set(pf)) or not all(0 < f <= 1 for f in pf):
        e.append("streaming.prefixes: frazioni crescenti e distinte in (0, 1]")
    if not all(0 < t <= 1 for t in st.tau_stop) or not 1 <= st.r_step <= 100 or st.within_pts < 0:
        e.append("streaming: tau_stop in (0, 1], r_step in [1, 100], within_pts >= 0")
    if e:
        raise ValueError("config non valida:\n  - " + "\n  - ".join(e))


def config_from_dict(data: Dict[str, Any]) -> Config:
    cfg = _build(Config, data, "")
    validate_config(cfg)
    return cfg


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(
    extra_files: Sequence[Path] = (),
    overrides: Sequence[str] = (),
    *,
    synthetic: bool = False,
    configs_dir: Path = CONFIGS_DIR,
) -> Config:
    """Carica e valida la configurazione completa."""
    merged: Dict[str, Any] = {}
    files: List[Path] = [configs_dir / f for f in BASE_FILES]
    if synthetic:
        files.append(configs_dir / SYNTHETIC_FILE)
    files.extend(Path(p) for p in extra_files)
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"file di config mancante: {path}")
        merged = deep_merge(merged, load_yaml(path))
    for item in overrides:
        merged = deep_merge(merged, _parse_set(item))
    return config_from_dict(merged)


def dump_config(cfg: Config, path: Path) -> None:
    from src.cascade.io import write_text

    write_text(path, yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True))
