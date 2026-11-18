"""Configurazione: YAML in ``configs/`` → dataclass tipizzate.

Ordine di merge (l'ultimo vince):
``base.yaml`` → ``camformer.yaml`` → ``egovlp.yaml`` → ``gate.yaml`` → ``gate_v2.yaml`` →
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

BASE_FILES = ("base.yaml", "camformer.yaml", "egovlp.yaml", "gate.yaml", "gate_v2.yaml")
SYNTHETIC_FILE = "synthetic.yaml"


# --------------------------------------------------------------------------- sezioni
@dataclass
class PathsCfg:
    scenario_csv: str = "{repo}/third_party/seeing_without_pixels/data_files/egoexo4d/scenario_cls.csv"
    split_sha256: Optional[str] = None  # SHA256 di scenario_csv: entra nei manifest (M07)
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
    traj_dir: str = "{run}/01_traj"  # uscita dello stadio 01 (M08): un <take_name>.npz per take
    pose_roots: List[str] = field(default_factory=list)  # radici del dataset con split_manifest.csv (stadio 01)


@dataclass
class DataCfg:
    window_s: float = 8.0
    n_windows: int = 1
    window_stride_s: Optional[float] = None
    window_placement: str = "center"  # center | uniform (K finestre su tutta la take)
    duration_source: str = "takes_json"  # takes_json | video | trajectory
    traj_source: str = "aria_csv"  # aria_csv | converted_npz | camformer_npz
    traj_fps: float = 20.0
    pose_src_fps: float = 30.0  # frequenza delle pose per frame di Ego-Exo4D (stadio 01)
    pose_max_gap_frames: int = 15  # buchi sorgente più lunghi: nessuna traiettoria (stadio 01)
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
    group_by: str = "take"  # take | site (M03)
    site_map: Optional[str] = None  # sito -> fold: split.json versionato, non ricalcolato (D6)


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
    freeze_encoder: bool = False  # M17: encoder del checkpoint congelato, si allena solo la testa
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
class EgovlpCfg:
    arch: EgovlpArch = field(default_factory=EgovlpArch)
    zero_shot: ZeroShotCfg = field(default_factory=ZeroShotCfg)
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
class HeadsCfg:
    source: str = "probe"  # probe (25 + logistica) | finetune (stadio 26)
    Cs: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    inner_cv: int = 3
    max_iter: int = 2000


@dataclass
class FusionCfg:
    mode: str = "replace"  # replace (L1 sostituisce L0, M05) | fuse (log-lineare, ablation)
    w_grid_step: float = 0.05


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
class CompareCfg:
    """CompareConfidence (M06): regressione logistica che stima P(L1 ha ragione | z)."""

    Cs: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0])
    inner_cv: int = 3


@dataclass
class GateCfg:
    reliability_alpha: float = 1.0
    emb_pca_dim: int = 16
    models: List[str] = field(default_factory=list)
    logreg_Cs: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    logreg_inner_cv: int = 3
    logreg_class_weight: str = "balanced"  # balanced | none
    lgbm: Dict[str, Any] = field(default_factory=dict)
    random_seeds: int = 10
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    r_step: int = 1
    budgets: List[float] = field(default_factory=lambda: [0.25, 0.5])
    within_pts: float = 0.01  # "entro 1 punto da always_L1" (costo minimo, curva di Pareto)
    improvement_deltas: List[float] = field(default_factory=lambda: [0.0, 0.01])  # δ di I(δ) (M09)
    bacc_min_takes: int = 5  # accuratezza bilanciata: classi con almeno tante take nello split (M14)
    transfer_budgets: List[float] = field(default_factory=lambda: [0.25, 0.5])  # soglie val → test (RQ1)
    operating_budget: float = 0.5
    operating_rule: str = "budget"  # budget | within_pts (N28: minimo costo entro within_pts da always_L1 su val)
    bootstrap: int = 1000
    bootstrap_seed: int = 0
    # N37: unità del bootstrap (take | session) e gruppi della CV interna dei gate (take | session).
    # Con session e senza colonna session nella cache si torna alla take, con un avviso nel log.
    bootstrap_unit: str = "take"
    inner_cv_groups: str = "take"
    comparisons: List[List[str]] = field(default_factory=list)  # coppie [A, B] per A − B (RQ2)
    reference: List[str] = field(default_factory=list)  # gate per la cascata per classe
    compare: CompareCfg = field(default_factory=CompareCfg)


@dataclass
class GateV2HeadsCfg:
    """Teste lineari rifatte dentro la CV annidata (stesso protocollo di scenario_preds/heads.py)."""

    Cs: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    inner_cv: int = 3  # CV per sessione che sceglie C e la temperatura
    max_iter: int = 3000
    class_weight: str = "balanced"  # balanced | none (come la testa di official_v2)


@dataclass
class GateV2Cfg:
    """Gate v2 (N37, stadio 56): CV annidata per sessione su tutte le take, regole a costo e a rischio.

    Livelli per ogni fold esterna k: (1) CV interna per sessione delle teste di CamFormer sul train esterno
    -> predizioni fuori campione su cui si addestra il gate; (2) cross-fitting per sessione del gate sul train
    esterno -> score fuori campione per calibratore e soglie; (3) teste, gate, calibratore e soglie congelati
    e applicati alla fold k."""

    outer_folds: int = 5
    inner_folds: int = 5  # livello 1
    gate_folds: int = 5  # livello 2
    repeats: List[int] = field(default_factory=lambda: [0, 1, 2])  # seed delle partizioni esterne
    outer_l0: str = "refit"  # refit (testa su tutto il train esterno) | inner_mean (media dei modelli interni)
    heads: GateV2HeadsCfg = field(default_factory=GateV2HeadsCfg)
    scorers: List[str] = field(
        default_factory=lambda: ["thr_maxprob", "logreg_A", "logreg_AC", "logreg_ABC", "lgbm_ABC", "class_rule"]
    )
    primary_scorer: str = "thr_maxprob"
    logreg_Cs: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0])
    logreg_inner_cv: int = 3
    lgbm: Dict[str, Any] = field(default_factory=dict)
    reliability_alpha: float = 1.0
    reliability_folds: int = 5  # N38: affidabilità delle righe di fit fuori fold (fold per sessione)
    calibration: str = "platt"  # platt | isotonic
    # N38: righe del train esterno su cui si stimano calibratore e soglie (risk, acc_target, budget). all = come la
    # run del 18/11; pretrain_unseen = solo le take non viste dal pre-addestramento di CamFormer (in uso reale ogni
    # take è nuova per CamFormer). Lo scorer si addestra sempre su tutte le righe. Con meno di
    # calibration_min_errors errori fra quelle righe si torna a all, con avviso e colonna cal_population.
    calibration_population: str = "all"
    calibration_min_errors: int = 10
    calibration_boot: int = 200  # repliche bootstrap per sessione del calibratore (r_up)
    delta: float = 0.1  # r_up = quantile 1 − δ; livello della garanzia della regola a rischio
    betas: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.05, 0.10, 0.20])
    alphas: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.10])
    risk_grid: int = 20  # coperture candidate 1/G, 2/G, ..., 1 (Bonferroni su G)
    max_error_rates: List[float] = field(default_factory=lambda: [0.01])  # regola acc_target (come within_pts)
    budgets: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    primary_rule: str = "cost"
    primary_param: float = 0.05
    subsets: List[str] = field(default_factory=lambda: ["all", "pretrain_unseen", "pretrain_seen", "official_eval"])
    primary_subset: str = "pretrain_unseen"
    comparisons: List[List[str]] = field(
        default_factory=lambda: [
            ["logreg_A", "thr_maxprob"],
            ["logreg_AC", "logreg_A"],
            ["logreg_ABC", "logreg_AC"],
            ["thr_maxprob", "class_rule"],
        ]
    )
    bootstrap: int = 1000
    bootstrap_seed: int = 0
    gmacs_per_window: Dict[str, float] = field(default_factory=lambda: {"L0": 0.121627, "L1": 92.430496})
    extra_gmacs_L0: List[float] = field(default_factory=lambda: [0.0])  # sensibilità: traiettoria per finestra
    extra_gmacs_L1: List[float] = field(default_factory=lambda: [0.0])  # sensibilità: decodifica per finestra


@dataclass
class CostE2ECfg:
    """Costo end-to-end (stadio 41, M10): tempo per fase di L0 e L1, a batch 1."""

    n_clips: int = 200  # campione stratificato per classe
    n_warmup: int = 5  # clip di riscaldamento scartate
    passes: int = 2  # si riporta la seconda passata (file in cache), la prima si dichiara
    seed: int = 0


@dataclass
class SiteDiagCfg:
    """Diagnostiche di sito (stadio 27, M02): sonda lineare sulle CLS di L1, solo train e val."""

    C: float = 0.1  # regolarizzazione della logistica (come check_site.py della v1)
    max_iter: int = 500
    cv_folds: int = 5  # GroupKFold per take e per sessione (leggibilità del sito, CV della classe)


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
    heads: HeadsCfg = field(default_factory=HeadsCfg)
    fusion: FusionCfg = field(default_factory=FusionCfg)
    cost: CostCfg = field(default_factory=CostCfg)
    cost_e2e: CostE2ECfg = field(default_factory=CostE2ECfg)
    site_diag: SiteDiagCfg = field(default_factory=SiteDiagCfg)
    gate: GateCfg = field(default_factory=GateCfg)
    gate_v2: GateV2Cfg = field(default_factory=GateV2Cfg)
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
    _one_of(e, "data.traj_source", d.traj_source, ["aria_csv", "converted_npz", "camformer_npz"])
    if d.pose_src_fps <= 0 or d.pose_max_gap_frames < 1:
        e.append("data.pose_src_fps > 0 e data.pose_max_gap_frames >= 1")
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
    _one_of(e, "folds.group_by", fo.group_by, ["take", "site"])
    if fo.group_by == "site" and not fo.site_map:
        e.append("folds.site_map è obbligatorio con folds.group_by=site")
    _one_of(e, "camformer.arch.pose_encoding", cam.arch.pose_encoding, ["abs7d", "rel9d", "rel9d_grav"])
    _one_of(e, "camformer.eval_source", cam.eval_source, ["fold_ensemble", "full"])
    _one_of(e, "camformer.train.monitor", cam.train.monitor, ["val_loss", "val_acc"])
    _one_of(e, "camformer.train.optimizer", cam.train.optimizer.lower(), ["adamw"])
    if cam.context_s is not None and cam.context_s < d.window_s:
        e.append("camformer.context_s deve essere >= data.window_s (o null)")
    if cam.freeze_encoder and not cam.pretrained:
        e.append("camformer.freeze_encoder richiede camformer.pretrained=true (congelare pesi casuali non ha senso)")
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
    _one_of(e, "fusion.mode", cfg.fusion.mode, ["fuse", "replace"])
    _one_of(e, "heads.source", cfg.heads.source, ["probe"])  # il fine-tuning di L1 non si porta (M12)
    if not 0 < cfg.fusion.w_grid_step <= 1:
        e.append("fusion.w_grid_step in (0, 1]")
    _one_of(e, "cost.counter", cfg.cost.counter, ["fvcore", "torch", "analytic"])
    _one_of(e, "gate.logreg_class_weight", g.logreg_class_weight, ["balanced", "none"])
    if g.operating_rule not in ("budget", "within_pts"):
        e.append("gate.operating_rule: budget | within_pts")
    if not all(0 < b <= 1 for b in g.budgets) or not 0 < g.operating_budget <= 1:
        e.append("gate.budgets e gate.operating_budget in (0, 1]")
    if not 1 <= g.r_step <= 100:
        e.append("gate.r_step in [1, 100]")
    if not g.improvement_deltas or not all(0 <= d < 1 for d in g.improvement_deltas):
        e.append("gate.improvement_deltas: lista non vuota di valori in [0, 1)")
    if g.bacc_min_takes < 1 or not 0 <= g.within_pts < 1:
        e.append("gate.bacc_min_takes >= 1 e gate.within_pts in [0, 1)")
    if not g.transfer_budgets or not all(0 < b <= 1 for b in g.transfer_budgets):
        e.append("gate.transfer_budgets: lista non vuota di valori in (0, 1]")
    if any(len(p) != 2 for p in g.comparisons):
        e.append("gate.comparisons: coppie [A, B]")
    if not g.compare.Cs or g.compare.inner_cv < 2:
        e.append("gate.compare.Cs non vuoto e gate.compare.inner_cv >= 2")
    ce = cfg.cost_e2e
    if ce.n_clips < 1 or ce.n_warmup < 0 or ce.passes < 1:
        e.append("cost_e2e: n_clips >= 1, n_warmup >= 0, passes >= 1")
    sd = cfg.site_diag
    if sd.C <= 0 or sd.max_iter < 1 or sd.cv_folds < 2:
        e.append("site_diag: C > 0, max_iter >= 1, cv_folds >= 2")
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
