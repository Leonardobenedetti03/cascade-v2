#!/bin/bash
# hpc/submit.sh - sottomette UNO stadio con le risorse giuste. Stampa il job id.
#   RUN_NAME=site_v2 hpc/submit.sh <stadio> [--after ID[:ID...]] [-- argomenti per lo script]
# Esempi:
#   RUN_NAME=site_v2 hpc/submit.sh 10                                  # array di NUM_SHARDS shard
#   RUN_NAME=site_v2 hpc/submit.sh 20 --after 123 -- --set camformer.seed=1
#   RUN_NAME=dbg_smoke RUN_CONFIG=site_v2 hpc/submit.sh 00 -- --inspect   # run di prova
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh
die() { echo "submit: $*" >&2; exit 2; }

STAGE="${1:?uso: hpc/submit.sh <stadio> [--after ID[:ID]] [-- argomenti]}"; shift
AFTER=""
if [[ "${1:-}" == "--after" ]]; then AFTER="$2"; shift 2; fi
if [[ "${1:-}" == "--" ]]; then shift; fi
export RUN_NAME="${RUN_NAME:?RUN_NAME mancante (site_v2 | id_v2 | dbg_*)}"
export RUN_CONFIG="${RUN_CONFIG:-$RUN_NAME}"
export STAGE_ARGS="$*"
[[ -f "configs/hpc/$RUN_CONFIG.yaml" ]] || die "manca configs/hpc/$RUN_CONFIG.yaml"
[[ -n "$DL4CV_ROOT" ]] || die "TODO(HPC): DL4CV_ROOT in hpc/env.sh o hpc/env.local.sh"

# stadio -> script, nodo, CPU, memoria, tempo massimo, tipo di array.
# Valori di partenza (stime): correggerli con i tempi del primo run (sacct).
declare -A TABLE=(
  [00]="00_inventory.py              cpu 4 8G  00:30:00 none"
  [01]="01_convert_trajectories.py   cpu 4 8G  01:00:00 none"
  [05]="05_folds.py                  cpu 1 4G  00:10:00 none"
  [10]="10_extract_frames.py         cpu 4 8G  01:00:00 shard"
  [15]="15_alignment_check.py        cpu 2 8G  00:20:00 none"
  [20]="20_camformer_finetune.py     gpu 4 16G 01:00:00 fold"
  [25]="25_egovlp_features.py        gpu 8 32G 01:00:00 none"
  [27]="27_site_diagnostics.py       cpu 4 16G 00:30:00 none"
  [28]="28_frame_baseline.py         gpu 4 16G 00:30:00 none"
  [30]="30_l1_heads.py               cpu 8 16G 00:30:00 none"
  [35]="35_fusion.py                 cpu 2 8G  00:15:00 none"
  [40]="40_cost.py                   gpu 2 16G 00:15:00 none"
  [41]="41_cost_e2e.py               gpu 4 16G 00:45:00 none"
  [45]="45_build_cache.py            cpu 4 16G 00:30:00 none"
  [50]="50_gate_eval.py              cpu 8 16G 01:00:00 none"
  [52]="52_aggregate_seeds.py        cpu 1 4G  00:10:00 none"
  [55]="55_report.py                 cpu 2 8G  00:20:00 none"
)
[[ -n "${TABLE[$STAGE]:-}" ]] || die "stadio sconosciuto: $STAGE"
read -r SCRIPT KIND CPUS MEM TIME ARRAY <<< "${TABLE[$STAGE]}"
export STAGE_SCRIPT="$SCRIPT" ARRAY_KIND="$ARRAY"
TIME="${TIME_OVERRIDE:-$TIME}"; MEM="${MEM_OVERRIDE:-$MEM}"

# Preflight sul login node (qui la rete c'e'): clone pulito e allineato a origin/main.
HEAD=$(git rev-parse HEAD)
if [[ "$RUN_NAME" != dbg_* ]]; then
  git fetch --quiet origin main
  [[ -z "$(git status --porcelain)" ]] || die "clone non pulito: git status"
  [[ "$HEAD" == "$(git rev-parse origin/main)" ]] || die "HEAD diverso da origin/main: git pull --ff-only"
fi
export EXPECTED_COMMIT="$HEAD"

OPTS=(--parsable --job-name="cv2_${RUN_NAME}_${STAGE}" --cpus-per-task="$CPUS" --mem="$MEM"
      --time="$TIME" --export=ALL --kill-on-invalid-dep=yes)
if [[ "$KIND" == gpu ]]; then
  OPTS+=(--partition="${GPU_PARTITION:?TODO(HPC) GPU_PARTITION}" --gres="$GPU_GRES")
else
  OPTS+=(--partition="${CPU_PARTITION:?TODO(HPC) CPU_PARTITION}")
fi
if [[ -n "$SLURM_ACCOUNT" ]]; then OPTS+=(--account="$SLURM_ACCOUNT"); fi
if [[ -n "$AFTER" ]]; then OPTS+=(--dependency="afterok:$AFTER"); fi
case "$ARRAY" in
  shard) OPTS+=(--array="0-$((NUM_SHARDS - 1))" --output="logs/slurm/%x-%A_%a.out") ;;
  fold)  OPTS+=(--array=0-5 --output="logs/slurm/%x-%A_%a.out") ;;   # 0..4 = fold, 5 = full
  none)  OPTS+=(--output="logs/slurm/%x-%j.out") ;;
esac
mkdir -p logs/slurm
sbatch "${OPTS[@]}" hpc/stage.sbatch | cut -d';' -f1
