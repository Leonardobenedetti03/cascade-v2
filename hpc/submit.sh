#!/bin/bash
# hpc/submit.sh - sottomette UNO stadio, oppure UNA catena di stadi in un solo job. Stampa il job id.
#   RUN_NAME=site_v2 hpc/submit.sh <stadio> [--after ID[:ID...]] [-- argomenti per lo script]
#   RUN_NAME=site_v2 hpc/submit.sh --chain <nome> "<stadio> [arg...]; <stadio> [arg...]" [--after ID[:ID...]]
# Esempi:
#   RUN_NAME=site_v2 hpc/submit.sh 10                                  # array di NUM_SHARDS shard
#   RUN_NAME=site_v2 hpc/submit.sh 20 --after 123 -- --set camformer.seed=1
#   RUN_NAME=dbg_smoke RUN_CONFIG=site_v2 hpc/submit.sh 00 -- --inspect   # run di prova
#   RUN_NAME=site_v2 hpc/submit.sh --chain B2 "20 --set camformer.seed=1; 20 --set camformer.seed=2"
# Catena (C14c-2, N27): gli stadi girano in sequenza nello stesso job, senza array (la QOS stud conta ogni
# elemento di un array come un job). Uno stadio con "?" dopo il numero (es. "15?") e' diagnostico: se
# fallisce la catena va avanti. Risorse del job: nodo GPU se almeno uno stadio e' GPU, CPU e memoria
# massime, tempo = somma dei tempi (x6 per le fold, xNUM_SHARDS per lo stadio 10), al massimo 24 ore.
# NO_REQUEUE=1: il job non torna mai in coda (hpc/final.sh: il test non si valuta due volte).
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh
source hpc/stages.sh
die() { echo "submit: $*" >&2; exit 2; }

CHAIN=""
STAGE_LIST=""
if [[ "${1:-}" == "--chain" ]]; then
  CHAIN="${2:?uso: hpc/submit.sh --chain <nome> \"<stadio> [arg...]; ...\" [--after ID]}"
  STAGE_LIST="${3:?catena $CHAIN senza stadi}"
  shift 3
  STAGE="$CHAIN"
else
  STAGE="${1:?uso: hpc/submit.sh <stadio> [--after ID[:ID]] [-- argomenti]}"; shift
fi
AFTER=""
if [[ "${1:-}" == "--after" ]]; then AFTER="$2"; shift 2; fi
if [[ "${1:-}" == "--" ]]; then shift; fi
[[ -z "$CHAIN" || $# -eq 0 ]] || die "con --chain gli argomenti vanno dentro la catena, non dopo --"
export RUN_NAME="${RUN_NAME:?RUN_NAME mancante (site_v2 | id_v2 | dbg_*)}"
export RUN_CONFIG="${RUN_CONFIG:-$RUN_NAME}"
export STAGE_ARGS="$*"
[[ -f "configs/hpc/$RUN_CONFIG.yaml" ]] || die "manca configs/hpc/$RUN_CONFIG.yaml"
[[ -n "$DL4CV_ROOT" ]] || die "TODO(HPC): DL4CV_ROOT in hpc/env.sh o hpc/env.local.sh"

if [[ -z "$CHAIN" ]]; then
  ROW=$(stage_row "$STAGE") || die "stadio sconosciuto: $STAGE"
  read -r SCRIPT KIND CPUS MEM TIME ARRAY <<< "$ROW"
  export STAGE_SCRIPT="$SCRIPT" ARRAY_KIND="$ARRAY" STAGE_LIST=""
else
  # risorse della catena: massimo di CPU e memoria, somma dei tempi
  KIND=cpu; CPUS=1; MEMG=1; SECS=0; N=0
  IFS=';' read -r -a SPECS <<< "$STAGE_LIST"
  for spec in "${SPECS[@]}"; do
    read -r st _ <<< "$spec"
    [[ -n "${st:-}" ]] || continue
    st="${st%\?}"
    ROW=$(stage_row "$st") || die "catena $CHAIN: stadio sconosciuto: $st"
    read -r _ k c m t a <<< "$ROW"
    [[ "$k" == gpu ]] && KIND=gpu
    (( c > CPUS )) && CPUS=$c
    (( ${m%G} > MEMG )) && MEMG=${m%G}
    n=1
    case "$a" in fold) n=6 ;; shard) n=$NUM_SHARDS ;; esac
    SECS=$(( SECS + n * $(hms_to_s "$t") ))
    N=$((N + 1))
  done
  (( N > 0 )) || die "catena $CHAIN senza stadi"
  (( SECS > 86400 )) && SECS=86400   # limite della QOS stud: 1 giorno per job
  MEM="${MEMG}G"; TIME=$(s_to_hms "$SECS"); ARRAY=none
  export STAGE_SCRIPT="" ARRAY_KIND="none" STAGE_LIST CHAIN_NAME="$CHAIN"
fi
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
if [[ -n "${SLURM_QOS:-}" ]]; then OPTS+=(--qos="$SLURM_QOS"); fi
if [[ -n "$AFTER" ]]; then OPTS+=(--dependency="afterok:$AFTER"); fi
if [[ "${NO_REQUEUE:-0}" == 1 ]]; then OPTS+=(--no-requeue); export MAX_REQUEUE=0; fi
case "$ARRAY" in
  shard) OPTS+=(--array="0-$((NUM_SHARDS - 1))" --output="logs/slurm/%x-%A_%a.out") ;;
  fold)  OPTS+=(--array=0-5 --output="logs/slurm/%x-%A_%a.out") ;;   # 0..4 = fold, 5 = full
  none)  OPTS+=(--output="logs/slurm/%x-%j.out") ;;
esac
mkdir -p logs/slurm
sbatch "${OPTS[@]}" hpc/stage.sbatch | cut -d';' -f1
