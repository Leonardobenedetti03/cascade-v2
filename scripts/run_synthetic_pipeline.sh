#!/bin/bash
# Pipeline completa su dati sintetici, su CPU. Cresce di commit in commit: ogni stadio
# portato aggiunge qui la sua riga (stesso ordine di hpc/pipeline.sh).
#   scripts/run_synthetic_pipeline.sh [--clean] [--folds "0 1 2 3 4 full"] [--set chiave=valore ...]
# DL4CV_ROOT di default: ./_synthetic_root (ignorata da git).
set -euo pipefail
cd "$(dirname "$0")/.."
export DL4CV_ROOT="${DL4CV_ROOT:-$PWD/_synthetic_root}"
RUN_NAME="${RUN_NAME:-synthetic}"
CLEAN=0
FOLDS="0 1 2 3 4 full"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) CLEAN=1; shift ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --set) EXTRA+=(--set "$2"); shift 2 ;;
    *) echo "argomento sconosciuto: $1"; exit 2 ;;
  esac
done
mkdir -p "$DL4CV_ROOT"
if [[ $CLEAN == 1 ]]; then rm -rf "$DL4CV_ROOT/runs/$RUN_NAME"; fi
PY="${PYTHON:-python3}"
# ${EXTRA[@]+...}: sicuro con array vuoto anche sulla bash 3.2 di macOS
COMMON=(--synthetic --run-name "$RUN_NAME" ${EXTRA[@]+"${EXTRA[@]}"})
step() { echo "== $*"; "$PY" "scripts/$1" "${COMMON[@]}" "${@:2}"; }

step 01_convert_trajectories.py
step 00_inventory.py
step 05_folds.py
# C07: 01_convert_trajectories.py   C08: 10, 15, 20, 25, 28, 30
# C09: 35   C10: 40, 41   C12: 27, 45, 50 --final, 52   C13: 55

RUN_DIR=$("$PY" -c "from src.cascade.paths import run_dir; print(run_dir('$RUN_NAME'))")
echo "== fatto: $RUN_DIR"
ls "$RUN_DIR" "$RUN_DIR/data"
