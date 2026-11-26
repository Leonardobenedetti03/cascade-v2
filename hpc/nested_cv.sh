#!/bin/bash
# hpc/nested_cv.sh - one CPU job with stage 56 (gate v2, nested CV by session on all takes, N37).
# Usage:  SCEN_RUN=$HOME/scenario_runs/official_v2 RUN_NAME=gate_v2 hpc/nested_cv.sh
#         N38: TAG=n38_rel CONFIG=configs/gate_v2_n38_rel.yaml SCEN_RUN=... RUN_NAME=gate_v2 hpc/nested_cv.sh
#         (TAG = subfolder of the outputs, default main; CONFIG = an extra config file, optional)
# The run in the report: TAG=n38_cal CONFIG=configs/gate_v2_n38_cal.yaml.
# It needs a scenario_preds run with the features (features/camformer.npz, features/egovlp.npz), the index and traj/.
# Before starting it, we wrote the values of configs/gate_v2.yaml in entry N37 of the decision log.
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh
: "${SCEN_RUN:?serve SCEN_RUN = run_dir di scenario_preds}" "${RUN_NAME:=gate_v2}" "${N_JOBS:=8}"
for f in index/takes.csv features/camformer.npz features/egovlp.npz configs/heads.json; do
  [[ -f "$SCEN_RUN/$f" ]] || { echo "manca $SCEN_RUN/$f" >&2; exit 2; }
done
mkdir -p logs/slurm
: "${TAG:=main}" "${CONFIG:=}"
if [[ -n "$CONFIG" && ! -f "$CONFIG" ]]; then echo "manca $CONFIG" >&2; exit 2; fi
export SCEN_RUN RUN_NAME N_JOBS TAG CONFIG
j=$(sbatch --parsable --export=ALL --account="$SLURM_ACCOUNT" ${SLURM_QOS:+--qos="$SLURM_QOS"} \
    --job-name="gate_v2_${RUN_NAME}" --partition="$CPU_PARTITION" --cpus-per-task="$N_JOBS" --mem=16G \
    --time=04:00:00 --output="logs/slurm/%x-%j.out" hpc/nested_cv.sbatch)
echo "gate v2: job $j  (run: $RUN_NAME, tag: $TAG${CONFIG:+, config: $CONFIG}, scenario_preds: $SCEN_RUN, $N_JOBS processi)"
