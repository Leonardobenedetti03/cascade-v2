#!/bin/bash
# hpc/nested_cv.sh - un job CPU con lo stadio 56 (gate v2, CV annidata per sessione su tutte le take, N37).
# Uso:  SCEN_RUN=$HOME/scenario_runs/official_v2 RUN_NAME=gate_v2 hpc/nested_cv.sh
#       N38: TAG=n38_rel CONFIG=configs/gate_v2_n38_rel.yaml SCEN_RUN=... RUN_NAME=gate_v2 hpc/nested_cv.sh
#       (TAG = sottocartella degli output, default main; CONFIG = file di config aggiuntivo, facoltativo)
# Serve una run di scenario_preds con le feature (features/camformer.npz, features/egovlp.npz), l'indice e traj/.
# Prima di lanciarlo: valori di configs/gate_v2.yaml scritti nella voce N37 del registro.
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
