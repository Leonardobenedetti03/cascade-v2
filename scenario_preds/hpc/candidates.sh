#!/bin/bash
# hpc/candidates.sh - un job GPU con le 6 run candidate (pose identità / diag(-1,-1,1) x 1, 3, 5 finestre),
# ognuna index -> traj -> feat-camformer -> heads --models camformer --skip-check, poi lo stadio select
# (registro, audit punto 1): sceglie sulla val e scrive in $HOME/scenario_runs/<RUN_NAME>_cand/select/.
# Uso:  RUN_NAME=official_v2 hpc/candidates.sh        Poi la run finale con hpc/submit.sh (vedi lì).
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh
mkdir -p logs/slurm
j=$(sbatch --parsable --export=ALL --account="$SLURM_ACCOUNT" --qos="$SLURM_QOS" --job-name="sc_${RUN_NAME}_cand" \
    --partition="$GPU_PARTITION" --gres="$GPU_GRES" --cpus-per-task=8 --mem=32G --time=12:00:00 \
    --output="logs/slurm/%x-%j.out" hpc/candidates.sbatch)
echo "candidate: job $j  (uscite: $HOME/scenario_runs/${RUN_NAME}_cand/)"
