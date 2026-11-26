#!/bin/bash
# hpc/input_variants.sh - one GPU job with the 6 candidate runs (identity / diag(-1,-1,1) poses x 1, 3, 5 windows),
# each one index -> traj -> feat-camformer -> heads --models camformer --skip-check, then the select stage
# (decision log, audit item 1): it picks on val and writes to $HOME/scenario_runs/<RUN_NAME>_cand/select/.
# Usage:  RUN_NAME=official_v2 hpc/input_variants.sh        Then the final run with hpc/features_submit.sh (see there).
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/features_env.sh
mkdir -p logs/slurm
j=$(sbatch --parsable --export=ALL --account="$SLURM_ACCOUNT" --qos="$SLURM_QOS" --job-name="sc_${RUN_NAME}_cand" \
    --partition="$GPU_PARTITION" --gres="$GPU_GRES" --cpus-per-task=8 --mem=32G --time=12:00:00 \
    --output="logs/slurm/%x-%j.out" hpc/input_variants.sbatch)
echo "candidate: job $j  (uscite: $HOME/scenario_runs/${RUN_NAME}_cand/)"
