#!/bin/bash
# hpc/features_submit.sh - submits the feature chain as 3 jobs (the stud QOS allows 4 in the queue):
#   A (CPU)  index -> traj
#   B (GPU)  feat-camformer -> feat-egovlp       after A
#   C (CPU)  heads                                after B; exits with code 3 if the pre-registered check fails
# Usage:  RUN_NAME=official_v1 hpc/features_submit.sh
# The cut (official_val.mode = split, half val and half test) and the check (val: bacc >= 0.40 and CI above chance;
# OOF train: recall > 0; all classes present) are the defaults fixed in the decision log. First hpc/input_variants.sh
# picks poses and windows on val; then this chain runs with the two --set printed by select (in select/selection.json).
# This is how we made the run in the report:
#   RUN_NAME=official_v2 hpc/features_submit.sh --set 'poses.json_cam_to_device=[[1,0,0],[0,1,0],[0,0,1]]' --set 'window.n_windows=5'
# If the index already exists with a different cut, redo it by hand first:
#   python scripts/01_extract_features.py index --overwrite --set ...  (the features are not recomputed).
# Secondary analysis by site: RUN_NAME=site_v1 hpc/features_submit.sh --set split.mode=site --skip-check
# Stages already complete with the same config are skipped, so rerunning after an error is safe.
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/features_env.sh
EXTRA=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --set) EXTRA="$EXTRA --set $2"; shift 2 ;;
        --skip-check) EXTRA="$EXTRA --skip-check"; shift ;;
        *) echo "argomento sconosciuto: $1" >&2; exit 2 ;;
    esac
done
export EXTRA_ARGS="$EXTRA"
mkdir -p logs/slurm
common=(--parsable --export=ALL --kill-on-invalid-dep=yes --account="$SLURM_ACCOUNT" --qos="$SLURM_QOS")
jA=$(STAGES="index traj" sbatch "${common[@]}" --job-name="sc_${RUN_NAME}_A" --partition="$CPU_PARTITION" \
     --cpus-per-task=4 --mem=16G --time=06:00:00 --output="logs/slurm/%x-%j.out" hpc/features.sbatch)
jB=$(STAGES="feat-camformer feat-egovlp" sbatch "${common[@]}" --job-name="sc_${RUN_NAME}_B" --partition="$GPU_PARTITION" \
     --gres="$GPU_GRES" --cpus-per-task=8 --mem=32G --time=10:00:00 --dependency="afterok:$jA" \
     --output="logs/slurm/%x-%j.out" hpc/features.sbatch)
jC=$(STAGES="heads" sbatch "${common[@]}" --job-name="sc_${RUN_NAME}_C" --partition="$CPU_PARTITION" \
     --cpus-per-task=4 --mem=16G --time=02:00:00 --dependency="afterok:$jB" \
     --output="logs/slurm/%x-%j.out" hpc/features.sbatch)
echo "A=$jA  B=$jB  C=$jC   (run_dir: $SCENARIO_RUN_DIR; log: logs/slurm/ e \$SCENARIO_RUN_DIR/logs/)"
