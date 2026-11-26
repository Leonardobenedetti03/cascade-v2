#!/bin/bash
# hpc/features.sbatch - runs the stages in $STAGES one after the other (e.g. "index traj"). Start it with hpc/features_submit.sh.
#SBATCH --open-mode=append
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?lanciare con hpc/features_submit.sh}"
source hpc/features_env.sh
source "$VENV/bin/activate"
set -u
echo "[$(date -u +%FT%TZ)] job ${SLURM_JOB_ID:-local} su $(hostname): stadi '$STAGES' run_dir=$SCENARIO_RUN_DIR"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
set -f  # no filename expansion in the arguments (the [[...]] matrices of --set)
for st in $STAGES; do
    # shellcheck disable=SC2086
    python scripts/01_extract_features.py "$st" --config "${CONFIG:-configs/features.yaml}" ${EXTRA_ARGS:-}
done
echo "[$(date -u +%FT%TZ)] fine"
