#!/bin/bash
# hpc/gate_v2.sbatch - eseguito da hpc/gate_v2.sh: stadio 56 (gate v2).
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?lanciare con hpc/gate_v2.sh}"
source hpc/env.sh
source "$VENV/bin/activate"
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # un thread per processo joblib
echo "[$(date -u +%FT%TZ)] gate v2 su scenario_preds: $SCEN_RUN -> run $RUN_NAME"
python scripts/56_gate_v2.py --run-name "$RUN_NAME" --scenario-run "$SCEN_RUN" --n-jobs "$N_JOBS"
echo "[$(date -u +%FT%TZ)] fine"
