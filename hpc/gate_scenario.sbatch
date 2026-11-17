#!/bin/bash
# hpc/gate_scenario.sbatch - eseguito da hpc/gate_scenario.sh: 40 -> 44 -> 50 -> 53 (senza --final).
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?lanciare con hpc/gate_scenario.sh}"
source hpc/env.sh
source "$VENV/bin/activate"
set -u
CFG="configs/hpc/${RUN_CONFIG:-scenario_v1}.yaml"
echo "[$(date -u +%FT%TZ)] gate su scenario_preds: $SCEN_RUN -> run $RUN_NAME"
python scripts/40_cost.py --run-name "$RUN_NAME" --config "$CFG"
python scripts/44_scenario_bridge.py --run-name "$RUN_NAME" --config "$CFG" --scenario-run "$SCEN_RUN"
python scripts/50_gate_eval.py --run-name "$RUN_NAME" --config "$CFG"
python scripts/53_scenario_analysis.py --run-name "$RUN_NAME" --config "$CFG"
echo "[$(date -u +%FT%TZ)] fine"
