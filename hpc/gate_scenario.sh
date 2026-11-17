#!/bin/bash
# hpc/gate_scenario.sh - un job CPU con gli stadi 40 (costi), 44 (cache da scenario_preds), 50 (gate) e 53
# (analisi), N28 e N32.
# Uso:  SCEN_RUN=$HOME/scenario_runs/official_v2 RUN_NAME=scenario_v1 hpc/gate_scenario.sh
# Il test non entra (né in 44 né in 50). Passo finale, una volta sola, a mano:
#   python scripts/44_scenario_bridge.py --run-name scenario_v1 --config configs/hpc/scenario_v1.yaml \
#       --scenario-run "$SCEN_RUN" --include-test --overwrite
#   python scripts/50_gate_eval.py --run-name scenario_v1 --config configs/hpc/scenario_v1.yaml --final
#   python scripts/53_scenario_analysis.py --run-name scenario_v1 --config configs/hpc/scenario_v1.yaml --final
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh
: "${SCEN_RUN:?serve SCEN_RUN = run_dir di scenario_preds}" "${RUN_NAME:=scenario_v1}"
[[ -f "$SCEN_RUN/predictions.csv" ]] || { echo "manca $SCEN_RUN/predictions.csv (scenario_preds heads)" >&2; exit 2; }
mkdir -p logs/slurm
export SCEN_RUN RUN_NAME
j=$(sbatch --parsable --export=ALL --account="$SLURM_ACCOUNT" ${SLURM_QOS:+--qos="$SLURM_QOS"} \
    --job-name="gate_${RUN_NAME}" --partition="$CPU_PARTITION" --cpus-per-task=8 --mem=16G --time=03:00:00 \
    --output="logs/slurm/%x-%j.out" hpc/gate_scenario.sbatch)
echo "gate: job $j  (run: $RUN_NAME, scenario_preds: $SCEN_RUN)"
