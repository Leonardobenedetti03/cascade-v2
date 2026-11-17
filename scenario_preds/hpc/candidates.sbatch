#!/bin/bash
# hpc/candidates.sbatch - eseguito da hpc/candidates.sh. Le candidate non usano mai il test (niente --eval-test).
#SBATCH --open-mode=append
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?lanciare con hpc/candidates.sh}"
source hpc/env.sh
source "$VENV/bin/activate"
set -u
BASE="$HOME/scenario_runs/${RUN_NAME}_cand"
CFG="${CONFIG:-configs/default.yaml}"
cands=()
for pose in id flip; do
    if [[ $pose == flip ]]; then P='[[-1,0,0],[0,-1,0],[0,0,1]]'; else P='[[1,0,0],[0,1,0],[0,0,1]]'; fi
    for w in 1 3 5; do
        export SCENARIO_RUN_DIR="$BASE/${pose}_w${w}"
        S=(--set "poses.json_cam_to_device=$P" --set "window.n_windows=$w")
        for st in index traj feat-camformer; do
            python -m scenario "$st" --config "$CFG" "${S[@]}"
        done
        python -m scenario heads --config "$CFG" "${S[@]}" --models camformer --skip-check
        cands+=("$SCENARIO_RUN_DIR")
    done
done
export SCENARIO_RUN_DIR="$BASE/select"
python -m scenario select --config "$CFG" --candidates "${cands[@]}"
echo "[$(date -u +%FT%TZ)] fine: vedi $BASE/select/selection.json"
