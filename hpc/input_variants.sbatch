#!/bin/bash
# hpc/input_variants.sbatch - eseguito da hpc/input_variants.sh. Le candidate non usano mai il test (niente --eval-test).
#SBATCH --open-mode=append
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?lanciare con hpc/input_variants.sh}"
source hpc/features_env.sh
source "$VENV/bin/activate"
set -u
BASE="$HOME/scenario_runs/${RUN_NAME}_cand"
CFG="${CONFIG:-configs/features.yaml}"
cands=()
for pose in id flip; do
    if [[ $pose == flip ]]; then P='[[-1,0,0],[0,-1,0],[0,0,1]]'; else P='[[1,0,0],[0,1,0],[0,0,1]]'; fi
    for w in 1 3 5; do
        export SCENARIO_RUN_DIR="$BASE/${pose}_w${w}"
        S=(--set "poses.json_cam_to_device=$P" --set "window.n_windows=$w")
        for st in index traj feat-camformer; do
            python scripts/01_extract_features.py "$st" --config "$CFG" "${S[@]}"
        done
        python scripts/01_extract_features.py heads --config "$CFG" "${S[@]}" --models camformer --skip-check
        cands+=("$SCENARIO_RUN_DIR")
    done
done
export SCENARIO_RUN_DIR="$BASE/select"
python scripts/01_extract_features.py select --config "$CFG" --candidates "${cands[@]}"
echo "[$(date -u +%FT%TZ)] fine: vedi $BASE/select/selection.json"
