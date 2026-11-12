#!/bin/bash
# hpc/final.sh - valutazione sul TEST: una volta sola, a punto di lavoro gia' deciso su val.
#   RUN_NAME=site_v2 hpc/final.sh
# Ogni esecuzione lascia una riga in <run_dir>/50_gate/FINAL_RUNS.log (verify_run la controlla).
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh
: "${RUN_NAME:?RUN_NAME mancante}"
hpc/verify_run.sh "$RUN_NAME" || { echo "verify_run fallito: il test non si guarda"; exit 1; }
read -r -p "Valutare il TEST di $RUN_NAME (seed: $SEEDS)? Scrivi il nome della run per confermare: " ans
[[ "$ans" == "$RUN_NAME" ]] || { echo "annullato"; exit 1; }
read -r -a SEED_LIST <<< "$SEEDS"
dep=""
for s in "${SEED_LIST[@]}"; do
  j=$(hpc/submit.sh 50 -- --set "camformer.seed=$s" --final)
  dep="${dep:+$dep:}$j"
done
if [[ ${#SEED_LIST[@]} -gt 1 ]]; then
  dep=$(hpc/submit.sh 52 --after "$dep" -- --seeds "${SEED_LIST[@]}")
fi
hpc/submit.sh 55 --after "$dep"
