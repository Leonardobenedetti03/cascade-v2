#!/bin/bash
# hpc/final.sh - valutazione sul TEST: una volta sola, a punto di lavoro gia' deciso su val.
#   RUN_NAME=site_v2 hpc/final.sh
# Ogni esecuzione lascia una riga in <run_dir>/50_gate/FINAL_RUNS.log (verify_run la controlla).
# Un solo job (catena: 50 --final per seed, 52, 55), perche' la QOS stud ammette 4 job in coda (N27),
# e senza requeue: un rilancio automatico valuterebbe il test una seconda volta.
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh
: "${RUN_NAME:?RUN_NAME mancante}"
hpc/verify_run.sh "$RUN_NAME" || { echo "verify_run fallito: il test non si guarda"; exit 1; }
read -r -p "Valutare il TEST di $RUN_NAME (seed: $SEEDS)? Scrivi il nome della run per confermare: " ans
[[ "$ans" == "$RUN_NAME" ]] || { echo "annullato"; exit 1; }
read -r -a SEED_LIST <<< "$SEEDS"
stages=""
for s in "${SEED_LIST[@]}"; do stages="$stages; 50 --set camformer.seed=$s --final"; done
if [[ ${#SEED_LIST[@]} -gt 1 ]]; then stages="$stages; 52 --seeds ${SEED_LIST[*]}"; fi
if [[ -f scripts/55_report.py ]]; then stages="$stages; 55"; fi
NO_REQUEUE=1 hpc/submit.sh --chain F "${stages#; }"
