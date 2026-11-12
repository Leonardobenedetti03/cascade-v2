#!/bin/bash
# hpc/pipeline.sh - sottomette la catena completa di una run, con dipendenze afterok.
#   RUN_NAME=site_v2 hpc/pipeline.sh              # seed da SEEDS (default "0 1 2")
#   RUN_NAME=id_v2 SEEDS="0" hpc/pipeline.sh
# Il test NON viene valutato: lo stadio 50 gira senza --final (vedi hpc/final.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh
: "${RUN_NAME:?RUN_NAME mancante}"
mkdir -p logs
PLAN="logs/pipeline_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).txt"
sub() {   # sub <etichetta> <argomenti di submit.sh>: stampa il job id, annota il piano
  local label="$1" id; shift
  id=$(hpc/submit.sh "$@")
  printf '%-10s %-8s %s\n' "$id" "$label" "$*" >> "$PLAN"
  echo "$id"
}

# A - dati (CPU)
j00=$(sub A   00)
j01=$(sub A   01 --after "$j00")
j05=$(sub A   05 --after "$j00")
# B - frame (array CPU), feature di L1 (GPU), diagnostica dell'allineamento
j10=$(sub B10 10 --after "$j00")
sub B15 15 --after "$j01" > /dev/null      # diagnostico: non blocca la catena
j25=$(sub B25 25 --after "$j10")
# C - stadi che non dipendono dal seed di CamFormer
j27=$(sub C   27 --after "$j25:$j05")
j28=$(sub C   28 --after "$j10:$j05")
j30=$(sub C   30 --after "$j25:$j05")
j40=$(sub C   40 --after "$j00")
j41=$(sub C   41 --after "$j01")
# B20 + C per seed: CamFormer (array 0-5), fusione, cache, gate su val
dep50=""
read -r -a SEED_LIST <<< "$SEEDS"
for s in "${SEED_LIST[@]}"; do
  seed=(--set "camformer.seed=$s")
  j20=$(sub "B20.s$s" 20 --after "$j01:$j05" -- "${seed[@]}")
  j35=$(sub "C.s$s"   35 --after "$j20:$j30" -- "${seed[@]}")
  j45=$(sub "C.s$s"   45 --after "$j35:$j40:$j28" -- "${seed[@]}")
  j50=$(sub "C.s$s"   50 --after "$j45" -- "${seed[@]}")
  dep50="${dep50:+$dep50:}$j50"
done
# aggregazione dei seed (solo con piu' seed) e report
dep55="$dep50:$j27:$j41"
if [[ ${#SEED_LIST[@]} -gt 1 ]]; then
  j52=$(sub C 52 --after "$dep50" -- --seeds "${SEED_LIST[@]}")
  dep55="$j52:$j27:$j41"
fi
j55=$(sub R 55 --after "$dep55")
echo "catena sottomessa (ultimo job: $j55). Piano: $PLAN"
cat "$PLAN"
