#!/bin/bash
# hpc/pipeline.sh - sottomette la run completa come catena di 4 job, con dipendenze afterok.
#   RUN_NAME=site_v2 hpc/pipeline.sh              # seed da SEEDS (default "0 1 2")
#   RUN_NAME=site_v2 hpc/pipeline.sh --redo-gate  # solo 50, 52, 55 di nuovo, con --overwrite (strada A, N27)
# Il test NON viene valutato: lo stadio 50 gira senza --final (vedi hpc/final.sh).
#
# C14c-2 (N27): la QOS stud ammette 4 job in coda per utente e conta ogni elemento di un array, quindi
# la catena del blueprint (circa 25 sottomissioni, piu' gli array di 10 e 20) non entra. Quattro job,
# ognuno con piu' stadi in sequenza (hpc/submit.sh --chain):
#   A  (CPU)  01 -> 00 -> 05 -> 10 -> 15 (diagnostico: se fallisce la catena va avanti)
#   B1 (GPU)  25 -> 28 -> 40 -> 41 -> 20 per il primo seed          dopo A
#   B2 (GPU)  20 per gli altri seed                                  dopo A, in parallelo a B1
#   C  (CPU)  27 -> 30 -> (35 -> 45 -> 50 per seed) -> 52 -> 55      dopo B1 e B2
# Con un solo seed B2 non c'e'. 55 entra solo se scripts/55_report.py esiste (C13b).
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh
: "${RUN_NAME:?RUN_NAME mancante}"
REDO_GATE=0
if [[ "${1:-}" == "--redo-gate" ]]; then REDO_GATE=1; shift; fi
[[ $# -eq 0 ]] || { echo "pipeline: argomento sconosciuto: $1" >&2; exit 2; }
mkdir -p logs
PLAN="logs/pipeline_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).txt"
sub() {   # sub <nome> <stadi> [--after ID[:ID]]: stampa il job id, annota il piano
  local name="$1" stages="$2" id; shift 2
  id=$(hpc/submit.sh --chain "$name" "$stages" "$@")
  printf '%-10s %-4s %-24s %s\n' "$id" "$name" "${2:-}" "$stages" >> "$PLAN"
  echo "$id"
}
read -r -a SEED_LIST <<< "$SEEDS"
[[ ${#SEED_LIST[@]} -gt 0 ]] || { echo "pipeline: SEEDS vuoto" >&2; exit 2; }
seed() { echo "--set camformer.seed=$1"; }

# parte di C che dipende dal seed, piu' aggregazione e report
gate_part() {   # gate_part <argomenti aggiunti a 50 e 52>
  local extra="$1" out="" s
  for s in "${SEED_LIST[@]}"; do
    out="$out; 50 $(seed "$s")${extra:+ $extra}"
  done
  if [[ ${#SEED_LIST[@]} -gt 1 ]]; then out="$out; 52 --seeds ${SEED_LIST[*]}${extra:+ $extra}"; fi
  if [[ -f scripts/55_report.py ]]; then out="$out; 55"; fi
  echo "${out#; }"
}

if [[ "$REDO_GATE" == 1 ]]; then
  # strada A: 00-45 restano quelli della run; 50-55 si rifanno al commit attuale (verify_run lo accetta)
  jg=$(sub G "$(gate_part --overwrite)")
  echo "gate sottomesso di nuovo (job $jg). Piano: $PLAN"
  cat "$PLAN"
  exit 0
fi

jA=$(sub A "01; 00; 05; 10; 15?")
jB1=$(sub B1 "25; 28; 40; 41; 20 $(seed "${SEED_LIST[0]}")" --after "$jA")
depC="$jB1"
if [[ ${#SEED_LIST[@]} -gt 1 ]]; then
  b2=""
  for s in "${SEED_LIST[@]:1}"; do b2="$b2; 20 $(seed "$s")"; done
  jB2=$(sub B2 "${b2#; }" --after "$jA")
  depC="$jB1:$jB2"
fi
c="27; 30"
for s in "${SEED_LIST[@]}"; do c="$c; 35 $(seed "$s"); 45 $(seed "$s")"; done
jC=$(sub C "$c; $(gate_part "")" --after "$depC")
echo "catena sottomessa (ultimo job: $jC). Piano: $PLAN"
cat "$PLAN"
