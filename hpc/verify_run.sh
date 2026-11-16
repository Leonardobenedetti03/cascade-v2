#!/bin/bash
# hpc/verify_run.sh <run> - controlli da superare PRIMA di leggere i numeri.
# Esce con 1 se un controllo fallisce. Non modifica nulla.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source hpc/env.sh
RUN_NAME="${1:-${RUN_NAME:?uso: hpc/verify_run.sh <run>}}"
RUN_DIR="$CASCADE_RUN_ROOT/$RUN_NAME"
[[ -d "$RUN_DIR" ]] || { echo "run_dir assente: $RUN_DIR"; exit 1; }
case "$RUN_NAME" in          # take attese per split (guida, sezioni 0 e 6)
  site_v2) EXPECTED="test=67 train=1138 val=68" ;;
  id_v2)   EXPECTED="train=988 val=285" ;;
  *)       EXPECTED="" ;;
esac
FAIL=0
ok()  { echo "  ok       $*"; }
bad() { echo "  FALLITO  $*"; FAIL=1; }

echo "1. commit dei manifest: uno solo per 00-45, uno solo (quello del clone) per 50-52"
# Strada A (N27): gli stadi 50-55 si possono rifare dopo, a un commit piu' recente, con
# hpc/pipeline.sh --redo-gate. Si accetta solo se il commit di 00-45 precede quello del clone e se
# tra i due non cambia nessuno script degli stadi 00-45; le modifiche in src/ e configs/ si elencano
# da controllare a mano (le chiavi di config cambiate le ferma gia' il manifest).
read -r UP DOWN <<< "$(python3 - "$RUN_DIR" <<'PY'
import json, pathlib, sys
up, down = set(), set()
for m in sorted(pathlib.Path(sys.argv[1], "_manifest").glob("*.json")):
    if m.name.startswith("55_"):          # il report si puo rigenerare dopo
        continue
    c = str(json.loads(m.read_text()).get("commit", "unknown"))
    (down if m.name.startswith(("50_", "52_")) else up).add(c)
print(",".join(sorted(up)) or "-", ",".join(sorted(down)) or "-")
PY
)"
HEAD=$(git rev-parse HEAD)
if [[ "$UP" == "-" ]]; then bad "nessun manifest degli stadi 00-45 in $RUN_DIR/_manifest"
elif [[ "$UP" == *,* ]]; then bad "commit diversi nei manifest di 00-45: $UP"
elif [[ "$DOWN" == *,* ]]; then bad "commit diversi nei manifest di 50-52: $DOWN"
elif [[ "$DOWN" != "-" && "$HEAD" != "$DOWN"* ]]; then bad "manifest di 50-52 al commit $DOWN, clone a ${HEAD:0:7}"
elif [[ "$DOWN" == "-" || "$UP" == "$DOWN" ]]; then
  if [[ "$HEAD" == "$UP"* ]]; then ok "commit $UP"; else bad "manifest al commit $UP, clone a ${HEAD:0:7}"; fi
elif ! git merge-base --is-ancestor "$UP" HEAD 2>/dev/null; then
  bad "il commit di 00-45 ($UP) non precede quello del clone (${HEAD:0:7})"
else
  CHANGED=$(git diff --name-only "$UP" HEAD -- scripts/ | grep -v -E '^scripts/5[0-9]_' || true)
  if [[ -n "$CHANGED" ]]; then bad "50-52 rifatti a ${HEAD:0:7}, ma da $UP sono cambiati script degli stadi 00-45: $(echo $CHANGED)"
  else
    ok "strada A: 00-45 al commit $UP, 50-52 al commit ${HEAD:0:7}"
    OTHER=$(git diff --name-only "$UP" HEAD -- src/ configs/ | tr '\n' ' ')
    if [[ -n "$OTHER" ]]; then echo "  info     da controllare (cambiati da $UP): $OTHER"; fi
  fi
fi

echo "2. nessun Traceback nei log"
TB=$(grep -rl "Traceback (most recent call last)" "$RUN_DIR/logs" 2>/dev/null | head -5)
if [[ -n "$TB" ]]; then bad "Traceback in: $TB"; else ok "log puliti"; fi

echo "3. take utilizzabili per split"
COUNTS=$(python3 - "$RUN_DIR/data/clips.csv" <<'PY'
import collections, csv, sys
takes = collections.defaultdict(set)
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        if r["has_video"] == "True" and r["has_traj"] == "True":
            takes[r["split"]].add(r["take_uid"])
print(" ".join(f"{k}={len(v)}" for k, v in sorted(takes.items())))
PY
)
if [[ -z "$EXPECTED" ]]; then ok "conteggi $COUNTS (nessun valore atteso per questa run)"
elif [[ "$COUNTS" == "$EXPECTED" ]]; then ok "$COUNTS"
else bad "trovato '$COUNTS', atteso '$EXPECTED'"; fi

echo "4. CamFormer: 5 fold + full per ogni seed"
for s in $SEEDS; do
  n=$(find "$RUN_DIR/20_camformer" -path "*seed${s}_*" -name summary.json 2>/dev/null | wc -l)
  if [[ "$n" -eq 6 ]]; then ok "seed $s: 6 modelli"; else bad "seed $s: $n modelli su 6"; fi
done

echo "5. split fissato e test guardato al massimo una volta per variante"
if ( cd "$SPLIT_DIR" && sha256sum --check --quiet SHA256SUMS ); then ok "SHA256 dello split"; else bad "split modificato"; fi
FINAL="$RUN_DIR/50_gate/FINAL_RUNS.log"
if [[ -f "$FINAL" ]]; then
  DUP=$(awk -F'\t' '{print $5 " " $6}' "$FINAL" | sort | uniq -d)
  N_DONE=$(find "$RUN_DIR/50_gate" -name FINAL_DONE | wc -l)
  N_LOG=$(awk -F'\t' '{print $5 " " $6}' "$FINAL" | sort -u | wc -l)
  if [[ -n "$DUP" ]]; then bad "test valutato piu' volte per: $DUP"; else ok "$(wc -l < "$FINAL") valutazioni finali registrate"; fi
  if [[ "$N_DONE" -ne "$N_LOG" ]]; then bad "$N_DONE FINAL_DONE ma $N_LOG varianti nel registro"; fi
else
  if [[ -n "$(find "$RUN_DIR/50_gate" -name FINAL_DONE 2>/dev/null)" ]]; then bad "FINAL_DONE senza FINAL_RUNS.log"
  else ok "test non ancora valutato"; fi
fi

if [[ "$FAIL" -eq 0 ]]; then echo "verify_run: tutto a posto ($RUN_NAME)"; else echo "verify_run: CONTROLLI FALLITI ($RUN_NAME)"; fi
exit "$FAIL"
