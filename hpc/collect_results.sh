#!/bin/bash
# hpc/collect_results.sh <run> - raccoglie i file PICCOLI di una run in un archivio da
# portare sul Mac e committare in results/<run>/. L'HPC non fa push (deploy key in sola lettura).
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh
RUN_NAME="${1:-${RUN_NAME:?uso: hpc/collect_results.sh <run>}}"
RUN_DIR="$CASCADE_RUN_ROOT/$RUN_NAME"
hpc/verify_run.sh "$RUN_NAME" || { echo "verify_run fallito: non raccolgo"; exit 1; }

OUT="$EXPORT_DIR/results/$RUN_NAME"
rm -rf "$OUT"; mkdir -p "$OUT"/{env,manifests,tables,logs}

# manifest e riepilogo della run
cp "$RUN_DIR"/_manifest/*.json "$OUT/manifests/"
python -m src.cascade.manifest "$RUN_DIR" > /dev/null && cp "$RUN_DIR/run_manifest.json" "$OUT/"

# tabelle, riassunti e config: solo json/csv/yaml sotto i 5 MB (mai parquet, npy, pt)
STAGES=()
for d in data splits configs 20_camformer 27_site 28_frame_baseline 30_l1_heads 35_fusion \
         40_cost 41_cost_e2e 45_cache 50_gate 52_seeds run_info; do
  if [[ -d "$RUN_DIR/$d" ]]; then STAGES+=("$d"); fi
done
( cd "$RUN_DIR" && find "${STAGES[@]}" -type f -size -5M \
    \( -name '*.json' -o -name '*.csv' -o -name '*.yaml' -o -name 'FINAL_*' \) -print0 \
  | xargs -0 -r cp --parents -t "$OUT/tables" )

# log (ognuno sotto 1 MB) e ambiente
find "$RUN_DIR/logs" -type f -size -1M -exec cp -t "$OUT/logs" {} +
cp "$VENV/pip_freeze.txt" "$VENV/modules.txt" "$OUT/env/" 2>/dev/null || python -m pip freeze > "$OUT/env/pip_freeze.txt"

# RUN.md: commit, split, job e comandi esatti, ricavati da manifest e run_info
python3 - "$RUN_DIR" "$RUN_NAME" "$(git rev-parse HEAD)" "$(cut -d' ' -f1 "$SPLIT_DIR/SHA256SUMS" | head -1)" > "$OUT/RUN.md" <<'PY'
import json, pathlib, sys
run_dir, run, commit, sha = pathlib.Path(sys.argv[1]), *sys.argv[2:5]
print(f"# Run {run}\n\n- commit: `{commit}`\n- split SHA256: `{sha}`\n")
print("## Job\n\n| stadio | job | nodo | data (UTC) | restart |\n|---|---|---|---|---|")
for f in sorted((run_dir / "run_info").glob("*.json")):
    i = json.loads(f.read_text())
    print(f"| {i['stage']} | {i['job']}_{i['task']} | {i['node']} | {i['date']} | {i['restart']} |")
print("\n## Comandi\n")
for f in sorted((run_dir / "_manifest").glob("*.json")):
    m = json.loads(f.read_text())
    print(f"- `{m['stage']}` [{m.get('part') or '-'}]: `{' '.join(map(str, m.get('argv', [])))}`")
print("\n## Note\n\n(da compilare a mano: anomalie, job rilanciati, decisioni)")
PY

TAR="$EXPORT_DIR/results_${RUN_NAME}_$(git rev-parse --short HEAD).tar.gz"
tar -czf "$TAR" -C "$EXPORT_DIR" "results/$RUN_NAME"
echo "archivio: $TAR ($(du -h "$TAR" | cut -f1))"
echo "sul Mac:  scp <hpc>:$TAR . && tar -xzf $(basename "$TAR") -C <repo> && git add results/$RUN_NAME && git commit"
