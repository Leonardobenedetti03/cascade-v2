#!/bin/bash
# hpc/stage.sbatch - template unico per tutti gli stadi.
# Non lanciarlo a mano: hpc/submit.sh passa partizione, risorse, array e dipendenze.
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
set -eo pipefail

cd "${SLURM_SUBMIT_DIR:?job da lanciare con hpc/submit.sh}"
source hpc/env.sh
if [[ -n "$MODULES" ]]; then read -r -a MODS <<< "$MODULES"; module purge; module load "${MODS[@]}"; fi
source "$VENV/bin/activate"
set -u
: "${STAGE_SCRIPT:?}" "${RUN_NAME:?}" "${RUN_CONFIG:?}" "${EXPECTED_COMMIT:?}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
die() { echo "[preflight] $*" >&2; exit 3; }

# --- 1. preflight: codice, submodule e split sono quelli attesi (nessuna rete sul nodo) ---
HEAD=$(git rev-parse HEAD)
[[ "$HEAD" == "$EXPECTED_COMMIT" ]] || die "HEAD $HEAD diverso dal commit del submit $EXPECTED_COMMIT"
if [[ "$RUN_NAME" != dbg_* ]]; then
  [[ -z "$(git status --porcelain)" ]] || die "clone non pulito: git status"
  if git submodule status | grep -q '^[-+U]'; then die "submodule non allineati: git submodule update --init"; fi
fi
( cd "$SPLIT_DIR" && sha256sum --check --quiet SHA256SUMS ) || die "split diverso da quello fissato"

# --- 2. log nella run_dir e RUN_INFO del job ---------------------------------------------
RUN_DIR=$(python -c "from src.cascade.paths import run_dir; print(run_dir('$RUN_NAME'))")
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/run_info"
LOG="$RUN_DIR/logs/slurm-${SLURM_JOB_NAME:-local}-${JOB}_${TASK}.out"
exec > >(tee -a "$LOG") 2>&1
ENV_SHA=$(python -m pip freeze 2>/dev/null | sha256sum | cut -d' ' -f1)
if [[ -f "$VENV/pip_freeze.sha256" && "$ENV_SHA" != "$(cat "$VENV/pip_freeze.sha256")" ]]; then
  echo "ATTENZIONE: ambiente diverso da quello di setup_env.sh"
fi
printf '{"stage":"%s","run":"%s","commit":"%s","job":"%s","task":"%s","node":"%s","date":"%s","restart":%s,"env_sha256":"%s","split_sha256":"%s"}\n' \
  "$STAGE_SCRIPT" "$RUN_NAME" "$HEAD" "$JOB" "$TASK" "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "${SLURM_RESTART_COUNT:-0}" "$ENV_SHA" "$(cut -d' ' -f1 "$SPLIT_DIR/SHA256SUMS" | head -1)" \
  > "$RUN_DIR/run_info/${SLURM_JOB_NAME:-local}-${JOB}_${TASK}.json"

# --- 3. argomenti del task di array ------------------------------------------------------
EXTRA=()
case "${ARRAY_KIND:-none}" in
  shard) EXTRA=(--shard-id "$TASK" --num-shards "$NUM_SHARDS") ;;
  fold)  if [[ "$TASK" == 5 ]]; then EXTRA=(--fold full); else EXTRA=(--fold "$TASK"); fi ;;
  none)  ;;
  *) die "ARRAY_KIND sconosciuto: $ARRAY_KIND" ;;
esac

# --- 4. requeue: SLURM manda USR1 180 s prima del limite di tempo -------------------------
# Lo stadio viene fermato e il job torna in coda; al rilancio i manifest fanno saltare le
# parti finite e lo stadio 20 riparte dall'ultima epoca salvata (ckpt_last.pt).
requeue() {
  echo "[$(date -u +%H:%M:%S)] USR1: requeue ${SLURM_RESTART_COUNT:-0}/$MAX_REQUEUE"
  kill -TERM "$PY_PID" 2>/dev/null || true
  wait "$PY_PID" 2>/dev/null || true
  if [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_RESTART_COUNT:-0}" -lt "$MAX_REQUEUE" ]]; then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  echo "limite di requeue raggiunto"; exit 4
}
trap requeue USR1

# --- 5. lo stadio, in background: cosi' il trap scatta mentre python lavora ---------------
echo "[$(date -u +%FT%TZ)] $STAGE_SCRIPT run=$RUN_NAME job=${JOB}_${TASK} node=$(hostname) commit=${HEAD:0:7}"
python -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.is_available())" 2>/dev/null || echo "torch non importabile"
# shellcheck disable=SC2086
python "scripts/$STAGE_SCRIPT" --run-name "$RUN_NAME" --config "configs/hpc/$RUN_CONFIG.yaml" \
  ${STAGE_ARGS:-} ${EXTRA[@]+"${EXTRA[@]}"} &
PY_PID=$!
set +e
wait "$PY_PID"; RC=$?
set -e
echo "[$(date -u +%FT%TZ)] fine $STAGE_SCRIPT (exit $RC)"
exit "$RC"
