#!/bin/bash
# hpc/check_cluster.sh <run> - controlli sul login node PRIMA della prima run.
# Non modifica nulla e non mette nulla in coda (sbatch --test-only convalida la richiesta e basta).
#   RUN_NAME=site_v2 hpc/check_cluster.sh
# Esce con 1 se un controllo indispensabile fallisce.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source hpc/env.sh
RUN_NAME="${1:-${RUN_NAME:?uso: hpc/check_cluster.sh <run>}}"
RUN_CONFIG="${RUN_CONFIG:-$RUN_NAME}"
FAIL=0
ok()   { echo "  ok       $*"; }
bad()  { echo "  FALLITO  $*"; FAIL=1; }
warn() { echo "  attenz.  $*"; }

echo "1. parametri del cluster (hpc/env.sh e hpc/env.local.sh)"
for v in SLURM_ACCOUNT CPU_PARTITION GPU_PARTITION DL4CV_ROOT; do
  if [[ -n "${!v}" ]]; then ok "$v=${!v}"; else bad "$v vuoto"; fi
done
if [[ -n "${SLURM_QOS:-}" ]]; then ok "SLURM_QOS=$SLURM_QOS"; else warn "SLURM_QOS vuoto: va bene solo se il cluster non richiede una QOS"; fi
[[ -f "configs/hpc/$RUN_CONFIG.yaml" ]] && ok "configs/hpc/$RUN_CONFIG.yaml" || bad "manca configs/hpc/$RUN_CONFIG.yaml"

echo "2. ambiente Python (hpc/setup_env.sh)"
if [[ -x "$VENV/bin/python" ]]; then
  PY="$VENV/bin/python"; ok "$VENV"
  "$PY" - <<'PY'
import importlib
import importlib.util
import sys

missing = []
for name in ("torch", "torchvision", "transformers", "fvcore", "sklearn", "pandas", "pyarrow", "lightgbm", "yaml"):
    try:
        mod = importlib.import_module(name)
        print(f"  ok       {name} {getattr(mod, '__version__', '')}")
    except Exception as exc:  # noqa: BLE001 - qualunque errore di import è un problema dell'ambiente
        print(f"  FALLITO  {name}: {type(exc).__name__}: {exc}")
        missing.append(name)
decoder = [n for n in ("decord", "av") if importlib.util.find_spec(n)]
if decoder:
    print(f"  ok       decoder video: {', '.join(decoder)}")
else:
    print("  FALLITO  nessun decoder video (decord o av)")
    missing.append("decoder")
try:
    import torch

    print(f"  info     CUDA visibile dal login node: {torch.cuda.is_available()} (di solito no: le GPU sono sui nodi)")
except Exception:  # noqa: BLE001
    pass
sys.exit(1 if missing else 0)
PY
  [[ $? -eq 0 ]] || FAIL=1
else
  PY=python3; bad "manca $VENV: hpc/setup_env.sh"
fi

echo "3. dati, pesi e split (percorsi di configs/base.yaml)"
"$PY" -m src.cascade.hpc_check --config "configs/hpc/$RUN_CONFIG.yaml" --run-name "$RUN_NAME" || FAIL=1
if ( cd "$SPLIT_DIR" && sha256sum --check --quiet SHA256SUMS ); then ok "SHA256SUMS di $SPLIT_DIR"; else bad "split diverso da SHA256SUMS"; fi

echo "4. richiesta a SLURM (sbatch --test-only: nulla viene messo in coda)"
if command -v sbatch > /dev/null 2>&1; then
  BASE=(--test-only --job-name=cv2_check --time=00:05:00 --cpus-per-task=1 --mem=1G)
  [[ -n "$SLURM_ACCOUNT" ]] && BASE+=(--account="$SLURM_ACCOUNT")
  [[ -n "${SLURM_QOS:-}" ]] && BASE+=(--qos="$SLURM_QOS")
  if sbatch "${BASE[@]}" --partition="$CPU_PARTITION" --wrap="true" 2>&1; then ok "richiesta CPU accettata"; else bad "richiesta CPU rifiutata"; fi
  if sbatch "${BASE[@]}" --partition="$GPU_PARTITION" --gres="$GPU_GRES" --wrap="true" 2>&1; then ok "richiesta GPU accettata"; else bad "richiesta GPU rifiutata (GPU_GRES=$GPU_GRES)"; fi
else
  warn "sbatch non trovato: questo non è il login node"
fi

echo "5. limiti del cluster (solo da leggere: confrontarli con la tabella di hpc/submit.sh)"
if command -v sacctmgr > /dev/null 2>&1 && [[ -n "${SLURM_QOS:-}" ]]; then
  sacctmgr -n -P show qos "$SLURM_QOS" format=Name,MaxWall,MaxTRESPU,MaxJobsPU,MaxSubmitPU 2>&1 | sed 's/^/  info     qos: /'
fi
if command -v sinfo > /dev/null 2>&1; then
  sinfo -p "$CPU_PARTITION,$GPU_PARTITION" -o "%P %a %l %D %G" 2>&1 | sed 's/^/  info     /'
fi

if [[ "$FAIL" -eq 0 ]]; then echo "check_cluster: tutto a posto ($RUN_NAME)"; else echo "check_cluster: CONTROLLI FALLITI ($RUN_NAME)"; fi
exit "$FAIL"
