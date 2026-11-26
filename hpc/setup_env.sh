#!/bin/bash
# hpc/setup_env.sh - creates the persistent Python environment, ONCE.
# Run it on the login node (it needs the network). After that no job installs anything.
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh
: "${TORCH_INDEX_URL:?TODO(HPC): TORCH_INDEX_URL in hpc/env.sh (es. https://download.pytorch.org/whl/cu121)}"

if [[ -n "$MODULES" ]]; then read -r -a MODS <<< "$MODULES"; module purge; module load "${MODS[@]}"; fi
if [[ -d "$VENV" ]]; then echo "$VENV esiste: rimuoverlo a mano per ricrearlo"; exit 1; fi

python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
python -m pip install -r env/requirements.txt

# Snapshot of the environment: its hash goes into the RUN_INFO.json of every job.
python -m pip freeze > "$VENV/pip_freeze.txt"
sha256sum < "$VENV/pip_freeze.txt" | cut -d' ' -f1 > "$VENV/pip_freeze.sha256"
{ module list 2>&1 || true; } > "$VENV/modules.txt"

python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"
python -c "import transformers; print('transformers', transformers.__version__)"
python -m pytest -q          # the test suite must pass on the cluster too
echo "ambiente pronto: $VENV (hash $(cut -c1-16 "$VENV/pip_freeze.sha256"))"
