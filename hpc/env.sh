#!/bin/bash
# hpc/env.sh - parametri del cluster e della run. File versionato.
# I valori personali (account, percorsi con il nome utente) vanno in hpc/env.local.sh,
# ignorato da git e letto per primo: qui sotto ogni variabile gia' impostata resta com'e'.
# Ogni TODO(HPC) si compila una volta, al primo accesso (docs/RUNBOOK_HPC.md).
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$_here/env.local.sh" ]]; then source "$_here/env.local.sh"; fi
unset _here

# --- cluster ---------------------------------------------------------------------------
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"   # TODO(HPC) sacctmgr show assoc user=$USER format=account,partition,qos
export CPU_PARTITION="${CPU_PARTITION:-}"   # TODO(HPC) sinfo -s
export GPU_PARTITION="${GPU_PARTITION:-}"   # TODO(HPC) sinfo -s
export GPU_GRES="${GPU_GRES:-gpu:1}"        # TODO(HPC) sinfo -o "%P %G"   (es. gpu:a100:1)
export MODULES="${MODULES:-}"               # TODO(HPC) module avail 2>&1 | grep -i -E "python|cuda"
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"  # TODO(HPC) build di torch per la CUDA del cluster

# --- percorsi --------------------------------------------------------------------------
export REPO_DIR="${REPO_DIR:-$HOME/cascade-v2}"
export VENV="${VENV:-$HOME/venvs/cascade-v2}"
export DL4CV_ROOT="${DL4CV_ROOT:-}"         # TODO(HPC) radice di dati e run (scratch o project)
export CASCADE_DATA_ROOT="${CASCADE_DATA_ROOT:-${DL4CV_ROOT:+$DL4CV_ROOT/data}}"
export CASCADE_RUN_ROOT="${CASCADE_RUN_ROOT:-${DL4CV_ROOT:+$DL4CV_ROOT/runs}}"
export EXPORT_DIR="${EXPORT_DIR:-${DL4CV_ROOT:+$DL4CV_ROOT/export}}"

# --- run -------------------------------------------------------------------------------
export NUM_SHARDS="${NUM_SHARDS:-8}"        # shard dello stadio 10 (job array CPU)
export MAX_REQUEUE="${MAX_REQUEUE:-3}"      # rilanci automatici al limite di tempo
export SEEDS="${SEEDS:-0 1 2}"              # seed di CamFormer (camformer.seed)
export SPLIT_DIR="${SPLIT_DIR:-splits/site_v1}"   # contiene scenario_site.csv e SHA256SUMS
