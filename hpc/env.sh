#!/bin/bash
# hpc/env.sh - cluster and run settings, read by hpc/nested_cv.sh and hpc/setup_env.sh.
# Personal values (account, paths with the user name) go in hpc/env.local.sh, which git ignores and which is
# read first: every variable already set there keeps its value.
# Each TODO(HPC) is filled in once, on first login. The values for the Bocconi cluster are in docs/hpc_runbook.md (N23).
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$_here/env.local.sh" ]]; then source "$_here/env.local.sh"; fi
unset _here

# Cluster
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"   # TODO(HPC) sacctmgr show assoc user=$USER format=account,partition,qos
export SLURM_QOS="${SLURM_QOS:-}"           # TODO(HPC) only if the cluster needs a QOS (sacctmgr, qos column)
export CPU_PARTITION="${CPU_PARTITION:-}"   # TODO(HPC) sinfo -s
export GPU_PARTITION="${GPU_PARTITION:-}"   # TODO(HPC) sinfo -s
export GPU_GRES="${GPU_GRES:-gpu:1}"        # TODO(HPC) sinfo -o "%P %G"   (e.g. gpu:a100:1)
export MODULES="${MODULES:-}"               # TODO(HPC) module avail 2>&1 | grep -i -E "python|cuda"
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"  # TODO(HPC) torch build for the cluster's CUDA

# Paths
export REPO_DIR="${REPO_DIR:-$HOME/cascade-v2}"
export VENV="${VENV:-$HOME/venvs/cascade-v2}"
export DL4CV_ROOT="${DL4CV_ROOT:-}"         # TODO(HPC) root of data and runs (scratch or project)
export CASCADE_DATA_ROOT="${CASCADE_DATA_ROOT:-${DL4CV_ROOT:+$DL4CV_ROOT/data}}"
export CASCADE_RUN_ROOT="${CASCADE_RUN_ROOT:-${DL4CV_ROOT:+$DL4CV_ROOT/runs}}"
export EXPORT_DIR="${EXPORT_DIR:-${DL4CV_ROOT:+$DL4CV_ROOT/export}}"

# Run (first pipeline)
export NUM_SHARDS="${NUM_SHARDS:-8}"        # shards of stage 10 (CPU job array)
export MAX_REQUEUE="${MAX_REQUEUE:-3}"      # automatic requeues at the time limit
export SEEDS="${SEEDS:-0 1 2}"              # CamFormer seeds (camformer.seed)
export SPLIT_DIR="${SPLIT_DIR:-splits/site_v1}"   # holds scenario_site.csv and SHA256SUMS
