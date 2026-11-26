#!/bin/bash
# hpc/features_env.sh - cluster settings and paths for the feature pipeline. Starting values from the cascade v2 runbook
# (checked on 15/11). Change the personal ones here or in hpc/features_env.local.sh (read first, not kept in git).
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$_here/features_env.local.sh" ]]; then source "$_here/features_env.local.sh"; fi
unset _here

# Cluster (QOS stud: 1 day per job, 4 jobs queued and 2 running per user)
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-3185472}"
export SLURM_QOS="${SLURM_QOS:-stud}"
export CPU_PARTITION="${CPU_PARTITION:-stud}"
export GPU_PARTITION="${GPU_PARTITION:-stud}"
export GPU_GRES="${GPU_GRES:-gpu:1}"
export VENV="${VENV:-$HOME/venvs/cascade-v2}"          # the cascade v2 environment already has everything (torch cu121, av, decord, sklearn)

# Data, weights, outputs
export CAMFORMER_REPO="${CAMFORMER_REPO:-$HOME/seeing_without_pixels}"   # git clone https://github.com/google-deepmind/seeing_without_pixels
export CAMFORMER_CKPT="${CAMFORMER_CKPT:-/home/3185472/cascade_v2_root/data/checkpoints/camformer/egoexo4d_dur8.pt}"
export EGOVLP_CKPT="${EGOVLP_CKPT:-/home/3185472/cascade_data/egovlp_ckpt/egovlp.pth}"
export ARIA_TRAJ_ROOT="${ARIA_TRAJ_ROOT:-/home/3185472/cascade_data/egoexo4d/takes}"   # <take>/trajectory/closed_loop_trajectory.csv
export RUN_NAME="${RUN_NAME:-official_v1}"
export SCENARIO_RUN_DIR="${SCENARIO_RUN_DIR:-$HOME/scenario_runs/$RUN_NAME}"
