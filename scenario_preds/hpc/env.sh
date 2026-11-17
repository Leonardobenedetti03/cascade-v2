#!/bin/bash
# hpc/env.sh - parametri del cluster e percorsi. Valori di partenza dal runbook di cascade v2 (verificati il 15/11):
# correggere quelli personali qui o in hpc/env.local.sh (letto per primo, non va versionato).
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$_here/env.local.sh" ]]; then source "$_here/env.local.sh"; fi
unset _here

# --- cluster (QOS stud: 1 giorno per job, 4 job in coda, 2 in esecuzione per utente) ----------------
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-3185472}"
export SLURM_QOS="${SLURM_QOS:-stud}"
export CPU_PARTITION="${CPU_PARTITION:-stud}"
export GPU_PARTITION="${GPU_PARTITION:-stud}"
export GPU_GRES="${GPU_GRES:-gpu:1}"
export VENV="${VENV:-$HOME/venvs/cascade-v2}"          # l'ambiente di cascade v2 ha già tutto (torch cu121, av, decord, sklearn)

# --- dati, pesi, uscite -------------------------------------------------------------------------------
export CAMFORMER_REPO="${CAMFORMER_REPO:-$HOME/seeing_without_pixels}"   # git clone https://github.com/google-deepmind/seeing_without_pixels
export CAMFORMER_CKPT="${CAMFORMER_CKPT:-/home/3185472/cascade_v2_root/data/checkpoints/camformer/egoexo4d_dur8.pt}"
export EGOVLP_CKPT="${EGOVLP_CKPT:-/home/3185472/cascade_data/egovlp_ckpt/egovlp.pth}"
export ARIA_TRAJ_ROOT="${ARIA_TRAJ_ROOT:-/home/3185472/cascade_data/egoexo4d/takes}"   # <take>/trajectory/closed_loop_trajectory.csv
export RUN_NAME="${RUN_NAME:-official_v1}"
export SCENARIO_RUN_DIR="${SCENARIO_RUN_DIR:-$HOME/scenario_runs/$RUN_NAME}"
