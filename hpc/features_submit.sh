#!/bin/bash
# hpc/features_submit.sh - sottomette la catena in 3 job (la QOS stud ne ammette 4 in coda):
#   A (CPU)  index -> traj
#   B (GPU)  feat-camformer -> feat-egovlp       dopo A
#   C (CPU)  heads                                dopo B; esce con codice 3 se il controllo pre-registrato fallisce
# Uso:  RUN_NAME=official_v1 hpc/features_submit.sh
# Taglio (official_val.mode = split, metà val e metà test) e controllo (val: bacc >= 0,40 e CI sopra il caso; train OOF:
# richiamo > 0; tutte le classi presenti) sono i default fissati nel registro. Prima: hpc/input_variants.sh sceglie sulla val
# pose e finestre; poi questa catena con le due --set stampate da select (in select/selection.json):
#   RUN_NAME=official_v2 hpc/features_submit.sh --set 'poses.json_cam_to_device=[[1,0,0],[0,1,0],[0,0,1]]' --set 'window.n_windows=1'
# se l'indice esiste già con un altro taglio,
# rifarlo prima a mano:  python scripts/01_extract_features.py index --overwrite --set ...  (le feature non si rifanno).
# Analisi secondaria per sito: RUN_NAME=site_v1 hpc/features_submit.sh --set split.mode=site --skip-check
# Gli stadi già completi con la stessa config si saltano: rilanciare dopo un errore è sicuro.
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/features_env.sh
EXTRA=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --set) EXTRA="$EXTRA --set $2"; shift 2 ;;
        --skip-check) EXTRA="$EXTRA --skip-check"; shift ;;
        *) echo "argomento sconosciuto: $1" >&2; exit 2 ;;
    esac
done
export EXTRA_ARGS="$EXTRA"
mkdir -p logs/slurm
common=(--parsable --export=ALL --kill-on-invalid-dep=yes --account="$SLURM_ACCOUNT" --qos="$SLURM_QOS")
jA=$(STAGES="index traj" sbatch "${common[@]}" --job-name="sc_${RUN_NAME}_A" --partition="$CPU_PARTITION" \
     --cpus-per-task=4 --mem=16G --time=06:00:00 --output="logs/slurm/%x-%j.out" hpc/features.sbatch)
jB=$(STAGES="feat-camformer feat-egovlp" sbatch "${common[@]}" --job-name="sc_${RUN_NAME}_B" --partition="$GPU_PARTITION" \
     --gres="$GPU_GRES" --cpus-per-task=8 --mem=32G --time=10:00:00 --dependency="afterok:$jA" \
     --output="logs/slurm/%x-%j.out" hpc/features.sbatch)
jC=$(STAGES="heads" sbatch "${common[@]}" --job-name="sc_${RUN_NAME}_C" --partition="$CPU_PARTITION" \
     --cpus-per-task=4 --mem=16G --time=02:00:00 --dependency="afterok:$jB" \
     --output="logs/slurm/%x-%j.out" hpc/features.sbatch)
echo "A=$jA  B=$jB  C=$jC   (run_dir: $SCENARIO_RUN_DIR; log: logs/slurm/ e \$SCENARIO_RUN_DIR/logs/)"
