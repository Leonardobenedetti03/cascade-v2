#!/bin/bash
# hpc/stages.sh - tabella degli stadi, letta da submit.sh (risorse) e da stage.sbatch (catene).
# Da includere con `source`, non da lanciare.
#
# stadio -> script, nodo, CPU, memoria, tempo massimo per elemento, tipo di array.
# Valori di partenza (stime): correggerli con i tempi del primo run (sacct).
# (funzione con case e non array associativo: gira anche con la bash 3.2 del Mac, dove si prova)
# Tipo di array: shard = stadio 10 diviso in NUM_SHARDS parti; fold = stadio 20, fold 0..4 e full;
# none = un solo processo. In una catena (submit.sh --chain) gli elementi girano uno dopo l'altro
# nello stesso job: fold diventa sei esecuzioni, shard una sola esecuzione su tutte le clip.
stage_row() {
  case "$1" in
    00) echo "00_inventory.py              cpu 4 8G  00:30:00 none" ;;
    01) echo "01_convert_trajectories.py   cpu 4 8G  01:00:00 none" ;;
    05) echo "05_folds.py                  cpu 1 4G  00:10:00 none" ;;
    10) echo "10_extract_frames.py         cpu 4 8G  01:00:00 shard" ;;
    15) echo "15_alignment_check.py        cpu 2 8G  00:20:00 none" ;;
    20) echo "20_camformer_finetune.py     gpu 4 16G 01:00:00 fold" ;;
    25) echo "25_egovlp_features.py        gpu 8 32G 01:00:00 none" ;;
    27) echo "27_site_diagnostics.py       cpu 4 16G 00:30:00 none" ;;
    28) echo "28_frame_baseline.py         gpu 4 16G 00:30:00 none" ;;
    30) echo "30_l1_heads.py               cpu 8 16G 00:30:00 none" ;;
    35) echo "35_fusion.py                 cpu 2 8G  00:15:00 none" ;;
    40) echo "40_cost.py                   gpu 2 16G 00:15:00 none" ;;
    41) echo "41_cost_e2e.py               gpu 4 16G 00:45:00 none" ;;
    45) echo "45_build_cache.py            cpu 4 16G 00:30:00 none" ;;
    50) echo "50_gate_eval.py              cpu 8 16G 01:00:00 none" ;;
    52) echo "52_aggregate_seeds.py        cpu 1 4G  00:10:00 none" ;;
    55) echo "55_report.py                 cpu 2 8G  00:20:00 none" ;;
    *) return 1 ;;
  esac
}

# Elementi di una fold: 0..4 e full (stesso ordine dell'array 0-5 di submit.sh, dove 5 = full).
export FOLD_PARTS="0 1 2 3 4 full"

# hms_to_s HH:MM:SS -> secondi; s_to_hms secondi -> HH:MM:SS
hms_to_s() { local h m s; IFS=: read -r h m s <<< "$1"; echo $(( 10#$h * 3600 + 10#$m * 60 + 10#$s )); }
s_to_hms() { printf '%02d:%02d:%02d' $(( $1 / 3600 )) $(( $1 % 3600 / 60 )) $(( $1 % 60 )); }
