# Runbook HPC

Comandi per lanciare una run sul cluster. I valori personali (account, partizione, QOS, percorsi) stanno in
`hpc/env.local.sh`, che git ignora; il resto è in `hpc/env.sh`. Il test si valuta una volta sola (`hpc/final.sh`).

## 1. Primo accesso (una volta)

1. Chiave GitHub del cluster (N23): deploy key `hpc-bocconi` di `cascade-v2`, in lettura e scrittura, in
   `~/.ssh/cascade_v2_deploy` sul cluster, con il nome breve `github-cascade-v2` in `~/.ssh/config`. Clone:
   `git clone --recurse-submodules github-cascade-v2:Leonardobenedetti03/cascade-v2.git ~/cascade-v2`, poi
   `git config user.name` e `user.email` uguali a quelli del Mac. Il codice si cambia solo dal Mac (CI verde).
2. `hpc/env.local.sh` (ignorato da git) con i valori verificati il 15/11/2026 (`sacctmgr`, `sinfo`, `sbatch --test-only`):

   ```bash
   export SLURM_ACCOUNT="3185472"
   export SLURM_QOS="stud"
   export CPU_PARTITION="stud"
   export GPU_PARTITION="stud"
   export GPU_GRES="gpu:1"                 # GPU MIG 3g.40gb, 4g.40gb o 7g.80gb: gpu:1 prende la prima libera
   export MODULES=""                       # Python 3.9 di sistema; il runtime CUDA arriva con la wheel di torch
   export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
   export DL4CV_ROOT="/home/3185472/cascade_v2_root"
   ```

   Limiti della QOS `stud`: 1 giorno per job, 12 CPU e 2 GPU per utente, 2 job in esecuzione e **4 in coda** (ogni
   elemento di un array conta come un job).
3. Ambiente Python, una volta, dal login node (serve la rete): `hpc/setup_env.sh`. Stampa le versioni di `torch` e
   `transformers` e lancia la suite di test.
4. Dati e pesi sotto `$DL4CV_ROOT/data`, senza copie (N23):

   ```bash
   R=/home/3185472/cascade_v2_root
   mkdir -p $R/data/egoexo4d $R/data/checkpoints/egovlp $R/data/checkpoints/torchvision $R/runs $R/export
   # le 1273 take delle due metà del subset, unite con link (albero costruito in v1 da make_traj_data.py)
   ln -s /home/3185472/cascade_data/egoexo4d/takes $R/data/egoexo4d/takes
   ln -s /home/3185472/cascade_data/egovlp_ckpt/egovlp.pth $R/data/checkpoints/egovlp/egovlp.pth
   curl -sSL -o $R/data/checkpoints/torchvision/mobilenet_v3_small-047dcff4.pth \
     https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth
   ```

   CamFormer (M17, N24): `egoexo4d_dur8.pt` da `camformer_checkpoints.zip` (Google Drive degli autori, link in
   `third_party/seeing_without_pixels/DATA.md`). Si scarica dal browser sul Mac e si copia con `scp` in
   `$R/data/checkpoints/camformer/egoexo4d_dur8.pt`; `python -m pytest -q tests/test_camformer.py -k real` lo carica.

   `takes.json` non c'è: `site_v2` prende la durata dal video. Le pose per lo stadio 01 si leggono dalle due radici
   in `paths.pose_roots` di `configs/hpc/site_v2.yaml`.
5. Controlli: `RUN_NAME=site_v2 hpc/check_cluster.sh`. Verifica parametri, import di `torch` e `transformers`,
   percorsi, hash dello split e `sbatch --test-only`; non mette nulla in coda.
6. Prova su una run `dbg_*` (salta i controlli sul clone pulito e non finisce mai nei risultati):

   ```bash
   RUN_NAME=dbg_smoke RUN_CONFIG=site_v2 hpc/submit.sh 00 -- --inspect
   ```

   `--inspect` stampa la struttura reale delle cartelle e il video e la traiettoria attesi per una take: conferma i
   percorsi di N23. Le correzioni si fanno sul Mac, con commit e CI verde, poi `git pull --ff-only`.

## 2. Run vera

Prima, dal Mac: CI verde sull'ultimo commit di `main`. Poi, sul cluster:

```bash
git pull --ff-only
RUN_NAME=site_v2 hpc/pipeline.sh      # seed da SEEDS (default "0 1 2"); il piano finisce in logs/pipeline_*.txt
squeue -u $USER                       # coda; sacct -j <id> per tempi e memoria
RUN_NAME=site_v2 hpc/verify_run.sh    # controlli prima di leggere i numeri
RUN_NAME=site_v2 hpc/final.sh         # il TEST, una volta, con conferma
RUN_NAME=site_v2 hpc/collect_results.sh   # archivio dei file piccoli da portare sul Mac in results/site_v2
```

Il commit nei manifest degli stadi 00–52 deve essere uguale a quello del clone (`verify_run.sh`): dopo la run non si
fa `git pull`. I tempi della tabella di `hpc/submit.sh` sono stime; si correggono con `sacct` o con
`TIME_OVERRIDE=hh:mm:ss MEM_OVERRIDE=16G` sul singolo lancio. I limiti della QOS li stampa `check_cluster.sh`.

## 3. Quando un job fallisce

- Exit 3: il preflight di `hpc/stage.sbatch` (commit diverso da quello del submit, clone non pulito, submodule o split
  non allineati). Il messaggio dice quale; il job non ha toccato nulla.
- Limite di tempo o nodo perso: il job torna in coda da solo (al massimo `MAX_REQUEUE` volte) e riprende dalle parti
  finite. Lo stadio 20 riparte dall'ultima epoca salvata.
- Rilancio di un solo stadio: `RUN_NAME=site_v2 hpc/submit.sh <stadio> [-- argomenti]`. I manifest saltano ciò che è
  completo con la stessa config; `-- --overwrite` rifà.
- Log: `<run_dir>/logs/` (uno per job, con la riga di partenza e di fine) e `logs/slurm/` nel clone.
- Un traceback nei log fa fallire `verify_run.sh`: va letto e risolto prima di guardare i risultati.
