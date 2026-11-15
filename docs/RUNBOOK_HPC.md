# Runbook HPC

Comandi per lanciare una run sul cluster. I valori personali (account, partizione, QOS, percorsi) stanno in
`hpc/env.local.sh`, che git ignora; il resto è in `hpc/env.sh`. Il test si valuta una volta sola (`hpc/final.sh`).

## 1. Primo accesso (una volta)

1. Clone con i submodule (deploy key in sola lettura): `git clone --recurse-submodules <url> ~/cascade-v2`.
2. Crea `hpc/env.local.sh` con i valori del cluster (esempio, da adattare):

   ```bash
   export SLURM_ACCOUNT="<account>"        # sacctmgr show assoc user=$USER format=account,partition,qos
   export SLURM_QOS="<qos>"                # lasciare vuoto se il cluster non la richiede
   export CPU_PARTITION="<partizione>"     # sinfo -s
   export GPU_PARTITION="<partizione>"
   export GPU_GRES="gpu:1"                 # sinfo -o "%P %G"
   export MODULES="<moduli python/cuda>"   # module avail 2>&1 | grep -i -E "python|cuda"
   export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"   # build di torch per la CUDA del cluster
   export DL4CV_ROOT="<radice di dati e run>"
   ```

3. Ambiente Python, una volta, dal login node (serve la rete): `hpc/setup_env.sh`. Stampa le versioni di `torch` e
   `transformers` e lancia la suite di test.
4. Copia dati e pesi sotto `$CASCADE_DATA_ROOT` (`$DL4CV_ROOT/data`): `egoexo4d/takes.json`, `egoexo4d/takes/`,
   `checkpoints/egovlp/egovlp.pth` e `checkpoints/torchvision/mobilenet_v3_small-047dcff4.pth`.
5. Controlli: `RUN_NAME=site_v2 hpc/check_cluster.sh`. Verifica parametri, import di `torch` e `transformers`,
   percorsi, hash dello split e `sbatch --test-only`; non mette nulla in coda.
6. Prova su una run `dbg_*` (salta i controlli sul clone pulito e non finisce mai nei risultati):

   ```bash
   RUN_NAME=dbg_smoke RUN_CONFIG=site_v2 hpc/submit.sh 00 -- --inspect
   ```

   `--inspect` stampa la struttura reale delle cartelle: serve a correggere i `TODO(HPC)` di `configs/base.yaml`
   (pattern dei percorsi, `pose_roots`). Le correzioni si fanno sul Mac, con commit e CI verde, poi `git pull --ff-only`.

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
