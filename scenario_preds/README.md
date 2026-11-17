# scenario_preds

Una predizione dello scenario per ogni take del subset Ego-Exo4D (1273 take), con due modelli
**congelati** e una **testa lineare riaddestrata** sopra ciascuno. Fornisce L0 (CamFormer) e L1
(EgoVLP) alla cascata di cascade-v2, che le legge con lo stadio ponte 44.

| Modello | Ingresso | Pesi | Embedding |
|---|---|---|---|
| CamFormer (L0) | traiettoria della camera RGB dai JSON `camera_pose`, 20 Hz, `rel9d_grav` | `egoexo4d_dur8.pt` (seeing_without_pixels, `DATA.md`) | 512 |
| EgoVLP (L1) | 4 frame RGB a 224 px dallo stream `*214-1*.mp4` | `EgoVLP_PT_BEST` (showlab/EgoVLP) | CLS a 768 |

Testa: StandardScaler + regressione logistica multinomiale, `class_weight: balanced`, sulle 6 classi
nell'ordine Basketball, Dance, Soccer, Cooking, Health, Bike repair (scritto anche in `classes.json`).
Music (0 take) e Bouldering (2 take) restano fuori dall'addestramento; le take di Bouldering ricevono
comunque una predizione, segnata nella colonna `note`.

## Protocollo

- **Divisione ufficiale** (colonna `split` dei manifest). Scelta del registro (17/11): opzione C,
  `split.official_val.mode: all_test` (default): train ufficiale 988 take per teste e gate, val ufficiale
  285 take = test. Nessuna val separata: tutto si sceglie sul train fuori campione. Le altre modalità
  restano per confronto:
  - `split`: quota `test_frac` al test (opzione A = 0.5, B = 0.6);
  - `all_test`: opzione C, tutta la val ufficiale è test; il controllo e il gate usano il train fuori
    campione (`check.split: train`);
  - `all_val`: nessun test.
- **Ogni take è predetta da un modello che non l'ha vista.** Train: fold per sessione
  (`StratifiedGroupKFold`), la fold k è predetta dalla testa addestrata sulle altre (`oof`). Si ripete
  con 3 partizioni (`split.fold_seeds`) e si mediano le probabilità. Val e test: media dei modelli di
  fold di tutti i seed (`fold_ensemble`). `conf_seed_sd` dice quanto la predizione cambia tra i seed.
- C scelto in CV interna raggruppata per sessione (log-loss); temperatura sui logit fuori campione della
  stessa CV. Con `balanced` le probabilità favoriscono le classi rare rispetto alle frequenze reali.
- Finestra di 8 s al centro della take (`window.*`).
- **Test sigillato**: senza `--eval-test` le predizioni del test non vengono scritte e le sue metriche
  non vengono calcolate. Con `--eval-test` finiscono in `sealed/`, separate da train e val.
- Divisione per sito (`split.mode: site`): solo analisi secondaria, con `--skip-check`.

## Controllo pre-registrato

Dopo le teste, `heads` verifica su L0 (`check.model`):
1. sulla val (o sul train fuori campione con l'opzione C): accuratezza bilanciata ≥ `check.min_bacc` e
   limite inferiore dell'intervallo bootstrap (ricampionando le sessioni) sopra il caso (1 / classi valutate);
2. sul train fuori campione: richiamo > 0 per ogni classe della testa, Health compresa.

Soglie del registro (17/11), già in config: sul train fuori campione accuratezza bilanciata ≥ 0,60
(`check.min_bacc`) E richiamo di ogni classe ≥ 0,20 (`check.min_recall`). Il limite inferiore del bootstrap
sopra il caso è riportato ma non richiesto (`check.ci_above_chance: false`). Se il controllo fallisce, `heads` scrive comunque predizioni e `check.json` ed esce con
codice 3: il job SLURM risulta fallito e i job a valle non partono (anche ai rilanci).

## Avvertenze

1. **Pose della camera RGB, non del device.** CamFormer è pre-addestrato sulla traiettoria del device
   Aria (assi x sinistra, y su, z avanti; gravità lungo −z del mondo). I JSON danno la camera RGB in
   convenzione OpenCV (x destra, y giù). Default (registro, 17/11): nessuna correzione,
   `json_cam_to_device` = identità, come in cascade-v2. La conversione del paper per le pose non Aria,
   `diag(-1, -1, 1)` (supplementare, 2.3), si adotta solo con una voce nel registro. La camera RGB di Aria però
   può essere montata ruotata di 90° rispetto al device, e il paper non ne parla. Prima della run:
   `python -m scenario check-up` (solo JSON, pochi minuti di CPU). Guarda dove punta in media l'alto del
   mondo nel frame della camera e da questo:
   - sceglie il rollio (0/90/180/270°) e suggerisce `json_cam_to_device` (`check_up/summary.json`,
     `matches_current` dice se coincide con la config);
   - controlla che l'asse z del mondo punti in alto: se in media la camera "guarda in alto", il mondo è
     capovolto e il token di gravità sarebbe sbagliato;
   - riporta l'inclinazione mediana per classe (le procedurali guardano più in basso).

   Non può vedere un'inclinazione fissa tra camera RGB e device: quella si stima solo con pose vere del
   device (`check-frame`, facoltativo). Se `check-up` cambia la matrice, va scritto nel registro.
2. **Leakage di CamFormer.** L'encoder è pre-addestrato (contrastivo, con le narrazioni, senza
   etichette di scenario) sulle take del train ufficiale di Ego-Exo4D. La colonna `camformer_pretrain`
   dice per ogni take se il pretraining l'ha vista. Con la divisione ufficiale val e test dovrebbero
   esserne fuori: lo conferma il log di `index`.

## Uso sul cluster

```bash
# una volta, dal login node
git clone https://github.com/google-deepmind/seeing_without_pixels ~/seeing_without_pixels
#   egoexo4d_dur8.pt da camformer_checkpoints.zip (link in DATA.md) -> $CAMFORMER_CKPT
vim hpc/env.sh                         # percorsi dei pesi e account (o hpc/env.local.sh)
source hpc/env.sh && source $VENV/bin/activate && python -m pytest -q   # ~1-2 min, CPU

# frequenza di un CSV della v1 (atteso ~1000 Hz se è la traiettoria MPS originale)
python -c "import pandas as p,sys; t=p.read_csv(sys.argv[1],usecols=['tracking_timestamp_us']).iloc[:,0]; print(1e6/t.diff().median(),'Hz')" <csv>

# 1) l'indice, per decidere il taglio val/test con i conteggi davanti (non sono risultati),
#    e il controllo del frame delle pose (--n-check 0 = tutte le take)
python -m scenario index && cat $SCENARIO_RUN_DIR/index/summary.json   # official_val_composition, sessions_train_and_eval
python -m scenario check-up --n-check 0 && cat $SCENARIO_RUN_DIR/check_up/summary.json
# 2) dopo aver scritto nel registro taglio e soglia:
RUN_NAME=official_v1 hpc/submit.sh          # taglio (all_test) e soglie sono già i default della config
#    (se l'indice è già stato fatto con mode=undecided, rifarlo: python -m scenario index --overwrite --set ...)
# 3) alla fine, una volta sola:
python -m scenario heads --eval-test --set ...   # stessi --set della run
```

A mano: `python -m scenario <stadio> [--set k=v] [--overwrite]`, stadi `index`, `traj`, `check-up`, `check-frame`,
`feat-camformer`, `feat-egovlp`, `heads [--eval-test] [--skip-check]`. Uno stadio completo con la stessa
config si salta; con una config diversa si ferma (`--overwrite`). Traiettorie e feature non dipendono da
split, seed e soglia: cambiarli rifà solo `index` e `heads`.

Controlli della run nei log: 1273 take (`data.expected_takes`, altrimenti `index` si ferma), sessioni
sia nel train sia in val/test (`index`), take con buchi nei JSON (`traj`, atteso 52).

## Uscite (`$SCENARIO_RUN_DIR`)

| File | Contenuto |
|---|---|
| `predictions.csv` | train e val, una riga per take: split, fold (`fold_s*`), etichetta, `cam_*`, `vid_*` (pred, conf, conf_seed_sd, fonte, probabilità per classe) |
| `sealed/predictions.csv` | il test, solo dopo `--eval-test` |
| `check.json` | esito del controllo pre-registrato |
| `classes.json` | ordine delle classi, classi escluse, seed |
| `heads/<modello>/predictions.csv`, `metrics.json`, `folds.csv` | per modello: predizioni, metriche (con intervallo bootstrap), C/T/accuratezza per fold e seed |
| `index/takes.csv`, `index/summary.json` | indice; conteggi di take e sessioni per split e classe, composizione della val ufficiale |
| `traj/status.csv`, `features/*.npz` | esiti delle traiettorie (con `n_missing_src`, `has_gap`), embedding per take |
| `logs/`, `configs/` | log e config effettiva di ogni stadio |
