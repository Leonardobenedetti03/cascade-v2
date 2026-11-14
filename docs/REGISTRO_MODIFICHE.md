# Registro delle modifiche metodologiche

Base del paragrafo «Deviations from the original plan» del report. Una riga per modifica (M01–M16, dal blueprint),
con il commit che la introduce; «previsto» = commit non ancora fatto. Sotto, le note di lavorazione: scelte prese
durante la ricostruzione che il blueprint non prevedeva o prevedeva in modo diverso.

## Modifiche M01–M16

| ID | Che cosa cambia | Prima (v1) | Perché | Commit |
|----|-----------------|------------|--------|--------|
| M01 | Repository nuovo, codice portato selettivamente | Estensioni non usate e script di lancio fuori dal repository | Il codice consegnato è quello che produce i risultati | 9c9147d (C01) … |
| M02 | Split per sito: test georgiatech, val nus + fair | Split ufficiali per take | L1 al 99% riconosce in parte il laboratorio; leave-one-site-out 61,7%; criteri fissati prima dei risultati | v1 3d5f3b4; v2 88384ba |
| M03 | Fold di train per sito | Fold per take | Le predizioni fuori campione devono simulare lo spostamento verso siti nuovi | 88384ba (C06) |
| M04 | CamFormer da zero | Checkpoint pubblico come default | Il pretraining ha visto circa il 95% di train e val | d134b40 (C08a) |
| M05 | Fusione `replace` di default | Fusione log-lineare | Δ = 0,0 nella v1; gli ensemble non migliorano il fronte; L1 diventa l'esperto | aefe79b (C09) |
| M06 | Gate per famiglie, CompareConfidence | Più di 15 gate senza riferimento | RQ2 tra famiglie citabili; la traiettoria è l'unica variabile tra `compare_A` e `compare_AC` | previsto C11 |
| M07 | Manifest con chiavi di split e fold | Cambio di fold non rilevato a valle | Impedisce di riusare in silenzio output calcolati con fold diverse | 1d8dd17 (C05) |
| M08 | Conversione delle pose come stadio 01 | Script esterno senza manifest | Riproducibilità | fe15eb0 (C07a), 31714af (C07b) |
| M09 | Accuratezza bilanciata, I(δ), trasferimento delle soglie | Solo accuratezza e metriche a budget | Squilibrio delle classi; metrica e analisi di Lebovitz | previsto C12 |
| M10 | Costo end-to-end e sensibilità al costo delle pose | Solo GMACs e latenza dei modelli | I GMACs non misurano decodifica e camera | f7e6325 (C10) |
| M11 | Un solo sistema di lancio, ambiente persistente | `slurm/` documentato ma non usato; script ad hoc | Riproducibilità e tempo | previsto C13 |
| M12 | Estensioni non portate: streaming, keystep, fine-tuning di L1, gate bayesiano | Presenti nel codice | Fuori dalle RQ; restano nella v1 | 41b0f62 (C04), ca2154f (C08b) |
| M13 | Baseline a un frame eseguita; tre livelli come analisi secondaria | Codice presente, mai eseguito | Senza, il confronto sul costo è incompleto | previsto C08 |
| M14 | Sei classi riportate; Bouldering fuori dalle metriche per classe; Music assente | Otto scenari dichiarati | Music non è nel sottoinsieme; 2 take non permettono stime | previsto C12 |
| M15 | Epoche e patience di CamFormer (D3) | 30 epoche, patience 5 | Fold 0 e full al tetto con la perdita in calo | d134b40 (C08a) |
| M16 | `results/` nel repository | Risultati solo sull'HPC e in un backup | Ogni numero del report ha run e commit | previsto dopo le run |

## Note di lavorazione

**N1 — C02 pubblicato con due test in errore.** Il commit 2e87fc4 (C02) è stato fatto con `22 passed, 2 errors`:
`test_clips` importava `synthetic.py`, portato solo in C03 (1f3b7e5). La CI è partita dal push di C03, quindi C01 e C02
non hanno una CI propria. La storia pubblicata non è stata riscritta. Da C03 in poi pytest è verde prima di ogni commit.

**N2 — Che cosa fa davvero la conversione delle pose (M08).** Letto da `hpc/make_traj_data.py` della v1:
- sorgente: per ogni radice del dataset, `split_manifest.csv` (colonne `take_name`, `camera_pose_file`) indica un JSON
  per take in `annotations/ego_pose/*/camera_pose/` con `camera_extrinsics` = matrici 3x4 mondo → camera per frame, a
  circa 30 Hz; non sono le pose a 1 kHz del device. Le radici sono due (il sottoinsieme era diviso tra due account);
- il tempo non è nel file: il frame `i` vale `i / 30` s dall'inizio del video, quindi la traiettoria nasce nel tempo
  del video con origine al frame 0 e `alignment.mode: same_origin` vale per costruzione (resta da guardarlo allo
  stadio 15); l'ipotesi «30 Hz» è fissata nel codice v1 e ora è `data.pose_src_fps`;
- camera scelta come il video usato (`aria01_214-1.mp4` → `aria01`); buchi oltre 15 frame sorgente (0,5 s) senza posa.

Tre differenze volute dalla v1, tutte in `src/cascade/traj_convert.py`:
1. uscita `.npz` su griglia completa con maschera `valid`, al posto di un CSV con righe mancanti (che il loader v1
   riempiva col campione più vicino: il primo campione di un buco lungo, a 50 ms dall'ultimo valido, risultava valido o
   no a seconda dell'arrotondamento in virgola mobile; provato su 273 buchi simulati: valido in 79);
2. i campioni prima della prima posa sorgente (v1: copia della prima posa, validi) sono `valid = False`;
3. quaternioni con segno continuo da un campione al successivo (stessa rotazione): il quaternione grezzo cambia
   segno ogni volta che la rotazione attraversa 2π.
La conversione è stata confrontata campione per campione con il codice v1 incorporato nei test (`test_traj_convert.py`).

**N3 — L'inventario (00) deve girare dopo lo stadio 01, non prima.** Con la sorgente `converted_npz`, `has_traj`
dell'inventario controlla l'esistenza degli `.npz`: se 00 gira per primo, ogni take risulta senza traiettoria. Lo stadio
01 non dipende da 00 (legge l'elenco delle take dallo split), quindi l'ordine diventa 01 → 00. Applicato in C07b: `00_inventory.py` pretende lo
stadio 01 con `converted_npz`, e l'ordine è cambiato in `hpc/pipeline.sh` e nella pipeline sintetica (nel blueprint
01 è dopo 00).

**N4 — Percorsi dello stadio 01.** `paths.traj_dir` vale `{run}/01_traj` (blueprint: `{data}/traj/{run_name}`):
`{run_name}` non è un segnaposto di `paths.resolve` e la cartella della run è già persistente. I file si chiamano
`<take_name>.npz` (blueprint: `<take_uid>`): lo split e la v1 identificano le take per nome.

**N5 — Da riportare nelle config HPC (C13).** I file della v1 mostrano come ha girato davvero il cluster:
`data.duration_source: video` (niente `takes.json`), `alignment.mode: same_origin`, `expected_split_counts` assente, e
`--account`, `--partition` e `--qos` di `hpc/convert.sbatch`. `hpc/submit.sh` del blueprint passa account e partizione ma
non la QOS: va aggiunta se il cluster la richiede.

**N6 — Dati sintetici e sorgente di default.** `synthetic.py` scrive anche le pose per frame (`split_manifest.csv`,
`annotations/ego_pose/...`, stesso formato della sorgente reale), così lo stadio 01 gira nella pipeline sintetica e in
un test che confronta l'inventario con `converted_npz` e con i file attesi. `data.traj_source` resta `aria_csv` nei
default (lo usano i sintetici e i test esistenti); `configs/hpc/site_v2.yaml` (C13) imposta `converted_npz` e
`alignment.mode: same_origin`.

**N7 — Valori del cluster, dagli sbatch della v1** (da mettere in `hpc/env.local.sh`, non versionato): stessi account,
partizione `stud` e QOS `stud` per CPU e GPU; GPU richiesta con `--gpus=1` (la v1 non usava `--gres`: confermare con
`sbatch --test-only` quale forma accetta il cluster); `module load miniconda3`; dati in `~/cascade_data`
(`DL4CV_ROOT` e `CASCADE_DATA_ROOT`), run in `~/cascade_runs` (`CASCADE_RUN_ROOT`). La v1 ricreava il venv in `/tmp`
a ogni job: la v2 usa un ambiente persistente (M11).

**N8 — Ripresa per epoca (D7).** `finetune.train_fold` salva `ckpt_last.pt` a fine epoca (scrittura atomica) con
modello, ottimizzatore, stato migliore, contatore della patience e storia. La chiave è l'hash delle chiavi di config
dello stadio 20 più fold e seed: un checkpoint di un'altra config, o non leggibile, si ignora. Lo stadio lo cancella
dopo aver scritto il manifest, e una fold già fermata dall'early stopping non viene prolungata. Limite: lo stato dei
generatori casuali (dropout) non si salva, quindi una ripresa non riproduce bit per bit un addestramento mai interrotto.

**N9 — fusion.py e heads.py anticipati.** `calibration.py` importa `log_softmax` da `fusion.py`, e `fusion.py`
importa `heads.py` (riga 78): i tre file si portano identici già in C08a come dipendenze. `heads.py` arriva con il suo
test (`test_heads`); il test di `fusion.py` e le modifiche di M05 restano in C09.

**N10 — Frame persistenti e CamFormer da zero.** `paths.frames_dir` diventa `{data}/frames/{run_name}` (era
`{run}/10_frames`); `ctx.path` passa `run_name` (valore di default, sovrascrivibile). M04 e M15 applicati in
`configs/camformer.yaml`: `pretrained: false`, 50 epoche, patience 8; i default delle dataclass non cambiano.
`TrajectoryWindowDataset.prefix_frac` e `role_datasets(prefix_frac, roles)` restano come nella v1 (servivano alla
variante causale, tolta in M12): da togliere in un commit di pulizia.

**N11 — `cache/checks.py` anticipato.** Nella v1 `check_loss` e `ProvenanceError` stanno in `cache/build.py` (stadio 45,
C12), ma gli stadi 25, 28 e 30 (C08b) li importano. Invece di anticipare `build.py` intero, i due simboli si portano
identici in `src/cascade/cache/checks.py` (con `tests/test_checks.py`); in C12 `build.py` li importerà da lì, senza
copie duplicate.

**N12 — `eval/style.py` anticipato.** Lo stadio 15 (figura di allineamento) importa `GRID`, `SERIES`, `SURFACE` e
`TEXT2` da `eval/report.py` (C13). Le quattro costanti (colori della v1) vivono in `src/cascade/eval/style.py`; in C13
`report.py` le importerà da lì. Le altre costanti di palette (`TEXT`, `REF_GRAY`, `SEQ_BLUE`) restano in `report.py`.

**N13 — Modelli e stadi di L1 e della baseline (C08b).** Portati dalla v1: `models/egovlp.py`, `models/frame_baseline.py`, `models/vendor/egovlp_video_transformer.py` (copia modificata dal submodule `third_party/EgoVLP`, 85 righe di differenza: va tenuta identica alla v1), `tests/test_egovlp.py` e gli stadi 15, 25, 28 e 30. Cambiano solo gli import: `check_loss` e `ProvenanceError` da `cache/checks.py` (N11), le costanti di colore da `eval/style.py` (N12). Non si porta `models/egovlp_finetune.py` (stadio 26, M12). La pipeline sintetica esegue 15 dopo il 10 e 25, 28 e 30 dopo i 20; la baseline a un frame gira perché `frame_baseline.enabled` è `true` in `configs/synthetic.yaml`.

**N14 — Fusione `replace` (C09, M05) e stadio 35 spostato in C12.** In `fusion.py` `replace` è il default (`choose_w` e `FusionCfg.mode`); `fuse` resta come ablation. Tolta la fusione appresa: `fusion_features`, `learned_fusion`, il parametro `learned` di `compare_fusions`, i campi `learned_features` e `compare_learned`, il valore `learned` di `fusion.mode` e il tag `_flearn` di `variant_tag`. Lo stadio 35 importa `assemble_camformer`, `join_l0_l1` e `load_l1` da `cache/build.py` (C12): si porta in C12 insieme a `build.py`, già senza la fusione appresa, invece di anticipare `build.py`. `fusion.py` non importa più `heads.py`, che resta per gli stadi 28 e 30 (N9). `test_label_y` della v1 usa `cache/schema.py` e va con `test_cache_schema` in C12.

**N15 (C10, M10).** Stadio 40 (GMACs con tre contatori) copiato da v1 senza modifiche. Stadio 41 nuovo: tempo per fase a batch 1 di L0 (lettura traiettoria, preprocessing, forward) e di L1 (apertura e decodifica, preprocessing, forward), su un campione stratificato per classe (cost_e2e.n_clips) con clip di riscaldamento scartate, sincronizzazione GPU, mediana e p90, due passate (si riporta la seconda, la prima si dichiara). Pesi casuali con l'architettura vera: il tempo non dipende dai valori. Il blueprint non fissa questi dettagli; scelta: il tempo di L0 comprende la lettura del file della traiettoria senza cache. Soglia di pareggio della traiettoria C_pose < (1 - r) * C_L1 - C_L0 - C_g in secondi e (se c'e' lo stadio 40) in GMACs.

**N16 (C11, M06).** Gate portato dalla v1 (la cartella `gate/` era vuota in v2) con le modifiche del blueprint. `models.py`: aggiunto `CompareGate` (sottoclasse di `LogRegGate`, `class_weight=None`, `target = "c1"`, score `η̂₁ − p_max`; il primo gruppo deve essere A) e il ramo `compare_` in `make_gate`; tolti `RidgeGate`, `LGBMRegGate` e il ramo bayesiano. `make_gate(name, gcfg)` perde il parametro `labels`, che serviva solo al gate bayesiano: lo stadio 50 (C12) lo chiama così. `features.py`: tolti i gruppi `mc` e `take` e l'affidabilità Beta-Binomiale (`FeatureBuilder` perde `reliability_mode`, `beta_n0`, `beta_q`, `prior_confusion`); restano A, B, C, `onehot` ed `emb` (il blueprint toglie solo `mc` e `take`, quindi `onehot` resta). `protocol.py`: `run_gate` sceglie la colonna bersaglio da `gate.target` con `TARGET_COLUMN = {"hard": "y", "c1": "c1"}`; tolti il bersaglio `soft`, il livello −1 (`level_minus1_forced`) e il ramo `needs_class`. `gate/bayes.py` non si porta. `GateCfg` perde `reliability`, `beta_n0`, `beta_q`, `prior_confusion` e `ridge_alphas`: `reliability: beta` richiedeva `bayes.py`, quindi resta solo l'affidabilità empirica (scelta presa qui, il blueprint non la fissa). `motion.py` (gruppo C) non era ancora in v2: copiato da v1 senza modifiche (TIENI). `configs/gate.yaml` segue le famiglie della sezione 5.2: modelli `always_L0, always_L1, random, oracle, thr_maxprob, thr_margin, thr_entropy, logreg_A, logreg_AC, logreg_ABC, lgbm_ABC, compare_A, compare_AC`; confronti `[compare_A, thr_maxprob]` e `[compare_AC, compare_A]` (le due coppie di RQ2) più `[logreg_AC, logreg_A]` (scelta mia: lo stesso effetto della traiettoria nella famiglia addestrata su y). Test: `test_gate` riscritto senza `cache/mock.py` (C12), con una cache finta costruita nel test e `roc_auc_score` di sklearn al posto di `src.eval.metrics.roc_auc`; `test_compare_gate` nuovo, con il test sintetico del blueprint (un L1 che riesce dove la velocità è alta); non si portano `test_level_minus1` né i test bayesiani. `tests/test_motion.py` portato da v1 senza modifiche. Le diagnostiche D1/D2 (AUROC di η̂₁ contro c1, tabella `diag_c1.csv`) vanno nello stadio 50 (C12). `lightgbm` su macOS richiede `libomp` (`brew install libomp`).
