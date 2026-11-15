# Registro delle modifiche metodologiche

Base del paragrafo «Deviations from the original plan» del report. Una riga per modifica (M01–M16, dal blueprint),
con il commit che la introduce; «previsto» = commit non ancora fatto. Sotto, le note di lavorazione: scelte prese
durante la ricostruzione che il blueprint non prevedeva o prevedeva in modo diverso.

## Modifiche M01–M16

| ID | Che cosa cambia | Prima (v1) | Perché | Commit |
|----|-----------------|------------|--------|--------|
| M01 | Repository nuovo, codice portato selettivamente | Estensioni non usate e script di lancio fuori dal repository | Il codice consegnato è quello che produce i risultati | 9c9147d (C01) … |
| M02 | Split per sito: test georgiatech, val nus + fair | Split ufficiali per take | L1 al 99% riconosce in parte il laboratorio; leave-one-site-out 61,7%; criteri fissati prima dei risultati | v1 3d5f3b4; v2 88384ba; stadio 27 52546c7 (C12c) |
| M03 | Fold di train per sito | Fold per take | Le predizioni fuori campione devono simulare lo spostamento verso siti nuovi | 88384ba (C06) |
| M04 | CamFormer da zero | Checkpoint pubblico come default | Il pretraining ha visto circa il 95% di train e val | d134b40 (C08a) |
| M05 | Fusione `replace` di default | Fusione log-lineare | Δ = 0,0 nella v1; gli ensemble non migliorano il fronte; L1 diventa l'esperto | aefe79b (C09) |
| M06 | Gate per famiglie, CompareConfidence | Più di 15 gate senza riferimento | RQ2 tra famiglie citabili; la traiettoria è l'unica variabile tra `compare_A` e `compare_AC` | 9b9b0fb (C11) |
| M07 | Manifest con chiavi di split e fold | Cambio di fold non rilevato a valle | Impedisce di riusare in silenzio output calcolati con fold diverse | 1d8dd17 (C05) |
| M08 | Conversione delle pose come stadio 01 | Script esterno senza manifest | Riproducibilità | fe15eb0 (C07a), 31714af (C07b) |
| M09 | Accuratezza bilanciata, I(δ), trasferimento delle soglie | Solo accuratezza e metriche a budget | Squilibrio delle classi; metrica e analisi di Lebovitz | 4d55e67 (C12b); 52546c7 (C12c) |
| M10 | Costo end-to-end e sensibilità al costo delle pose | Solo GMACs e latenza dei modelli | I GMACs non misurano decodifica e camera | f7e6325 (C10) |
| M11 | Un solo sistema di lancio, ambiente persistente | `slurm/` documentato ma non usato; script ad hoc | Riproducibilità e tempo | previsto C13 |
| M12 | Estensioni non portate: streaming, keystep, fine-tuning di L1, gate bayesiano | Presenti nel codice | Fuori dalle RQ; restano nella v1 | 41b0f62 (C04), ca2154f (C08b) |
| M13 | Baseline a un frame eseguita; tre livelli come analisi secondaria | Codice presente, mai eseguito | Senza, il confronto sul costo è incompleto | previsto C08 |
| M14 | Sei classi riportate; Bouldering fuori dalle metriche per classe; Music assente | Otto scenari dichiarati | Music non è nel sottoinsieme; 2 take non permettono stime | 4d55e67 (C12b); 52546c7 (C12c) |
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

**N17 (C12a, M05).** Cache e fusione portati dalla v1: `cache/build.py`, `cache/schema.py`, `cache/mock.py` e gli stadi 35 e 45. `build.py` importa `check_loss` e `ProvenanceError` da `cache/checks.py` (N11) e li riesporta (i test li importano anche da lì); tolte le colonne e i vettori MC dropout (`CAM_VECTORS`, `MC_VECTORS`, `MC_SCALARS`), perché il gruppo `mc` del gate non esiste più (N16). `schema.py`: tolto il ramo `fusion_mode == "learned"` (un `w` NaN ora è un errore), le colonne opzionali MC e `take_duration`. `mock.py`: tolta l'opzione `mc`; con `choose_w` in modalità `replace` la fusione finta coincide con L1. Stadio 35: tolti `learned_fusion`, `fused_learned.parquet` e il tag `_l1ft`; output in `35_fusion/seed<S>_hz<F>/`. Stadio 45: tolti il ramo della fusione appresa e la colonna `take_duration` (serviva all'analisi degli errori, che non si porta). `manifest.py`: le chiavi dello stadio 35 aggiungono `camformer.eval_source`, `gate.bootstrap` e `gate.bootstrap_seed`; quelle del 45 aggiungono `data.window_s`, `data.min_traj_coverage`, `data.traj_fps`, `camformer.eval_source`, `fusion` e `frame_baseline` (prima un cambio di fusione o di baseline non invalidava la cache; scelta mia, il blueprint non elenca le chiavi). Test: `test_cache_build` (con `eval_source=full` e `load_l1`), `test_cache_schema` con `test_label_y` riscritto (il corpo originale della v1 non era stato letto) e `test_mock_fusion_is_replace`. Non c'è un test pytest di integrazione per 35 e 45: li esegue `scripts/run_synthetic_pipeline.sh`, che li ha dopo 30 e dopo 41.

**N18 — C12b: valutazione dei gate (stadi 50 e 52).** Scelte prese durante la scrittura di M09 e M14, dove il
blueprint e il codice divergevano o non dicevano:

- `gate.within_pts` resta uno scalare (0,01) e indica il costo minimo «entro un punto da always_L1»; i δ di I(δ)
  stanno in `gate.improvement_deltas` = [0; 0,01]. Il blueprint elencava i δ dentro `within_pts`. Campi nuovi di
  `GateCfg`: `bacc_min_takes` = 5 e `transfer_budgets` = [0,25; 0,5].
- Nessuna decisione forzata del livello −1: il vettore `forced` della v1 non esiste più (livello non portato, M12/M13).
- I(δ) = C_L1 / C(r*_δ), con il numeratore letto dalla curva (`cost[r = 100] − cost[r = 0]`). Con `replace` il minimo
  è C_L1 / (C_L0 + C_L1) ≈ 0,9987: sotto 1 la cascata non risparmia nulla.
- Accuratezza bilanciata (`bacc_*`) = media sulle classi del richiamo per take, con le sole classi che hanno almeno 5
  take nello split (Bouldering resta fuori, M14). Se nessuna classe arriva a 5, la metrica è NaN e non un errore.
  Il gate di riferimento per la cascata per classe è quello con `bacc_area` maggiore su val.
- Trasferimento delle soglie (RQ1, solo con `--final`): la soglia scelta su val per un budget di costo
  (C_L0 + r·C_L1 ≤ b·(C_L0 + C_L1), quindi a b = 0,5 la quota è appena sotto il 50%) applicata al test, con lo
  sforamento di costo e il divario di accuratezza rispetto alla soglia ricalcolata sul test, che è solo un riferimento.
  Si saltano i gate a r fisso e l'oracolo.
- Diagnostica D1: AUROC di η̂₁ = score + p_max contro c1 per `compare_A` e `compare_AC`, su train fuori campione e
  val, mai sul test (la funzione solleva un errore). Celle A/B/D/N per split e classe in `cells.csv`.
- `--final` sullo stadio 50 rigira sempre (non consulta il manifest), scrive `FINAL_DONE` e una riga in
  `FINAL_RUNS.log`; senza `--final` lo stadio si rifiuta di girare se `FINAL_DONE` esiste nella cartella.
- Lo stadio 52 aggrega anche `threshold_transfer` e `diag_c1`; cade il suffisso `_l1ft` (fine-tuning di L1 non portato).
- Pipeline sintetica: lo stadio 50 gira con `gate.bacc_min_takes=1`, perché con 2 take per classe in val e test
  la soglia di 5 lascerebbe la bacc sempre NaN e il percorso non verrebbe esercitato.
- Le moltiplicazioni per i pesi bootstrap usano `einsum` e non `@`: con numpy 2.0 e Accelerate (macOS arm64)
  `@` emette avvisi spuri («divide by zero / overflow / invalid value in matmul») su numeri finiti; verificato
  che i risultati coincidono con `einsum` (differenza 0 sui conteggi, ~1e-11 sui costi).
- Non portati dalla v1: `three_level`, `error_analysis`, `_cohen_d`, il gate `bayes_*` e `src.eval.metrics`
  (l'AUROC viene da scikit-learn).
- Sul Mac (numpy 2.0.2 con Accelerate; Python 3.9 non permette numpy più recenti) le pipeline stampano centinaia di
  `RuntimeWarning: ... encountered in matmul` da `gate/features.py` e dalla regressione logistica di scikit-learn.
  Sono avvisi spuri: `@` ed `einsum` danno gli stessi numeri e tutti i valori sono finiti. Non si filtrano, per non
  nascondere avvisi veri; su Linux con OpenBLAS non dovrebbero comparire (da controllare sull'HPC in C13).

Il commit di C12b (4d55e67) è entrato nelle righe M09 e M14 della tabella con C12c.

**N19 — C12c: diagnostiche di sito (stadio 27; M02, M09, M14).** Scelte prese dove blueprint (12.6) e brief divergevano.

- Le tre analisi del blueprint vanno in tre file: leave-one-site-out per sito e per classe in `loso.csv` (a),
  leggibilità del sito dalle CLS in `site_probe.json` (b; è nuova: nella v1 `check_leak.py` predice la classe, non il
  sito), baseline «classe più frequente del sito» in `site_majority.json` (c). In `site_probe.json` c'è anche il
  controllo di `check_leak.py` (CV della classe per take e per sessione, `label_cv`): il salto rispetto al
  leave-one-site-out è la misura della scorciatoia di sito.
- Perimetro: train e val, mai il test. Lo stadio gira fuori da `50 --final`, come 25 e 30. La v1 usava tutte le take,
  quindi i numeri non sono confrontabili: il 61,7% citato in M02 resta il numero della v1, con quel perimetro.
- Il sito è il prefisso del `take_uid`; la sessione è il `take_uid` senza l'ultimo suffisso (come in `check_leak.py`).
  Sonda: StandardScaler + LogisticRegression con C = 0,1 e max_iter = 500 (`site_diag.C`, `site_diag.max_iter`), CV con
  `site_diag.cv_folds` = 5. Gli avvisi di convergenza non si nascondono: si contano in `fits` di `site_probe.json`.
- Nel leave-one-site-out si valutano solo le clip la cui classe è nel train di quel giro (come `check_site.py`); le
  escluse sono contate. `reported` segue `gate.bacc_min_takes` (M14: Bouldering resta fuori dalle medie per classe).
- I ruoli per la baseline vengono da `load_scenario` (indici di `labels.yaml`) e non da `roles_from_scenario`, che usa
  la colonna `label` del CSV, in un altro ordine: servono i nomi delle classi. `site_majority_accuracy` è invariata:
  in-sample sul train e su tutte le take dello split, anche quelle senza video.
- Lo stadio `27_site` è in `STAGE_KEYS` (`site_diag`, `gate.bacc_min_takes`) e tra gli stadi che dipendono dallo split
  (M07). Nessuna variante per seed: usa le CLS dello stadio 25 e non dipende da CamFormer. `hpc/submit.sh`,
  `hpc/pipeline.sh` e `hpc/collect_results.sh` lo prevedevano già.
- Dati sintetici: tutti i `take_uid` iniziano con `syn`, quindi c'è un solo sito. La sonda si salta (`skipped` nel
  JSON, `loso.csv` con la sola intestazione) e lo stadio scrive comunque i tre file; la logica con più siti è provata
  in `tests/test_site_diag.py` su array costruiti a mano.
- Aperti, non toccati qui: M13 dice ancora «previsto C08», ma `three_level` non è stato portato (N18), quindi la
  cascata a tre livelli non è valutata nello stadio 50; il blueprint (11.2) ha `transfer_budgets` da 0,1 a 0,9, la
  config [0,25; 0,5].
- L'hash di C12c entra in M02, M09 e M14 con il commit successivo (C13), come per gli altri.

**N20 — C13a: lancio sull'HPC.**
- Tag `results-v1` del repository v1: punta ad `a06691b`, il commit che ha prodotto i 49 manifest, come dice il
  blueprint. `a5f66f2`, visto con `rev-parse --short`, è l'oggetto del tag annotato e non un commit. Nessuna
  discrepanza; gli script di lancio della v1 sono in `8a4d9c2`, più recente.
- `hpc/env.sh` ha `SLURM_QOS` e `hpc/submit.sh` passa `--qos` quando è impostata. I valori personali (account,
  partizione, QOS) stanno in `hpc/env.local.sh`, fuori dal repository.
- La tabella degli stadi di `hpc/submit.sh` è una funzione con `case` e non un array associativo: stessi valori, ma
  gira anche con la bash 3.2 del Mac, quindi `tests/test_hpc_dryrun.py` la prova in locale oltre che in CI.
- `configs/hpc/site_v2.yaml` contiene solo ciò che distingue la run: il nome, `data.traj_source: converted_npz`
  (il blueprint 11.2 lo mette in `base.yaml`, che nel codice ha `aria_csv` perché i sintetici lo usano: scelta,
  `base.yaml` invariato), la baseline a un frame accesa (M13) e lo zero-shot spento (D8: nessun modello di testo da
  scaricare sui nodi). Un test verifica che le sole chiavi diverse da `base.yaml` siano queste.
- `hpc/check_cluster.sh` e `src/cascade/hpc_check.py`: controlli sul login node (parametri, import di `torch` e
  `transformers`, percorsi di dati e pesi, hash dello split, `sbatch --test-only`). `transformers>=4.30` è in
  `env/requirements.txt` e `hpc/setup_env.sh` ne stampa la versione; che sia importabile sul cluster si vede solo lì.
- `docs/RUNBOOK_HPC.md`, citato da `hpc/env.sh`, ora esiste.
- Aperti: `id_v2` non si può lanciare (`splits/id_v1/` ha solo `.gitkeep`; il confronto H4 resta fuori da C13); i
  `TODO(HPC)` di `base.yaml` (pattern dei percorsi, `pose_roots`, `alignment.mode`, `vertical_axis`) si chiudono con la
  prova `dbg_smoke` e con lo stadio 15; `tests/test_hpc_dryrun.py` esenta `55_report.py` finché non esiste (C13b).
- L'hash di C12c (`52546c7`) è ora in M02, M09 e M14; quello di C13a entra in M11 con il commit successivo.
