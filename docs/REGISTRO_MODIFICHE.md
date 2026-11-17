# Registro delle modifiche metodologiche

Base del paragrafo «Deviations from the original plan» del report. Una riga per modifica (M01–M16 dal blueprint, M17 decisa dopo),
con il commit che la introduce; «previsto» = commit non ancora fatto. Sotto, le note di lavorazione: scelte prese
durante la ricostruzione che il blueprint non prevedeva o prevedeva in modo diverso.

## Modifiche M01–M17

| ID | Che cosa cambia | Prima (v1) | Perché | Commit |
|----|-----------------|------------|--------|--------|
| M01 | Repository nuovo, codice portato selettivamente | Estensioni non usate e script di lancio fuori dal repository | Il codice consegnato è quello che produce i risultati | 9c9147d (C01) … |
| M02 | Split per sito: test georgiatech, val nus + fair | Split ufficiali per take | L1 al 99% riconosce in parte il laboratorio; leave-one-site-out 61,7%; criteri fissati prima dei risultati | v1 3d5f3b4; v2 88384ba; stadio 27 52546c7 (C12c) |
| M03 | Fold di train per sito | Fold per take | Le predizioni fuori campione devono simulare lo spostamento verso siti nuovi | 88384ba (C06) |
| M04 | CamFormer da zero (**sostituita da M17**) | Checkpoint pubblico come default | Il pretraining ha visto circa il 95% di train e val | d134b40 (C08a) |
| M05 | Fusione `replace` di default | Fusione log-lineare | Δ = 0,0 nella v1; gli ensemble non migliorano il fronte; L1 diventa l'esperto | aefe79b (C09) |
| M06 | Gate per famiglie, CompareConfidence | Più di 15 gate senza riferimento | RQ2 tra famiglie citabili; la traiettoria è l'unica variabile tra `compare_A` e `compare_AC` | 9b9b0fb (C11) |
| M07 | Manifest con chiavi di split e fold | Cambio di fold non rilevato a valle | Impedisce di riusare in silenzio output calcolati con fold diverse | 1d8dd17 (C05) |
| M08 | Conversione delle pose come stadio 01 | Script esterno senza manifest | Riproducibilità | fe15eb0 (C07a), 31714af (C07b) |
| M09 | Accuratezza bilanciata, I(δ), trasferimento delle soglie | Solo accuratezza e metriche a budget | Squilibrio delle classi; metrica e analisi di Lebovitz | 4d55e67 (C12b); 52546c7 (C12c) |
| M10 | Costo end-to-end e sensibilità al costo delle pose | Solo GMACs e latenza dei modelli | I GMACs non misurano decodifica e camera | f7e6325 (C10) |
| M11 | Un solo sistema di lancio, ambiente persistente | `slurm/` documentato ma non usato; script ad hoc | Riproducibilità e tempo | 9bbe819 (C13a); catena a 4 job: previsto (C14c-2) |
| M12 | Estensioni non portate: streaming, keystep, fine-tuning di L1, gate bayesiano | Presenti nel codice | Fuori dalle RQ; restano nella v1 | 41b0f62 (C04), ca2154f (C08b) |
| M13 | Baseline a un frame eseguita; tre livelli come analisi secondaria | Codice presente, mai eseguito | Senza, il confronto sul costo è incompleto | previsto C08 |
| M14 | Sei classi riportate; Bouldering fuori dalle metriche per classe; Music assente | Otto scenari dichiarati | Music non è nel sottoinsieme; 2 take non permettono stime | 4d55e67 (C12b); 52546c7 (C12c) |
| M15 | Epoche e patience di CamFormer (D3) | 30 epoche, patience 5 | Fold 0 e full al tetto con la perdita in calo | d134b40 (C08a) |
| M16 | `results/` nel repository | Risultati solo sull'HPC e in un backup | Ogni numero del report ha run e commit | previsto dopo le run |
| M17 | CamFormer dal checkpoint del paper, encoder congelato, si allena solo la testa; EgoVLP invariato (congelato, testa logistica) | v1: checkpoint con fine-tuning end-to-end; blueprint (M04): da zero | Decisione del gruppo (16/11): modelli pre-addestrati su molti più dati, si addestrano solo i classificatori. Limiti dichiarati in N24 | previsto (C14d) |
| M18 | L0 e L1 da `scenario_preds` (teste logistiche su CamFormer ed EgoVLP congelati, 6 classi, 3 seed), cache del gate dallo stadio 44 al posto di 00–45; divisione ufficiale con la val tagliata a metà (val/test) | Stadi 00–45 di cascade-v2, divisione per sito (M02/M03) | Il controllo su L0 della run `site_v2` è fallito (siti mai visti per CamFormer congelato); la divisione per sito resta analisi secondaria. Dettagli in N28–N31 | previsto (C15) |

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

**N21 — C14a: correzioni bloccanti dell'audit di Giorgio (15/11).** Verificate sul codice prima di correggerle.
- B1: con `data.traj_source: converted_npz` (la run `site_v2`) l'inventario cercava la traiettoria in
  `paths.traj_pattern`, cioè il CSV di Aria, e non trovava i `.npz` dello stadio 01: tutte le take sarebbero finite
  in `missing_takes.csv`. Ora `clips.resolve_take_paths` usa `<paths.traj_dir>/<take_name>.npz`; `traj_pattern`
  resta per `aria_csv`. Il test di M08 metteva a mano l'override di `traj_pattern` e così nascondeva il bug: tolto,
  il test fallisce col codice vecchio e passa col nuovo. `data.traj_source` e `paths.traj_dir` entrano nelle chiavi
  del manifest dello stadio 00.
- B2: in `hpc/pipeline.sh` gli stadi 15 e 41 partivano dopo lo stadio 01, ma leggono `data/clips.csv` dello stadio
  00: ora dipendono da `$j00`; il 41 anche da `$j40`, perché scrive `breakeven_pose_gmacs` solo se
  `40_cost/<hz>/cost.json` esiste già (trovato da Giorgio). Aggiornato `tests/test_hpc_dryrun.py`.
- B3: `paths.pose_roots` vuoto non era controllato prima della run. `src/cascade/hpc_check.py` (quindi
  `hpc/check_cluster.sh`), con `converted_npz`, dà FALLITO se la lista è vuota o se una radice non ha
  `split_manifest.csv`. `takes.json` mancante resta un avviso, come concordato: la v1 sul cluster girava senza (N5)
  e la durata ripiega sul video. È un errore solo con `alignment.mode: device_us`, dove gli start in µs stanno
  soltanto lì (prima `hpc_check` dava sempre FALLITO e lo stadio 00 sempre un avviso).
  `pose_roots` in `base.yaml` resta un `TODO(HPC)`.
- I6: `cost.expected.camformer` da 0,02 a 0,12 GMACs. Con l'architettura di `base.yaml` e il contesto di 8 s a 20 Hz
  (T = 160 più il token di gravità) la formula analitica dà 0,1216 GMACs, lo stesso valore di
  `cost.override_gmacs.L0` dei sintetici; con 0,02 il sanity check dello stadio 40 avrebbe segnalato uno scarto di
  6× su un conteggio corretto. Un test lega il valore atteso alla formula.
- Fonte: PDF di Giorgio del 15/11. La sua `correzioni_provate_v2.patch` non era disponibile: le modifiche sono
  scritte da capo e vanno confrontate con la patch. B4 (stadio 55) non ha una soluzione temporanea: arriva con
  C13b, prima della run vera.
- Restano per C14b: I1, I2, I3, I4, I7, m1, m3, m4, m5.

**N22 — C14b, prima parte: I1, I2, I4 e il test della config vera (I8).** Da `correzioni_provate_v2.patch` di Giorgio,
applicata dove non coincideva già con C14a; ogni test nuovo fallisce col codice di prima.
- I1: in `heads.fit_head` le fold interne si contavano sulle clip e non sulle take, e Bouldering (una take nel fit
  delle fold 1–4 di `site_v1`) portava `n_splits` a 1: CV saltata e C = 0,001 fisso, senza avviso. Ora le classi con
  una sola take restano fuori dalla CV interna e dentro il fit finale; se nessuna classe ha due take si usa `Cs[0]` con
  un avviso. Vale anche per la testa di L½. Test: `tests/test_heads_rare_class.py` (fold versionate, C invariato
  aggiungendo la take rara, colonne allineate, classe rara imparata).
- I2: le chiavi dello stadio 00 (`CLIP_KEYS`, ora con `paths.split_sha256`) passano agli stadi 10, 20, 25, 27, 28, 30,
  35, 41, 45 e 50, come `SPLIT_KEYS` in M07; il 41 ha le sue chiavi. Rifare lo stadio 00 con un'altra definizione delle
  clip dà `ConfigMismatch` invece di riusare output vecchi. Lo stadio 10 salta comunque i `.npy` già presenti: dopo
  un cambio delle finestre serve `--overwrite`.
- I4: lo stadio 50 valuta L½ quando `half_probs_cal` è nella cache (`per_class.csv`, `confusion.csv`). La cascata a
  tre livelli resta non portata (decisione aperta su M13).
- I8: `tests/test_site_v2_chain.py` lancia 01 → 00 → 05 → 20 sui sintetici con `configs/hpc/site_v2.yaml` (fold per
  take, perché i sintetici hanno un solo sito). `tests/test_hpc_dryrun.py` ricava le dipendenze di `pipeline.sh` da
  ciò che ogni script legge: è il test che ha trovato la dipendenza 41 → 40.
- Restano per la seconda parte di C14b: I3 (bootstrap e confronti sul train, metrica per fold), I7, m1, m3, m4, m5.

**N23 — C14c, prima parte: valori del cluster e percorsi veri (ricognizione del 15/11 in sola lettura).**
Verificati sul login node con `sacctmgr`, `sinfo`, `scontrol`, `sbatch --test-only` (nulla in coda) e `find`; la v1
(`hpc/*.sbatch`, `hpc/make_traj_data.py`) indicava dove guardare.
- Cluster: account 3185472, partizione e QOS `stud`. Le GPU sono partizioni MIG (`3g.40gb`, `4g.40gb`, `7g.80gb`):
  `--gres=gpu:1` è accettato e resta il default (la v1 usava `--gpus=1`, accettato anch'esso). Python 3.9.25 di sistema,
  rete dal login node verso PyPI, PyTorch e GitHub: `MODULES` vuoto, torch `cu121`. I valori stanno in
  `hpc/env.local.sh` (non versionato) e, per memoria, in `docs/RUNBOOK_HPC.md`.
- `paths.pose_roots`: le due metà del subset, `/home/3185472/egoexo4d_split_45G_physical` (992 take) e
  `/home/3218863/Oasis_Deep_Vision_DSBA/egoexo4d_split_45G` (281), ognuna con `split_manifest.csv` e
  `annotations/ego_pose/{train,val}`. 992 + 281 = 1273 = 1138 + 68 + 67, e l'elenco coincide con
  `splits/site_v1/scenario_site.csv`. I due manifest hanno colonne diverse; lo stadio 01 legge solo `take_name` e
  `camera_pose_file`, presenti in entrambi. Le radici vanno in `configs/hpc/site_v2.yaml` e non in `base.yaml`
  (i sintetici hanno le loro); `tests/test_site_v2_chain.py` le rimette sui sintetici.
- `takes_root`: il blueprint presuppone una sola cartella con tutte le take, il subset ne ha due. Scelta: riusare
  l'albero di link costruito in v1 (`/home/3185472/cascade_data/egoexo4d/takes`, 1273 take, 0 link rotti, ogni
  `frame_aligned_videos` punta alla radice vera), collegato come `$DL4CV_ROOT/data/egoexo4d/takes`. Nelle cartelle
  ci sono anche i `closed_loop_trajectory.csv` della v1: `site_v2` (`converted_npz`) non li legge. Alternativa
  scartata per ora: uno script versionato che ricostruisce i link dai manifest (stesso risultato, più lavoro).
- `video_pattern`: il prefisso dello stream varia (`aria01` 878, `aria02` 102, `aria` 10, `aria1` 2 take nella prima
  metà); ogni take ha un solo `*214-1*.mp4`, che `video_glob` trova. Invariato.
- `takes.json` non c'è: `data.duration_source: video` in `site_v2.yaml`, esplicito invece del ripiego silenzioso
  (come la v1, N5). Con `alignment.mode: device_us` resterebbe un errore di `hpc_check` (B3).
- Pesi: EgoVLP dal file della v1 (link), MobileNetV3-Small scaricato dal login node (hash `047dcff4` verificato).
- Chiave GitHub: deploy key di `cascade-v2` in lettura e scrittura, solo sul cluster. Il codice si cambia dal Mac.
- Aperto (C14c, seconda parte): la QOS `stud` ammette 4 job in coda e 2 in esecuzione per utente, e ogni elemento di
  un array conta. `hpc/pipeline.sh` ne sottomette circa 20, più gli array degli stadi 10 e 20: SLURM rifiuterebbe il
  quinto `sbatch`. Va riorganizzato in pochi job con più stadi in sequenza, come faceva la v1 (`chain.sbatch`).
- Restano `TODO(HPC)` da chiudere con lo stadio 15: `alignment.mode`, `data.vertical_axis`, `frames.rotate_k90`.

**N24 — C14d: CamFormer pre-addestrato e congelato (M17, sostituisce M04).** Decisione del gruppo del 16/11: per
entrambi i livelli si usano i pesi pubblicati e si addestra solo il classificatore. Il blueprint diceva il contrario
(pagina 11, «Addestramento. Da zero (M04)», e tabella del registro a pagina 45): la scelta presa è M17.
- EgoVLP non cambia: era già congelato (EgoVLP_PT_BEST, stadio 25) con la regressione logistica dello stadio 30.
- CamFormer: `configs/camformer.yaml` passa a `pretrained: true` e al nuovo `freeze_encoder: true`. Con
  `freeze_encoder` (`finetune.make_optimizer`) i pesi dell'encoder non richiedono gradiente, l'ottimizzatore riceve
  solo la testa lineare (lr × `head_lr_mult`) e in addestramento l'encoder resta in `eval` (niente dropout). Senza,
  il comportamento è quello di prima (fine-tuning end-to-end). La config rifiuta `freeze_encoder` senza
  `pretrained`; `synthetic.yaml` resta a pesi casuali con `freeze_encoder: false`. La chiave entra nel manifest dello
  stadio 20 con il blocco `camformer` (già incluso) e in `summary.json`.
- Checkpoint: `egoexo4d_dur8.pt` da `camformer_checkpoints.zip` (Drive degli autori, `third_party/seeing_without_pixels/DATA.md`),
  in `paths.camformer_ckpt`. È quello indicato dagli autori per Ego-Exo4D: `pose_encoding` `rel9d_grav` e contesto di
  8 s, uguali a `camformer.arch` e a `data.window_s` (nessun cambio di config). `hpc_check` lo richiede (con
  `pretrained` vero lo controllava già).
- Limiti da dichiarare nel report: (1) il pretraining ha visto circa il 95% delle take di train e val (il motivo di
  M04); se abbia visto anche il test va verificato sullo split degli autori. Con take già viste la confidenza di L0
  sulla val può essere troppo alta, e le soglie del gate si calibrano sulla val. (2) Le nostre pose sono quelle della
  camera RGB convertite a 20 Hz (N2), non quelle del device a 1 kHz del pretraining: con l'encoder congelato la
  differenza conta (il blueprint la riteneva irrilevante solo con CamFormer da zero). Controllo fissato prima dei
  risultati: dopo lo stadio 20, l'accuratezza bilanciata di L0 sulla val deve stare chiaramente sopra quella della
  classe di maggioranza su tutti i seed; altrimenti ci si ferma prima di guardare il test. (3) I tre seed cambiano
  solo l'inizializzazione della testa e l'ordine dei batch: la varianza tra seed sarà piccola.
- Test: `test_freeze_encoder_trains_only_the_head`, `test_without_freeze_the_encoder_is_finetuned`,
  `test_freeze_needs_pretrained_weights` (`tests/test_camformer.py`); `test_hpc_check` aggiornato (il checkpoint
  CamFormer ora serve); `test_resume` usa `freeze_encoder=false` (il modello giocattolo non ha un encoder separato).

**N25 — C14e: ambiente sull'account di Giorgio e caricamento del checkpoint EgoVLP vero (16/11).** La run gira
dall'account HPC di Giorgio (3195149): la home di Leo è quasi piena (circa 47 GB su 50) e lo spazio conta per
proprietario. Il dataset resta dov'è e si legge dalle home di Leo e Jacopo; `egovlp.pth` è diventato leggibile da tutti
(`chmod go+r`, il file non è stato toccato). Sull'account di Giorgio: clone con una deploy key propria («HPC giorgio»,
lettura e scrittura), `hpc/env.local.sh` con `SLURM_ACCOUNT=3195149` e `DL4CV_ROOT=$HOME/cascade_v2_root`, ambiente in
`~/venvs/cascade-v2` (5,6 GB). Con l'indice `cu121` pip installa torch 2.5.1 e torchvision 0.20.1, non la 2.8 del Mac:
la suite passa uguale. `check_cluster.sh site_v2`: tutto a posto, compresi i tre pesi e `sbatch --test-only`.
- Unico test fallito: `test_egovlp.py::test_real_checkpoint_loads`, `ModuleNotFoundError: No module named
  'parse_config'`. Il checkpoint EgoVLP_PT_BEST salva anche la config di addestramento come oggetto
  `parse_config.ConfigParser` del repository EgoVLP. `load_state` ora usa un Unpickler che sostituisce le classi non
  importabili con un segnaposto mai usato (serve solo `state_dict`); scartata l'alternativa di mettere
  `third_party/EgoVLP` nel path, che importerebbe il loro codice di addestramento. Test:
  `test_load_state_ignores_classes_missing_from_the_path`, che riproduce l'errore.
- Il checkpoint CamFormer `egoexo4d_dur8.pt` (scaricato dal Drive degli autori e copiato con `scp`) si carica:
  `test_camformer.py::test_real_checkpoint_loads` passa sul cluster.

**N26 — C14f: prova dbg_smoke sul cluster, chiusi i tre TODO(HPC) dello stadio 15 (16/11).** Run `dbg_smoke` con
`site_v2` dall'account di Giorgio, commit 3b5f6b4. Stadio 01 (job 768476, 3 min): 1273 take convertite su 1273, 2 con
righe prima della prima posa, 52 con buchi. Stadio 00 (job 768529, 3 min): 1273 clip, tutte utilizzabili, 3 take con
problemi segnalate (da guardare prima della run vera, non bloccano). Stadio 15 (job 768535, 2 min): 6 take della val
(nus_covidtest_36_2, nus_cooking_10_4, nus_covidtest_01_2, fair_cooking_06_4, nus_cooking_17_5, nus_covidtest_32_3).
- `frames.rotate_k90 = 0`: in tutte le 48 miniature i frame sono dritti (tavolo e mani in basso, soffitto in alto).
- `data.vertical_axis = 2`: la quota resta entro circa ±0,25 m dalla posa iniziale, anche nelle take di cucina in cui
  la persona si sposta di metri; un asse orizzontale darebbe escursioni di metri. Gli abbassamenti coincidono con i
  frame in cui la persona si china (lavello in nus_cooking_10_4 tra 50 e 95 s, tavolo in nus_covidtest_32_3 tra 47 e
  63 s).
- `alignment.mode = same_origin`, `offset_s = 0`: in tutte e sei le take la traiettoria copre il video dall'inizio
  alla fine, senza tratti vuoti a un capo (uno sfasamento di secondi si vedrebbe lì), e gli abbassamenti cadono sui
  frame corrispondenti. Limite: con 8 frame per take il controllo esclude sfasamenti di qualche secondo, non di
  frazioni di secondo; rispetto a una finestra di 8 s l'effetto residuo è piccolo. `device_us` non serve.
- Blueprint e realtà: la sequenza «00 --inspect, poi 15» del RUNBOOK non funziona, perché `--inspect` non scrive il
  manifest dello stadio 00 e lo stadio 15 lo richiede (`FileNotFoundError` nel job 768518). Scelta presa: nessuna
  modifica al codice (è il comportamento voluto di `--inspect`); il RUNBOOK ora elenca 01, 00 completo, 15.
- Solo commenti e documentazione: i valori delle config non cambiano (config caricate identiche prima e dopo).

**N27 — C14c-2: la run come catena di 4 job, strada A per gli stadi 50–55, take corte (16/11).** Chiude il punto
aperto di N23: la QOS `stud` ammette 4 job in coda per utente e conta ogni elemento di un array, e `hpc/pipeline.sh`
del blueprint (17.4: circa 25 sottomissioni, più gli array dello stadio 10 e dello stadio 20) avrebbe visto rifiutato
il quinto `sbatch`. Blueprint e realtà non coincidono; scelta presa: quattro job, ognuno con più stadi in sequenza,
come il `chain.sbatch` della v1.
- `hpc/submit.sh --chain <nome> "<stadio> [argomenti]; ..."` sottomette un job che esegue gli stadi in ordine
  (`STAGE_LIST`). Risorse: nodo GPU se almeno uno stadio è GPU, CPU e memoria massime, tempo = somma dei tempi della
  tabella (×6 per le fold dello stadio 20, ×`NUM_SHARDS` per lo stadio 10), al massimo 24 ore. La tabella degli stadi
  passa da `submit.sh` al nuovo `hpc/stages.sh`, letto anche da `stage.sbatch`. `submit.sh <stadio>` resta invariato
  per i rilanci singoli.
- `hpc/stage.sbatch` in modalità catena: le fold dello stadio 20 (0–4 e `full`) girano una dopo l'altra; lo stadio 10
  gira una volta su tutte le clip, senza shard (manifest `shard000of001`, che lo stadio 25 accetta). Uno stadio con
  `?` (lo stadio 15) è diagnostico: se fallisce, la catena va avanti, come nel blueprint dove 15 «non blocca nulla»;
  uno script che manca e ha `?` si salta. Al primo errore di uno stadio normale il job esce con il suo codice e
  `--kill-on-invalid-dep` cancella i job a valle. Il requeue al limite di tempo resta: il job riparte dal primo stadio
  e i manifest saltano quelli completi.
- `hpc/pipeline.sh`: A (CPU) 01 → 00 → 05 → 10 → 15?; B1 (GPU) 25 → 28 → 40 → 41 → 20 per il primo seed; B2 (GPU) 20
  per gli altri seed, in parallelo a B1 (8 + 4 CPU e 2 GPU: dentro i 12 CPU e 2 GPU della QOS); C (CPU) 27 → 30 → 35 e
  45 per seed → 50 per seed → 52 → 55. Con un seed solo B2 e 52 non ci sono; 55 entra solo se lo script esiste
  (C13b). Tempi stimati con `NUM_SHARDS=8`: A 10 h, B1 8 h 30, B2 12 h, C circa 6 h 30; sono le stime della tabella,
  da correggere con `sacct` dopo la run.
- `hpc/final.sh`: un solo job (50 `--final` per seed, 52, 55) invece di cinque, che con tre seed superavano la QOS.
  Con `NO_REQUEUE=1` (`--no-requeue` e `MAX_REQUEUE=0`): un rilancio automatico valuterebbe il test una seconda volta.
- Strada A (decisione del 16/11): la run vera parte prima di C14b-2 (I3, I7, m1–m5) e di C13b (stadio 55), che
  cambiano soprattutto gli stadi 50–55. Questi si rifanno dopo, al commit nuovo, con `hpc/pipeline.sh --redo-gate`
  (un job: 50 per seed e 52 con `--overwrite`, poi 55), senza rifare 00–45. Il blueprint chiede un solo commit nei
  manifest 00–52 (principio 8 e `verify_run`): `verify_run.sh` ora accetta un commit per 00–45 e uno, uguale al
  clone, per 50–52, solo se il primo precede il secondo (`git merge-base --is-ancestor`) e tra i due non cambia
  nessuno script `scripts/` degli stadi 00–45 (altrimenti FALLITO); le differenze in `src/` e `configs/` le elenca
  da controllare a mano, perché lo stesso modulo può servire sia a monte sia al gate. Le chiavi di config cambiate
  le ferma già il manifest (`ConfigMismatch`). Limite da dichiarare nel report: i risultati citano due commit. Se
  C14b-2 tocca codice degli stadi 00–45 (per esempio I7, «cache ridotta», nello stadio 45), la strada A non basta:
  si rifà da quello stadio in poi.
- Test (`tests/test_hpc_dryrun.py`, 28 invece di 13): le dipendenze della catena si ricavano ancora da ciò che ogni
  script legge, ora contando l'ordine dentro il job; 4 job entro i limiti della QOS; contenuto e risorse di ogni
  job; `--redo-gate`; `--chain` con stadi sconosciuti; `NO_REQUEUE`. `stage.sbatch` ora gira nei test, con un
  `python` finto: ordine degli stadi, sei fold, stadio 10 senza shard, stadio diagnostico che fallisce, arresto al
  primo errore, script mancante saltato, task singolo di array, commit sbagliato (exit 3). `verify_run.sh` su run_dir
  finte: un commit, strada A accettata, script a monte cambiato, 50–52 non al commit del clone, commit misti a monte.
  I test che eseguono `stage.sbatch` e `verify_run.sh` richiedono `sha256sum` e altrimenti si saltano. Provati in
  sandbox (Python 3.9, bash 5); 17 dei nuovi test falliscono con gli script di prima.
- Le 3 «take con problemi» dello stadio 00 di `dbg_smoke` (N26) sono `short_take`, tutte nel train e tutte Basketball:
  sfu_basketball_08_40 (3,6 s), sfu_basketball_11_40 (7,1 s), sfu_basketball_11_44 (6,6 s). Con `data.short_take:
  keep` la clip è la take intera. Gli stadi a valle le reggono: lo stadio 10 prende i frame dentro la finestra vera,
  la copertura della traiettoria (stadi 20 e 45, `slice_window`) si calcola sulla durata della finestra e non su 8 s,
  CamFormer prende sequenze di lunghezza variabile con la maschera di padding. Il blueprint non tratta il caso.
  Scelta presa: si tengono (`keep`, nessuna modifica). Con l'encoder congelato (M17) entrano solo nell'addestramento
  della testa lineare: 3 esempi con un contesto più corto, nessuno in val o test. Alternativa scartata: `drop`, cioè
  addestrare solo su finestre piene come il pretraining di `egoexo4d_dur8.pt`.

**N28 — C15: gate sulle predizioni di `scenario_preds`, stadio ponte 44, 6 classi (17/11).** Decisione del gruppo
del 17/11 (pomeriggio): L0 e L1 vengono dal pacchetto `scenario_preds` di Giorgio (M18); di `cascade-v2` restano
costi (40), gate e valutazione (50).
- `configs/labels.yaml` a 6 classi, nell'ordine delle teste di `scenario_preds`: Basketball, Dance, Soccer, Cooking,
  Health, Bike repair. Music (0 take) e Bouldering (2 take) fuori (come M14). `mock.py` a 6 classi. Lo stadio 00 con
  il CSV di CamFormer a 8 scenari non si usa più: `test_real_scenario_csv_mapping_by_name` è saltato con il motivo;
  i conteggi di `test_converted_inventory` scendono da 46 a 34 take sintetiche (quelle di Music e Bouldering escono).
- Nuovo `scripts/44_scenario_bridge.py`: scrive `45_cache/<variante>/cache.parquet` con lo stesso schema e lo stesso
  manifest dello stadio 45 (lo stadio 50 non cambia). Probabilità = `cam_p_*`/`vid_p_*` di `predictions.csv` (già
  calibrate dalla testa), logit = log delle probabilità, `T = 1`; fusione `replace` (`fused_probs` = L1, `w = 0`);
  `fold` = `fold_s0` sul train, −1 altrove; embedding da `features/*.npz`; `mot_*` dalle traiettorie `traj/*.npz`
  (solo campioni validi) nelle stesse finestre delle embedding, mediate; costi dello stadio 40 × numero di finestre.
  Il test entra solo con `--include-test` (da `sealed/predictions.csv`), al passo finale; senza, 44 si rifiuta di
  girare se `predictions.csv` contiene righe del test.
- Take senza embedding CamFormer (audit 17/11, punto 3): fuori dalla cache, elencate in `no_l0.csv` con la
  correttezza di L1. Regola fissata ora: nel risultato finale vanno a EgoVLP e si contano come costo L0 + L1
  (accuratezza e costo della cascata si correggono con queste take prima di riportarli).
- Punto di lavoro (stadio 50): nuova chiave `gate.operating_rule`. `within_pts` (default di questa run) = quota di
  escalation minima con accuratezza su val ≥ accuratezza di always_L1 su val − `within_pts` (0,01); se nessuna quota
  basta, tutto a L1. `budget` = la regola di prima (50% del costo di always_L1). Con `within_pts` la colonna
  `budget_frac` del punto di lavoro vale −1 (lo stadio 52 raggruppa per quella colonna e scarterebbe i NaN). I budget
  25% e 50% e la curva completa restano nei risultati come confronto.
- `gate.bacc_min_takes` da 5 a 1: una classe esce dall'accuratezza bilanciata solo con 0 take (come in N29).
- Lancio: `configs/hpc/scenario_v1.yaml`, `hpc/gate_scenario.sh` (un job CPU: 40 → 44 → 50, senza `--final`).
  Lo stadio 52 non serve: la media sui seed è già dentro `scenario_preds`.
- Test: `tests/test_stage_44.py` (schema, take senza L0, Bouldering fuori, costi × finestre, test sigillato,
  catena 44 → 50 anche con `--final`, classi in ordine diverso, righe del test in `predictions.csv`);
  `test_operating_point_within_pts_on_val` e `test_stage_50_without_final` controllano la nuova regola.

**N29 — Modifiche a `scenario_preds` per l'audit del 17/11 (passo A).** Config di default allineata al piano del
pomeriggio: `split.official_val.mode = split` (val ufficiale tagliata una volta a metà, per sessione e stratificata
per classe, `test_frac` 0,5, seed 0). Controllo su L0 (`check`): sulla val accuratezza bilanciata ≥ 0,40 e limite
inferiore dell'intervallo bootstrap 95% (per sessione) sopra il caso; sul train fuori campione richiamo > 0 per ogni
classe; nuovo `require_all_classes`: fallisce se una delle 6 classi ha 0 take nella val o nel train
(`heads.min_takes_metric = 1`: una classe esce dall'accuratezza bilanciata solo con 0 take). Take senza embedding
CamFormer: restano nelle predizioni con `cam_pred` vuoto, contate (`n_no_embedding`); `feat-camformer` si ferma solo
oltre il 10% (`camformer.max_missing_frac`). Assi delle pose: due varianti pre-registrate, identità (default) e
diag(−1, −1, 1); con 1, 3 e 5 finestre fanno 6 candidate (`hpc/candidates.sh`), e il nuovo stadio `select` sceglie
sulla val la massima accuratezza bilanciata di CamFormer (a parità meno finestre, poi identità); EgoVLP usa lo stesso
numero di finestre. `check-up` resta solo diagnosi. 18 test (15 di Giorgio + 3).

**N30 — Correzione di N24, punto (2).** La differenza dal pretraining non è la frequenza (20 Hz contro 1 kHz: il
pretraining usa comunque traiettorie ricampionate a 20 Hz) ma il sistema di riferimento: le nostre pose sono quelle
della camera RGB (estrinseche dei JSON), il pretraining usa la traiettoria del dispositivo Aria. Per questo la
correzione degli assi è una scelta pre-registrata sulla val (N29) e non un valore fisso.

**N31 — Limiti da dichiarare e analisi decise prima della run (17/11).**
- Variante delle pose e finestre si scelgono sulla stessa val su cui si fa il controllo su L0: il controllo è un po'
  ottimistico; il test resta intatto.
- Il gate impara sulle probabilità fuori campione del train, che in `scenario_preds` sono la media di 3 seed (un
  modello per seed), e si applica a val e test, che sono la media di 15 modelli (5 fold × 3 seed): probabilità meno
  estreme. Il punto di lavoro si sceglie sulla val, quindi sulla stessa distribuzione a cui si applica.
- Il pretraining di CamFormer ha visto le take del train ufficiale (N24): le probabilità OOF del train possono essere
  troppo sicure. Si riportano i risultati separati per `camformer_pretrain`.
- Val e test sono piccoli (circa 140 take ciascuno: 1 punto ≈ 1,4 take): accuratezza e costo della cascata con
  intervallo bootstrap per take/sessione.
- Sessioni in comune tra train e val/test (log di `index`): risultato principale su tutte le take, più un'analisi di
  controllo senza quelle sessioni.
- Analisi aggiuntive, solo valutazione (il gate non cambia): accuratezza di CamFormer per fasce di movimento della
  camera (`mot_*`) sul train fuori campione, in totale e dentro ogni classe; confronto con la regola fissa
  «classe procedurale prevista da L0 → EgoVLP».

**N32 — C16: analisi del gate decise prima della run (N31), stadio 53.** Lo stadio 50 scrive in `scores.parquet`,
per ogni gate, la decisione al punto di lavoro (`esc__<gate>`, quota dei seed che scalano) e, se ci sono, sessione,
sito, `camformer_pretrain` e costi. Il nuovo `scripts/53_scenario_analysis.py` (solo val; il test con `--final`, dopo
50 `--final`) calcola, al punto di lavoro: (1) accuratezza, accuratezza bilanciata e costo **contando le take senza
L0** di `no_l0.csv` come mandate a EgoVLP con costo L0 + L1 (regola di N28); (2) lo stesso senza le sessioni che
compaiono anche nel train (nella run vera 19 sessioni, 160 take di val+test su 285); (3) intervalli bootstrap 95%
ricampionando le sessioni, anche della differenza di accuratezza da always_L1; (4) la regola fissa
`rule_procedural` (scala se la classe prevista da L0 è procedurale); (5) i risultati separati per
`camformer_pretrain`; (6) l'accuratezza di L0 sul train fuori campione per quintili di `mot_energy` (soglie dal
train), in totale e per classe, con figura. Gira in `hpc/gate_scenario.sh` dopo lo stadio 50. Il gate non cambia.
Test: `test_analysis_53` e la catena finale in `test_bridge_cache_and_gate` (`tests/test_stage_44.py`).

**N33 — Risultati di val congelati prima del test (17/11, sera).** Run `scenario_preds` `official_v2` (commit
`ebbf20a` di cascade-v2 per il gate) e gate `scenario_v1`. Nessun numero del test è stato guardato; dopo questa voce
non si cambia più nulla del metodo e il test si valuta una volta sola.
- Candidate (stadio `select`, bacc di CamFormer sulla val, 143 take): identità con 5 finestre 0,824 (CI 0,759–0,914),
  identità 3 finestre 0,790, diag(−1,−1,1) 5 finestre 0,789, identità 1 finestra 0,679, diag 3 finestre 0,659,
  diag 1 finestra 0,566. Scelta: **identità, 5 finestre**.
- Indice: 1273 take; val 143 e test 142 dalla val ufficiale, tutte e 6 le classi da entrambe le parti (Soccer 7 in
  val); 19 sessioni in comune tra train e val/test (160 take su 285), di cui 10 sessioni e 85 take nella val.
- Controllo su L0 superato: bacc val 0,824 (CI 0,759–0,914) contro caso 0,167; nessuna classe sotto soglia o
  mancante; 0 take senza embedding. CamFormer: acc 0,924 / bacc 0,861 sul train fuori campione, acc 0,909 sulla val.
  EgoVLP: acc e bacc 1,000 sia sul train fuori campione sia sulla val (L1 al tetto).
- Gate, punto di lavoro sulla val (entro 1 punto da always_L1 = 1,000): tutti i gate appresi e a soglia hanno
  acc 0,993 (1 errore su 143) e bacc 0,983; costo rispetto ad always_L1: lgbm_ABC 0,255, logreg_ABC 0,274,
  logreg_AC 0,281, thr_maxprob/thr_entropy/logreg_A/compare_A/compare_AC 0,288, thr_margin 0,302; oracolo 0,092;
  random 0,895 (acc 0,993). Regola `rule_procedural`: acc 0,979, costo 0,246 (non rispetta il criterio).
  Differenza di accuratezza da always_L1 (bootstrap per sessione, 37 sessioni): da −0,025 a 0 per i gate.
- Senza le sessioni in comune (58 take di val): CamFormer acc 0,810; ai punti di lavoro scelti su tutta la val i gate
  hanno acc 0,983 con costo 0,51–0,59 (lgbm_ABC 0,512).
- Moto (train fuori campione, quintili di `mot_energy`): acc di L0 0,74 nel quintile più basso contro 0,95–0,99
  negli altri; dentro la classe sale con il moto per Basketball (0,84 → 0,98), Cooking (0,68 → 0,90), Bike repair
  (0,63 → 1,00); Health va al contrario ma con 7 take nel secondo quintile.
- `camformer_pretrain`: tutte le 143 take di val sono nella val del pretraining di CamFormer (non nel suo train).
- Limite: il punto di lavoro è scelto e misurato sulla stessa val ed è al bordo (1 errore): sul test può cambiare
  di una take in più o in meno.

**N34 — Test, valutazione unica (17/11, 22:50).** Dopo N33 (commit 362f5e2): previsioni sigillate di `scenario_preds`
(`heads --eval-test`, job 770706) e stadi 44 `--include-test`, 50 `--final`, 53 `--final` (job 770707); una riga in
`50_gate/FINAL_RUNS.log`, `FINAL_DONE` presente. Output copiati in `results/scenario_v1/` (senza i `.parquet`,
esclusi da `.gitignore`).
- Test, 142 take: CamFormer acc 0,915, bacc 0,791 (CI 0,627–0,907); EgoVLP acc e bacc 1,000; 0 take senza L0.
- Gate con la soglia scelta sulla val: acc 1,000 e bacc 1,000 per tutti i gate appresi e a soglia; costo rispetto ad
  always_L1: lgbm_ABC 0,245, logreg_ABC 0,262, logreg_AC 0,269, thr_maxprob/thr_margin/thr_entropy/logreg_A/
  compare_A/compare_AC 0,283; oracolo 0,093; random acc 0,992 con costo 0,871. Regola `rule_procedural`: acc 0,979,
  bacc 0,941, costo 0,240. Differenza di accuratezza da always_L1 (bootstrap per sessione, 40 sessioni): 0 per i gate.
- Senza le sessioni in comune col train (9 sessioni, 75 take; restano 67 take): CamFormer acc 0,821; gate acc 1,000 con
  costo 0,50–0,58 (lgbm_ABC 0,503); regola procedurale acc 0,955.
- Quota mandata a EgoVLP per classe (test): thr_maxprob Basketball 0,11, Dance 0,07, Soccer 0,67, Cooking 0,75,
  Health 0,80, Bike repair 0,88; lgbm_ABC 0,11, 0,02, 0,67, 0,69, 0,77, 0,75.
- RQ2 (confronti appaiati, 1000 ricampionamenti): `compare_AC` e `compare_A` identici; `logreg_AC` − `logreg_A`
  sull'area della curva −0,0006 (CI −0,0020 / 0,0006, p 0,38). Le feature di traiettoria e moto non migliorano in
  modo significativo il gate rispetto alla sola confidenza di CamFormer; i gate con il gruppo C scalano 3–5 take in
  meno sia su val sia su test (tendenza, non prova).

