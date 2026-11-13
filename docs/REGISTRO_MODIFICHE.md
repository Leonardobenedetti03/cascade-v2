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
| M04 | CamFormer da zero | Checkpoint pubblico come default | Il pretraining ha visto circa il 95% di train e val | previsto C08 |
| M05 | Fusione `replace` di default | Fusione log-lineare | Δ = 0,0 nella v1; gli ensemble non migliorano il fronte; L1 diventa l'esperto | previsto C09 |
| M06 | Gate per famiglie, CompareConfidence | Più di 15 gate senza riferimento | RQ2 tra famiglie citabili; la traiettoria è l'unica variabile tra `compare_A` e `compare_AC` | previsto C11 |
| M07 | Manifest con chiavi di split e fold | Cambio di fold non rilevato a valle | Impedisce di riusare in silenzio output calcolati con fold diverse | 1d8dd17 (C05) |
| M08 | Conversione delle pose come stadio 01 | Script esterno senza manifest | Riproducibilità | fe15eb0 (C07a), 31714af (C07b) |
| M09 | Accuratezza bilanciata, I(δ), trasferimento delle soglie | Solo accuratezza e metriche a budget | Squilibrio delle classi; metrica e analisi di Lebovitz | previsto C12 |
| M10 | Costo end-to-end e sensibilità al costo delle pose | Solo GMACs e latenza dei modelli | I GMACs non misurano decodifica e camera | previsto C10 |
| M11 | Un solo sistema di lancio, ambiente persistente | `slurm/` documentato ma non usato; script ad hoc | Riproducibilità e tempo | previsto C13 |
| M12 | Estensioni non portate: streaming, keystep, fine-tuning di L1, gate bayesiano | Presenti nel codice | Fuori dalle RQ; restano nella v1 | 41b0f62 (C04) |
| M13 | Baseline a un frame eseguita; tre livelli come analisi secondaria | Codice presente, mai eseguito | Senza, il confronto sul costo è incompleto | previsto C08 |
| M14 | Sei classi riportate; Bouldering fuori dalle metriche per classe; Music assente | Otto scenari dichiarati | Music non è nel sottoinsieme; 2 take non permettono stime | previsto C12 |
| M15 | Epoche e patience di CamFormer (D3) | 30 epoche, patience 5 | Fold 0 e full al tetto con la perdita in calo | previsto C08 |
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
