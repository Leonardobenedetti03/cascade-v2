# Risultati della cascata CamFormer → gate → EgoVLP: prima valutazione (v1) e CV annidata per sessione (v2)

*cascade-v2, 18/11/2026. Fonti: `results/scenario_v1/` (v1, registro N33–N34) e `results/scenario_cv/` (v2, registro
N37, job SLURM 772601, commit 96e391e). Tutti i numeri sono letti dai file di output, nessuno è ricalcolato a mano.*

## 0. In una riga

La cascata porta l'accuratezza da quella di CamFormer (0,88–0,92) a quella di EgoVLP (0,99–1,00) usando EgoVLP su
circa un terzo delle take, cioè con circa un terzo del calcolo di "solo EgoVLP". La v2 conferma il messaggio della v1
con una valutazione molto più solida: la semplice soglia sulla confidenza di CamFormer basta (i gate appresi non
aggiungono nulla), e il risparmio sulle sessioni nuove era sottostimato. LightGBM nella run 772601 non è valutabile
per un errore nella feature B, corretto nella run correttiva `n38_rel`.

## 1. Cosa è in comune

| | |
|---|---|
| Compito | Scenario di una take di Ego-Exo4D, 6 classi: Basketball, Dance, Soccer (fisiche), Cooking, Health, Bike repair (procedurali) |
| Dati | 1273 take; 1271 nelle 6 classi (2 di Bouldering escluse), 294 sessioni |
| L0 (economico) | CamFormer **congelato** (traiettoria della camera, 5 finestre, pose identità) + testa logistica. Costo 0,61 GMAC per take |
| L1 (costoso) | EgoVLP **congelato** (video, 5 finestre) + testa logistica. Costo 462,15 GMAC per take |
| Gate | Decide per ogni take se basta CamFormer o se chiamare EgoVLP. Feature: A = confidenza di CamFormer (p_max, margine, entropia), B = affidabilità della classe predetta, C = moto della camera |
| Cosa si addestra | Solo le teste logistiche e i gate. Le embedding di CamFormer ed EgoVLP sono calcolate una volta e riusate |
| Costo | "% di solo EgoVLP" = calcolo della cascata diviso il calcolo di EgoVLP su tutte le take |

## 2. La prima valutazione (v1, 17/11)

**Come è fatta.**
- Divisione fissa per sessione: train 988, val 143, test 142 (val ufficiale tagliata a metà per sessione e classe).
- Teste: 5 fold per sessione sul train, con predizioni fuori campione (OOF) sul train.
- Gate addestrati sulle OOF del train. Punto di lavoro scelto **sulla val**: minor uso di EgoVLP con accuratezza entro
  1 punto da "sempre EgoVLP".
- Test valutato **una sola volta** (17/11, 22:50, N34).

**Risultati** (costo in % di sempre EgoVLP):

| | Val (143) | Test (142) |
|---|---|---|
| Solo CamFormer | acc 0,909 | acc 0,915 |
| Solo EgoVLP | acc 1,000 | acc 1,000 |
| `lgbm_ABC` (scelto come principale) | acc 0,993, 25,5% | acc 1,000, 24,5% |
| `thr_maxprob` (soglia su p_max) | acc 0,993, 28,8% | acc 1,000, 28,3% |
| Oracolo | 9,2% | 9,3% |
| Gate casuale | acc 0,993, 87,1% | acc 0,992, 87,1% |
| Regola "manda le procedurali" | acc 0,979, 24,6% | acc 0,979, 24,0% |

- Sessioni nuove del test (67 take): CamFormer 0,82; cascata acc 1,000 con costo 50–58%.
- RQ2 (la traiettoria aiuta il gate?): no, p = 0,38.

**Problemi emersi** (analisi di Giorgio, 18/11):
1. Valutazione su pochi dati: 12–13 errori di CamFormer in val e in test.
2. Punto di lavoro al bordo, deciso da una sola sessione (`georgiatech_bike_06_10`): 1 errore su 143.
3. Risparmio gonfiato dalle sessioni già viste nel train (56% di val+test).
4. Costo usato solo come vincolo implicito; score delle logistiche non calibrati.
5. CV interna dei gate per take invece che per sessione.
6. Predizioni del train (media di 3 modelli) e di val/test (media di 15) su scale diverse.

## 3. La valutazione in CV annidata per sessione (v2, 18/11)

**Perché.** Il test della v1 era già stato aperto, quindi non si poteva usarlo per giudicare un metodo corretto dopo
averlo visto. Serviva una valutazione nuova, in cui ogni take fosse predetta da modelli che non hanno mai visto la sua
sessione. Disegno pre-registrato in N37 **prima** della run; la v1 non è stata cancellata.

**Come è fatta.**

| Livello | Cosa succede |
|---|---|
| Esterno | 1271 take divise in 5 fold per sessione (stratificate per classe), ripetuto 3 volte: 15 fold esterne. Ogni take viene valutata una volta per ripetizione |
| Teste (dentro ogni fold) | Teste di CamFormer rifatte con 5 fold interne per sessione → OOF per addestrare il gate; teste di CamFormer ed EgoVLP riaddestrate sul resto → predizioni sulla fold esterna |
| Gate (dentro ogni fold) | Addestrati in cross-fitting per sessione; CV interna per **sessione**; logistiche **senza** pesi di classe |
| Calibrazione | Platt: dallo score al rischio che CamFormer sbagli; limite prudente `r_up` dal bootstrap per sessione |
| Regole di soglia | **costo β** (scala se il rischio prudente ≥ β; β = 0,01 / 0,02 / 0,05 / 0,10 / 0,20); **rischio garantito** α (Clopper-Pearson con Bonferroni); **entro 1 punto** (la regola della v1); budget |
| Risultato principale | Take non viste dal pre-addestramento di CamFormer (337 take, 108 sessioni), `thr_maxprob`, regola a costo con β = 0,05 |
| Intervalli e test | Bootstrap per sessione (1000), confronti con correzione di Holm |

Run: 2 h 39 min su CPU (circa 2000 regressioni logistiche delle teste), nessun errore.

### 3.1 Risultato principale

| Take non viste (337) | Accuratezza | Quota a EgoVLP | GMAC per take | % di solo EgoVLP |
|---|---|---|---|---|
| Solo CamFormer | 0,882 | 0% | 0,6 | 0,1% |
| **Cascata, `thr_maxprob`, β = 0,05** | **0,995** [0,984 – 1,000] | **34,4%** [25,8 – 46,6] | **159,7** | **34,6%** [25,9 – 46,7], **2,9 volte meno** |
| Solo EgoVLP | 1,000 | 100% | 462,2 | 100% |
| Oracolo | 1,000 | 11,8% | 55,0 | 11,9% |

Su **tutte** le 1271 take: accuratezza 0,997, quota a EgoVLP 29,7%, 29,9% del calcolo (3,3 volte meno).

### 3.2 Curva su β (`thr_maxprob`)

β è il prezzo di una chiamata a EgoVLP espresso in errori: β = 0,05 vuol dire "accetto fino a 20 chiamate per ogni
errore evitato". Non si sceglie il β "migliore" guardando i risultati: si riporta la curva.

| β | 0,01 | 0,02 | **0,05** | 0,10 | 0,20 |
|---|---|---|---|---|---|
| Non viste: accuratezza | 1,000 | 0,998 | **0,995** | 0,985 | 0,980 |
| Non viste: % di solo EgoVLP | 47,1% | 41,4% | **34,6%** | 27,6% | 21,1% |
| Tutte: accuratezza | 0,999 | 0,998 | 0,997 | 0,992 | 0,986 |
| Tutte: % di solo EgoVLP | 43,9% | 37,9% | 29,9% | 23,2% | 16,1% |

### 3.3 Confronti pre-registrati (take non viste, p corretti con Holm)

| Confronto | Risultato | Lettura |
|---|---|---|
| `logreg_A` contro `thr_maxprob` | AUROC uguale (0,921 contro 0,921); quota −0,8 punti (p = 0,048) | Un gate appreso non aggiunge nulla di rilevante alla soglia su p_max (Prop. 2) |
| `logreg_AC` contro `logreg_A` (RQ2) | AUROC −0,002 (p = 1,0); quota −1,6 punti (p = 0,22) | La traiettoria non aiuta: risultato nullo confermato |
| `logreg_ABC` contro `logreg_AC` | Nessuna differenza | L'affidabilità per classe non aggiunge nulla |
| `thr_maxprob` contro regola per classe | AUROC 0,921 contro 0,798 (p < 0,001); acc +1 punto, quota +5 punti (n.s. dopo Holm) | La soglia ordina meglio gli errori; nelle classi procedurali tiene su CamFormer il 14% delle take (circa 1 errore su 14), la regola nessuna |

### 3.4 Controlli

- **Rischio garantito:** su tutte le take il limite sull'errore delle take lasciate a CamFormer regge in 15 fold su 15
  per α = 0,02 e 0,05 e in 13 su 15 per α = 0,10; **sulle non viste in 12, 9 e 7 su 15**: sul sottoinsieme principale
  la garanzia non regge, perché calibratore e soglie imparano soprattutto da take viste dal pre-addestramento
  (errore 2, N38). Correzione: calibrazione solo sulle non viste del train, run correttiva `n38_cal` (N39).
- **EgoVLP:** 0,999 in CV. Sbaglia solo 4 take, tutte della sessione `upenn_0720_Dance_3`, in una sola ripetizione.
- **Calibrazione:** sulle take non viste il rischio è sottostimato (previsto 9,3%, osservato 11,8%); la regola a costo
  usa il limite prudente `r_up`, che compensa in parte (vedi garanzia sopra).
- **`lgbm_ABC`: errore nella feature B, confermato (N38).** AUROC circa 0,70–0,76, molto sotto gli altri gate. La
  feature di affidabilità sulle righe di addestramento era calcolata togliendo la sola sessione della riga: è un
  leave-one-out target encoding (a totale fisso il valore cresce con gli errori della propria sessione; correlazione
  con l'errore 0,6–0,95 dentro classe). Gli alberi la sfruttano e imparano una regola rovesciata; la logistica quasi
  no. La valutazione resta pulita (nessun risultato gonfiato), ma `lgbm_ABC` di questa run **non è valutabile**.
  Correzione (N38): affidabilità delle righe di fit fuori fold per sessione, come nel gate v1; sul train LightGBM torna
  a 0,93–0,94, alla pari degli altri gate. Run correttiva `n38_rel`, riportata accanto (N39).

## 4. Cosa è cambiato tra v1 e v2

| | v1 (test, 142 take) | v2, tutte (1271) | v2, non viste (337) |
|---|---|---|---|
| Solo CamFormer | 0,915 | 0,924 | 0,882 |
| Solo EgoVLP | 1,000 | 0,999 | 1,000 |
| Oracolo, quota a EgoVLP | 9,3% | 7,7% | 11,9% |
| **Stessa regola** (entro 1 punto), `thr_maxprob` | acc 1,000, quota 28,3% | acc 0,990, quota 19,0% | acc 0,985, quota 23,5% |
| Regola a costo β = 0,05, `thr_maxprob` | – | acc 0,997, quota 29,7% | acc 0,995, quota 34,4% |
| `lgbm_ABC`, entro 1 punto | acc 1,000, quota 24,5% | non valutabile (errore feature B, N38) | non valutabile |
| Regola per classe | acc 0,979, quota 24,0% | acc 0,985, quota 22,7% | acc 0,985, quota 29,5% |
| AUROC `thr_maxprob` | 0,983 | 0,935 | 0,921 |
| RQ2 | nulla (p = 0,38) | nulla | nulla (p = 1,0) |
| Errori di CamFormer su cui si valuta | 12 | circa 96 per ripetizione | circa 40 |

1. **A parità di regola si usa meno EgoVLP (19–24% invece di 28%), ma l'accuratezza scende al limite di 1 punto
   (0,985–0,990 invece di 1,000).** L'1,000 del test v1 dipendeva in buona parte dalla dimensione piccola del test.
2. **La regola nuova a costo è più prudente:** 0,995–0,997 di accuratezza con il 30–34% di EgoVLP, e una curva su β al
   posto di un solo punto.
3. **Il gate di riferimento diventa `thr_maxprob`** (Prop. 2). Su LightGBM la run 772601 non dice nulla per l'errore
   della feature B (N38); il confronto valido viene dalla run correttiva `n38_rel` (N39).
4. **La soglia su p_max batte chiaramente la regola per classe** nell'ordinare gli errori (AUROC 0,92–0,94 contro
   0,78–0,80). Nella v1 sembravano quasi equivalenti.
5. **Le sessioni nuove costano meno di quanto diceva la v1:** 30–34% invece di 50–58% (in v1 erano solo 67 take).
6. **Intervalli molto più stretti** e punto di lavoro non più deciso da una singola sessione.

## 5. Limiti da dichiarare nel report

- Il disegno della v2 è stato deciso dopo aver visto il test della v1: la valutazione è nuova, il metodo non è cieco.
- EgoVLP è praticamente perfetto: il gate deve solo prevedere quando CamFormer sbaglia (Prop. 2); il compito a 6
  classi è facile per EgoVLP.
- Il pre-addestramento di CamFormer ha visto 934 delle 1271 take: per questo il risultato principale è sulle 337 non
  viste.
- La garanzia della regola a rischio assume take scambiabili, ma take della stessa sessione non lo sono.
- Costo misurato solo in GMAC dei due encoder: non include decodifica del video né stima della traiettoria.
- Encoder pre-addestrati e congelati; solo le teste e i gate sono addestrati.
- La v2 ha una nota tecnica minore: "solo EgoVLP" vale 462,15 GMAC per take nella v2 e 461,54 nella v1 (differenza
  0,13%, irrilevante per le conclusioni).

## 6. Dove sono i file

| | v1 | v2 |
|---|---|---|
| Cartella | `results/scenario_v1/` | `results/scenario_cv/` |
| Tabelle principali | `50_gate/.../operating_point.csv`, `metrics.csv`, `comparisons.csv` | `metrics.csv`, `cost_vs_egovlp.csv`, `comparisons.csv`, `summary.json` |
| Per take | `official_v2/heads/*/predictions.csv` | `takes.csv.gz` |
| Registro | N33, N34 | N37 (pre-registrazione), N38 (risultati e due errori trovati dopo la run), N39 (run correttive `n38_rel`, `n38_cal`) |
