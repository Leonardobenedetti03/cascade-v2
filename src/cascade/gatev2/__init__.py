"""Gate v2 (N37): CV annidata per sessione su tutte le take, gate calibrato, regole a costo e a rischio.

Moduli:
- ``data``: take, embedding congelate di CamFormer ed EgoVLP e moto, da una run di scenario_preds;
- ``folds``: partizioni stratificate per classe e raggruppate per sessione;
- ``heads``: teste lineari con C e temperatura in CV per sessione (protocollo di scenario_preds);
- ``scorers``: feature del gate (gruppi A, B, C) e scorer (soglia, logistica, LightGBM, regola per classe);
- ``calibrate``: dallo score al rischio di errore di CamFormer, con limite superiore bootstrap;
- ``rules``: regole di decisione (costo β, rischio garantito α, accuratezza obiettivo, budget);
- ``nested``: i tre livelli della CV annidata per una partizione esterna;
- ``evaluate``: metriche, curve, AUROC, confronti appaiati per sessione, calibrazione, analisi procedurale.

Vincolo di informazione (come la v1): al momento della decisione il gate vede solo l'uscita di CamFormer e la
traiettoria. Label ed EgoVLP entrano solo come target (addestramento) e per contare gli errori (valutazione).
"""
