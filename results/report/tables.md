# Tables of the report

Made by `scripts/06_make_report_tables.py` from `results/nested_cv/report_n38_cal/`.

## Table 1. RQ1: accuracy and compute

Main columns: takes not used to pre-train CamFormer (n = 337). Last two columns: all takes (n = 1,271).
Compute is the percentage of EgoVLP-only GMAC per take. The cascade uses the threshold gate with the cost rule at β = 0.05.

| System | Acc [95% CI] | bAcc | Comp. % | Acc (all) | Comp. % (all) |
|---|---|---|---|---|---|
| CamFormer only | .882 [.826, .922] | .803 | 0.1 | .924 | 0.1 |
| EgoVLP only | 1.000 | 1.000 | 100.0 | .999 | 100.0 |
| Oracle gate | 1.000 | 1.000 | 11.9 | 1.000 | 7.7 |
| Class rule | .985 [.963, .999] | .989 | 29.6 | .985 | 22.8 |
| Cascade (threshold) | .997 [.989, 1.000] | .992 | 42.4 | .998 | 38.7 |

Compute interval of the cascade: [32.5, 55.5]%.

## Table 2. RQ2: gate ablation

Takes not used to pre-train CamFormer (n = 337). Escalated and Acc at the cost rule with β = 0.05 (the class rule at its own share). Errors left: CamFormer errors per repeat left when the gate escalates the 30% of takes with the highest calibrated risk. ΔAUROC: against the row above, or class rule against threshold.

| Gate | Features | AUROC [95% CI] | Escalated (%) | Acc | Errors left at 30% | ΔAUROC (p_Holm) |
|---|---|---|---|---|---|---|
| Threshold | p_max | .914 [.844, .957] | 42.2 | .997 | 3.3 |  |
| Logistic | A | .915 [.845, .957] | 45.2 | .997 | 4.0 | +.001 (1.0) |
| Logistic | A+C | .911 [.845, .958] | 44.3 | .996 | 4.0 | −.004 (1.0) |
| Logistic | A+B+C | .908 [.837, .960] | 42.9 | .995 | 4.3 | −.003 (1.0) |
| LightGBM | A+B+C | .902 [.836, .953] | 56.8 | .996 | 3.3 | – |
| Class rule | class | .784 [.677, .875] | 29.5 | .985 | 5.0 | −.130 (.016) |

## Table 3. CamFormer's errors follow the activity type

Takes not used to pre-train CamFormer, threshold gate with the cost rule at β = 0.05; escalation and errors left are per repeat.

| Class | Takes | CamFormer acc | Escalated (%) | Errors left |
|---|---|---|---|---|
| Dance | 116 | 1.00 | 19 | 0 |
| Basketball | 112 | 0.93 | 21 | 0 |
| Soccer | 13 | 0.95 | 56 | 0 |
| Cooking | 39 | 0.73 | 91 | 0 |
| Health | 35 | 0.74 | 97 | 0 |
| Bike repair | 22 | 0.47 | 91 | 1.0 |
