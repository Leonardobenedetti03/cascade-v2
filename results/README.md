# Results

Everything here was produced by the code in this repository on the Bocconi cluster and committed as it came out.
The SLURM log of each run sits next to its outputs.

## Folders

- `nested_cv/report_n38_cal/`: **the run in the report** (`scripts/03_nested_cv.py` with
  `configs/gate_v2_n38_cal.yaml`, job 773282).
- `nested_cv/reliability_fix_n38_rel/`: the same run without the change in calibration (job 773281). Not in the
  report.
- `nested_cv/preregistered_772601/`: the run we planned in advance, before we found the leak in the class-reliability
  feature. Not in the report.
- `report/`: Figures 2 and 3 and Tables 1, 2 and 3 of the report, made from `nested_cv/report_n38_cal/` by
  `scripts/05_make_report_figures.py` and `scripts/06_make_report_tables.py`, with the numbers behind each figure and
  table as CSV files and the tables as the report rounds them in `tables.md`.
- `features/official_v2/`: the per-take predictions of the CamFormer and EgoVLP heads (`scripts/01_extract_features.py`).
  The report uses them for the labels and to know which takes CamFormer was pre-trained on.
- `features/input_variant_selection/`: how we chose the pose convention and the number of windows (6 candidates,
  validation split).
- `first_evaluation_official_split/`: our first evaluation on the official split, made with the old pipeline. Its code
  is in the tag `archive/development`. Not in the report.
- `first_evaluation_vs_nested_cv.md`: a summary we wrote on 18/11 comparing the first evaluation with the
  pre-registered nested CV run (`preregistered_772601`). Not in the report.

## Files of the report run

Each row of the CSV files is one subset, system, rule and parameter. The main result is the row with
`subset = pretrain_unseen`, `system = thr_maxprob`, `rule = cost`, `param = 0.05`.

| File | What it has | Used in |
|---|---|---|
| `summary.json` | the main result in one place | abstract, 5.2 |
| `models.csv` | accuracy and errors of CamFormer alone and EgoVLP alone | Tab. 1, 3.4, 4 |
| `metrics.csv` | accuracy with 95% CI, balanced accuracy, share sent to EgoVLP, errors left | Tab. 1, Tab. 2, 5.2 |
| `cost_vs_egovlp.csv` | GMAC per take, % of EgoVLP's compute, speed-up | Tab. 1, 5.2 |
| `auc.csv` | AUROC of each gate at spotting CamFormer's errors | Tab. 2 |
| `curves.csv` | accuracy at every share of takes sent to EgoVLP | Tab. 2, Fig. 2 |
| `comparisons.csv` | paired differences between gates, with Holm-corrected p-values | Tab. 2, 5.3 |
| `procedural.csv` | takes predicted as procedural that stay on CamFormer | 5.3 |
| `importance.csv` | gate coefficients and LightGBM split gains, per fold | 5.4 |
| `calibration.csv` | predicted risk against observed error rate | 5.5, Fig. 3 |
| `thresholds.csv` | the threshold of every rule in every fold | 5.5 |
| `takes.csv.gz` | one row per take and repeat: both predictions, gate scores and risks | Tab. 3 |
| `outer_folds.csv`, `heads.csv` | fold of each take; settings of the heads in each fold | 5.1 |
| `cost_sensitivity.csv` | cost with an extra per-window cost added (zero in this run) | not used |
| `config.yaml`, `job_773282.out` | the config that was actually used and the SLURM log | |

The other two runs in `nested_cv/` contain the same files.

To check the numbers of the report against these files: `python scripts/04_check_report_numbers.py`.
