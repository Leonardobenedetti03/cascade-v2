# Look When You Need

Code for our DL4CV project (Bocconi, 2026): *Look When You Need: Camera Trajectories Decide When to Turn On
Pixels*, by Leonardo Benedetti, Giorgio Papa and Jacopo Briseghella.

We classify Ego-Exo4D takes into six scenarios (Basketball, Dance, Soccer, Cooking, Health, Bike repair) with two
frozen models:

- **CamFormer** reads only the camera trajectory. It is very cheap (0.61 GMAC per take) and good on physical
  activities, but it confuses the procedural ones, where the head barely moves.
- **EgoVLP** reads the video frames. It is almost always right, but it costs 462 GMAC per take.

CamFormer runs on every take. A small gate, which never looks at pixels, decides whether to also run EgoVLP. On the
337 takes that CamFormer was not pre-trained on, the simplest gate (a threshold on CamFormer's confidence) gets
0.997 accuracy, against 1.000 for EgoVLP alone, while using 42% of EgoVLP's compute. Adding camera-motion features
to the gate does not help. All results come from a nested cross-validation split by recording session.

## What is where

```
src/cascade/data/        loading takes, labels and camera poses; windows; motion features; session folds
src/cascade/models/      the two frozen encoders, the gates, GMAC counting
src/cascade/training/    the logistic heads on top of the encoders, calibration of the gate's risk
src/cascade/evaluation/  nested cross-validation, decision rules, metrics, compute cost
src/cascade/utils/       config, command line, paths, logging
scripts/                 the entry points, numbered in the order they run
configs/                 YAML configs (gate_v2_n38_cal.yaml is the run in the report)
results/                 the outputs used in the report, see results/README.md
hpc/                     SLURM scripts we used on the Bocconi cluster, see docs/hpc_runbook.md
docs/                    the decision log and the HPC runbook
tests/                   unit tests, CPU only, synthetic data
third_party/             CamFormer (seeing_without_pixels) and EgoVLP, as git submodules
```

## Setup

```bash
git clone --recurse-submodules https://github.com/Leonardobenedetti03/cascade-v2.git
cd cascade-v2
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision        # on a GPU machine, pick the build for your CUDA version
pip install -r env/requirements.txt
python -m pytest -q                  # should give 116 passed, 6 skipped
```

We used Python 3.9. There is also a conda file, `env/environment.yml`.

You only need the data and the checkpoints to recompute the features (step 1 below). Everything after that runs on
the files already in `results/`.

- **Data.** Ego-Exo4D needs a licence (https://ego-exo4d-data.org). We used an 80 GB subset with 1,273 takes, kept
  in two folders. Each folder has `split_manifest.csv`, `annotations/` and the downscaled videos under `takes/`.
  Set their paths in `data.roots` in `configs/features.yaml`.
- **Checkpoints.** CamFormer: `egoexo4d_dur8.pt`, linked from `third_party/seeing_without_pixels/DATA.md`. EgoVLP:
  `EgoVLP_PT_BEST` from the EgoVLP repository. Both models stay frozen.

The scripts find these paths through environment variables:

```bash
export CAMFORMER_REPO=$PWD/third_party/seeing_without_pixels
export CAMFORMER_CKPT=/path/to/egoexo4d_dur8.pt
export EGOVLP_CKPT=/path/to/egovlp.pth
export SCENARIO_RUN_DIR=/path/to/runs/official_v2    # where step 1 writes
export CASCADE_RUN_ROOT=/path/to/runs                # where steps 2 and 3 write
export CASCADE_DATA_ROOT=/path/to/data
```

## Reproducing the report

### 1. Features and heads (GPU)

This step extracts the CamFormer and EgoVLP embeddings for every take and fits a logistic head on each. We give
CamFormer the camera poses as they are in the dataset (identity rotation) and use 5 windows of 8 seconds per take.
We picked these two settings on the validation split out of six candidates; that comparison is in
`results/features/input_variant_selection/`.

```bash
SET="--set poses.json_cam_to_device=[[1,0,0],[0,1,0],[0,0,1]] --set window.n_windows=5"
python scripts/01_extract_features.py index $SET
python scripts/01_extract_features.py traj $SET
python scripts/01_extract_features.py feat-camformer $SET
python scripts/01_extract_features.py feat-egovlp $SET
python scripts/01_extract_features.py heads $SET
python scripts/01_extract_features.py heads --eval-test $SET    # predictions on the official test split
```

On the cluster we ran the same chain with `hpc/features_submit.sh` (see `docs/hpc_runbook.md`). Stages that are
already done are skipped. More details on this step are in [The feature pipeline](#the-feature-pipeline) below.

### 2. Compute cost

```bash
python scripts/02_count_gmac.py --run-name gate_v2
```

This counts the GMACs of one window for each model. We wrote the values from our cluster run into
`configs/gate_v2.yaml` (`gmacs_per_window`) before running step 3, and step 3 reads them from there. If you recount
on a different machine you may see small differences (we saw 0.01% for EgoVLP), which come from the torch and
fvcore versions.

### 3. Nested cross-validation (CPU, about one hour)

This is where every table in the report comes from.

```bash
python scripts/03_nested_cv.py --run-name gate_v2 --scenario-run $SCENARIO_RUN_DIR \
    --tag n38_cal --config configs/gate_v2_n38_cal.yaml --n-jobs 8
```

The output goes to `$CASCADE_RUN_ROOT/gate_v2/56_gate_v2/n38_cal/`. Our copy is in
`results/nested_cv/report_n38_cal/`. On the cluster: `TAG=n38_cal CONFIG=configs/gate_v2_n38_cal.yaml
SCEN_RUN=$SCENARIO_RUN_DIR hpc/nested_cv.sh`.

The output folders are still called `40_cost/` (step 2) and `56_gate_v2/` (step 3), after the stage numbers we used
during development. We left them as they are so that the code is exactly the code that produced the results.

There are three runs of this step in `results/nested_cv/`, and the report uses only the last one:

- `preregistered_772601`: the run we planned in advance. After it finished we found a leak in one gate feature (the
  reliability of the predicted class used the take's own outcome).
- `reliability_fix_n38_rel`: the same run with that feature computed out of fold.
- `report_n38_cal`: the fix, plus the gate's calibration fitted only on takes CamFormer was not pre-trained on.
  **This is the run in the report.**

### 4. Check the numbers in the report

```bash
python scripts/04_check_report_numbers.py
```

This recomputes 162 numbers of the report from `results/` and prints them next to the values in the paper. All of
them match. It needs neither data nor a GPU, so you can run it right after cloning.

### 5. Figures and tables of the report

```bash
python scripts/05_make_report_figures.py
python scripts/06_make_report_tables.py
```

These draw Figures 2 and 3 and build Tables 1, 2 and 3 from the same files, into `results/report/`. Each figure comes
with the numbers behind it as a CSV, and `results/report/tables.md` shows the tables rounded as in the report.
Figure 1 is a diagram of the cascade drawn in LaTeX, so it has no script. Like step 4, this runs in a few seconds
right after cloning.

## The feature pipeline

Step 1 is our version of a package Giorgio wrote for the project (`scenario_preds`). It gives every take of the subset
one scenario prediction from each frozen model, with a linear head trained on top:

- **CamFormer (L0)** reads the trajectory of the RGB camera, taken from the per-frame `camera_pose` JSON files and
  resampled at 20 Hz (`rel9d_grav` encoding). Weights `egoexo4d_dur8.pt`, embedding of size 512.
- **EgoVLP (L1)** reads 4 RGB frames at 224 px from the `*214-1*.mp4` stream. Weights `EgoVLP_PT_BEST`, 768-d CLS.

The head is a StandardScaler plus a multinomial logistic regression with `class_weight: balanced`, on the 6 classes in
the order Basketball, Dance, Soccer, Cooking, Health, Bike repair. Music (0 takes) and Bouldering (2 takes) are left
out of training; the Bouldering takes still get a prediction, flagged in the `note` column.

The stages are `index`, `traj`, `feat-camformer`, `feat-egovlp` and `heads`, plus two diagnostics (`check-up`,
`check-frame`) and `select`. Each one writes a completion marker last. Rerunning a stage that is complete with the
same config does nothing; with a different config it stops until you pass `--overwrite`. Trajectories and features do
not depend on the split, the seeds or the check, so changing those only reruns `index` and `heads`.

**Protocol.** This step uses the official split of Ego-Exo4D. We cut the official val once in half, by session and
stratified by class, into our val (143 takes) and test (142). Every take is predicted by a head that has not seen it:
on train, a head trained on the other folds (5 folds by session) predicts each fold, and this is repeated with 3 fold
partitions whose probabilities are averaged; val and test get the mean of all the fold models. C is chosen by an inner
CV grouped by session (log-loss), and a temperature is fitted on the out-of-sample logits of that same CV. Without
`--eval-test` the test predictions are neither written nor scored; with it they go to a separate `sealed/` folder.

**Pre-registered check.** After the heads, `heads` checks CamFormer (values in `check` of `configs/features.yaml`): on
val, balanced accuracy ≥ 0.40 and the lower end of the session-bootstrap interval above chance; on the out-of-sample
train, recall > 0 for every class; all 6 classes present in both. If the check fails, `heads` still writes its outputs
and `check.json`, then exits with code 3, so the SLURM jobs that depend on it do not start.

**Two things to keep in mind.**

1. *RGB camera poses, not device poses.* CamFormer was pre-trained on the trajectory of the Aria device (x left, y up,
   z forward, gravity along −z). The JSON files give the RGB camera in the OpenCV convention (x right, y down). We
   pre-registered two choices for `poses.json_cam_to_device`, the identity and the paper's conversion
   `diag(-1, -1, 1)`, and picked the identity on val (`select`, see `results/features/input_variant_selection/`).
   `check-up` and `check-frame` are diagnostics for this frame question and were not used to decide.
2. *CamFormer's pre-training saw most of the takes.* The encoder was pre-trained (contrastively, with the narrations,
   without scenario labels) on the takes of the official Ego-Exo4D train. The `camformer_pretrain` column of the index
   says, for each take, whether pre-training used it. This is why the main subset of the report, `pretrain_unseen`,
   is the 337 takes CamFormer was not trained on.

The report uses this step for the features, the labels and the `camformer_pretrain` flag. The heads are refit inside
the nested cross-validation of step 3. The evaluation on the official split itself was our first evaluation, kept in
`results/first_evaluation_official_split/`.

**Outputs** (in `$SCENARIO_RUN_DIR`):

- `predictions.csv`: train and val, one row per take, with split, folds (`fold_s*`), label and, for each model, the
  prediction, its confidence, `conf_seed_sd` (spread across the fold partitions) and the class probabilities;
- `sealed/predictions.csv`: the test, only after `--eval-test`;
- `check.json` (outcome of the check) and `classes.json` (class order, excluded classes, seeds);
- `heads/<model>/`: predictions, metrics with bootstrap intervals, and C, T and accuracy for each fold and seed;
- `index/takes.csv` and `index/summary.json`: the index, and counts of takes and sessions by split and class;
- `traj/status.csv` and `features/*.npz`: the outcome of each trajectory, and the embeddings per take;
- `logs/` and `configs/`: the log and the effective config of each stage.

## Where each part of the report comes from

Result files are in `results/nested_cv/report_n38_cal/` unless another folder is given.

| Report | Code | Results |
|---|---|---|
| 3.1 Cascade | `models/camformer.py`, `models/egovlp.py`, `training/heads.py` | `results/features/official_v2/` |
| 3.3 Gate features and families | `models/gates.py`, `data/motion.py` | `importance.csv` |
| 3.4 From scores to decisions | `training/calibration.py`, `evaluation/rules.py` | `thresholds.csv`, `calibration.csv` |
| 3.5 Cost accounting | `evaluation/cost.py`, `models/gmac/` | `cost_vs_egovlp.csv` |
| 4 Datasets | `data/takes.py`, `data/poses.py`, `data/windows.py` | `results/features/` |
| 5.1 Protocol | `evaluation/nested_cv.py`, `data/folds.py` | `outer_folds.csv` |
| Tab. 1, Fig. 2, 5.2 (RQ1) | `evaluation/metrics.py`, `evaluation/curves.py` | `metrics.csv`, `cost_vs_egovlp.csv`, `curves.csv`; drawn in `results/report/` |
| Tab. 2, 5.3 (RQ2) | `evaluation/metrics.py`, `evaluation/curves.py` | `auc.csv`, `curves.csv`, `comparisons.csv`, `procedural.csv`; table in `results/report/` |
| Tab. 3 | `scripts/06_make_report_tables.py` | `takes.csv.gz`; table in `results/report/` |
| 5.4 Why motion does not help | `models/gates.py` | `importance.csv` |
| 5.5, Fig. 3 Calibration and guarantee | `training/calibration.py`, `evaluation/rules.py` | `calibration.csv`, `thresholds.csv`; figure in `results/report/` |

Code paths are under `src/cascade/` unless they start with `scripts/`.

## Names used in the code

- **L0 / L1**: CamFormer / EgoVLP.
- **`pretrain_unseen`**: the takes not used to pre-train CamFormer (337 takes). This is the main subset in the
  report. `all` is all 1,271 takes.
- **`thr_maxprob`**: the threshold gate on CamFormer's top-class probability. `logreg_A`, `logreg_AC`, `logreg_ABC`
  and `lgbm_ABC` are the learned gates, named after the feature groups they use (A confidence, B class reliability,
  C camera motion). `class_rule` sends every take predicted as procedural to EgoVLP.
- **`rhat` / `rup`**: the calibrated risk that CamFormer is wrong, and its conservative version (r↑ in the report).
- **rule `cost` with param β**: send a take to EgoVLP when `rup` ≥ β. The report uses β = 0.05.
- **`rho`**: the share of takes sent to EgoVLP.

## Development history

Every decision we took along the way is written down in `docs/decision_log.md`. The repository used to also
contain our first pipeline and a first evaluation on the official split. We moved them out of the main branch in
commit C26 without touching the code above. The tag
[`archive/development`](https://github.com/Leonardobenedetti03/cascade-v2/tree/archive/development) points to the
commit just before that, so the full history is still there.
