# Decision log

This is the record of every methodological change and every non-obvious choice we made while rebuilding the
project, written as we went. The "Deviations from the original plan" paragraph of the report is based on it.

How to read it:

- **M01–M20** are the changes with respect to our plan (the "blueprint") and to our first codebase (v1), one row
  each, with the commit that brings them in.
- **N1–N44** are working notes: choices the blueprint did not foresee, or foresaw differently, and the results we
  froze before looking at the next step.
- **C01, C14d, ...** are commit labels. Each commit message starts with its label, so you can find it with
  `git log --grep`. We cite labels and not hashes, because labels survive a rewrite of the history.
- **D1–D8** are decisions taken in the blueprint. **B1–B4, I1–I8, m1–m5** are items of Giorgio's code audit of 15/11.
- Dates are day/month, all in 2026.

The notes describe the code as it was at that time. Much of it (stages 00–55, `hpc/pipeline.sh`, `src/cascade/gate/`,
...) belonged to our first pipeline and is now only in the tag `archive/development`. The code that produced the
report was moved into its current folders in C26, without changing it (N41). The main renames are:

| At the time | Now |
|---|---|
| `scenario_preds/` (Giorgio's feature package), `python -m scenario` | `scripts/01_extract_features.py`, `src/cascade/data/`, `training/heads.py` |
| `scripts/40_cost.py` (stage 40) | `scripts/02_count_gmac.py` (output folder still `40_cost/`) |
| `scripts/56_gate_v2.py` (stage 56) | `scripts/03_nested_cv.py` (output folder still `56_gate_v2/`) |
| `src/cascade/gatev2/` | `models/gates.py`, `training/calibration.py`, `training/nested_heads.py`, `evaluation/{nested_cv,rules,metrics}.py`, `data/{folds,gate_inputs}.py` |
| `hpc/gate_v2.sh`, `scenario_preds/hpc/*` | `hpc/nested_cv.sh`, `hpc/features_*.sh`, `hpc/input_variants.sh` |
| `results/scenario_v1/` | `results/first_evaluation_official_split/` and `results/features/` |
| `results/scenario_cv/`, `results/scenario_cv_n38/` | `results/nested_cv/` |
| `docs/REGISTRO_MODIFICHE.md`, `docs/RUNBOOK_HPC.md` | this file, `docs/hpc_runbook.md` |

## Changes M01–M20

The Commit column was brought up to date on 26/11 (C29); before that, several rows said "planned".

| ID | What changes | Before (v1) | Why | Commit |
|----|--------------|-------------|-----|--------|
| M01 | New repository, code carried over selectively | Unused extensions, launch scripts outside the repository | The code we hand in is the code that produces the results | C01 … |
| M02 | Split by site: test georgiatech, val nus + fair | Official splits by take | L1 at 99% partly recognises the lab; leave-one-site-out 61.7%; criteria fixed before the results | v1 3d5f3b4; C06; stage 27 in C12c |
| M03 | Train folds by site | Folds by take | Out-of-sample predictions must simulate the shift to new sites | C06 |
| M04 | CamFormer from scratch (**replaced by M17**) | Public checkpoint as the default | Pre-training saw about 95% of train and val | C08a |
| M05 | `replace` fusion by default | Log-linear fusion | Δ = 0.0 in v1; ensembles do not improve the front; L1 becomes the expert | C09 |
| M06 | Gates by family, CompareConfidence | More than 15 gates with no reference | RQ2 between families we can cite; the trajectory is the only variable between `compare_A` and `compare_AC` | C11 |
| M07 | Manifests with split and fold keys | A change of folds went unnoticed downstream | Prevents silently reusing outputs computed on different folds | C05 |
| M08 | Pose conversion as stage 01 | External script with no manifest | Reproducibility | C07a, C07b |
| M09 | Balanced accuracy, I(δ), threshold transfer | Only accuracy and metrics at a budget | Class imbalance; Lebovitz's metric and analysis | C12b; C12c |
| M10 | End-to-end cost and sensitivity to the pose cost | Only GMACs and model latency | GMACs do not measure decoding and the camera | C10 |
| M11 | A single launch system, persistent environment | `slurm/` documented but unused; ad hoc scripts | Reproducibility and time | C13a; 4-job chain in C14c-2 |
| M12 | Extensions not carried over: streaming, keystep, L1 fine-tuning, Bayesian gate | Present in the code | Outside the RQs; they stay in v1 | C04, C08b |
| M13 | One-frame baseline run; three levels as a secondary analysis | Code present, never run | Without it the cost comparison is incomplete | planned for C08; the three-level cascade was never carried over (N18, N19), and M18 made the question moot |
| M14 | Six classes reported; Bouldering out of the per-class metrics; Music absent | Eight scenarios declared | Music is not in the subset; 2 takes allow no estimate | C12b; C12c |
| M15 | Epochs and patience of CamFormer (D3) | 30 epochs, patience 5 | Fold 0 and full hit the cap with the loss still going down | C08a |
| M16 | `results/` in the repository | Results only on the HPC and in a backup | Every number in the report has a run and a commit | C18, C23, C25 |
| M17 | CamFormer from the paper's checkpoint, frozen encoder, only the head is trained; EgoVLP unchanged (frozen, logistic head) | v1: checkpoint fine-tuned end to end; blueprint (M04): from scratch | Group decision (16/11): the models were pre-trained on far more data, so we only train the classifiers. Limits stated in N24 | C14d |
| M18 | L0 and L1 from `scenario_preds` (logistic heads on frozen CamFormer and EgoVLP, 6 classes, 3 seeds), gate cache from stage 44 instead of stages 00–45; official split with the val cut in half (val/test) | Stages 00–45 of cascade-v2, split by site (M02/M03) | The L0 check of the `site_v2` run failed (sites never seen by a frozen CamFormer); the split by site stays as a secondary analysis. Details in N28–N31 | C15 (code in the repo in C21) |
| M19 | Extra evaluation with a nested CV by session on all 1,271 takes (stage 56, gate v2): heads, gate, calibrator and thresholds refit inside every fold; cost rules and rules with a risk guarantee; per-session options in the gate v1 | A single val/test split; operating point on 143 takes, at the edge | Giorgio's analysis of 18/11. It adds to the first evaluation, it does not replace it: N33 and N34 stay as the first evaluation. Details in N37 | C22, run 772601, results in C23, N38 |
| M20 | Gate v2, fixes found after run 772601: (1) reliability of the predicted class computed out of fold by session for the fit rows (`reliability_folds`); (2) option `calibration_population` (calibrator and thresholds fitted only on the train takes CamFormer did not see in pre-training); `TAG` and `CONFIG` in the HPC scripts | (1) reliability that left out only the row's session; (2) calibration on the whole train | (1) leave-one-out target encoding, LightGBM learned a reversed rule; (2) risk underestimated on the unseen takes, guarantee that does not hold (N38). Corrective runs `n38_rel` and `n38_cal` next to the pre-registered one. Details in N38–N40 | C24, runs in C25 |

## Working notes

**N1 — C02 was pushed with two failing tests.** C02 was committed with `22 passed, 2 errors`: `test_clips` imported
`synthetic.py`, which only came in C03. CI started with the push of C03, so C01 and C02 have no CI run of their own.
We did not rewrite the published history for this. From C03 on, pytest was green before every commit.

**N2 — What the pose conversion really does (M08).** Read from `hpc/make_traj_data.py` in v1:
- source: for each dataset root, `split_manifest.csv` (columns `take_name`, `camera_pose_file`) points to one JSON per
  take in `annotations/ego_pose/*/camera_pose/`, with `camera_extrinsics` = 3x4 world → camera matrices per frame, at
  about 30 Hz. These are not the 1 kHz device poses. There are two roots (the subset was split across two accounts);
- time is not in the file: frame `i` is at `i / 30` s from the start of the video. So the trajectory lives in video
  time with its origin at frame 0, and `alignment.mode: same_origin` holds by construction (still to be looked at in
  stage 15). The "30 Hz" assumption was hardcoded in v1 and is now `data.pose_src_fps`;
- the camera is chosen to match the video we use (`aria01_214-1.mp4` → `aria01`); gaps longer than 15 source frames
  (0.5 s) get no pose.

Three intended differences from v1, all in `src/cascade/traj_convert.py`:
1. the output is an `.npz` on the full grid with a `valid` mask, instead of a CSV with missing rows. The v1 loader
   filled those rows with the nearest sample, so the first sample of a long gap, 50 ms after the last valid one, came
   out valid or not depending on floating-point rounding (tried on 273 simulated gaps: valid in 79);
2. samples before the first source pose (v1: a copy of the first pose, valid) are `valid = False`;
3. quaternions keep a continuous sign from one sample to the next (same rotation): the raw quaternion flips sign each
   time the rotation crosses 2π.
We compared the conversion sample by sample with the v1 code embedded in the tests (`test_traj_convert.py`).

**N3 — The inventory (00) must run after stage 01, not before.** With the `converted_npz` source, the inventory's
`has_traj` checks that the `.npz` files exist: if 00 runs first, every take looks like it has no trajectory. Stage 01
does not depend on 00 (it reads the list of takes from the split), so the order becomes 01 → 00. Applied in C07b:
`00_inventory.py` requires stage 01 with `converted_npz`, and the order changed in `hpc/pipeline.sh` and in the
synthetic pipeline (in the blueprint 01 comes after 00).

**N4 — Paths of stage 01.** `paths.traj_dir` is `{run}/01_traj` (blueprint: `{data}/traj/{run_name}`): `{run_name}` is
not a placeholder of `paths.resolve`, and the run folder is already persistent. Files are named `<take_name>.npz`
(blueprint: `<take_uid>`), because the split and v1 identify takes by name.

**N5 — To carry into the HPC configs (C13).** The v1 files show how the cluster really ran: `data.duration_source:
video` (no `takes.json`), `alignment.mode: same_origin`, no `expected_split_counts`, and the `--account`,
`--partition` and `--qos` of `hpc/convert.sbatch`. The blueprint's `hpc/submit.sh` passes account and partition but
not the QOS: it has to be added if the cluster needs it.

**N6 — Synthetic data and default source.** `synthetic.py` also writes per-frame poses (`split_manifest.csv`,
`annotations/ego_pose/...`, same format as the real source), so stage 01 runs in the synthetic pipeline and in a test
that compares the inventory from `converted_npz` with the expected files. `data.traj_source` stays `aria_csv` in the
defaults (synthetic data and existing tests use it); `configs/hpc/site_v2.yaml` (C13) sets `converted_npz` and
`alignment.mode: same_origin`.

**N7 — Cluster values, from the v1 sbatch files** (to go in `hpc/env.local.sh`, not in git): same account, partition
`stud` and QOS `stud` for CPU and GPU; GPU requested with `--gpus=1` (v1 did not use `--gres`: check with `sbatch
--test-only` which form the cluster accepts); `module load miniconda3`; data in `~/cascade_data` (`DL4CV_ROOT` and
`CASCADE_DATA_ROOT`), runs in `~/cascade_runs` (`CASCADE_RUN_ROOT`). v1 rebuilt the venv in `/tmp` at every job; v2
uses a persistent environment (M11).

**N8 — Resuming per epoch (D7).** `finetune.train_fold` saves `ckpt_last.pt` at the end of each epoch (atomic write)
with the model, the optimizer, the best state, the patience counter and the history. The key is the hash of the stage
20 config keys plus fold and seed: a checkpoint from another config, or one that cannot be read, is ignored. The stage
deletes it after writing the manifest, and a fold already stopped by early stopping is not extended. Limit: the state
of the random generators (dropout) is not saved, so a resumed run does not reproduce an uninterrupted one bit for bit.

**N9 — fusion.py and heads.py brought in early.** `calibration.py` imports `log_softmax` from `fusion.py`, and
`fusion.py` imports `heads.py` (line 78), so the three files come in unchanged in C08a as dependencies. `heads.py`
comes with its test (`test_heads`); the test of `fusion.py` and the M05 changes stay in C09.

**N10 — Persistent frames and CamFormer from scratch.** `paths.frames_dir` becomes `{data}/frames/{run_name}` (it was
`{run}/10_frames`); `ctx.path` passes `run_name` (a default value that can be overridden). M04 and M15 are applied in
`configs/camformer.yaml`: `pretrained: false`, 50 epochs, patience 8; the dataclass defaults do not change.
`TrajectoryWindowDataset.prefix_frac` and `role_datasets(prefix_frac, roles)` stay as in v1 (they served the causal
variant, removed in M12), to be removed in a clean-up commit.

**N11 — `cache/checks.py` brought in early.** In v1, `check_loss` and `ProvenanceError` live in `cache/build.py`
(stage 45, C12), but stages 25, 28 and 30 (C08b) import them. Instead of bringing in the whole `build.py` early, we
copy the two symbols unchanged into `src/cascade/cache/checks.py` (with `tests/test_checks.py`); in C12 `build.py`
imports them from there, with no duplicate copies.

**N12 — `eval/style.py` brought in early.** Stage 15 (alignment figure) imports `GRID`, `SERIES`, `SURFACE` and `TEXT2`
from `eval/report.py` (C13). The four constants (the v1 colours) live in `src/cascade/eval/style.py`, and in C13
`report.py` imports them from there. The other palette constants (`TEXT`, `REF_GRAY`, `SEQ_BLUE`) stay in `report.py`.

**N13 — Models and stages of L1 and of the baseline (C08b).** Carried over from v1: `models/egovlp.py`,
`models/frame_baseline.py`, `models/vendor/egovlp_video_transformer.py` (a modified copy of the submodule
`third_party/EgoVLP`, 85 lines of difference, to be kept identical to v1), `tests/test_egovlp.py` and stages 15, 25,
28 and 30. Only the imports change: `check_loss` and `ProvenanceError` from `cache/checks.py` (N11), the colour
constants from `eval/style.py` (N12). We do not carry over `models/egovlp_finetune.py` (stage 26, M12). The synthetic
pipeline runs 15 after 10, and 25, 28 and 30 after 20; the one-frame baseline runs because `frame_baseline.enabled` is
`true` in `configs/synthetic.yaml`.

**N14 — `replace` fusion (C09, M05), and stage 35 moved to C12.** In `fusion.py`, `replace` is the default
(`choose_w` and `FusionCfg.mode`); `fuse` stays as an ablation. We removed the learned fusion: `fusion_features`,
`learned_fusion`, the `learned` parameter of `compare_fusions`, the fields `learned_features` and `compare_learned`, the
value `learned` of `fusion.mode` and the tag `_flearn` of `variant_tag`. Stage 35 imports `assemble_camformer`,
`join_l0_l1` and `load_l1` from `cache/build.py` (C12), so it comes in C12 together with `build.py`, already without the
learned fusion, instead of bringing `build.py` in early. `fusion.py` no longer imports `heads.py`, which stays for
stages 28 and 30 (N9). The v1 `test_label_y` uses `cache/schema.py` and goes with `test_cache_schema` in C12.

**N15 (C10, M10).** Stage 40 (GMACs with three counters) copied from v1 unchanged. Stage 41 is new: time of each phase
at batch size 1, for L0 (reading the trajectory, preprocessing, forward pass) and for L1 (opening and decoding,
preprocessing, forward pass), on a sample stratified by class (`cost_e2e.n_clips`), with warm-up clips discarded, GPU
synchronisation, median and p90, two passes (we report the second and state the first). Random weights with the real
architecture: time does not depend on the values. The blueprint does not fix these details. Our choice: the L0 time
includes reading the trajectory file, with no cache. Break-even threshold for the trajectory:
C_pose < (1 - r) * C_L1 - C_L0 - C_g, in seconds and (if stage 40 is there) in GMACs.

**N16 (C11, M06).** Gate carried over from v1 (the `gate/` folder was empty in v2) with the blueprint's changes.
- `models.py`: added `CompareGate` (a subclass of `LogRegGate`, `class_weight=None`, `target = "c1"`, score
  `η̂₁ − p_max`; the first group must be A) and the `compare_` branch in `make_gate`; removed `RidgeGate`,
  `LGBMRegGate` and the Bayesian branch. `make_gate(name, gcfg)` loses the `labels` parameter, which only the Bayesian
  gate needed; stage 50 (C12) calls it that way.
- `features.py`: removed the `mc` and `take` groups and the Beta-Binomial reliability (`FeatureBuilder` loses
  `reliability_mode`, `beta_n0`, `beta_q`, `prior_confusion`); A, B, C, `onehot` and `emb` remain (the blueprint only
  removes `mc` and `take`, so `onehot` stays).
- `protocol.py`: `run_gate` picks the target column from `gate.target` with `TARGET_COLUMN = {"hard": "y", "c1":
  "c1"}`; removed the `soft` target, level −1 (`level_minus1_forced`) and the `needs_class` branch. `gate/bayes.py` is
  not carried over.
- `GateCfg` loses `reliability`, `beta_n0`, `beta_q`, `prior_confusion` and `ridge_alphas`: `reliability: beta` needed
  `bayes.py`, so only the empirical reliability remains (our choice; the blueprint does not fix it).
- `motion.py` (group C) was not in v2 yet: copied from v1 unchanged.
- `configs/gate.yaml` follows the families of blueprint section 5.2: models `always_L0, always_L1, random, oracle,
  thr_maxprob, thr_margin, thr_entropy, logreg_A, logreg_AC, logreg_ABC, lgbm_ABC, compare_A, compare_AC`; comparisons
  `[compare_A, thr_maxprob]` and `[compare_AC, compare_A]` (the two RQ2 pairs) plus `[logreg_AC, logreg_A]` (our choice:
  the same effect of the trajectory in the family trained on y).
- Tests: `test_gate` rewritten without `cache/mock.py` (C12), with a fake cache built inside the test and scikit-learn's
  `roc_auc_score` instead of `src.eval.metrics.roc_auc`; `test_compare_gate` is new, with the blueprint's synthetic test
  (an L1 that succeeds where the speed is high); `test_level_minus1` and the Bayesian tests are not carried over.
  `tests/test_motion.py` carried over from v1 unchanged.
- The D1/D2 diagnostics (AUROC of η̂₁ against c1, table `diag_c1.csv`) go into stage 50 (C12).
- On macOS `lightgbm` needs `libomp` (`brew install libomp`).

**N17 (C12a, M05).** Cache and fusion carried over from v1: `cache/build.py`, `cache/schema.py`, `cache/mock.py` and
stages 35 and 45.
- `build.py` imports `check_loss` and `ProvenanceError` from `cache/checks.py` (N11) and re-exports them (the tests also
  import them from there); removed the MC dropout columns and vectors (`CAM_VECTORS`, `MC_VECTORS`, `MC_SCALARS`),
  because the gate's `mc` group no longer exists (N16).
- `schema.py`: removed the `fusion_mode == "learned"` branch (a NaN `w` is now an error), the optional MC columns and
  `take_duration`. `mock.py`: removed the `mc` option; with `choose_w` in `replace` mode the fake fusion equals L1.
- Stage 35: removed `learned_fusion`, `fused_learned.parquet` and the `_l1ft` tag; output in `35_fusion/seed<S>_hz<F>/`.
  Stage 45: removed the learned-fusion branch and the `take_duration` column (it was for the error analysis, which we
  do not carry over).
- `manifest.py`: the keys of stage 35 add `camformer.eval_source`, `gate.bootstrap` and `gate.bootstrap_seed`; those of
  stage 45 add `data.window_s`, `data.min_traj_coverage`, `data.traj_fps`, `camformer.eval_source`, `fusion` and
  `frame_baseline` (before, a change of fusion or baseline did not invalidate the cache; our choice, the blueprint does
  not list the keys).
- Tests: `test_cache_build` (with `eval_source=full` and `load_l1`), `test_cache_schema` with `test_label_y` rewritten
  (we had not read the original v1 body) and `test_mock_fusion_is_replace`. There is no pytest integration test for 35
  and 45: `scripts/run_synthetic_pipeline.sh` runs them, after 30 and after 41.

**N18 — C12b: evaluation of the gates (stages 50 and 52).** Choices made while writing M09 and M14, where the blueprint
and the code diverged or said nothing:
- `gate.within_pts` stays a scalar (0.01) and means the lowest cost "within one point of always_L1"; the δ values of
  I(δ) are in `gate.improvement_deltas` = [0, 0.01]. The blueprint listed the δ inside `within_pts`. New `GateCfg`
  fields: `bacc_min_takes` = 5 and `transfer_budgets` = [0.25, 0.5].
- No forced decisions of level −1: the v1 `forced` vector no longer exists (level not carried over, M12/M13).
- I(δ) = C_L1 / C(r*_δ), with the numerator read from the curve (`cost[r = 100] − cost[r = 0]`). With `replace` the
  minimum is C_L1 / (C_L0 + C_L1) ≈ 0.9987: below 1 the cascade saves nothing.
- Balanced accuracy (`bacc_*`) = mean over classes of the per-take recall, using only the classes with at least 5 takes
  in the split (Bouldering stays out, M14). If no class reaches 5, the metric is NaN, not an error. The reference gate
  for the per-class cascade is the one with the highest `bacc_area` on val.
- Threshold transfer (RQ1, only with `--final`): the threshold chosen on val for a cost budget
  (C_L0 + r·C_L1 ≤ b·(C_L0 + C_L1), so at b = 0.5 the share is just below 50%) is applied to test, with the cost overrun
  and the accuracy gap with respect to the threshold recomputed on test, which is only a reference. Fixed-r gates and
  the oracle are skipped.
- Diagnostic D1: AUROC of η̂₁ = score + p_max against c1 for `compare_A` and `compare_AC`, on out-of-sample train and
  on val, never on test (the function raises an error). Cells A/B/D/N per split and class in `cells.csv`.
- `--final` on stage 50 always reruns (it does not check the manifest) and writes `FINAL_DONE` and a line in
  `FINAL_RUNS.log`; without `--final` the stage refuses to run if `FINAL_DONE` is in the folder.
- Stage 52 also aggregates `threshold_transfer` and `diag_c1`; the `_l1ft` suffix goes away (L1 fine-tuning not carried
  over).
- Synthetic pipeline: stage 50 runs with `gate.bacc_min_takes=1`, because with 2 takes per class in val and test a
  threshold of 5 would leave bacc always NaN and that code path untested.
- Multiplications by the bootstrap weights use `einsum` and not `@`: with numpy 2.0 and Accelerate (macOS arm64) `@`
  gives spurious warnings ("divide by zero / overflow / invalid value in matmul") on finite numbers. We checked that
  the results match `einsum` (difference 0 on counts, ~1e-11 on costs).
- Not carried over from v1: `three_level`, `error_analysis`, `_cohen_d`, the `bayes_*` gate and `src.eval.metrics`
  (AUROC comes from scikit-learn).
- On the Mac (numpy 2.0.2 with Accelerate; Python 3.9 does not allow newer numpy) the pipelines print hundreds of
  `RuntimeWarning: ... encountered in matmul` from `gate/features.py` and from scikit-learn's logistic regression. They
  are spurious: `@` and `einsum` give the same numbers and every value is finite. We do not filter them, so as not to
  hide real warnings; on Linux with OpenBLAS they should not appear (to check on the HPC in C13).

The C12b commit entered rows M09 and M14 of the table together with C12c.

**N19 — C12c: site diagnostics (stage 27; M02, M09, M14).** Choices made where the blueprint (12.6) and the brief
diverged.
- The blueprint's three analyses go to three files: leave-one-site-out per site and per class in `loso.csv` (a), how
  readable the site is from the CLS tokens in `site_probe.json` (b; this is new: in v1 `check_leak.py` predicts the
  class, not the site), and the baseline "most frequent class of the site" in `site_majority.json` (c).
  `site_probe.json` also has the check of `check_leak.py` (CV of the class by take and by session, `label_cv`): the
  jump with respect to leave-one-site-out measures the site shortcut.
- Scope: train and val, never test. The stage runs outside `50 --final`, like 25 and 30. v1 used all takes, so the
  numbers are not comparable: the 61.7% cited in M02 stays the v1 number, with that scope.
- The site is the prefix of the `take_uid`; the session is the `take_uid` without its last suffix (as in
  `check_leak.py`). Probe: StandardScaler + LogisticRegression with C = 0.1 and max_iter = 500 (`site_diag.C`,
  `site_diag.max_iter`), CV with `site_diag.cv_folds` = 5. Convergence warnings are not hidden: they are counted in
  `fits` of `site_probe.json`.
- In leave-one-site-out we evaluate only the clips whose class is in the train of that round (as in
  `check_site.py`); the excluded ones are counted. `reported` follows `gate.bacc_min_takes` (M14: Bouldering stays out
  of the per-class means).
- The roles for the baseline come from `load_scenario` (indices of `labels.yaml`) and not from `roles_from_scenario`,
  which uses the CSV's `label` column, in a different order: we need the class names. `site_majority_accuracy` is
  unchanged: in-sample on train and on all takes of the split, including those without video.
- Stage `27_site` is in `STAGE_KEYS` (`site_diag`, `gate.bacc_min_takes`) and among the stages that depend on the split
  (M07). No per-seed variant: it uses the CLS tokens of stage 25 and does not depend on CamFormer. `hpc/submit.sh`,
  `hpc/pipeline.sh` and `hpc/collect_results.sh` already expected it.
- Synthetic data: every `take_uid` starts with `syn`, so there is a single site. The probe is skipped (`skipped` in the
  JSON, `loso.csv` with only the header), and the stage still writes the three files; the logic with several sites is
  tested in `tests/test_site_diag.py` on hand-built arrays.
- Open, not touched here: M13 still says "planned C08", but `three_level` was not carried over (N18), so the
  three-level cascade is not evaluated in stage 50; the blueprint (11.2) has `transfer_budgets` from 0.1 to 0.9, the
  config [0.25, 0.5].
- The C12c commit enters M02, M09 and M14 with the next commit (C13), as for the others.

**N20 — C13a: launching on the HPC.**
- Tag `results-v1` of the v1 repository: it points to `a06691b`, the commit that produced the 49 manifests, as the
  blueprint says. `a5f66f2`, seen with `rev-parse --short`, is the annotated tag object, not a commit. No discrepancy;
  the v1 launch scripts are in `8a4d9c2`, which is more recent.
- `hpc/env.sh` has `SLURM_QOS`, and `hpc/submit.sh` passes `--qos` when it is set. Personal values (account, partition,
  QOS) are in `hpc/env.local.sh`, outside the repository.
- The stage table of `hpc/submit.sh` is a function with `case` and not an associative array: same values, but it also
  runs with the bash 3.2 of the Mac, so `tests/test_hpc_dryrun.py` tests it locally as well as in CI.
- `configs/hpc/site_v2.yaml` holds only what makes the run different: its name, `data.traj_source: converted_npz` (the
  blueprint, 11.2, puts it in `base.yaml`, which has `aria_csv` in the code because synthetic data uses it; our choice:
  `base.yaml` unchanged), the one-frame baseline turned on (M13) and zero-shot turned off (D8: no text model to
  download on the nodes). A test checks that these are the only keys that differ from `base.yaml`.
- `hpc/check_cluster.sh` and `src/cascade/hpc_check.py`: checks on the login node (settings, imports of `torch` and
  `transformers`, data and weight paths, split hash, `sbatch --test-only`). `transformers>=4.30` is in
  `env/requirements.txt` and `hpc/setup_env.sh` prints its version; whether it imports on the cluster can only be seen
  there.
- The HPC runbook, cited by `hpc/env.sh`, now exists.
- Open: `id_v2` cannot be launched (`splits/id_v1/` has only `.gitkeep`; the H4 comparison stays out of C13); the
  `TODO(HPC)` items of `base.yaml` (path patterns, `pose_roots`, `alignment.mode`, `vertical_axis`) are closed with the
  `dbg_smoke` trial and stage 15; `tests/test_hpc_dryrun.py` exempts `55_report.py` until it exists (C13b).
- The C12c commit is now in M02, M09 and M14; C13a enters M11 with the next commit.

**N21 — C14a: blocking fixes from Giorgio's audit (15/11).** Each one checked on the code before fixing it.
- B1: with `data.traj_source: converted_npz` (the `site_v2` run) the inventory looked for the trajectory in
  `paths.traj_pattern`, i.e. the Aria CSV, and did not find the stage 01 `.npz` files: every take would have ended up
  in `missing_takes.csv`. Now `clips.resolve_take_paths` uses `<paths.traj_dir>/<take_name>.npz`; `traj_pattern` stays
  for `aria_csv`. The M08 test set the `traj_pattern` override by hand and so hid the bug: we removed it, and the test
  now fails with the old code and passes with the new one. `data.traj_source` and `paths.traj_dir` enter the manifest
  keys of stage 00.
- B2: in `hpc/pipeline.sh` stages 15 and 41 started after stage 01, but they read the `data/clips.csv` of stage 00: now
  they depend on `$j00`, and 41 also on `$j40`, because it writes `breakeven_pose_gmacs` only if
  `40_cost/<hz>/cost.json` already exists (found by Giorgio). `tests/test_hpc_dryrun.py` updated.
- B3: an empty `paths.pose_roots` was not checked before the run. `src/cascade/hpc_check.py` (and so
  `hpc/check_cluster.sh`), with `converted_npz`, reports FAILED if the list is empty or a root has no
  `split_manifest.csv`. A missing `takes.json` stays a warning, as agreed: v1 ran without it on the cluster (N5) and
  the duration falls back to the video. It is an error only with `alignment.mode: device_us`, where the µs start times
  live only there (before, `hpc_check` always reported FAILED and stage 00 always a warning). `pose_roots` in
  `base.yaml` stays a `TODO(HPC)`.
- I6: `cost.expected.camformer` from 0.02 to 0.12 GMACs. With the `base.yaml` architecture and the 8 s context at
  20 Hz (T = 160 plus the gravity token) the analytic formula gives 0.1216 GMACs, the same value as
  `cost.override_gmacs.L0` in the synthetic config; with 0.02 the sanity check of stage 40 would have flagged a 6× gap
  on a correct count. A test ties the expected value to the formula.
- Source: Giorgio's PDF of 15/11. His `correzioni_provate_v2.patch` was not available, so the changes were written
  from scratch, to be compared with the patch. B4 (stage 55) has no temporary fix: it comes with C13b, before the real
  run.
- Left for C14b: I1, I2, I3, I4, I7, m1, m3, m4, m5.

**N22 — C14b, first part: I1, I2, I4 and the test of the real config (I8).** From Giorgio's
`correzioni_provate_v2.patch`, applied where it did not already match C14a; every new test fails with the code from
before.
- I1: in `heads.fit_head` the inner folds were counted on clips and not on takes, and Bouldering (one take in the fit of
  folds 1–4 of `site_v1`) brought `n_splits` down to 1: CV skipped and C = 0.001 fixed, with no warning. Now classes
  with a single take stay out of the inner CV and inside the final fit; if no class has two takes we use `Cs[0]` with a
  warning. The same applies to the L½ head. Test: `tests/test_heads_rare_class.py` (folds from the repository, C
  unchanged when the rare take is added, aligned columns, rare class learned).
- I2: the keys of stage 00 (`CLIP_KEYS`, now with `paths.split_sha256`) pass to stages 10, 20, 25, 27, 28, 30, 35, 41,
  45 and 50, like `SPLIT_KEYS` in M07; stage 41 has its own keys. Rerunning stage 00 with another clip definition gives
  `ConfigMismatch` instead of reusing old outputs. Stage 10 still skips the `.npy` files already there: after a change
  of windows you need `--overwrite`.
- I4: stage 50 evaluates L½ when `half_probs_cal` is in the cache (`per_class.csv`, `confusion.csv`). The three-level
  cascade is still not carried over (open decision on M13).
- I8: `tests/test_site_v2_chain.py` runs 01 → 00 → 05 → 20 on synthetic data with `configs/hpc/site_v2.yaml` (folds by
  take, because synthetic data has a single site). `tests/test_hpc_dryrun.py` derives the dependencies of
  `pipeline.sh` from what each script reads: this is the test that found the 41 → 40 dependency.
- Left for the second part of C14b: I3 (bootstrap and comparisons on train, metric per fold), I7, m1, m3, m4, m5.

**N23 — C14c, first part: cluster values and real paths (read-only survey of 15/11).** Checked on the login node with
`sacctmgr`, `sinfo`, `scontrol`, `sbatch --test-only` (nothing queued) and `find`; v1 (`hpc/*.sbatch`,
`hpc/make_traj_data.py`) showed where to look.
- Cluster: account 3185472, partition and QOS `stud`. The GPUs are MIG partitions (`3g.40gb`, `4g.40gb`, `7g.80gb`):
  `--gres=gpu:1` is accepted and stays the default (v1 used `--gpus=1`, also accepted). System Python 3.9.25, network
  from the login node to PyPI, PyTorch and GitHub: `MODULES` empty, torch `cu121`. The values are in `hpc/env.local.sh`
  (not in git) and, for the record, in the HPC runbook.
- `paths.pose_roots`: the two halves of the subset, `/home/3185472/egoexo4d_split_45G_physical` (992 takes) and
  `/home/3218863/Oasis_Deep_Vision_DSBA/egoexo4d_split_45G` (281), each with `split_manifest.csv` and
  `annotations/ego_pose/{train,val}`. 992 + 281 = 1,273 = 1,138 + 68 + 67, and the list matches
  `splits/site_v1/scenario_site.csv`. The two manifests have different columns; stage 01 reads only `take_name` and
  `camera_pose_file`, which both have. The roots go in `configs/hpc/site_v2.yaml` and not in `base.yaml` (synthetic
  data has its own); `tests/test_site_v2_chain.py` puts them back for synthetic data.
- `takes_root`: the blueprint assumes a single folder with all the takes, the subset has two. Choice: reuse the tree
  of links built in v1 (`/home/3185472/cascade_data/egoexo4d/takes`, 1,273 takes, 0 broken links, every
  `frame_aligned_videos` points to the real root), linked as `$DL4CV_ROOT/data/egoexo4d/takes`. The folders also hold
  the v1 `closed_loop_trajectory.csv` files, which `site_v2` (`converted_npz`) does not read. Alternative set aside for
  now: a script in the repository that rebuilds the links from the manifests (same result, more work).
- `video_pattern`: the stream prefix varies (`aria01` 878, `aria02` 102, `aria` 10, `aria1` 2 takes in the first half);
  each take has a single `*214-1*.mp4`, which `video_glob` finds. Unchanged.
- There is no `takes.json`: `data.duration_source: video` in `site_v2.yaml`, explicit instead of the silent fallback
  (as in v1, N5). With `alignment.mode: device_us` it would still be an error of `hpc_check` (B3).
- Weights: EgoVLP from the v1 file (a link), MobileNetV3-Small downloaded from the login node (hash `047dcff4`
  checked).
- GitHub key: a read-write deploy key of `cascade-v2`, only on the cluster. Code is changed from the Mac.
- Open (C14c, second part): the `stud` QOS allows 4 queued jobs and 2 running per user, and every element of an array
  counts. `hpc/pipeline.sh` submits about 20, plus the arrays of stages 10 and 20: SLURM would reject the fifth
  `sbatch`. It has to be reorganised into a few jobs with several stages in sequence, as v1 did (`chain.sbatch`).
- `TODO(HPC)` items still to close with stage 15: `alignment.mode`, `data.vertical_axis`, `frames.rotate_k90`.

**N24 — C14d: CamFormer pre-trained and frozen (M17, replaces M04).** Group decision of 16/11: for both levels we use
the published weights and train only the classifier. The blueprint said the opposite (page 11, "Training. From
scratch (M04)", and the log table on page 45): the choice we took is M17.
- EgoVLP does not change: it was already frozen (EgoVLP_PT_BEST, stage 25) with the logistic regression of stage 30.
- CamFormer: `configs/camformer.yaml` moves to `pretrained: true` and the new `freeze_encoder: true`. With
  `freeze_encoder` (`finetune.make_optimizer`) the encoder weights do not require gradients, the optimizer gets only
  the linear head (lr × `head_lr_mult`), and during training the encoder stays in `eval` mode (no dropout). Without
  it, the behaviour is the one from before (end-to-end fine-tuning). The config refuses `freeze_encoder` without
  `pretrained`; `synthetic.yaml` keeps random weights with `freeze_encoder: false`. The key enters the manifest of stage
  20 with the `camformer` block (already included) and `summary.json`.
- Checkpoint: `egoexo4d_dur8.pt` from `camformer_checkpoints.zip` (the authors' Drive,
  `third_party/seeing_without_pixels/DATA.md`), in `paths.camformer_ckpt`. It is the one the authors point to for
  Ego-Exo4D: `pose_encoding` `rel9d_grav` and an 8 s context, the same as `camformer.arch` and `data.window_s` (no
  config change). `hpc_check` requires it (with `pretrained` true it already checked it).
- Limits to state in the report: (1) pre-training saw about 95% of the train and val takes (the reason for M04);
  whether it also saw the test has to be checked on the authors' split. With takes already seen, the confidence of L0
  on val can be too high, and the gate thresholds are calibrated on val. (2) Our poses are those of the RGB camera
  converted to 20 Hz (N2), not the 1 kHz device poses of pre-training: with a frozen encoder the difference matters
  (the blueprint considered it irrelevant only with CamFormer trained from scratch). Check fixed before the results:
  after stage 20, the balanced accuracy of L0 on val must be clearly above that of the majority class on every seed;
  otherwise we stop before looking at the test. (3) The three seeds change only the initialisation of the head and the
  order of the batches, so the variance across seeds will be small.
- Tests: `test_freeze_encoder_trains_only_the_head`, `test_without_freeze_the_encoder_is_finetuned`,
  `test_freeze_needs_pretrained_weights` (`tests/test_camformer.py`); `test_hpc_check` updated (the CamFormer checkpoint
  is now needed); `test_resume` uses `freeze_encoder=false` (the toy model has no separate encoder).

**N25 — C14e: environment on Giorgio's account and loading the real EgoVLP checkpoint (16/11).** The run goes from
Giorgio's HPC account (3195149): Leo's home is almost full (about 47 GB out of 50) and space is counted per owner. The
dataset stays where it is and is read from Leo's and Jacopo's homes; `egovlp.pth` was made readable by everyone
(`chmod go+r`; the file itself was not touched). On Giorgio's account: a clone with its own deploy key ("HPC
giorgio", read-write), `hpc/env.local.sh` with `SLURM_ACCOUNT=3195149` and `DL4CV_ROOT=$HOME/cascade_v2_root`, an
environment in `~/venvs/cascade-v2` (5.6 GB). With the `cu121` index pip installs torch 2.5.1 and torchvision 0.20.1,
not the 2.8 of the Mac; the test suite passes all the same. `check_cluster.sh site_v2`: all fine, including the three
weight files and `sbatch --test-only`.
- The only failing test: `test_egovlp.py::test_real_checkpoint_loads`, `ModuleNotFoundError: No module named
  'parse_config'`. The EgoVLP_PT_BEST checkpoint also stores the training config as a `parse_config.ConfigParser`
  object from the EgoVLP repository. `load_state` now uses an Unpickler that replaces the classes it cannot import with
  a placeholder that is never used (we only need `state_dict`). We set aside the alternative of putting
  `third_party/EgoVLP` on the path, which would import their training code. Test:
  `test_load_state_ignores_classes_missing_from_the_path`, which reproduces the error.
- The CamFormer checkpoint `egoexo4d_dur8.pt` (downloaded from the authors' Drive and copied with `scp`) loads:
  `test_camformer.py::test_real_checkpoint_loads` passes on the cluster.

**N26 — C14f: `dbg_smoke` trial on the cluster, the three stage-15 TODO(HPC) items closed (16/11).** Run `dbg_smoke`
with `site_v2` from Giorgio's account, at commit C14e. Stage 01 (job 768476, 3 min): 1,273 takes converted out of
1,273, 2 with rows before the first pose, 52 with gaps. Stage 00 (job 768529, 3 min): 1,273 clips, all usable, 3
takes with problems flagged (to look at before the real run, not blocking). Stage 15 (job 768535, 2 min): 6 val takes
(nus_covidtest_36_2, nus_cooking_10_4, nus_covidtest_01_2, fair_cooking_06_4, nus_cooking_17_5, nus_covidtest_32_3).
- `frames.rotate_k90 = 0`: in all 48 thumbnails the frames are upright (table and hands at the bottom, ceiling at the
  top).
- `data.vertical_axis = 2`: the height stays within about ±0.25 m of the initial pose, even in the cooking takes where
  the person moves by metres; a horizontal axis would give ranges of metres. The dips match the frames where the
  person bends down (sink in nus_cooking_10_4 between 50 and 95 s, table in nus_covidtest_32_3 between 47 and 63 s).
- `alignment.mode = same_origin`, `offset_s = 0`: in all six takes the trajectory covers the video from start to end,
  with no empty stretch at either end (a shift of seconds would show there), and the dips fall on the matching frames.
  Limit: with 8 frames per take the check rules out shifts of a few seconds, not of fractions of a second; compared
  with an 8 s window the remaining effect is small. `device_us` is not needed.
- Blueprint and reality: the RUNBOOK sequence "00 --inspect, then 15" does not work, because `--inspect` does not
  write the stage 00 manifest and stage 15 requires it (`FileNotFoundError` in job 768518). Choice: no code change
  (that is how `--inspect` is meant to behave); the RUNBOOK now lists 01, the full 00, then 15.
- Only comments and documentation: the config values do not change (configs load identically before and after).

**N27 — C14c-2: the run as a chain of 4 jobs, "road A" for stages 50–55, short takes (16/11).** This closes the open
point of N23: the `stud` QOS allows 4 queued jobs per user and counts every element of an array, and the blueprint's
`hpc/pipeline.sh` (17.4: about 25 submissions, plus the arrays of stage 10 and stage 20) would have had its fifth
`sbatch` rejected. Blueprint and reality do not match. Our choice: four jobs, each with several stages in sequence,
like the v1 `chain.sbatch`.
- `hpc/submit.sh --chain <name> "<stage> [arguments]; ..."` submits a job that runs the stages in order
  (`STAGE_LIST`). Resources: a GPU node if at least one stage needs a GPU, the maximum CPU and memory, time = sum of the
  times in the table (×6 for the folds of stage 20, ×`NUM_SHARDS` for stage 10), at most 24 hours. The stage table
  moves from `submit.sh` to the new `hpc/stages.sh`, which `stage.sbatch` also reads. `submit.sh <stage>` stays as it
  was, for single reruns.
- `hpc/stage.sbatch` in chain mode: the folds of stage 20 (0–4 and `full`) run one after the other; stage 10 runs once
  on all clips, without shards (manifest `shard000of001`, which stage 25 accepts). A stage marked `?` (stage 15) is a
  diagnostic: if it fails the chain goes on, as in the blueprint where 15 "blocks nothing"; a missing script marked `?`
  is skipped. At the first error of a normal stage the job exits with that stage's code, and `--kill-on-invalid-dep`
  cancels the downstream jobs. Requeue at the time limit stays: the job restarts from the first stage and the manifests
  skip the complete ones.
- `hpc/pipeline.sh`: A (CPU) 01 → 00 → 05 → 10 → 15?; B1 (GPU) 25 → 28 → 40 → 41 → 20 for the first seed; B2 (GPU) 20
  for the other seeds, in parallel with B1 (8 + 4 CPUs and 2 GPUs: within the 12 CPUs and 2 GPUs of the QOS); C (CPU)
  27 → 30 → 35 and 45 per seed → 50 per seed → 52 → 55. With a single seed, B2 and 52 are not there; 55 is included
  only if the script exists (C13b). Estimated times with `NUM_SHARDS=8`: A 10 h, B1 8 h 30, B2 12 h, C about 6 h 30.
  These are the table estimates, to be corrected with `sacct` after the run.
- `hpc/final.sh`: a single job (50 `--final` per seed, 52, 55) instead of five, which with three seeds went over the
  QOS. With `NO_REQUEUE=1` (`--no-requeue` and `MAX_REQUEUE=0`): an automatic rerun would evaluate the test a second
  time.
- Road A (decision of 16/11): the real run starts before C14b-2 (I3, I7, m1–m5) and C13b (stage 55), which mostly
  change stages 50–55. These are redone afterwards, at the new commit, with `hpc/pipeline.sh --redo-gate` (one job: 50
  per seed and 52 with `--overwrite`, then 55), without redoing 00–45. The blueprint asks for a single commit in the
  00–52 manifests (principle 8 and `verify_run`): `verify_run.sh` now accepts one commit for 00–45 and one, the same as
  the clone, for 50–52, only if the first precedes the second (`git merge-base --is-ancestor`) and no `scripts/` file of
  stages 00–45 changes between them (otherwise FAILED). It lists the differences in `src/` and `configs/` to check by
  hand, because the same module can serve both upstream and the gate. Changed config keys are already stopped by the
  manifest (`ConfigMismatch`). Limit to state in the report: the results cite two commits. If C14b-2 touches the code of
  stages 00–45 (for instance I7, "reduced cache", in stage 45), road A is not enough and we redo from that stage on.
- Tests (`tests/test_hpc_dryrun.py`, 28 instead of 13): the chain dependencies are still derived from what each script
  reads, now taking into account the order inside the job; 4 jobs within the QOS limits; content and resources of each
  job; `--redo-gate`; `--chain` with unknown stages; `NO_REQUEUE`. `stage.sbatch` now runs in the tests, with a fake
  `python`: order of the stages, six folds, stage 10 without shards, a diagnostic stage that fails, stop at the first
  error, missing script skipped, single array task, wrong commit (exit 3). `verify_run.sh` on fake run folders: one
  commit, road A accepted, upstream script changed, 50–52 not at the clone's commit, mixed commits upstream. The tests
  that run `stage.sbatch` and `verify_run.sh` need `sha256sum` and are skipped otherwise. Tried in the sandbox (Python
  3.9, bash 5); 17 of the new tests fail with the scripts from before.
- The 3 "takes with problems" of stage 00 in `dbg_smoke` (N26) are `short_take`, all in train and all Basketball:
  sfu_basketball_08_40 (3.6 s), sfu_basketball_11_40 (7.1 s), sfu_basketball_11_44 (6.6 s). With `data.short_take: keep`
  the clip is the whole take. The downstream stages handle them: stage 10 takes the frames inside the real window, the
  trajectory coverage (stages 20 and 45, `slice_window`) is computed on the length of the window and not on 8 s, and
  CamFormer takes sequences of variable length with the padding mask. The blueprint does not cover the case. Choice:
  keep them (`keep`, no change). With the frozen encoder (M17) they only enter the training of the linear head: 3
  examples with a shorter context, none in val or test. Alternative set aside: `drop`, i.e. training only on full
  windows, like the pre-training of `egoexo4d_dur8.pt`.

**N28 — C15: gate on the `scenario_preds` predictions, bridge stage 44, 6 classes (17/11).** Group decision of 17/11
(afternoon): L0 and L1 come from Giorgio's `scenario_preds` package (M18); what stays from `cascade-v2` is the cost
(40), the gate and the evaluation (50).
- `configs/labels.yaml` with 6 classes, in the order of the `scenario_preds` heads: Basketball, Dance, Soccer, Cooking,
  Health, Bike repair. Music (0 takes) and Bouldering (2 takes) out (as in M14). `mock.py` with 6 classes. Stage 00 with
  CamFormer's 8-scenario CSV is no longer used: `test_real_scenario_csv_mapping_by_name` is skipped, with the reason;
  the counts of `test_converted_inventory` go from 46 to 34 synthetic takes (the Music and Bouldering ones leave).
- New `scripts/44_scenario_bridge.py`: it writes `45_cache/<variant>/cache.parquet` with the same schema and manifest
  as stage 45 (stage 50 does not change). Probabilities = `cam_p_*`/`vid_p_*` of `predictions.csv` (already calibrated
  by the head), logits = log of the probabilities, `T = 1`; `replace` fusion (`fused_probs` = L1, `w = 0`); `fold` =
  `fold_s0` on train, −1 elsewhere; embeddings from `features/*.npz`; `mot_*` from the trajectories `traj/*.npz` (valid
  samples only) in the same windows as the embeddings, averaged; costs of stage 40 × number of windows. The test enters
  only with `--include-test` (from `sealed/predictions.csv`), at the final step; without it, 44 refuses to run if
  `predictions.csv` contains test rows.
- Takes without a CamFormer embedding (audit of 17/11, item 3): out of the cache, listed in `no_l0.csv` with the
  correctness of L1. Rule fixed now: in the final result they go to EgoVLP and count as cost L0 + L1 (the accuracy and
  cost of the cascade are corrected with these takes before reporting them).
- Operating point (stage 50): new key `gate.operating_rule`. `within_pts` (the default of this run) = the smallest
  escalation share with accuracy on val ≥ accuracy of always_L1 on val − `within_pts` (0.01); if no share is enough,
  everything goes to L1. `budget` = the rule from before (50% of the cost of always_L1). With `within_pts` the column
  `budget_frac` of the operating point is −1 (stage 52 groups by that column and would drop NaNs). The 25% and 50%
  budgets and the full curve stay in the results for comparison.
- `gate.bacc_min_takes` from 5 to 1: a class leaves the balanced accuracy only with 0 takes (as in N29).
- Launch: `configs/hpc/scenario_v1.yaml`, `hpc/gate_scenario.sh` (one CPU job: 40 → 44 → 50, without `--final`).
  Stage 52 is not needed: the mean over seeds is already inside `scenario_preds`.
- Tests: `tests/test_stage_44.py` (schema, takes without L0, Bouldering out, costs × windows, sealed test, chain 44 → 50
  also with `--final`, classes in a different order, test rows in `predictions.csv`);
  `test_operating_point_within_pts_on_val` and `test_stage_50_without_final` check the new rule.

**N29 — Changes to `scenario_preds` after the audit of 17/11 ("step A").** The default config follows the plan of the
afternoon: `split.official_val.mode = split` (official val cut once in half, by session and stratified by class,
`test_frac` 0.5, seed 0). Check on L0 (`check`): on val, balanced accuracy ≥ 0.40 and the lower end of the 95% bootstrap
interval (by session) above chance; on the out-of-sample train, recall > 0 for every class; new `require_all_classes`:
it fails if one of the 6 classes has 0 takes in val or in train (`heads.min_takes_metric = 1`: a class leaves the
balanced accuracy only with 0 takes). Takes without a CamFormer embedding stay in the predictions with an empty
`cam_pred` and are counted (`n_no_embedding`); `feat-camformer` stops only above 10% (`camformer.max_missing_frac`).
Pose axes: two pre-registered variants, identity (the default) and diag(−1, −1, 1); with 1, 3 and 5 windows they make 6
candidates (`hpc/candidates.sh`), and the new `select` stage picks on val the highest balanced accuracy of CamFormer
(on ties fewer windows, then the identity); EgoVLP uses the same number of windows. `check-up` stays only a
diagnostic. 18 tests (15 by Giorgio + 3).

**N30 — Correction of N24, point (2).** The difference from pre-training is not the rate (20 Hz against 1 kHz:
pre-training also uses trajectories resampled at 20 Hz) but the reference frame: our poses are those of the RGB camera
(the JSON extrinsics), while pre-training uses the trajectory of the Aria device. This is why the axis correction is
a choice pre-registered on val (N29) and not a fixed value.

**N31 — Limits to state and analyses decided before the run (17/11).**
- The pose variant and the windows are chosen on the same val as the L0 check: the check is a little optimistic; the
  test stays untouched.
- The gate learns on the out-of-sample train probabilities, which in `scenario_preds` are the mean of 3 seeds (one model
  per seed), and is applied to val and test, which are the mean of 15 models (5 folds × 3 seeds): less extreme
  probabilities. The operating point is chosen on val, so on the same distribution it is applied to.
- CamFormer's pre-training saw the takes of the official train (N24): the OOF train probabilities can be too
  confident. We report the results split by `camformer_pretrain`.
- Val and test are small (about 140 takes each: 1 point ≈ 1.4 takes): accuracy and cost of the cascade with a bootstrap
  interval by take/session.
- Sessions shared between train and val/test (in the `index` log): main result on all takes, plus a control analysis
  without those sessions.
- Extra analyses, evaluation only (the gate does not change): CamFormer accuracy by bands of camera motion (`mot_*`) on
  the out-of-sample train, overall and within each class; comparison with the fixed rule "procedural class predicted
  by L0 → EgoVLP".

**N32 — C16: gate analyses decided before the run (N31), stage 53.** Stage 50 writes to `scores.parquet`, for each
gate, the decision at the operating point (`esc__<gate>`, share of seeds that escalate) and, when available, session,
site, `camformer_pretrain` and costs. The new `scripts/53_scenario_analysis.py` (val only; test with `--final`, after
50 `--final`) computes, at the operating point: (1) accuracy, balanced accuracy and cost **counting the takes without
L0** of `no_l0.csv` as sent to EgoVLP at cost L0 + L1 (rule of N28); (2) the same without the sessions that also appear
in train (in the real run 19 sessions, 160 val+test takes out of 285); (3) 95% bootstrap intervals resampling
sessions, also for the accuracy difference from always_L1; (4) the fixed rule `rule_procedural` (escalate if the class
predicted by L0 is procedural); (5) the results split by `camformer_pretrain`; (6) the accuracy of L0 on the
out-of-sample train by quintiles of `mot_energy` (thresholds from train), overall and per class, with a figure. It
runs in `hpc/gate_scenario.sh` after stage 50. The gate does not change. Tests: `test_analysis_53` and the final chain
in `test_bridge_cache_and_gate` (`tests/test_stage_44.py`).

**N33 — Val results frozen before the test (17/11, evening).** `scenario_preds` run `official_v2` (commit C16 of
cascade-v2 for the gate) and gate run `scenario_v1`. No test number was looked at; after this entry nothing in the
method changes, and the test is evaluated only once.
- Candidates (`select` stage, CamFormer bacc on val, 143 takes): identity with 5 windows 0.824 (CI 0.759–0.914),
  identity 3 windows 0.790, diag(−1,−1,1) 5 windows 0.789, identity 1 window 0.679, diag 3 windows 0.659, diag 1 window
  0.566. Choice: **identity, 5 windows**.
- Index: 1,273 takes; val 143 and test 142 from the official val, all 6 classes on both sides (Soccer 7 in val); 19
  sessions shared between train and val/test (160 takes out of 285), of which 10 sessions and 85 takes in val.
- L0 check passed: val bacc 0.824 (CI 0.759–0.914) against chance 0.167; no class below the threshold or missing; 0
  takes without an embedding. CamFormer: acc 0.924 / bacc 0.861 on the out-of-sample train, acc 0.909 on val. EgoVLP:
  acc and bacc 1.000 both on the out-of-sample train and on val (L1 at the ceiling).
- Gates, operating point on val (within 1 point of always_L1 = 1.000): all the learned and threshold gates have acc
  0.993 (1 error out of 143) and bacc 0.983; cost relative to always_L1: lgbm_ABC 0.255, logreg_ABC 0.274, logreg_AC
  0.281, thr_maxprob/thr_entropy/logreg_A/compare_A/compare_AC 0.288, thr_margin 0.302; oracle 0.092; random 0.895 (acc
  0.993). Rule `rule_procedural`: acc 0.979, cost 0.246 (it does not meet the criterion). Accuracy difference from
  always_L1 (bootstrap by session, 37 sessions): from −0.025 to 0 for the gates.
- Without the shared sessions (58 val takes): CamFormer acc 0.810; at the operating points chosen on the whole val the
  gates have acc 0.983 with cost 0.51–0.59 (lgbm_ABC 0.512).
- Motion (out-of-sample train, quintiles of `mot_energy`): L0 acc 0.74 in the lowest quintile against 0.95–0.99 in the
  others; within the class it goes up with motion for Basketball (0.84 → 0.98), Cooking (0.68 → 0.90), Bike repair
  (0.63 → 1.00); Health goes the other way, but with 7 takes in the second quintile.
- `camformer_pretrain`: all 143 val takes are in the val split of CamFormer's pre-training (not in its train).
- Limit: the operating point is chosen and measured on the same val and sits at the edge (1 error): on the test it can
  move by one take either way.

**N34 — Test, evaluated once (17/11, 22:50).** After N33 (commit C17): sealed predictions of `scenario_preds` (`heads
--eval-test`, job 770706) and stages 44 `--include-test`, 50 `--final`, 53 `--final` (job 770707); one line in
`50_gate/FINAL_RUNS.log`, `FINAL_DONE` present. Outputs copied to `results/scenario_v1/` (without the `.parquet`
files, excluded by `.gitignore`).
- Test, 142 takes: CamFormer acc 0.915, bacc 0.791 (CI 0.627–0.907); EgoVLP acc and bacc 1.000; 0 takes without L0.
- Gates with the threshold chosen on val: acc 1.000 and bacc 1.000 for all the learned and threshold gates; cost
  relative to always_L1: lgbm_ABC 0.245, logreg_ABC 0.262, logreg_AC 0.269,
  thr_maxprob/thr_margin/thr_entropy/logreg_A/compare_A/compare_AC 0.283; oracle 0.093; random acc 0.992 at cost
  0.871. Rule `rule_procedural`: acc 0.979, bacc 0.941, cost 0.240. Accuracy difference from always_L1 (bootstrap by
  session, 40 sessions): 0 for the gates.
- Without the sessions shared with train (9 sessions, 75 takes; 67 takes remain): CamFormer acc 0.821; gates acc 1.000
  with cost 0.50–0.58 (lgbm_ABC 0.503); procedural rule acc 0.955.
- Share sent to EgoVLP per class (test): thr_maxprob Basketball 0.11, Dance 0.07, Soccer 0.67, Cooking 0.75, Health
  0.80, Bike repair 0.88; lgbm_ABC 0.11, 0.02, 0.67, 0.69, 0.77, 0.75.
- RQ2 (paired comparisons, 1,000 resamples): `compare_AC` and `compare_A` identical; `logreg_AC` − `logreg_A` on the
  area under the curve −0.0006 (CI −0.0020 / 0.0006, p 0.38). The trajectory and motion features do not significantly
  improve the gate over CamFormer's confidence alone; the gates with group C escalate 3–5 fewer takes on both val and
  test (a trend, not evidence).

**N35 — Report figures (17/11, night).** New `scripts/60_figures.py` (C19): it reads only `results/scenario_v1/` (no new
computation; method and numbers unchanged after N34) and writes PDF and PNG files to `results/scenario_v1/figures/`.
- `fig1_accuracy_cost`: accuracy against cost (% of always_L1) on val and test; curves of lgbm_ABC, thr_maxprob and
  random (mean over seeds), points for always_L0, always_L1, the oracle and `rule_procedural` from
  `53_analysis/op_all.csv`; the operating point of lgbm_ABC is circled.
- `fig2_escalation_by_class`: share of takes sent to EgoVLP per class (lgbm_ABC), val and test.
- `fig3_motion_l0`: CamFormer acc per quintile of `mot_energy` (out-of-sample train), integer x axis; for the single
  classes only the quintiles with at least 10 takes.
- Note for the report: on the test thr_maxprob reaches acc 1.000 at a lower cost than lgbm_ABC; the operating point
  stays the one chosen on val (N33), consistent with an RQ2 that is not significant (N34).

These figures belong to the first evaluation; they are now in `results/first_evaluation_official_split/figures/`.

**N36 — `scenario_preds` in the repository (17/11, night).** We copied Giorgio's package into `scenario_preds/` with
the step A changes (`select` stage, `hpc/candidates.sh`, checks on missing classes and embeddings; 18 tests), that is
exactly the version used for `official_v2` and so for all the results of N33 and N34. `.ruff_cache` and the working
files are left out. The code was not reformatted: it is excluded from black and ruff in `pyproject.toml` (like
`third_party`), and its tests run from its own folder (`cd scenario_preds && python -m pytest -q`), not with the
repository's. `hpc/env.sh` holds only the default cluster paths and account; personal values go in
`hpc/env.local.sh`, not in git. (Since C26 these files live in `scripts/01_extract_features.py`, `src/cascade/` and
`tests/features/`, still excluded from black and ruff, and their tests run with the others.)

**N37 — Gate v2 (stage 56) and fixes to the gate v1 (18/11). Draft to approve with Giorgio before the run.**
After N34 the official test can no longer judge a method changed after looking at it. The gate v2 redoes the
evaluation with a nested CV by session on all the takes of the 6 classes. **It is not a blind evaluation**: the
official val and test were already analysed (N33, N34 and the analysis of 18/11, in which the session
`georgiatech_bike_06` motivated the risk rule), and the choice of poses and of the 5 windows (`select`, N33) was made
on val. The results of N33 and N34 stay reported as the "first evaluation".
- Fixes to the gate v1 (`gate/protocol.py`, `eval/gate_eval.py`, `eval/curves.py`, `configs/gate.yaml`): options
  `gate.inner_cv_groups` and `gate.bootstrap_unit` (`take` | `session`). With `session` the gates' inner CV groups by
  session (before it grouped by take, i.e. by row: no grouping), and the intervals of curves, comparisons, AUROC and
  per-class accuracy resample sessions. Without a `session` column in the cache it falls back to take with a warning.
  **The default is `take` in `configs/gate.yaml`**, so the `scenario_v1` run (N33, N34) stays reproducible with stage
  50 as it is; the per-session version is turned on with `--config configs/hpc/gate_v1_session.yaml`. The gate v2
  (stage 56) always works by session.
- `src/cascade/scenario_io.py`: `compute_windows`, `take_motion`, `load_npz` moved from stage 44 unchanged, so that the
  gate v2 computes motion as in the v1 cache.
- New package `src/cascade/gatev2/` and stage `scripts/56_gate_v2.py` (CPU; `hpc/gate_v2.sh`). It reads the frozen
  embeddings of scenario_preds (not the predictions) and, for each outer fold k (by session, stratified): (1) inner CV
  by session of the CamFormer heads on the outer train, which gives the out-of-sample predictions the gate is trained
  on; (2) cross-fitting by session of the gate, which gives out-of-sample scores for the calibrator and the thresholds;
  (3) heads refit on the outer train, gate, calibrator and thresholds frozen and applied to fold k. Heads as in
  official_v2 (logistic regression, `class_weight: balanced`, C and temperature by CV by session). Gate target
  e = 1 − c0 (if EgoVLP is always right this equals the expected benefit); if EgoVLP is wrong the log says so.
- Gates: `thr_maxprob` (the reference), `logreg_A`, `logreg_AC`, `logreg_ABC` (logistic regressions WITHOUT class
  weights), `lgbm_ABC` (secondary, for continuity with v1; outside the pre-registered comparisons), `class_rule`
  (escalates the predicted procedural takes). Features as in v1 but with the logit of p_max; the per-class reliability
  (B) leaves out the row's session. Platt calibrator on out-of-sample scores, `r_up` = 90% quantile over 200 session
  bootstraps.
- Rules (all fixed on the outer train and applied to the outer fold): `cost` (escalate if `r_up` ≥ β; variant
  `cost_hat` with `r_hat`), `risk` (largest coverage with a Clopper-Pearson bound ≤ α on a fixed grid of 20 coverages,
  each one at level 1 − δ/20: Bonferroni; the fixed sequence proposed on 18/11 would stop at once, because with few
  kept takes the bound is wide), `acc_target` (≤ 1% of errors let through, like `within_pts`), `budget` (10, 20, 30%).
- **Values to pre-register** (defaults of `configs/gate_v2.yaml`): folds 5/5/5, 3 outer partitions (seeds 0, 1, 2),
  β ∈ {0.01, 0.02, 0.05, 0.10, 0.20}, α ∈ {0.02, 0.05, 0.10}, δ = 0.1, costs per window L0 0.121627 and L1 92.430496
  GMACs (stage 40 of scenario_v1, 5 windows). **Main result: the takes not seen in CamFormer's pre-training
  (`pretrain_unseen`), `thr_maxprob`, cost rule with β = 0.05**. β is the price of one EgoVLP call in errors, a value
  judgement and not a parameter to optimise: the result is reported as a **curve over all β** (errors left against
  share sent to EgoVLP), and **no β is chosen after looking at the results**. β = 0.05 is only the point of the summary
  and of the comparisons; any other β called "best" afterwards must be presented as an exploratory analysis. Metrics:
  accuracy and share sent to EgoVLP ("camera on"), with a 95% bootstrap CI by session (1,000 resamples). Secondary: all
  takes, `pretrain_seen`, official val + test (`official_eval`), the other rules and parameters.
- Pre-registered comparisons (A − B, AUROC, accuracy and share sent to EgoVLP at the main point, Holm correction per
  subset and metric): `logreg_A` − `thr_maxprob`, `logreg_AC` − `logreg_A` (motion), `logreg_ABC` − `logreg_AC`
  (per-class reliability), `thr_maxprob` − `class_rule` (per-take decision against per-class rule; the per-class rule
  uses its natural decision). Pre-registered analysis: among the takes predicted as procedural, how many stay on
  CamFormer and how many of those are right (`procedural.csv`).
- Cost against EgoVLP alone (`cost_vs_egovlp.csv`, added on 18/11 before the run): for each β of the cost rule, for
  `class_rule` and for the references, GMACs per take of the cascade (5 × (C_L0 + ρ · C_L1)) against 5 × C_L1 for
  EgoVLP alone on every take (no CamFormer in front), with share, saving, reduction factor, totals over the subset and
  an interval from the session bootstrap of ρ. The point at the main β (0.05) goes into `summary.json`
  (`cost_vs_egovlp_primary`) and into the log.
- Extra descriptive analysis (a separate script, after the run, on the stage 56 outputs): out-of-sample correctness of
  CamFormer against `mot_energy` with class fixed effects (logistic regression), to separate the effect of motion from
  that of the class (in v1 the stillest quintile was almost all procedural).
- Limits to state: Clopper-Pearson assumes independent takes (sessions are not, so the guarantee of `risk` is
  optimistic and should be read with the session CI); CamFormer's pre-training saw 936 of the 1,273 takes, which in the
  CV also enter the evaluation (this is why the main result is on `pretrain_unseen`); the heads on the outer train use
  more data than the inner ones the gate is trained on (variant `outer_l0: inner_mean` as a check); the calibrator
  learns mostly from takes seen in pre-training (fewer errors), so on the unseen ones it can underestimate the risk
  (check: `calibration.csv` per subset); the cost counts only GMACs (`extra_gmacs_L0`/`extra_gmacs_L1` for sensitivity,
  not measured).
- Manifest: `STAGE_KEYS["56_gate_v2"] = ["gate_v2", "data.vertical_axis"]`. Without keys, stage 56 did not record the
  config, and a run with a changed `gate_v2` looked "already complete"; now it gives an error like the other stages.
  `hpc/gate_v2.sh`: time limit 4 h (the duration on real data was not measured yet).
- Tests: `tests/test_gatev2.py` (17, including `test_stage_56_refuses_changed_gate_v2_config`): Clopper-Pearson bound
  (0.0237 with 0 errors out of 96, δ = 0.1), risk rule with Bonferroni, thresholds of `acc_target` and `budget`, cost
  rule = β, calibrator (risk recovered, `r_up` ≥ `r_hat`, wider with less data, a single class, isotonic), session
  folds, reliability without the row's own session, **no take of the outer fold in a head or a gate of that fold** and
  correct cross-fitting, chain of stage 56 on a fake run, main point off the grid refused; fixes to v1 (CV groups,
  session bootstrap weights, wider CIs with homogeneous sessions). Suite: 307 passed, 10 skipped; the 5 tests that need
  `torch`/`av` (stages 10 and 44) do not run in the sandbox, and the chain 44 → 50 `--final` → 53 was tried by hand with
  a simulated stage 40.

**N38 — Results of the nested CV (run 772601) and two errors found after the run (18/11, evening).** SLURM job 772601
(code C22), 2 h 39 min, COMPLETED; outputs in `results/scenario_cv/` (C23), checked byte by byte against the archive on
the cluster. These are the pre-registered results (N37) and they are not overwritten: the stage 56 manifest gives an
error if the config changes under the same tag. The fixes are evaluated under new tags (N39) and reported next to them.
- Counts: 1,271 takes, 294 sessions (median 1 take per session), 0 takes without CamFormer, EgoVLP or motion. CamFormer
  out of sample 0.924 (all), 0.882 (the 337 not seen in pre-training); EgoVLP 0.999 (4 errors, all in
  `upenn_0720_Dance_3`, in a single repeat).
- **Main result (pre-registered)**, unseen takes, `thr_maxprob`, cost β = 0.05: accuracy 0.995 [0.984, 1.000], share
  sent to EgoVLP 34.4% [25.8, 46.6], 159.7 against 462.2 GMACs per take = 34.6% of EgoVLP alone (2.9 times less). All
  takes: 0.997 at 29.7%. Oracle 11.8% (unseen), 7.7% (all).
- Curve over β (unseen, `thr_maxprob`): accuracy 1.000 / 0.998 / 0.995 / 0.985 / 0.980 with 47.1 / 41.4 / 34.6 / 27.6 /
  21.1% of the compute of EgoVLP alone (β = 0.01 / 0.02 / 0.05 / 0.10 / 0.20).
- Same rule as v1 (`acc_target` 0.01): all takes 0.990 at 19.0%; unseen 0.985 at 23.5%, i.e. beyond the 1-point
  tolerance (v1, test: 1.000 at 28.3%).
- Pre-registered comparisons (unseen, Holm): `logreg_A` = `thr_maxprob` (AUROC 0.921 against 0.921; share −0.8 points,
  p = 0.048); RQ2 null (`logreg_AC` − `logreg_A`: AUROC −0.002, p = 1; share −1.6 points, p = 0.22); `logreg_ABC` =
  `logreg_AC`; `thr_maxprob` beats the per-class rule in AUROC (0.92 against 0.80, p < 0.001), not in accuracy and share
  after Holm.
- Guarantee of the risk rule (`thr_maxprob`, error of the takes left to CamFormer ≤ α in the outer fold): on all takes
  15/15 folds for α = 0.02 and 0.05, 13/15 for α = 0.10; **on the unseen takes 12/15, 9/15 and 7/15**: on the main
  subset the guarantee does not hold (error 2).
- **Error 1 (code): reliability of the predicted class (group B).** For the gates' training rows,
  `FeatureBuilder.reliability` left out only the row's own session. Inside the training set, the value
  (correct − correct in the session)/(predicted − predicted in the session) is a decreasing function of the session's
  outcomes; with one-take sessions (the median) it is exactly a leave-one-out target encoding. On the real data the
  within-class correlation with the error is 0.74 / 0.60 / 0.84 / 0.95 / 0.93 / 0.82 (Giorgio's analysis, recomputed
  independently with the same numbers); LightGBM learns a reversed rule (`lgbm_ABC` AUROC 0.738 on all takes, 0.695 on
  the unseen ones). The evaluation stays clean (no evaluated take in the fits): the effect makes the gate worse, it
  does not make the results optimistic. The test "changing the outcome of one row does not change its value" does not
  catch the error (the change enters both the total and the session, and cancels out); the right test is the
  within-class correlation with the error. Fix (`gatev2/scorers.py`, Giorgio): for the fit rows the reliability is
  computed OUT OF FOLD, with `reliability_folds` (5) folds by session stratified by predicted class, as in the gate v1,
  which left the fold out; for new rows it is still computed on all the fit rows. Trial on the official_v2 train (986
  takes, CV by session 5 × 3): LightGBM A 0.911–0.925, A + old B 0.835, A + fixed B 0.932–0.940; logistic regression
  about 0.94 in every case. We evaluated and set aside the variant "predicted class one-hot" (no leak by construction,
  equivalent AUROC), and kept the out-of-fold version for consistency with the gate v1. A/B trial on the fake run:
  among the per-take columns only those of `logreg_ABC` (and `lgbm_ABC`) change; all the others are identical bit for
  bit.
- **Error 2 (method): calibration on the unseen takes.** The calibrator and the thresholds are fitted on the outer
  train, where about 73% of the takes were seen in CamFormer's pre-training (L0 0.94 against 0.88 on the unseen ones).
  On the 337 unseen takes the risk is underestimated (predicted 9.3%, observed 11.8%; middle bands 3.1% against 8.9%
  and 28% against 40%): `acc_target` gives 0.985 and the guarantee of `risk` holds in 9 folds out of 15 with α = 0.05.
  New option `gate_v2.calibration_population` (`all` = as in run 772601, default unchanged; `pretrain_unseen` =
  calibrator and thresholds of `risk`, `acc_target`, `budget` fitted only on the outer-train takes not seen in
  pre-training; the scorer is always trained on all of them). With fewer than `calibration_min_errors` (10) errors among
  those rows it falls back to all rows, with the column `cal_population` = `all_fallback` in `thresholds.csv`. It uses
  only train metadata: the decision on the evaluated takes does not change form.
- Added limits: the session bootstrap intervals do not include the variability of retraining (Bates, Hastie and
  Tibshirani, 2023); the EgoVLP head picks C = 100 (the edge of the grid) in 11 folds out of 15 and T down to 0.05 (the
  edge): saturated EgoVLP probabilities, harmless for the gate, which uses only its class; the official val and test
  takes are in the validation list of CamFormer's pre-training [TO CHECK whether it was used for the checkpoint].
- Withdrawn: "the LightGBM advantage of v1 is not confirmed" (`lgbm_ABC` of run 772601 cannot be evaluated).

**N39 — Pre-registration of the corrective runs (18/11, evening, before the runs).** Code C24; same `scenario_preds`
run (`official_v2`), same `RUN_NAME=gate_v2`, new tags next to `main` (run 772601).
- `TAG=n38_rel CONFIG=configs/gate_v2_n38_rel.yaml`: ONLY the fix of error 1 (`calibration_population: all`).
- `TAG=n38_cal CONFIG=configs/gate_v2_n38_cal.yaml`: errors 1 and 2 (`calibration_population: pretrain_unseen`).
- Everything else identical to run 772601: the N37 config (β up to 0.20, `lgbm_ABC` among the scorers, already the
  default in `configs/gate_v2.yaml`), seeds, partitions, heads, rules, subsets, comparisons. `N_JOBS` may change (it is
  only parallelism).
- **Pre-registered checks:** in `n38_rel` the gates that do not use B (`thr_maxprob`, `logreg_A`, `logreg_AC`,
  `class_rule`) and the references must give numbers identical to run 772601 (`metrics.csv`, `auc.csv`,
  `cost_vs_egovlp.csv`); if not, the results of the corrective runs are not used. In `n38_cal` the scores (`s__*` in
  `takes.csv.gz`) must be identical to `n38_rel`; only calibration and thresholds change. We report how many folds fall
  back to `all_fallback`.
- To report: first the pre-registered results (N38); then `n38_rel` (LightGBM can be evaluated) and `n38_cal`
  (calibration on the unseen takes), stated as corrections made after the run. We do not pick the "best" version
  afterwards: the main β stays 0.05 and the main result stays on `pretrain_unseen`.
- Tests (`tests/test_gatev2.py`, Giorgio): out-of-fold reliability computed correctly; within-class correlation with
  the error < 0.3 with sessions of 1 and of 6 takes (the code of run 772601 gives +1.00 and +0.62); calibration only on
  the unseen takes and fallback to all rows with few errors; default `all` unchanged; unknown population refused; n38
  configs equal to run 772601 except `calibration_population`.

**N40 — C25: results of the corrective runs (18/11, night).** Jobs 773281 (`n38_rel`) and 773282 (`n38_cal`),
committed as they came out, with their SLURM logs; now in `results/nested_cv/reliability_fix_n38_rel/` and
`results/nested_cv/report_n38_cal/`. We checked the pre-registered checks of N39 on the committed files on 26/11:
- `n38_rel` against run 772601: the rows of `thr_maxprob`, `logreg_A`, `logreg_AC`, `class_rule` and of the references
  are identical in `metrics.csv`, `auc.csv` and `cost_vs_egovlp.csv`. With the fix, `lgbm_ABC` has AUROC 0.918 on the
  unseen takes (0.695 in run 772601), in line with the other gates.
- `n38_cal` against `n38_rel`: the six score columns `s__*` of `takes.csv.gz` are identical. No fold fell back:
  `cal_population` is `pretrain_unseen` in all 1,530 rows of `thresholds.csv`.
- Main result of `n38_cal` (unseen takes, `thr_maxprob`, cost β = 0.05): accuracy 0.997 [0.989, 1.000], share sent to
  EgoVLP 42.2% [32.4, 55.3], 42.4% of the compute of EgoVLP alone (2.4 times less). The risk guarantee holds in 15, 13
  and 12 folds out of 15 for α = 0.02, 0.05 and 0.10 on the main subset.
- The report is written on `n38_cal`. `results/README.md` lists the three runs and what each one is.

**N41 — C26: reorganisation of the repository, form only (25/11).** To follow the course rules (data, model, training
and evaluation in separate places, clean names), we moved the code that produced the report into
`src/cascade/{data,models,training,evaluation,utils}` and three numbered scripts (`01_extract_features.py`,
`02_count_gmac.py`, `03_nested_cv.py`), and renamed the HPC scripts and the result folders. Before that we put the tag
`archive/development` on C25, so the full history and the first pipeline stay available.
- No logic changed: only file moves, imports and paths. The output folders keep the stage names `40_cost/` and
  `56_gate_v2/` (constants in the code), so that the code is exactly the one that produced the results.
- Moved out of the main branch (still in the tag): the first pipeline (scripts 00–35, 41–53, 60, the synthetic pipeline
  script), the v1 modules, `src/data`, `tools`, `splits`, `configs/{ablations,hpc,keystep_labels.yaml}`, the old HPC
  scripts, `report/`, `docs/history` and 33 test files. Three test functions that used removed code were dropped.
- Checks in the sandbox (Python 3.9): the syntax tree of every moved file is the same apart from imports and paths (7
  declared exceptions: paths, script names and the removed test functions); the vendored files are identical byte for byte; the
  default config loads identically; on synthetic data the feature pipeline with the real CamFormer and the whole chain
  of stage 56 give 2,113 output files identical byte for byte between the old and the new layout (stage 40 identical
  apart from latencies). pytest: 116 passed, 6 skipped. The only visible difference is the logger name in log lines.

**N42 — C27–C29: documentation for the submission (26/11).** No logic changed in any of these commits.
- C27: `README.md` (setup, the four steps, where each part of the report comes from), `results/README.md`, and
  `scripts/04_check_report_numbers.py`, which recomputes 162 numbers of the report from `results/` and finds them all
  equal. While writing it we found one difference, Table 1 of the report: the interval of "CamFormer only" on the 337
  takes is .922 in the files and .923 in the report; to fix in the report. The sentence of section 5.5 on the
  guarantee holding in 15, 13 and 12 folds out of 15 is confirmed (N40).
- C28: comments and docstrings in English (52 Python files with the same syntax tree once docstrings are removed; 27
  other files where only comments changed, plus 6 labels: the Makefile help, two CI step names, the package
  description). Log and error messages stay in Italian, because they appear in the committed logs.
- C29: this log translated into English (`docs/decision_log.md`, it was `docs/REGISTRO_MODIFICHE.md`); the HPC runbook
  rewritten for the current scripts (`docs/hpc_runbook.md`); the notes of the feature package merged into the README;
  the summary of 18/11 translated (`results/first_evaluation_vs_nested_cv.md`). In this file we replaced the commit
  hashes with commit labels.

**N43 — C30: figures and tables of the report in `results/report/` (26/11).** The course rules ask for the plots and
key tables in the results folder. We could not find the LaTeX sources of the report, which had their own plotting
scripts, so we wrote two new scripts that read only `results/nested_cv/report_n38_cal/` (and the labels of the feature
run), checked against the PDF of the report:
- `scripts/05_make_report_figures.py` draws Figure 2 (accuracy against compute for the cost rule over β, with the
  ranking line, the oracle and EgoVLP alone, on the unseen takes and on all takes) and Figure 3 (reliability of the
  threshold gate's risk, 10 equal-count bins). It also writes the numbers behind each figure as CSV. The figures
  show the same values as the PDF; fonts and axis style differ a little.
- `scripts/06_make_report_tables.py` builds Tables 1, 2 and 3 as CSV files and as `tables.md`, rounded like the
  report. Every number matches the report except the one already known: the upper end of the interval of
  "CamFormer only" in Table 1 is .922, not .923 (N42).
- Figure 1 is a diagram of the cascade, not a result, so it has no script.
- No code of the pipeline changes, and no result file is rewritten: the two scripts only add files under
  `results/report/`.

**N44 — Rewrite of the commit messages (27/11).** Before the submission we rewrote the messages of all the commits
up to C30, once, with `git filter-branch --msg-filter`. The messages up to C26 were in Italian and are now in
English, and we removed two attribution lines (`Co-Authored-By`, `Claude-Session`) from the message of C11. Nothing
else changed: every commit keeps the same tree (the same files, byte for byte), author, committer and dates, which
we checked commit by commit before pushing. The tag `archive/development` now points to the rewritten C25.

The rewrite changes every commit hash. Some files in `results/` were written by the code and quote the hash of the
commit that produced them, and we left them as they are: `commit=96e391e` in the log of job 772601 (C22),
`commit=1048eae` in the logs of jobs 773281 and 773282 (C24), and `362f5e2` in `FINAL_RUNS.log` and `FINAL_DONE` of
the first evaluation (C17). This table maps the old hashes to the new ones:

| Commit | Old hash | New hash |
|---|---|---|
| C01 | `9c9147d` | `1bb4460` |
| C01 | `14a0f72` | `f6b444a` |
| C01 | `3c5c1fd` | `2dd6b8f` |
| C01 | `b90a6a8` | `5e7f8da` |
| C02 | `2e87fc4` | `0e83e71` |
| C03 | `1f3b7e5` | `7a38a91` |
| C04 | `41b0f62` | `5374c4f` |
| C05 | `1d8dd17` | `4d4a657` |
| C06 | `88384ba` | `69ac3b9` |
| C07a | `fe15eb0` | `cc0df4c` |
| C07b | `31714af` | `a6e51d9` |
| C08a | `d134b40` | `ad3375f` |
| C08b | `ca2154f` | `4d28f1d` |
| C09 | `aefe79b` | `490c641` |
| C10 | `f7e6325` | `fe1a123` |
| C11 | `9b9b0fb` | `654f916` |
| C12a | `d3fad63` | `cd91f71` |
| C12b | `4d55e67` | `39dbdb6` |
| C12c | `52546c7` | `9a7cfef` |
| C13a | `9bbe819` | `fc3576b` |
| C14a | `d25e685` | `2c45162` |
| C14b-1 | `a2b4615` | `02b8fce` |
| C14c-1 | `b93cee3` | `25e48a7` |
| C14d | `869fe21` | `7dc50dd` |
| C14e | `3b5f6b4` | `19d44b7` |
| C14f | `9ab51f9` | `ab0a426` |
| C14c-2 | `181895c` | `261c54d` |
| C15 | `2fa1aad` | `d6cfa15` |
| C16 | `ebbf20a` | `06e8c77` |
| C17 | `362f5e2` | `f808f6d` |
| C18 | `5311c3e` | `d4eb957` |
| C19 | `2e66d57` | `c82a594` |
| C20 | `e7d43a1` | `36f7951` |
| C21 | `6bcffa3` | `f5d5db8` |
| C22 | `96e391e` | `72b62da` |
| C23 | `e0cc174` | `dcd4e94` |
| C24 | `1048eae` | `124a4f7` |
| C25 | `0408643` | `c770919` |
| C26 | `3e0c5a6` | `578a8aa` |
| C27 | `d240216` | `c935a64` |
| C28 | `853057e` | `6cee536` |
| C29 | `d16dc05` | `243ff94` |
| C30 | `5df9d88` | `64736b1` |

