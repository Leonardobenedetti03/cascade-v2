# The cascade CamFormer → gate → EgoVLP: first evaluation (v1) and nested CV by session (v2)

*Written on 18/11, translated on 26/11. Sources: `first_evaluation_official_split/` and `features/official_v2/` (v1,
decision log N33–N34) and `nested_cv/preregistered_772601/` (v2, decision log N37, SLURM job 772601, commit C22). All
numbers are read from the output files; none was recomputed by hand.*

**This summary is not the report.** It compares our first evaluation with the pre-registered nested CV run, the day
both were done. After it we found two errors in that run and made two corrective runs (N38–N40). The report uses the
last of them, `nested_cv/report_n38_cal/`, whose main numbers differ from the ones below (see the README).

## 0. In one line

The cascade lifts the accuracy from CamFormer's level (0.88–0.92) to EgoVLP's (0.99–1.00) while calling EgoVLP on
about a third of the takes, so with about a third of the compute of "EgoVLP alone". v2 confirms the message of v1 with
a much more solid evaluation: a plain threshold on CamFormer's confidence is enough (the learned gates add nothing),
and the saving on new sessions was underestimated. LightGBM cannot be evaluated in run 772601 because of an error in
feature B, fixed in the corrective run `n38_rel`.

## 1. What both evaluations share

- **Task:** the scenario of an Ego-Exo4D take, 6 classes: Basketball, Dance, Soccer (physical), Cooking, Health, Bike
  repair (procedural).
- **Data:** 1,273 takes; 1,271 in the 6 classes (the 2 Bouldering takes are left out), 294 sessions.
- **L0 (cheap):** **frozen** CamFormer (camera trajectory, 5 windows, identity poses) + a logistic head. Cost 0.61 GMAC
  per take.
- **L1 (expensive):** **frozen** EgoVLP (video, 5 windows) + a logistic head. Cost 462.15 GMAC per take.
- **Gate:** for each take it decides whether CamFormer is enough or EgoVLP is needed. Features: A = CamFormer's
  confidence (p_max, margin, entropy), B = reliability of the predicted class, C = camera motion.
- **What is trained:** only the logistic heads and the gates. The CamFormer and EgoVLP embeddings are computed once
  and reused.
- **Cost:** "% of EgoVLP alone" = compute of the cascade divided by the compute of EgoVLP on every take.

## 2. The first evaluation (v1, 17/11)

**How it works.**
- A fixed split by session: train 988, val 143, test 142 (the official val cut in half by session and class).
- Heads: 5 folds by session on train, with out-of-fold (OOF) predictions on train.
- Gates trained on the train OOF predictions. Operating point chosen **on val**: the lowest use of EgoVLP with accuracy
  within 1 point of "always EgoVLP".
- Test evaluated **only once** (17/11, 22:50, N34).

**Results** (cost as % of always EgoVLP):

| | Val (143) | Test (142) |
|---|---|---|
| CamFormer alone | acc 0.909 | acc 0.915 |
| EgoVLP alone | acc 1.000 | acc 1.000 |
| `lgbm_ABC` (chosen as the main gate) | acc 0.993, 25.5% | acc 1.000, 24.5% |
| `thr_maxprob` (threshold on p_max) | acc 0.993, 28.8% | acc 1.000, 28.3% |
| Oracle | 9.2% | 9.3% |
| Random gate | acc 0.993, 87.1% | acc 0.992, 87.1% |
| Rule "send the procedural ones" | acc 0.979, 24.6% | acc 0.979, 24.0% |

- New sessions in the test (67 takes): CamFormer 0.82; cascade acc 1.000 at a cost of 50–58%.
- RQ2 (does the trajectory help the gate?): no, p = 0.38.

**Problems that came up** (Giorgio's analysis, 18/11):
1. Evaluation on very little data: 12–13 CamFormer errors in val and in test.
2. Operating point at the edge, decided by a single session (`georgiatech_bike_06_10`): 1 error out of 143.
3. Saving inflated by sessions already seen in train (56% of val+test).
4. Cost used only as an implicit constraint; scores of the logistic gates not calibrated.
5. Inner CV of the gates by take instead of by session.
6. Train predictions (mean of 3 models) and val/test predictions (mean of 15) on different scales.

## 3. The evaluation with a nested CV by session (v2, 18/11)

**Why.** The v1 test had already been opened, so we could not use it to judge a method fixed after seeing it. We needed
a new evaluation in which every take is predicted by models that never saw its session. The design was pre-registered
in N37 **before** the run; v1 was not deleted.

**How it works.**
- **Outer level:** the 1,271 takes split into 5 folds by session (stratified by class), repeated 3 times: 15 outer
  folds. Each take is evaluated once per repeat.
- **Heads, inside each fold:** CamFormer heads refit with 5 inner folds by session, which gives the OOF predictions to
  train the gate; CamFormer and EgoVLP heads retrained on the rest, which predict the outer fold.
- **Gates, inside each fold:** trained by cross-fitting by session; inner CV by **session**; logistic gates **without**
  class weights.
- **Calibration:** Platt, from the score to the risk that CamFormer is wrong; conservative bound `r_up` from the
  session bootstrap.
- **Threshold rules:** **cost β** (escalate if the conservative risk ≥ β; β = 0.01 / 0.02 / 0.05 / 0.10 / 0.20);
  **guaranteed risk** α (Clopper-Pearson with Bonferroni); **within 1 point** (the v1 rule); budget.
- **Main result:** the takes not seen in CamFormer's pre-training (337 takes, 108 sessions), `thr_maxprob`, cost rule
  with β = 0.05.
- **Intervals and tests:** session bootstrap (1,000), comparisons with Holm correction.

Run: 2 h 39 min on CPU (about 2,000 logistic regressions for the heads), no errors.

### 3.1 Main result

| Unseen takes (337) | Accuracy | Share sent to EgoVLP | GMAC per take | % of EgoVLP alone |
|---|---|---|---|---|
| CamFormer alone | 0.882 | 0% | 0.6 | 0.1% |
| **Cascade, `thr_maxprob`, β = 0.05** | **0.995** [0.984 – 1.000] | **34.4%** [25.8 – 46.6] | **159.7** | **34.6%** [25.9 – 46.7], **2.9 times less** |
| EgoVLP alone | 1.000 | 100% | 462.2 | 100% |
| Oracle | 1.000 | 11.8% | 55.0 | 11.9% |

On **all** 1,271 takes: accuracy 0.997, share sent to EgoVLP 29.7%, 29.9% of the compute (3.3 times less).

### 3.2 Curve over β (`thr_maxprob`)

β is the price of one EgoVLP call expressed in errors: β = 0.05 means "I accept up to 20 calls for each error
avoided". We do not pick the "best" β by looking at the results: we report the curve.

| β | 0.01 | 0.02 | **0.05** | 0.10 | 0.20 |
|---|---|---|---|---|---|
| Unseen: accuracy | 1.000 | 0.998 | **0.995** | 0.985 | 0.980 |
| Unseen: % of EgoVLP alone | 47.1% | 41.4% | **34.6%** | 27.6% | 21.1% |
| All: accuracy | 0.999 | 0.998 | 0.997 | 0.992 | 0.986 |
| All: % of EgoVLP alone | 43.9% | 37.9% | 29.9% | 23.2% | 16.1% |

### 3.3 Pre-registered comparisons (unseen takes, Holm-corrected p)

- **`logreg_A` against `thr_maxprob`:** same AUROC (0.921 against 0.921); share −0.8 points (p = 0.048). A learned gate
  adds nothing relevant to the threshold on p_max (Prop. 2).
- **`logreg_AC` against `logreg_A` (RQ2):** AUROC −0.002 (p = 1.0); share −1.6 points (p = 0.22). The trajectory does
  not help: the null result is confirmed.
- **`logreg_ABC` against `logreg_AC`:** no difference. The per-class reliability adds nothing.
- **`thr_maxprob` against the per-class rule:** AUROC 0.921 against 0.798 (p < 0.001); acc +1 point, share +5 points
  (not significant after Holm). The threshold ranks the errors better; in the procedural classes it keeps 14% of the
  takes on CamFormer (about 1 error in 14), the rule none.

### 3.4 Checks

- **Guaranteed risk:** on all takes the bound on the error of the takes left to CamFormer holds in 15 folds out of 15
  for α = 0.02 and 0.05, and in 13 out of 15 for α = 0.10; **on the unseen takes in 12, 9 and 7 out of 15**. On the main
  subset the guarantee does not hold, because the calibrator and the thresholds learn mostly from takes seen in
  pre-training (error 2, N38). Fix: calibration only on the unseen train takes, corrective run `n38_cal` (N39).
- **EgoVLP:** 0.999 in the CV. It gets only 4 takes wrong, all from the session `upenn_0720_Dance_3`, in a single
  repeat.
- **Calibration:** on the unseen takes the risk is underestimated (predicted 9.3%, observed 11.8%); the cost rule uses
  the conservative bound `r_up`, which partly makes up for it (see the guarantee above).
- **`lgbm_ABC`: error in feature B, confirmed (N38).** AUROC about 0.70–0.76, far below the other gates. The
  reliability feature on the training rows was computed by leaving out only the row's own session: that is a
  leave-one-out target encoding (with a fixed total, the value grows with the errors of the row's own session;
  correlation with the error 0.6–0.95 within class). The trees exploit it and learn a reversed rule; the logistic
  regression hardly does. The evaluation stays clean (no inflated result), but `lgbm_ABC` in this run **cannot be
  evaluated**. Fix (N38): reliability of the fit rows computed out of fold by session, as in the gate v1; on train
  LightGBM goes back to 0.93–0.94, on par with the other gates. Corrective run `n38_rel`, reported next to it (N39).

## 4. What changed between v1 and v2

| | v1 (test, 142 takes) | v2, all (1,271) | v2, unseen (337) |
|---|---|---|---|
| CamFormer alone | 0.915 | 0.924 | 0.882 |
| EgoVLP alone | 1.000 | 0.999 | 1.000 |
| Oracle, share sent to EgoVLP | 9.3% | 7.7% | 11.9% |
| **Same rule** (within 1 point), `thr_maxprob` | acc 1.000, share 28.3% | acc 0.990, share 19.0% | acc 0.985, share 23.5% |
| Cost rule β = 0.05, `thr_maxprob` | – | acc 0.997, share 29.7% | acc 0.995, share 34.4% |
| `lgbm_ABC`, within 1 point | acc 1.000, share 24.5% | cannot be evaluated (feature B error, N38) | cannot be evaluated |
| Per-class rule | acc 0.979, share 24.0% | acc 0.985, share 22.7% | acc 0.985, share 29.5% |
| AUROC `thr_maxprob` | 0.983 | 0.935 | 0.921 |
| RQ2 | null (p = 0.38) | null | null (p = 1.0) |
| CamFormer errors the evaluation rests on | 12 | about 96 per repeat | about 40 |

1. **With the same rule we use less EgoVLP (19–24% instead of 28%), but the accuracy drops to the 1-point limit
   (0.985–0.990 instead of 1.000).** The 1.000 of the v1 test was largely due to how small the test was.
2. **The new cost rule is more cautious:** 0.995–0.997 accuracy with 30–34% of EgoVLP, and a curve over β instead of a
   single point.
3. **The reference gate becomes `thr_maxprob`** (Prop. 2). On LightGBM run 772601 says nothing because of the feature B
   error (N38); the valid comparison comes from the corrective run `n38_rel` (N39).
4. **The threshold on p_max clearly beats the per-class rule** at ranking the errors (AUROC 0.92–0.94 against
   0.78–0.80). In v1 they looked almost equivalent.
5. **New sessions cost less than v1 said:** 30–34% instead of 50–58% (in v1 there were only 67 such takes).
6. **Much narrower intervals**, and an operating point no longer decided by a single session.

## 5. Limits to state in the report

- The v2 design was decided after seeing the v1 test: the evaluation is new, but the method is not blind.
- EgoVLP is practically perfect, so the gate only has to predict when CamFormer is wrong (Prop. 2); the 6-class task is
  easy for EgoVLP.
- CamFormer's pre-training saw 934 of the 1,271 takes: this is why the main result is on the 337 unseen ones.
- The guarantee of the risk rule assumes exchangeable takes, but takes of the same session are not.
- Cost measured only as GMACs of the two encoders: it includes neither video decoding nor trajectory estimation.
- Encoders pre-trained and frozen; only the heads and the gates are trained.
- A minor technical note: "EgoVLP alone" is 462.15 GMAC per take in v2 and 461.54 in v1 (a 0.13% difference, irrelevant
  for the conclusions).

## 6. Where the files are

All paths are under `results/`.

| | v1 | v2 |
|---|---|---|
| Folder | `first_evaluation_official_split/` and `features/official_v2/` | `nested_cv/preregistered_772601/` |
| Main tables | `50_gate/.../operating_point.csv`, `metrics.csv`, `comparisons.csv` | `metrics.csv`, `cost_vs_egovlp.csv`, `comparisons.csv`, `summary.json` |
| Per take | `features/official_v2/heads/*/predictions.csv` | `takes.csv.gz` |
| Decision log | N33, N34 | N37 (pre-registration), N38 (results and the two errors found after the run), N39 (corrective runs `n38_rel`, `n38_cal`) |
