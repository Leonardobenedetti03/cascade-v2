# HPC runbook

How we ran the two GPU/CPU-heavy steps on the Bocconi cluster (SLURM). The commands below are the ones in `hpc/`.
Steps 2 to 4 of the README do not need the cluster: they run on a laptop from the files in `results/`.

Personal values (account, user paths) never go in git. Each settings file reads a `.local.sh` twin first, which git
ignores, and keeps every variable already set there:

- `hpc/env.sh` (read by `hpc/setup_env.sh` and `hpc/nested_cv.sh`) → `hpc/env.local.sh`
- `hpc/features_env.sh` (read by `hpc/features_submit.sh` and `hpc/input_variants.sh`) → `hpc/features_env.local.sh`

## The cluster

Values checked on the login node on 15/11 with `sacctmgr`, `sinfo` and `sbatch --test-only` (decision log, N23):

- partition and QOS `stud`, for both CPU and GPU;
- limits of the `stud` QOS: 1 day per job, 12 CPUs and 2 GPUs per user, 2 jobs running and **4 in the queue** (every
  element of a job array counts as one job);
- GPUs are MIG slices (`3g.40gb`, `4g.40gb`, `7g.80gb`); `--gres=gpu:1` takes the first free one;
- system Python 3.9, no modules needed; the CUDA runtime comes with the torch wheel (`cu121` index);
- the login node can reach PyPI, PyTorch and GitHub, the compute nodes are not assumed to.

The runs in `results/` were made from Giorgio's account, because Leo's home was almost full (N25). The dataset stayed
in Leo's and Jacopo's homes and was only read from there.

## 1. First login (once)

1. **Clone.** The cluster has its own deploy key for this repository. Clone with the submodules:

   ```bash
   git clone --recurse-submodules <url-of-cascade-v2> ~/cascade-v2
   cd ~/cascade-v2
   ```

   We changed code only on the Mac, with green CI, and then ran `git pull --ff-only` on the cluster.

2. **Settings.** Create `hpc/env.local.sh`, for example:

   ```bash
   export SLURM_ACCOUNT="<your account>"
   export SLURM_QOS="stud"
   export CPU_PARTITION="stud"
   export GPU_PARTITION="stud"
   export GPU_GRES="gpu:1"
   export MODULES=""
   export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
   export DL4CV_ROOT="$HOME/cascade_v2_root"
   ```

   and, if the defaults in `hpc/features_env.sh` do not fit your account, `hpc/features_env.local.sh` with
   `SLURM_ACCOUNT`, `CAMFORMER_REPO`, `CAMFORMER_CKPT`, `EGOVLP_CKPT` and `SCENARIO_RUN_DIR`.

3. **Environment.** From the login node, because it needs the network:

   ```bash
   hpc/setup_env.sh
   ```

   It creates the venv in `$VENV` (default `~/venvs/cascade-v2`), installs torch and `env/requirements.txt`, prints
   the versions and runs the test suite. It refuses to run if the venv already exists. No job installs anything.

4. **Data and weights.**
   - Ego-Exo4D subset: two folders, each with `split_manifest.csv`, `annotations/` and `takes/`. Their paths go in
     `data.roots` of `configs/features.yaml`.
   - CamFormer: the code is the submodule `third_party/seeing_without_pixels` (or any clone, via `CAMFORMER_REPO`);
     the checkpoint `egoexo4d_dur8.pt` comes from `camformer_checkpoints.zip`, linked in its `DATA.md`. We downloaded
     it on the Mac and copied it with `scp`.
   - EgoVLP: `EgoVLP_PT_BEST` from the EgoVLP repository, in `EGOVLP_CKPT`.

## 2. Features and heads (step 1 of the README)

First we chose the pose convention and the number of windows on the validation split. One GPU job runs 6 candidate
runs (identity or diag(-1,-1,1) poses, times 1, 3 or 5 windows), each up to the CamFormer heads, and then the
`select` stage:

```bash
RUN_NAME=official_v2 hpc/input_variants.sh
```

The choice is written to `$HOME/scenario_runs/official_v2_cand/select/selection.json` (ours: identity, 5 windows;
copied to `results/features/input_variant_selection/`). Then the real run, as 3 chained jobs (index and
trajectories on CPU, the two encoders on GPU, the heads on CPU):

```bash
RUN_NAME=official_v2 hpc/features_submit.sh \
    --set 'poses.json_cam_to_device=[[1,0,0],[0,1,0],[0,0,1]]' --set 'window.n_windows=5'
```

The heads job exits with code 3 if the pre-registered check on CamFormer fails, and then nothing downstream starts.
At the very end, once, the predictions on the official test split (we ran this as job 770706, N34):

```bash
python scripts/01_extract_features.py heads --eval-test \
    --set 'poses.json_cam_to_device=[[1,0,0],[0,1,0],[0,0,1]]' --set 'window.n_windows=5'
```

The run folder is `$SCENARIO_RUN_DIR` (default `$HOME/scenario_runs/$RUN_NAME`).

## 3. Compute cost (step 2)

`scripts/02_count_gmac.py` needs no data. We ran it on the cluster as stage 40, inside an earlier CPU job of the first
evaluation (decision log, N28), and copied the values it gave into `gmacs_per_window` of `configs/gate_v2.yaml`
(N37). It can also run on a laptop, with small differences due to the torch and fvcore versions (see the README).

## 4. Nested cross-validation (step 3)

One CPU job with 8 processes, about an hour per run:

```bash
SCEN_RUN=$HOME/scenario_runs/official_v2 RUN_NAME=gate_v2 \
    TAG=n38_cal CONFIG=configs/gate_v2_n38_cal.yaml hpc/nested_cv.sh
```

`TAG` is the output subfolder (default `main`) and `CONFIG` an extra config file on top of `configs/gate_v2.yaml`.
The three runs in `results/nested_cv/` are `TAG=main` with no `CONFIG` (job 772601), `TAG=n38_rel` with
`configs/gate_v2_n38_rel.yaml` (job 773281) and `TAG=n38_cal` with `configs/gate_v2_n38_cal.yaml` (job 773282, the
run in the report). Before submitting, the script checks that the feature run has
`$SCEN_RUN/index/takes.csv`, `$SCEN_RUN/features/*.npz` and `$SCEN_RUN/configs/heads.json`.

Outputs go to `$CASCADE_RUN_ROOT/gate_v2/56_gate_v2/<TAG>/`. We copied that folder and the SLURM log into
`results/nested_cv/` without changing anything.

## When a job fails

- **Rerunning is safe.** Every stage writes a completion marker last. A stage that is already complete with the same
  config is skipped; one with a different config stops and asks for `--overwrite` (feature pipeline) or a new `TAG`
  (nested CV), so outputs of different configs never mix.
- **Exit code 3** from the heads job: the pre-registered check failed. Read `check.json` in the run folder.
- **Logs:** `logs/slurm/` in the clone (one file per job) and `$SCENARIO_RUN_DIR/logs/` for the feature pipeline.
- **Queue and resources:** `squeue -u $USER`, and `sacct -j <job id>` for time and memory after the end.
