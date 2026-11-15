"""Gli script di ``hpc/`` con un ``sbatch`` finto: sintassi, dipendenze, array, risorse (M11).

Gli script girano in una copia isolata (un repository git a parte), con un ``sbatch`` che registra
gli argomenti invece di sottomettere. Così si prova la logica di ``submit.sh`` e ``pipeline.sh`` in
locale e in CI, senza SLURM. ``stage.sbatch`` non viene eseguito: gira solo sul cluster.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, NamedTuple, Set

import pytest

from src import REPO_ROOT

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("git") is None, reason="servono bash e git"
)

FAKE_SBATCH = """#!/bin/bash
d="${FAKE_SBATCH_DIR:?}"
n=$(cat "$d/n" 2>/dev/null || echo 100)
n=$((n + 1))
echo "$n" > "$d/n"
{
  printf '%s\\t%s\\t%s\\t%s\\t' "$n" "${STAGE_SCRIPT:-}" "${STAGE_ARGS:-}" "${ARRAY_KIND:-}"
  printf '%s ' "$@"
  printf '\\n'
} >> "$d/calls.tsv"
echo "$n"
"""

ALL_STAGES = ["00", "01", "05", "10", "15", "20", "25", "27", "28", "30", "35", "40", "41", "45", "50", "52", "55"]
GPU_STAGES = {"20", "25", "28", "40", "41"}
# Script ancora da scrivere: il test lo ricorda quando compaiono (togliere da qui).
NOT_YET = {"55_report.py"}


class Call(NamedTuple):
    id: str
    script: str
    args: str
    array_kind: str
    opts: Dict[str, str]

    @property
    def stage(self) -> str:
        return self.opts["job-name"].rsplit("_", 1)[1]

    @property
    def deps(self) -> Set[str]:
        dep = self.opts.get("dependency")
        return set(dep.split(":")[1:]) if dep else set()


class Sandbox(NamedTuple):
    root: Path
    fake_dir: Path
    env: Dict[str, str]


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    root = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "hpc", root / "hpc", ignore=shutil.ignore_patterns("env.local.sh"))
    shutil.copytree(REPO_ROOT / "configs" / "hpc", root / "configs" / "hpc")
    (root / "splits" / "site_v1").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "splits" / "site_v1" / "SHA256SUMS", root / "splits" / "site_v1" / "SHA256SUMS")
    git = ["git", "-c", "user.email=t@example.org", "-c", "user.name=t"]
    subprocess.run(git + ["init", "-q"], cwd=root, check=True)
    subprocess.run(git + ["add", "-A"], cwd=root, check=True)
    subprocess.run(git + ["commit", "-q", "-m", "sandbox"], cwd=root, check=True)
    fake_dir = tmp_path / "fake"
    fake_dir.mkdir()
    (fake_dir / "bin").mkdir()
    sbatch = fake_dir / "bin" / "sbatch"
    sbatch.write_text(FAKE_SBATCH)
    sbatch.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "CASCADE_", "SBATCH_"))}
    for k in ("CPU_PARTITION", "GPU_PARTITION", "GPU_GRES", "MODULES", "TORCH_INDEX_URL", "DL4CV_ROOT", "EXPORT_DIR"):
        env.pop(k, None)
    for k in ("NUM_SHARDS", "MAX_REQUEUE", "SEEDS", "SPLIT_DIR", "RUN_NAME", "RUN_CONFIG", "STAGE_ARGS", "VENV"):
        env.pop(k, None)
    env.update(
        PATH=f"{fake_dir / 'bin'}{os.pathsep}{env.get('PATH', '')}",
        FAKE_SBATCH_DIR=str(fake_dir),
        DL4CV_ROOT=str(tmp_path / "dl4cv"),
        SLURM_ACCOUNT="acct",
        SLURM_QOS="stud",
        CPU_PARTITION="cpuq",
        GPU_PARTITION="gpuq",
        GPU_GRES="gpu:a:1",
        NUM_SHARDS="4",
        SEEDS="0 1",
        RUN_NAME="dbg_dry",
        RUN_CONFIG="site_v2",
        VENV=str(tmp_path / "no_venv"),
    )
    return Sandbox(root, fake_dir, env)


def run(sb: Sandbox, *cmd: str, **env: str) -> subprocess.CompletedProcess[str]:
    full = dict(sb.env, **env)
    return subprocess.run(["bash", *cmd], cwd=sb.root, env=full, capture_output=True, text=True)


def read_calls(sb: Sandbox) -> List[Call]:
    calls: List[Call] = []
    path = sb.fake_dir / "calls.tsv"
    if not path.exists():
        return calls
    for line in path.read_text().splitlines():
        cid, script, args, kind, rest = line.split("\t", 4)
        opts: Dict[str, str] = {}
        for tok in rest.split():
            if tok.startswith("--"):
                key, _, value = tok[2:].partition("=")
                opts[key] = value
        calls.append(Call(cid, script, args, kind, opts))
    return calls


EXPECTED_SCRIPTS = {
    "check_cluster.sh",
    "collect_results.sh",
    "env.sh",
    "final.sh",
    "pipeline.sh",
    "setup_env.sh",
    "submit.sh",
    "verify_run.sh",
}


def _bash_major() -> int:
    r = subprocess.run(["bash", "-c", 'echo "${BASH_VERSINFO[0]}"'], capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


def test_scripts_have_valid_syntax():
    names = {p.name for p in (REPO_ROOT / "hpc").glob("*.sh")}
    assert EXPECTED_SCRIPTS <= names, f"mancano: {sorted(EXPECTED_SCRIPTS - names)}"
    old_bash = _bash_major() < 4
    for script in sorted((REPO_ROOT / "hpc").glob("*.sh")) + [REPO_ROOT / "hpc" / "stage.sbatch"]:
        r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        if r.returncode != 0 and old_bash and "unexpected EOF" in r.stderr:
            continue  # la bash 3.2 del Mac sbaglia sugli heredoc con apostrofi dentro $( ): la CI (bash 5) li prova
        assert r.returncode == 0, f"{script.name}: {r.stderr}"


def test_submit_resources_and_options(sandbox):
    r = run(sandbox, "hpc/submit.sh", "20", "--", "--set", "camformer.seed=1")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "101"
    (call,) = read_calls(sandbox)
    assert call.script == "20_camformer_finetune.py"
    assert call.args == "--set camformer.seed=1"
    assert call.array_kind == "fold"
    o = call.opts
    assert o["partition"] == "gpuq" and o["gres"] == "gpu:a:1"
    assert o["account"] == "acct" and o["qos"] == "stud"
    assert o["array"] == "0-5" and o["cpus-per-task"] == "4" and o["mem"] == "16G" and o["time"] == "01:00:00"
    assert o["kill-on-invalid-dep"] == "yes" and "parsable" in o
    assert o["job-name"] == "cv2_dbg_dry_20"
    assert "dependency" not in o


def test_submit_cpu_stage_has_no_gres_and_shard_array(sandbox):
    assert run(sandbox, "hpc/submit.sh", "10").returncode == 0
    (call,) = read_calls(sandbox)
    assert call.opts["partition"] == "cpuq" and "gres" not in call.opts
    assert call.opts["array"] == "0-3" and call.array_kind == "shard"


def test_submit_dependency_and_overrides(sandbox):
    r = run(sandbox, "hpc/submit.sh", "50", "--after", "7:8", TIME_OVERRIDE="00:05:00", MEM_OVERRIDE="2G")
    assert r.returncode == 0, r.stderr
    (call,) = read_calls(sandbox)
    assert call.opts["dependency"] == "afterok:7:8"
    assert call.opts["time"] == "00:05:00" and call.opts["mem"] == "2G"


def test_submit_qos_is_optional(sandbox):
    assert run(sandbox, "hpc/submit.sh", "05", SLURM_QOS="").returncode == 0
    (call,) = read_calls(sandbox)
    assert "qos" not in call.opts and call.opts["account"] == "acct"


def test_submit_rejects_bad_input(sandbox):
    r = run(sandbox, "hpc/submit.sh", "99")
    assert r.returncode == 2 and "stadio sconosciuto" in r.stderr
    r = run(sandbox, "hpc/submit.sh", "05", RUN_CONFIG="non_esiste")
    assert r.returncode == 2 and "manca configs/hpc/non_esiste.yaml" in r.stderr
    r = run(sandbox, "hpc/submit.sh", "05", DL4CV_ROOT="")
    assert r.returncode == 2 and "DL4CV_ROOT" in r.stderr
    assert read_calls(sandbox) == []


def test_every_stage_points_to_an_existing_script(sandbox):
    seen = set()
    for stage in ALL_STAGES:
        assert run(sandbox, "hpc/submit.sh", stage).returncode == 0, stage
    for call in read_calls(sandbox):
        seen.add(call.script)
        exists = (REPO_ROOT / "scripts" / call.script).exists()
        if call.script in NOT_YET:
            assert not exists, f"{call.script} esiste: toglierlo da NOT_YET"
        else:
            assert exists, f"hpc/submit.sh punta a scripts/{call.script}, che non esiste"
    assert len(seen) == len(ALL_STAGES)


def test_pipeline_chain(sandbox):
    r = run(sandbox, "hpc/pipeline.sh")
    assert r.returncode == 0, r.stderr
    calls = read_calls(sandbox)
    assert len(calls) == 11 + 4 * 2 + 2  # 11 stadi comuni, 4 per seed, 52 e 55
    by: Dict[str, List[Call]] = {}
    for c in calls:
        by.setdefault(c.stage, []).append(c)
    one = {st: cs[0].id for st, cs in by.items() if len(cs) == 1}
    s20, s35, s45, s50 = ([c.id for c in by[st]] for st in ("20", "35", "45", "50"))
    assert all(c.opts["account"] == "acct" and c.opts["qos"] == "stud" for c in calls)
    assert {st for st, cs in by.items() if "gres" in cs[0].opts} == GPU_STAGES
    # stadi comuni
    assert by["01"][0].deps == set()
    assert by["00"][0].deps == {one["01"]}
    assert by["05"][0].deps == {one["00"]}
    assert by["10"][0].deps == {one["00"]}
    assert by["15"][0].deps == {one["01"]}
    assert by["25"][0].deps == {one["10"]}
    assert by["27"][0].deps == {one["25"], one["05"]}
    assert by["28"][0].deps == {one["10"], one["05"]}
    assert by["30"][0].deps == {one["25"], one["05"]}
    assert by["40"][0].deps == {one["00"]}
    assert by["41"][0].deps == {one["01"]}
    # per seed: 20 → 35 → 45 → 50
    for i, seed in enumerate(("0", "1")):
        assert by["20"][i].args == f"--set camformer.seed={seed}" and by["20"][i].opts["array"] == "0-5"
        assert by["20"][i].deps == {one["01"], one["05"]}
        assert by["35"][i].deps == {s20[i], one["30"]}
        assert by["45"][i].deps == {s35[i], one["40"], one["28"]}
        assert by["50"][i].deps == {s45[i]}
        assert by["50"][i].args == f"--set camformer.seed={seed}" and "--final" not in by["50"][i].args
    # aggregazione e report
    assert by["52"][0].deps == set(s50) and by["52"][0].args == "--seeds 0 1"
    assert by["55"][0].deps == {one["52"], one["27"], one["41"]}
    assert by["10"][0].opts["array"] == "0-3"


def test_pipeline_single_seed_skips_52(sandbox):
    r = run(sandbox, "hpc/pipeline.sh", SEEDS="0")
    assert r.returncode == 0, r.stderr
    calls = read_calls(sandbox)
    stages = [c.stage for c in calls]
    assert "52" not in stages and len(calls) == 11 + 4 + 1
    s50 = next(c.id for c in calls if c.stage == "50")
    s55 = next(c for c in calls if c.stage == "55")
    assert s50 in s55.deps and len(s55.deps) == 3


def test_verify_run_needs_a_run_dir(sandbox):
    r = run(sandbox, "hpc/verify_run.sh", "dbg_dry")
    assert r.returncode == 1 and "run_dir assente" in r.stdout


def test_check_cluster_fails_without_venv(sandbox):
    r = run(sandbox, "hpc/check_cluster.sh", "site_v2")
    assert r.returncode == 1
    assert "FALLITO  manca" in r.stdout and "no_venv" in r.stdout
    assert "check_cluster: CONTROLLI FALLITI" in r.stdout


def test_check_cluster_reports_missing_parameters(sandbox):
    r = run(sandbox, "hpc/check_cluster.sh", "site_v2", SLURM_ACCOUNT="", CPU_PARTITION="")
    assert r.returncode == 1
    assert "FALLITO  SLURM_ACCOUNT vuoto" in r.stdout and "FALLITO  CPU_PARTITION vuoto" in r.stdout
