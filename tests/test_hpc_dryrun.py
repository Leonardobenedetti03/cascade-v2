"""Gli script di ``hpc/`` con un ``sbatch`` finto: sintassi, dipendenze, array, risorse (M11).

Gli script girano in una copia isolata (un repository git a parte), con un ``sbatch`` che registra
gli argomenti invece di sottomettere. Così si prova la logica di ``submit.sh`` e ``pipeline.sh`` in
locale e in CI, senza SLURM. ``stage.sbatch`` gira con un ``python`` finto che registra gli stadi
lanciati (C14c-2, N27: ordine della catena, fold, stadi diagnostici, arresto al primo errore);
``verify_run.sh`` su una run_dir finta (strada A: 50-52 rifatti a un commit successivo).
"""

from __future__ import annotations

import ast
import os
import re
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
  printf '%s\\t%s\\t%s\\t%s\\t%s\\t' "$n" "${STAGE_SCRIPT:-}" "${STAGE_ARGS:-}" "${ARRAY_KIND:-}" "${STAGE_LIST:-}"
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
    stage_list: str
    opts: Dict[str, str]

    @property
    def chain(self) -> List[str]:
        """Elementi della catena (``submit.sh --chain``): "20 --set camformer.seed=1", "15?", ..."""
        return [x.strip() for x in self.stage_list.split(";") if x.strip()]

    @property
    def chain_stages(self) -> List[str]:
        return [x.split()[0].rstrip("?") for x in self.chain]

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
    shutil.copytree(REPO_ROOT / "splits" / "site_v1", root / "splits" / "site_v1")
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
        cid, script, args, kind, stage_list, rest = line.split("\t", 5)
        opts: Dict[str, str] = {}
        for tok in rest.split():
            if tok.startswith("--"):
                key, _, value = tok[2:].partition("=")
                opts[key] = value
        calls.append(Call(cid, script, args, kind, stage_list, opts))
    return calls


EXPECTED_SCRIPTS = {
    "check_cluster.sh",
    "collect_results.sh",
    "env.sh",
    "final.sh",
    "pipeline.sh",
    "setup_env.sh",
    "stages.sh",
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


def required_stages(script: str) -> Set[str]:
    """Numeri degli stadi di cui ``scripts/<script>`` legge l'output: i nomi di stadio che compaiono
    come stringhe nel sorgente (``ctx.require("00_inventory")``, anche dentro un ciclo), meno il
    proprio, più lo stadio 00 se lo script chiama ``ctx.read_clips()``."""
    src = (REPO_ROOT / "scripts" / script).read_text()
    own = script[:2]
    names = {
        n.value[:2]
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and re.fullmatch(r"\d\d_[a-z0-9_]+", n.value)
    }
    if "read_clips(" in src:
        names.add("00")
    return names - {own}


def chains(sb: Sandbox, *args: str, **env: str) -> Dict[str, Call]:
    r = run(sb, "hpc/pipeline.sh", *args, **env)
    assert r.returncode == 0, r.stderr
    return {c.stage: c for c in read_calls(sb)}


def test_pipeline_dependencies_cover_what_each_stage_reads(sandbox):
    """Le dipendenze di ``pipeline.sh`` si ricavano da ciò che gli script richiedono, non da una
    tabella copiata: ogni stadio deve venire dopo uno stadio di ciascuno di quelli di cui legge
    l'output, prima nello stesso job oppure in un job da cui il suo dipende (afterok)."""
    by_name = chains(sandbox)
    by_id = {c.id: c for c in by_name.values()}

    def ancestors(c: Call) -> Set[str]:
        seen: Set[str] = set()
        todo = list(c.deps)
        while todo:
            j = todo.pop()
            if j not in seen:
                seen.add(j)
                todo += list(by_id[j].deps)
        return seen

    submitted = {st for c in by_name.values() for st in c.chain_stages}
    for c in by_name.values():
        before = {st for j in ancestors(c) for st in by_id[j].chain_stages}
        for i, st in enumerate(c.chain_stages):
            script = (REPO_ROOT / "scripts").glob(f"{st}_*.py")
            script = next(script, None)
            if script is None:
                continue
            missing = (required_stages(script.name) & submitted) - before - set(c.chain_stages[:i])
            assert not missing, f"job {c.stage}: lo stadio {st} legge {sorted(missing)} ma non viene dopo"


def test_pipeline_is_four_jobs_within_the_qos(sandbox):
    """La QOS stud: 4 job in coda (ogni elemento di un array conta), 2 in esecuzione, 12 CPU e 2 GPU."""
    by = chains(sandbox, SEEDS="0 1 2")
    assert list(by) == ["A", "B1", "B2", "C"]
    assert all("array" not in c.opts and c.script == "" and c.array_kind == "none" for c in by.values())
    assert all(c.opts["account"] == "acct" and c.opts["qos"] == "stud" for c in by.values())
    assert {n for n, c in by.items() if "gres" in c.opts} == {"B1", "B2"}
    # B1 e B2 girano insieme: CPU e GPU sommate entro i limiti
    assert int(by["B1"].opts["cpus-per-task"]) + int(by["B2"].opts["cpus-per-task"]) <= 12
    assert all(int(c.opts["time"].split(":")[0]) <= 24 for c in by.values())


def test_pipeline_chain(sandbox):
    by = chains(sandbox, SEEDS="0 1 2")
    a, b1, b2, c = by["A"], by["B1"], by["B2"], by["C"]
    s = "--set camformer.seed="
    assert a.chain == ["01", "00", "05", "10", "15?"]  # 15 diagnostico: non ferma la catena
    assert b1.chain == ["25", "28", "40", "41", f"20 {s}0"]
    assert b2.chain == [f"20 {s}1", f"20 {s}2"]
    assert c.chain[:2] == ["27", "30"]
    assert c.chain[2:8] == [f"35 {s}0", f"45 {s}0", f"35 {s}1", f"45 {s}1", f"35 {s}2", f"45 {s}2"]
    assert c.chain[8:12] == [f"50 {s}0", f"50 {s}1", f"50 {s}2", "52 --seeds 0 1 2"]
    assert not any("--final" in x for x in c.chain)
    has55 = (REPO_ROOT / "scripts" / "55_report.py").exists()
    assert c.chain[12:] == (["55"] if has55 else [])
    assert a.deps == set() and b1.deps == {a.id} and b2.deps == {a.id} and c.deps == {b1.id, b2.id}
    # risorse: massimo di CPU e memoria, somma dei tempi (x6 le fold, x NUM_SHARDS=4 lo stadio 10)
    assert a.opts["partition"] == "cpuq" and a.opts["cpus-per-task"] == "4" and a.opts["mem"] == "8G"
    assert a.opts["time"] == "06:00:00"  # 1h + 0h30 + 0h10 + 4 x 1h + 0h20
    assert b1.opts["partition"] == "gpuq" and b1.opts["gres"] == "gpu:a:1"
    assert b1.opts["cpus-per-task"] == "8" and b1.opts["mem"] == "32G" and b1.opts["time"] == "08:30:00"
    assert b2.opts["time"] == "12:00:00" and c.opts["partition"] == "cpuq"
    assert {x.opts["job-name"] for x in by.values()} == {f"cv2_dbg_dry_{n}" for n in by}


def test_pipeline_single_seed_has_no_b2_and_no_52(sandbox):
    by = chains(sandbox, SEEDS="0")
    assert list(by) == ["A", "B1", "C"]
    assert by["C"].deps == {by["B1"].id}
    assert not any(x.startswith("52") for x in by["C"].chain)


def test_pipeline_redo_gate(sandbox):
    """Strada A: solo 50 (per seed) e 52 di nuovo, con --overwrite, in un job senza dipendenze."""
    by = chains(sandbox, "--redo-gate", SEEDS="0 1")
    (g,) = by.values()
    assert g.stage == "G" and g.deps == set()
    assert g.chain[:3] == [
        "50 --set camformer.seed=0 --overwrite",
        "50 --set camformer.seed=1 --overwrite",
        "52 --seeds 0 1 --overwrite",
    ]


def test_submit_chain_rejects_unknown_stage_and_extra_args(sandbox):
    r = run(sandbox, "hpc/submit.sh", "--chain", "X", "01; 99")
    assert r.returncode == 2 and "stadio sconosciuto: 99" in r.stderr
    r = run(sandbox, "hpc/submit.sh", "--chain", "X", "01", "--", "--overwrite")
    assert r.returncode == 2 and "dentro la catena" in r.stderr
    assert read_calls(sandbox) == []


def test_submit_no_requeue(sandbox):
    assert run(sandbox, "hpc/submit.sh", "--chain", "F", "50 --final", NO_REQUEUE="1").returncode == 0
    (call,) = read_calls(sandbox)
    assert "no-requeue" in call.opts and call.chain == ["50 --final"]


# --- stage.sbatch con un python finto -------------------------------------------------------------

FAKE_PYTHON = """#!/bin/bash
if [[ "$1" == -c ]]; then case "$2" in *run_dir*) echo "$FAKE_RUN_DIR" ;; esac; exit 0; fi
if [[ "$1" == -m ]]; then exit 0; fi
echo "$*" >> "$FAKE_SBATCH_DIR/py.log"
for f in ${FAKE_FAIL:-}; do [[ "$1" == *"$f"* ]] && exit 7; done
exit 0
"""

needs_sha256 = pytest.mark.skipif(shutil.which("sha256sum") is None, reason="serve sha256sum")


def run_stage(sb: Sandbox, **env: str):
    """Lancia ``hpc/stage.sbatch`` come farebbe SLURM, con un python finto; restituisce
    (risultato, stadi lanciati: una riga per esecuzione di python)."""
    (sb.root / "scripts").mkdir(exist_ok=True)
    for script in (REPO_ROOT / "scripts").glob("[0-9][0-9]_*.py"):  # stessi nomi, contenuto vuoto
        (sb.root / "scripts" / script.name).write_text("")
    git = ["git", "-c", "user.email=t@example.org", "-c", "user.name=t"]
    subprocess.run(git + ["add", "-A"], cwd=sb.root, check=True)
    subprocess.run(git + ["commit", "-q", "-m", "scripts", "--allow-empty"], cwd=sb.root, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=sb.root, capture_output=True, text=True).stdout.strip()
    py = sb.fake_dir / "bin" / "python"
    py.write_text(FAKE_PYTHON)
    py.chmod(0o755)
    venv = sb.fake_dir / "venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / "bin" / "activate").write_text("")
    full = dict(
        sb.env,
        SLURM_SUBMIT_DIR=str(sb.root),
        VENV=str(venv),
        EXPECTED_COMMIT=head,
        FAKE_RUN_DIR=str(sb.fake_dir / "run"),
        STAGE_SCRIPT="",
        ARRAY_KIND="none",
    )
    full.update(env)
    r = subprocess.run(["bash", "hpc/stage.sbatch"], cwd=sb.root, env=full, capture_output=True, text=True)
    log = sb.fake_dir / "py.log"
    lines = log.read_text().splitlines() if log.exists() else []
    return r, [ln.split(" --run-name")[0].replace(".py", "") + ln.split("site_v2.yaml")[1] for ln in lines]


@needs_sha256
def test_stage_chain_runs_stages_in_order_with_folds(sandbox):
    r, ran = run_stage(sandbox, STAGE_LIST="01; 10; 20 --set camformer.seed=1; 15?", CHAIN_NAME="T")
    assert r.returncode == 0, r.stdout + r.stderr
    folds = [
        f"scripts/20_camformer_finetune --set camformer.seed=1 --fold {f}" for f in ("0", "1", "2", "3", "4", "full")
    ]
    assert ran == [
        "scripts/01_convert_trajectories",
        "scripts/10_extract_frames",  # una sola esecuzione, senza --shard-id: tutte le clip
        *folds,
        "scripts/15_alignment_check",
    ]
    assert "catena T completata" in r.stdout


@needs_sha256
def test_stage_chain_diagnostic_failure_does_not_stop(sandbox):
    r, ran = run_stage(sandbox, STAGE_LIST="15?; 05", FAKE_FAIL="15_")
    assert r.returncode == 0, r.stdout + r.stderr
    assert ran == ["scripts/15_alignment_check", "scripts/05_folds"]
    assert "stadio diagnostico, la catena continua" in r.stdout


@needs_sha256
def test_stage_chain_stops_at_the_first_failure(sandbox):
    r, ran = run_stage(sandbox, STAGE_LIST="01; 00; 05", FAKE_FAIL="00_")
    assert r.returncode == 7
    assert ran == ["scripts/01_convert_trajectories", "scripts/00_inventory"]
    assert "interrotta a 00_inventory.py" in r.stdout


@needs_sha256
def test_stage_chain_skips_missing_optional_script(sandbox):
    r, ran = run_stage(sandbox, STAGE_LIST="05; 55?")
    if (REPO_ROOT / "scripts" / "55_report.py").exists():
        pytest.skip("55_report.py esiste")
    assert r.returncode == 0, r.stdout + r.stderr
    assert ran == ["scripts/05_folds"] and "non esiste, salto" in r.stdout


@needs_sha256
def test_stage_single_fold_task(sandbox):
    r, ran = run_stage(
        sandbox,
        STAGE_LIST="",
        STAGE_SCRIPT="20_camformer_finetune.py",
        ARRAY_KIND="fold",
        STAGE_ARGS="--set camformer.seed=2",
        SLURM_ARRAY_TASK_ID="5",
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert ran == ["scripts/20_camformer_finetune --set camformer.seed=2 --fold full"]


@needs_sha256
def test_stage_rejects_wrong_commit(sandbox):
    r, ran = run_stage(sandbox, STAGE_LIST="01", EXPECTED_COMMIT="0" * 40)
    assert r.returncode == 3 and ran == []


# --- verify_run.sh: commit dei manifest (strada A) -----------------------------------------------


def make_run(sb: Sandbox, name: str, up: str, down: str = "") -> Path:
    """run_dir finta e completa: manifest di 00-45 al commit ``up`` e di 50-52 a ``down``."""
    import json

    rd = Path(sb.env["DL4CV_ROOT"]) / "runs" / name
    (rd / "_manifest").mkdir(parents=True)
    for st in ("00_inventory", "05_folds", "20_camformer.fold0", "45_cache"):
        (rd / "_manifest" / f"{st}.json").write_text(json.dumps({"commit": up}))
    if down:
        for st in ("50_gate.seed0_hz20.main", "52_seeds.hz20.main"):
            (rd / "_manifest" / f"{st}.json").write_text(json.dumps({"commit": down}))
    (rd / "data").mkdir()
    (rd / "data" / "clips.csv").write_text("take_uid,split,has_video,has_traj\nt1,train,True,True\n")
    (rd / "logs").mkdir()
    for s in ("0", "1"):
        for f in ("fold0", "fold1", "fold2", "fold3", "fold4", "full"):
            d = rd / "20_camformer" / f"seed{s}_hz20" / f
            d.mkdir(parents=True)
            (d / "summary.json").write_text("{}")
    return rd


def commit(sb: Sandbox, path: str, text: str) -> str:
    f = sb.root / path
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    git = ["git", "-c", "user.email=t@example.org", "-c", "user.name=t"]
    subprocess.run(git + ["add", "-A"], cwd=sb.root, check=True)
    subprocess.run(git + ["commit", "-q", "-m", path], cwd=sb.root, check=True)
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=sb.root, capture_output=True, text=True)
    return out.stdout.strip()


@needs_sha256
def test_verify_run_one_commit(sandbox):
    c = commit(sandbox, "scripts/00_inventory.py", "a")
    make_run(sandbox, "dbg_v", c, c)
    r = run(sandbox, "hpc/verify_run.sh", "dbg_v")
    assert r.returncode == 0, r.stdout
    assert f"ok       commit {c}" in r.stdout


@needs_sha256
def test_verify_run_accepts_gate_redone_later(sandbox):
    up = commit(sandbox, "scripts/00_inventory.py", "a")
    commit(sandbox, "scripts/50_gate_eval.py", "nuovo gate")
    down = commit(sandbox, "src/cascade/eval/gate_eval.py", "nuovo")
    make_run(sandbox, "dbg_v", up, down)
    r = run(sandbox, "hpc/verify_run.sh", "dbg_v")
    assert r.returncode == 0, r.stdout
    assert f"strada A: 00-45 al commit {up}, 50-52 al commit {down}" in r.stdout
    assert "src/cascade/eval/gate_eval.py" in r.stdout  # elencato da controllare


@needs_sha256
def test_verify_run_rejects_upstream_script_changed(sandbox):
    up = commit(sandbox, "scripts/00_inventory.py", "a")
    down = commit(sandbox, "scripts/20_camformer_finetune.py", "cambiato")
    make_run(sandbox, "dbg_v", up, down)
    r = run(sandbox, "hpc/verify_run.sh", "dbg_v")
    assert r.returncode == 1
    assert "scripts/20_camformer_finetune.py" in r.stdout and "FALLITO" in r.stdout


@needs_sha256
def test_verify_run_rejects_gate_not_at_head(sandbox):
    up = commit(sandbox, "scripts/00_inventory.py", "a")
    down = commit(sandbox, "scripts/50_gate_eval.py", "b")
    commit(sandbox, "README.md", "dopo")
    make_run(sandbox, "dbg_v", up, down)
    r = run(sandbox, "hpc/verify_run.sh", "dbg_v")
    assert r.returncode == 1 and f"manifest di 50-52 al commit {down}" in r.stdout


@needs_sha256
def test_verify_run_rejects_mixed_upstream(sandbox):
    a = commit(sandbox, "scripts/00_inventory.py", "a")
    rd = make_run(sandbox, "dbg_v", a)
    b = commit(sandbox, "README.md", "b")
    (rd / "_manifest" / "30_l1_heads.json").write_text(f'{{"commit": "{b}"}}')
    r = run(sandbox, "hpc/verify_run.sh", "dbg_v")
    assert r.returncode == 1 and "commit diversi nei manifest di 00-45" in r.stdout


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
