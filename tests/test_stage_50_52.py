"""Stadi 50 (gate e valutazione) e 52 (aggregazione sui seed): output, semantica di ``--final``,
tracciabilità. Girano sulla cache finta (``--mock``) e su CSV costruiti a mano."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from src.cascade.eval.gate_eval import OUTPUT_TABLES

ROOT = Path(__file__).resolve().parents[1]
SMALL = [  # stessi override di conftest: il dataset sintetico della sessione è uno solo
    "synthetic.takes_per_class={train: 5, val: 2, test: 2}",
    "synthetic.duration_s=[6.0, 8.0]",
    "synthetic.video_size=32",
    "synthetic.traj_hz=50.0",
]
GATE = [
    "gate.models=[thr_maxprob,logreg_A,compare_A,compare_AC]",
    "gate.bootstrap=15",
    "gate.random_seeds=2",
    "gate.seeds=[0]",
    "gate.comparisons=[[compare_A,thr_maxprob]]",
]


def _script(name):
    spec = importlib.util.spec_from_file_location("stage_" + name.replace(".py", ""), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _argv(run, *extra, tag="main"):
    out = ["--run-name", run, "--synthetic", "--tag", tag, *extra]
    for o in SMALL + GATE:
        out += ["--set", o]
    return out


def _env(monkeypatch, syn_root):
    monkeypatch.setenv("DL4CV_ROOT", str(syn_root))
    monkeypatch.delenv("CASCADE_DATA_ROOT", raising=False)
    monkeypatch.delenv("CASCADE_RUN_ROOT", raising=False)
    from src.cascade.paths import run_dir

    return run_dir


def _out(run_dir, run, tag="main"):
    (d,) = list((run_dir(run) / "50_gate").glob(f"seed*_hz*/{tag}"))
    return d


MOCK = ["--mock", "--mock-sizes", "500", "150", "150"]


def test_stage_50_without_final(syn_root, monkeypatch):
    run_dir = _env(monkeypatch, syn_root)
    _script("50_gate_eval.py").main(_argv("t50a", *MOCK))
    d = _out(run_dir, "t50a")
    for k in OUTPUT_TABLES:
        assert (d / f"{k}.csv").exists(), k
    assert (d / "scores.parquet").exists() and (d / "config.yaml").exists()
    assert not (d / "FINAL_DONE").exists() and not (run_dir("t50a") / "50_gate" / "FINAL_RUNS.log").exists()
    for k in ("metrics", "curves", "operating_point", "comparisons"):
        assert "test" not in set(pd.read_csv(d / f"{k}.csv").split), k
    assert "test" not in set(pd.read_parquet(d / "scores.parquet").split)
    op = pd.read_csv(d / "operating_point.csv")
    b = op[(op.split == "val") & (op.point == "budget") & ~op.model.isin(["always_L1"])]
    assert (b.cost_frac_L1 <= 0.5 + 1e-6).all()
    _script("50_gate_eval.py").main(_argv("t50a", *MOCK))  # secondo giro senza --final: salta


def test_stage_50_final_logs_every_run_and_locks(syn_root, monkeypatch):
    run_dir = _env(monkeypatch, syn_root)
    mod = _script("50_gate_eval.py")
    mod.main(_argv("t50b", *MOCK, "--final"))
    mod.main(_argv("t50b", *MOCK, "--final"))  # --final rigira sempre e lascia una riga per esecuzione
    d = _out(run_dir, "t50b")
    lines = (run_dir("t50b") / "50_gate" / "FINAL_RUNS.log").read_text().strip().split("\n")
    assert len(lines) == 2
    f = lines[0].split("\t")
    assert len(f) == 7 and f[1].startswith("commit=") and f[3].startswith("cache_sha=")
    assert f[4].startswith("variant=seed") and f[5] == "tag=main"  # colonne 5 e 6 per verify_run.sh
    assert (d / "FINAL_DONE").exists()
    assert "test" in set(pd.read_csv(d / "metrics.csv").split)
    assert len(pd.read_csv(d / "threshold_transfer.csv")) > 0
    with pytest.raises(RuntimeError, match="FINAL_DONE"):  # niente rilancio senza --final sopra una finale
        mod.main(_argv("t50b", *MOCK))
    mod.main(_argv("t50b", *MOCK, tag="abl"))  # altro tag: libero
    assert not (_out(run_dir, "t50b", "abl") / "FINAL_DONE").exists()


def _fake_50(run_dir, run, seed, shift, traj):
    d = run_dir(run) / "50_gate" / f"seed{seed}_{traj}" / "x"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "split": ["val", "val"],
            "model": ["compare_A", "oracle"],
            "metric": ["area", "area"],
            "value": [0.90 + shift, 0.95],
            "ci_low": [0.85, 0.9],
            "ci_high": [0.95, 0.99],
        }
    ).to_csv(d / "metrics.csv", index=False)
    pd.DataFrame(
        {"split": ["val"], "a": ["compare_A"], "b": ["thr_maxprob"], "metric": ["area"], "diff": [0.01 + shift]}
    ).to_csv(d / "comparisons.csv", index=False)
    pd.DataFrame(
        {
            "split": ["test"],
            "model": ["compare_A"],
            "budget_frac": [0.5],
            "cost_over_budget": [0.02 + shift],
            "acc_transfer": [0.9],
            "acc_reopt": [0.91],
            "acc_gap": [-0.01],
            "bacc_gap": [-0.02],
        }
    ).to_csv(d / "threshold_transfer.csv", index=False)
    pd.DataFrame({"split": ["val"], "model": ["compare_A"], "auroc": [0.6 + shift]}).to_csv(
        d / "diag_c1.csv", index=False
    )


def test_stage_52_aggregates_seeds(syn_root, syn_ctx, monkeypatch):
    run_dir = _env(monkeypatch, syn_root)
    traj = syn_ctx.traj_tag
    for seed, shift in ((0, 0.0), (1, 0.02), (2, 0.04)):
        _fake_50(run_dir, "t52", seed, shift, traj)
    mod = _script("52_aggregate_seeds.py")
    argv = ["--run-name", "t52", "--synthetic", "--tag", "x"]
    for o in SMALL:
        argv += ["--set", o]
    mod.main(argv)
    out = run_dir("t52") / "52_seeds" / traj / "x"
    m = pd.read_csv(out / "metrics_seeds.csv").set_index(["split", "model", "metric"])
    assert m.loc[("val", "compare_A", "area"), "n_seeds"] == 3
    assert m.loc[("val", "compare_A", "area"), "mean"] == pytest.approx(0.92)
    assert m.loc[("val", "oracle", "area"), "std"] == pytest.approx(0.0)
    for name in ("comparisons_seeds", "threshold_transfer_seeds", "diag_c1_seeds"):
        assert (out / f"{name}.csv").exists(), name
    t = pd.read_csv(out / "threshold_transfer_seeds.csv")
    assert t.cost_over_budget_mean.iloc[0] == pytest.approx(0.04)
    with pytest.raises(RuntimeError, match="almeno 2 seed"):
        mod.main(argv + ["--seeds", "0"])


def test_aggregate_handles_missing_tables():
    mod = _script("52_aggregate_seeds.py")
    a = pd.DataFrame({"split": ["val"], "model": ["m"], "metric": ["area"], "value": [0.5], "ci_low": [0.4]})
    out = mod.aggregate([a, a.assign(value=0.7)], ["split", "model", "metric"], "value")
    assert out["mean"].iloc[0] == pytest.approx(0.6) and out["n_seeds"].iloc[0] == 2
