"""Stadio 44 (ponte da scenario_preds alla cache del gate, N28): schema, take senza L0, classi fuori
dalla testa, costi per finestra, test sigillato fino a --include-test, e catena 44 -> 50 (anche --final)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cascade.labels import load_labels

ROOT = Path(__file__).resolve().parents[1]
COST = ["cost.latency_reps=1", "cost.timesformer_frames=[4]"]  # stadio 40 veloce: stesse chiavi in 44 e 50
GATE = [
    "gate.models=[always_L0,always_L1,random,oracle,thr_maxprob,logreg_AC,compare_A,compare_AC]",
    "gate.bootstrap=15",
    "gate.random_seeds=2",
    "gate.seeds=[0]",
    "gate.comparisons=[[compare_AC,compare_A]]",
]


def _script(name):
    spec = importlib.util.spec_from_file_location("stage_" + name.replace(".py", ""), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_scenario_run(d: Path, n_windows: int = 3, seed: int = 0) -> Path:
    """Run di scenario_preds finta: L0 buono sulle classi fisiche e debole sulle procedurali, L1 ~95%."""
    rng = np.random.default_rng(seed)
    names = list(load_labels().names)
    proc = load_labels().procedural_mask
    rows, cam, vid = [], {}, {}
    (d / "traj").mkdir(parents=True)
    (d / "sealed").mkdir()
    (d / "features").mkdir()
    (d / "index").mkdir()
    (d / "configs").mkdir()
    i = 0
    for split, n_per in (("train", 40), ("val", 10), ("test", 10)):
        for c, name in enumerate(names + ["Bouldering"]):
            for j in range(n_per if name != "Bouldering" else 1):
                take = f"{split}_{name.replace(' ', '')}_{j}"
                in_head = name in names
                acc0 = 0.55 if in_head and proc[c] else 0.95
                p0 = rng.dirichlet(np.ones(6))
                p1 = rng.dirichlet(np.ones(6))
                if in_head:
                    k0 = c if rng.random() < acc0 else rng.integers(6)
                    k1 = c if rng.random() < 0.95 else rng.integers(6)
                    p0[k0] += 2.0
                    p1[k1] += 2.0
                p0, p1 = p0 / p0.sum(), p1 / p1.sum()
                row = {
                    "take_uid": f"uid{i}",
                    "take_name": take,
                    "site": "s",
                    # metà delle sessioni (j pari) è in comune tra train e val/test
                    "session": f"{name}_{j}" if j % 2 == 0 else f"{take}_sess",
                    "split": split,
                    "fold": (i % 5) if split == "train" else -1,
                    "fold_s0": (i % 5) if split == "train" else -1,
                    "fold_s1": ((i + 1) % 5) if split == "train" else -1,
                    "label": name,
                    "label_idx": c if in_head else -1,
                    "in_head": in_head,
                    "camformer_pretrain": bool(i % 2),
                }
                no_l0 = split == "val" and name == "Cooking" and j == 0  # una take senza embedding CamFormer
                for k, n in enumerate(names):
                    row[f"cam_p_{n}"] = np.nan if no_l0 else p0[k]
                    row[f"vid_p_{n}"] = p1[k]
                rows.append(row)
                cam[take] = np.full(512, np.nan) if no_l0 else rng.normal(size=512)
                vid[take] = rng.normal(size=768)
                t = np.arange(0, 30, 0.05)
                amp = 0.8 if not proc[min(c, 5)] else 0.1
                pos = np.stack([amp * np.sin(t), 0.1 * np.cos(t), 1.6 + 0.05 * np.sin(2 * t)], 1)
                q = np.tile([0, 0, 0, 1.0], (len(t), 1))
                np.savez(
                    d / "traj" / f"{take}.npz", t=t, poses=np.concatenate([pos, q], 1), valid=np.ones(len(t), bool)
                )
                i += 1
    df = pd.DataFrame(rows)
    df[df.split != "test"].to_csv(d / "predictions.csv", index=False)
    df[df.split == "test"].to_csv(d / "sealed" / "predictions.csv", index=False)
    df.assign(duration_s=30.0)[["take_uid", "take_name", "duration_s", "session", "camformer_pretrain"]].to_csv(
        d / "index" / "takes.csv", index=False
    )
    for m, e in (("camformer", cam), ("egovlp", vid)):
        np.savez(d / "features" / f"{m}.npz", take_name=np.array(list(e)), X=np.stack(list(e.values())))
    cfg = {
        "labels": {"classes": names},
        "window": {"length_s": 8.0, "n_windows": n_windows, "placement": "center"},
        "poses": {"fps": 20.0, "json_cam_to_device": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
    }
    (d / "configs" / "heads.json").write_text(json.dumps(cfg))
    return d


@pytest.fixture()
def env(tmp_path, monkeypatch):
    (tmp_path / "root").mkdir()
    monkeypatch.setenv("DL4CV_ROOT", str(tmp_path / "root"))
    monkeypatch.delenv("CASCADE_DATA_ROOT", raising=False)
    monkeypatch.delenv("CASCADE_RUN_ROOT", raising=False)
    from src.cascade.paths import run_dir

    scen = fake_scenario_run(tmp_path / "scen")
    _script("40_cost.py").main(_argv())
    return run_dir("b44"), scen


def _argv(*extra, gate=False):
    out = ["--run-name", "b44", *extra]
    for o in COST + (GATE if gate else []):
        out += ["--set", o]
    return out


def _gate_argv(*extra):
    return _argv(*extra, gate=True)


def test_bridge_cache_and_gate(env):
    from src.cascade.cache.schema import validate_cache
    from src.data.cache import read_table

    rd, scen = env
    _script("44_scenario_bridge.py").main(_argv("--scenario-run", str(scen)))
    (vdir,) = list((rd / "45_cache").glob("seed*_hz*"))
    t = read_table(vdir / "cache.parquet")
    validate_cache(t, load_labels(), cam_dim=512, vid_dim=768)
    assert set(t.df.split) == {"train", "val"}  # test sigillato
    assert "val_Cooking_0" not in set(t.df.clip_id)  # senza L0: fuori dalla cache...
    no_l0 = pd.read_csv(vdir / "no_l0.csv")
    assert list(no_l0.take_name) == ["val_Cooking_0"] and set(no_l0.c1) <= {0, 1}  # ...ma elencata
    assert not t.df.clip_id.str.contains("Bouldering").any()
    cost = json.loads((rd / "40_cost" / "hz20" / "cost.json").read_text())
    assert np.allclose(t.df.gmacs_L0, 3 * cost["gmacs_L0"]) and np.allclose(t.df.gmacs_L1, 3 * cost["gmacs_L1"])
    assert t.df[[c for c in t.df.columns if c.startswith("mot_")]].notna().all().all()
    assert np.allclose(t.arrays["fused_probs"], t.arrays["vid_probs_cal"]) and (t.df.w == 0).all()
    tr = t.df.split == "train"
    assert (t.df.fold[tr].between(0, 4)).all() and (t.df.fold[~tr] == -1).all()
    summ = json.loads((vdir / "summary.json").read_text())
    assert summ["splits"]["val"]["n_no_l0"] == 1 and summ["counts"]["n_out_of_head"] == 2  # Bouldering train+val
    _script("50_gate_eval.py").main(_gate_argv())
    (gdir,) = list((rd / "50_gate").glob("seed*_hz*/main"))
    op = pd.read_csv(gdir / "operating_point.csv")
    assert "test" not in set(op.split)
    ref = op[(op.point == "budget") & (op.model == "always_L1")].acc.iloc[0]
    sel = op[(op.point == "budget") & op.model.isin(["thr_maxprob", "logreg_AC", "compare_AC"])]
    assert (sel.acc >= ref - 0.01 - 1e-9).all()
    # passo finale: il test entra solo con --include-test, poi 50 --final
    _script("44_scenario_bridge.py").main(_argv("--scenario-run", str(scen), "--include-test", "--overwrite"))
    assert "test" in set(read_table(vdir / "cache.parquet").df.split)
    _script("50_gate_eval.py").main(_gate_argv("--final"))
    assert (gdir / "FINAL_DONE").exists()
    assert "test" in set(pd.read_csv(gdir / "operating_point.csv").split)
    _script("53_scenario_analysis.py").main(_argv("--final", "--n-boot", "50"))
    assert "test" in set(pd.read_csv(rd / "53_analysis" / vdir.name / "main" / "op_all.csv").split)


def test_analysis_53(env):
    """N32: take senza L0 contate come L1, sessioni in comune, bootstrap per sessione, regola procedurale, moto."""
    rd, scen = env
    _script("44_scenario_bridge.py").main(_argv("--scenario-run", str(scen)))
    _script("50_gate_eval.py").main(_gate_argv())
    with pytest.raises(RuntimeError, match="FINAL_DONE"):
        _script("53_scenario_analysis.py").main(_argv("--final"))
    _script("53_scenario_analysis.py").main(_argv("--n-boot", "50"))
    (out,) = list((rd / "53_analysis").glob("seed*_hz*/main"))
    op = pd.read_csv(out / "op_all.csv")
    assert set(op.split) == {"val"} and {"always_L1", "rule_procedural", "logreg_AC"} <= set(op.model)
    l1 = op[op.model == "always_L1"].iloc[0]
    assert l1.n_no_l0 == 1 and l1.n_takes == 60 and l1.cost_frac_L1 == pytest.approx(1.0)  # 59 in cache + 1 senza L0
    l0 = op[op.model == "always_L0"].iloc[0]
    assert l0.escalation_rate == pytest.approx(1 / 60)  # solo la take senza L0 va a L1
    nosh = pd.read_csv(out / "op_no_shared.csv")
    assert set(nosh.n_takes) == {30}  # metà delle take di val ha una sessione propria
    boot = pd.read_csv(out / "op_boot.csv")
    assert {"acc", "cost_frac_L1", "acc_minus_always_L1"} == set(boot.metric) and (boot.lo <= boot.hi).all()
    rule = op[op.model == "rule_procedural"].iloc[0]
    assert 0.4 < rule.escalation_rate < 0.7  # circa le take previste procedurali
    pre = pd.read_csv(out / "by_pretrain.csv")
    assert set(pre.subset) == {"camformer_pretrain=True", "camformer_pretrain=False"}
    mot = pd.read_csv(out / "motion_l0.csv")
    assert set(mot.quintile) == {1, 2, 3, 4, 5} and (out / "motion_l0.png").exists()
    summ = json.loads((out / "summary.json").read_text())
    assert summ["n_no_l0"] == {"val": 1} and summ["n_takes_shared"]["val"] == 30


def test_bridge_rejects_wrong_classes(env):
    rd, scen = env
    cfg = json.loads((scen / "configs" / "heads.json").read_text())
    cfg["labels"]["classes"] = cfg["labels"]["classes"][::-1]
    (scen / "configs" / "heads.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="classi"):
        _script("44_scenario_bridge.py").main(_argv("--scenario-run", str(scen)))


def test_bridge_refuses_test_rows_in_predictions(env):
    rd, scen = env
    p = pd.read_csv(scen / "predictions.csv")
    pd.concat([p, pd.read_csv(scen / "sealed" / "predictions.csv")]).to_csv(scen / "predictions.csv", index=False)
    with pytest.raises(RuntimeError, match="test"):
        _script("44_scenario_bridge.py").main(_argv("--scenario-run", str(scen)))
