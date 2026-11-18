"""Gate v2 (N37, stadio 56): regole, calibratore, fold per sessione, feature senza fughe, CV annidata senza fughe
di informazione, catena completa su una run finta di scenario_preds; correzioni del gate v1 (sessioni)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cascade.config import load_config
from src.cascade.gatev2.calibrate import Calibrator, ece
from src.cascade.gatev2.folds import check_grouping, session_folds
from src.cascade.gatev2.rules import (
    acc_target_threshold,
    budget_threshold,
    clopper_pearson_upper,
    escalate,
    risk_threshold,
)
from src.cascade.gatev2.scorers import FeatureBuilder, make_scorer

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["Basketball", "Dance", "Soccer", "Cooking", "Health", "Bike repair"]
PROC = np.array([False, False, False, True, True, True])
FAST = [
    "gate_v2.repeats=[0]",
    "gate_v2.outer_folds=3",
    "gate_v2.inner_folds=3",
    "gate_v2.gate_folds=3",
    "gate_v2.heads.Cs=[1.0]",
    "gate_v2.heads.inner_cv=2",
    "gate_v2.logreg_Cs=[0.1,1.0]",
    "gate_v2.logreg_inner_cv=2",
    "gate_v2.bootstrap=30",
    "gate_v2.calibration_boot=10",
    "gate_v2.scorers=[thr_maxprob,logreg_AC,logreg_ABC,class_rule]",
    "gate_v2.comparisons=[[logreg_AC,thr_maxprob],[thr_maxprob,class_rule]]",
]


def fake_scenario_run(d: Path, n_sessions=8, takes_per_session=4, seed=0, d0=32, d1=24) -> Path:
    """CamFormer debole sulle procedurali (errori su cui lavora il gate), EgoVLP quasi perfetto."""
    rng = np.random.default_rng(seed)
    for sub in ("index", "features", "configs", "traj"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    mu0, mu1 = rng.normal(size=(6, d0)) * 1.2, rng.normal(size=(6, d1)) * 3.0
    rows, cam, vid = [], {}, {}
    i = 0
    for c, name in enumerate(NAMES + ["Bouldering"]):
        for s in range(n_sessions if name != "Bouldering" else 1):
            sess = f"site{s % 3}_{name.replace(' ', '').lower()}_{s:02d}"
            off = rng.normal(size=d0) * 0.5
            for t in range(takes_per_session):
                take = f"{sess}_{t}"
                split = ["train", "train", "val", "test"][s % 4]
                cc = min(c, 5)
                x0 = mu0[cc] + off + rng.normal(size=d0) * (2.2 if PROC[cc] else 1.0)
                cam[take] = np.full(d0, np.nan) if i == 5 else x0  # una take senza embedding CamFormer
                vid[take] = mu1[cc] + rng.normal(size=d1)
                pre = "train" if split == "train" and t % 3 else ("val" if split != "train" else "none")
                rows.append(
                    {
                        "take_uid": f"u{i}",
                        "take_name": take,
                        "label": name,
                        "session": sess,
                        "site": f"site{s % 3}",
                        "split": split,
                        "duration_s": 30.0,
                        "camformer_pretrain": pre,
                    }
                )
                tt = np.arange(0, 30, 0.05)
                amp = 0.1 if PROC[cc] else 0.8
                pos = np.stack([amp * np.sin(tt * (1 + t)), 0.1 * np.cos(tt), 1.6 + 0.05 * np.sin(2 * tt)], 1)
                q = np.tile([0, 0, 0, 1.0], (len(tt), 1))
                np.savez(
                    d / "traj" / f"{take}.npz", t=tt, poses=np.concatenate([pos, q], 1), valid=np.ones(len(tt), bool)
                )
                i += 1
    pd.DataFrame(rows).to_csv(d / "index" / "takes.csv", index=False)
    for m, e in (("camformer", cam), ("egovlp", vid)):
        np.savez(d / "features" / f"{m}.npz", take_name=np.array(list(e)), X=np.stack(list(e.values())))
    cfg = {"labels": {"classes": NAMES}, "window": {"length_s": 8.0, "n_windows": 3, "placement": "center"}}
    (d / "configs" / "heads.json").write_text(json.dumps({**cfg, "poses": {"fps": 20.0}}))
    return d


def _gcfg(*sets):
    return load_config([], list(FAST) + list(sets)).gate_v2


# --------------------------------------------------------------------------- regole
def test_clopper_pearson_matches_hand_values():
    assert clopper_pearson_upper(0, 96, 0.9) == pytest.approx(0.0237, abs=2e-4)
    assert clopper_pearson_upper(1, 96, 0.9) == pytest.approx(0.0399, abs=2e-4)
    assert clopper_pearson_upper(5, 5, 0.9) == 1.0 and clopper_pearson_upper(0, 0, 0.9) == 1.0


def test_risk_threshold_picks_largest_valid_coverage_with_bonferroni():
    rng = np.random.default_rng(0)
    s = np.sort(rng.random(2000))
    e = (np.arange(2000) >= 1600).astype(int)  # errori solo nel 20% di take con lo score più alto
    info = risk_threshold(s, e, alpha=0.02, delta=0.1, grid=20)
    kept = s <= info["threshold"]
    assert info["coverage_cal"] == pytest.approx(0.8, abs=1e-9)  # coperture oltre 0,8 includono errori
    assert e[kept].sum() == 0
    assert clopper_pearson_upper(int(e[kept].sum()), int(kept.sum()), 1 - 0.1 / 20) <= 0.02
    assert risk_threshold(s, np.ones_like(e), 0.05, 0.1, 20)["threshold"] == -np.inf  # nessuna copertura valida
    assert risk_threshold(s, np.zeros_like(e), 0.05, 0.1, 20)["threshold"] == np.inf  # tiene tutto


def test_acc_target_and_budget_thresholds():
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    e = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    lam = acc_target_threshold(s, e, max_error_rate=0.1)  # al più 1 errore lasciato passare
    assert lam == 0.8 and e[s <= lam].sum() == 1
    assert acc_target_threshold(s, np.zeros(10, int), 0.0) == np.inf
    for rho in (0.1, 0.3, 0.5):
        assert (s > budget_threshold(s, rho)).mean() <= rho + 1e-12
    assert budget_threshold(s, 0.0) == np.inf and budget_threshold(s, 1.0) == -np.inf


def test_cost_rule_threshold_is_beta_itself():
    r_up = np.array([0.01, 0.049, 0.05, 0.2])
    r_hat = r_up / 2
    assert escalate("cost", 0.05, None, r_hat, r_up, np.nan).tolist() == [False, False, True, True]
    assert escalate("cost_hat", 0.05, None, r_hat, r_up, np.nan).tolist() == [False, False, False, True]
    with pytest.raises(ValueError):
        escalate("boh", 0.05, None, r_hat, r_up, np.nan)


# --------------------------------------------------------------------------- calibratore
def test_platt_calibrator_recovers_risk_and_upper_bound():
    rng = np.random.default_rng(1)
    s = rng.uniform(0.001, 0.999, 4000)
    true = 1 / (1 + np.exp(-(-1.0 + 1.5 * np.log(s / (1 - s)))))
    e = (rng.random(4000) < true).astype(int)
    sess = np.repeat(np.arange(400), 10)
    cal = Calibrator("platt", n_boot=50, delta=0.1, seed=0).fit(s, e, sess)
    r, up = cal.predict(s)
    assert ece(r, e) < 0.03
    assert np.abs(r - true).max() < 0.08
    assert (up >= r).all()
    small = Calibrator("platt", n_boot=50, delta=0.1, seed=0).fit(s[:200], e[:200], sess[:200])
    _, up_small = small.predict(s)
    r_small, _ = small.predict(s)
    assert np.mean(up_small - r_small) > np.mean(up - r)  # meno dati -> limite più largo


def test_calibrator_single_class_and_isotonic():
    s = np.linspace(0.01, 0.99, 50)
    r, up = Calibrator("platt", n_boot=5).fit(s, np.zeros(50), np.arange(50)).predict(s)
    assert np.allclose(r, r[0]) and r[0] < 0.05
    r2, up2 = Calibrator("isotonic", n_boot=5).fit(s, (s > 0.5).astype(float), np.arange(50)).predict(s)
    assert np.all(np.diff(r2) >= -1e-12) and (up2 >= r2).all()
    with pytest.raises(ValueError):
        Calibrator("beta")


# --------------------------------------------------------------------------- fold e feature
def test_session_folds_never_split_a_session():
    rng = np.random.default_rng(0)
    sess = np.repeat([f"s{i}" for i in range(60)], 5)
    y = np.repeat(rng.integers(0, 6, 60), 5)
    f = session_folds(y, sess, 5, seed=3)
    check_grouping(f, sess)
    assert set(f) == set(range(5))
    with pytest.raises(RuntimeError):
        check_grouping(np.r_[f[:-1], (f[-1] + 1) % 5], sess)
    with pytest.raises(ValueError):
        session_folds(y[:10], sess[:10], 5, 0)  # 2 sessioni per 5 fold


def test_training_reliability_is_out_of_fold():
    """N38: ogni riga di fit riceve l'affidabilità calcolata sulle ALTRE fold (fold per sessione, stratificate per
    classe predetta); le righe nuove usano tutte le righe di fit."""
    from src.cascade.gatev2.folds import session_folds

    rng = np.random.default_rng(0)
    sess = np.repeat([f"s{i}" for i in range(40)], 5)
    P = rng.dirichlet(np.ones(6) * 0.5, size=len(sess))
    pred = P.argmax(1)
    y = np.where(rng.random(len(P)) < 0.7, pred, (pred + 1) % 6)
    corr = (pred == y).astype(float)
    fb = FeatureBuilder(["A", "B"], PROC, alpha=1.0, n_folds=5).fit(P, y, sess, seed=3)
    fold = session_folds(pred, sess, 5, 3)
    for j in range(5):
        out = fold != j
        for c in np.unique(pred[fold == j]):
            n, k = (pred[out] == c).sum(), corr[out][pred[out] == c].sum()
            assert np.allclose(fb.train_reliability[(fold == j) & (pred == c)], (k + 1) / (n + 2))
    k, n = corr[pred == 3].sum(), (pred == 3).sum()
    assert fb.reliability(np.eye(6)[[3]])[0] == pytest.approx((k + 1) / (n + 2))
    assert fb.transform(P, np.zeros((len(P), 9)), training=True).shape == (len(P), 5)
    with pytest.raises(RuntimeError):
        fb.transform(P[:10], np.zeros((10, 9)), training=True)


@pytest.mark.parametrize("per_session", [1, 6])
def test_training_reliability_is_not_correlated_with_own_error(per_session):
    """Il difetto della run del 18/11: escludendo la sola sessione, dentro un training set il valore di una riga è una
    funzione decrescente degli esiti della sua sessione (con sessioni di una take, la mediana dei dati, è un
    leave-one-out esatto: correlazione +1,00 con l'errore; +0,62 con 6 take). Fuori fold resta piccola."""
    rng = np.random.default_rng(1)
    sess = np.repeat([f"s{i}" for i in range(60)], per_session)
    P = np.tile(np.eye(6)[3] * 0.8 + 0.2 / 6, (len(sess), 1))  # tutte predette Cooking
    err_rate = np.repeat(rng.uniform(0, 0.8, 60), per_session)
    y = np.where(rng.random(len(sess)) < err_rate, 0, 3)
    r = FeatureBuilder(["A", "B"], PROC).fit(P, y, sess, seed=0).train_reliability
    assert abs(np.corrcoef(r, (y != 3).astype(float))[0, 1]) < 0.3


def test_scorers_direction_and_unknown_names():
    g = _gcfg()
    P = np.array([[0.9, 0.02, 0.02, 0.02, 0.02, 0.02], [0.2, 0.1, 0.1, 0.4, 0.1, 0.1]])
    assert make_scorer("thr_maxprob", g, PROC).score(P, None).tolist() == pytest.approx([0.1, 0.6])
    assert make_scorer("class_rule", g, PROC).score(P, None).tolist() == [0.0, 1.0]
    for bad in ("logreg_C", "logreg_AX", "foo"):
        with pytest.raises(ValueError):
            make_scorer(bad, g, PROC)


# --------------------------------------------------------------------------- CV annidata
def test_nested_cv_has_no_leakage(tmp_path, monkeypatch):
    """Nessuna take della fold esterna entra in una testa o in un gate di quella fold; il gate si addestra solo su
    predizioni interne fuori campione e il cross-fitting non vede la fold del gate che valuta."""
    from src.cascade.gatev2 import nested
    from src.cascade.gatev2.data import load_scenario_run
    from src.cascade.gatev2.scorers import LogRegScorer

    data = load_scenario_run(fake_scenario_run(tmp_path / "scen"), NAMES, PROC, motion=False)
    g = _gcfg("gate_v2.scorers=[thr_maxprob,logreg_A]", "gate_v2.comparisons=[[logreg_A,thr_maxprob]]")
    rows_of = {tuple(np.round(x, 9)): i for i, x in enumerate(data.X_cam) if np.isfinite(x).all()}
    rows_of.update({tuple(np.round(x, 9)): i for i, x in enumerate(data.X_vid)})
    fits, gate_fits = [], []
    real_fit, real_lr_fit = nested.fit_head, LogRegScorer.fit

    def spy_fit(X, y, groups, n_classes, hcfg, seed):
        fits.append({rows_of[tuple(np.round(x, 9))] for x in X})
        return real_fit(X, y, groups, n_classes, hcfg, seed)

    def spy_lr_fit(self, P, M, y, e, sessions, seed=0):
        gate_fits.append(set(np.asarray(sessions).astype(str)))
        return real_lr_fit(self, P, M, y, e, sessions, seed)

    monkeypatch.setattr(nested, "fit_head", spy_fit)
    monkeypatch.setattr(LogRegScorer, "fit", spy_lr_fit)
    fold = nested.outer_assignment(data, g, 0)
    for k in range(g.outer_folds):
        fits.clear()
        gate_fits.clear()
        res = nested.run_fold(data, g, 0, k, fold)
        test_rows = set(np.flatnonzero(fold == k))
        test_sess = set(data.sessions[fold == k])
        assert fits and all(not (f & test_rows) for f in fits)
        assert gate_fits and all(not (s & test_sess) for s in gate_fits)
        # cross-fitting: g_folds fit sul sottoinsieme + 1 fit finale su tutto il train esterno
        assert len(gate_fits) == g.gate_folds + 1
        assert all(len(s) < len(gate_fits[-1]) for s in gate_fits[:-1])
        assert set(res["takes"].take_name) == set(data.takes.take_name[fold == k])
        th = pd.DataFrame(res["thresholds"])
        cost = th[th.rule == "cost"]
        assert np.allclose(cost.threshold, cost.param)


def test_outer_l0_inner_mean_and_bad_value(tmp_path):
    from src.cascade.gatev2 import nested
    from src.cascade.gatev2.data import load_scenario_run

    data = load_scenario_run(fake_scenario_run(tmp_path / "scen"), NAMES, PROC, motion=False)
    g = _gcfg("gate_v2.scorers=[thr_maxprob]", "gate_v2.comparisons=[]", "gate_v2.outer_l0=inner_mean")
    fold = nested.outer_assignment(data, g, 0)
    t = nested.run_fold(data, g, 0, 0, fold)["takes"]
    assert np.isfinite(t.p_max[t.has_l0]).all()
    g.outer_l0 = "boh"
    with pytest.raises(ValueError):
        nested.run_fold(data, g, 0, 0, fold)


# --------------------------------------------------------------------------- stadio 56
def _script(name):
    spec = importlib.util.spec_from_file_location("stage_" + name.replace(".py", ""), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def env(tmp_path, monkeypatch):
    (tmp_path / "root").mkdir()
    monkeypatch.setenv("DL4CV_ROOT", str(tmp_path / "root"))
    monkeypatch.delenv("CASCADE_DATA_ROOT", raising=False)
    monkeypatch.delenv("CASCADE_RUN_ROOT", raising=False)
    from src.cascade.paths import run_dir

    return run_dir("g56"), fake_scenario_run(tmp_path / "scen")


def _argv(scen, *extra):
    out = ["--run-name", "g56", "--scenario-run", str(scen)]
    for o in FAST:
        out += ["--set", o]
    return out + list(extra)  # gli override del test vincono su FAST


def test_stage_56_chain(env):
    rd, scen = env
    _script("56_gate_v2.py").main(_argv(scen))
    d = rd / "56_gate_v2" / "main"
    m = pd.read_csv(d / "metrics.csv")
    assert {"always_L0", "always_L1", "oracle", "thr_maxprob", "class_rule"} <= set(m.system)
    assert {"cost", "cost_hat", "risk", "acc_target", "budget", "natural", "fixed"} <= set(m.rule)
    assert set(m.subset) == {"all", "pretrain_unseen", "pretrain_seen", "official_eval"}
    allm = m[m.subset == "all"]
    l0 = allm[allm.system == "always_L0"].iloc[0]
    l1 = allm[allm.system == "always_L1"].iloc[0]
    orc = allm[allm.system == "oracle"].iloc[0]
    assert l1.acc >= orc.acc - 1e-12 and orc.acc >= l0.acc and 0 < orc.rho < 1
    assert l0.rho == pytest.approx(1 / 192)  # solo la take senza CamFormer va a EgoVLP
    # regola a costo: β più piccolo -> più escalation
    c = allm[(allm.system == "thr_maxprob") & (allm.rule == "cost")].sort_values("param")
    assert c.rho.is_monotonic_decreasing
    assert (m.acc_ci_low <= m.acc + 1e-12).all() and (m.acc <= m.acc_ci_high + 1e-12).all()
    comp = pd.read_csv(d / "comparisons.csv")
    assert {"auroc", "acc_at_primary", "rho_at_primary"} <= set(comp.metric)
    assert (comp.p_holm >= comp.p_two_sided - 1e-12).all()
    auc = pd.read_csv(d / "auc.csv")
    assert auc.area.between(0, 1).all()
    t = pd.read_csv(d / "takes.csv.gz")
    assert t.groupby("repeat").take_name.nunique().eq(192).all()
    s = json.loads((d / "summary.json").read_text())
    assert s["primary"]["subset"] == "pretrain_unseen" and "acc" in s["primary"]
    assert s["counts"]["n_out_of_head"] == 4 and s["counts"]["n_no_l0"] == 1
    cv = pd.read_csv(d / "cost_vs_egovlp.csv")
    cvall = cv[cv.subset == "all"]
    eg = cvall[cvall.system == "always_L1"].iloc[0]
    assert eg.cost_frac_of_egovlp_only == pytest.approx(1.0) and eg.speedup == pytest.approx(1.0)
    cb = cvall[(cvall.system == "thr_maxprob") & (cvall.rule == "cost")].sort_values("param")
    assert len(cb) and (cb.cost_frac_of_egovlp_only < 1.0 + 1e-9).all() and cb.speedup.is_monotonic_increasing
    assert (cb.cost_frac_ci_low <= cb.cost_frac_of_egovlp_only + 1e-12).all()
    assert cv.primary_beta.any() and "cost_vs_egovlp_primary" in s
    proc = pd.read_csv(d / "procedural.csv")
    nat = proc[(proc.subset == "all") & (proc.system == "class_rule") & (proc.rule == "natural")].iloc[0]
    assert nat.kept_on_L0 == 0  # la regola per classe scala tutte le procedurali predette
    mtime = (d / "metrics.csv").stat().st_mtime
    _script("56_gate_v2.py").main(_argv(scen))  # già fatto: non rigira
    assert (d / "metrics.csv").stat().st_mtime == mtime


def test_stage_56_refuses_changed_gate_v2_config(env):
    rd, scen = env
    _script("56_gate_v2.py").main(_argv(scen))
    d = rd / "56_gate_v2" / "main"
    mtime = (d / "metrics.csv").stat().st_mtime
    with pytest.raises(Exception, match="gate_v2"):  # config diversa dal manifest: errore, non "già fatto"
        _script("56_gate_v2.py").main(_argv(scen, "--set", "gate_v2.calibration_boot=7"))
    assert (d / "metrics.csv").stat().st_mtime == mtime


def test_stage_56_rejects_unregistered_primary_point(env):
    rd, scen = env
    with pytest.raises(ValueError, match="griglia"):
        _script("56_gate_v2.py").main(_argv(scen, "--set", "gate_v2.primary_param=0.07"))
    with pytest.raises(ValueError, match="scorer"):
        _script("56_gate_v2.py").main(_argv(scen, "--set", "gate_v2.comparisons=[[lgbm_AC,thr_maxprob]]"))


# --------------------------------------------------------------------------- correzioni del gate v1 (N37)
def test_v1_cv_groups_use_session_when_available():
    from src.cascade.gate.protocol import cv_groups

    df = pd.DataFrame({"take_uid": ["a", "b", "c"], "session": ["s1", "s1", "s2"]})
    assert cv_groups(df, "session").tolist() == ["s1", "s1", "s2"]
    assert cv_groups(df, "take").tolist() == ["a", "b", "c"]
    assert cv_groups(df.drop(columns="session"), "session").tolist() == ["a", "b", "c"]
    with pytest.raises(ValueError):
        cv_groups(df, "site")


def test_v1_group_bootstrap_weights_move_sessions_together():
    from src.cascade.eval.curves import group_resample_weights

    groups = np.array(["s1", "s1", "s1", "s2", "s3", "s3"])
    W = group_resample_weights(groups, 200, seed=0)
    assert W.shape == (200, 6)
    assert (W[:, 0] == W[:, 1]).all() and (W[:, 1] == W[:, 2]).all() and (W[:, 4] == W[:, 5]).all()
    assert (W[:, [0, 3, 4]].sum(1) == 3).all()  # 3 sessioni estratte per replica


def test_v1_session_bootstrap_is_wider_with_clustered_sessions():
    from src.cascade.eval.curves import group_resample_weights, resample_weights

    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 10)
    x = np.repeat(rng.integers(0, 2, 20), 10).astype(float)  # take identiche dentro la sessione
    sd_take = (resample_weights(200, 500, 1) @ x / 200).std()
    sd_sess = (group_resample_weights(groups, 500, 1) @ x / 200).std()
    assert sd_sess > 2 * sd_take


# --------------------------------------------------------------------------- N38: calibrazione sulle non viste
def test_calibration_population_uses_only_unseen_training_rows(tmp_path):
    from src.cascade.gatev2 import nested
    from src.cascade.gatev2.data import load_scenario_run

    data = load_scenario_run(fake_scenario_run(tmp_path / "scen"), NAMES, PROC, motion=False)
    g = _gcfg(
        "gate_v2.scorers=[thr_maxprob]",
        "gate_v2.comparisons=[]",
        "gate_v2.calibration_population=pretrain_unseen",
        "gate_v2.calibration_min_errors=1",
    )
    fold = nested.outer_assignment(data, g, 0)
    unseen = data.takes.camformer_pretrain.to_numpy() != "train"
    for k in range(g.outer_folds):
        th = pd.DataFrame(nested.run_fold(data, g, 0, k, fold)["thresholds"])
        tr0 = (fold != k) & data.has_l0
        assert set(th.cal_population) == {"pretrain_unseen"}
        assert (th.n_cal == int((tr0 & unseen).sum())).all()
    g.calibration_min_errors = 10**6  # troppo pochi errori: si torna a tutte le righe, dichiarandolo
    th = pd.DataFrame(nested.run_fold(data, g, 0, 0, fold)["thresholds"])
    assert set(th.cal_population) == {"all_fallback"}
    assert (th.n_cal == int(((fold != 0) & data.has_l0).sum())).all()


def test_default_calibration_population_is_unchanged(tmp_path):
    from src.cascade.gatev2 import nested
    from src.cascade.gatev2.data import load_scenario_run

    data = load_scenario_run(fake_scenario_run(tmp_path / "scen"), NAMES, PROC, motion=False)
    g = _gcfg("gate_v2.scorers=[thr_maxprob]", "gate_v2.comparisons=[]")
    assert g.calibration_population == "all"
    fold = nested.outer_assignment(data, g, 0)
    th = pd.DataFrame(nested.run_fold(data, g, 0, 0, fold)["thresholds"])
    assert set(th.cal_population) == {"all"} and (th.n_cal == int(((fold != 0) & data.has_l0).sum())).all()


def test_stage_56_rejects_unknown_calibration_population(env):
    rd, scen = env
    with pytest.raises(ValueError, match="calibration_population"):
        _script("56_gate_v2.py").main(_argv(scen, "--set", "gate_v2.calibration_population=val"))


def test_n38_configs_reproduce_the_18_11_run_except_the_corrections():
    """Le config della run correttiva cambiano solo calibration_population rispetto alla run del 18/11."""
    import yaml

    base = load_config([]).gate_v2
    for name, pop in (("gate_v2_n38_rel.yaml", "all"), ("gate_v2_n38_cal.yaml", "pretrain_unseen")):
        g = load_config([ROOT / "configs" / name]).gate_v2
        raw = yaml.safe_load((ROOT / "configs" / name).read_text())["gate_v2"]
        assert set(raw) == {"betas", "scorers", "calibration_population"}
        assert g.betas == [0.01, 0.02, 0.05, 0.10, 0.20] and "lgbm_ABC" in g.scorers
        assert g.calibration_population == pop
        for f in ("outer_folds", "inner_folds", "gate_folds", "repeats", "primary_rule", "primary_param", "subsets"):
            assert getattr(g, f) == getattr(base, f)
