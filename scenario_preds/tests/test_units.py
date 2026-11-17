import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from scenario.index import scenario_of, session_of, site_of
from scenario.poses import Trajectory, apply_frame_correction, estimate_frame, extrinsics_to_c2w
from scenario.windows import compute_windows


def test_scenario_mapping():
    assert scenario_of("Cooking an Omelet") == "Cooking"
    assert scenario_of("Making Milk Tea") == "Cooking"
    assert scenario_of("First Aid - CPR") == "Health"
    assert scenario_of("Covid-19 Rapid Antigen Test") == "Health"
    assert scenario_of("Remove a Wheel") == "Bike repair"
    assert scenario_of("Rock Climbing") == "Bouldering"
    assert scenario_of("Basketball") == "Basketball"
    with pytest.raises(ValueError):
        scenario_of("Knitting")


def test_site_session():
    assert site_of("upenn_0706_Dance_4_5") == "upenn"
    assert session_of("uniandes_dance_016_42") == "uniandes_dance_016"


def test_windows():
    assert compute_windows(4.0, 8.0) == [(0.0, 4.0)]
    assert compute_windows(20.0, 8.0) == [(6.0, 14.0)]
    w = compute_windows(20.0, 8.0, 3, "center")
    assert w[0] == (0.0, 8.0) and w[-1] == (12.0, 20.0)
    assert len(compute_windows(100.0, 8.0, 4, "uniform")) == 4


def test_extrinsics_inversion():
    R = Rotation.random(5, random_state=0).as_matrix()
    c = np.random.default_rng(0).normal(size=(5, 3))
    t = -np.einsum("nij,nj->ni", R, c)  # mondo -> camera: t = -R c
    centre, q = extrinsics_to_c2w(np.concatenate([R, t[:, :, None]], 2))
    assert np.allclose(centre, c)
    assert np.allclose(Rotation.from_quat(q).as_matrix(), np.transpose(R, (0, 2, 1)))


def _traj(R, c, fps=20.0, src="x"):
    n = len(c)
    return Trajectory(np.arange(n) / fps, np.concatenate([c, Rotation.from_matrix(R).as_quat()], 1), np.ones(n, bool), src)


def test_estimate_frame_recovers_rotations():
    rng = np.random.default_rng(1)
    n = 600
    rv = np.cumsum(rng.normal(scale=0.05, size=(n, 3)), axis=0)  # moto rotazionale casuale
    Rc = Rotation.from_rotvec(rv).as_matrix()
    c = np.cumsum(rng.normal(scale=0.01, size=(n, 3)), axis=0)
    X = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.diag([1, -1, -1])
    W = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    Rd = np.einsum("ij,njk,kl->nil", W, Rc, X)
    res = estimate_frame(_traj(Rc, c), _traj(Rd, c @ W.T))
    assert np.allclose(res["json_cam_to_device"], X, atol=1e-3)
    assert np.allclose(res["json_world_rotation"], W, atol=1e-3)
    assert res["residual_deg_median"] < 0.1 and res["shift_s"] == 0
    # e la correzione applicata ai JSON riproduce le pose del device
    corr = apply_frame_correction(_traj(Rc, c).poses, X, W)
    assert np.allclose(Rotation.from_quat(corr[:, 3:]).as_matrix(), Rd, atol=1e-6)


def test_cross_fit_is_out_of_sample():
    from scenario.heads import cross_fit

    rng = np.random.default_rng(0)
    n, C = 300, 3
    y = rng.integers(0, C, n)
    X = rng.normal(size=(n, 8)) + 2.0 * np.eye(C, 8)[y]
    takes = pd.DataFrame(
        {
            "label_idx": y,
            "fold": np.where(np.arange(n) < 240, np.arange(n) % 4, -1),
            "split": np.where(np.arange(n) < 240, "train", "val"),
            "session": [f"s{i // 2}" for i in range(n)],
        }
    )
    takes.loc[5, "label_idx"] = -1  # take fuori dalla testa: predetta, non usata nel fit
    X[7] = np.nan  # take senza embedding: nessuna predizione
    hcfg = {"Cs": [0.1, 1.0], "inner_cv": 3, "max_iter": 500, "class_weight": None}
    res, folds = cross_fit(X, takes, C, hcfg, 0)
    P = res["probs"]
    assert len(folds) == 4
    assert np.isnan(P[7]).all() and np.isfinite(P[5]).all()
    assert np.allclose(P[np.isfinite(P).all(1)].sum(1), 1)
    assert set(res["source"][:240]) - {""} == {"oof"} and set(res["source"][240:]) == {"fold_ensemble"}
    acc = (P[240:].argmax(1) == y[240:]).mean()
    assert acc > 0.8


def test_cut_official_val():
    from scenario.index import cut_official_val

    rows = []
    for c, n_sess in (("A", 10), ("B", 4), ("C", 1)):
        for k in range(n_sess):
            for t in range(1 + k % 3):
                rows.append({"label": c, "session": f"{c}{k}", "take_name": f"{c}{k}_{t}"})
    val = pd.DataFrame(rows)
    out = cut_official_val(val, {"mode": "split", "test_frac": 0.5, "seed": 0})
    val["split"] = out
    assert not (set(val.session[val.split == "val"]) & set(val.session[val.split == "test"]))
    for c in ("A", "B"):
        g = val[val.label == c]
        assert set(g.split) == {"val", "test"}
        assert abs((g.split == "test").mean() - 0.5) < 0.25
    assert val[val.label == "C"].split.nunique() == 1  # una sola sessione: tutta da una parte
    assert set(cut_official_val(val, {"mode": "all_test", "test_frac": 0.5, "seed": 0})) == {"test"}
    assert set(cut_official_val(val, {"mode": "undecided", "test_frac": 0.5, "seed": 0})) == {"val"}


def test_pose_json_gap_count(tmp_path):
    import json

    from scenario.poses import from_pose_json

    E = np.concatenate([np.eye(3), np.zeros((3, 1))], 1).tolist()
    frames = [i for i in range(5, 300) if not (100 <= i < 110)]  # inizio al frame 5, buco di 10 frame
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"aria01": {"camera_extrinsics": {str(i): E for i in frames}}}))
    pc = {"json_src_fps": 30.0, "fps": 20.0, "json_max_gap_frames": 15,
          "json_cam_to_device": np.eye(3).tolist(), "json_world_rotation": np.eye(3).tolist()}
    tr = from_pose_json(f, "x/aria01_214-1.mp4", pc)
    assert tr.n_missing_src == 10 and tr.first_src_frame == 5
    assert tr.valid[tr.t > 1.0].all() and not tr.valid[0]  # buco corto interpolato, inizio mancante non valido


def _device_poses(n, pitch_deg, rng):
    """Device Aria (x sx, y su, z avanti) con imbardata casuale, inclinazione fissa e un po' di rumore."""
    yaw = rng.uniform(-np.pi, np.pi, n)
    p = np.radians(pitch_deg + rng.normal(scale=3, size=n))
    z = np.stack([np.cos(yaw) * np.cos(p), np.sin(yaw) * np.cos(p), np.sin(p)], 1)
    y = np.stack([-np.cos(yaw) * np.sin(p), -np.sin(yaw) * np.sin(p), np.cos(p)], 1)
    x = np.cross(y, z)
    return np.stack([x, y, z], 2)  # colonne = assi del device nel mondo


@pytest.mark.parametrize("k", [0, 1, 3])
def test_check_up_recovers_roll(k):
    from scenario.poses import roll_candidates, suggest_cam_to_device, up_statistics

    rng = np.random.default_rng(k)
    Rd = _device_poses(500, -15.0, rng)
    X = roll_candidates()[k]
    Rc = Rd @ X.T  # R_wd = R_wc X
    st = up_statistics(_traj(Rc, np.zeros((500, 3))))
    sug = suggest_cam_to_device(np.array([st["up_x"], st["up_y"], st["up_z"]]))
    assert sug["roll_k"] == k and sug["residual_roll_deg"] < 5
    assert np.allclose(sug["json_cam_to_device"], X, atol=1e-6)
    assert st["pitch_deg_median"] == pytest.approx(-15.0, abs=2)  # con k = 0 la camera è OpenCV dritta


def test_check_up_flags_world_upside_down():
    from scenario.poses import up_statistics

    rng = np.random.default_rng(0)
    Rd = _device_poses(300, -15.0, rng)
    flip = np.diag([1.0, -1.0, -1.0])  # mondo con z verso il basso
    Rc = np.einsum("ij,njk->nik", flip, Rd) @ np.diag([-1.0, -1.0, 1.0]).T
    assert up_statistics(_traj(Rc, np.zeros((300, 3))))["pitch_deg_median"] > 10


def test_check_counts_missing_classes_and_embeddings():
    """Audit punti 2 e 3: classi a 0 take e take senza embedding sono contate, non tolte in silenzio."""
    import pandas as pd

    from scenario.heads import metrics, preregistered_check

    classes = ["A", "B", "C"]
    rows = []
    for split in ("train", "val"):
        for c, n in ((0, 6), (1, 6), (2, 6 if split == "train" else 0)):
            for j in range(n):
                rows.append({"split": split, "label_idx": c, "pred_idx": c, "pred": classes[c], "session": f"{split}{c}{j}"})
    rows.append({"split": "val", "label_idx": 0, "pred_idx": np.nan, "pred": None, "session": "val_x"})  # senza embedding
    df = pd.DataFrame(rows)
    m = metrics(df, classes, "val", 1)
    assert m["n"] == 12 and m["n_no_embedding"] == 1 and m["bacc_classes"] == ["A", "B"]
    hcfg = {"min_takes_metric": 1, "n_boot": 50, "ci_level": 0.95}
    ccfg = {"model": "camformer", "split": "val", "min_bacc": 0.4, "min_recall": None, "ci_above_chance": True}
    assert preregistered_check(df, classes, dict(ccfg, require_all_classes=False), hcfg)["passed"]
    res = preregistered_check(df, classes, dict(ccfg, require_all_classes=True), hcfg)
    assert not res["passed"] and res["classes_missing_in_split"] == ["C"] and res["n_no_embedding"] == 1
