"""Pipeline completa su dati sintetici (CPU): index -> traj -> check-frame -> feat-* -> heads.

Usa il codice vero di CamFormer (serve il clone del repository: variabile CAMFORMER_REPO o
../ext/seeing_without_pixels) con pesi casuali, ed EgoVLP con un'architettura ridotta.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

REPO = Path(os.environ.get("CAMFORMER_REPO", Path(__file__).resolve().parents[2] / "ext" / "seeing_without_pixels"))
FPS_VIDEO = 30
DUR = 10.0
X_TRUE = Rotation.from_euler("y", 90, degrees=True).as_matrix()  # camera -> device del finto Aria

pytestmark = pytest.mark.skipif(not (REPO / "models" / "cm_encoder.py").exists(), reason="repository CamFormer assente")


def _video(path: Path, color: int, n: int) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=FPS_VIDEO)
        s.width, s.height, s.pix_fmt = 96, 64, "yuv420p"
        for i in range(n):
            img = np.full((64, 96, 3), (color + 3 * i) % 255, np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def _motion(kind: int, n: int, rng):
    t = np.arange(n) / FPS_VIDEO
    amp = [0.8, 0.1, 0.3][kind]
    rv = np.stack([amp * np.sin(2 * np.pi * (kind + 1) * 0.5 * t), 0.2 * np.cos(t), rng.normal(scale=0.01, size=n)], 1)
    Rd = Rotation.from_rotvec(rv).as_matrix()
    c = np.stack([np.sin(t) * amp, np.cos(t) * 0.1, 1.6 + 0.05 * np.sin((kind + 1) * t)], 1)
    return Rd, c


def make_dataset(tmp: Path):
    rng = np.random.default_rng(0)
    roots = [tmp / "phys", tmp / "proc"]
    csv_root = tmp / "aria"
    specs = [("Basketball", 0, "unc", 0), ("Cooking an Omelet", 1, "iiith", 1), ("First Aid - CPR", 2, "nus", 1)]
    rows = {0: [], 1: []}
    n = int(DUR * FPS_VIDEO)
    for scen, kind, site, ri in specs:
        for j in range(15):
            name = f"{site}_{scen.split()[0].lower()}_{j:03d}_1"
            split = "val" if j >= 12 else "train"
            root = roots[ri]
            _video(root / "takes" / name / "frame_aligned_videos/downscaled/448/aria01_214-1.mp4", 40 * kind, n)
            Rd, c = _motion(kind, n, rng)
            Rc = Rd @ X_TRUE.T  # R_wd = R_wc X  ->  R_wc = R_wd X^T
            E = np.concatenate([np.transpose(Rc, (0, 2, 1)), -np.einsum("nji,nj->ni", Rc, c)[:, :, None]], 2)
            pose_rel = f"annotations/ego_pose/{split}/camera_pose/{name}.json"
            (root / pose_rel).parent.mkdir(parents=True, exist_ok=True)
            (root / pose_rel).write_text(json.dumps({"metadata": {}, "aria01": {"camera_extrinsics": {str(i): E[i].tolist() for i in range(n)}}}))
            ts = (np.arange(n) / FPS_VIDEO * 1e6).astype(np.int64) + 123456
            q = Rotation.from_matrix(Rd).as_quat()
            cdir = csv_root / name / "trajectory"
            cdir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {"tracking_timestamp_us": ts, "tx_world_device": c[:, 0], "ty_world_device": c[:, 1], "tz_world_device": c[:, 2],
                 "qx_world_device": q[:, 0], "qy_world_device": q[:, 1], "qz_world_device": q[:, 2], "qw_world_device": q[:, 3]}
            ).to_csv(cdir / "closed_loop_trajectory.csv", index=False)
            rows[ri].append({"take_uid": f"uid-{name}", "take_name": name, "split": split, "scenario": scen, "camera_pose_file": pose_rel})
    for ri, root in enumerate(roots):
        pd.DataFrame(rows[ri]).to_csv(root / "split_manifest.csv", index=False)
    return roots, csv_root


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    import torch

    from scenario.camformer import build_encoder
    from scenario.config import load_config
    from scenario.egovlp import build_model

    tmp = tmp_path_factory.mktemp("syn")
    roots, csv_root = make_dataset(tmp)
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    torch.manual_seed(0)
    enc = build_encoder(cfg["camformer"], REPO)
    torch.save({"state_dict": {f"model.{k}": v for k, v in enc.state_dict().items()}}, tmp / "cam.pt")
    small = dict(cfg["egovlp"], embed_dim=64, depth=1, num_heads=2, proj_dim=16)
    vid = build_model(small)
    torch.save({"state_dict": {f"module.{k}": v for k, v in vid.state_dict().items()}, "config": None}, tmp / "vid.pth")
    os.environ.update(SCENARIO_RUN_DIR=str(tmp / "run"), CAMFORMER_REPO=str(REPO), CAMFORMER_CKPT=str(tmp / "cam.pt"),
                      EGOVLP_CKPT=str(tmp / "vid.pth"), ARIA_TRAJ_ROOT=str(csv_root))
    sets = [f"data.roots={json.dumps([str(r) for r in roots])}", "egovlp.embed_dim=64", "egovlp.depth=1", "egovlp.num_heads=2",
            "egovlp.proj_dim=16", "egovlp.num_workers=0", "split.n_folds=3", "heads.Cs=[0.1,1.0]",
            "data.expected_takes=45", "split.official_val.mode=split", "split.fold_seeds=[0,1]",
            "heads.min_takes_metric=1", "heads.n_boot=200", "check.min_bacc=0.3", "check.split=val",
            # i dati sintetici hanno 3 classi su 6 e pesi casuali: niente controllo sulle classi mancanti né sull'intervallo
            "check.require_all_classes=false", "check.ci_above_chance=false"]
    return tmp / "run", sets


def _main(stage, sets, *extra):
    from scenario.__main__ import main

    main([stage, *sum([["--set", s] for s in sets], []), *extra])


def test_pipeline(run):
    run_dir, sets = run
    for stage in ("index", "traj", "feat-camformer", "feat-egovlp", "heads"):
        _main(stage, sets)
    takes = pd.read_csv(run_dir / "index" / "takes.csv")
    assert len(takes) == 45 and set(takes.label) == {"Basketball", "Cooking", "Health"}
    assert (takes[takes.split == "train"][["fold_s0", "fold_s1"]] >= 0).all().all()
    ev = takes[takes.split_official == "val"]
    assert set(ev.split) == {"val", "test"}
    for _, g in ev.groupby("label"):  # stratificato: ogni classe da entrambe le parti
        assert set(g.split) == {"val", "test"}
    assert not (set(ev.session[ev.split == "val"]) & set(ev.session[ev.split == "test"]))
    pred = pd.read_csv(run_dir / "predictions.csv")
    assert "test" not in set(pred.split)  # test sigillato
    assert not (run_dir / "sealed").exists()
    assert pred.cam_pred.notna().all() and pred.vid_pred.notna().all()
    assert set(pred.cam_pred_source) == {"oof", "fold_ensemble"}
    probs = pred[[c for c in pred.columns if c.startswith("cam_p_")]].to_numpy()
    assert np.allclose(probs.sum(1), 1)
    m = json.loads((run_dir / "heads" / "camformer" / "metrics.json").read_text())
    assert "test" not in m and m["val"]["bacc_ci"] is not None
    folds = pd.read_csv(run_dir / "heads" / "camformer" / "folds.csv")
    assert set(folds.seed) == {0, 1}
    chk = json.loads((run_dir / "check.json").read_text())
    assert chk["passed"] and chk["split"] == "val" and chk["chance"] == pytest.approx(1 / 3)
    classes = json.loads((run_dir / "classes.json").read_text())
    assert classes["classes"][:3] == ["Basketball", "Dance", "Soccer"]
    # rilancio: lo stadio si salta; con un'altra config si ferma
    _main("heads", sets)
    from scenario.io import StageConflict

    with pytest.raises(StageConflict):
        _main("heads", sets + ["heads.Cs=[1.0]"])
    # soglia irraggiungibile: codice 3 (ferma la catena), anche al rilancio
    with pytest.raises(SystemExit) as e:
        _main("heads", sets + ["check.min_bacc=1.01"], "--overwrite")
    assert e.value.code == 3
    with pytest.raises(SystemExit):
        _main("heads", sets + ["check.min_bacc=1.01"])
    # il test si apre solo con --eval-test, in un file a parte
    _main("heads", sets, "--eval-test", "--overwrite")
    sealed = pd.read_csv(run_dir / "sealed" / "predictions.csv")
    assert set(sealed.split) == {"test"}
    assert "test" in json.loads((run_dir / "heads" / "camformer" / "metrics.json").read_text())


def test_guards(run):
    run_dir, sets = run
    _main("index", sets)
    # check sul test vietato; soglia non fissata vietata
    for bad in (["check.split=test"], ["check.min_bacc=null"]):
        with pytest.raises(ValueError):
            _main("heads", sets + bad, "--overwrite")
    # taglio non deciso: index gira, heads no; le feature non si rifanno (non dipendono dallo split)
    _main("index", sets + ["split.official_val.mode=undecided"], "--overwrite")
    with pytest.raises(RuntimeError, match="undecided"):
        _main("heads", sets + ["split.official_val.mode=undecided"])
    _main("feat-camformer", sets + ["split.official_val.mode=undecided"])  # già fatto, nessun conflitto
    # opzione C: tutta la val ufficiale va al test, il controllo usa il train fuori campione
    _main("index", sets + ["split.official_val.mode=all_test"], "--overwrite")
    with pytest.raises(ValueError, match="all_test"):  # senza val il controllo non può usare la val
        _main("heads", sets + ["split.official_val.mode=all_test"], "--overwrite")
    _main("heads", sets + ["split.official_val.mode=all_test", "check.split=train"], "--overwrite")
    chk = json.loads((run_dir / "check.json").read_text())
    assert chk["split"] == "train" and chk["passed"] and chk["min_recall"] is None
    assert "test" not in set(pd.read_csv(run_dir / "predictions.csv").split)  # test sigillato anche con all_test
    # richiamo minimo irraggiungibile: il controllo fallisce anche con l'accuratezza bilanciata sopra soglia
    with pytest.raises(SystemExit) as e:
        _main("heads", sets + ["split.official_val.mode=all_test", "check.split=train", "check.min_bacc=0.0",
                               "check.min_recall=1.01"], "--overwrite")
    assert e.value.code == 3
    chk = json.loads((run_dir / "check.json").read_text())
    assert chk["cond1_bacc_ok"] and not chk["cond2_recall_ok"] and chk["classes_below_min_recall"]
    _main("index", sets, "--overwrite")


def test_check_frame_and_json_source(run):
    run_dir, sets = run
    _main("index", sets)
    _main("check-up", sets, "--n-check", "0")
    up = json.loads((run_dir / "check_up" / "summary.json").read_text())
    assert up["n_takes"] == 45 and "suggested" in up
    _main("check-frame", sets, "--n-check", "3")
    s = json.loads((run_dir / "check_frame" / "summary.json").read_text())
    assert np.allclose(s["json_cam_to_device"], X_TRUE, atol=1e-2), s
    assert np.allclose(s["json_world_rotation"], np.eye(3), atol=1e-2), s
    # con la correzione stimata, le pose JSON coincidono con quelle del CSV
    from scenario.poses import from_aria_csv, from_pose_json
    from scenario.config import load_config

    t = pd.read_csv(run_dir / "index" / "takes.csv").iloc[0]
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    pc = dict(cfg["poses"], json_cam_to_device=s["json_cam_to_device"], json_world_rotation=s["json_world_rotation"])
    a = from_pose_json(Path(t.pose_json_path), t.video_path, pc)
    b = from_aria_csv(Path(t.aria_csv_path), 20.0)
    n = min(len(a.t), len(b.t))
    ang = (Rotation.from_quat(a.poses[:n, 3:]).inv() * Rotation.from_quat(b.poses[:n, 3:])).magnitude()
    assert np.degrees(np.median(ang)) < 1.0


def test_require_all_classes(run):
    """Audit punto 2: una classe della testa senza take nello split del controllo fa fallire il controllo."""
    run_dir, sets = run
    _main("index", sets)
    with pytest.raises(SystemExit) as e:
        _main("heads", sets + ["check.require_all_classes=true"], "--overwrite")
    assert e.value.code == 3
    chk = json.loads((run_dir / "check.json").read_text())
    assert chk["cond1_bacc_ok"] and not chk["cond3_all_classes_ok"]
    assert chk["classes_missing_in_split"] == ["Bike repair", "Dance", "Soccer"]
    assert chk["n_no_embedding"] == 0
    _main("heads", sets, "--overwrite")


def test_select(run, tmp_path):
    """Audit punto 1: la scelta di pose e finestre si fa sulla val, tra candidate che differiscono solo lì."""
    run_dir, sets = run
    flip = "poses.json_cam_to_device=[[-1,0,0],[0,-1,0],[0,0,1]]"
    cands = []
    for name, extra in (("id1", []), ("flip1", [flip]), ("id3", ["window.n_windows=3"])):
        d = tmp_path / name
        os.environ["SCENARIO_RUN_DIR"] = str(d)
        for st in ("index", "traj", "feat-camformer"):
            _main(st, sets + extra)
        _main("heads", sets + extra, "--models", "camformer", "--skip-check")
        cands.append(str(d))
    os.environ["SCENARIO_RUN_DIR"] = str(tmp_path / "sel")
    _main("select", sets, "--candidates", *cands)
    out = json.loads((tmp_path / "sel" / "select" / "selection.json").read_text())
    assert {c["variant"] for c in out["candidates"]} == {"identita", "diag(-1,-1,1)"}
    best = max(out["candidates"], key=lambda c: c["val_bacc"])
    assert out["chosen"]["val_bacc"] == pytest.approx(best["val_bacc"])
    assert any(x.startswith("window.n_windows=") for x in out["set_for_final_run"])
    # una candidata con un'altra config (non solo pose e finestre) non è ammessa
    d = tmp_path / "otherC"
    os.environ["SCENARIO_RUN_DIR"] = str(d)
    for st in ("index", "traj", "feat-camformer"):
        _main(st, sets)
    _main("heads", sets + ["heads.Cs=[1.0]"], "--models", "camformer", "--skip-check")
    os.environ["SCENARIO_RUN_DIR"] = str(tmp_path / "sel")
    with pytest.raises(RuntimeError, match="diverse"):
        _main("select", sets, "--candidates", cands[0], str(d))
    # una candidata che ha aperto il test non è ammessa
    os.environ["SCENARIO_RUN_DIR"] = cands[0]
    _main("heads", sets, "--models", "camformer", "--skip-check", "--eval-test", "--overwrite")
    os.environ["SCENARIO_RUN_DIR"] = str(tmp_path / "sel")
    with pytest.raises(RuntimeError, match="test"):
        _main("select", sets, "--candidates", *cands)
    os.environ["SCENARIO_RUN_DIR"] = str(run_dir)
