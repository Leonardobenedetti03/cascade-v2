"""Fixture comuni: run_dir temporanea e dataset sintetico piccolo (CPU, pochi secondi)."""

from __future__ import annotations

import pytest

from src.cascade.cli import open_stage, stage_parser

SMALL = [
    "synthetic.takes_per_class={train: 5, val: 2, test: 2}",
    "synthetic.duration_s=[6.0, 8.0]",
    "synthetic.video_size=32",
    "synthetic.traj_hz=50.0",
]


def make_ctx(root, name="test_stage", extra=(), synthetic=True):
    import os

    os.environ["DL4CV_ROOT"] = str(root)
    os.environ.pop("CASCADE_DATA_ROOT", None)
    os.environ.pop("CASCADE_RUN_ROOT", None)
    argv = ["--run-name", "t"]
    if synthetic:
        argv.append("--synthetic")
    for o in list(SMALL if synthetic else []) + list(extra):
        argv += ["--set", o]
    args = stage_parser(None).parse_args(argv)
    return open_stage(args, name)


@pytest.fixture(scope="session")
def syn_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("dl4cv")
    make_ctx(root)  # genera il dataset una volta per sessione
    return root


@pytest.fixture
def syn_ctx(syn_root, monkeypatch):
    monkeypatch.setenv("DL4CV_ROOT", str(syn_root))
    return make_ctx(syn_root)
