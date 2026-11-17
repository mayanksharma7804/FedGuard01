"""Checks that scripts/import_runs.py applies before accepting a run from another machine."""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from import_runs import check_run  # noqa: E402
from run_experiments import normalise, run_name  # noqa: E402

from fedguard.config import load_config  # noqa: E402
from fedguard.utils import REPO_ROOT  # noqa: E402


def fake_run(fingerprint="abc123"):
    cfg = normalise(load_config(REPO_ROOT / "configs" / "k10_base.yaml"))
    files = {"config.yaml": yaml.safe_dump(cfg).encode(), "metrics_val.json": b"{}", "rounds.csv": b"round\n1\n",
             "weights_log.csv": b"round,client\n1,0\n",
             "metrics_test.json": json.dumps({"data_fingerprint": fingerprint}).encode()}
    return run_name(cfg), cfg, files


def test_valid_run_is_accepted():
    name, _, files = fake_run()
    assert check_run(files, name, "abc123")[0]


def test_run_from_other_data_is_refused():
    name, _, files = fake_run(fingerprint="other")
    ok, msg = check_run(files, name, "abc123")
    assert not ok and "other data" in msg


def test_renamed_or_edited_run_is_refused():
    name, cfg, files = fake_run()
    assert not check_run(files, name + "x", "abc123")[0]                  # folder renamed
    cfg["seed"] = 43
    files["config.yaml"] = yaml.safe_dump(cfg).encode()                   # config edited
    assert not check_run(files, name, "abc123")[0]


def test_incomplete_run_is_refused():
    name, _, files = fake_run()
    del files["rounds.csv"]
    assert not check_run(files, name, "abc123")[0]
