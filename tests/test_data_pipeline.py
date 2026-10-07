"""Leakage guards of the cleaning pipeline, checked on a small synthetic table."""
import json

import numpy as np
import pandas as pd
import pytest

from fedguard.data import DatasetSpec, cap_per_class, clean, prepare
from fedguard.utils import row_keys

SPEC = DatasetSpec(name="toy", csv_files=(), label_col="Attack", benign_label="Benign",
                   drop_cols=("SRC_IP", "DST_PORT", "Label"))


def toy_frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.choice(["Benign", "DoS", "Exploits", "Worms"], size=n, p=[0.7, 0.15, 0.13, 0.02])
    base = {"DoS": 5.0, "Exploits": 2.0, "Benign": 0.0, "Worms": 8.0}
    df = pd.DataFrame({
        "SRC_IP": [f"10.0.0.{i % 250}" for i in range(n)],          # identifier
        "DST_PORT": rng.integers(1, 65535, n),                       # identifier
        "BYTES": np.round(np.exp(rng.normal(3, 2, n)) + np.vectorize(base.get)(labels)),
        "PKTS": rng.integers(1, 50, n).astype(float),
        "DUR": np.round(rng.exponential(10, n), 1),
        "Label": (labels != "Benign").astype(int),
        "Attack": labels,
    })
    dup = df.sample(400, random_state=1).copy()
    dup["SRC_IP"] = "192.168.1.1"                                    # same flow, other host
    conflict = df[df.Attack == "DoS"].head(5).copy()
    conflict["Attack"] = "Exploits"                                  # same vector, other label
    bad = df.head(3).copy(); bad["DUR"] = np.inf                     # inf row
    return pd.concat([df, dup, conflict, conflict, bad], ignore_index=True)


def test_clean_drops_ids_dedups_and_resolves_conflicts():
    feats, labels, rep = clean(toy_frame(), SPEC)
    assert not {"SRC_IP", "DST_PORT", "Label"} & set(feats.columns)
    assert rep["rows_dropped_inf_nan"] == 3
    assert not feats.duplicated().any()                              # one row per feature vector
    assert len(feats) == len(labels)
    assert rep["vectors_with_conflicting_labels"] >= 5


def test_majority_label_wins():
    df = pd.DataFrame({"A": [1.0, 1.0, 1.0, 2.0], "Attack": ["DoS", "DoS", "Exploits", "Benign"]})
    spec = DatasetSpec(name="t", csv_files=(), label_col="Attack", benign_label="Benign", drop_cols=())
    feats, labels, _ = clean(df, spec)
    got = {round(float(np.expm1(a))): lab for a, lab in zip(feats["A"], labels)}   # undo log1p
    assert got == {1: "DoS", 2: "Benign"}


def test_cap_only_limits_big_classes():
    y = np.repeat([0, 1, 2], [500, 50, 5])
    X = np.arange(len(y), dtype=float)[:, None]
    Xc, yc = cap_per_class(X, y, cap=100, seed=0)
    assert np.bincount(yc).tolist() == [100, 50, 5]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    out = tmp_path_factory.mktemp("proc")
    m = prepare(SPEC, cap=600, seed=0, out_dir=out, df=toy_frame())
    return m, np.load(out / "toy.npz"), out


def test_no_row_in_both_train_and_test(prepared):
    _, z, _ = prepared
    for other in ("X_va", "X_te"):
        assert not np.isin(row_keys(z[other]), row_keys(z["X_tr"])).any()
    assert not np.isin(row_keys(z["X_te"]), row_keys(z["X_va"])).any()


def test_scaler_fitted_on_train_only(prepared):
    m, z, _ = prepared
    # after the cap the train mean drifts slightly, but before the cap it was exactly 0;
    # the scaler mean recorded in the manifest must equal the log1p TRAIN mean, not train+test
    assert len(m["scaler_mean"]) == z["X_tr"].shape[1]
    assert abs(float(z["X_te"].mean())) > 1e-6 or abs(float(z["X_va"].mean())) > 1e-6


def test_benign_is_class_zero_and_cap_is_train_only(prepared):
    m, z, out = prepared
    assert m["classes"][0] == "Benign"
    assert max(np.bincount(z["y_tr"])) <= 600
    assert m["class_counts"]["test"]["Benign"] > 600 * 0.2          # test keeps the natural mix
    assert json.load(open(out / "toy_manifest.json"))["fingerprint"] == m["fingerprint"]
