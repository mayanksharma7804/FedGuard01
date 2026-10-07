"""Checks on the REAL processed NF-UNSW-NB15-v2 files. Skipped until scripts/prepare_data.py has run."""
import json

import numpy as np
import pytest

from fedguard.data import DATASETS, load_processed
from fedguard.partition import load_partition, partition_path
from fedguard.utils import REPO_ROOT, row_keys

NPZ = REPO_ROOT / "data" / "processed" / "nf_unsw.npz"
pytestmark = pytest.mark.skipif(not NPZ.exists(), reason="run scripts/prepare_data.py first")


@pytest.fixture(scope="module")
def data():
    return load_processed("nf_unsw")


def test_no_identifier_columns(data):
    assert not set(DATASETS["nf_unsw"].drop_cols) & set(data.features)
    assert len(data.features) == data.n_features == 38


def test_splits_do_not_share_rows(data):
    tr, va, te = row_keys(data.X_tr), row_keys(data.X_va), row_keys(data.X_te)
    assert not np.isin(te, tr).any()
    assert not np.isin(va, tr).any()
    assert not np.isin(te, va).any()


def test_no_duplicate_rows_inside_train(data):
    assert len(np.unique(row_keys(data.X_tr))) == len(data.X_tr)


def test_every_class_in_every_split(data):
    for y in (data.y_tr, data.y_va, data.y_te):
        assert len(np.unique(y)) == data.n_classes


def test_cap_respected(data):
    m = json.load(open(REPO_ROOT / "data" / "processed" / "nf_unsw_manifest.json"))
    assert np.bincount(data.y_tr).max() <= m["per_class_cap_train"]


@pytest.mark.parametrize("alpha", [0.1, 0.5, 100])
@pytest.mark.parametrize("k", [5, 10])
def test_saved_partitions_match_data(data, alpha, k):
    path = partition_path("nf_unsw", alpha, k, 42)
    if not path.exists():
        pytest.skip("run scripts/make_partitions.py first")
    parts, meta = load_partition(path, fingerprint=data.fingerprint)
    allidx = np.concatenate(parts)
    assert len(allidx) == len(np.unique(allidx)) == len(data.y_tr)
