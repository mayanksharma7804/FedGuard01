import numpy as np
import pytest

from fedguard.partition import dirichlet_partition, load_partition, save_partition

Y = np.repeat(np.arange(6), [3000, 1500, 800, 400, 120, 30])


@pytest.mark.parametrize("alpha", [0.1, 0.5, 100])
@pytest.mark.parametrize("k", [5, 10])
def test_disjoint_cover_and_min_size(alpha, k):
    parts = dirichlet_partition(Y, k, alpha, min_size=50, seed=1)
    allidx = np.concatenate(parts)
    assert len(allidx) == len(Y)                          # covers every row ...
    assert len(np.unique(allidx)) == len(Y)               # ... exactly once (disjoint)
    assert min(len(p) for p in parts) >= 50


def test_same_seed_same_partition():
    a = dirichlet_partition(Y, 5, 0.5, min_size=50, seed=7)
    b = dirichlet_partition(Y, 5, 0.5, min_size=50, seed=7)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))


def test_small_alpha_is_more_skewed_than_large_alpha():
    def mean_classes(alpha):
        parts = dirichlet_partition(Y, 5, alpha, min_size=20, seed=3)
        return np.mean([np.mean(np.bincount(Y[p], minlength=6) > 0.05 * len(p)) for p in parts])
    assert mean_classes(0.1) < mean_classes(100)


def test_impossible_min_size_raises():
    with pytest.raises(RuntimeError):
        dirichlet_partition(Y, 10, 0.1, min_size=len(Y), seed=0, max_tries=5)


def test_save_load_roundtrip_and_fingerprint_guard(tmp_path):
    parts = dirichlet_partition(Y, 5, 0.5, min_size=50, seed=2)
    path = save_partition(parts, Y, 6, dataset="toy", alpha=0.5, seed=2, min_size=50,
                          fingerprint="abc", root=tmp_path)
    loaded, meta = load_partition(path, fingerprint="abc")
    assert all(np.array_equal(x, z) for x, z in zip(parts, loaded))
    assert meta["sizes"] == [len(p) for p in parts]
    with pytest.raises(ValueError):
        load_partition(path, fingerprint="different")
