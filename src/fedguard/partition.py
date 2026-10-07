"""Non-IID client partitions with a Dirichlet split (plan section 7.2, Figure 7.2).

Small alpha (0.1) = every client sees only a few classes; alpha = 100 = almost IID.
Partitions are saved once and reused by every experiment, so all methods train on exactly the
same split. Each file stores the fingerprint of the processed data it was made from.
"""
import json
from pathlib import Path

import numpy as np

from fedguard.utils import REPO_ROOT


def dirichlet_partition(y, n_clients, alpha, min_size=200, seed=42, max_tries=1000):
    """Split row indices of `y` into `n_clients` disjoint parts.

    For every class, a share vector p ~ Dirichlet(alpha) decides how that class's rows are
    divided among clients. Retries until every client has at least `min_size` rows.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        parts = [[] for _ in range(n_clients)]
        for c in np.unique(y):
            ids = rng.permutation(np.flatnonzero(y == c))
            p = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(p) * len(ids)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(ids, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.array(sorted(p), dtype=np.int64) for p in parts]
    raise RuntimeError(f"Could not give every client >= {min_size} rows with alpha={alpha}, "
                       f"{n_clients} clients in {max_tries} tries. Lower min_size.")


def partition_path(dataset, alpha, n_clients, seed, root=None):
    root = Path(root or REPO_ROOT / "partitions")
    return root / f"{dataset}_alpha{alpha:g}_k{n_clients}_seed{seed}.npz"


def class_counts(parts, y, n_classes):
    return [np.bincount(y[p], minlength=n_classes).tolist() for p in parts]


def save_partition(parts, y, n_classes, *, dataset, alpha, seed, min_size, fingerprint, root=None):
    path = partition_path(dataset, alpha, len(parts), seed, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"dataset": dataset, "alpha": alpha, "n_clients": len(parts), "seed": seed,
            "min_size": min_size, "data_fingerprint": fingerprint,
            "sizes": [int(len(p)) for p in parts], "class_counts": class_counts(parts, y, n_classes)}
    np.savez_compressed(path, meta=np.array(json.dumps(meta)),
                        **{f"client_{k}": p for k, p in enumerate(parts)})
    return path


def load_partition(path, fingerprint=None):
    """Return (list of index arrays, meta dict). Refuses a file made from different data."""
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    if fingerprint is not None and meta["data_fingerprint"] != fingerprint:
        raise ValueError(f"{Path(path).name} was made from different processed data "
                         f"({meta['data_fingerprint']} != {fingerprint}); re-run scripts/make_partitions.py")
    parts = [z[f"client_{k}"] for k in range(meta["n_clients"])]
    return parts, meta
