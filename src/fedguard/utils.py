"""Small helpers shared across the project: seeding, hashing, paths."""
import hashlib
import random
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    """Fix every source of randomness we use (Python, NumPy, PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def array_fingerprint(*arrays) -> str:
    """Short hash of array contents, used to tie partition files to one processed dataset."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def row_keys(X: np.ndarray) -> np.ndarray:
    """One opaque key per row (its exact bytes), for overlap and duplicate checks with np.isin / np.unique."""
    X = np.ascontiguousarray(X)
    return X.view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()
