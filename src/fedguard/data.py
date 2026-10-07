"""Leakage-safe data pipeline (plan Phase 2, Figure 7.1).

Order matters and must not change:
  1. drop identifier columns          (IPs, ports, random IDs)
  2. drop inf / NaN rows
  3. log1p(clip(x, 0)) as float32     fixed formula, learns nothing from the data, so no leakage;
                                      done BEFORE de-duplication because two different raw values can
                                      become the same float32 number, i.e. the same row for the model
  4. resolve conflicting labels       (same feature vector, different labels -> majority label)
  5. drop duplicate feature vectors   (BEFORE the split, so no row is in both train and test)
  6. stratified split                 train / val / test
  7. StandardScaler                   fitted on TRAIN only
  8. per-class cap                    TRAIN only; val and test keep the natural distribution
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fedguard.utils import REPO_ROOT, array_fingerprint, sha256_file


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv_files: tuple            # paths relative to the repo root
    label_col: str              # multi-class label column
    benign_label: str           # becomes class index 0
    drop_cols: tuple            # identifiers + redundant label columns, removed before anything else
    class_groups: dict = field(default_factory=dict)   # optional raw label -> grouped label


DATASETS = {
    "nf_unsw": DatasetSpec(
        name="nf_unsw",
        csv_files=("data/raw/NF-UNSW-NB15-v2/data/NF-UNSW-NB15-v2.csv",),
        label_col="Attack",
        benign_label="Benign",
        # IPs/ports identify hosts, DNS_QUERY_ID is a random transaction number: all are
        # identifiers a model could memorise. 'Label' is the binary twin of 'Attack'.
        drop_cols=("IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT",
                   "DNS_QUERY_ID", "Label"),
    ),
}


@dataclass
class ProcessedData:
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_va: np.ndarray
    y_va: np.ndarray
    X_te: np.ndarray
    y_te: np.ndarray
    classes: list
    features: list
    fingerprint: str

    @property
    def n_features(self):
        return self.X_tr.shape[1]

    @property
    def n_classes(self):
        return len(self.classes)


# ----------------------------------------------------------------------------- loading
def load_raw(spec: DatasetSpec, cache_dir=None) -> pd.DataFrame:
    """Read the raw CSV(s) once, then reuse a parquet copy (seconds instead of ~20 s)."""
    cache_dir = Path(cache_dir or REPO_ROOT / "data" / "processed")
    cache = cache_dir / f"{spec.name}_raw.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    frames = [pd.read_csv(REPO_ROOT / f, low_memory=False) for f in spec.csv_files]
    df = pd.concat(frames, ignore_index=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


# ----------------------------------------------------------------------------- cleaning
def clean(df: pd.DataFrame, spec: DatasetSpec):
    """Steps 1-5. Returns (log1p float32 features DataFrame, label Series, report dict)."""
    report = {"rows_raw": int(len(df))}
    df = df.drop(columns=[c for c in spec.drop_cols if c in df.columns])
    report["dropped_columns"] = [c for c in spec.drop_cols]
    if spec.class_groups:
        df[spec.label_col] = df[spec.label_col].map(lambda v: spec.class_groups.get(v, v))
    features = [c for c in df.columns if c != spec.label_col]
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")

    num = df[features].to_numpy(dtype=np.float64)
    bad = ~np.isfinite(num).all(axis=1)
    df = df.loc[~bad].copy()
    report["rows_dropped_inf_nan"] = int(bad.sum())
    df[features] = np.log1p(np.clip(num[~bad], 0, None)).astype(np.float32)

    # One row per unique feature vector. If a vector appears with several labels, keep the most
    # frequent label (ties -> alphabetical, so the result is deterministic).
    counts = df.groupby(features + [spec.label_col], sort=False).size().rename("n").reset_index()
    report["rows_exact_duplicates"] = int(len(df) - len(counts))
    counts = counts.sort_values(["n", spec.label_col], ascending=[False, True], kind="mergesort")
    n_labels = counts.groupby(features, sort=False)[spec.label_col].transform("nunique")
    report["vectors_with_conflicting_labels"] = int(counts.loc[n_labels > 1]
                                                    .drop_duplicates(subset=features).shape[0])
    unique = counts.drop_duplicates(subset=features, keep="first")
    report["rows_after_dedup"] = int(len(unique))
    report["class_counts_after_dedup"] = unique[spec.label_col].value_counts().to_dict()
    return unique[features].reset_index(drop=True), unique[spec.label_col].reset_index(drop=True), report


def class_order(labels, benign: str) -> list:
    """Benign first (index 0), then the attack classes alphabetically."""
    others = sorted(set(labels) - {benign})
    return [benign] + others


# ----------------------------------------------------------------------------- split / scale / cap
def split(X, y, seed, test_size=0.2, val_size=0.1):
    """Stratified train / val / test. val_size is a share of what is left after the test split."""
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=val_size, stratify=y_tmp,
                                              random_state=seed)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def fit_transform(X_tr, X_va, X_te):
    """Inputs are already log1p'ed (in clean). The scaler only ever sees TRAIN rows."""
    scaler = StandardScaler().fit(X_tr)
    out = [scaler.transform(a).astype(np.float32) for a in (X_tr, X_va, X_te)]
    return (*out, scaler)


def cap_per_class(X, y, cap, seed):
    """Keep at most `cap` rows per class (rare classes stay whole). TRAIN only."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        keep.append(idx if len(idx) <= cap else rng.choice(idx, cap, replace=False))
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep]


# ----------------------------------------------------------------------------- full pipeline
def prepare(spec: DatasetSpec, cap=20_000, seed=42, out_dir=None, df=None):
    """Run steps 1-7 and save <name>.npz + <name>_manifest.json. Returns the manifest dict."""
    out_dir = Path(out_dir or REPO_ROOT / "data" / "processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    from_disk = df is None              # tests pass a small DataFrame instead of the real CSV
    df = load_raw(spec) if from_disk else df
    feats, labels, report = clean(df, spec)

    classes = class_order(labels.unique(), spec.benign_label)
    cmap = {c: i for i, c in enumerate(classes)}
    X = feats.to_numpy(dtype=np.float32)
    y = labels.map(cmap).to_numpy(dtype=np.int64)

    X_tr, y_tr, X_va, y_va, X_te, y_te = split(X, y, seed)
    X_tr, X_va, X_te, scaler = fit_transform(X_tr, X_va, X_te)
    rows_train_before_cap = len(y_tr)
    X_tr, y_tr = cap_per_class(X_tr, y_tr, cap, seed)

    fp = array_fingerprint(X_tr, y_tr, X_va, y_va, X_te, y_te)
    np.savez_compressed(out_dir / f"{spec.name}.npz", X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va,
                        X_te=X_te, y_te=y_te)
    count = lambda yy: {classes[i]: int(n) for i, n in zip(*np.unique(yy, return_counts=True))}
    manifest = {
        "dataset": spec.name,
        "source_files": {f: sha256_file(REPO_ROOT / f) for f in spec.csv_files} if from_disk else {},
        "fingerprint": fp,
        "seed": seed,
        "per_class_cap_train": cap,
        "split": {"test_size": 0.2, "val_size_of_rest": 0.1, "stratified": True},
        "transform": "log1p(clip(x, 0)) as float32 before de-duplication, then StandardScaler fitted on train only",
        "classes": classes,
        "features": list(feats.columns),
        "cleaning": report,
        "rows": {"train_before_cap": rows_train_before_cap, "train": int(len(y_tr)),
                 "val": int(len(y_va)), "test": int(len(y_te))},
        "class_counts": {"train": count(y_tr), "val": count(y_va), "test": count(y_te)},
        "scaler_mean": scaler.mean_.round(6).tolist(),
        "scaler_scale": scaler.scale_.round(6).tolist(),
    }
    with open(out_dir / f"{spec.name}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_processed(name="nf_unsw", data_dir=None) -> ProcessedData:
    data_dir = Path(data_dir or REPO_ROOT / "data" / "processed")
    z = np.load(data_dir / f"{name}.npz")
    with open(data_dir / f"{name}_manifest.json") as f:
        m = json.load(f)
    arrays = {k: z[k] for k in ("X_tr", "y_tr", "X_va", "y_va", "X_te", "y_te")}
    fp = array_fingerprint(*arrays.values())
    if fp != m["fingerprint"]:
        raise ValueError(f"{name}.npz does not match its manifest (re-run scripts/prepare_data.py)")
    return ProcessedData(**arrays, classes=m["classes"], features=m["features"], fingerprint=fp)
