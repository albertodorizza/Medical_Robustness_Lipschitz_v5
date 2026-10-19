"""
shared/splits.py
================
Train / validation / test index splits, stratified and persisted to disk.

Why the split lives in a file instead of being re-derived from a seed
--------------------------------------------------------------------
Previously each script rebuilt the split independently with

    random_split(dataset, [0.8*n, 0.2*n], generator=manual_seed(29))

so ``finetuning_cls.py``, ``drift_analysis_cls.py`` and ``corruption_experiments_cls.py``
only agreed as long as the dataset length, the seed argument and the torch
version stayed identical.  Materialising the indices in a JSON file makes the
coupling explicit: every script loads the same file, and a mismatch fails
loudly instead of silently evaluating on different images.

The split is also three-way (train / val / test) rather than two-way, so that
hyperparameters and the stopping epoch can be selected on the validation split
while the test split stays untouched until the final evaluation.

Typical use::

    split = load_or_make_split(dataset, args.split_dir, args.split_seed)
    train_ds = Subset(dataset, split["train"])
    val_ds   = Subset(dataset, split["val"])
    test_ds  = Subset(dataset, split["test"])
"""

import hashlib
import json
import os

import numpy as np


# =============================================================================
# Split construction
# =============================================================================
def make_split(
    n: int,
    targets: list[int] | None = None,
    fractions: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 0,
) -> dict[str, list[int]]:
    """
    Build a train/val/test index split.

    Args:
        n         : Number of samples in the dataset.
        targets   : Per-sample class index.  When provided the split is
                    stratified, so each subset keeps the same class
                    proportions.  Pass ``None`` for a plain random split.
        fractions : ``(train, val, test)``; must sum to 1.
        seed      : RNG seed.  Vary it to obtain independent repetitions.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"``, each a sorted list
        of integer indices into the dataset.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {fractions}")

    rng = np.random.default_rng(seed)
    idx = np.arange(n)

    if targets is None:
        rng.shuffle(idx)
        n_tr = int(round(fractions[0] * n))
        n_va = int(round(fractions[1] * n))
        return {
            "train": sorted(int(i) for i in idx[:n_tr]),
            "val":   sorted(int(i) for i in idx[n_tr:n_tr + n_va]),
            "test":  sorted(int(i) for i in idx[n_tr + n_va:]),
        }

    # Stratified: split each class independently, then merge
    targets = np.asarray(targets)
    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for c in np.unique(targets):
        c_idx = idx[targets == c]
        rng.shuffle(c_idx)
        n_tr = int(round(fractions[0] * len(c_idx)))
        n_va = int(round(fractions[1] * len(c_idx)))
        out["train"] += c_idx[:n_tr].tolist()
        out["val"]   += c_idx[n_tr:n_tr + n_va].tolist()
        out["test"]  += c_idx[n_tr + n_va:].tolist()

    return {k: sorted(int(i) for i in v) for k, v in out.items()}


# =============================================================================
# Dataset introspection
# =============================================================================
def infer_targets(dataset) -> list[int] | None:
    """
    Extract one class index per sample, for stratification.

    Handles the dataset classes used in this project:
      * ``KvasirDatasetNpy`` / ``KvasirLoaderNpy`` -> ``dataset.labels``
      * ``KvasirDataset``                          -> ``dataset.samples``

    Returns ``None`` when the labels cannot be recovered, in which case the
    split falls back to a plain random partition.
    """
    labels = getattr(dataset, "labels", None)
    if labels is not None:
        return [int(v) for v in np.asarray(labels).reshape(-1)]

    samples = getattr(dataset, "samples", None)
    if samples is not None:
        return [int(lbl) for _, lbl in samples]

    return None


def _fingerprint(n: int, targets: list[int] | None) -> str:
    """
    Short hash of the dataset, used to detect a stale split file.

    The hash covers ALL labels, not a prefix: Kvasir-v2 is stored class by
    class, so any two label vectors agree on their first few hundred entries
    and a prefix hash would not notice a dataset change.
    """
    h = hashlib.sha256()
    h.update(str(n).encode())
    if targets is None:
        h.update(b"none")
    else:
        h.update(np.asarray(targets, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


# =============================================================================
# Persistence
# =============================================================================
def split_path(split_dir: str, split_seed: int) -> str:
    """Canonical location of the split file for a given seed."""
    return os.path.join(split_dir, f"split_seed{split_seed}.json")


def load_or_make_split(
    dataset,
    split_dir: str,
    split_seed: int = 0,
    fractions: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> dict[str, list[int]]:
    """
    Load ``<split_dir>/split_seed<seed>.json`` if present, otherwise create it.

    The file stores a fingerprint of the dataset.  If the dataset changes
    (different number of images, different labels) the fingerprint no longer
    matches and this function raises, rather than silently using a split that
    points at the wrong images.

    Args:
        dataset    : Any dataset exposing ``__len__``; labels are read through
                     :func:`infer_targets` when available.
        split_dir  : Directory holding the split files.
        split_seed : Seed identifying this particular partition.
        fractions  : ``(train, val, test)`` proportions, used only on creation.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"``.
    """
    os.makedirs(split_dir, exist_ok=True)
    path = split_path(split_dir, split_seed)

    n = len(dataset)
    targets = infer_targets(dataset)
    fp = _fingerprint(n, targets)

    if os.path.exists(path):
        with open(path) as f:
            blob = json.load(f)
        if blob.get("fingerprint") != fp:
            raise RuntimeError(
                f"Split file '{path}' was built for a different dataset "
                f"(fingerprint {blob.get('fingerprint')} != {fp}). "
                "Delete it explicitly if you intend to regenerate it."
            )
        return {k: blob[k] for k in ("train", "val", "test")}

    split = make_split(n, targets, fractions, split_seed)
    blob = {
        "seed":        split_seed,
        "fractions":   list(fractions),
        "n_samples":   n,
        "stratified":  targets is not None,
        "fingerprint": fp,
        **split,
    }
    with open(path, "w") as f:
        json.dump(blob, f, indent=2)

    print(f"[SPLIT] created {path}  "
          f"(train={len(split['train'])}, val={len(split['val'])}, "
          f"test={len(split['test'])}, stratified={targets is not None})")
    return split
