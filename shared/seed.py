"""
shared/seed.py
==============
Reproducibility utilities shared across all modules.

What ``reset_seed`` covers, and what it deliberately does not
------------------------------------------------------------
Seeding ``torch`` / ``numpy`` / ``random`` and locking cuDNN fixes the
shuffling order, the weight initialisation and the convolution algorithm
choice.  Three things it did NOT cover before, and now does:

* **PYTHONHASHSEED.** Set here for completeness, but note the interpreter
  reads it at startup: exporting it from the launcher is what actually makes
  ``hash()`` stable.  The corruption pipeline never relies on ``hash()``
  (``corruption_common.corruption_seed_key`` uses ``zlib.crc32`` precisely for
  this reason), so this is a belt-and-braces measure.
* **DataLoader workers.** ``worker_init_fn`` and a seeded ``generator`` are
  needed whenever a worker draws random numbers; without them the per-worker
  numpy / random state depends on the worker count.  The datasets in this
  project apply no random augmentation, so this changes nothing today, but it
  removes a silent failure mode the moment augmentation is added.
* **Deterministic kernels.** ``torch.use_deterministic_algorithms`` is opt-in
  (``strict=True``) rather than always-on: some backward kernels have no
  deterministic implementation and would raise, which would be a worse
  outcome than a bitwise-different-but-statistically-identical run.
"""

import os
import random

import numpy as np
import torch


def reset_seed(seed: int, strict: bool = False) -> None:
    """
    Set all relevant random seeds for reproducible experiments.

    Covers PyTorch (CPU + CUDA), NumPy and Python's ``random``.  Locks cuDNN
    to deterministic mode and disables the auto-tuner so that convolution
    algorithm selection is stable across runs.

    Args:
        seed   : Integer seed value.
        strict : Also request deterministic CUDA kernels via
                 ``torch.use_deterministic_algorithms(True)`` and set
                 ``CUBLAS_WORKSPACE_CONFIG``.  Off by default because a few
                 backward kernels have no deterministic implementation and
                 would raise at training time.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if strict:
        # cuBLAS needs a fixed workspace for its reductions to be reproducible.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:                       # pragma: no cover
            print(f"[SEED] deterministic algorithms unavailable ({exc}), "
                  "continuing without them.")


def seed_worker(worker_id: int) -> None:
    """
    ``worker_init_fn`` for :class:`torch.utils.data.DataLoader`.

    PyTorch gives each worker a distinct torch seed derived from the parent
    generator, but leaves numpy and ``random`` untouched.  Deriving both from
    that torch seed makes a worker's random state a function of (parent seed,
    worker id) only, so the same data is produced whatever ``num_workers`` is.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def loader_generator(seed: int) -> torch.Generator:
    """
    Seeded generator for a shuffling :class:`~torch.utils.data.DataLoader`.

    Passing this as ``generator=`` makes the shuffling order depend on ``seed``
    alone, rather than on how much of the global torch RNG the rest of the
    script happened to consume before the loader was built.
    """
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g
