"""
shared/runtime.py
=================
Small runtime helpers shared by the entry-point scripts.

``pin_cpu_cores`` replaces the inline block that each script used to run at
import time::

    _cores_to_use = 20
    _num_cores = os.cpu_count()
    os.sched_setaffinity(0, set(range(_num_cores - _cores_to_use, _num_cores)))
"""

import os


def pin_cpu_cores(n_cores: int | None, torch_threads: int = 2) -> None:
    """
    Pin the process to the last ``n_cores`` CPUs and cap the torch thread pool.

    Args:
        n_cores       : How many cores to reserve, counted from the end of the
                        core list.  Pass ``None`` or 0 to leave affinity
                        untouched 
        torch_threads : Value for ``torch.set_num_threads`` and
                        ``set_num_interop_threads``.

    Never raises: an unsupported platform or an out-of-range core count only
    produces a warning.
    """
    import torch

    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(torch_threads)
    except RuntimeError:
        # Can only be set once per process, before any parallel work starts.
        pass

    if not n_cores:
        return

    if not hasattr(os, "sched_setaffinity"):
        print("[RUNTIME] CPU pinning not supported on this platform, skipping.")
        return

    total = os.cpu_count() or 1
    if n_cores > total:
        print(f"[RUNTIME] requested {n_cores} cores but only {total} available, "
              "skipping CPU pinning.")
        return

    try:
        os.sched_setaffinity(0, set(range(total - n_cores, total)))
        print(f"[RUNTIME] pinned to the last {n_cores} of {total} cores.")
    except OSError as exc:
        print(f"[RUNTIME] could not set CPU affinity ({exc}), continuing.")
