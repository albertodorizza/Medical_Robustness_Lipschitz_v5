"""
shared/wandb_logging.py
=======================
One place that decides where runs are logged, so the experiment scripts do
not each invent their own entity, project name and offline/online mode.

Naming convention
-----------------
    entity   medical_robustness            (the team; override with WANDB_ENTITY)
    project  <stage>-<task>                e.g. finetuning-segmentation
    run      <dataset>_<arch>_<freeze>_lambda<L>_seed<N>[_extra]

Six projects in total, one per (stage, task) pair, which keeps the run lists
short enough to browse while still separating things that are never compared
directly:

    finetuning-classification    finetuning-segmentation
    corruption-classification    corruption-segmentation
    drift-classification         drift-segmentation

The run name carries the dataset and the seed, so repetitions of the same
configuration are adjacent when sorted and are trivially grouped in the UI.
"""

import os

DEFAULT_ENTITY = "medical_robustness"

FINETUNING = "finetuning"
CORRUPTION = "corruption"
DRIFT = "drift"


def project_name(stage: str, task: str) -> str:
    """``"finetuning" + "segmentation" -> "finetuning-segmentation"``."""
    return f"{stage}-{task}"


def run_label(dataset: str, arch: str, freeze_mode: str, lambda_sn,
              split_seed: int = 0, extra: str = "") -> str:
    """
    Build a readable run name.
    """
    name = (f"{dataset}_{arch}_{freeze_mode}"
            f"_lambda{lambda_sn:g}_seed{int(split_seed)}")
    return f"{name}_{extra}" if extra else name


def init(stage: str, task: str, name: str, config: dict,
         mode: str = "online", enabled: bool = True):
    """
    Start a run, or do nothing when logging is disabled.

    Args:
        stage   : one of finetuning / corruption / drift.
        task    : ``"classification"`` or ``"segmentation"``.
        name    : run name
        config  : hyperparameters to record.
        mode    : ``"online"``, ``"offline"`` or ``"disabled"``.  Offline runs
                  are synced later with ``wandb sync <dir>``.
        enabled : ``False`` skips initialisation entirely and returns ``None``.
        
    Returns:
        The wandb run, or ``None`` when disabled.
    """
    if not enabled:
        return None

    import wandb

    return wandb.init(
        entity=os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY),
        project=project_name(stage, task),
        name=name,
        config=config,
        mode=mode,
        settings=wandb.Settings(init_timeout=180),
    )


def finish(enabled: bool = True) -> None:
    """Close the active run if there is one."""
    if not enabled:
        return
    import wandb
    if wandb.run is not None:
        wandb.finish()
