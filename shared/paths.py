"""
shared/paths.py
===============
One definition of where results are written, shared by every script.

Layout
------
Every artefact lives under ``<results_root>/<stage>/<task>/<dataset>/<run>``:

    OUTPUTS_MEDICAL/
      finetuning/
        classification/kvasir_v2/dinov2_head+last_lambda100/
            dinov2_head+last.pth
            dinov2_head+last_log.json
        segmentation/kvasir_seg/dinov2_head+last_lambda100_seed1/
      corruption_experiments/
        classification/kvasir_v2/dinov2_head+last_lambda100_rawfeat/
            metrics.json
        segmentation/kvasir_seg/dinov2_head+last_lambda100_on_etis_larib_rawfeat/
      drift_analysis/
        classification/kvasir_v2/drift_gaussian/dinov2_head+last_lambda100/
        segmentation/kvasir_seg/drift_Contrast/dinov2_head+last_lambda100/

Run names
---------
``<arch>_<freeze>_lambda<L>[_seed<N>]``, e.g. ``dinov2_head+last_lambda100`` or
``resnet_head_lambda0_seed2``.   That makes the scheme multi-seed safe: two repetitions of the same
configuration differ by the suffix and never overwrite each other, and
``analysis.io`` recovers the seed from the same suffix in order to
average over repetitions.
"""

import os

# Stage names, kept as constants so a typo fails at import rather than
# silently creating a new folder tree.
FINETUNING = "finetuning"
CORRUPTION = "corruption_experiments"
DRIFT = "drift_analysis"

CLASSIFICATION = "classification"
SEGMENTATION = "segmentation"


def run_name(arch: str, freeze_mode: str, lambda_sn, split_seed: int = 0) -> str:
    """
    Canonical name of a single fine-tuning configuration.

    Args:
        arch        : short architecture name, e.g. ``"dinov2"``.
        freeze_mode : ``"head"`` or ``"head+last"``.
        lambda_sn   : spectral-norm coefficient; formatted with %g so 100.0
                      and 100 give the same string.
        split_seed  :  append ``_seed<N>``.

    Returns:
        e.g. ``"resnet_head_lambda0_seed2"``.
    """
    
    return f"{arch}_{freeze_mode}_lambda{lambda_sn:g}_seed{int(split_seed)}"


def ensure_seed_suffix(name: str, split_seed: int) -> str:
    """
    Guarantee that a run folder name carries an explicit ``_seed<N>``.

    Older checkpoints, and hand-made folders, may have no suffix at all.
    Adding it means seed 0 is spelled out like every other seed, so runs of the
    same configuration sort together and :func:`parse_run_name` never has to
    fall back to a default.

    Args:
        name       : folder name, e.g. ``dinov2_head+last_lambda100``.
        split_seed : seed to append when the name does not already carry one.
    """
    return name if "_seed" in name else f"{name}_seed{int(split_seed)}"


def run_folder_from_checkpoint(checkpoint: str | None, fallback: str,
                               split_seed: int) -> str:
    """
    Name of the evaluation run folder derived from a checkpoint path.

    Every evaluation stage (corruption, drift) writes its results under a
    folder named after the fine-tuning run that produced the checkpoint, so
    that ``parse_run_name`` recovers the architecture, the freeze mode, lambda
    and the seed from the path alone.  This is the single definition of that
    rule.

    Args:
        checkpoint : path to the ``.pth``; the parent folder name is used.
        fallback   : base name when there is no checkpoint (e.g. the arch).
        split_seed : seed to append when the folder name does not carry one.
    """
    base = (os.path.basename(os.path.dirname(checkpoint)) if checkpoint
            else str(fallback))
    return ensure_seed_suffix(base, split_seed)


def stage_dir(results_root: str, stage: str, task: str, dataset: str) -> str:
    """
    Directory holding every run of one (stage, task, dataset) combination.

    This is what the scripts receive as ``--output_dir`` / ``--save_root``
    parent; each script then creates its own run subfolder inside it.
    """
    return os.path.join(results_root, stage, task, dataset)


def split_file_dir(split_root: str, dataset: str) -> str:
    """
    Directory holding the split files of one dataset.

    Splits are namespaced by dataset but NOT by task: a dataset belongs to a
    single task, so ``<split_root>/<dataset>/split_seed<N>.json`` is already
    unambiguous, and keeping it flat means the training script and every
    evaluation script resolve the same path with the same two inputs.
    """
    return os.path.join(split_root, dataset)


def parse_run_name(name: str) -> dict:
    """
    Inverse of :func:`run_name`, used by the result-collection scripts.

    Returns a dict with ``arch``, ``freeze``, ``lambda_sn`` and ``seed``.
    Unparseable names return the raw string under ``arch`` and seed 0, so a
    stray folder degrades to a visible oddity in a table instead of crashing
    the whole aggregation.
    """
    seed = 0
    base = name
    if "_seed" in base:
        base, _, tail = base.rpartition("_seed")
        if tail.isdigit():
            seed = int(tail)
        else:                       # not actually a seed suffix, put it back
            base = name

    if "_lambda" not in base:
        return {"arch": name, "freeze": "?", "lambda_sn": None, "seed": seed}

    head, _, lam = base.rpartition("_lambda")
    arch, _, freeze = head.partition("_")
    try:
        lam_val = float(lam)
    except ValueError:
        lam_val = None
    return {"arch": arch, "freeze": freeze, "lambda_sn": lam_val, "seed": seed}
