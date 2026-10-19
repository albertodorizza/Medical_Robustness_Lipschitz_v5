"""
drift_analysis/drift_common.py
==============================
Code shared by the two entry points of the ``drift_analysis`` stage.

``drift_analysis_cls.py`` (classification) and ``drift_analysis_seg.py``
(segmentation) validate the same two bounds on the same head, and differ only
in the model, in what a "prediction" is (one label vs one label per pixel) and
in the task metric the bound is finally expressed in.  Everything that is the
same for both lives here, so the two cannot drift apart: the head constants,
the architecture lookup, the perturbation naming and the shape of the result
file.

The result file
---------------
Both scripts write the same three blocks, so ``analysis.io`` reads
them with one loader and the corruption stage writes the same skeleton:

* ``metadata`` -- what was run and when.
* ``config``   -- the arguments needed to reproduce it.
* ``summary``  -- the scalars the tables use, with task-neutral names
  (``true_drop``, ``predicted_drop``, ``true_flip_rate``, ...), so the two
  tasks land in the same columns.
"""

import os
from datetime import datetime

import torch

from shared.ModelFactory import compute_spectral_norm, get_head


# =============================================================================
# Architecture lookup
# =============================================================================
# Short key used by the hooks and the checkpoints -> (model_type, arch), where
# model_type is the short name finetuning_cls.py stores and arch is the name
# ModelFactory / SegmentationModels expect.
ARCH_ALIASES = {
    "resnet":   ("resnet", "resnet34"),
    "resnet34": ("resnet", "resnet34"),
    "vit":      ("vit",    "vit_b_16"),
    "vit_b_16": ("vit",    "vit_b_16"),
    "dinov2":   ("dinov2", "dinov2"),
}


def resolve_architecture(model_path: str,
                         override: str | None = None) -> tuple[str, str]:
    """
    Return ``(model_type, arch)`` for a checkpoint.

    Checkpoints written by the current ``finetuning_cls.py`` carry an ``"arch"``
    field, which is authoritative.  Older ones do not, so the file name is used
    as a fallback; that path exists only for backwards compatibility.  An
    explicit ``override`` always wins, which is what lets the segmentation
    launcher pass ``--arch`` and skip the guessing entirely.

    Args:
        model_path : path to the ``.pth``.
        override   : value of ``--arch``, or None to read the checkpoint.

    Raises:
        ValueError : when the architecture cannot be determined.
    """
    if override:
        if override not in ARCH_ALIASES:
            raise ValueError(f"Unknown architecture '{override}'. "
                             f"Choose one of: {sorted(ARCH_ALIASES)}.")
        return ARCH_ALIASES[override]

    try:
        blob = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and blob.get("arch") in ARCH_ALIASES:
            return ARCH_ALIASES[blob["arch"]]
    except Exception as exc:
        print(f"  [WARN] could not read checkpoint metadata ({exc}); "
              "falling back to the file name.")

    name = os.path.basename(model_path).lower()
    for key in ("resnet", "vit", "dinov2"):
        if key in name:
            return ARCH_ALIASES[key]

    raise ValueError(
        f"Cannot determine the architecture for '{model_path}'. "
        "Pass --arch, or re-train with the current finetuning_cls.py so the "
        "checkpoint stores an 'arch' field."
    )


# =============================================================================
# Head constants
# =============================================================================
def head_gamma(model: torch.nn.Module) -> float:
    """
    ``Gamma = max_{a != b} ||w_a - w_b||_2``, the Lipschitz constant of the MARGIN.

    Always at most ``sqrt(2) * sigma_max(W)``: Gamma bounds how far the logit
    GAP between two classes can move, which is what decides the prediction,
    whereas sigma_max bounds how far the whole logit VECTOR can move.  Costs one
    K x K distance matrix.

    ``.flatten(1)`` is a no-op for the ``nn.Linear`` weight ``(K, D)`` of the
    classification head and turns the ``(K, D, 1, 1)`` kernel of the
    segmentation head into the same ``(K, D)`` matrix, which is why one
    function serves both tasks.
    """
    W = get_head(model).weight.detach().flatten(1)
    return torch.cdist(W, W, p=2).max().item()


def head_constants(model: torch.nn.Module) -> tuple[float, float]:
    """
    Return ``(sigma_max, gamma)`` of the head.

    ``sigma_max`` bounds the change of the logit vector and is what the paper
    reports next to the spectral penalty; ``gamma`` bounds the change of the
    logit gap, which is what decides whether a prediction changes.

    Both go through ``shared.ModelFactory.get_head``, so the classification
    head and the 1x1 convolution head of the segmentation models are handled
    by the same code and cannot diverge.
    """
    return compute_spectral_norm(model).item(), head_gamma(model)


# =============================================================================
# Naming
# =============================================================================
def perturbation_tag(kind: str, eps: float, severity: int) -> str:
    """
    Short, filesystem-safe name of a perturbation setting.

    ``gaussian_eps0.01`` for the random probe, ``Contrast_s3`` for an
    ImageNet-C key.  Used in the output file names and in the WandB run label,
    so the same setting is spelled the same way everywhere.
    """
    if kind.lower() == "gaussian":
        return f"gaussian_eps{eps:g}"
    return f"{kind.replace(' ', '')}_s{int(severity)}"


def is_gaussian(kind: str) -> bool:
    """Whether ``--perturbation`` selects the random probe rather than ImageNet-C."""
    return kind.lower() == "gaussian"


# =============================================================================
# Result file
# =============================================================================
def result_header(task: str, run_name: str, arch: str, args,
                  output_dir: str, n_eval: int) -> dict:
    """
    The ``metadata`` and ``config`` blocks shared by both tasks.

    Written by both scripts so that one loader in ``analysis.io`` can
    read either, and so that the drift files look like the corruption ones.

    Args:
        task       : ``"classification"`` or ``"segmentation"``.
        run_name   : output folder name, ``<arch>_<freeze>_lambda<L>_seed<N>``.
        arch       : full architecture name, e.g. ``"dinov2"``.
        args       : parsed command line.
        output_dir : where the results are written.
        n_eval     : number of images actually evaluated.
    """
    return {
        "metadata": {
            "task":       task,
            "model":      run_name,
            "arch":       arch,
            "model_path": args.model_path,
            "dataset":    args.dataset,
            "eval_split": args.eval_split,
            "n_eval_images": int(n_eval),
            "timestamp":  datetime.now().isoformat(),
            "output_dir": str(output_dir),
        },
        "config": {
            "arch":         arch,
            "dataset":      args.dataset,
            "model_path":   args.model_path,
            "eval_split":   args.eval_split,
            "perturbation": args.perturbation,
            "severity":     args.severity,
            "eps":          args.eps,
            "seed":         args.seed,
            "split_seed":   args.split_seed,
            "batch_size":   args.batch_size,
            "num_classes":  args.num_classes,
            "input_size":   list(args.input_size),
        },
    }


def headless_pyplot():
    """
    ``matplotlib.pyplot`` with the Agg backend, or None when it is missing.

    The sweep launchers run under ``nohup`` on nodes with no display, where the
    default backend either fails to import or blocks; Agg has to be selected
    before pyplot is first imported, which is why every plot goes through here.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib is not available, skipping the plot.")
        return None
    return plt
