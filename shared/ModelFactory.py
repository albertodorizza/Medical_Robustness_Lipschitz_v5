"""
shared/ModelFactory.py
======================
Centralised factory for building ResNet-34, ViT-B/16, and DINOv2 models
with an optional fine-tuned checkpoint, plus shared utilities for
accessing the classification head and computing its spectral norm.

Functions
---------
build_model            : Instantiate a backbone and optionally load a checkpoint.
get_head               : Return the ``nn.Linear`` classification head of any
                         supported model.
compute_spectral_norm  : Return the largest singular value of the head weight
                         matrix (differentiable; used both for regularization
                         in ``finetuning_cls.py`` and for bound analysis in
                         ``drift_analysis_cls.py``).
"""

import os

import torch
import torch.nn as nn
import torchvision.models as models

from shared.DINOv2Wrapper import DINOv2Wrapper

# ─────────────────────────────────────────────────────────────────────────────
#  Supported architecture identifiers
# ─────────────────────────────────────────────────────────────────────────────
SUPPORTED_ARCHS = ("resnet34", "vit_b_16", "dinov2")


def peek_num_classes(checkpoint: str) -> int:
    """
    Read ``num_classes`` from a checkpoint without building any model.

    ``finetuning_cls.py`` stores it at save time precisely so that evaluation
    scripts do not have to know it in advance -- this matters for Hyper-Kvasir
    and Kvasir-Capsule, whose class count is discovered from the folder tree
    and is therefore not a fixed constant like Kvasir-v2's 8.  Building the
    head with the wrong count and only then loading the checkpoint would fail
    with a shape mismatch in ``load_state_dict``, so this must be called BEFORE
    :func:`build_model` whenever ``--num_classes`` is not given explicitly.

    Args:
        checkpoint : Path to a ``.pth`` written by ``finetuning_cls.py``.

    Raises:
        FileNotFoundError : If the path does not exist.
        KeyError           : If the checkpoint predates this field.
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "num_classes" not in state:
        raise KeyError(
            f"'{checkpoint}' has no 'num_classes' field; it was probably "
            "written by an older version of finetuning_cls.py. Pass --num_classes "
            "explicitly instead."
        )
    return int(state["num_classes"])


def build_model(
    arch: str,
    num_classes: int,
    checkpoint: str | None = None,
    device: torch.device = torch.device("cpu"),
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a classification backbone and optionally load fine-tuned weights.

    The classification head is replaced with a fresh ``nn.Linear`` layer for
    ``num_classes`` classes **before** any checkpoint is loaded, so that the
    checkpoint's head weights are restored correctly.

    Supported architectures:

    +------------+-------------------------------------------+
    | ``arch``   | Backbone                                  |
    +============+===========================================+
    | resnet34   | ResNet-34 (torchvision)                   |
    +------------+-------------------------------------------+
    | vit_b_16   | ViT-B/16  (torchvision)                   |
    +------------+-------------------------------------------+
    | dinov2     | DINOv2 ViT-B/14 via torch.hub +           |
    |            | :class:`~shared.DINOv2Wrapper.DINOv2Wrapper` |
    +------------+-------------------------------------------+

    Args:
        arch        : Architecture identifier — one of ``SUPPORTED_ARCHS``.
        num_classes : Number of target output classes.
        checkpoint  : Path to a ``.pth`` file produced by ``finetuning_cls.py``.
                      The file may contain either a raw ``state_dict`` or a
                      dict with key ``"model_state_dict"`` or ``"model"``.
                      Pass ``None`` to skip weight loading.
        device      : Target torch device (CPU or CUDA).
        pretrained  : If ``True`` and ``checkpoint`` is ``None``, loads
                      ImageNet-pretrained weights.  For DINOv2 the weights
                      are always downloaded from ``facebookresearch/dinov2``.

    Returns:
        ``nn.Module`` in eval mode on ``device``.

    Raises:
        ValueError       : If ``arch`` is not in ``SUPPORTED_ARCHS``.
        FileNotFoundError: If ``checkpoint`` path does not exist on disk.
        RuntimeError     : If loading the checkpoint fails (e.g. shape mismatch).
    """
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Choose one of: {SUPPORTED_ARCHS}."
        )

    # ── Build backbone with fresh classification head ─────────────────────────
    if arch == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet34(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif arch == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.vit_b_16(weights=weights)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)

    elif arch == "dinov2":
        backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            pretrained=pretrained,
        )
        m = DINOv2Wrapper(backbone, num_classes)

    # ── Load fine-tuned checkpoint ────────────────────────────────────────────
    if checkpoint is not None:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        try:
            state = torch.load(checkpoint, map_location=device)
            # Support checkpoints that wrap the state_dict in a container dict
            if isinstance(state, dict):
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                elif "model" in state:
                    state = state["model"]
            m.load_state_dict(state, strict=True)
            print(f"  Loaded checkpoint: {checkpoint}")
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load checkpoint '{checkpoint}': {exc}"
            ) from exc

    return m.to(device).eval()


def get_head(model: nn.Module) -> nn.Linear:
    """
    Return the linear classification head of any supported model.

    Args:
        model : A model built by :func:`build_model`.

    Returns:
        The ``nn.Linear`` head module.

    Raises:
        ValueError : If no recognised head attribute is found on ``model``.
    """
    if hasattr(model, "fc"):
        return model.fc          # ResNet-34
    if hasattr(model, "heads"):
        return model.heads.head  # ViT-B/16
    if hasattr(model, "head"):
        return model.head        # DINOv2Wrapper
    raise ValueError(
        "Cannot locate the classification head. "
        "Expected model.fc, model.heads.head, or model.head."
    )


def compute_spectral_norm(model: nn.Module) -> torch.Tensor:
    """
    Return the spectral norm (largest singular value) of the classification
    head weight matrix.

    This function is shared between ``finetuning_cls.py`` (where the value is
    added to the loss and must remain attached to the computation graph) and
    ``drift_analysis_cls.py`` (where ``.item()`` is called to obtain a plain
    float for logging and bound computation).

    Concretely, it computes σ_max(W_head) via full SVD, which appears in
    the Lipschitz upper bound::

        ‖W·z − W·z̃‖₂  ≤  σ_max(W) · ‖z − z̃‖₂

    Head lookup is delegated to :func:`get_head` so that the same call works
    for ResNet-34, ViT-B/16, and DINOv2 without any architecture-specific
    branching.

    Args:
        model : Model instance built by :func:`build_model`.

    Returns:
        Scalar ``torch.Tensor`` (``requires_grad=True`` when called during
        training; call ``.item()`` to detach for logging or bound computation).
    """
    head = get_head(model)
    return torch.linalg.svdvals(head.weight.flatten(1))[0] # largest singular value σ_max
