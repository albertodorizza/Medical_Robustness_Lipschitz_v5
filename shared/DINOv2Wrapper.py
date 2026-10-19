"""
shared/DINOv2Wrapper.py
=======================
Thin ``nn.Module`` wrapper that adapts a DINOv2 backbone to the same
classification interface used by torchvision's ResNet-34 and ViT-B/16.

All three supported architectures expose a ``model.head`` attribute via
:func:`shared.ModelFactory.get_head`, keeping the spectral-norm computation
in ``finetuning_cls.py`` and the hook registration in ``drift_analysis_cls.py``
architecture-agnostic.
"""

import torch
import torch.nn as nn

class DINOv2Wrapper(nn.Module):
    """
    Wraps a pretrained DINOv2 backbone with a linear classification head.

    DINOv2 (via ``torch.hub``) returns the CLS token as its global image
    representation.  This wrapper adds a single ``nn.Linear`` layer on top
    of those features so that DINOv2 can be used as a drop-in replacement
    for ResNet-34 or ViT-B/16 throughout the codebase.

    The head is accessible as ``model.head`` (same as
    :func:`shared.ModelFactory.get_head` expects).

    Args:
        backbone    : Pretrained DINOv2 model loaded via ``torch.hub``.
                      Expected to expose ``backbone.embed_dim`` and to return
                      a tensor of shape ``(B, embed_dim)`` from its
                      ``forward()`` (CLS token output).
        num_classes : Number of target classes for the classification head.

    Attributes:
        backbone (nn.Module) : Feature extractor (frozen or partially trainable,
                               depending on the ``freeze_mode`` set in
                               :func:`~finetuning.finetuning_cls.set_trainable_layers`).
        head (nn.Linear)     : Linear classification head mapping embeddings
                               to class logits.
    """

    def __init__(self, backbone, num_classes):
        super().__init__()

        # Feature extractor (e.g., DINOv2 ViT backbone)
        self.backbone = backbone

        # Linear classification head
        # Maps embedding space → class logits
        self.head = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, x):
        """
        Forward pass: extract CLS token features and project to logits.

        Args:
            x : Input images of shape ``(B, C, H, W)``.

        Returns:
            Logits tensor of shape ``(B, num_classes)``.
        """
        # DINOv2 backbone returns the CLS token as a (B, embed_dim) vector
        features = self.backbone(x)
        return self.head(features)