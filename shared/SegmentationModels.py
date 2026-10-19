"""
shared/SegmentationModels.py
============================
Segmentation counterparts of the classification models built by
:mod:`shared.ModelFactory`.

Why a separate wrapper is needed
--------------------------------
The classification backbones destroy spatial information *inside* their own
``forward()``, before anything reaches the head:

* ResNet-34 : ``avgpool`` collapses ``(B, 512, 7, 7)`` to ``(B, 512)``
* ViT-B/16  : ``x = x[:, 0]`` keeps only the CLS token and discards the patches
* DINOv2    : ``backbone(x)`` returns the pooled CLS vector

So swapping the head is not enough; the wrappers below stop the backbone
*before* the pooling step and expose the full feature map.

The whole Lipschitz analysis requires a head whose constant is exactly
computable.  A 1x1 convolution is the same affine map applied independently at
every spatial location, therefore:
* its Lipschitz constant is ``sigma_max(W)`` with ``W`` of shape ``(K, D)``,
  exactly as in the classification case, so ``compute_spectral_norm`` works
  unchanged;
* the per-pixel margin analysis carries over with no modification;
* the spectral-norm penalty is the same one-line term.

One consequence worth stating in the paper: a 1x1 convolution commutes with
bilinear upsampling, because both are linear and the convolution is applied
identically at every location.  Margins and feature drift can therefore be
measured on the low-resolution grid, before upsampling, and the result is the
same as measuring them at full resolution.  ``forward(return_features=True)``
exposes exactly that.

Architectures and feature grids at 224x224 input:

+------------+---------------------+-----------+----------------+
| ``arch``   | Feature source      | Grid      | Embedding dim  |
+============+=====================+===========+================+
| resnet34   | ``layer4`` output   | 7 x 7     | 512            |
+------------+---------------------+-----------+----------------+
| vit_b_16   | patch tokens        | 14 x 14   | 768            |
+------------+---------------------+-----------+----------------+
| dinov2     | ``x_norm_patchtokens`` | 16 x 16 | 768           |
+------------+---------------------+-----------+----------------+
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

SUPPORTED_ARCHS = ("resnet34", "vit_b_16", "dinov2")


# =============================================================================
# Backbone feature extractors
# =============================================================================
class ResNetFeatures(nn.Module):
    """
    ResNet-34 truncated after ``layer4``, returning ``(B, 512, H/32, W/32)``.

    The submodules keep their original names (``layer1`` ... ``layer4``) so
    that the freezing logic can address them exactly as in the classification
    pipeline.
    """

    embed_dim = 512

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet34(weights=weights)
        self.conv1, self.bn1, self.relu = m.conv1, m.bn1, m.relu
        self.maxpool = m.maxpool
        self.layer1, self.layer2 = m.layer1, m.layer2
        self.layer3, self.layer4 = m.layer3, m.layer4

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)                    # (B, 512, Hf, Wf)


class ViTFeatures(nn.Module):
    """
    ViT-B/16 patch tokens reshaped to a spatial grid, ``(B, 768, H/16, W/16)``.

    torchvision's ``forward()`` selects ``x[:, 0]`` (the CLS token) right after
    the encoder, so the encoder is called directly here and the CLS token is
    dropped instead, keeping the patch tokens.
    """

    embed_dim = 768

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = models.vit_b_16(weights=weights)
        self.vit.heads = nn.Identity()           # never used, drop the weights

    def forward(self, x):
        v = self.vit
        x = v._process_input(x)                  # (B, N, D)
        cls = v.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)           # (B, N+1, D)
        x = v.encoder(x)                         # includes the final LayerNorm
        x = x[:, 1:, :]                          # drop CLS -> (B, N, D)
        return tokens_to_grid(x)


class DINOv2Features(nn.Module):
    """
    DINOv2 ViT-B/14 patch tokens as a spatial grid, ``(B, 768, H/14, W/14)``.

    ``forward()`` on the hub model returns only the pooled CLS vector, so
    ``forward_features()`` is used instead: it returns a dict whose
    ``x_norm_patchtokens`` entry holds the normalised patch tokens.
    """

    embed_dim = 768

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", pretrained=pretrained
        )
        self.embed_dim = self.backbone.embed_dim

    def forward(self, x):
        out = self.backbone.forward_features(x)
        return tokens_to_grid(out["x_norm_patchtokens"])


def tokens_to_grid(tokens: torch.Tensor) -> torch.Tensor:
    """
    Reshape a patch-token sequence ``(B, N, D)`` into a grid ``(B, D, s, s)``.

    Args:
        tokens : Patch tokens, with any non-spatial prefix tokens already
                 removed by the caller.

    Raises:
        ValueError : If ``N`` is not a perfect square, which means a prefix
                     token (CLS or a DINOv2 register) was not stripped.
    """
    B, N, D = tokens.shape
    side = int(round(math.sqrt(N)))
    if side * side != N:
        raise ValueError(
            f"{N} patch tokens do not form a square grid; a CLS or register "
            "token was probably left in the sequence."
        )
    return tokens.transpose(1, 2).reshape(B, D, side, side).contiguous()


# =============================================================================
# Segmentation model
# =============================================================================
class SegmentationModel(nn.Module):
    """
    Backbone feature map -> 1x1 convolution head -> bilinear upsampling.

    Attributes:
        backbone (nn.Module) : One of the feature extractors above.  Exposes
                               ``embed_dim``.
        head (nn.Conv2d)     : 1x1 convolution mapping ``embed_dim`` channels
                               to ``num_classes`` logits per location.  Named
                               ``head`` so that ``ModelFactory.get_head`` and
                               ``compute_spectral_norm`` work unchanged.

    Args:
        backbone    : Feature extractor.
        num_classes : Number of segmentation classes (2 for binary: background
                      and foreground).
        out_size    : Output resolution ``(H, W)``.  ``None`` keeps the input
                      resolution, which is what the training loop wants.
    """

    def __init__(self, backbone: nn.Module, num_classes: int,
                 out_size: tuple[int, int] | None = None):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.out_size = out_size
        self.head = nn.Conv2d(backbone.embed_dim, num_classes, kernel_size=1)

    def head_from_features(self, feats: torch.Tensor,
                           size: tuple[int, int]) -> torch.Tensor:
        """
        Everything ``forward`` does after the backbone: 1x1 convolution, then
        bilinear upsampling to ``size``.

        Split out so that the cached-feature training path in
        ``finetuning_seg.py`` (``freeze_mode="head"``, see
        :mod:`shared.featcache`) runs the *same* code as the normal path
        instead of a copy of it that could drift out of step.

        Args:
            feats : Backbone feature grid ``(B, D, Hf, Wf)``.
            size  : Output resolution ``(H, W)``.
        """
        logits_lr = self.head(feats)             # (B, K, Hf, Wf)
        return F.interpolate(logits_lr, size=size, mode="bilinear",
                             align_corners=False)

    def forward(self, x, return_features: bool = False):
        """
        Args:
            x               : Input images ``(B, 3, H, W)``.
            return_features : If ``True``, also return the pre-head feature map
                              and the low-resolution logits.  Used by the drift
                              and margin analyses, which work on the feature
                              grid rather than at full resolution.

        Returns:
            ``logits`` of shape ``(B, K, H, W)``, or the triple
            ``(logits, features, logits_lowres)`` when ``return_features``.
        """
        size = self.out_size or x.shape[-2:]
        feats = self.backbone(x)                 # (B, D, Hf, Wf)
        logits_lr = self.head(feats)             # (B, K, Hf, Wf)
        logits = F.interpolate(logits_lr, size=size, mode="bilinear",
                               align_corners=False)
        if return_features:
            return logits, feats, logits_lr
        return logits


# =============================================================================
# Factory
# =============================================================================
def build_segmentation_model(
    arch: str,
    num_classes: int = 2,
    checkpoint: str | None = None,
    device: torch.device = torch.device("cpu"),
    pretrained: bool = True,
) -> SegmentationModel:
    """
    Build a segmentation model, optionally loading fine-tuned weights.

    Mirrors :func:`shared.ModelFactory.build_model`: the head is created for
    ``num_classes`` **before** the checkpoint is loaded, so the saved head
    weights are restored correctly.

    Args:
        arch        : One of ``SUPPORTED_ARCHS``.
        num_classes : 2 for binary polyp segmentation.
        checkpoint  : ``.pth`` produced by ``finetuning_seg.py``; may hold a raw
                      ``state_dict`` or a dict with ``"model_state_dict"``.
        device      : Target device.
        pretrained  : Load ImageNet / DINOv2 pretrained backbone weights when
                      no checkpoint is given.

    Returns:
        :class:`SegmentationModel` in eval mode on ``device``.
    """
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Unknown architecture '{arch}'. "
                         f"Choose one of: {SUPPORTED_ARCHS}.")

    if arch == "resnet34":
        backbone = ResNetFeatures(pretrained)
    elif arch == "vit_b_16":
        backbone = ViTFeatures(pretrained)
    else:
        backbone = DINOv2Features(pretrained)

    model = SegmentationModel(backbone, num_classes)

    if checkpoint is not None:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=True)
        print(f"  Loaded checkpoint: {checkpoint}")

    return model.to(device).eval()


def set_trainable_layers_seg(model: SegmentationModel, freeze_mode: str) -> None:
    """
    Freeze the backbone and selectively unfreeze parts of it.

    Uses the same two strategies as the classification pipeline, so that the
    segmentation results are directly comparable:

    * ``"head"``      : only the 1x1 convolution head is trainable.
    * ``"head+last"`` : head plus the last backbone stage — ``layer4`` for
                        ResNet-34, the last 4 encoder blocks and the final
                        LayerNorm for ViT-B/16 and DINOv2.

    This function ONLY sets ``requires_grad``.  It used to also build and
    return an ``Adam(lr=1e-4)``, which ``finetuning_seg.py`` threw away and
    rebuilt through ``build_optimizer``: the hard-coded learning rate silently
    ignored ``--lr``, ``--optimizer`` and ``--weight_decay`` for anyone calling
    this directly.  Build the optimizer at the call site, after this returns.

    Args:
        model       : A :class:`SegmentationModel`.
        freeze_mode : ``"head"`` or ``"head+last"``.

    Raises:
        ValueError : If ``freeze_mode`` is not recognised.
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    if freeze_mode == "head+last":
        bb = model.backbone
        if isinstance(bb, ResNetFeatures):
            for p in bb.layer4.parameters():
                p.requires_grad = True
        elif isinstance(bb, ViTFeatures):
            layers = bb.vit.encoder.layers
            for i in range(len(layers) - 4, len(layers)):
                for p in layers[i].parameters():
                    p.requires_grad = True
            for p in bb.vit.encoder.ln.parameters():
                p.requires_grad = True
        elif isinstance(bb, DINOv2Features):
            for block in bb.backbone.blocks[-4:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in bb.backbone.norm.parameters():
                p.requires_grad = True
    elif freeze_mode != "head":
        raise ValueError(f"Unknown freeze_mode '{freeze_mode}'. "
                         "Choose 'head' or 'head+last'.")
