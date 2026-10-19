"""
corruption_experiments/corruption_metrics.py
============================================
Feature-drift metrics (MSE and cosine similarity) between clean and corrupted
features, restricted to a region of the image.

These are the feature side of the decomposition both corruption scripts use:
the same quantity measured inside the corrupted region, outside it, and over
the whole image.  Comparing ``MSE_in`` with ``MSE_out`` is what shows the
corruption propagating beyond its own spatial support.

Three choices that are not obvious
----------------------------------
**An empty mask returns NaN, not 0.0.**  At ``perturbation_ratio = 0.9`` the
non-corrupted region can disappear entirely once it is downsampled onto the
feature grid.  A 0.0 would enter the average as if it were a measurement and
pull ``MSE_out`` down; NaN says there is no measurement.

**The mask is projected by area, the same way for every architecture.**  The
correct weight for a feature cell is the FRACTION of its footprint that is
corrupted, which is an area average (``adaptive_avg_pool2d``).  Nearest
sampling picks one pixel per cell and is biased whenever the mask boundary is
not stride-aligned.

**Prefix tokens are declared, not inferred.**  ``feat[:, 1:, :]`` assumes
exactly one CLS token; DINOv2 checkpoints with registers have 1 CLS + 4
registers, and that slice would fold a register into the patch grid.
``N_PREFIX_TOKENS`` makes the count explicit.

``compute_metrics`` also returns the total mask weight, so the caller can
average batches as ``sum(v_i * w_i) / sum(w_i)`` instead of taking a plain mean
that overweights the last partial batch.
"""

import math
import torch
import torch.nn.functional as F

# Number of non-spatial tokens at the START of the token sequence.
# Add an entry when you add a backbone; do not infer it.
N_PREFIX_TOKENS = {
    "resnet34": 0,
    "vit_b_16": 1,      # CLS
    "dinov2": 1,        # CLS (no registers in dinov2_vitb14)
    "dinov2_reg": 5,    # CLS + 4 registers
}

TOKEN_ARCHS = {"vit_b_16", "dinov2", "dinov2_reg"}


def to_grid(arch: str, feat: torch.Tensor) -> torch.Tensor:
    """
    Canonical feature form ``(B, D, Hf, Wf)`` for any backbone.

    Conv nets are already in that shape.  Token sequences ``(B, T, D)`` have
    their prefix tokens removed and the remaining patch tokens reshaped to a
    square grid.

    The check is on the tensor RANK, not only on ``arch``: the segmentation
    wrappers in ``shared/SegmentationModels.py`` already return a
    ``(B, D, Hf, Wf)`` grid for every backbone, including ViT and DINOv2,
    because they reshape the patch tokens themselves before this function ever
    sees them.  Keying only on ``arch`` would send those already-gridded
    features down the token path and try to unpack a 4-D shape into three
    names, which is exactly the "too many values to unpack" crash this check
    prevents.

    Args:
        arch : architecture key, see ``N_PREFIX_TOKENS``.
        feat : hooked feature, ``(B, C, Hf, Wf)`` or ``(B, T, D)``.

    Returns:
        ``(B, D, Hf, Wf)`` tensor.

    Raises:
        ValueError : if the patch tokens do not form a square grid.
    """
    if arch not in TOKEN_ARCHS or feat.dim() == 4:
        return feat                                   # (B, C, Hf, Wf)

    n_prefix = N_PREFIX_TOKENS[arch]
    tokens = feat[:, n_prefix:, :]                    # (B, N, D)
    B, N, D = tokens.shape
    side = int(round(math.sqrt(N)))
    if side * side != N:
        raise ValueError(
            f"{arch}: {N} patch tokens is not a square grid. "
            f"Check N_PREFIX_TOKENS['{arch}'] (currently {n_prefix})."
        )
    return tokens.transpose(1, 2).reshape(B, D, side, side).contiguous()


def project_mask(masks: torch.Tensor, hw: tuple[int, int],
                 soft: bool) -> torch.Tensor:
    """
    Project an image-space mask ``(B, 1, H, W)`` onto the feature grid.

    Args:
        masks : image-space mask.
        hw    : target ``(Hf, Wf)``.
        soft  : ``False`` gives a hard 0/1 weight, where a cell counts if more
                than half of it is corrupted; ``True`` keeps the continuous
                coverage in [0, 1].

    Returns:
        ``(B, 1, Hf, Wf)`` weight map.
    """
    w = F.adaptive_avg_pool2d(masks.float(), hw)      # (B, 1, Hf, Wf)
    return w if soft else (w > 0.5).float()


def compute_metrics(arch: str,
                    feat_clean: torch.Tensor,
                    feat_corr: torch.Tensor,
                    masks: torch.Tensor,
                    metric_names: list[str],
                    normalize_features: bool = False,
                    soft_masking: bool = False,
                    eps: float = 1e-8):
    """
    Masked MSE / cosine similarity between clean and corrupted features.

    Args:
        arch              : architecture key, see ``N_PREFIX_TOKENS``.
        feat_clean        : hooked feature, ``(B,C,H,W)`` or ``(B,T,D)``.
        feat_corr         : same shape.
        masks             : ``(B, 1, H, W)`` float mask in image space.
        metric_names      : subset of ``["mse", "cos"]``.
        normalize_features: L2-normalise the channel dimension first.
        soft_masking      : continuous weights instead of a 0.5 threshold.
        eps               : guard against division by zero.

    Returns:
        ``(results, weight)`` where ``results`` maps each metric name to a
        scalar tensor (NaN if the mask is empty), and ``weight`` is the total
        mask weight of this batch.  Average batches with
        ``sum(v_i * w_i) / sum(w_i)``.

    Raises:
        ValueError : if ``masks`` is not 4-D.
    """
    if masks.dim() != 4:
        raise ValueError(f"masks must be 4-D (B,1,H,W), got {masks.dim()}-D")

    # .float() is a no-op in full precision and an upcast under AMP: the
    # forward pass may run in bf16/fp16, but MSE and cosine are accumulated in
    # float32 so the metrics do not depend on the precision of the backbone.
    fc = to_grid(arch, feat_clean).float()            # (B, D, Hf, Wf)
    fr = to_grid(arch, feat_corr).float()

    if normalize_features:
        fc = F.normalize(fc, p=2, dim=1)
        fr = F.normalize(fr, p=2, dim=1)

    Hf, Wf = fc.shape[-2:]
    w = project_mask(masks, (Hf, Wf), soft_masking).squeeze(1)   # (B, Hf, Wf)

    w_total = w.sum()
    w_sum = float(w_total)
    if w_sum <= eps:
        # No location selected, so there is no measurement: NaN, never 0.0.
        nan = torch.tensor(float("nan"), device=fc.device)
        return {m: nan for m in metric_names}, 0.0

    results: dict[str, torch.Tensor] = {}

    if "mse" in metric_names:
        # Mean over channels, then mask-weighted mean over locations.
        diff2 = (fc - fr).pow(2).mean(dim=1)                     # (B, Hf, Wf)
        results["mse"] = (diff2 * w).sum() / w_total

    if "cos" in metric_names:
        num = (fc * fr).sum(dim=1)
        den = (fc.pow(2).sum(dim=1) * fr.pow(2).sum(dim=1)).sqrt()
        cos_map = num / (den + eps)                              # (B, Hf, Wf)
        results["cos"] = (cos_map * w).sum() / w_total

    return results, w_sum
