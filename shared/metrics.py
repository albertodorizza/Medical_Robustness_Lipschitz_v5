"""
shared/metrics.py
=================
The single definition of foreground Dice and IoU used by every stage.

Why this file exists
--------------------
The three stages used to compute Dice three different ways, and the numbers
they produced were therefore not comparable even though the paper puts them in
adjacent tables:

============================  ==========================  ====================
stage                          empty image (no GT, no P)   formula
============================  ==========================  ====================
``finetuning_seg.py``          counted as **1.0**          ``(2T+1e-6)/(S+1e-6)``
``corruption_experiments_seg`` excluded (NaN)              ``2T/S``
``drift_analysis_seg.py``      excluded (NaN)              ``2T/S`` on the
                                                           feature grid
============================  ==========================  ====================

The smoothing constant is the harmful one.  On an image with no lesion where
the model also predicts nothing, ``(2*0 + 1e-6)/(0 + 1e-6) = 1.0``: a perfect
score awarded for a measurement that was never taken.  Averaged over a dataset
with many lesion-free frames it inflates the reported Dice by exactly the
fraction of such frames, and it does so more for a conservative model than for
a sensitive one, which is the opposite of what the metric is meant to rank.

The convention adopted here, and now used everywhere:

* **no smoothing.**  ``Dice = 2|P∩G| / (|P|+|G|)``.
* **an undefined case is NaN, not a score.**  ``|P|+|G| = 0`` means there was
  nothing to measure.  Those images are dropped from the average.
* **the number of images actually scored travels with the mean.**  Reporting
  ``0.89`` without saying it is over 197 of 200 images hides exactly the thing
  the NaN convention was introduced to expose, so every helper here returns the
  count alongside the value and the callers write it into their logs.

Note the asymmetry that remains, deliberately: a lesion-free image on which the
model *does* predict foreground scores 0, because ``|P|+|G| > 0`` and the
measurement is well defined.  Only the both-empty case is undefined.
"""

import numpy as np
import torch

__all__ = ["foreground_dice", "foreground_iou", "nanmean_count",
           "SMOOTH_FREE_CONVENTION"]

SMOOTH_FREE_CONVENTION = (
    "Dice = 2|P n G| / (|P|+|G|), no smoothing; images where |P|+|G| = 0 are "
    "undefined and excluded from the mean, and n reports how many were scored."
)


def foreground_dice(pred: torch.Tensor, target: torch.Tensor,
                    region: torch.Tensor | None = None) -> torch.Tensor:
    """
    Per-image foreground Dice, optionally restricted to a spatial region.

    Args:
        pred   : ``(B, H, W)`` predicted class indices.
        target : ``(B, H, W)`` ground-truth class indices.
        region : ``(B, H, W)`` boolean mask selecting which pixels to score.
                 ``None`` scores the whole image.  This is what gives the
                 inside/outside decomposition of the corruption stage.

    Returns:
        ``(B,)`` float tensor, NaN where the selected region contains neither
        predicted nor ground-truth foreground.
    """
    p = (pred == 1)
    g = (target == 1)
    if region is not None:
        p = p & region
        g = g & region
    p = p.float()
    g = g.float()

    inter = (p * g).sum(dim=(1, 2))
    denom = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2))
    return torch.where(denom > 0, 2.0 * inter / denom,
                       torch.full_like(denom, float("nan")))


def foreground_iou(pred: torch.Tensor, target: torch.Tensor,
                   region: torch.Tensor | None = None) -> torch.Tensor:
    """
    Per-image foreground IoU, same conventions as :func:`foreground_dice`.

    Args:
        pred   : ``(B, H, W)`` predicted class indices.
        target : ``(B, H, W)`` ground-truth class indices.
        region : ``(B, H, W)`` boolean mask, or ``None`` for the whole image.

    Returns:
        ``(B,)`` float tensor, NaN where the union is empty.
    """
    p = (pred == 1)
    g = (target == 1)
    if region is not None:
        p = p & region
        g = g & region
    p = p.float()
    g = g.float()

    inter = (p * g).sum(dim=(1, 2))
    union = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2)) - inter
    return torch.where(union > 0, inter / union,
                       torch.full_like(union, float("nan")))


def nanmean_count(values) -> tuple[float, int, int]:
    """
    Mean over defined entries, with the counts that make it interpretable.

    Args:
        values : 1-D tensor, numpy array, list of tensors, or list of floats.

    Returns:
        ``(mean, n_valid, n_total)``.  ``mean`` is NaN when nothing was
        defined.  Always report ``n_valid`` next to the mean: a Dice averaged
        over 140 of 200 images is a different claim from one averaged over 200.
    """
    if isinstance(values, (list, tuple)):
        if not values:
            return float("nan"), 0, 0
        if torch.is_tensor(values[0]):
            values = torch.cat([v.reshape(-1) for v in values])
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    x = np.asarray(values, dtype=float).reshape(-1)

    n_total = int(x.size)
    ok = ~np.isnan(x)
    n_valid = int(ok.sum())
    mean = float(x[ok].mean()) if n_valid else float("nan")
    return mean, n_valid, n_total
