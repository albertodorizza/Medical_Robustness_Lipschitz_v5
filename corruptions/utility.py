"""
corruptions/utility.py
==================================
Shared low-level helper for applying ImageNet-C perturbation functions to a
masked region of an image.  Used by both GlobalCorruptions and LocalCorruptions.

How a perturbation is confined to a mask
----------------------------------------
A perturbation is a function of a whole array, not of individual pixels, so
"apply it only inside a mask" has to be made precise.  This module perturbs the
WHOLE IMAGE and then composites the result through the mask.

It used to crop the tight bounding box of the mask instead, perturb that crop,
and composite it back.  Cheaper in principle, but wrong for any perturbation
whose output depends on a statistic of its input.  ImageNet-C's Contrast is
``(x - mean(ROI)) * c + mean(ROI)`` and Brightness works in HSV over the whole
array, so under the bounding-box convention the same nominal severity was a
different perturbation for each mask shape:

* ``circular`` with ``invert=True`` corrupts the ring outside a central disk,
  whose bounding box is the ENTIRE image, so the reference mean was global;
* ``squared`` calls this function once per patch, so every 32x32 patch was
  normalised against its own local mean;
* ``superpixel`` sat in between and moved with the image content.

Measured on Kvasir-SEG with ``analysis/appendix/perturbation_magnitude.py``, Contrast
at severity 5 and ratio 0.3 changed the masked pixels by 73 grey levels under
``circular`` and 30 under ``squared`` -- the same declared severity, less than
half the actual perturbation.  A comparison across shapes made that way mixed
spatial propagation, which is the object of study, with a change in how hard
the corruption hit.

Perturbing the whole image removes the dependence: the function always sees the
same input, so severity means the same thing whatever the mask looks like, and
only the compositing step is spatial.  Gaussian Noise and Defocus Blur are
unchanged in distribution by this (noise is pixel-independent; blur reads a
neighbourhood either way), and comparisons across architectures, settings or
lambda were never affected, since those are paired and every configuration sees
the identical corrupted image.
"""

import numpy as np


def _apply_perturbations(image, corr_mask, perturbation_fn, severity):
    """
    Apply `perturbation_fn` only to the pixels of `image` where `corr_mask == 1`.

    The perturbation is computed on the whole image and then composited through
    the mask, so its output does not depend on the shape of the mask.  See the
    module docstring for why that matters.

    Args:
        image          : (H, W, C) uint8 numpy array - original RGB image.
        corr_mask      : (H, W) binary numpy array - 1 where perturbation applies.
        perturbation_fn: callable f(image, severity) -> ndarray (ImageNet-C).
        severity       : int in [0, 5].  0 means "no perturbation".

    Returns:
        output : (H, W, C) uint8 array; unmasked pixels are bit-identical to the
                 input.

    Raises:
        ValueError : if a square-only ImageNet-C function is given a
                     non-square image, or if the perturbation changes the shape.
    """
    # A few ImageNet-C functions assume a square input.
    square_corruptions = ("zoom_blur", "snow", "fog")
    if (image.shape[0] != image.shape[1]) and \
            perturbation_fn.__name__ in square_corruptions:
        raise ValueError(
            f"'{perturbation_fn.__name__}' requires a square image."
        )

    mask = corr_mask.astype(bool)

    if severity <= 0 or not np.any(mask):
        return image.copy()

    perturbed = perturbation_fn(image.copy(), severity)

    # Several ImageNet-C functions return a PIL image or a float array; force a
    # numpy array, then clip before the cast, since a few of them overshoot
    # [0, 255] and a bare astype would wrap around.
    if not isinstance(perturbed, np.ndarray):
        perturbed = np.array(perturbed)
    if perturbed.shape != image.shape:
        raise ValueError(
            f"Perturbed image shape {perturbed.shape} does not match "
            f"original image shape {image.shape}"
        )
    perturbed = np.clip(perturbed, 0, 255).astype(image.dtype)

    return np.where(mask[:, :, None], perturbed, image)
