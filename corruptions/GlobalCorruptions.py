"""
corruptions/GlobalCorruptions.py
=================================
Applies an ImageNet-C perturbation to the **entire image** (no spatial mask).

The resulting corruption mask is a boolean array of all True values (every
pixel is considered corrupted), which is consistent with the mask convention
used by `LocalCorruptions.py` and expected by `corruption_experiments_cls.py`
when computing masked feature metrics.
"""

from corruptions import imagenet_c
import numpy as np
from .utility import _apply_perturbations

def global_corr(image, perturbation_types, severity_level):
    """
    Apply a single ImageNet-C perturbation to the whole image.

    Args:
        image             : (H, W, C) uint8 numpy array — input RGB image.
        perturbation_types: list of perturbation type strings; only the first
                            element is used (e.g. ["Gaussian Noise"]).
        severity_level    : int in [1, 5] — corruption severity.

    Returns:
        corrupted_image : (H, W, C) numpy array with the perturbation applied.
        corrupted_mask  : (H, W) boolean array, all True (whole image corrupted).

    Notes:
        - Only the first element of `perturbation_types` is applied.
          Multiple types are not combined in the global mode; use the squared
          corruption in LocalCorruptions.py for multi-type mixing.
        - Normalization to [0, 1] is NOT done here; it is handled downstream
          by transforms.ToTensor() in the DataLoader.
    """

    # Check that the input image is valid
    if image is None:
        raise Exception(f"Image is None!")

    # Create a corruption mask covering the entire image (all pixels set to True)
    # Shape is (H, W) because the mask is applied spatially, independent of channels
    corrupted_mask = np.ones(image.shape[:2], dtype=bool)

    # Retrieve the perturbation function corresponding to the first perturbation
    # specified in the perturbation_types list from the ImageNet-C transformations
    perturbation_fn = imagenet_c.return_list_transformations()[perturbation_types[0]]

    # Apply the perturbation
    corrupted_image = _apply_perturbations(image, corrupted_mask, perturbation_fn, severity_level)

    # Normalization to [0, 1] is not done here because it is handled later
    # inside transforms.ToTensor() in the DatasetLoader
    # output = corrupted_image / 255.0
    
    return corrupted_image, corrupted_mask
