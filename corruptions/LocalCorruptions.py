"""
corruptions/LocalCorruptions.py
================================
Three local (spatially selective) corruption modes for image robustness testing.

All three functions share the same interface:
    corrupted_image, corruption_mask = <function>(image, perturbation_type,
                                                  perturbation_ratio, severity_level, ...)

`corruption_mask` is a boolean (H, W) array where True indicates pixels
that received the perturbation.  This mask is consumed by
`corruption_experiments_cls.py` to compute masked feature-map metrics.

Normalization to [0, 1] is NOT done here; it is handled by
transforms.ToTensor() in the DataLoader.

Modes
-----
- `squared`    : fixed-size square patches randomly placed, each possibly
                 receiving a different perturbation type.
- `circular`   : the outer ring of the image (outside a central circle) is
                 corrupted; the circle radius scales with `perturbation_ratio`.
- `superpixel` : a random subset of SLIC superpixels is corrupted until the
                 total corrupted area reaches `perturbation_ratio`.
"""

from skimage.segmentation import slic
import numpy as np
from corruptions import imagenet_c
from .utility import _apply_perturbations


# --------------------------------------------------------
# Squared
# --------------------------------------------------------
def squared(image, perturbation_types, perturbation_ratio, severity_level, squared_size):
    """
    Apply perturbations to randomly selected square patches.

    When multiple `perturbation_types` are given, the total `perturbation_ratio`
    is split equally among them and each type is applied to a non-overlapping
    set of patches.

    Args:
        image             : (H, W, C) uint8 numpy array.
        perturbation_types: list of perturbation type strings (ImageNet-C keys).
        perturbation_ratio: float in (0, 1] — fraction of image area to corrupt.
        severity_level    : int in [1, 5].
        squared_size      : int — side length (in pixels) of each square patch.

    Returns:
        corrupted_image   : (H, W, C) numpy array.
        merged_mask       : (H, W) boolean array — union of all patch masks.
    """

    # Ensure that the input image is valid
    if image is None:
        raise Exception(f"Image is None!")
    

    # Generate the mapping between perturbation types and their patch masks
    corr_masks_dict = _get_patches_to_perturb(perturbation_types, perturbation_ratio, squared_size, image)

    # Merge all masks into a single corruption masks 
    merged_corruption_mask = _merge_corr_masks(image, corr_masks_dict)

    # Create a copy of the input image to avoid modifying the original one
    corrupted_image = image.copy()

    # Iterate over each perturbation type
    for ptype in corr_masks_dict.keys():

         # Retrieve the perturbation function from ImageNet-C
        perturbation_fn = imagenet_c.return_list_transformations()[ptype]

        # All the patches of one type are perturbed in a SINGLE call, on the
        # union of their masks.  The loop used to call _apply_perturbations once
        # per patch, feeding it the partially corrupted image each time; now that
        # the perturbation is computed on the whole image, that would make patch
        # k's result depend on patches 1..k-1 through the image statistics, and
        # would run a full-image perturbation once per patch for no reason.
        # Different types still get separate calls, since they are different
        # functions applied to disjoint patch sets.
        type_mask = np.zeros(image.shape[:2], dtype=bool)
        for mask in corr_masks_dict[ptype]:
            type_mask |= mask.astype(bool)
        corrupted_image = _apply_perturbations(
            corrupted_image, type_mask, perturbation_fn, severity_level)

    # Normalize to [0, 1] moved to transforms.ToTensor() in DatasetLoader
    #output = corrupted_image / 255.0

    return corrupted_image, merged_corruption_mask


def _get_patches_to_perturb(perturbation_types, perturbation_ratio, squared_size, image):  
    """
    Build a dict mapping each perturbation type to its list of patch masks.

    Patches are randomly selected without replacement, and different types
    receive non-overlapping sets of patches (achieved by removing selected
    indices from the pool after each assignment).

    When multiple types are provided, `perturbation_ratio` is divided equally
    among them so that the total corrupted area stays at `perturbation_ratio`.

    Example output:
        {
          "Gaussian Noise": [mask_patch_2, mask_patch_7, ...],
          "Brightness":     [mask_patch_1, mask_patch_4, ...],
        }

    Args:
        perturbation_types: list of perturbation type strings.
        perturbation_ratio: target fraction of image area to corrupt (total).
        squared_size      : side length of each square patch in pixels.
        image             : (H, W, C) numpy array.

    Returns:
        corruption_masks: dict {ptype: [binary_mask, ...]}.

    Raises:
        ValueError if duplicate types are passed, if patches don't fit, or if
        more patches are needed than available.
    """

    # Check if there are multiple perturation types with same name defined by the user
    if len(perturbation_types) != len(set(perturbation_types)):
        raise ValueError("Please insert only one type of corruption with more perturbation ratio instead of many equal corruption types.")

    mask_shape_no_channels = image.shape[:2]

    # Divide the overall perturbation ratio with all the perturbation types
    if len(perturbation_types) > 1: 
        perturbation_ratio = perturbation_ratio / len(perturbation_types)

    # Calculate the number of patches needed for covering the ratio area required
    H, W, _ = image.shape
    ph, pw = [squared_size, squared_size]

    n_h = int(np.floor(H / ph))
    n_w = int(np.floor(W / pw))
    total_patches = n_h * n_w
    
    # If no full patches fit
    if n_h == 0 and n_w == 0:  # no total_patches==0 beacuse we can have a patch along only x or only y
        raise ValueError("No full patches fit inside the image with given patch size.")

    # Overestimate perturbation ratio (ceil ensures at least desired coverage)
    needed_patches = int(np.ceil(total_patches * perturbation_ratio))

    if needed_patches > total_patches: 
        raise ValueError("Needed patches are more than total patches, please insert a different perturbation ratio.")

    # Create dict: each key -> zero-initialized binary mask
    corruption_masks = {ptype: [np.zeros(mask_shape_no_channels, dtype=bool)] 
                        for ptype in perturbation_types}
    
    available_idxs = np.arange(total_patches) # 0 to total_patches-1 

    for ptype in perturbation_types:            # otherwise check the overlapping between mask: np.any(np.logical_and(m1, m2)) and regenerate until no overlapping
        # Randomly select patches
        selected_idx = np.random.choice(available_idxs, needed_patches, replace=False) 

        corruption_masks[ptype].pop(0) # Remove first zero-initialized binary mask

        # Generate all patches and save those (by concatenation) on relative corruption_masks key
        # Example: 
        #   'Snow' --> (binary_corr_mask of patch 2), (binary_corr_mask of patch 20), (binary_corr_mask of patch 5), ...
        #   'Gaussian Noise' --> (binary_corr_mask of patch 1), (binary_corr_mask of patch 15), (binary_corr_mask of patch 3), ...
        #
        for idx in selected_idx:
            mask = np.zeros((H, W), dtype=bool)
            # Calculate patch indexes
            i = idx // n_w 
            j = idx % n_w

            # Compute coordinates of the patch
            y0, y1 = i * ph, min((i + 1) * ph, H)  # clamp to image height
            x0, x1 = j * pw, min((j + 1) * pw, W)  # clamp to image width
            
            # Create patch mask and add it to the map
            mask[y0:y1, x0:x1] = 1 
            corruption_masks[ptype].append(mask)

        # Update the available indexes for ensuring no overlapping between different perturbation types masks 
        available_idxs = [x for x in available_idxs if x not in selected_idx]

    return corruption_masks


def _merge_corr_masks(image, corr_masks_dict):
    """
    Merge all per-patch masks into a single boolean mask via logical OR.

    Args:
        image         : (H, W, C) numpy array — used only for shape.
        corr_masks_dict: dict {ptype: [mask, ...]} from _get_patches_to_perturb.

    Returns:
        combined_mask : (H, W) boolean array — True where any perturbation is applied.
    """
    if corr_masks_dict is None:
        raise ValueError("The masks dictionary is empty.")

    # Get sizes of the image
    H, W, _ = image.shape

    # Initialize combined mask
    combined_mask = np.zeros((H, W), dtype=bool)
    
    # Merge all masks with logical OR
    for ptype in corr_masks_dict.keys():
        for mask in corr_masks_dict[ptype]:
            combined_mask = np.logical_or(combined_mask, mask)
        
    return combined_mask




# --------------------------------------------------------
# Circular
# --------------------------------------------------------
def circular(image, perturbation_type, perturbation_ratio, severity_level, position="center", invert=True):
    """
    Corrupt a centred circular region, or everything outside it.

    The radius is derived from the requested AREA, not from the linear size:

        invert=True  (corrupt outside):  r = sqrt((1 - ratio) * H*W / pi)
        invert=False (corrupt inside):   r = sqrt(ratio * H*W / pi)

    so `perturbation_ratio` is the fraction of pixels actually corrupted.
    A centred disk inscribed in a square covers pi/4 = 78.5% of it, which bounds
    the reachable range: [0.215, 1.0] with invert=True, [0, 0.785] with
    invert=False.  An unreachable value raises instead of silently producing a
    different amount of corruption.

    Args:
        image             : (H, W, C) uint8 numpy array.
        perturbation_type : single perturbation type string (ImageNet-C key).
        perturbation_ratio: float in (0, 1] — fraction of pixels to corrupt.
        severity_level    : int in [1, 5].
        position          : circle center; currently only "center" is supported.
        invert            : True  -> corrupt outside a central disk. (radius in [0.215, 1.0])
                            False -> corrupt inside a central disk.
                                     Models a specular reflection, a bubble or
                                     an instrument shadow, i.e. a blob INSIDE
                                     the field of view.  (radius in [0, 0.785])

    Returns:
        corrupted_image : (H, W, C) numpy array.
        mask            : (H, W) boolean array — True where the corruption
                          was applied.
    """
    if image is None:
        raise Exception(f"Image is None!")
    if not 0.0 < perturbation_ratio <= 1.0:
        raise ValueError(f"perturbation_ratio must be in (0, 1], got {perturbation_ratio}")
    if position != "center":
        raise NotImplementedError("Only 'center' position is supported.")



    H, W = image.shape[:2]
    area = float(H * W)
    r_max = min(H, W) / 2.0                      # largest inscribed disk
    max_disk_frac = np.pi * r_max ** 2 / area    # = pi/4 for a square image


    if invert:
        # corrupted = area - pi r^2  ->  r = sqrt((1 - ratio) * area / pi)
        radius = np.sqrt((1.0 - perturbation_ratio) * area / np.pi)
        if radius > r_max:
            raise ValueError(
                f"ratio={perturbation_ratio} not reachable with invert=True: "
                f"the minimum is {1.0 - max_disk_frac:.3f} for a {H}x{W} image."
            )
    else:
        # corrupted = pi r^2  ->  r = sqrt(ratio * area / pi)
        radius = np.sqrt(perturbation_ratio * area / np.pi)
        if radius > r_max:
            raise ValueError(
                f"ratio={perturbation_ratio} not reachable with invert=False: "
                f"the maximum is {max_disk_frac:.3f} for a {H}x{W} image."
            )


    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]

    # Create circular mask
    disk = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

    # Create the inside/outside circle mask
    mask = ~disk if invert else disk

    # Get the first perturbation in the config file    
    perturbation_fn = imagenet_c.return_list_transformations()[perturbation_type]
   
    # Apply the perturbation
    corrupted_image = _apply_perturbations(image, mask, perturbation_fn, severity_level)
    
    # Normalize to [0, 1] moved to transforms.ToTensor() in DatasetLoader
    #output = corrupted_image / 255.0

    return corrupted_image, mask




# --------------------------------------------------------
# Superpixel
# --------------------------------------------------------

def superpixel(image, perturbation_type, perturbation_ratio, severity_level, n_segments=100, compactness=10):
    """
    Corrupt a random subset of SLIC superpixels.

    Superpixels are added to the corrupted set in random order until the
    total corrupted area reaches `perturbation_ratio` of the image area.
    This produces an organically shaped mask that respects image content
    boundaries (unlike uniform random patches).

    Args:
        image             : (H, W, C) uint8 numpy array.
        perturbation_type : single perturbation type string (ImageNet-C key).
        perturbation_ratio: float in (0, 1] — target fraction of image to corrupt.
        severity_level    : int in [1, 5].
        n_segments        : approximate number of superpixels (SLIC parameter).
        compactness       : balance between color and spatial proximity (SLIC).
                            Higher values → more square superpixels.

    Returns:
        corrupted_image : (H, W, C) numpy array.
        mask            : (H, W) boolean array — True for corrupted superpixels.
    """

    # Save useful params for computation
    perturbation_fn = imagenet_c.return_list_transformations()[perturbation_type]

    # Get superpixels labels
    image_float = image / 255.0
    superpixel_labels = slic(image_float, n_segments=n_segments, compactness=compactness)
 
    # Create corruption mask with random superpixels 
    mask = _get_superpixel_mask(image, superpixel_labels, perturbation_ratio)
    mask = mask.astype(bool)

    # apply perturbation on the corruption mask
    corrupted_image = _apply_perturbations(image, mask, perturbation_fn, severity_level)

    # Normalize to [0, 1] moved to transforms.ToTensor() in DatasetLoader
    #output = corrupted_image / 255.0

    return corrupted_image, mask


def _get_superpixel_mask(image, sp_labels, perturbation_ratio):
    """
    Greedily select superpixels (in random order) until the target area is reached.

    Args:
        image             : (H, W, C) numpy array — used for shape and area.
        sp_labels         : (H, W) integer array of superpixel labels from SLIC.
        perturbation_ratio: target fraction of total image pixels to corrupt.

    Returns:
        binary_mask : (H, W) uint8 array — 1 for corrupted superpixels.
    """

    # Get sizes of the image
    H, W, _ = image.shape

    # Get unique labels returned by slic and shuffle them randomly 
    unique_labels = np.unique(sp_labels)
    np.random.shuffle(unique_labels)

    # Compute area (pixel count) of each superpixel 
    label_sizes = np.bincount(sp_labels.ravel())

    # Select labels until the total covered area ≈ perturbation_ratio 
    target_area = perturbation_ratio * H * W
    selected_labels = []
    covered_area = 0

    for label in unique_labels:
        covered_area += label_sizes[label]
        selected_labels.append(label)
        if covered_area >= target_area:
            break

    # Create binary mask
    binary_mask = np.isin(sp_labels, selected_labels).astype(np.uint8)

    return binary_mask


