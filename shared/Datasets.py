"""
shared/Datasets.py
==================
Dataset classes for Kvasir-v2 used across finetuning and drift analysis.

Two variants are provided:

* :class:`KvasirDataset`    — reads JPEG/PNG images on-the-fly via PIL.
  Suitable for initial experimentation or when disk I/O is not a bottleneck.

* :class:`KvasirDatasetNpy` — reads pre-processed images from a memory-mapped
  ``.npy`` file.  Eliminates per-epoch JPEG decoding and resize overhead,
  giving significantly faster data loading at the cost of a one-time
  preprocessing step.

See ``scripts/preprocess_kvasir.py`` (not included here) for how to generate
the ``.npy`` files consumed by :class:`KvasirDatasetNpy`.
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from shared.constants import IMAGENET_MEAN, IMAGENET_STD
import zlib

# =============================================================================
# KvasirDataset — on-the-fly PIL-based loader
# =============================================================================
class KvasirDataset(Dataset):
    """
    Minimal ImageFolder-style dataset for Kvasir-v2.

    Expects the root directory to contain one sub-folder per class,
    each holding ``.png`` / ``.jpg`` / ``.jpeg`` image files.

    Args:
        data_dir  : Path to the dataset root (e.g. ``kvasir-dataset-v2/``).
        transform : ``torchvision.transforms`` pipeline applied to every image.
                    If ``None``, raw PIL images are returned.
    """
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.samples = []  # Stores (image_path, label)

        # Get class folders (directories only, sorted for a stable label map)
        classes = sorted(d for d in os.listdir(data_dir)
                         if os.path.isdir(os.path.join(data_dir, d)))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.classes = classes

        print(f"Loading dataset {data_dir} into memory...")
        
        # Iterate over each class folder.  The file list is sorted because the
        # split files in shared/splits.py store integer indices: an unordered
        # os.scandir would make index i point at a different image on another
        # machine or after any change to the directory.
        for cls in classes:
            cls_dir = os.path.join(data_dir, cls)
            for name in sorted(os.listdir(cls_dir)):
                path = os.path.join(cls_dir, name)
                if os.path.isfile(path) and name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((path, self.class_to_idx[cls]))

        # Same attribute name as KvasirDatasetNpy, so shared.splits can
        # stratify either loader without special-casing.
        self.labels = np.array([lbl for _, lbl in self.samples], dtype=np.int64)

        print(f"Dataset loaded: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Load image and convert to RGB
        img = Image.open(path).convert("RGB")

        # Apply preprocessing transform if provided
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


# =============================================================================
# KvasirDatasetNpy — memory-mapped NumPy loader
# =============================================================================

class KvasirDatasetNpy(Dataset):
    """
    High-performance dataset for Kvasir images stored as NumPy arrays.

    Assumptions:

    * Images are stored in ``<data_dir>/kvasir_images.npy`` as a single
      array of shape ``(N, H, W, C)`` with ``dtype=uint8``.
    * Labels are stored in ``<data_dir>/kvasir_labels.npy`` as a 1-D integer
      array of length ``N``.
    * Both files are aligned by index (``images[i]`` corresponds to
      ``labels[i]``).

    Key performance properties:

    * Uses memory-mapping (``mmap_mode="r"``) so the OS loads pages on demand
      rather than reading the full array into RAM at startup.
    * HWC → CHW transposition and normalisation are done in ``__getitem__``
      with zero additional memory allocation (``torch.from_numpy`` is
      zero-copy for contiguous arrays).

    ImageNet normalization constants are imported from
    :mod:`shared.constants` to guarantee consistency with the transforms
    used in :class:`KvasirDataset`.

    Args:
        data_dir : Directory containing ``kvasir_images.npy`` and
                   ``kvasir_labels.npy``.
    """

    # Broadcastable to (C, H, W) tensors — class-level to avoid re-creating per call
    _MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    _STD  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir (str): Directory containing:
                - kvasir_images.npy
                - kvasir_labels.npy
        """

        # Memory-map the image array (no full load into RAM)
        self.images = np.load(os.path.join(data_dir, "kvasir_images.npy"), mmap_mode="r")

        # Labels can also be memory-mapped (small overhead, but consistent)
        self.labels = np.load(os.path.join(data_dir, "kvasir_labels.npy"))
        
    def __len__(self):
        """Return total number of samples."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve a single (image, label) pair.

        Processing steps:

        1. Read one row from the memory-mapped array (triggers a page fault
           only for the needed slice).
        2. Zero-copy NumPy → Torch conversion (``torch.from_numpy``).
        3. HWC → CHW transposition.
        4. Cast to float32 and scale to [0, 1].
        5. Apply ImageNet normalization in-place.
        """


        # np.array() copies the mmap slice into a writable buffer so that
        # torch.from_numpy can alias it without holding a reference to the mmap
        img = torch.from_numpy(np.array(self.images[idx]))

        # Rearrange to (C, H, W) and apply ImageNet normalisation
        img = img.permute(2, 0, 1).float().div_(255.0)
        img.sub_(self._MEAN).div_(self._STD)

        return img, torch.tensor(int(self.labels[idx]), dtype=torch.long)


