"""
shared/DatasetRegistry.py
=========================
One place that knows where every dataset lives on disk and what task it serves.
Both dataset classes return the same contract:

* classification -> ``(image_tensor, label)``   with ImageNet normalisation
* segmentation   -> ``(image_tensor, mask)``    mask is ``(H, W)`` int64

and both expose ``.labels`` so that :mod:`shared.splits` can stratify and fingerprint them..

// SISTEMA
Segmentation datasets stratify on lesion **size** (three terciles) rather than
on a class index.  Polyp area varies by two orders of magnitude and Dice is
strongly size-dependent, so a plain random split can put most of the small
lesions in one fold and make results look unstable across seeds for a reason
that has nothing to do with the model.  The strata are cached next to the
dataset because computing them requires reading every mask once.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from shared.constants import IMAGENET_MEAN, IMAGENET_STD

cv2.setNumThreads(0)

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


# =============================================================================
# Dataset specification
# =============================================================================
@dataclass
class DatasetSpec:
    """
    Everything the pipeline needs to know about a dataset.

    Attributes:
        name        : Registry key, used as ``--dataset``.
        task        : ``"classification"`` or ``"segmentation"``.
        root        : Absolute path on disk.
        num_classes : Number of output classes; 2 for binary segmentation.
        modality    : Free-form label used only to group results in tables.
        images_dir  : Segmentation only — subfolder holding the images.
        masks_dir   : Segmentation only — subfolder holding the masks.
        mask_prefix : Segmentation only — prefix that mask filenames add to the
                      image stem (ETIS uses ``p1.tif`` for ``1.tif``).
        sources     : Segmentation only — list of root folders to MERGE into a
                      single dataset, each containing its own
                      ``images_dir``/``masks_dir``.  When set, ``root`` is used
                      only as a writable place for the cached file and
                      no images are read from it.  See ``aircare_all``.
        notes       : Short description.
    """
    name: str
    task: str
    root: str
    num_classes: int = 2
    modality: str = "endoscopy"
    images_dir: str = "images"
    masks_dir: str = "masks"
    mask_prefix: str = ""
    sources: list[str] | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)


# =============================================================================
# Registry
# =============================================================================
# Every path below is relative to it.
#
# Read from the environment so the registry is not tied to one machine: set
# MEDICAL_ROOT before launching anything, or edit the default below.  The
# launchers export it, so a single value in one place reaches all six stages.
MEDICAL_ROOT = os.environ.get(
    "MEDICAL_ROOT", "/home/datasets/medical/"
)

# Common prefix for the four AIRCARE / Brescia head-and-neck sites, so it is
# not repeated in each DatasetSpec below.
_AIRCARE_BASE = f"{MEDICAL_ROOT}/aircare/external/Brescia/1_LAR_OC_OP_HP_datasets_Bresia"




DATASETS: dict[str, DatasetSpec] = {

    # ── Classification ───────────────────────────────────────────────────────
    "kvasir_v2": DatasetSpec(
        name="kvasir_v2", task="classification",
        root=f"{MEDICAL_ROOT}/kvasir-dataset-v2",
        num_classes=8,
        notes="Pogorelov et al. 2017 — 8 GI findings, 1000 images each.",
    ),
    "hyperkvasir": DatasetSpec(
        name="hyperkvasir", task="classification",
        root=f"{MEDICAL_ROOT}/hyper-kvasir/labeled-images",
        num_classes=0,          # discovered from the folder tree
        notes=("Borgli et al. 2020 — labelled images only. Classes are the "
               "leaf folders of the tract/category/class tree, so the label "
               "set is discovered at load time. Strongly imbalanced."),
    ),
    "kvasir_capsule": DatasetSpec(
        name="kvasir_capsule", task="classification",
        root=f"{MEDICAL_ROOT}/kvasir-capsule/labelled_images",
        num_classes=0,          # discovered from the folder tree
        notes=("Smedsrud et al. 2021 — video capsule endoscopy. Very "
               "imbalanced."),
    ),

    # SISTEMA
    # ── Segmentation: polyps ─────────────────────────────────────────────────
    # Any of these can be used either as a training set with its own 70/10/20
    # split (protocol A), or as a zero-shot external test set for a model
    # trained elsewhere (protocol B).  The launchers default to Kvasir-SEG for
    # training and the other three for external evaluation, but nothing here
    # prevents the opposite. Training on one and testing zero-shot on the others is the standard
    # protocol in this sub-field.
    "kvasir_seg": DatasetSpec(
        name="kvasir_seg", task="segmentation",
        root=f"{MEDICAL_ROOT}/Kvasir-SEG",
        images_dir="images", masks_dir="masks",
        notes="Jha et al. 2020 — 1000 polyp images.",
    ),
    "cvc_clinicdb": DatasetSpec(
        name="cvc_clinicdb", task="segmentation",
        root=f"{MEDICAL_ROOT}/CVC-ClinicDB/PNG",
        images_dir="Original", masks_dir="Ground Truth",
        notes="Bernal et al. 2015 — 612 frames.",
    ),
    "cvc_colondb": DatasetSpec(
        name="cvc_colondb", task="segmentation",
        root=f"{MEDICAL_ROOT}/CVC-ColonDB",
        images_dir="images", masks_dir="masks",
        notes="380 frames.",
    ),
    "etis_larib": DatasetSpec(
        name="etis_larib", task="segmentation",
        root=f"{MEDICAL_ROOT}/ETIS-LaribPolypDB",
        images_dir="images", masks_dir="masks",
        mask_prefix="p",        # masks are p1.tif for image 1.tif
        notes="Silva et al. 2014 — 196 frames.",
    ),

    # ── Segmentation: AIRCARE head-and-neck (Brescia) ────────────────────────
    # Four anatomical sites under the same collection, each with its own
    # images/labels/masks folders.  Different anatomical site and imaging
    # conditions from the polyp sets.
    #
    # "labels" (bounding boxes) is not used here; only "images" and "masks"
    # matter for segmentation.  Folder names are kept exactly as they are on
    # disk, including the space+parenthesis in "Larynx_(611_images)" and the
    # "Orophaynx" spelling.  
    "aircare_hypopharynx": DatasetSpec(
        name="aircare_hypopharynx", task="segmentation",
        root=f"{_AIRCARE_BASE}/Hypopharynx",
        images_dir="images", masks_dir="masks",
        modality="endoscopy_head_neck",
        notes="Brescia hypopharynx, lesion segmentation.",
    ),
    "aircare_larynx": DatasetSpec(
        name="aircare_larynx", task="segmentation",
        root=f"{_AIRCARE_BASE}/Larynx_(611_images)",
        images_dir="images", masks_dir="masks",
        modality="endoscopy_head_neck",
        notes="Brescia larynx, lesion segmentation, 611 images.",
    ),
    "aircare_oral_cavity": DatasetSpec(
        name="aircare_oral_cavity", task="segmentation",
        root=f"{_AIRCARE_BASE}/Oral_cavity",
        images_dir="images", masks_dir="masks",
        modality="endoscopy_head_neck",
        notes="Brescia oral cavity, lesion segmentation.",
    ),
    "aircare_oropharynx": DatasetSpec(
        name="aircare_oropharynx", task="segmentation",
        root=f"{_AIRCARE_BASE}/Orophaynx",
        images_dir="images", masks_dir="masks",
        modality="endoscopy_head_neck",
        notes="Brescia oropharynx, lesion segmentation.",
    ),

    # Merged view of the four sites above, as one site-agnostic binary
    # lesion/background dataset.  Each site alone is small while pooling
    # gives 39+611+246+135 = 1031 images and a usable 70/10/20 split.
    #
    # root is only a writable location for the cached strata file; images come from `sources`.
    "aircare_all": DatasetSpec(
        name="aircare_all", task="segmentation",
        root=f"{MEDICAL_ROOT}/aircare/external/Brescia",
        images_dir="images", masks_dir="masks",
        modality="endoscopy_head_neck",
        sources=[
            f"{_AIRCARE_BASE}/Hypopharynx",
            f"{_AIRCARE_BASE}/Larynx_(611_images)",
            f"{_AIRCARE_BASE}/Oral_cavity",
            f"{_AIRCARE_BASE}/Orophaynx",
        ],
        notes=("Pooled Hypopharynx + Larynx + Oral_cavity + Orophaynx "
               "(1031 images). Binary lesion vs. background."),
    ),
}


def get_spec(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. "
                       f"Available: {sorted(DATASETS)}")
    return DATASETS[name]


def list_datasets(task: str | None = None) -> list[str]:
    """Registry keys, optionally filtered by task."""
    return sorted(k for k, s in DATASETS.items()
                  if task is None or s.task == task)


# =============================================================================
# Shared image helpers
# =============================================================================
_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def read_rgb(path: str, size: tuple[int, int] | None = None) -> np.ndarray:
    """Read an image as uint8 HWC RGB, optionally resized to ``(H, W)``."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None and img.shape[:2] != tuple(size):
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(img)


def read_mask(path: str, size: tuple[int, int] | None = None) -> np.ndarray:
    """
    Read a binary mask as uint8 ``{0, 1}``.

    Nearest-neighbour resizing is mandatory here: bilinear would create
    fractional labels at object boundaries.  The threshold at 127 tolerates the
    JPEG artefacts present in several of these datasets, whose masks are not
    exactly 0/255.
    """
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if size is not None and m.shape[:2] != tuple(size):
        m = cv2.resize(m, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def normalize_image(img_u8: np.ndarray) -> torch.Tensor:
    """uint8 HWC RGB -> ImageNet-normalised CHW float tensor."""
    t = torch.from_numpy(np.ascontiguousarray(img_u8)).permute(2, 0, 1).contiguous()
    return t.float().div_(255.0).sub_(_MEAN).div_(_STD)


def _find_images(root: str) -> list[str]:
    """All image files under ``root``, recursively, sorted for determinism."""
    return sorted(
        os.path.join(dp, f)
        for dp, _, fns in os.walk(root) for f in fns
        if f.lower().endswith(IMG_EXT)
    )


# =============================================================================
# Classification dataset
# =============================================================================
class FolderClassificationDataset(Dataset):
    """
    Classification from a folder tree, where every **leaf** folder is a class.

    Handles both flat layouts (``kvasir-dataset-v2/<class>/*.jpg``) and nested
    ones (``hyper-kvasir/labeled-images/<tract>/<category>/<class>/*.jpg``): a
    folder is a class if it directly contains images.  The class name is its
    path relative to the root, so nested names stay unique and readable, e.g.
    ``lower-gi-tract/pathological-findings/polyps``.

    Args:
        spec       : Registry entry.
        image_size : ``(H, W)`` to resize to.
        max_per_class : Optional cap on images per class.  Hyper-Kvasir and
                        Kvasir-Capsule are extremely imbalanced (some classes
                        have tens of thousands of frames, others a few dozen);
                        capping keeps the class weights from being dominated by
                        one or two categories.  ``None`` uses everything.
        min_per_class : Drop classes with fewer images than this.  A class with
                        6 examples cannot be split 70/10/20 in any meaningful
                        way.
    """

    def __init__(self, spec: DatasetSpec, image_size=(224, 224),
                 max_per_class: int | None = None, min_per_class: int = 20,
                 seed: int = 0):
        self.spec = spec
        self.image_size = tuple(image_size)

        if not os.path.isdir(spec.root):
            raise FileNotFoundError(f"Dataset root not found: {spec.root}")

        # Collect leaf folders that directly contain images
        per_class: dict[str, list[str]] = {}
        for dirpath, _, filenames in os.walk(spec.root):
            files = sorted(f for f in filenames if f.lower().endswith(IMG_EXT))
            if not files:
                continue
            cls = os.path.relpath(dirpath, spec.root).replace(os.sep, "/")
            per_class[cls] = [os.path.join(dirpath, f) for f in files]

        # Drop tiny classes, then optionally cap the large ones
        rng = np.random.default_rng(seed)
        kept = {}
        for cls in sorted(per_class):
            paths = per_class[cls]
            if len(paths) < min_per_class:
                print(f"  [{spec.name}] dropping class '{cls}' "
                      f"({len(paths)} images < min_per_class={min_per_class})")
                continue
            if max_per_class is not None and len(paths) > max_per_class:
                idx = np.sort(rng.choice(len(paths), max_per_class, replace=False))
                paths = [paths[i] for i in idx]
            kept[cls] = paths

        if not kept:
            raise ValueError(f"No usable classes found under {spec.root}")

        self.classes = sorted(kept)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: list[tuple[str, int]] = [
            (p, self.class_to_idx[c]) for c in self.classes for p in kept[c]
        ]
        self.labels = np.array([l for _, l in self.samples], dtype=np.int64)
        self.num_classes = len(self.classes)

        print(f"  [{spec.name}] {len(self.samples)} images, "
              f"{self.num_classes} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = read_rgb(path, self.image_size)
        return normalize_image(img), torch.tensor(int(label), dtype=torch.long)

    def raw(self, idx) -> tuple[np.ndarray, int]:
        """Uint8 image and label, for the corruption pipeline."""
        path, label = self.samples[idx]
        return read_rgb(path, self.image_size), int(label)


# =============================================================================
# Segmentation dataset
# =============================================================================
class SegmentationDataset(Dataset):
    """
    Image/mask segmentation dataset with tolerant filename pairing.

    These four polyp datasets do not agree on naming: Kvasir-SEG and CVC use
    identical stems, ETIS prefixes its masks with ``p``, and extensions differ
    between the two folders.  Pairing is therefore done on the **stem**, with
    an optional prefix, and unpaired files are reported rather than silently
    dropped — a silent drop would change the dataset size and invalidate the
    split fingerprint without any visible sign.

    Args:
        spec       : Registry entry.
        image_size : ``(H, W)`` to resize to.
    """

    def __init__(self, spec: DatasetSpec, image_size=(224, 224)):
        self.spec = spec
        self.image_size = tuple(image_size)

        # A dataset is either a single root, or a merge of several (spec.sources).
        # Pairing is done PER ROOT rather than on one global stem index: two
        # collections can reuse the same filename for different images, and a
        # global index would silently pair an image with another root's mask.
        roots = spec.sources if spec.sources else [spec.root]

        self.pairs: list[tuple[str, str]] = []
        self.sites: list[str] = []          # provenance, one entry per pair
        unpaired: list[str] = []

        for root in roots:
            img_root = os.path.join(root, spec.images_dir)
            msk_root = os.path.join(root, spec.masks_dir)
            for d in (img_root, msk_root):
                if not os.path.isdir(d):
                    raise FileNotFoundError(f"{spec.name}: missing folder {d}")

            # Index masks by stem (recursive: some AIRCARE folders are nested)
            masks_by_stem = {
                os.path.splitext(os.path.basename(p))[0]: p
                for p in _find_images(msk_root)
            }

            site = os.path.basename(root.rstrip("/"))
            for img_path in _find_images(img_root):
                stem = os.path.splitext(os.path.basename(img_path))[0]
                mask = (masks_by_stem.get(spec.mask_prefix + stem)
                        or masks_by_stem.get(stem))
                if mask is None:
                    unpaired.append(os.path.basename(img_path))
                else:
                    self.pairs.append((img_path, mask))
                    self.sites.append(site)

        if unpaired:
            print(f"  [{spec.name}] WARNING: {len(unpaired)} images without a "
                  f"mask were skipped, e.g. {unpaired[:3]}. "
                  "Check images_dir/masks_dir/mask_prefix in the registry.")
        if not self.pairs:
            raise ValueError(
                f"{spec.name}: no image/mask pairs matched under {roots}."
            )

        self.num_classes = spec.num_classes
        self._strata = None
        print(f"  [{spec.name}] {len(self.pairs)} image/mask pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]
        img = read_rgb(img_path, self.image_size)
        msk = read_mask(msk_path, self.image_size)
        return normalize_image(img), torch.from_numpy(msk.astype(np.int64))

    def raw(self, idx) -> tuple[np.ndarray, np.ndarray]:
        """Uint8 image and 0/1 mask, for the corruption pipeline."""
        img_path, msk_path = self.pairs[idx]
        return read_rgb(img_path, self.image_size), read_mask(msk_path, self.image_size)

    # -- stratification --------------------------------------------------------
    @property
    def labels(self) -> np.ndarray:
        """
        Lesion-size stratum per sample, so that ``shared.splits`` can stratify.

        Three terciles of foreground area.  Cached to a small ``.npy`` next to
        the dataset, since building it reads every mask once.
        """
        if self._strata is not None:
            return self._strata

        # The name is part of the cache key: merged and single-site datasets
        # can share a root folder, and the sample count alone would not always
        # tell them apart.
        cache = os.path.join(self.spec.root,
                             f".strata_{self.spec.name}_{len(self.pairs)}.npy")
        if os.path.exists(cache):
            self._strata = np.load(cache)
            return self._strata

        print(f"  [{self.spec.name}] computing lesion-size strata ...")
        areas = np.array([read_mask(m, self.image_size).mean()
                          for _, m in self.pairs])
        q1, q2 = np.quantile(areas, [1 / 3, 2 / 3])
        self._strata = np.digitize(areas, [q1, q2]).astype(np.int64)
        try:
            np.save(cache, self._strata)
        except OSError:
            pass                                  # read-only dataset folder
        return self._strata


# =============================================================================
# Factory
# =============================================================================
def build_dataset(name: str, image_size=(224, 224), **kwargs):
    """
    Instantiate a registered dataset.

    Args:
        name       : Registry key (see :func:`list_datasets`).
        image_size : ``(H, W)``.
        **kwargs   : Forwarded to the dataset class (e.g. ``max_per_class``).

    Returns:
        A :class:`SegmentationDataset` or :class:`FolderClassificationDataset`.
    """
    spec = get_spec(name)
    if spec.task == "segmentation":
        return SegmentationDataset(spec, image_size)
    return FolderClassificationDataset(spec, image_size, **kwargs)
