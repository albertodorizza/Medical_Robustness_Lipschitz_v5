"""
corruption_experiments/corruption_common.py
===========================================
Code shared by the two entry points of the ``corruption_experiments`` stage.

``corruption_experiments_cls.py`` (classification) and
``corruption_experiments_seg.py`` (segmentation) apply the same corruption with
the same per-sample seed, and differ only in the model, the output metric, and
the target the dataset returns.  Keeping the common part here is what
guarantees that a given (image, config) pair receives the same mask and the
same noise realisation in both tasks.

Speed
-----
Three things in this module exist only to make the sweep faster, without
changing any number it produces:

* ``CorruptedDataset`` takes a LIST of configs and corrupts one image with all
  of them in a single ``__getitem__``.  The image is then decoded, resized and
  pushed through the clean forward pass ONCE per chunk instead of once per
  combination, which is where most of the old runtime went.
* SLIC is memoised (see ``_CachedSlic``).  The segmentation of an image depends
  only on the image and on the SLIC parameters, never on ratio, severity or
  perturbation type, so every ``superpixel`` combination of a chunk reuses the
  same segments.
* Images travel to the GPU as uint8 and are normalised there
  (``normalize_batch``), which is 4x less host memory and 4x less PCIe traffic
  than sending float32 tensors.
"""

import os
import zlib
import hashlib
from collections import OrderedDict

import torch
from box import Box
import numpy as np
from torch.utils.data import Dataset
from corruptions import LocalCorruptions as _local_corruptions
from corruptions.GlobalCorruptions import global_corr
from corruptions.LocalCorruptions import circular, squared, superpixel
from shared.constants import IMAGENET_MEAN, IMAGENET_STD


# =============================================================================
# SLIC memoisation
# =============================================================================
# How many segmentations one worker keeps.  Read from the environment so the
# value reaches DataLoader workers under both the fork and the spawn start
# method; a plain module constant would be lost with spawn.
SLIC_CACHE_SIZE = int(os.environ.get("SLIC_CACHE_SIZE", "256"))


class _CachedSlic:
    """
    LRU wrapper around ``skimage.segmentation.slic``.

    SLIC is the most expensive CPU step of the sweep, and it is recomputed
    identically for every ``superpixel`` combination of the same image: the
    segmentation depends on the pixels and on ``n_segments`` / ``compactness``,
    never on the ratio, the severity or the perturbation type.  Only the RANDOM
    CHOICE of which superpixels to corrupt depends on those, and that choice
    stays outside the cache, so the corruptions are exactly the ones the
    uncached code produced.

    The key is a hash of the image content rather than a dataset index, so the
    cache can never hand back the segmentation of a different image.  Hashing
    about a megabyte costs roughly a millisecond, against the tens of
    milliseconds SLIC itself takes.

    Wrapping is done here instead of editing ``corruptions/LocalCorruptions.py``
    so the corruption definitions themselves stay untouched.
    """

    def __init__(self, fn, maxsize: int):
        self._fn = fn
        self._maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()

    def __call__(self, image, *args, **kwargs):
        # Positional extras are not part of the key, so do not risk caching.
        if self._maxsize <= 0 or args:
            return self._fn(image, *args, **kwargs)

        arr = np.ascontiguousarray(image)
        digest = hashlib.blake2b(memoryview(arr).cast("B"),
                                 digest_size=16).digest()
        key = (digest, arr.shape, arr.dtype.str, tuple(sorted(kwargs.items())))

        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit

        segments = self._fn(image, **kwargs)
        # int32 halves the memory of skimage's int64 labels: a 224x224 map is
        # 200 kB, so a 256-entry cache costs about 50 MB per worker.
        segments = np.ascontiguousarray(segments, dtype=np.int32)
        self._cache[key] = segments
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return segments


# Installed at import time so DataLoader workers get it too.
_local_corruptions.slic = _CachedSlic(_local_corruptions.slic, SLIC_CACHE_SIZE)


# =============================================================================
# Default hook layers per architecture
# =============================================================================
RESNET34_HOOK_LAYERS = ["layer1", "layer2", "layer3", "layer4"]

VIT_B16_HOOK_LAYERS = (
    [f"encoder.layers.encoder_layer_{i}" for i in range(12)] + ["encoder.ln"]
)

DINOV2_HOOK_LAYERS = (
    [f"backbone.blocks.{i}" for i in range(12)] + ["backbone.norm"]
)


# The segmentation wrappers of ``shared/SegmentationModels.py`` hold the same
# backbones one or two attributes deeper, so the module paths differ from the
# classification ones and the lists above would silently hook nothing.
#
# The DEEPEST probe is deliberately absent from each list: for every backbone it
# is the same tensor the segmentation model already reports as its pre-head
# feature map (ResNet returns ``layer4`` directly; the ViT and DINOv2 wrappers
# drop the prefix tokens of ``encoder.ln`` / ``backbone.norm`` and reshape them,
# which is exactly what ``corruption_metrics.to_grid`` does before measuring).
# Hooking it too would compute the same numbers twice and put two rows at the
# same depth in the analyses.
SEG_RESNET34_HOOK_LAYERS = [f"backbone.layer{i}" for i in (1, 2, 3)]

SEG_VIT_B16_HOOK_LAYERS = [
    f"backbone.vit.encoder.layers.encoder_layer_{i}" for i in range(12)
]

SEG_DINOV2_HOOK_LAYERS = [f"backbone.backbone.blocks.{i}" for i in range(12)]


def get_default_hook_layers(arch: str, task: str = "classification") -> list[str]:
    """
    Layers to hook for ``arch``, as dot-separated ``named_modules()`` keys.

    Args:
        arch : ``"resnet34"``, ``"vit_b_16"`` or ``"dinov2"``.
        task : ``"classification"`` or ``"segmentation"``; the segmentation
               wrappers nest the backbone, so the paths differ.

    Returns:
        List of layer names; empty for an unknown architecture, which lets the
        caller fall back to whatever ``--hook_layers`` provided.
    """
    table = {
        "classification": {
            "resnet34": RESNET34_HOOK_LAYERS,
            "vit_b_16": VIT_B16_HOOK_LAYERS,
            "dinov2":   DINOV2_HOOK_LAYERS,
        },
        "segmentation": {
            "resnet34": SEG_RESNET34_HOOK_LAYERS,
            "vit_b_16": SEG_VIT_B16_HOOK_LAYERS,
            "dinov2":   SEG_DINOV2_HOOK_LAYERS,
        },
    }[task]
    return table.get(arch, [])


def register_hooks(model, layer_names: list[str]):
    """
    Register forward hooks on named sub-modules to capture their outputs.

    The captured tensor is stored in ``features[name]`` and overwritten on every
    forward pass, so a caller that needs to compare two passes must copy the
    dict between them.  Handles are returned so they can be removed after the
    evaluation loop.

    Shared by both corruption scripts so that "the layer called X" means the
    same module in the classification and the segmentation sweep.

    Args:
        model       : the model to hook.
        layer_names : dot-separated sub-module names as returned by
                      ``model.named_modules()``.

    Returns:
        ``(features, handles)``: ``features`` is updated in place on each
        forward pass, ``handles`` is the list to call ``.remove()`` on.
    """
    features: dict = {}
    handles: list = []

    named = dict(model.named_modules())
    for name in layer_names:
        if name not in named:
            print(f"  [WARN] Layer '{name}' not found in model - skipping.")
            continue

        def _make_hook(n: str):
            def _hook(module, inp, out):
                # Raw output tensor:
                #   ResNet : (B, C, H, W)
                #   ViT    : (B, T, D)  with T = N_patches + prefix tokens
                features[n] = out.detach()
            return _hook

        handles.append(named[name].register_forward_hook(_make_hook(name)))

    return features, handles


# =============================================================================
# Corruption dispatch
# =============================================================================
def corruption_seed_key(cfg: Box) -> tuple[int, int]:
    """
    Return ``(cfg_hash, base_seed)``, the two halves of the per-sample seed.

    ``zlib.crc32`` and not ``hash()``: Python randomises string hashes per
    process unless PYTHONHASHSEED is set, so ``hash()`` would silently break
    reproducibility across runs.

    Args:
        cfg : single-experiment config.

    Returns:
        ``(cfg_hash, base_seed)`` where ``cfg_hash`` identifies the corruption
        configuration and ``base_seed`` is ``cfg.seed``.
    """
    key = "|".join(str(cfg.get(k)) for k in
                   ("corruption", "perturbation_type", "perturbation_ratio",
                    "severity_level", "squared_size"))
    return zlib.crc32(key.encode()) & 0x7FFFFFFF, int(cfg.seed)


def apply_corruption(image: np.ndarray, cfg: Box, sample_seed: int):
    """
    Apply the configured corruption to one uint8 HWC RGB image.

    The corruption functions and the ImageNet-C functions both draw from the
    GLOBAL numpy RNG, for the mask and for the noise respectively.  Seeding it
    deterministically here and restoring it afterwards means the same image
    receives the same mask and the same noise realisation for every
    architecture, every run, and any number of DataLoader workers, which is
    what makes the comparison paired.

    Args:
        image       : ``(H, W, C)`` uint8 RGB array.
        cfg         : single-experiment config.
        sample_seed : seed for this (config, sample) pair.

    Returns:
        ``(corrupted_image, corruption_mask)``.

    Raises:
        ValueError : if ``cfg.corruption`` is not a known corruption shape.
    """
    _state = np.random.get_state()
    np.random.seed(sample_seed)
    try:
        c = cfg.corruption
        if c == "squared":
            return squared(image, [cfg.perturbation_type],
                           cfg.perturbation_ratio, cfg.severity_level,
                           cfg.squared_size)
        if c == "circular":
            return circular(image, cfg.perturbation_type,
                            cfg.perturbation_ratio, cfg.severity_level)
        if c == "superpixel":
            return superpixel(image, cfg.perturbation_type,
                              cfg.perturbation_ratio, cfg.severity_level)
        if c == "global":
            return global_corr(image, [cfg.perturbation_type],
                               cfg.severity_level)
        raise ValueError(
            f"Invalid corruption '{c}'. "
            "Choose: squared, circular, superpixel, global."
        )
    finally:
        np.random.set_state(_state)


# Corruptions that move pixels.  In segmentation the target would need the same
# warp; in classification the label is global and does not change, so the guard
# only applies to the first case.
GEOMETRIC = {"Elastic", "Zoom Blur"}


def unsupported_reason(cfg: Box, task: str) -> str | None:
    """
    Why this combination cannot be run, or ``None`` when it can.

    Checked before the sweep is chunked, so a rejected combination is dropped
    once with a clear message instead of failing inside a DataLoader worker.

    Args:
        cfg  : single-experiment config.
        task : ``"classification"`` or ``"segmentation"``.
    """
    if task == "segmentation" and cfg.perturbation_type in GEOMETRIC:
        return (f"'{cfg.perturbation_type}' moves pixels; the segmentation "
                "target would need the same warp. Not supported.")
    return None


# =============================================================================
# Corrupted dataset, one class for both tasks
# =============================================================================
def _to_chw_uint8(img_hwc: np.ndarray) -> torch.Tensor:
    """``(H, W, C)`` uint8 numpy to a contiguous ``(C, H, W)`` uint8 tensor."""
    return torch.from_numpy(
        np.ascontiguousarray(img_hwc.astype(np.uint8))
    ).permute(2, 0, 1).contiguous()


class CorruptedDataset(Dataset):
    """
    Wrap a registry dataset and corrupt each image with SEVERAL configs at once.

    One class covers both tasks because it goes through ``.raw(idx)``, which
    ``FolderClassificationDataset`` and ``SegmentationDataset`` both expose.
    The target is an int in the first case and an ``(H, W)`` mask in the
    second, and that is the only difference in what is returned.

    Returning K corrupted versions of the same image, instead of one, is what
    lets the evaluation loop read and decode the image once and run the clean
    forward pass once for the whole chunk.  It also means every ``superpixel``
    config of the chunk hits the SLIC cache warmed by the first one.

    Per ``__getitem__``:

    * ``clean``     -- ``(3, H, W)`` uint8.
    * ``corrupted`` -- ``(K, 3, H, W)`` uint8, one per config.
    * ``cmask``     -- ``(K, 1, H, W)`` uint8, 1 where the corruption was applied.
    * ``target``    -- scalar int64 label, or ``(H, W)`` int64 mask.

    Images stay uint8 on purpose: normalisation happens on the GPU through
    :func:`normalize_batch`, which keeps the host memory of a K-wide batch
    below that of the single-config float32 batch this replaced.

    Args:
        base    : dataset built by ``shared.DatasetRegistry.build_dataset``.
        cfgs    : list of single-experiment configs.
        indices : subset of dataset indices, normally the test split.  Kept
                  here instead of wrapping in ``torch.utils.data.Subset``
                  because the mask seed uses the ORIGINAL dataset index, so an
                  image gets the same corruption whichever split it lands in.
    """

    def __init__(self, base, cfgs: list[Box], indices=None):
        if not cfgs:
            raise ValueError("CorruptedDataset needs at least one config.")

        self.base = base
        self.cfgs = list(cfgs)
        self.indices = list(range(len(base))) if indices is None else list(indices)
        self._keys = [corruption_seed_key(c) for c in self.cfgs]

    def _sample_seed(self, k: int, idx: int) -> int:
        """Deterministic seed per (config, sample); identical across archs."""
        cfg_hash, base_seed = self._keys[k]
        ss = np.random.SeedSequence([base_seed, int(idx), cfg_hash])
        return int(ss.generate_state(1)[0])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        # Seed on the ORIGINAL dataset index, not on the position in the split.
        idx = self.indices[i]
        image, target = self.base.raw(idx)

        corrupted, masks = [], []
        for k, cfg in enumerate(self.cfgs):
            corr, cmask = apply_corruption(image, cfg, self._sample_seed(k, idx))
            corrupted.append(_to_chw_uint8(corr))
            cm = torch.from_numpy(np.asarray(cmask, dtype=np.uint8))
            if cm.ndim == 2:
                cm = cm.unsqueeze(0)
            masks.append(cm)

        # ndarray target means a segmentation mask, anything else is a label.
        tgt = (torch.from_numpy(target.astype(np.int64))
               if isinstance(target, np.ndarray)
               else torch.tensor(int(target), dtype=torch.long))

        return (
            _to_chw_uint8(image),
            torch.stack(corrupted),        # (K, 3, H, W)
            torch.stack(masks),            # (K, 1, H, W)
            tgt,
        )


# =============================================================================
# GPU-side normalisation
# =============================================================================
_NORM_CACHE: dict = {}


def normalize_batch(x_uint8: torch.Tensor) -> torch.Tensor:
    """
    ``(B, 3, H, W)`` uint8 to an ImageNet-normalised float32 batch.

    Same arithmetic as ``shared.DatasetRegistry.normalize_image``, moved to
    wherever the tensor already is.  On the GPU it is essentially free, and it
    lets the DataLoader ship uint8, which is four times less to allocate, pin
    and copy.

    Args:
        x_uint8 : uint8 batch on any device.

    Returns:
        float32 tensor on the same device.
    """
    stats = _NORM_CACHE.get(x_uint8.device)
    if stats is None:
        mean = torch.tensor(IMAGENET_MEAN, device=x_uint8.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x_uint8.device).view(1, 3, 1, 1)
        stats = (mean, std)
        _NORM_CACHE[x_uint8.device] = stats
    mean, std = stats
    return x_uint8.float().div_(255.0).sub_(mean).div_(std)


# =============================================================================
# Mixed precision
# =============================================================================
def amp_dtype(enabled: bool, device: torch.device):
    """
    Which dtype to run the forward passes in, or ``None`` for plain float32.

    bfloat16 is preferred over float16 when the GPU supports it: same speed,
    the same exponent range as float32, so no scaling and nothing to overflow.
    Everything downstream of the forward pass stays in float32 because
    ``compute_metrics`` upcasts, so the metrics keep full precision.

    Args:
        enabled : value of ``--amp``.
        device  : target device; AMP is skipped on CPU.
    """
    if not enabled or device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# =============================================================================
# Regions and aggregation
# =============================================================================
def region_masks(cmask: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    The three regions the feature metrics are decomposed over.

    No thresholding here.  Projecting the mask onto the feature grid, and
    choosing between a binary and a continuous weight, belong to
    ``corruption_metrics.project_mask``, which is the only place the 0.5
    threshold should appear.

    Args:
        cmask : ``(B, 1, H, W)`` float mask in image space.

    Returns:
        Dict mapping ``"corrupted"``, ``"non_corrupted"``, ``"global"`` to a
        mask of the same shape.
    """
    return {
        "corrupted":     cmask,
        "non_corrupted": 1.0 - cmask,
        "global":        torch.ones_like(cmask),
    }


def weighted_nanmean(pairs: list[tuple[float, float]]) -> float:
    """
    Average per-batch values weighted by how much each batch contributed.

    Args:
        pairs : ``(value, weight)`` per batch, where the weight is the total
                mask weight returned by ``compute_metrics``.  A plain mean
                would give the last, partially filled batch the same influence
                as a full one.  NaN entries (empty mask) are skipped.

    Returns:
        The weighted mean, or NaN when no batch contributed.
    """
    v = np.array([p[0] for p in pairs], dtype=float)
    w = np.array([p[1] for p in pairs], dtype=float)
    ok = ~np.isnan(v) & (w > 0)
    if not ok.any():
        return float("nan")
    return float((v[ok] * w[ok]).sum() / w[ok].sum())


def to_float(t) -> float:
    """Scalar tensor to float; NaN when ``t`` is None, so nanmean skips it."""
    return float("nan") if t is None else float(t.item())


def nanmean_chunks(chunks: list) -> float:
    """
    Mean over a list of per-batch tensors or scalars, ignoring NaN.

    ``.cpu()`` before ``.numpy()``: the per-batch tensors collected in the
    evaluation loops still live on the GPU, since nothing moves them.

    Args:
        chunks : list of 1-D tensors, or of plain floats.

    Returns:
        The mean, or NaN when there is nothing to average.
    """
    if not chunks:
        return float("nan")
    if torch.is_tensor(chunks[0]):
        x = torch.cat(chunks).detach().cpu().numpy()
    else:
        x = np.asarray(chunks, dtype=float)
    return float(np.nanmean(x)) if x.size else float("nan")


# =============================================================================
# Sweep configuration
# =============================================================================
def make_experiment_cfg(base_cfg: Box, corruption: str,
                        perturbation_ratio: float, perturbation_family: str,
                        perturbation_type: str, severity_level: int,
                        squared_size: int | None = None) -> Box:
    """
    Build the config of one combination from the sweep config.

    Args:
        base_cfg            : config produced by ``parse_args``.
        corruption          : corruption shape (``"circular"``, ...).
        perturbation_ratio  : fraction of the image to corrupt.
        perturbation_family : family label, e.g. ``"NOISE"``.
        perturbation_type   : perturbation name, e.g. ``"Gaussian Noise"``.
        severity_level      : severity in [1, 5].
        squared_size        : patch side in pixels, only for ``"squared"``.

    Returns:
        A :class:`~box.Box` describing one experiment.
    """
    exp = Box(dict(base_cfg))
    for k in ("corruption_types", "perturbation_ratios", "severity_levels",
              "perturbation_types", "squared_sizes"):
        exp.pop(k, None)
    exp.corruption          = corruption
    exp.perturbation_ratio  = perturbation_ratio
    exp.perturbation_family = perturbation_family
    exp.perturbation_type   = perturbation_type
    exp.severity_level      = severity_level
    if squared_size is not None:
        exp.squared_size = squared_size
    return exp


def sweep_combinations(cfg: Box) -> list[tuple]:
    """
    Every ``(family, ptype, corruption, ratio, severity, size)`` of the sweep.

    Materialising them in a list instead of using five nested loops means the
    progress-bar total is ``len(...)`` and cannot drift from what the loop
    actually runs.

    The corruption shape varies slowly here, so consecutive combinations tend
    to share it.  That is what makes a chunk of ``--combo_chunk`` combinations
    reuse the same SLIC segmentation over and over.

    Args:
        cfg : sweep config.

    Returns:
        List of tuples; ``size`` is None for every shape but ``"squared"``.
    """
    return [
        (family, ptype, corruption, ratio, severity, size)
        for family, ptypes in cfg.perturbation_types.items()
        for ptype in ptypes
        for corruption in cfg.corruption_types
        for ratio in cfg.perturbation_ratios
        for severity in cfg.severity_levels
        for size in (cfg.squared_sizes if corruption == "squared" else [None])
    ]


def build_experiment_cfgs(cfg: Box, task: str) -> list[Box]:
    """
    Every runnable combination of the sweep, expanded and filtered.

    Args:
        cfg  : sweep config.
        task : ``"classification"`` or ``"segmentation"``.

    Returns:
        List of single-experiment configs, in sweep order.
    """
    out = []
    for family, ptype, corruption, ratio, severity, size in sweep_combinations(cfg):
        exp = make_experiment_cfg(cfg, corruption, ratio, family, ptype,
                                  severity, size)
        reason = unsupported_reason(exp, task)
        if reason:
            print(f"  [SKIP] {corruption}/{ptype}: {reason}")
            continue
        out.append(exp)
    return out


def chunked(items: list, size: int) -> list[list]:
    """
    Split ``items`` into consecutive groups of at most ``size``.

    Each group is one pass over the dataset, so the group size trades memory
    (K corrupted copies of every image in a batch) against speed (one image
    decode and one clean forward pass shared by the whole group).
    """
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]


# =============================================================================
# Run naming
# =============================================================================
def variant_suffix(normalize_features: bool, soft_masking: bool) -> str:
    """
    Name of the feature / masking variant of a run.

    One of ``rawfeat``, ``normfeat``, ``rawfeat_softmasking``,
    ``normfeat_softmasking``.  It is appended to the run folder so the four
    variants of one checkpoint never overwrite each other.
    """
    feat = "normfeat" if normalize_features else "rawfeat"
    mask = "_softmasking" if soft_masking else ""
    return f"{feat}{mask}"
