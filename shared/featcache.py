"""
shared/featcache.py
===================
Backbone-feature cache for ``freeze_mode="head"`` runs, shared by
``finetuning/finetuning_cls.py`` and ``finetuning/finetuning_seg.py``.

Why this is exact, and not an approximation
-------------------------------------------
With ``freeze_mode="head"`` the backbone is never updated, and three further
properties of this pipeline make its output a pure function of the image:

* every BatchNorm is forced to ``.eval()`` at the top of each epoch in both
  training loops, so it always uses the pretrained running statistics and never
  the statistics of the current batch;
* the registry's preprocessing is deterministic -- resize plus ImageNet
  normalisation, no augmentation anywhere (see ``shared/DatasetRegistry.py``);
* ViT-B/16 and DINOv2 are built with dropout 0, and ResNet-34 has none.

So for a fixed ``(dataset, model_type, input_size)`` the backbone output for a
given image is identical at every epoch and identical across every run that
shares that triple -- ``lambda_sn`` and ``split_seed`` only change the head and
the order in which samples are visited.  Computing the features once and
replaying them is therefore the same arithmetic as before, not a cheaper
approximation of it: what disappears is a forward pass whose result was already
known.

The saving is the whole backbone, forward and backward, for every epoch of
every ``head`` run after the first one of its group.  Half of each sweep grid
is ``freeze_mode=head``.

What this does NOT cover
------------------------
``freeze_mode="head+last"``: there the last backbone stage is trainable, so its
output changes at every optimizer step and there is nothing to cache.  (The
INPUT of that last stage is cacheable in principle, but that is a separate,
more invasive change and is not done here.)  Those runs keep the original path,
untouched.

RNG safety
----------
The extraction DataLoader is built with its own ``torch.Generator``, so it does
not draw its base seed from the global torch RNG.  A run that has to extract
the cache and a run that finds it already on disk therefore consume the global
RNG identically, and both consume it exactly as the pre-cache code did -- the
shuffling order of every epoch and the head initialisation are unchanged, so
results stay comparable with runs produced before this file existed.

Concurrency
-----------
A sweep launches many jobs that share the same cache entry.  Extraction is
serialised with an exclusive ``flock`` and the file is published with
``os.replace``, so the first job to arrive computes it while the others wait a
few seconds and then read it; no job ever observes a half-written ``.npy``.

Layout on disk
--------------
``$FEATCACHE_DIR`` (default ``featcache/`` in the working directory)::

    <dataset>_<model_type>_<task>_<digest>.npy   backbone features, float32
    <dataset>_<task>_targets_<digest>.npy        segmentation masks, uint8

``digest`` covers the dataset key, the input size, the number of samples and
the full list of file paths, so a dataset that gains, loses or reorders images
gets a different entry instead of silently reusing a stale one.
"""

import hashlib
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:                                   # POSIX only; absent on Windows
    import fcntl
except ImportError:                    # pragma: no cover
    fcntl = None


__all__ = ["features_cls", "features_seg", "index_loader", "default_dir"]


# =============================================================================
# Cache identity
# =============================================================================
def default_dir() -> str:
    """Where cache files live.  Set ``FEATCACHE_DIR`` to move it off the repo."""
    return os.environ.get("FEATCACHE_DIR", "featcache")


def _sample_paths(dataset) -> list[str]:
    """File paths of every sample, in dataset order (empty if unknown)."""
    if hasattr(dataset, "samples"):            # FolderClassificationDataset
        return [p for p, _ in dataset.samples]
    if hasattr(dataset, "pairs"):              # SegmentationDataset
        return [p for p, _ in dataset.pairs]
    return []


def _digest(dataset, dataset_name: str, input_size) -> str:
    """
    Fingerprint of the exact sample list the features were computed from.

    Length alone is not enough: two runs can see the same number of images
    after a dataset is re-exported or a folder is renamed, and reusing features
    computed from different files would silently corrupt every result
    downstream.
    """
    h = hashlib.sha1()
    size = tuple(int(s) for s in input_size)
    h.update(f"{dataset_name}|{size}|{len(dataset)}".encode())
    for p in _sample_paths(dataset):
        h.update(p.encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


# =============================================================================
# Filesystem helpers
# =============================================================================
class _Lock:
    """Exclusive inter-process lock on ``<path>.lock``; a no-op without fcntl."""

    def __init__(self, path: str):
        self.path = path + ".lock"
        self.fd = None

    def __enter__(self):
        if fcntl is None:
            return self
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.fd = open(self.path, "w")
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.fd.close()
            self.fd = None
        return False


def _save_atomic(path: str, arr: np.ndarray) -> None:
    """Write to a private temporary file, then publish it with a rename."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def _place(t: torch.Tensor, device: torch.device, name: str) -> torch.Tensor:
    """
    Put the cache on the GPU when it comfortably fits, otherwise leave it in
    host memory.

    A sweep runs several jobs per card, so taking half of what is free would be
    antisocial even when it works; the fallback still skips the backbone, it
    only pays a host-to-device copy per batch, which is a fraction of what it
    replaces.
    """
    nbytes = t.numel() * t.element_size()
    if device.type != "cuda":
        return t
    free, _total = torch.cuda.mem_get_info(device)
    if nbytes < 0.25 * free:
        return t.to(device)
    print(f"  [featcache] {name} is {nbytes / 2**20:.0f} MB and only "
          f"{free / 2**20:.0f} MB are free on {device}; keeping it in host "
          "memory and streaming it per batch.")
    try:
        return t.pin_memory() if nbytes < 2 * 2**30 else t
    except RuntimeError:
        return t


# =============================================================================
# Feature extraction
# =============================================================================
def _cls_features(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Activation immediately before the linear head, mirroring each backbone's
    own ``forward`` exactly:

    * ResNet-34 (torchvision) -- everything except ``m.fc``.
    * ViT-B/16 (torchvision)  -- ``_process_input`` + class token + encoder
      (torchvision's Encoder already applies the positional embedding, dropout
      and the final LayerNorm), keeping the CLS token: everything except
      ``m.heads``.
    * DINOv2Wrapper           -- ``m.backbone(x)``: everything except ``m.head``.
    """
    if hasattr(model, "fc"):                   # ResNet-34
        m = model
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        return torch.flatten(m.avgpool(x), 1)

    if hasattr(model, "heads"):                # ViT-B/16
        m = model
        x = m._process_input(x)
        cls = m.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = m.encoder(x)
        return x[:, 0]

    if hasattr(model, "backbone"):             # DINOv2Wrapper
        return model.backbone(x)

    raise ValueError("Unrecognised backbone: no .fc, .heads or .backbone.")


def _seg_features(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Feature grid feeding the 1x1 convolution head of a SegmentationModel."""
    return model.backbone(x)


@torch.no_grad()
def _extract(model, dataset, device, feature_fn, want_targets: bool,
             batch_size: int, num_workers: int):
    """
    One ordered pass over the whole dataset, returning ``(features, targets)``
    as numpy arrays indexed exactly like ``dataset``.

    ``targets`` is ``None`` for classification, where the labels are already
    available as ``dataset.labels`` and need no cache of their own.
    """
    was_training = model.training
    model.eval()

    # Own generator: building a DataLoader iterator draws a base seed, and with
    # generator=None that seed comes from the GLOBAL torch RNG.  Taking it from
    # here instead is what keeps a cache-miss run and a cache-hit run on the
    # same RNG stream (see the module docstring).
    gen = torch.Generator()
    gen.manual_seed(0)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        generator=gen)

    feats = targets = None
    i = 0
    for images, y in loader:
        f = feature_fn(model, images.to(device, non_blocking=True))
        f = f.float().cpu().numpy()
        if feats is None:
            feats = np.empty((len(dataset), *f.shape[1:]), dtype=np.float32)
            if want_targets:
                targets = np.empty((len(dataset), *y.shape[1:]), dtype=np.uint8)
        n = f.shape[0]
        feats[i:i + n] = f
        if want_targets:
            targets[i:i + n] = y.numpy().astype(np.uint8)
        i += n

    if i != len(dataset):
        raise RuntimeError(f"expected {len(dataset)} features, extracted {i}")

    if was_training:
        model.train()
    return feats, targets


def _build_or_load(model, dataset, dataset_name, model_type, task, input_size,
                   device, feature_fn, want_targets, batch_size, num_workers,
                   cache_dir):
    cdir = cache_dir or default_dir()
    os.makedirs(cdir, exist_ok=True)
    d = _digest(dataset, dataset_name, input_size)

    fpath = os.path.join(cdir, f"{dataset_name}_{model_type}_{task}_{d}.npy")
    tpath = (os.path.join(cdir, f"{dataset_name}_{task}_targets_{d}.npy")
             if want_targets else None)

    def _ready() -> bool:
        return (os.path.exists(fpath)
                and (tpath is None or os.path.exists(tpath)))

    if not _ready():
        # Sweep jobs sharing this entry queue here; the first one computes it,
        # the rest fall through to the load below a few seconds later.
        with _Lock(fpath):
            if not _ready():
                print(f"  [featcache] extracting {task} features for "
                      f"{dataset_name}/{model_type} -> {fpath}")
                feats, targets = _extract(model, dataset, device, feature_fn,
                                          want_targets, batch_size, num_workers)
                _save_atomic(fpath, feats)
                if tpath is not None:
                    _save_atomic(tpath, targets)

    print(f"  [featcache] using {fpath}")
    feats = _place(torch.from_numpy(np.load(fpath)), device, "features")
    if tpath is None:
        return feats
    targets = _place(torch.from_numpy(np.load(tpath)), device, "targets")
    return feats, targets


# =============================================================================
# Public API
# =============================================================================
def features_cls(model, dataset, dataset_name: str, model_type: str,
                 input_size, device: torch.device, batch_size: int = 128,
                 num_workers: int = 4, cache_dir: str | None = None
                 ) -> torch.Tensor:
    """
    Pre-head activations of the frozen backbone, ``(N, D)``, in dataset order.

    Call it with the model already built (pretrained backbone loaded, fresh
    head -- the head itself is not used here).
    """
    return _build_or_load(model, dataset, dataset_name, model_type,
                          "cls", input_size, device, _cls_features,
                          want_targets=False, batch_size=batch_size,
                          num_workers=num_workers, cache_dir=cache_dir)


def features_seg(model, dataset, dataset_name: str, model_type: str,
                 input_size, device: torch.device, batch_size: int = 32,
                 num_workers: int = 4, cache_dir: str | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Feature grids of the frozen backbone, ``(N, D, Hf, Wf)``, plus the masks,
    ``(N, H, W)`` uint8, both in dataset order.

    The masks are cached alongside because the cached training path never opens
    an image again, and re-reading them from disk every epoch would put back
    most of the IO the cache is there to remove.  They are keyed by dataset
    only, so the three architectures share one copy.
    """
    return _build_or_load(model, dataset, dataset_name, model_type,
                          "seg", input_size, device, _seg_features,
                          want_targets=True, batch_size=batch_size,
                          num_workers=num_workers, cache_dir=cache_dir)


def index_loader(indices, batch_size: int, shuffle: bool,
                 drop_last: bool = False) -> DataLoader:
    """
    DataLoader over sample INDICES rather than over samples.

    Two reasons for the indirection.  It keeps the random number stream
    identical to the image path -- the same ``RandomSampler`` over the same
    number of items, so every epoch visits the samples in exactly the order it
    did before this cache existed.  And it gathers a batch with one indexing op
    on the cached tensor instead of slicing every sample separately and
    stacking them, which for segmentation means one copy of 32 feature maps
    rather than 32 copies of one.

    ``num_workers=0`` on purpose: the data is already a resident tensor, there
    is no JPEG left to decode, and worker processes would only add IPC to move
    something that is already in place.
    """
    idx = torch.as_tensor(np.asarray(indices), dtype=torch.long)
    return DataLoader(TensorDataset(idx), batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last, num_workers=0)
