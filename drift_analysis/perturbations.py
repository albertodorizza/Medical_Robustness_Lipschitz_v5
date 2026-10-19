"""
drift_analysis/perturbations.py
===============================
Pixel-space perturbations shared by ``drift_analysis_cls.py`` and
``drift_analysis_seg.py``.

Everything is applied in PIXEL space: the normalised batch is de-normalised to
[0, 1], perturbed, and re-normalised.  That is the domain the ImageNet-C
functions are defined on, so the severity levels here mean exactly what they
mean in ``corruption_experiments``, and the two stages stay comparable.
"""

import numpy as np
import torch
from shared.constants import IMAGENET_MEAN, IMAGENET_STD

_MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

# Cached per device: these are called twice per batch, and re-uploading two
# tiny tensors every time shows up in the profile of a long sweep.
_STATS_CACHE: dict = {}


def _stats(device) -> tuple[torch.Tensor, torch.Tensor]:
    """ImageNet mean and std, materialised once per device."""
    stats = _STATS_CACHE.get(device)
    if stats is None:
        stats = (_MEAN.to(device), _STD.to(device))
        _STATS_CACHE[device] = stats
    return stats


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalised (B,3,H,W) -> [0,1]."""
    mean, std = _stats(x.device)
    return x * std + mean


def normalize(x01: torch.Tensor) -> torch.Tensor:
    """[0,1] (B,3,H,W) -> ImageNet-normalised."""
    mean, std = _stats(x01.device)
    return (x01 - mean) / std


def _to_uint8_hwc(x01: torch.Tensor) -> np.ndarray:
    """(B,3,H,W) float in [0,1] -> (B,H,W,3) uint8, the format ImageNet-C wants."""
    arr = x01.clamp(0, 1).permute(0, 2, 3, 1).detach().cpu().numpy()
    return (arr * 255.0).round().astype(np.uint8)


def _from_float_hwc(arr: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    """(B,H,W,3) float in [0,255] -> normalised (B,3,H,W) tensor on like.device."""
    t = torch.from_numpy(np.ascontiguousarray(arr)).float().div_(255.0)
    t = t.permute(0, 3, 1, 2).clamp(0, 1).to(like.device)
    return normalize(t)


# =============================================================================
# Perturbations
# =============================================================================
def gaussian_perturbation(x: torch.Tensor, eps: float = 0.01,
                          generator: torch.Generator | None = None
                          ) -> torch.Tensor:
    """
    x_tilde = clip(x + eps * N(0, I), 0, 1), applied in PIXEL space [0, 1].

    Args:
        x         : normalised batch (B, 3, H, W).
        eps       : noise std in [0, 1] units.  0.01 = 2.55 grey levels.
        generator : pass a seeded torch.Generator for reproducible runs.
    """
    x01 = denormalize(x)
    noise = torch.randn(x01.shape, generator=generator,
                        device="cpu" if generator is not None else x01.device)
    noise = noise.to(x01.device)
    return normalize((x01 + eps * noise).clamp(0, 1))


def imagenet_c_perturbation(x: torch.Tensor, name: str,
                            severity: int = 1) -> torch.Tensor:
    """
    Apply any ImageNet-C corruption in the space it was designed for.

    Args:
        x        : normalised batch (B, 3, H, W).
        name     : key of `corruptions.imagenet_c.return_list_transformations()`,
                   e.g. "Contrast", "Gaussian Noise", "Defocus Blur".
        severity : 1..5.

    Using this instead of a hand-written contrast function also means the
    severity semantics in drift_analysis are IDENTICAL to those in
    corruption_experiments, which was not the case before.
    """
    from corruptions import imagenet_c

    fn = imagenet_c.return_list_transformations()[name]
    imgs = _to_uint8_hwc(denormalize(x))                 # (B,H,W,3) uint8
    out = np.stack([np.asarray(fn(im.copy(), severity), dtype=np.float32)
                    for im in imgs])                     # (B,H,W,3) in [0,255]
    return _from_float_hwc(out, x)


def make_perturbation_fn(kind: str, severity: int = 1, eps: float = 0.01,
                         seed: int | None = None):
    """
    Build the callable used by `analyze_drift`.

    Args:
        kind : "gaussian" for the small random probe, or any ImageNet-C name
               ("Contrast", "Gaussian Noise", "Brightness", "Defocus Blur", ...).
        severity : only used for ImageNet-C corruptions.
        eps      : only used for "gaussian".
        seed     : seed for the gaussian probe, so the same noise realisation
                   is used for every architecture (paired comparison).
    """
    if kind.lower() == "gaussian":
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        return lambda x: gaussian_perturbation(x, eps=eps, generator=gen)
    return lambda x: imagenet_c_perturbation(x, kind, severity=severity)
