"""
shared/constants.py
===================
Project-wide constants shared across all modules.

Centralising these values ensures that every script (finetuning,
drift analysis, corruption experiments) uses identical normalization
statistics.
"""

# ── ImageNet normalization ────────────────────────────────────────────────────
# These statistics are the standard per-channel mean and standard deviation
# computed over the ImageNet training set.  

IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD:  list[float] = [0.229, 0.224, 0.225]
