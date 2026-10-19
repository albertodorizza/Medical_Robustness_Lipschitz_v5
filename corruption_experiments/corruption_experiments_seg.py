"""
corruption_experiments/corruption_experiments_seg.py
====================================================
Robustness evaluation of fine-tuned segmentation models under localized and
global corruptions.

Segmentation counterpart of ``corruption_experiments_cls.py``.  For every
combination of (corruption shape x ratio x severity x perturbation type) it:

1. Applies the corruption on-the-fly, through
   `corruption_experiments.corruption_common`, so masks and noise are
   generated exactly as in the classification script.
2. Runs a forward pass on the clean image and one on the corrupted image.
3. Computes MSE and cosine similarity between clean and corrupted feature maps
   in three spatial regions: corrupted, non-corrupted, full image.
4. Records Dice and IoU on clean and corrupted inputs.
5. Saves everything to ``metrics.json``.

``--hook_depth`` additionally measures the same drift at every intermediate
block of the backbone and writes it to a ``layer_metrics`` block, so the depth
profile that exists for classification can be drawn for segmentation too.  The
deepest block is deliberately not hooked: for every backbone it is the tensor
already reported as ``feature_metrics`` (ResNet returns ``layer4``; the ViT and
DINOv2 wrappers reshape the tokens of the final norm), so hooking it would
record the same measurement twice.

What segmentation adds over classification
------------------------------------------
Output degradation is measured separately
inside and outside the corrupted region, exactly mirroring the ``MSE_in`` /
``MSE_out`` feature decomposition.  In classification the output is a single
label, so only the feature side could be decomposed; here both sides can, and
the two can be compared directly.

**Overlap between the corruption and the lesion.**  Every sample is assigned to
one of three strata according to how much of the ground-truth lesion falls
under the corruption mask: ``none`` (no overlap), ``partial``, ``full``.  The
interesting question is the ``none`` stratum: if an artefact that never touches
the polyp still degrades its segmentation, the corruption has propagated
through the representation, which is precisely the claim the feature-level
metrics are meant to support, and here it is measured at the output, in a
clinically readable unit.

Chunked sweep
-------------
Combinations are processed ``--combo_chunk`` at a time.  Every image of a chunk
is read, decoded and forwarded CLEAN exactly once, and only the corrupted
forward pass is repeated per combination: with K combinations per chunk the
sweep does K+1 forward passes per batch instead of 2K, and reads each image
once instead of K times.  The clean Dice of a chunk is therefore measured once
and shared by its combinations, which is also what it always was numerically,
just recomputed K times before.

Usage (see ``run_corruption_experiments_seg.sh`` for the full sweep)::

    python -m corruption_experiments.corruption_experiments_seg \
        --arch       dinov2 \
        --checkpoint /path/to/dinov2_head+last.pth \
        --dataset    kvasir_seg \
        --output_dir /path/to/output
"""

import argparse
import json
import os
import sys
import torch

from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from box import Box
import numpy as np

from tqdm import tqdm

# Ensure the project root is on sys.path when running the script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corruption_experiments.corruption_common import (
    CorruptedDataset, amp_dtype, build_experiment_cfgs, chunked,
    get_default_hook_layers, nanmean_chunks, normalize_batch, region_masks,
    register_hooks, to_float, variant_suffix, weighted_nanmean,
)
from corruption_experiments.corruption_metrics import compute_metrics
from shared.DatasetRegistry import build_dataset, get_spec
from shared.paths import (parse_run_name, run_folder_from_checkpoint,
                          split_file_dir)
from shared.runtime import pin_cpu_cores
from shared.metrics import (foreground_dice, foreground_iou,
                            nanmean_count)
from shared.SegmentationModels import build_segmentation_model
from shared.seed import reset_seed
from shared.splits import load_or_make_split
import shared.wandb_logging as wb

REGION_TYPES = ["corrupted", "non_corrupted", "global"]


# =============================================================================
# Segmentation metrics
# =============================================================================
def regional_dice(pred, target, region):
    """
    Per-image foreground Dice restricted to a spatial region.

    Thin alias over :func:`shared.metrics.foreground_dice`, kept because the
    region argument is what gives the inside/outside decomposition and reads
    better under that name here.  The definition itself lives in one place now,
    shared with ``finetuning_seg.py`` and ``drift_analysis_seg.py``, so the
    "clean Dice" column of the corruption tables and the Dice of the baseline
    table are the same estimator rather than two that differ on lesion-free
    images.
    """
    return foreground_dice(pred, target, region)


def overlap_stratum(target, cmask, tol=0.01):
    """
    Classify how much of the lesion is covered by the corruption.

    Args:
        target : ``(B, H, W)`` ground truth.
        cmask  : ``(B, H, W)`` boolean corruption mask.
        tol    : fractions below ``tol`` count as no overlap and above
                 ``1 - tol`` as full, absorbing single-pixel boundary effects.

    Returns:
        List of ``"none" | "partial" | "full" | "no_lesion"`` of length B.
    """
    lesion = (target == 1)
    area = lesion.sum(dim=(1, 2)).float()
    covered = (lesion & cmask).sum(dim=(1, 2)).float()
    frac = torch.where(area > 0, covered / area.clamp(min=1),
                       torch.full_like(area, -1.0))

    out = []
    for f in frac.tolist():
        if f < 0:
            out.append("no_lesion")
        elif f <= tol:
            out.append("none")
        elif f >= 1 - tol:
            out.append("full")
        else:
            out.append("partial")
    return out


# =============================================================================
# WandB logging helper
# =============================================================================
def _log_to_wandb(result: dict, cfg: Box) -> None:
    """
    Log the scalar metrics of a single experiment.

    Twin of ``_log_to_wandb`` in ``corruption_experiments_cls.py``: same config
    keys, a different output metric (Dice instead of accuracy), plus the column
    the classification task cannot produce, i.e. the Dice drop on the samples
    where the corruption never touches the lesion.

    Args:
        result : one result dict returned by :func:`run_chunk`.
        cfg    : config of the experiment that produced it.
    """
    import wandb

    log = {
        "dice_clean":           result["dice_clean"],
        "dice_corrupted":       result["dice_corrupted"],
        "dice_drop":            result["dice_drop"],
        "dice_drop_no_overlap": result["dice_drop_by_overlap"]["none"],
        "effective_ratio":      result["effective_ratio"],
        "corruption":           cfg.corruption,
        "perturbation_family":  cfg.perturbation_family,
        "perturbation_type":    cfg.perturbation_type,
        "ratio":                cfg.perturbation_ratio,
        "severity":             cfg.severity_level,
    }
    if cfg.corruption == "squared":
        log["squared_size"] = cfg.squared_size
    wandb.log(log)


# =============================================================================
# Chunk runner
# =============================================================================
def run_chunk(cfg: Box, exp_cfgs: list[Box], model, base_ds, eval_idx,
              device, features: dict | None = None,
              hook_layers: list[str] | None = None) -> list[dict]:
    """
    Run a group of combinations in a single pass over the evaluation split.

    Args:
        cfg      : sweep config, read for the loader and device settings.
        exp_cfgs : the combinations of this chunk, in sweep order.
        model    : segmentation model in eval mode, shared across the sweep.
        base_ds  : dataset built from the registry.
        eval_idx : indices to evaluate on.
        device   : target device.

    Returns:
        One result dict per config in ``exp_cfgs``, in the same order.
    """
    reset_seed(cfg.seed)
    K = len(exp_cfgs)

    for exp in exp_cfgs:
        print(f"\n[EXP] corruption={exp.corruption}  "
              f"ratio={exp.perturbation_ratio}  "
              f"type={exp.perturbation_type}  severity={exp.severity_level}")

    ds = CorruptedDataset(base_ds, exp_cfgs, indices=eval_idx)
    loader = DataLoader(
        ds,
        batch_size         = cfg.batch_size,
        shuffle            = False,
        num_workers        = cfg.num_workers,
        pin_memory         = torch.cuda.is_available(),
        persistent_workers = (cfg.num_workers > 0),
    )

    metric_names = list(cfg.metric_names)
    dtype = amp_dtype(cfg.amp, device)
    features = features if features is not None else {}
    hook_layers = list(hook_layers or [])

    # One accumulator set per combination of the chunk.
    feat_metrics = [{rt: {mn: [] for mn in metric_names} for rt in REGION_TYPES}
                    for _ in range(K)]
    # The same, once per hooked layer.  Empty when nothing is hooked, which is
    # the single-probe behaviour the sweep had before the depth study.
    layer_metrics = [{ly: {rt: {mn: [] for mn in metric_names}
                           for rt in REGION_TYPES}
                      for ly in hook_layers} for _ in range(K)]
    dice_corr = [[] for _ in range(K)]
    iou_corr = [[] for _ in range(K)]
    dice_corr_in = [[] for _ in range(K)]
    dice_corr_out = [[] for _ in range(K)]
    by_overlap = [{"none": [], "partial": [], "full": [], "no_lesion": []}
                  for _ in range(K)]
    eff_ratio_sum = [0.0] * K

    # The clean forward pass is shared by the whole chunk, so are its metrics.
    dice_clean, iou_clean = [], []
    n_total = 0

    model.eval()
    with torch.inference_mode():
        for clean_u8, corr_u8, cmask_u8, target in tqdm(
                loader, desc=f"{cfg.arch} chunk of {K}", leave=False):
            clean_u8 = clean_u8.to(device, non_blocking=True)
            corr_u8  = corr_u8.to(device, non_blocking=True)   # (B, K, 3, H, W)
            cmask_u8 = cmask_u8.to(device, non_blocking=True)  # (B, K, 1, H, W)
            target   = target.to(device, non_blocking=True)

            # return_features gives the pre-head feature map, which is where
            # the drift metrics are defined.
            with torch.autocast(device_type=device.type, dtype=dtype,
                                enabled=dtype is not None):
                logits_c, feat_c, _ = model(normalize_batch(clean_u8),
                                            return_features=True)

            # The hooks overwrite ``features`` on every forward pass, so the
            # clean activations have to be taken out of the dict before the
            # corrupted pass runs.  ``.detach()`` already happened in the hook;
            # this is a reference copy, not a clone, which is why the corrupted
            # tensors are read into their own names below.
            feat_c_layers = {ly: features[ly] for ly in hook_layers
                             if ly in features}

            pred_c = logits_c.argmax(1)
            full = torch.ones_like(pred_c, dtype=torch.bool)
            d_c = regional_dice(pred_c, target, full)
            dice_clean.append(d_c)
            iou_clean.append(foreground_iou(pred_c, target))
            n_total += target.size(0)

            for k in range(K):
                exp = exp_cfgs[k]
                cmask = cmask_u8[:, k].float()

                with torch.autocast(device_type=device.type, dtype=dtype,
                                    enabled=dtype is not None):
                    logits_k, feat_k, _ = model(normalize_batch(corr_u8[:, k]),
                                                return_features=True)

                # -- Feature-level drift, per region --------------------------
                # The region masks are the same for the head input and for
                # every hooked layer: compute_metrics projects them onto
                # whatever feature grid it is handed.
                regions = region_masks(cmask)
                for rt, mk in regions.items():
                    vals, w = compute_metrics(cfg.arch, feat_c, feat_k, mk,
                                              metric_names,
                                              exp.normalize_features,
                                              exp.soft_masking)
                    for mn in metric_names:
                        feat_metrics[k][rt][mn].append((to_float(vals.get(mn)), w))

                # -- The same, at every hooked layer --------------------------
                for ly in hook_layers:
                    if ly not in feat_c_layers or ly not in features:
                        continue
                    fc, fk = feat_c_layers[ly], features[ly]
                    for rt, mk in regions.items():
                        vals, w = compute_metrics(cfg.arch, fc, fk, mk,
                                                  metric_names,
                                                  exp.normalize_features,
                                                  exp.soft_masking)
                        for mn in metric_names:
                            layer_metrics[k][ly][rt][mn].append(
                                (to_float(vals.get(mn)), w))

                # -- Output-level degradation ---------------------------------
                pred_k = logits_k.argmax(1)
                cm2d = cmask.squeeze(1) > 0.5

                d_k = regional_dice(pred_k, target, full)
                dice_corr[k].append(d_k)
                iou_corr[k].append(foreground_iou(pred_k, target))
                dice_corr_in[k].append(regional_dice(pred_k, target, cm2d))
                dice_corr_out[k].append(regional_dice(pred_k, target, ~cm2d))

                # -- Degradation split by lesion / corruption overlap ---------
                for stratum, dc, dk in zip(overlap_stratum(target, cm2d),
                                           d_c.tolist(), d_k.tolist()):
                    if not (np.isnan(dc) or np.isnan(dk)):
                        by_overlap[k][stratum].append(dc - dk)

                eff_ratio_sum[k] += cmask.mean(dim=(1, 2, 3)).sum().item()

    d_clean, n_dice_clean, _ = nanmean_count(dice_clean)
    i_clean, _, _ = nanmean_count(iou_clean)

    results = []
    for k, exp in enumerate(exp_cfgs):
        agg_feat = {rt: {mn: weighted_nanmean(feat_metrics[k][rt][mn])
                         for mn in metric_names}
                    for rt in REGION_TYPES}
        agg_layers = {ly: {rt: {mn: weighted_nanmean(layer_metrics[k][ly][rt][mn])
                                for mn in metric_names}
                           for rt in REGION_TYPES}
                      for ly in hook_layers}

        d_corr, n_corr, n_img = nanmean_count(dice_corr[k])
        _, n_in, _ = nanmean_count(dice_corr_in[k])
        _, n_out, _ = nanmean_count(dice_corr_out[k])
        result = {
            "corruption_type":     exp.corruption,
            "perturbation_ratio":  exp.perturbation_ratio,
            "effective_ratio":     eff_ratio_sum[k] / max(n_total, 1),
            "severity_level":      exp.severity_level,
            "perturbation_family": exp.perturbation_family,
            "perturbation_name":   exp.perturbation_type,
            "dice_clean":          d_clean,
            "dice_corrupted":      d_corr,
            "dice_drop":           d_clean - d_corr,
            "iou_clean":           i_clean,
            "iou_corrupted":       nanmean_chunks(iou_corr[k]),
            "dice_corrupted_in":   nanmean_chunks(dice_corr_in[k]),
            "dice_corrupted_out":  nanmean_chunks(dice_corr_out[k]),
            "dice_drop_by_overlap": {
                key: (float(np.mean(v)) if v else float("nan"))
                for key, v in by_overlap[k].items()
            },
            "n_by_overlap":        {key: len(v) for key, v in by_overlap[k].items()},
            "n_samples":           n_total,
            # How many images each Dice mean is actually over.  They differ
            # from n_samples by the lesion-free frames on which the model also
            # predicted nothing, which are undefined rather than perfect; the
            # "outside" column drops the most, since at ratio 0.9 there is
            # often no uncorrupted region left to score.
            "n_dice_clean":        n_dice_clean,
            "n_dice":              n_corr,
            "n_dice_in":           n_in,
            "n_dice_out":          n_out,
            "n_images":            n_img,
            "feature_metrics":     agg_feat,
            # Absent (empty) when the sweep runs without hooks, so a reader of
            # an older file sees the same JSON it always did.  The deepest
            # probe is NOT in here: it is the head input above.
            "layer_metrics":       agg_layers,
        }

        print(f"  [{exp.corruption} r={exp.perturbation_ratio} "
              f"s={exp.severity_level} {exp.perturbation_type}]  "
              f"dice_clean={d_clean:.4f}  dice_corr={d_corr:.4f}  "
              f"drop={d_clean - d_corr:.4f}  "
              f"(no-overlap drop={result['dice_drop_by_overlap']['none']:.4f})")
        results.append(result)

    return results


# =============================================================================
# Argument parsing
# =============================================================================
def parse_args() -> Box:
    """
    Parse the command line and return it as a :class:`~box.Box`.

    ``--perturbation_types`` is a JSON string mapping a family name to a list
    of perturbation names, exactly as in ``corruption_experiments_cls.py``.
    """
    p = argparse.ArgumentParser(
        description="Robustness evaluation of segmentation models."
    )

    # -- Paths ----------------------------------------------------------------
    p.add_argument("--dataset", type=str, default="kvasir_seg",
                   help="Registry key, see shared/DatasetRegistry.py.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Parent directory; one run subfolder is created here.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a .pth checkpoint written by finetuning_seg.py.")

    # -- Model ----------------------------------------------------------------
    p.add_argument("--arch", type=str, required=True,
                   choices=["resnet34", "vit_b_16", "dinov2"])
    p.add_argument("--num_classes", type=int, default=2,
                   help="2 for binary segmentation (background and lesion).")
    p.add_argument("--input_size", type=int, nargs=2, default=[224, 224],
                   metavar=("H", "W"))

    # -- Corruption sweep -----------------------------------------------------
    p.add_argument("--corruption_types", type=str, nargs="+",
                   default=["circular", "superpixel", "squared"])
    p.add_argument("--perturbation_ratios", type=float, nargs="+",
                   default=[0.3, 0.6, 0.9])
    p.add_argument("--severity_levels", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--perturbation_types", type=str,
                   default='{"NOISE": ["Gaussian Noise"], "BLUR": ["Defocus Blur"],'
                           ' "COLOR": ["Brightness", "Contrast"]}',
                   help="JSON mapping a family name to perturbation names.")
    p.add_argument("--squared_sizes", type=int, nargs="+", default=[32],
                   help="Must divide the image size exactly; 32 tiles 224.")
    p.add_argument("--metric_names", type=str, nargs="+", default=["mse", "cos"])
    p.add_argument("--hook_depth", action=argparse.BooleanOptionalAction,
                   default=False,
                   help=("Also measure the feature drift at the intermediate "
                         "blocks of the backbone, not only at the pre-head "
                         "feature map, so the depth profile can be plotted for "
                         "segmentation as it is for classification. Off by "
                         "default: it multiplies the metric work by the number "
                         "of hooked layers."))
    p.add_argument("--hook_layers", type=str, nargs="*", default=None,
                   help=("Layers to hook when --hook_depth is on. Defaults to "
                         "every backbone block EXCEPT the deepest, which is "
                         "already measured as the pre-head feature map."))
    p.add_argument("--normalize_features", action="store_true", default=False,
                   help="L2-normalize feature maps before computing metrics.")
    p.add_argument("--soft_masking", action=argparse.BooleanOptionalAction,
                   default=True,
                   help=("Weight each feature cell by its pooled mask value in "
                         "[0, 1]. This is the default; --no-soft_masking "
                         "applies a hard 0.5 threshold instead and drops the "
                         "'_softmasking' suffix from the output folder."))

    # -- DataLoader / device --------------------------------------------------
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gpu",         type=int, default=0)
    p.add_argument("--pin_cores",   type=int, default=0,
                   help="Pin the process to the last N CPU cores (0 = off).")
    p.add_argument("--combo_chunk", type=int, default=4,
                   help=("How many combinations share one pass over the data. "
                         "Higher is faster (the clean forward pass and the "
                         "image decoding are done once per chunk) but holds "
                         "that many corrupted copies of each batch in memory. "
                         "1 restores the old one-combination-per-pass "
                         "behaviour."))
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help=("Run the forward passes in bfloat16/float16 on CUDA. "
                         "Metrics are still accumulated in float32. Use "
                         "--no-amp for bit-identical float32 inference."))
    p.add_argument("--seed",        type=int, default=29,
                   help="Base seed for the per-sample corruption masks.")
    p.add_argument("--split_dir",   type=str, default="splits",
                   help="Directory holding the split file written by "
                        "finetuning_seg.py.")
    p.add_argument("--split_seed",  type=int, default=0,
                   help="Which split file to read (must match finetuning_seg.py).")
    p.add_argument("--eval_split",  type=str, default="test",
                   choices=["test", "all"],
                   help=("'test' uses the held-out split of --dataset. 'all' "
                         "uses every image, which is what an EXTERNAL dataset "
                         "needs: the model never saw any of it, so holding "
                         "part of it out would only discard evaluation data."))

    # -- WandB ----------------------------------------------------------------
    p.add_argument("--use_wandb", action="store_true", default=False)
    p.add_argument("--wandb_mode", type=str, default="online",
                   choices=["online", "offline", "disabled"],
                   help="'offline' buffers to disk; sync later with "
                        "`wandb sync <dir>`.")

    args = p.parse_args()
    args.perturbation_types = json.loads(args.perturbation_types)
    return Box(vars(args))


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    """Parse the config, run the corruption sweep, save the results."""
    cfg = parse_args()

    spec = get_spec(cfg.dataset)
    if spec.task != "segmentation":
        raise ValueError(f"Dataset '{cfg.dataset}' is a {spec.task} dataset. "
                         "Use corruption_experiments_cls.py for classification.")

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu)
    print(f"Device: {device}")

    pin_cpu_cores(cfg.pin_cores, torch_threads=2)
    reset_seed(cfg.seed)

    # Dataset and split are built once and reused by every combination.
    base_ds = build_dataset(cfg.dataset, image_size=tuple(cfg.input_size))
    if cfg.eval_split == "test":
        split = load_or_make_split(base_ds,
                                   split_file_dir(cfg.split_dir, cfg.dataset),
                                   cfg.split_seed)
        eval_idx = split["test"]
    else:
        eval_idx = list(range(len(base_ds)))
    print(f"Evaluating on {len(eval_idx)} images ({cfg.eval_split})")

    model = build_segmentation_model(cfg.arch, cfg.num_classes,
                                     cfg.checkpoint, device)

    # Optional depth study: hook the intermediate blocks of the backbone.  The
    # deepest probe is never in this list -- it is the pre-head feature map the
    # model already returns -- so the layers here are strictly the ones the
    # single-probe sweep could not see.  Empty list => the old behaviour.
    if not cfg.hook_depth:
        hook_layers = []
    elif cfg.hook_layers is not None:
        hook_layers = list(cfg.hook_layers)
    else:
        hook_layers = get_default_hook_layers(cfg.arch, "segmentation")
    features, handles = register_hooks(model, hook_layers)
    if hook_layers:
        print(f"Hooked {len(features)} layer(s) for the depth profile: "
              f"{list(features) if features else hook_layers}")

    # The checkpoint folder is named <arch>_<freeze>_lambda<L>_seed<N>
    # (shared/paths.run_name), so it can be parsed back instead of being
    # treated as an opaque string.  The folder name keeps the "_on_<dataset>"
    # part because the same checkpoint is evaluated on several external sets.
    ckpt_dir = run_folder_from_checkpoint(cfg.checkpoint, cfg.arch,
                                          cfg.split_seed)
    info = parse_run_name(ckpt_dir)
    variant = variant_suffix(cfg.normalize_features, cfg.soft_masking)
    run_name = f"{ckpt_dir}_on_{cfg.dataset}_{variant}"

    out_dir = Path(cfg.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    print(f"Output directory: {out_dir}")

    wb.init(stage=wb.CORRUPTION, task="segmentation",
            name=wb.run_label(cfg.dataset, info["arch"], info["freeze"],
                              info["lambda_sn"] or 0, info["seed"], variant),
            config=dict(cfg), mode=cfg.wandb_mode, enabled=cfg.use_wandb)

    exp_cfgs = build_experiment_cfgs(cfg, "segmentation")
    chunks = chunked(exp_cfgs, cfg.combo_chunk)
    dtype = amp_dtype(cfg.amp, device)
    print(f"Total combinations: {len(exp_cfgs)} in {len(chunks)} chunk(s) of "
          f"up to {cfg.combo_chunk}  |  amp={dtype}")

    all_results: list[dict] = []
    with tqdm(total=len(exp_cfgs), desc="Overall progress") as pbar:
        for chunk in chunks:
            results = run_chunk(cfg, chunk, model, base_ds, eval_idx, device,
                                features=features, hook_layers=hook_layers)
            for exp, result in zip(chunk, results):
                all_results.append(result)
                if cfg.use_wandb:
                    _log_to_wandb(result, exp)
            pbar.update(len(chunk))

    for h in handles:
        h.remove()

    # -- Save aggregated metrics ----------------------------------------------
    metrics = {
        "metadata": {
            "task":            "segmentation",
            "model":           run_name,
            "arch":            cfg.arch,
            "checkpoint":      cfg.checkpoint,
            "dataset":         cfg.dataset,
            "eval_split":      cfg.eval_split,
            "num_experiments": len(all_results),
            "timestamp":       datetime.now().isoformat(),
            "output_dir":      str(out_dir),
        },
        "config": {
            "num_classes":         cfg.num_classes,
            "input_size":          cfg.input_size,
            "batch_size":          cfg.batch_size,
            "combo_chunk":         cfg.combo_chunk,
            "amp":                 str(dtype) if dtype is not None else "off",
            "seed":                cfg.seed,
            "split_seed":          cfg.split_seed,
            "corruption_types":    cfg.corruption_types,
            "perturbation_ratios": cfg.perturbation_ratios,
            "severity_levels":     cfg.severity_levels,
            "squared_sizes":       cfg.squared_sizes,
            "metric_names":        cfg.metric_names,
            "hook_depth":          bool(cfg.hook_depth),
            "hook_layers":         list(hook_layers),
            "normalize_features":  cfg.normalize_features,
            "soft_masking":        cfg.soft_masking,
        },
        "summary": {
            "mean_dice_clean":     nanmean_chunks(
                [r["dice_clean"] for r in all_results]),
            "mean_dice_corrupted": nanmean_chunks(
                [r["dice_corrupted"] for r in all_results]),
            "mean_dice_drop":      nanmean_chunks(
                [r["dice_drop"] for r in all_results]),
            "mean_dice_drop_no_overlap": nanmean_chunks(
                [r["dice_drop_by_overlap"]["none"] for r in all_results]),
        },
        "all_experiments": all_results,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics -> {metrics_path}")

    # -- WandB final summary --------------------------------------------------
    if cfg.use_wandb:
        import wandb

        wandb.summary["final/num_experiments"] = len(all_results)
        for k, v in metrics["summary"].items():
            wandb.summary[f"final/{k}"] = v
    wb.finish(enabled=cfg.use_wandb)

    print(f"\nExperiment complete! Results in {out_dir}")


if __name__ == "__main__":
    main()
