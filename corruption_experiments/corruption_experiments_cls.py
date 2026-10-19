"""
corruption_experiments/corruption_experiments_cls.py
================================================
Robustness evaluation of fine-tuned ResNet-34, ViT-B/16 and DINOv2 under
localized and global corruptions.

For every combination of (corruption shape x ratio x severity x perturbation
type) the script:

1. Applies the corruption on-the-fly, through
   `corruption_experiments.corruption_common`, so masks and noise are
   generated exactly as in ``corruption_experiments_seg.py``.
2. Runs a forward pass on the clean image and one on the corrupted image.
3. Computes MSE and cosine similarity between clean and corrupted features at
   every hooked layer, in three spatial regions: corrupted, non-corrupted,
   full image.
4. Records top-1 accuracy on clean and corrupted inputs.
5. Saves everything to ``metrics.json``.

Supported architectures : ``resnet34``, ``vit_b_16``, ``dinov2``
Supported corruptions   : ``circular``, ``superpixel``, ``squared``, ``global``
Supported metrics       : ``mse``, ``cos``

Chunked sweep
-------------
Combinations are processed ``--combo_chunk`` at a time.  Every image of a chunk
is read, decoded and forwarded CLEAN exactly once, and only the corrupted
forward pass is repeated per combination: with K combinations per chunk the
sweep does K+1 forward passes per batch instead of 2K, and reads each image
once instead of K times.  The clean accuracy of a chunk is therefore measured
once and shared by its combinations, which is also what it always was
numerically, just recomputed K times before.

Masking strategies
------------------
The image-space corruption mask is projected onto the feature grid by area
averaging, so a feature cell receives the fraction of its footprint that is
corrupted.  By default (``--soft_masking``) that fraction is kept as a
continuous weight; ``--no-soft_masking`` thresholds it at 0.5 instead and drops
the ``_softmasking`` suffix from the output folder.  In both cases an empty
mask yields NaN, never 0.0, so a batch with no selected location is skipped by
the weighted average instead of pulling it towards zero.

Datasets are referred to by registry key (see ``shared/DatasetRegistry.py``),
never by path, and ``--num_classes`` is read from the checkpoint, which is what
Hyper-Kvasir and Kvasir-Capsule need since their class count is discovered at
training time.

Reproducibility
---------------
Each corruption mask, and the noise realisation inside it, is drawn from an RNG
seeded by ``(--seed, sample index, corruption config)``.  The same image
therefore receives the same corruption for every architecture, every run,
any number of DataLoader workers and any ``--combo_chunk``, which makes the
cross-architecture comparison paired.

The evaluation split is read from the file written by ``finetuning_cls.py`` (see
`shared.splits`), not re-derived from a seed.

Usage (see ``run_corruption_experiments_cls.sh`` for the full sweep)::

    python -m corruption_experiments.corruption_experiments_cls \
        --arch       dinov2 \
        --checkpoint /path/to/dinov2_head+last.pth \
        --dataset    kvasir_v2 \
        --output_dir /path/to/output
"""

import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from box import Box
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
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
from shared.ModelFactory import build_model, peek_num_classes
from shared.paths import (parse_run_name, run_folder_from_checkpoint,
                          split_file_dir)
from shared.runtime import pin_cpu_cores
from shared.seed import reset_seed
from shared.splits import load_or_make_split
import shared.wandb_logging as wb

REGION_TYPES = ["corrupted", "non_corrupted", "global"]


# =============================================================================
# WandB logging helper
# =============================================================================
def _log_to_wandb(result: dict, experiment_cfg: Box) -> None:
    """
    Log the scalar metrics of a single experiment.

    Per-layer feature metrics are aggregated separately and logged once at the
    end of the sweep in :func:`main`.

    Args:
        result         : one result dict returned by :func:`run_chunk`.
        experiment_cfg : config of the experiment that produced it.
    """
    import wandb

    log = {
        "accuracy_clean":      result["accuracy_clean"],
        "accuracy_corrupted":  result["accuracy_corrupted"],
        "accuracy_drop":       result["accuracy_drop"],
        "effective_ratio":     result["effective_ratio"],
        "corruption":          experiment_cfg.corruption,
        "perturbation_family": experiment_cfg.perturbation_family,
        "perturbation_type":   experiment_cfg.perturbation_type,
        "ratio":               experiment_cfg.perturbation_ratio,
        "severity":            experiment_cfg.severity_level,
    }
    if experiment_cfg.corruption == "squared":
        log["squared_size"] = experiment_cfg.squared_size
    wandb.log(log)


# =============================================================================
# Chunk runner
# =============================================================================
def run_chunk(
    cfg: Box,
    exp_cfgs: list[Box],
    model: nn.Module,
    base_ds,
    eval_idx: list[int],
    device: torch.device,
) -> list[dict]:
    """
    Run a group of combinations in a single pass over the evaluation split.

    ``base_ds`` and ``eval_idx`` arrive ready from :func:`main`: the dataset
    and the split are the same for every combination of the sweep, and
    rebuilding them on each iteration meant re-walking the folder tree dozens
    of times per run.

    Args:
        cfg      : sweep config, read for the loader and device settings.
        exp_cfgs : the combinations of this chunk, in sweep order.
        model    : model in eval mode, shared across the whole sweep.
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
              f"family={exp.perturbation_family}  "
              f"type={exp.perturbation_type}  "
              f"severity={exp.severity_level}  "
              f"normalize_features={exp.normalize_features}  "
              f"soft_masking={exp.soft_masking}")

    hook_layers = cfg.hook_layers or get_default_hook_layers(cfg.arch)
    features, handles = register_hooks(model, hook_layers)

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

    # Per-batch (value, weight) pairs, one structure per combination:
    # combo -> layer -> region -> metric -> list
    all_metrics = [
        {layer: {rt: {mn: [] for mn in metric_names} for rt in REGION_TYPES}
         for layer in hook_layers}
        for _ in range(K)
    ]

    # The clean forward pass is shared by the whole chunk, so is its accuracy.
    correct_clean = 0
    total = 0
    correct_corr = [0] * K
    eff_ratio_sum = [0.0] * K

    model.eval()
    with torch.inference_mode():
        for clean_u8, corr_u8, cmask_u8, labels in tqdm(
            loader, desc=f"{cfg.arch} chunk of {K}", leave=False
        ):
            clean_u8 = clean_u8.to(device, non_blocking=True)
            corr_u8  = corr_u8.to(device, non_blocking=True)   # (B, K, 3, H, W)
            cmask_u8 = cmask_u8.to(device, non_blocking=True)  # (B, K, 1, H, W)
            labels   = labels.to(device, non_blocking=True)

            # Forward pass - clean image, once for the whole chunk
            features.clear()
            with torch.autocast(device_type=device.type, dtype=dtype,
                                enabled=dtype is not None):
                out_clean = model(normalize_batch(clean_u8))
            feats_clean = {k: v.clone() for k, v in features.items()}

            correct_clean += (out_clean.argmax(1) == labels).sum().item()
            total         += labels.size(0)

            for k in range(K):
                exp = exp_cfgs[k]
                cmask = cmask_u8[:, k].float()

                # Forward pass - corrupted image
                features.clear()
                with torch.autocast(device_type=device.type, dtype=dtype,
                                    enabled=dtype is not None):
                    out_corr = model(normalize_batch(corr_u8[:, k]))
                feats_corr = features

                correct_corr[k] += (out_corr.argmax(1) == labels).sum().item()

                # Fraction of pixels actually corrupted, reported alongside the
                # nominal ratio because the two do not always coincide.
                eff_ratio_sum[k] += cmask.mean(dim=(1, 2, 3)).sum().item()

                # Feature metrics at every hooked layer, in the three regions
                regions = region_masks(cmask)
                for layer in hook_layers:
                    if layer not in feats_clean or layer not in feats_corr:
                        continue
                    fc, fr = feats_clean[layer], feats_corr[layer]

                    for rt, mk in regions.items():
                        vals, w = compute_metrics(
                            cfg.arch, fc, fr, mk,
                            metric_names, exp.normalize_features,
                            exp.soft_masking)
                        for mn in metric_names:
                            all_metrics[k][layer][rt][mn].append(
                                (to_float(vals.get(mn)), w))

    for h in handles:
        h.remove()

    acc_clean = correct_clean / max(total, 1)

    results = []
    for k, exp in enumerate(exp_cfgs):
        # Aggregate the per-batch (value, weight) pairs into per-experiment means.
        agg = {
            layer: {
                rt: {mn: weighted_nanmean(pairs) for mn, pairs in rt_dict.items()}
                for rt, rt_dict in layer_dict.items()
            }
            for layer, layer_dict in all_metrics[k].items()
        }

        acc_corr = correct_corr[k] / max(total, 1)
        print(f"  [{exp.corruption} r={exp.perturbation_ratio} "
              f"s={exp.severity_level} {exp.perturbation_type}]  "
              f"acc_clean={acc_clean:.4f}  acc_corr={acc_corr:.4f}  "
              f"drop={acc_clean - acc_corr:.4f}")

        results.append({
            "corruption_type":     exp.corruption,
            "perturbation_ratio":  exp.perturbation_ratio,
            "effective_ratio":     eff_ratio_sum[k] / max(total, 1),
            "severity_level":      exp.severity_level,
            "perturbation_family": exp.perturbation_family,
            "perturbation_name":   exp.perturbation_type,
            "accuracy_clean":      acc_clean,
            "accuracy_corrupted":  acc_corr,
            "accuracy_drop":       acc_clean - acc_corr,
            "n_samples":           total,
            "layer_metrics":       agg,
        })

    return results


# =============================================================================
# Aggregation across experiments
# =============================================================================
def aggregate_results(all_results: list[dict], cfg: Box) -> dict:
    """
    Average the per-layer metrics over every experiment of the sweep.

    ``numpy.nanmean`` is used so NaN values (empty mask regions) are ignored
    rather than propagated.

    Args:
        all_results : result dicts from :func:`run_chunk`.
        cfg         : sweep config, read for ``metric_names``.

    Returns:
        ``{layer: {region: {metric: mean}}}``.
    """
    if not all_results:
        return {}

    layer_names = list(all_results[0].get("layer_metrics", {}).keys())

    aggregated: dict = {}
    for layer in layer_names:
        aggregated[layer] = {}
        for rt in REGION_TYPES:
            aggregated[layer][rt] = {}
            for mn in cfg.metric_names:
                values = [
                    r["layer_metrics"][layer][rt][mn]
                    for r in all_results
                    if mn in r.get("layer_metrics", {}).get(layer, {}).get(rt, {})
                ]
                aggregated[layer][rt][mn] = (float(np.nanmean(values))
                                             if values else float("nan"))
    return aggregated


# =============================================================================
# Argument parsing
# =============================================================================
def parse_args() -> Box:
    """
    Parse the command line and return it as a :class:`~box.Box`.

    ``--perturbation_types`` is a JSON string mapping a family name to a list
    of perturbation names, e.g.
    ``'{"NOISE": ["Gaussian Noise"], "BLUR": ["Defocus Blur"]}'``.
    """
    p = argparse.ArgumentParser(
        description="Robustness evaluation of classification models."
    )

    # -- Paths ----------------------------------------------------------------
    p.add_argument("--dataset", type=str, default="kvasir_v2",
                   help="Registry key, see shared/DatasetRegistry.py.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Parent directory; one run subfolder is created here.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a .pth checkpoint written by finetuning_cls.py.")

    # -- Model ----------------------------------------------------------------
    p.add_argument("--arch", type=str, default="resnet34",
                   choices=["resnet34", "vit_b_16", "dinov2"])
    p.add_argument("--num_classes", type=int, default=None,
                   help="Left unset, it is read from --checkpoint (required "
                        "for Hyper-Kvasir / Kvasir-Capsule, whose class count "
                        "is discovered at training time, not fixed).")
    p.add_argument("--input_size", type=int, nargs=2, default=[224, 224],
                   metavar=("H", "W"))
    p.add_argument("--hook_layers", type=str, nargs="*", default=None,
                   help="Layers to hook. Defaults to all main layers of the "
                        "architecture.")

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
                   help=("Patch side(s) in pixels for the 'squared' shape. "
                         "Must divide the image size exactly, otherwise the "
                         "patch grid cannot cover the image and the requested "
                         "ratio is unreachable: at 224 px, 50 leaves a 4x4 "
                         "grid covering only 80%% of the image, while 32 "
                         "gives 7x7 and full coverage."))
    p.add_argument("--metric_names", type=str, nargs="+", default=["mse", "cos"])
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
                   help="Directory holding the split file written by finetuning_cls.py.")
    p.add_argument("--split_seed",  type=int, default=0,
                   help="Which split file to read (must match finetuning_cls.py).")
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
    if spec.task != "classification":
        raise ValueError(f"Dataset '{cfg.dataset}' is a {spec.task} dataset. "
                         "Use corruption_experiments_seg.py for segmentation.")

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

    # Build the model once and reuse it across all experiments.
    if cfg.num_classes is None:
        if not cfg.checkpoint:
            raise ValueError("--num_classes is required when no --checkpoint "
                             "is given (nothing to read it from).")
        cfg.num_classes = peek_num_classes(cfg.checkpoint)
        print(f"  num_classes = {cfg.num_classes} (read from checkpoint)")
    model = build_model(cfg.arch, cfg.num_classes, cfg.checkpoint, device)

    # The checkpoint folder is named <arch>_<freeze>_lambda<L>_seed<N>
    # (shared/paths.run_name), so it can be parsed back instead of being
    # treated as an opaque string: the WandB run name then follows the same
    # convention as finetuning and drift.
    ckpt_dir = run_folder_from_checkpoint(cfg.checkpoint, cfg.arch,
                                          cfg.split_seed)
    info = parse_run_name(ckpt_dir)
    variant = variant_suffix(cfg.normalize_features, cfg.soft_masking)
    run_name = f"{ckpt_dir}_{variant}"

    output_dir = Path(cfg.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    print(f"Output directory: {output_dir}")

    wb.init(stage=wb.CORRUPTION, task="classification",
            name=wb.run_label(cfg.dataset, info["arch"], info["freeze"],
                              info["lambda_sn"] or 0, info["seed"], variant),
            config=dict(cfg), mode=cfg.wandb_mode, enabled=cfg.use_wandb)

    exp_cfgs = build_experiment_cfgs(cfg, "classification")
    chunks = chunked(exp_cfgs, cfg.combo_chunk)
    dtype = amp_dtype(cfg.amp, device)
    print(f"Total combinations: {len(exp_cfgs)} in {len(chunks)} chunk(s) of "
          f"up to {cfg.combo_chunk}  |  amp={dtype}")

    all_results: list[dict] = []
    with tqdm(total=len(exp_cfgs), desc="Overall progress") as pbar:
        for chunk in chunks:
            results = run_chunk(cfg, chunk, model, base_ds, eval_idx, device)
            for exp, result in zip(chunk, results):
                all_results.append(result)
                if cfg.use_wandb:
                    _log_to_wandb(result, exp)
            pbar.update(len(chunk))

    # -- Save aggregated metrics ----------------------------------------------
    aggregated = aggregate_results(all_results, cfg)

    final_metrics = {
        "metadata": {
            "task":            "classification",
            "model":           run_name,
            "arch":            cfg.arch,
            "checkpoint":      cfg.checkpoint,
            "dataset":         cfg.dataset,
            "eval_split":      cfg.eval_split,
            "num_experiments": len(all_results),
            "timestamp":       datetime.now().isoformat(),
            "output_dir":      str(output_dir),
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
            "normalize_features":  cfg.normalize_features,
            "soft_masking":        cfg.soft_masking,
        },
        "summary": {
            "aggregated_layer_metrics": aggregated,
            "mean_accuracy_clean":      nanmean_chunks(
                [r["accuracy_clean"] for r in all_results]),
            "mean_accuracy_corrupted":  nanmean_chunks(
                [r["accuracy_corrupted"] for r in all_results]),
            "mean_accuracy_drop":       nanmean_chunks(
                [r["accuracy_drop"] for r in all_results]),
        },
        "all_experiments": all_results,
    }

    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nSaved metrics -> {metrics_path}")

    # -- WandB final summary --------------------------------------------------
    if cfg.use_wandb:
        import wandb

        summary = {"final/num_experiments": len(all_results)}
        for k, v in final_metrics["summary"].items():
            if k != "aggregated_layer_metrics":
                summary[f"final/{k}"] = v
        for layer, layer_dict in aggregated.items():
            safe = layer.replace(".", "_")
            for rt, mn_dict in layer_dict.items():
                for mn, val in mn_dict.items():
                    summary[f"final/layer_{safe}/{rt}/{mn}"] = val
        for k, v in summary.items():
            wandb.summary[k] = v
    wb.finish(enabled=cfg.use_wandb)

    print(f"\nExperiment complete! Results in {output_dir}")


if __name__ == "__main__":
    main()
