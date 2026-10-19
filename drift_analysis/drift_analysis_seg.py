"""
drift_analysis/drift_analysis_seg.py
====================================
Empirical validation of the Lipschitz bound for segmentation models, plus the
prediction-level (margin) analysis that turns a bound on logits into a bound on
Dice.

Segmentation counterpart of ``drift_analysis_cls.py``.  Three quantities are
measured per pixel, for every image of the evaluation split:

1. ``||dz||``      - feature drift between clean and perturbed input
2. ``||dlogits||`` - the induced change in the per-pixel logit vector
3. ``rho_Lip``     - the ratio ``||dlogits|| / (sigma_max(W) ||dz||)``, i.e. how
                     tight the Lipschitz bound is in practice

Everything is computed on the **low-resolution feature grid**, before the
bilinear upsampling.  This is exact, not an approximation: a 1x1 convolution is
the same affine map at every location, so it commutes with bilinear
interpolation, and the margins and drifts measured on the grid are the ones
that determine the full-resolution prediction.  It is also far cheaper.

From logits to Dice
-------------------
A bound on the logit change does not by itself bound any metric a clinician
reads.  The margin closes that gap.  Let ``m_i`` be the per-pixel top1-top2
margin on the clean input and

    Gamma = max_{a != b} ||w_a - w_b||_2

the Lipschitz constant of the *margin* (for binary segmentation this is simply
``||w_0 - w_1||``).  A pixel keeps its label whenever ``m_i > Gamma ||dz_i||``,
so the number of pixels that can flip is bounded by

    n_flip <= #{ i : m_i <= Gamma ||dz_i|| }

Two bounds on the Dice score are computed from the at-risk set ``R``:

* loose (absolute change), with ``S = |P| + |G|``::

      |dDice| <= 3n / (S - n),     n = |R|

* exact worst case on the Dice DROP, the one stated in the paper.  With
  ``T = |P & G|``, ``N = |P| + |G|``, ``n_tp = |R & P & G|`` (true positives
  at risk) and ``n_bg = |R minus (P | G)|`` (background at risk)::

      Dice - Dice' <= 2 [n_tp (N - T) + n_bg T] / (N (N - n_tp + n_bg))

  Only those two kinds of flip can lower Dice, and the bound is attained when
  exactly those pixels flip, so no bound built from R, P and G is tighter.

Both are checked per image against what actually happened, and both
violation counts must be zero.

Note ``Gamma <= sqrt(2) sigma_max(W)``, so the margin constant is always at
least as tight as the spectral norm, and unlike it the resulting statement is
about the segmentation metric rather than about the logits.

The result file has the same ``metadata`` / ``config`` / ``summary`` skeleton as
``drift_analysis_cls.py`` and as the corruption stage, with task-neutral names in
``summary``, so ``analysis.io`` reads both tasks with one loader.

Usage (see ``run_drift_analysis_seg.sh`` for the full sweep)::

    python -m drift_analysis.drift_analysis_seg \
        --model_path /path/to/dinov2_head+last.pth \
        --arch dinov2 --dataset kvasir_seg \
        --output_dir /path/to/output \
        --perturbation gaussian --eps 0.01
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Ensure the project root is on sys.path when running the script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift_analysis.drift_common import (
    head_constants, headless_pyplot, is_gaussian, perturbation_tag,
    result_header,
)
from drift_analysis.perturbations import make_perturbation_fn
from shared.DatasetRegistry import build_dataset, get_spec
from shared.metrics import foreground_dice
from shared.paths import (parse_run_name, run_folder_from_checkpoint,
                          split_file_dir)
from shared.runtime import pin_cpu_cores
from shared.SegmentationModels import build_segmentation_model
from shared.seed import reset_seed
from shared.splits import load_or_make_split
import shared.wandb_logging as wb


# =============================================================================
# Dice sensitivity
# =============================================================================
def dice_change_bound(n_flip, pred_area, gt_area):
    """
    Upper bound on ``|Dice(corrupted) - Dice(clean)|`` given ``n_flip`` pixels
    that may change label.

    Args:
        n_flip    : ``(B,)`` number of pixels allowed to flip.
        pred_area : ``(B,)`` ``|P|``, predicted foreground on the clean input.
        gt_area   : ``(B,)`` ``|G|``, ground-truth foreground.

    Returns:
        ``(B,)`` bound, clipped to 1 since Dice lives in [0, 1].
    """
    S = pred_area + gt_area
    denom = (S - n_flip).clamp(min=1.0)
    return (3.0 * n_flip / denom).clamp(max=1.0)


def dice_drop_bound_exact(at_risk, pred, target):
    """
    Exact worst-case bound on ``Dice(clean) - Dice(corrupted)``.

    Args:
        at_risk : ``(B, H, W)`` bool, pixels failing the safe condition.
        pred    : ``(B, H, W)`` clean prediction (class indices).
        target  : ``(B, H, W)`` ground truth (class indices).

    Returns:
        ``((B,) bound, (B,) n_tp, (B,) n_bg)``.  The bound is NaN where
        ``|P| + |G| = 0``, i.e. where Dice itself is undefined.
    """
    p, g = pred == 1, target == 1
    count = lambda m: m.flatten(1).sum(1).float()
    T = count(p & g)
    N = count(p) + count(g)
    n_tp = count(at_risk & p & g)
    n_bg = count(at_risk & ~p & ~g)
    denom = (N * (N - n_tp + n_bg)).clamp(min=1.0)
    bound = 2.0 * (n_tp * (N - T) + n_bg * T) / denom
    bound = torch.where(N > 0, bound, torch.full_like(bound, float("nan")))
    return bound, n_tp, n_bg


# =============================================================================
# Analysis
# =============================================================================
@torch.inference_mode()
def analyze(model, loader, perturb_fn, device, sigma, gamma):
    """
    Collect per-pixel drift, bound tightness and margin statistics.

    No autocast here on purpose: the violation count computed by :func:`run`
    is a correctness check on an exact inequality, and reduced precision would
    turn float noise into spurious violations.

    Args:
        model      : segmentation model in eval mode.
        loader     : yields ``(image, target_mask)``.
        perturb_fn : callable ``(x) -> x_tilde``.
        device     : target device.
        sigma      : ``sigma_max(W)`` of the head.
        gamma      : ``Gamma`` of the head.

    Returns:
        Dict of 1-D numpy arrays: per-pixel entries flattened over the batch,
        per-image entries one value per image.
    """
    acc = {k: [] for k in ("dz_norm", "dlogit_norm", "ratio", "margin",
                           "flipped", "certified_flip")}
    per_image = {k: [] for k in ("dice_clean", "dice_corr", "dice_bound",
                                 "dice_bound_exact", "n_tp_risk", "n_bg_risk",
                                 "n_flip_frac")}

    model.eval()
    for images, target in tqdm(loader, desc="drift", leave=False):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        perturbed = perturb_fn(images)

        _, feat_c, logit_c = model(images, return_features=True)
        _, feat_k, logit_k = model(perturbed, return_features=True)

        Hf, Wf = feat_c.shape[-2:]

        # (B, Hf, Wf) per-pixel norms
        dz = (feat_k - feat_c).norm(dim=1)
        dl = (logit_k - logit_c).norm(dim=1)
        ratio = dl / (sigma * dz + 1e-12)

        # Per-pixel top1-top2 margin on the clean prediction
        top2 = logit_c.topk(2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).clamp(min=0)

        pred_c, pred_k = logit_c.argmax(1), logit_k.argmax(1)
        flipped = pred_c != pred_k
        # Sufficient condition for a flip to be POSSIBLE, not for it to happen:
        # this is the certified side of the bound.
        certified = margin <= gamma * dz

        for name, t in (("dz_norm", dz), ("dlogit_norm", dl), ("ratio", ratio),
                        ("margin", margin), ("flipped", flipped.float()),
                        ("certified_flip", certified.float())):
            acc[name].append(t.reshape(-1).cpu())

        # -- Per-image Dice and its predicted bound ---------------------------
        # The ground truth is brought to the feature grid with nearest
        # interpolation, so every quantity below lives on the same grid.
        tgt_lr = torch.nn.functional.interpolate(
            target[:, None].float(), size=(Hf, Wf), mode="nearest"
        ).squeeze(1).long()

        # Same estimator as everywhere else (shared.metrics), but evaluated on
        # the feature grid rather than at full resolution: this is a
        # LOW-RESOLUTION Dice and is systematically higher than the
        # full-resolution one, because boundary pixels -- where most of the
        # disagreement lives -- are averaged away by the coarse grid.  It is
        # the right quantity here, since the bound is a statement about the
        # pixels of that grid, but it must be labelled as such in the paper and
        # never compared directly against the Dice of the other two stages.
        def _dice(pred):
            return foreground_dice(pred, tgt_lr)

        n_flip = certified.flatten(1).sum(1).float()
        pred_area = (pred_c == 1).flatten(1).sum(1).float()
        gt_area = (tgt_lr == 1).flatten(1).sum(1).float()

        per_image["dice_clean"].append(_dice(pred_c).cpu())
        per_image["dice_corr"].append(_dice(pred_k).cpu())
        per_image["dice_bound"].append(
            dice_change_bound(n_flip, pred_area, gt_area).cpu())
        exact, n_tp, n_bg = dice_drop_bound_exact(certified, pred_c, tgt_lr)
        per_image["dice_bound_exact"].append(exact.cpu())
        per_image["n_tp_risk"].append(n_tp.cpu())
        per_image["n_bg_risk"].append(n_bg.cpu())
        per_image["n_flip_frac"].append((n_flip / (Hf * Wf)).cpu())

    out = {k: torch.cat(v).numpy() for k, v in acc.items()}
    out.update({k: torch.cat(v).numpy() for k, v in per_image.items()})
    return out


# =============================================================================
# Plot
# =============================================================================
def _plot(res, sel, sigma, stem, summary):
    """
    Scatter of observed vs bounded drift, the tightness histogram, and the
    per-image Dice bound against the observed Dice change.

    Args:
        res     : dict returned by :func:`analyze`.
        sel     : indices of the subsampled per-pixel values.
        sigma   : ``sigma_max(W)``, used to rebuild the upper bound.
        stem    : output path without the ``_plot.png`` suffix.
        summary : used for the figure title.
    """
    plt = headless_pyplot()
    if plt is None:
        return

    upper = sigma * res["dz_norm"][sel]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    lim = float(max(upper.max(), res["dlogit_norm"][sel].max()))
    ax[0].scatter(upper, res["dlogit_norm"][sel], s=2, alpha=0.2)
    ax[0].plot([0, lim], [0, lim], "k--", lw=1, label="Identity")
    ax[0].set_xlabel(r"Upper bound $\sigma\|\Delta z\|_2$")
    ax[0].set_ylabel(r"Actual drift $\|\Delta \mathrm{logits}\|_2$")
    ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].hist(res["ratio"][sel], bins=60)
    ax[1].set_xlabel(r"Ratio $\rho_{\mathrm{Lip}}$")
    ax[1].set_ylabel("Count"); ax[1].grid(alpha=.3)

    ok = ~np.isnan(res["dice_clean"]) & ~np.isnan(res["dice_bound_exact"])
    drop = res["dice_clean"][ok] - res["dice_corr"][ok]
    ax[2].scatter(res["dice_bound_exact"][ok], drop, s=8, alpha=.5)
    lim2 = float(max(res["dice_bound_exact"][ok].max(initial=0.0), 1e-3))
    ax[2].plot([0, lim2], [0, lim2], "k--", lw=1, label="Identity")
    ax[2].set_xlabel("Exact bound on the Dice drop")
    ax[2].set_ylabel("Observed Dice drop")
    ax[2].legend(); ax[2].grid(alpha=.3)

    fig.suptitle(f"{os.path.basename(stem)}   "
                 f"pixel flip={summary['true_flip_rate']:.4f}   "
                 f"certified={summary['certified_flip_rate']:.4f}")
    fig.tight_layout()
    fig.savefig(f"{stem}_plot.png", dpi=180)
    plt.close(fig)
    print(f"Saved -> {stem}_plot.png")


def _check_bound(observed, bound, tol=1e-6):
    """Per-image violations and mean tightness (bound / observed) of a bound."""
    valid = ~np.isnan(observed) & ~np.isnan(bound)
    n_valid = int(valid.sum())
    violations = int(((observed[valid] - bound[valid]) > tol).sum())
    meaningful = valid & (observed > 1e-3)
    tightness = (float(np.mean(bound[meaningful] / observed[meaningful]))
                 if meaningful.any() else float("nan"))
    return {"violations": violations, "rate": violations / max(n_valid, 1),
            "tightness": tightness}


# =============================================================================
# Main
# =============================================================================
def run(args: argparse.Namespace, device: torch.device) -> None:
    """
    Load one checkpoint, run the drift analysis, write the results.

    Args:
        args   : parsed command line.
        device : target device.
    """
    spec = get_spec(args.dataset)
    if spec.task != "segmentation":
        raise ValueError(f"Dataset '{args.dataset}' is a {spec.task} dataset. "
                         "Use drift_analysis_cls.py for classification.")

    # -- Dataset --------------------------------------------------------------
    base_ds = build_dataset(args.dataset, image_size=tuple(args.input_size))
    if args.eval_split == "test":
        split = load_or_make_split(base_ds,
                                   split_file_dir(args.split_dir, args.dataset),
                                   args.split_seed)
        ds = Subset(base_ds, split["test"])
    else:
        ds = base_ds
    print(f"Evaluating on {len(ds)} images ({args.eval_split})")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers=(args.num_workers > 0))

    # -- Model ----------------------------------------------------------------
    model = build_segmentation_model(args.arch, args.num_classes,
                                     args.model_path, device)
    sigma, gamma = head_constants(model)
    print(f"Loaded model: {args.arch}")
    print(f"sigma_max = {sigma:.4f}   Gamma = {gamma:.4f}   "
          f"(sqrt(2)*sigma = {np.sqrt(2) * sigma:.4f})")

    # -- Perturbation ---------------------------------------------------------
    perturb_fn = make_perturbation_fn(args.perturbation,
                                      severity=args.severity,
                                      eps=args.eps, seed=args.seed)
    print(f"Perturbation: {args.perturbation}"
          + (f"  (eps={args.eps})" if is_gaussian(args.perturbation)
             else f"  (severity={args.severity})"))

    res = analyze(model, loader, perturb_fn, device, sigma, gamma)

    # Subsample the per-pixel arrays before writing: a full run holds tens of
    # millions of values and the JSON would be unusable.
    rng = np.random.default_rng(args.seed)
    n_keep = min(args.max_saved_pixels, len(res["dz_norm"]))
    sel = rng.choice(len(res["dz_norm"]), n_keep, replace=False)

    # -- Summary --------------------------------------------------------------
    # Each bound is a per-image statement, so it is checked per image.
    # The loose bound is on the absolute change, the exact one on the drop.
    drop = res["dice_clean"] - res["dice_corr"]
    loose = _check_bound(np.abs(drop), res["dice_bound"])
    exact = _check_bound(drop, res["dice_bound_exact"])

    # Task-neutral key names: 'true_drop' is a Dice drop here and an accuracy
    # drop in drift_analysis_cls.py, which is what lets one loader in
    # analysis.io put both tasks in the same table.
    summary = {
        "sigma_max":            sigma,
        "gamma":                gamma,
        "mean_dz_norm":         float(res["dz_norm"].mean()),
        "mean_dlogit_norm":     float(res["dlogit_norm"].mean()),
        "mean_ratio":           float(res["ratio"].mean()),
        "std_ratio":            float(res["ratio"].std()),
        "true_flip_rate":       float(res["flipped"].mean()),
        "certified_flip_rate":  float(res["certified_flip"].mean()),
        "mean_dice_clean":      float(np.nanmean(res["dice_clean"])),
        "mean_dice_corrupted":  float(np.nanmean(res["dice_corr"])),
        "true_drop":            float(np.nanmean(res["dice_clean"]
                                                 - res["dice_corr"])),
        # Loose bound 3n/(S-n) on |dDice|; key names kept for older loaders.
        "predicted_drop":       float(np.nanmean(res["dice_bound"])),
        "bound_violations":     loose["violations"],
        "bound_violation_rate": loose["rate"],
        "bound_holds":          loose["violations"] == 0,
        "mean_tightness":       loose["tightness"],
        # Exact worst-case bound on the Dice drop (the one in the paper).
        "predicted_drop_exact":       float(np.nanmean(res["dice_bound_exact"])),
        "bound_violations_exact":     exact["violations"],
        "bound_violation_rate_exact": exact["rate"],
        "bound_holds_exact":          exact["violations"] == 0,
        "mean_tightness_exact":       exact["tightness"],
        "n_pixels":             int(len(res["dz_norm"])),
        "n_images":             int(len(res["dice_clean"])),
        "n_images_dice":        int((~np.isnan(res["dice_clean"])).sum()),
        "dice_resolution":      "feature_grid",
    }
    print(json.dumps(summary, indent=2))

    # -- Output location ------------------------------------------------------
    # One subfolder per fine-tuning run, named exactly as the checkpoint folder
    # so that parse_run_name recovers arch / freeze / lambda / seed from the
    # path.  The evaluation dataset is in the FILE name instead, because the
    # same checkpoint is evaluated on several external sets.
    run_name = run_folder_from_checkpoint(args.model_path, args.arch,
                                          args.split_seed)
    info = parse_run_name(run_name)
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    tag = perturbation_tag(args.perturbation, args.eps, args.severity)
    stem = os.path.join(out_dir, f"{args.arch}_{args.dataset}_{tag}")

    # -- WandB ----------------------------------------------------------------
    # Logged as summary values rather than a time series: this script produces
    # one measurement per (checkpoint, dataset, perturbation), not a curve.
    wandb_run = wb.init(
        stage=wb.DRIFT, task="segmentation",
        name=wb.run_label(args.dataset, info["arch"], info["freeze"],
                          info["lambda_sn"] or 0, info["seed"], tag),
        config=vars(args), mode=args.wandb_mode, enabled=args.use_wandb)
    if wandb_run is not None:
        import wandb
        for k, v in summary.items():
            wandb.summary[k] = v

    # -- Save -----------------------------------------------------------------
    blob = result_header("segmentation", run_name, args.arch, args, out_dir,
                         len(ds))
    blob["summary"] = summary
    blob["pixels"] = {
        "dz_norm":     res["dz_norm"][sel].tolist(),
        "dlogit_norm": res["dlogit_norm"][sel].tolist(),
        "ratio":       res["ratio"][sel].tolist(),
        "margin":      res["margin"][sel].tolist(),
    }
    blob["per_image"] = {k: res[k].tolist() for k in
                         ("dice_clean", "dice_corr", "dice_bound",
                          "dice_bound_exact", "n_tp_risk", "n_bg_risk",
                          "n_flip_frac")}

    with open(f"{stem}_results.json", "w") as f:
        json.dump(blob, f, indent=2)
    print(f"Saved -> {stem}_results.json")

    _plot(res, sel, sigma, stem, summary)
    wb.finish(enabled=args.use_wandb)


# =============================================================================
# Entry point
# =============================================================================
def _build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""
    p = argparse.ArgumentParser(
        description="Lipschitz bound analysis for segmentation models."
    )

    # -- Paths ----------------------------------------------------------------
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to a .pth checkpoint written by finetuning_seg.py.")
    p.add_argument("--dataset", type=str, default="kvasir_seg",
                   help="Registry key, see shared/DatasetRegistry.py.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Parent directory; one run subfolder is created here.")

    # -- Model ----------------------------------------------------------------
    p.add_argument("--arch", type=str, required=True,
                   choices=["resnet34", "vit_b_16", "dinov2"])
    p.add_argument("--num_classes", type=int, default=2,
                   help="2 for binary segmentation (background and lesion).")
    p.add_argument("--input_size", type=int, nargs=2, default=[224, 224],
                   metavar=("H", "W"))

    # -- Perturbation ---------------------------------------------------------
    p.add_argument("--perturbation", type=str, default="gaussian",
                   help="'gaussian' for the small random probe, or an "
                        "ImageNet-C key ('Contrast', 'Defocus Blur', ...).")
    p.add_argument("--severity", type=int, default=3, choices=[1, 2, 3, 4, 5],
                   help="Severity level, used only for ImageNet-C keys.")
    p.add_argument("--eps", type=float, default=0.01,
                   help="Std of the gaussian probe, in [0, 1] pixel units "
                        "(0.01 = 2.55 grey levels). Used only with 'gaussian'.")

    # -- DataLoader / device --------------------------------------------------
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gpu",         type=int, default=0)
    p.add_argument("--pin_cores",   type=int, default=0,
                   help="Pin the process to the last N CPU cores (0 = off).")
    p.add_argument("--seed",        type=int, default=29,
                   help="Seed of the perturbation, so the same noise "
                        "realisation is used for every architecture.")
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
    p.add_argument("--max_saved_pixels", type=int, default=200000,
                   help="Per-pixel values written to the JSON.")

    # -- WandB ----------------------------------------------------------------
    p.add_argument("--use_wandb", action="store_true", default=False)
    p.add_argument("--wandb_mode", type=str, default="online",
                   choices=["online", "offline", "disabled"],
                   help="'offline' buffers to disk; sync later with "
                        "`wandb sync <dir>`.")
    return p


if __name__ == "__main__":
    cli_args = _build_parser().parse_args()

    dev = torch.device(f"cuda:{cli_args.gpu}"
                       if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(cli_args.gpu)
    print(f"Device: {dev}")

    pin_cpu_cores(cli_args.pin_cores, torch_threads=2)
    reset_seed(cli_args.seed)
    run(cli_args, dev)
