"""
drift_analysis/drift_analysis_cls.py
================================
Empirically validates the Lipschitz upper bound on the classification head.

For every image *x* of the evaluation split a perturbed copy *x_tilde* is
created via one of two strategies (selected with ``--perturbation``):

* ``gaussian`` (default) - ``x_tilde = clip(x + eps * N(0, I), 0, 1)``, with eps
  given by ``--eps`` and expressed in ``[0, 1]`` pixel units.
* any ImageNet-C key (``"Contrast"``, ``"Gaussian Noise"``, ``"Brightness"``,
  ``"Defocus Blur"``, ...) at the severity given by ``--severity``.

Both are applied in pixel space by `drift_analysis.perturbations`: the batch is
de-normalised to ``[0, 1]``, converted to uint8 HWC RGB, perturbed and
re-normalised.  That is the domain the ImageNet-C functions are defined on, and
it makes the severity levels here identical to the ones
``corruption_experiments_cls.py`` uses.

The script then measures two things.

1. The plain logit-vector bound:

   * **dz**      = z(x) - z(x_tilde)      - feature drift at the penultimate layer.
   * **dlogits** = f(x) - f(x_tilde)      - logit difference at the head output.
   * **upper**   = ||W||_2 * ||dz||_2     - Lipschitz upper bound.
   * **ratio**   = ||dlogits||_2 / upper  - bound tightness, should be <= 1.

2. The margin-based bound, which turns "the logits moved" into "the prediction
   changed" (see the "From Feature Drift to Task Metrics" section of the paper,
   and the mirror implementation in ``drift_analysis_seg.py``):

   * **Gamma**   = max_{a != b} ||w_a - w_b||_2, the Lipschitz constant of the
     MARGIN, always <= sqrt(2) * ||W||_2, computed once from the head weights.
   * **m(x)**    = top1 - top2 logit gap on the CLEAN prediction.
   * **safe**    = m(x) > Gamma * ||dz(x)||_2 - the certified condition of
     Proposition 1: if true, the prediction on x CANNOT change.
   * **flipped** = argmax(logits(x_tilde)) != argmax(logits(x)) - observed.
   * **certified flip rate**  = fraction of samples where ``safe`` is false
     (the worst-case bound on FlipRate from the paper).
   * **bound violation rate** = fraction of ``safe`` samples that flipped
     anyway.  This is a correctness check, not a statistic: the bound is a
     mathematical guarantee, so it MUST be exactly 0; anything else means the
     head is not being treated as the linear map the theory assumes.
   * **observed / predicted accuracy drop** = the population-level bound
     against the drop actually measured with the labels.

Results are written to ``--output_dir/<run>/`` as:

* ``<arch>_<dataset>_<tag>_results.json`` - per-sample arrays plus the summary.
* ``<arch>_<dataset>_<tag>_plot.png``     - scatter (upper vs. drift) and the
  ratio histogram.

The result file has the same ``metadata`` / ``config`` / ``summary`` skeleton as
``drift_analysis_seg.py`` and as the corruption stage, with task-neutral names
in ``summary``, so ``analysis.io`` reads both tasks with one loader.

Usage (see ``run_drift_analysis_cls.sh`` for the full sweep)::

    python -m drift_analysis.drift_analysis_cls \
        --model_path   /path/to/dinov2_head+last.pth \
        --dataset      kvasir_v2 \
        --output_dir   /path/to/output \
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
    result_header, resolve_architecture,
)
from drift_analysis.perturbations import make_perturbation_fn
from shared.DatasetRegistry import build_dataset, get_spec
from shared.ModelFactory import build_model, get_head, peek_num_classes
from shared.paths import (parse_run_name, run_folder_from_checkpoint,
                          split_file_dir)
from shared.runtime import pin_cpu_cores
from shared.seed import reset_seed
from shared.splits import load_or_make_split
import shared.wandb_logging as wb


# =============================================================================
# Drift measurement
# =============================================================================
def analyze_drift(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_type: str,
    perturb_fn,
    sigma: float,
    gamma: float,
    max_batches: int | None = None,
) -> dict[str, torch.Tensor]:
    """
    Measure per-sample feature drift and compare it to the two bounds.

    A forward hook captures the feature vector *z* at the penultimate layer.
    For each image *x* a perturbed copy *x_tilde* is forwarded and:

    * ``upper``  = ``sigma * ||z(x) - z(x_tilde)||``  - logit-vector bound.
    * ``drift``  = ``||logits(x) - logits(x_tilde)||`` - observed change.
    * ``ratio``  = drift / upper - bound tightness, <= 1.
    * ``margin`` = top1 - top2 logit gap on the CLEAN prediction.
    * ``safe``   = ``margin > gamma * ||dz||`` - Proposition 1's certified
      condition: this prediction cannot flip.
    * ``flipped``       = observed argmax change.
    * ``correct_clean`` = whether the clean prediction matches the label.

    Hook attachment points per architecture:

    +----------+---------------------------------------+
    | resnet   | ``model.fc``          input (B, 512)  |
    +----------+---------------------------------------+
    | vit      | ``model.heads.head``  input (B, 768)  |
    +----------+---------------------------------------+
    | dinov2   | ``model.head``        input (B, 768)  |
    +----------+---------------------------------------+

    Args:
        model       : model in eval mode.
        dataloader  : yields ``(x, label)``.
        device      : target device.
        model_type  : ``"resnet"``, ``"vit"`` or ``"dinov2"``.
        perturb_fn  : callable ``(x) -> x_tilde``.
        sigma       : ``sigma_max(W)`` of the head.
        gamma       : ``Gamma`` of the head.
        max_batches : stop early; None means the whole split.

    Returns:
        Dict of 1-D tensors, one entry per sample.
    """
    head = get_head(model)
    model.eval()

    upper_bounds:  list[torch.Tensor] = []
    actual_drifts: list[torch.Tensor] = []
    ratios:        list[torch.Tensor] = []
    margins:       list[torch.Tensor] = []
    safe_flags:    list[torch.Tensor] = []
    flipped_flags: list[torch.Tensor] = []
    correct_flags: list[torch.Tensor] = []
    feats:         list[torch.Tensor] = []

    # -- Hook: the feature vector fed to the classification head --------------
    def _hook(module, inp, out):
        feat = inp[0]
        # ResNet safety: some implementations still provide (B, C, 1, 1).
        if feat.ndim == 4:
            feat = feat.flatten(1)
        feats.append(feat)

    if model_type == "resnet":
        handle = model.fc.register_forward_hook(_hook)
    elif model_type == "vit":
        handle = model.heads.head.register_forward_hook(_hook)
    elif model_type == "dinov2":
        handle = model.head.register_forward_hook(_hook)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         "Choose 'resnet', 'vit', or 'dinov2'.")

    try:
        # inference_mode rather than no_grad: same guarantee, slightly less
        # bookkeeping. No autocast here on purpose -- the violation count below
        # is a correctness check on an exact inequality, and reduced precision
        # would turn float noise into spurious violations.
        with torch.inference_mode():
            for i, (x, y) in enumerate(tqdm(dataloader, desc="drift",
                                            leave=False)):
                if max_batches is not None and i >= max_batches:
                    break

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                x_tilde = perturb_fn(x)

                # Forward - clean image
                feats.clear()
                model(x)
                z = feats[0]
                logits_clean = head(z)

                # Forward - perturbed image
                feats.clear()
                model(x_tilde)
                z_tilde = feats[0]
                logits_pert = head(z_tilde)

                dz_norm      = torch.norm(z - z_tilde, dim=1)
                dlogits_norm = torch.norm(logits_clean - logits_pert, dim=1)
                upper        = sigma * dz_norm

                upper_bounds.append(upper.cpu())
                actual_drifts.append(dlogits_norm.cpu())
                ratios.append((dlogits_norm / (upper + 1e-12)).cpu())

                # -- Margin-based bound -----------------------------------
                top2 = logits_clean.topk(2, dim=1)
                y_pred = top2.indices[:, 0]
                margin = (top2.values[:, 0] - top2.values[:, 1]).clamp(min=0)
                safe = margin > gamma * dz_norm          # certified: cannot flip
                flipped = logits_pert.argmax(1) != y_pred
                correct = y_pred == y

                margins.append(margin.cpu())
                safe_flags.append(safe.cpu())
                flipped_flags.append(flipped.cpu())
                correct_flags.append(correct.cpu())
    finally:
        handle.remove()

    return {
        "upper":         torch.cat(upper_bounds),
        "drift":         torch.cat(actual_drifts),
        "ratio":         torch.cat(ratios),
        "margin":        torch.cat(margins),
        "safe":          torch.cat(safe_flags),
        "flipped":       torch.cat(flipped_flags),
        "correct_clean": torch.cat(correct_flags),
    }


# =============================================================================
# Plot
# =============================================================================
def _plot(upper, drift, ratio, stem: str, summary: dict) -> None:
    """
    Scatter of the bound against the observed drift, plus the ratio histogram.

    Args:
        upper   : per-sample upper-bound values, as a list.
        drift   : per-sample observed drift, as a list.
        ratio   : per-sample ratio, as a list.
        stem    : output path without the ``_plot.png`` suffix.
        summary : used for the figure title.
    """
    plt = headless_pyplot()
    if plt is None:
        return

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    lim = float(max(max(upper), max(drift))) if upper else 1.0
    ax[0].scatter(upper, drift, s=20, alpha=0.35)
    ax[0].plot([0, lim], [0, lim], "k--", lw=2, label="Identity")
    ax[0].set_xlabel(r"Upper bound $\|W\|_2 \|\Delta z\|_2$")
    ax[0].set_ylabel(r"Actual drift $\|\Delta \mathrm{logits}\|_2$")
    ax[0].set_title("Upper bound vs. actual drift")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    ax[1].hist(ratio, bins=40, alpha=0.75)
    ax[1].set_xlabel(r"Ratio  $\|\Delta \mathrm{logits}\|_2$ / upper bound")
    ax[1].set_ylabel("Count")
    ax[1].set_title("Ratio distribution")
    ax[1].grid(alpha=0.3)

    fig.suptitle(f"{os.path.basename(stem)}   "
                 f"flip={summary['true_flip_rate']:.4f}   "
                 f"certified={summary['certified_flip_rate']:.4f}")
    fig.tight_layout()
    fig.savefig(f"{stem}_plot.png", dpi=180)
    plt.close(fig)
    print(f"Saved -> {stem}_plot.png")


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
    if spec.task != "classification":
        raise ValueError(f"Dataset '{args.dataset}' is a {spec.task} dataset. "
                         "Use drift_analysis_seg.py for segmentation.")

    # -- Dataset --------------------------------------------------------------
    # The indices come from the split file written by finetuning_cls.py, so the
    # evaluated images are the ones the model never saw.
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
    model_type, arch = resolve_architecture(args.model_path, args.arch)
    if args.num_classes is None:
        args.num_classes = peek_num_classes(args.model_path)
        print(f"  num_classes = {args.num_classes} (read from checkpoint)")

    model = build_model(arch=arch, num_classes=args.num_classes,
                        checkpoint=args.model_path, device=device,
                        pretrained=False)
    sigma, gamma = head_constants(model)
    print(f"Loaded model: {arch} ({model_type})")
    print(f"sigma_max = {sigma:.4f}   Gamma = {gamma:.4f}   "
          f"(sqrt(2)*sigma = {np.sqrt(2) * sigma:.4f})")

    # -- Perturbation ---------------------------------------------------------
    perturb_fn = make_perturbation_fn(args.perturbation,
                                      severity=args.severity,
                                      eps=args.eps, seed=args.seed)
    print(f"Perturbation: {args.perturbation}"
          + (f"  (eps={args.eps})" if is_gaussian(args.perturbation)
             else f"  (severity={args.severity})"))

    res = analyze_drift(model, loader, device, model_type, perturb_fn,
                        sigma, gamma)

    upper, drift, ratio = res["upper"], res["drift"], res["ratio"]
    margin, safe = res["margin"], res["safe"]
    flipped, correct = res["flipped"], res["correct_clean"]

    # -- Summary --------------------------------------------------------------
    # "safe" (margin > Gamma*dz) is a CERTIFICATE that the sample cannot flip.
    # A safe sample that flipped anyway is a correctness bug, not noise: this
    # must be exactly 0, unlike the statistical quantities around it.
    n_valid = int(safe.numel())
    violations = int((safe & flipped).sum().item())

    true_flip_rate = float(flipped.float().mean().item())
    certified_flip_rate = float((~safe).float().mean().item())

    clean_acc = float(correct.float().mean().item())
    true_drop = float((correct & flipped).float().mean().item())
    predicted_drop = float((correct & ~safe).float().mean().item())

    # How conservative the certified rate is against the observed one; NaN when
    # nothing flipped, where bound and reality trivially agree at 0.
    tightness = (certified_flip_rate / true_flip_rate
                 if true_flip_rate > 0 else float("nan"))

    # Task-neutral key names: 'true_drop' is an accuracy drop here and a Dice
    # drop in drift_analysis_seg.py, which is what lets one loader in
    # analysis.io put both tasks in the same table.
    summary = {
        "sigma_max":            sigma,
        "gamma":                gamma,
        "mean_upper":           float(upper.mean()),
        "mean_drift":           float(drift.mean()),
        "mean_ratio":           float(ratio.mean()),
        "std_ratio":            float(ratio.std()),
        "min_ratio":            float(ratio.min()),
        "max_ratio":            float(ratio.max()),
        "clean_acc":            clean_acc,
        "true_flip_rate":       true_flip_rate,
        "certified_flip_rate":  certified_flip_rate,
        "true_drop":            true_drop,
        "predicted_drop":       predicted_drop,
        # Must be 0.  Anything else means the bound is broken somewhere in the
        # chain feature -> head -> logit, and no result should be reported
        # until it is understood.
        "bound_violations":     violations,
        "bound_violation_rate": violations / max(n_valid, 1),
        "bound_holds":          violations == 0,
        "mean_tightness":       tightness,
        "n_samples":            n_valid,
    }
    print(json.dumps(summary, indent=2))

    # -- Output location ------------------------------------------------------
    # One subfolder per fine-tuning run, named exactly as the checkpoint folder
    # so that parse_run_name recovers arch / freeze / lambda / seed from the
    # path, the same way the corruption stage does it.
    run_name = run_folder_from_checkpoint(args.model_path, arch, args.split_seed)
    info = parse_run_name(run_name)
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    tag = perturbation_tag(args.perturbation, args.eps, args.severity)
    stem = os.path.join(out_dir, f"{arch}_{args.dataset}_{tag}")

    # -- WandB ----------------------------------------------------------------
    # Logged as summary values rather than a time series: this script produces
    # one measurement per (checkpoint, dataset, perturbation), not a curve.
    wandb_run = wb.init(
        stage=wb.DRIFT, task="classification",
        name=wb.run_label(args.dataset, info["arch"], info["freeze"],
                          info["lambda_sn"] or 0, info["seed"], tag),
        config=vars(args), mode=args.wandb_mode, enabled=args.use_wandb)
    if wandb_run is not None:
        import wandb
        for k, v in summary.items():
            wandb.summary[k] = v
        wandb.log({
            "ratio_hist":  wandb.Histogram(ratio.numpy().tolist()),
            "upper_hist":  wandb.Histogram(upper.numpy().tolist()),
            "drift_hist":  wandb.Histogram(drift.numpy().tolist()),
            "margin_hist": wandb.Histogram(margin.numpy().tolist()),
        })

    # -- Save -----------------------------------------------------------------
    blob = result_header("classification", run_name, arch, args, out_dir, len(ds))
    blob["summary"] = summary
    blob["per_sample"] = {
        "upper":   upper.numpy().tolist(),
        "drift":   drift.numpy().tolist(),
        "ratio":   ratio.numpy().tolist(),
        "margin":  margin.numpy().tolist(),
        "safe":    safe.numpy().astype(int).tolist(),
        "flipped": flipped.numpy().astype(int).tolist(),
    }

    with open(f"{stem}_results.json", "w") as f:
        json.dump(blob, f, indent=2)
    print(f"Saved -> {stem}_results.json")

    _plot(upper.numpy().tolist(), drift.numpy().tolist(),
          ratio.numpy().tolist(), stem, summary)
    wb.finish(enabled=args.use_wandb)


# =============================================================================
# Entry point
# =============================================================================
def _build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""
    p = argparse.ArgumentParser(
        description="Lipschitz bound analysis for classification models."
    )

    # -- Paths ----------------------------------------------------------------
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to a .pth checkpoint written by finetuning_cls.py.")
    p.add_argument("--dataset", type=str, default="kvasir_v2",
                   help="Registry key, see shared/DatasetRegistry.py.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Parent directory; one run subfolder is created here.")

    # -- Model ----------------------------------------------------------------
    p.add_argument("--arch", type=str, default=None,
                   choices=["resnet34", "vit_b_16", "dinov2"],
                   help="Left unset, it is read from the checkpoint metadata, "
                        "falling back to the file name for older ones.")
    p.add_argument("--num_classes", type=int, default=None,
                   help="Left unset, it is read from --model_path (required "
                        "for Hyper-Kvasir / Kvasir-Capsule, whose class count "
                        "is discovered at training time, not fixed).")
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
