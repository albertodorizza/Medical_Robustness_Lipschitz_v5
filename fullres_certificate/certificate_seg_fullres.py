"""
fullres_certificate/certificate_seg_fullres.py
==============================================
Margin certificate for segmentation, stated on the FULL-RESOLUTION prediction
(the one whose Dice is reported in the clean-metric table), next to the
feature-grid version used so far.  Experimental: kept apart from
``drift_analysis/`` and writes to its own output tree.

    python -m fullres_certificate.certificate_seg_fullres \
        --model_path /path/to/dinov2_head+last.pth --arch dinov2 \
        --dataset kvasir_seg --output_dir /path/to/out \
        --perturbation gaussian --eps 0.01


Why the grid certificate does not cover the final mask
------------------------------------------------------
The head is a 1x1 convolution on the feature grid, and the logits are then
upsampled bilinearly to the image size.  The grid certificate says that a CELL
keeps its label when ``rho_i > Gamma ||dz_i||``.  A full-resolution pixel on the
lesion boundary, between a foreground and a background cell, has an
interpolated margin close to zero: it can flip although none of its cells is
at risk.  "Every cell safe" does not imply "every pixel safe".


The full-resolution condition
-----------------------------
Bilinear upsampling (align_corners=False) writes every output pixel ``p`` as
a convex combination of grid cells, ``l(p) = sum_i a_i(p) l_i`` with
``a_i >= 0`` and ``sum_i a_i = 1``.  For any two classes ``a, b`` the logit gap
is linear in the logits, so

    (g_a - g_b)(p) = sum_i a_i (g_a - g_b)_i ,

and the change of each cell's gap under the drift is at most
``||w_a - w_b|| ||dz_i|| <= Gamma ||dz_i||`` (the bias cancels).  The gap at ``p``
therefore changes by at most

    r(p) = sum_i a_i Gamma ||dz_i||  =  bilinear(Gamma ||dz||)(p) ,

i.e. the SAME upsampling applied to the per-cell risk.  If the top1-top2
margin of the upsampled logits exceeds ``r(p)``, every gap of the predicted
class stays positive and the pixel cannot flip.  The at-risk set ``R`` built
from this condition is plugged unchanged into the exact Dice-drop bound of
``drift_analysis_seg.dice_drop_bound_exact``, which holds for any set of
pixels that contains every pixel that can flip.

Both versions are computed in the same pass, on the same images and the same
perturbation, so the grid columns of the output reproduce the numbers already
in the paper and the full-resolution columns are directly comparable.


Diagnostics written next to the bounds
--------------------------------------
* ``violations_pixel_*``   flipped pixels that the condition called safe. The
                           full-resolution one must be zero; it is the check
                           of the derivation above.
* ``flip_unseen_by_grid``  full-resolution pixels that flipped although none of
                           the grid cells they are interpolated from flipped:
                           changes the grid analysis cannot see at all.
* ``risk_boundary_only``   full-resolution at-risk pixels none of whose cells
                           is at risk: the extra looseness that the boundary
                           costs.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reused read-only from the existing pipeline, so the two versions cannot
# differ in anything but the resolution.
from drift_analysis.drift_analysis_seg import _check_bound, dice_drop_bound_exact
from drift_analysis.drift_common import head_constants, perturbation_tag
from shared.metrics import foreground_dice

# Slack on the full-resolution comparison only (see certify_batch).  The grid
# comparison is left exactly as in drift_analysis_seg.py, so its columns
# reproduce the numbers already in the paper.
REL_TOL, ABS_TOL = 1e-5, 1e-6


def _upsample(t: torch.Tensor, size) -> torch.Tensor:
    """The upsampling of ``SegmentationModel.forward``, on a (B, H, W) map."""
    x = t[:, None] if t.is_floating_point() else t[:, None].float()
    return F.interpolate(x, size=size, mode="bilinear",
                         align_corners=False).squeeze(1)


def _margin(logits: torch.Tensor) -> torch.Tensor:
    """Top1-top2 logit gap, (B, K, H, W) -> (B, H, W)."""
    top2 = logits.topk(2, dim=1).values
    return (top2[:, 0] - top2[:, 1]).clamp(min=0)


def certify_batch(feat_c, feat_k, logit_lr_c, logit_lr_k, logit_c, logit_k,
                  target, gamma):
    """
    Grid and full-resolution certificate for one batch.

    Args:
        feat_c, feat_k         : (B, D, Hf, Wf) head input, clean / perturbed.
        logit_lr_c, logit_lr_k : (B, K, Hf, Wf) head output on the grid.
        logit_c, logit_k       : (B, K, H, W) upsampled logits, i.e. the
                                 model output.
        target                 : (B, H, W) full-resolution ground truth.
        gamma                  : margin sensitivity of the head.

    Returns:
        Dict of (B,) tensors, one value per image.
    """
    size = logit_c.shape[-2:]
    Hf, Wf = feat_c.shape[-2:]
    count = lambda m: m.flatten(1).sum(1).float()
    out = {}

    risk_g = gamma * (feat_k - feat_c).norm(dim=1)             # (B, Hf, Wf)

    # -- Feature grid: the existing analysis --------------------------------
    at_risk_g = _margin(logit_lr_c) <= risk_g
    pred_gc, pred_gk = logit_lr_c.argmax(1), logit_lr_k.argmax(1)
    flipped_g = pred_gc != pred_gk
    tgt_lr = F.interpolate(target[:, None].float(), size=(Hf, Wf),
                           mode="nearest").squeeze(1).long()
    out["dice_clean_grid"] = foreground_dice(pred_gc, tgt_lr)
    out["dice_corr_grid"] = foreground_dice(pred_gk, tgt_lr)
    out["bound_grid"], _, _ = dice_drop_bound_exact(at_risk_g, pred_gc, tgt_lr)
    out["at_risk_grid"] = count(at_risk_g) / (Hf * Wf)
    out["flip_grid"] = count(flipped_g) / (Hf * Wf)
    out["viol_pix_grid"] = count(flipped_g & ~at_risk_g)

    # -- Full resolution ------------------------------------------------------
    # The float32 interpolation can put a pixel whose margin equals its risk
    # on the wrong side of the comparison; the relative slack only enlarges
    # the at-risk set, so the bound stays valid.
    risk_f = _upsample(risk_g, size)
    at_risk_f = _margin(logit_c) <= risk_f * (1 + REL_TOL) + ABS_TOL
    pred_fc, pred_fk = logit_c.argmax(1), logit_k.argmax(1)
    flipped_f = pred_fc != pred_fk
    n_pix = float(size[0] * size[1])
    out["dice_clean_full"] = foreground_dice(pred_fc, target)
    out["dice_corr_full"] = foreground_dice(pred_fk, target)
    out["bound_full"], _, _ = dice_drop_bound_exact(at_risk_f, pred_fc, target)
    out["at_risk_full"] = count(at_risk_f) / n_pix
    out["flip_full"] = count(flipped_f) / n_pix
    out["viol_pix_full"] = count(flipped_f & ~at_risk_f)

    # -- What the grid misses -------------------------------------------------
    # A pixel "touches" a cell when that cell has a non-zero interpolation
    # weight at it; the weights are non-negative, so the upsampled indicator
    # is > 0 exactly when at least one touching cell is set.
    touched_risk = _upsample(at_risk_g.float(), size) > 0
    touched_flip = _upsample(flipped_g.float(), size) > 0
    out["risk_boundary_only"] = count(at_risk_f & ~touched_risk) / n_pix
    out["flip_unseen_by_grid"] = count(flipped_f & ~touched_flip) / n_pix
    return out


@torch.inference_mode()
def analyze(model, loader, perturb_fn, device, gamma):
    """Run :func:`certify_batch` over the loader; no autocast, see the module
    docstring of ``drift_analysis_seg`` for why."""
    model.eval()
    per_image = {}
    for images, target in tqdm(loader, desc="fullres", leave=False):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        perturbed = perturb_fn(images)
        logit_c, feat_c, lr_c = model(images, return_features=True)
        logit_k, feat_k, lr_k = model(perturbed, return_features=True)
        res = certify_batch(feat_c, feat_k, lr_c, lr_k, logit_c, logit_k,
                            target, gamma)
        for k, v in res.items():
            per_image.setdefault(k, []).append(v.cpu())
    return {k: torch.cat(v).numpy() for k, v in per_image.items()}


def summarize(res: dict) -> dict:
    """Run-level means and the violation counts, for both resolutions."""
    s = {}
    for r in ("grid", "full"):
        clean, corr, bound = (res[f"dice_clean_{r}"], res[f"dice_corr_{r}"],
                              res[f"bound_{r}"])
        chk = _check_bound(clean - corr, bound)
        s.update({
            f"dice_clean_{r}":     float(np.nanmean(clean)),
            f"dice_corr_{r}":      float(np.nanmean(corr)),
            f"true_drop_{r}":      float(np.nanmean(clean - corr)),
            f"cert_drop_{r}":      float(np.nanmean(bound)),
            f"at_risk_{r}":        float(np.mean(res[f"at_risk_{r}"])),
            f"flip_{r}":           float(np.mean(res[f"flip_{r}"])),
            f"violations_dice_{r}":  chk["violations"],
            f"violations_pixel_{r}": int(res[f"viol_pix_{r}"].sum()),
        })
    s["risk_boundary_only"] = float(np.mean(res["risk_boundary_only"]))
    s["flip_unseen_by_grid"] = float(np.mean(res["flip_unseen_by_grid"]))
    s["n_images"] = int(len(res["dice_clean_full"]))
    return s


def run(args, device):
    # Heavy imports here, so certify_batch can be tested without torchvision.
    from drift_analysis.perturbations import make_perturbation_fn
    from shared.DatasetRegistry import build_dataset, get_spec
    from shared.paths import (parse_run_name, run_folder_from_checkpoint,
                              split_file_dir)
    from shared.SegmentationModels import build_segmentation_model
    from shared.splits import load_or_make_split

    if get_spec(args.dataset).task != "segmentation":
        raise ValueError(f"'{args.dataset}' is not a segmentation dataset")

    base_ds = build_dataset(args.dataset, image_size=tuple(args.input_size))
    if args.eval_split == "test":
        split = load_or_make_split(base_ds,
                                   split_file_dir(args.split_dir, args.dataset),
                                   args.split_seed)
        ds = Subset(base_ds, split["test"])
    else:
        ds = base_ds
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available())

    model = build_segmentation_model(args.arch, args.num_classes,
                                     args.model_path, device)
    sigma, gamma = head_constants(model)
    perturb_fn = make_perturbation_fn(args.perturbation, severity=args.severity,
                                      eps=args.eps, seed=args.seed)
    print(f"{args.arch} on {args.dataset} ({len(ds)} images)  "
          f"sigma={sigma:.4f}  Gamma={gamma:.4f}  [{args.perturbation}]")

    res = analyze(model, loader, perturb_fn, device, gamma)
    summary = {"sigma_max": sigma, "gamma": gamma, **summarize(res)}
    print(json.dumps(summary, indent=2))

    run_name = run_folder_from_checkpoint(args.model_path, args.arch,
                                          args.split_seed)
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    tag = perturbation_tag(args.perturbation, args.eps, args.severity)
    path = os.path.join(out_dir, f"{args.arch}_{args.dataset}_{tag}_fullres.json")
    blob = {"run_name": run_name, "run": parse_run_name(run_name),
            "arch": args.arch, "train_dataset": args.train_dataset or args.dataset,
            "eval_dataset": args.dataset, "perturbation": args.perturbation,
            "tag": tag, "config": vars(args), "summary": summary,
            "per_image": {k: v.tolist() for k, v in res.items()}}
    with open(path, "w") as f:
        json.dump(blob, f)
    print(f"Saved -> {path}")


def _build_parser():
    p = argparse.ArgumentParser(description="Full-resolution margin certificate "
                                            "for segmentation (experimental).")
    p.add_argument("--model_path", required=True)
    p.add_argument("--arch", required=True,
                   choices=["resnet34", "vit_b_16", "dinov2"])
    p.add_argument("--dataset", default="kvasir_seg")
    p.add_argument("--train_dataset", default=None,
                   help="Recorded in the output only. Defaults to --dataset.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--input_size", type=int, nargs=2, default=[224, 224])
    p.add_argument("--perturbation", default="gaussian")
    p.add_argument("--severity", type=int, default=3)
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=29,
                   help="Seed of the perturbation, same default as "
                        "drift_analysis_seg.py.")
    p.add_argument("--split_dir", default="splits")
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--eval_split", default="test", choices=["test", "all"])
    return p


if __name__ == "__main__":
    a = _build_parser().parse_args()
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(a.gpu)
    from shared.seed import reset_seed
    reset_seed(a.seed)
    run(a, dev)
