"""
tests/test_segmentation_models.py
==================================
Standalone smoke test for shared/SegmentationModels.py.

What it checks, for each of the three architectures:

1. the forward pass runs and produces the expected shapes;
2. the feature grid returned by ``return_features=True`` has the expected
   (D, Hf, Wf) — this is the check that catches "a prefix token was not
   stripped" or "the wrong hook point was used";
3. the head really is the linear map the Lipschitz analysis assumes:
   logits_lr == head.weight @ feats + head.bias, verified numerically;
4. compute_spectral_norm runs on the 1x1 conv head and returns a positive,
   finite number;
5. a forward pass on a clean and a lightly corrupted image gives a small,
   non-zero feature drift (sanity check that something is actually flowing
   through the model, not e.g. two identical outputs from a bug).

Uses one real Kvasir-SEG image when --root is given, otherwise falls back to
random tensors -- shapes and the linear-head identity do not need real data,
only the drift sanity check and the visual output are more meaningful with a
real image.

If --root is given, also saves one PNG per architecture under --out showing
the input image, the ground-truth mask (when available), the prediction, and
an overlay.

Two modes:

* no --checkpoint  : the backbone is only ImageNet/DINOv2-pretrained and the
                     1x1 head has random weights, so the prediction is NOT
                     expected to trace the lesion -- the check is structural
                     (a spatially coherent blob vs. noise or a constant map).
* --checkpoint DIR : loads a fine-tuned checkpoint per architecture (see
                     --checkpoint below for the expected layout) and adds a
                     "fine-tuned" row to the preview, so the vanilla and the
                     trained prediction sit side by side.

Run:
    python -m tests.test_segmentation_models
    python -m tests.test_segmentation_models --root /data/.../Kvasir-SEG \
        --out OUTPUTS/seg_model_preview

    # compare vanilla ImageNet weights against a fine-tuned checkpoint
    python -m tests.test_segmentation_models --root /data/.../Kvasir-SEG \
        --checkpoint_dir /data/.../finetuning/segmentation/kvasir_seg/RUN \
        --freeze_mode head+last
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.ModelFactory import compute_spectral_norm, get_head
from shared.SegmentationModels import (DINOv2Features, ResNetFeatures,
                                       SegmentationModel, ViTFeatures,
                                       build_segmentation_model)

EXPECTED_GRID = {
    "resnet34": (7, 7, 512),
    "vit_b_16": (14, 14, 768),
    "dinov2":   (16, 16, 768),
}


def load_one_image(root: str | None, size=(224, 224)):
    """
    Return ``(x, img_u8, mask_u8)`` for one Kvasir-SEG sample.

    ``x`` is the ImageNet-normalised ``(1, 3, H, W)`` tensor fed to the model.
    ``img_u8`` is the same image as ``(H, W, 3)`` uint8, for plotting.
    ``mask_u8`` is the ground-truth mask as ``(H, W)`` in ``{0, 1}``, or
    ``None`` when ``root`` is not given or no matching mask is found.

    Reads the first image of Kvasir-SEG when ``root`` is given; otherwise
    returns random noise and no mask.  Shapes and the linear-head identity
    check do not need real data; only the drift check and the visual output
    are more meaningful with a real image.
    """
    if root:
        from shared.DatasetRegistry import normalize_image, read_mask, read_rgb
        img_dir = os.path.join(root, "images")
        msk_dir = os.path.join(root, "masks")
        files = sorted(f for f in os.listdir(img_dir)
                       if f.lower().endswith((".jpg", ".png")))
        if not files:
            raise FileNotFoundError(f"No images found in {img_dir}")
        stem = os.path.splitext(files[0])[0]
        path = os.path.join(img_dir, files[0])
        print(f"  using real image: {path}")

        img_u8 = read_rgb(path, size)
        x = normalize_image(img_u8).unsqueeze(0)

        mask_u8 = None
        if os.path.isdir(msk_dir):
            matches = [f for f in os.listdir(msk_dir)
                      if os.path.splitext(f)[0] == stem]
            if matches:
                mask_u8 = read_mask(os.path.join(msk_dir, matches[0]), size)
        return x, img_u8, mask_u8

    print("  --root not given, using random noise instead of a real image")
    return torch.randn(1, 3, *size), None, None


def build_backbone(arch: str):
    if arch == "resnet34":
        return ResNetFeatures(pretrained=True)
    if arch == "vit_b_16":
        return ViTFeatures(pretrained=True)
    if arch == "dinov2":
        return DINOv2Features(pretrained=True)
    raise ValueError(arch)


# Short CLI names (as used by finetuning_seg.py checkpoints) -> registry keys.
SHORT_NAME = {"resnet34": "resnet", "vit_b_16": "vit", "dinov2": "dinov2"}


def find_checkpoint(checkpoint_dir: str, arch: str,
                    freeze_mode: str) -> str | None:
    """
    Locate the checkpoint for one architecture inside a run directory.

    Expects the layout written by ``finetuning_seg.py``:
    ``<checkpoint_dir>/<short_name>_<freeze_mode>.pth``, e.g.
    ``resnet_head+last.pth``.  Returns ``None`` if the file does not exist,
    so the caller can fall back to the vanilla-only comparison for that one
    architecture instead of failing the whole run.
    """
    short = SHORT_NAME[arch]
    path = os.path.join(checkpoint_dir, f"{short}_{freeze_mode}.pth")
    return path if os.path.exists(path) else None


def build_model(arch: str, num_classes: int, checkpoint: str | None = None):
    """
    Build a SegmentationModel, either vanilla-pretrained or fine-tuned.

    With ``checkpoint=None`` this is equivalent to
    ``SegmentationModel(build_backbone(arch), num_classes)`` with a random
    head.  With a checkpoint, it delegates to
    :func:`shared.SegmentationModels.build_segmentation_model`, which builds
    the same architecture and then loads the saved weights -- so the vanilla
    and fine-tuned models are guaranteed to be structurally identical, and
    only the weights differ.
    """
    if checkpoint is None:
        return SegmentationModel(build_backbone(arch), num_classes)
    return build_segmentation_model(arch, num_classes, checkpoint=checkpoint)


def _pred_row(ax_row, img_u8, mask_u8, pred, row_title, has_gt: bool):
    """Fill one row of the preview grid: input | (GT) | prediction | overlay."""
    ax_row[0].imshow(img_u8)
    ax_row[0].set_ylabel(row_title, fontsize=10)

    col = 1
    if has_gt:
        ax_row[col].imshow(mask_u8, cmap="gray", vmin=0, vmax=1)
        col += 1

    ax_row[col].imshow(pred, cmap="gray", vmin=0, vmax=1)
    col += 1

    ax_row[col].imshow(img_u8)
    ax_row[col].contour(pred, levels=[0.5], colors="lime", linewidths=1.5)
    if has_gt:
        ax_row[col].contour(mask_u8, levels=[0.5], colors="red", linewidths=1.5)

    for a in ax_row:
        a.set_xticks([]); a.set_yticks([])


def save_prediction_preview(arch: str, img_u8, mask_u8, logits, out_dir: str,
                            logits_ft=None):
    """
    Save a PNG with the input image, ground truth, prediction, and overlay.

    Args:
        arch      : architecture name, used in the title and file name.
        img_u8    : ``(H, W, 3)`` uint8 RGB, or ``None`` (random-noise input:
                    nothing meaningful to show, the function does nothing).
        mask_u8   : ``(H, W)`` ground-truth mask in ``{0, 1}``, or ``None``.
        logits    : ``(1, K, H, W)`` full-resolution logits, vanilla model.
        out_dir   : directory the PNG is written to.
        logits_ft : same shape, from a fine-tuned checkpoint.  When given, a
                    second row is added below the vanilla one, so the two
                    predictions sit in the same figure at the same scale.

    Without a checkpoint, the "prediction" row is not expected to trace the
    lesion: the head has random weights, so the check there is structural (a
    spatially coherent blob vs. noise or a constant map).  With a checkpoint,
    the second row shows what the same architecture learned to do with the
    same input, and the improvement (or lack of it) is directly visible.
    """
    if img_u8 is None:
        print("  (no real image given, skipping the visual preview)")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_gt = mask_u8 is not None
    n_cols = 4 if has_gt else 3
    n_rows = 2 if logits_ft is not None else 1

    fig, ax = plt.subplots(n_rows, n_cols,
                           figsize=(3.2 * n_cols, 3.4 * n_rows),
                           squeeze=False)

    pred = logits[0].argmax(0).cpu().numpy()
    _pred_row(ax[0], img_u8, mask_u8, pred,
             "vanilla\n(ImageNet-pretrained)", has_gt)

    if logits_ft is not None:
        pred_ft = logits_ft[0].argmax(0).cpu().numpy()
        _pred_row(ax[1], img_u8, mask_u8, pred_ft, "fine-tuned", has_gt)

    # Column headers on the top row only
    headers = ["input"] + (["ground truth"] if has_gt else []) + \
              ["prediction", "overlay (green=pred, red=GT)" if has_gt
               else "overlay (green=pred)"]
    for a, title in zip(ax[0], headers):
        a.set_title(title, fontsize=10)

    fig.suptitle(f"{arch} -- segmentation prediction "
                 + ("comparison" if logits_ft is not None
                    else "sanity check"))
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{arch}_prediction_preview.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved visual preview -> {path}")


def check_one(arch: str, x: torch.Tensor, num_classes: int = 2,
             img_u8=None, mask_u8=None, out_dir: str | None = None,
             checkpoint: str | None = None) -> bool:
    """
    Run every check for one architecture; return True if all pass.

    Structural checks (shape, linear-head identity, spectral norm) always run
    on the VANILLA model, since they test the architecture, not the weights --
    identical whether the head is random or fine-tuned.

    When ``out_dir`` is given (and a real image was loaded), a visual preview
    is saved.  If ``checkpoint`` also resolves to an existing file, the preview
    additionally shows the fine-tuned prediction next to the vanilla one, via
    :func:`save_prediction_preview`.
    """
    print(f"\n=== {arch} ===")
    ok = True

    model = build_model(arch, num_classes)          # vanilla, random head
    model.eval()

    with torch.no_grad():
        logits, feats, logits_lr = model(x, return_features=True)

    logits_ft = None
    if checkpoint is not None:
        print(f"  loading fine-tuned checkpoint: {checkpoint}")
        model_ft = build_model(arch, num_classes, checkpoint=checkpoint)
        model_ft.eval()
        with torch.no_grad():
            logits_ft = model_ft(x)

    if out_dir is not None:
        save_prediction_preview(arch, img_u8, mask_u8, logits, out_dir,
                                logits_ft=logits_ft)

    # -- 1. output shape at full resolution -----------------------------------
    expected_full = (x.shape[0], num_classes, x.shape[2], x.shape[3])
    print(f"  logits (full res)   : {tuple(logits.shape)}  "
          f"(expected {expected_full})")
    if tuple(logits.shape) != expected_full:
        print("  FAIL: unexpected full-resolution shape")
        ok = False

    # -- 2. feature grid shape --------------------------------------------------
    Hf_exp, Wf_exp, D_exp = EXPECTED_GRID[arch]
    print(f"  feats (pre-head)    : {tuple(feats.shape)}  "
          f"(expected (1, {D_exp}, {Hf_exp}, {Wf_exp}))")
    if tuple(feats.shape) != (x.shape[0], D_exp, Hf_exp, Wf_exp):
        print("  FAIL: unexpected feature grid shape "
              "-- a prefix token was probably not stripped")
        ok = False

    print(f"  logits_lr           : {tuple(logits_lr.shape)}  "
          f"(expected (1, {num_classes}, {Hf_exp}, {Wf_exp}))")
    if tuple(logits_lr.shape) != (x.shape[0], num_classes, Hf_exp, Wf_exp):
        print("  FAIL: logits_lr should match the feature grid resolution")
        ok = False

    # -- 3. the head really is the linear map the theory assumes ---------------
    head = get_head(model)
    W = head.weight.detach()                       # (K, D, 1, 1)
    b = head.bias.detach()
    recon = torch.einsum("bdhw,kdxy->bkhw", feats, W) + b.view(1, -1, 1, 1)
    err = (recon - logits_lr).abs().max().item()
    print(f"  linear-head check   : max|W*z+b - logits_lr| = {err:.2e}")
    if err > 1e-4:
        print("  FAIL: the head is not exactly linear -- "
              "the Lipschitz bound would not apply")
        ok = False

    # -- 4. spectral norm ---------------------------------------------------
    sigma = compute_spectral_norm(model).item()
    gamma = torch.cdist(W.flatten(1), W.flatten(1), p=2).max().item()
    print(f"  sigma_max(W)        : {sigma:.4f}")
    print(f"  Gamma (margin const): {gamma:.4f}  "
          f"(sqrt(2)*sigma = {1.4142 * sigma:.4f}, Gamma should be <=)")
    if not (sigma > 0 and torch.isfinite(torch.tensor(sigma))):
        print("  FAIL: sigma_max is not a positive finite number")
        ok = False
    if gamma > 1.4143 * sigma:
        print("  FAIL: Gamma exceeds its theoretical upper bound sqrt(2)*sigma")
        ok = False

    # -- 5. feature drift under a small perturbation ----------------------------
    with torch.no_grad():
        x_pert = (x + 0.01 * torch.randn_like(x)).clamp(-3, 3)
        _, feats_pert, _ = model(x_pert, return_features=True)
    dz = (feats_pert - feats).norm(dim=1).mean().item()
    print(f"  mean ||dz|| under small noise: {dz:.4f}  "
          "(should be > 0 and not huge)")
    if not (0 < dz < 50):
        print("  FAIL: drift is zero, NaN, or implausibly large")
        ok = False

    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None,
                   help="Path to the Kvasir-SEG folder (contains images/, masks/).")
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--out", type=str, default="OUTPUTS/seg_model_preview",
                   help="Where to save the per-architecture PNG previews.")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help=("Directory containing fine-tuned checkpoints, in the "
                         "layout written by finetuning_seg.py: "
                         "'<short_name>_<freeze_mode>.pth', e.g. "
                         "'resnet_head+last.pth'. When given, the preview "
                         "shows the fine-tuned prediction next to the "
                         "vanilla one, for whichever architectures have a "
                         "matching file."))
    p.add_argument("--freeze_mode", type=str, default="head+last",
                   choices=["head", "head+last"],
                   help="Which checkpoint variant to load from --checkpoint_dir.")
    args = p.parse_args()

    x, img_u8, mask_u8 = load_one_image(args.root)

    results = {}
    for arch in ("resnet34", "vit_b_16", "dinov2"):
        checkpoint = None
        if args.checkpoint_dir is not None:
            checkpoint = find_checkpoint(args.checkpoint_dir, arch,
                                         args.freeze_mode)
            if checkpoint is None:
                print(f"\n=== {arch} ===\n  [WARN] no checkpoint found for "
                      f"'{arch}' ({args.freeze_mode}) under "
                      f"{args.checkpoint_dir}; showing vanilla only.")
        try:
            results[arch] = check_one(arch, x, args.num_classes,
                                      img_u8=img_u8, mask_u8=mask_u8,
                                      out_dir=args.out, checkpoint=checkpoint)
        except Exception as exc:
            print(f"\n=== {arch} ===\n  EXCEPTION: {type(exc).__name__}: {exc}")
            results[arch] = False

    print("\n" + "=" * 40)
    for arch, ok in results.items():
        print(f"  {arch:10s} {'PASS' if ok else 'FAIL'}")
    print("=" * 40)

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
