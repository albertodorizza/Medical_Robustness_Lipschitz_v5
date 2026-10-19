"""
corruptions/preview.py
============================
Render a grid of corrupted images so the corruptions can be checked by eye.

``analysis/appendix/effective_ratio.py`` verifies the mask AREA numerically; this tool
verifies everything that a number cannot show: that the mask lands where it
should, that the corruption is visible at the requested severity, that
`invert` flips the region, and that nothing outside the mask was touched.

Run it after any change to ``corruptions/`` and before launching a sweep.

Examples
--------
One page per shape, all four perturbations, on real images::

    python -m corruptions.preview \\
        --images /path/to/kvasir-dataset-v2 \\
        --out OUTPUTS/corruption_preview

Sweep the severity of a single perturbation::

    python -m corruptions.preview --mode severity \\
        --ptype "Gaussian Noise" --shape superpixel

Sweep the ratio, and compare the two circular variants::

    python -m corruptions.preview --mode ratio --shape circular

What to look for
----------------
* the mask overlay lines up with the corrupted pixels;
* the "difference" row is black OUTSIDE the mask -- any signal there means the
  corruption leaked out of its support;
* the printed area matches the requested ratio;
* severity 1 is visible but mild, severity 5 is strong.
"""

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corruptions.GlobalCorruptions import global_corr
from corruptions.LocalCorruptions import circular, squared, superpixel

DEFAULT_PTYPES = ["Gaussian Noise", "Defocus Blur", "Brightness", "Contrast"]


# =============================================================================
# Images
# =============================================================================
def _synthetic_images(n: int, size: int) -> np.ndarray:
    """
    Structured synthetic images, used when no real data is given.

    Uniform noise is deliberately avoided: SLIC collapses it into a single
    segment, so the superpixel shape would look like a full-image corruption.
    """
    rng = np.random.default_rng(0)
    xx, yy = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    out = []
    for _ in range(n):
        fx, fy, ph = rng.uniform(3, 9), rng.uniform(3, 9), rng.uniform(0, 6.3)
        img = np.stack([
            np.sin(fx * xx + ph) * 0.5 + 0.5,
            np.cos(fy * yy + ph) * 0.5 + 0.5,
            (xx * yy + 0.2 * np.sin(fx * yy)) % 1.0,
        ], -1) * 255.0
        out.append(img.astype(np.uint8))
    return np.stack(out)


def load_images(folder, n, size):
    """Load up to `n` images from `folder` (recursively), or synthesise them."""
    if not folder:
        print("[images] synthetic structured images "
              "(pass --images for real data)")
        return _synthetic_images(n, size)

    import cv2
    paths = sorted(
        os.path.join(dp, f)
        for dp, _, fns in os.walk(folder) for f in fns
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )
    if not paths:
        raise FileNotFoundError(f"No images under {folder}")

    # Spread the picks across the folder so several classes are represented.
    step = max(1, len(paths) // n)
    picked = paths[::step][:n]
    imgs = [cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB),
                       (size, size)) for p in picked]
    print(f"[images] {len(imgs)} real images from {folder}")
    return np.stack(imgs)


# =============================================================================
# Corruption dispatch
# =============================================================================
def apply_one(image, shape, ptype, ratio, severity, squared_size=32,
              invert=True, seed=0):
    """Apply one corruption, with a deterministic RNG, and return (img, mask)."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        if shape == "global":
            return global_corr(image, [ptype], severity)
        if shape == "circular":
            return circular(image, ptype, ratio, severity, invert=invert)
        if shape == "superpixel":
            return superpixel(image, ptype, ratio, severity)
        if shape == "squared":
            return squared(image, [ptype], ratio, severity, squared_size)
        raise ValueError(f"Unknown shape '{shape}'")
    finally:
        np.random.set_state(state)


# =============================================================================
# Plotting
# =============================================================================
def _draw_cell(ax, img, mask, title, show_mask_outline=True):
    """Show one corrupted image with its mask outline and a short caption."""
    ax.imshow(img)
    if show_mask_outline and mask is not None and mask.any() and not mask.all():
        # A single contour line is easier to read than a translucent overlay.
        ax.contour(mask.astype(float), levels=[0.5], colors="cyan",
                   linewidths=0.8)
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def page_shapes(imgs, args, plt):
    """One figure per shape: rows = images, columns = perturbation types."""
    figs = {}
    shapes = args.shapes
    for shape in shapes:
        n_rows = len(imgs)
        n_cols = len(args.ptypes) + 1          # +1 for the clean reference
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(1.7 * n_cols, 1.8 * n_rows),
                                 squeeze=False)
        for r, img in enumerate(imgs):
            _draw_cell(axes[r][0], img, None, "clean" if r == 0 else "")
            for c, ptype in enumerate(args.ptypes, start=1):
                out, mask = apply_one(img, shape, ptype, args.ratio,
                                      args.severity, args.squared_size,
                                      args.invert, seed=r)
                frac = float(np.asarray(mask, bool).mean())
                _draw_cell(axes[r][c], out, mask,
                           f"{ptype}\narea={frac:.2f}" if r == 0
                           else f"area={frac:.2f}")
        fig.suptitle(f"shape={shape}   ratio={args.ratio}   "
                     f"severity={args.severity}"
                     + (f"   invert={args.invert}" if shape == "circular" else ""),
                     fontsize=9)
        fig.tight_layout()
        figs[f"shapes_{shape}"] = fig
    return figs


def page_severity(imgs, args, plt):
    """One figure: rows = images, columns = severity 1..5."""
    sev = [1, 2, 3, 4, 5]
    fig, axes = plt.subplots(len(imgs), len(sev) + 1,
                             figsize=(1.7 * (len(sev) + 1), 1.8 * len(imgs)),
                             squeeze=False)
    for r, img in enumerate(imgs):
        _draw_cell(axes[r][0], img, None, "clean" if r == 0 else "")
        for c, s in enumerate(sev, start=1):
            out, mask = apply_one(img, args.shape, args.ptype, args.ratio, s,
                                  args.squared_size, args.invert, seed=r)
            _draw_cell(axes[r][c], out, mask, f"severity {s}" if r == 0 else "")
    fig.suptitle(f"{args.ptype}   shape={args.shape}   ratio={args.ratio}",
                 fontsize=9)
    fig.tight_layout()
    return {"severity_sweep": fig}


def page_ratio(imgs, args, plt):
    """One figure: rows = images, columns = corruption ratio."""
    ratios = args.ratios
    fig, axes = plt.subplots(len(imgs), len(ratios) + 1,
                             figsize=(1.7 * (len(ratios) + 1), 1.8 * len(imgs)),
                             squeeze=False)
    for r, img in enumerate(imgs):
        _draw_cell(axes[r][0], img, None, "clean" if r == 0 else "")
        for c, ratio in enumerate(ratios, start=1):
            try:
                out, mask = apply_one(img, args.shape, args.ptype, ratio,
                                      args.severity, args.squared_size,
                                      args.invert, seed=r)
            except ValueError as exc:
                # Unreachable ratios raise on purpose; show why.
                axes[r][c].text(0.5, 0.5, str(exc)[:60], fontsize=5,
                                ha="center", va="center", wrap=True)
                axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
                continue
            frac = float(np.asarray(mask, bool).mean())
            _draw_cell(axes[r][c], out, mask,
                       f"r={ratio}\narea={frac:.2f}" if r == 0
                       else f"area={frac:.2f}")
    fig.suptitle(f"{args.ptype}   shape={args.shape}   "
                 f"severity={args.severity}", fontsize=9)
    fig.tight_layout()
    return {"ratio_sweep": fig}


def page_leakage(imgs, args, plt):
    """
    Clean / corrupted / mask / |difference| for one setting.

    The difference panel is the important one: it must be black everywhere
    outside the mask.  Any structure there means the corruption escaped its
    spatial support, which would invalidate the IN/OUT decomposition.
    """
    fig, axes = plt.subplots(len(imgs), 4,
                             figsize=(4 * 1.9, 1.9 * len(imgs)), squeeze=False)
    for r, img in enumerate(imgs):
        out, mask = apply_one(img, args.shape, args.ptype, args.ratio,
                              args.severity, args.squared_size, args.invert,
                              seed=r)
        diff = np.abs(out.astype(np.int16) - img.astype(np.int16)).max(-1)
        outside = diff[~np.asarray(mask, bool)]
        leak = float(outside.max()) if outside.size else 0.0

        _draw_cell(axes[r][0], img, None, "clean" if r == 0 else "", False)
        _draw_cell(axes[r][1], out, mask, "corrupted" if r == 0 else "")
        axes[r][2].imshow(np.asarray(mask, float), cmap="gray", vmin=0, vmax=1)
        axes[r][2].set_title("mask" if r == 0 else "", fontsize=7)
        axes[r][2].set_xticks([]); axes[r][2].set_yticks([])
        axes[r][3].imshow(diff, cmap="magma", vmin=0, vmax=max(diff.max(), 1))
        axes[r][3].set_title(f"|diff|  max outside={leak:.0f}"
                             if r == 0 else f"max outside={leak:.0f}",
                             fontsize=7)
        axes[r][3].set_xticks([]); axes[r][3].set_yticks([])

        if leak > 0:
            print(f"  [WARN] image {r}: corruption changed pixels outside the "
                  f"mask (max delta {leak:.0f}/255)")
    fig.suptitle(f"leakage check   {args.ptype}   shape={args.shape}   "
                 f"ratio={args.ratio}   severity={args.severity}", fontsize=9)
    fig.tight_layout()
    return {"leakage": fig}


# =============================================================================
# Main
# =============================================================================
def main():
    p = argparse.ArgumentParser(
        description="Render corrupted images for visual inspection."
    )
    p.add_argument("--mode", default="shapes",
                   choices=["shapes", "severity", "ratio", "leakage", "all"],
                   help="Which page(s) to produce.")
    p.add_argument("--images", type=str, default=None,
                   help="Folder of real images; synthetic ones if omitted.")
    p.add_argument("--n", type=int, default=3, help="Images per page.")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--out", type=str, default="OUTPUTS/corruption_preview")

    p.add_argument("--shapes", nargs="+",
                   default=["circular", "superpixel", "squared", "global"],
                   help="Shapes for --mode shapes.")
    p.add_argument("--shape", type=str, default="superpixel",
                   help="Single shape for the severity/ratio/leakage modes.")
    p.add_argument("--ptypes", nargs="+", default=DEFAULT_PTYPES,
                   help="Perturbations for --mode shapes.")
    p.add_argument("--ptype", type=str, default="Gaussian Noise",
                   help="Single perturbation for the other modes.")
    p.add_argument("--ratio", type=float, default=0.3)
    p.add_argument("--ratios", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    p.add_argument("--severity", type=int, default=3)
    p.add_argument("--squared_size", type=int, default=32)
    p.add_argument("--invert", action="store_true", default=True,
                   help="circular: corrupt outside the disk (default).")
    p.add_argument("--no-invert", dest="invert", action="store_false",
                   help="circular: corrupt inside the disk.")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    imgs = load_images(args.images, args.n, args.size)

    modes = (["shapes", "severity", "ratio", "leakage"]
             if args.mode == "all" else [args.mode])
    builders = {"shapes": page_shapes, "severity": page_severity,
                "ratio": page_ratio, "leakage": page_leakage}

    for mode in modes:
        print(f"[preview] building '{mode}' ...")
        for name, fig in builders[mode](imgs, args, plt).items():
            path = os.path.join(args.out, f"{name}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {path}")

    print(f"\nDone. Open the PNGs in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
