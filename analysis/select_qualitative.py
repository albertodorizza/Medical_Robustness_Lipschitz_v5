"""
analysis/select_qualitative.py
==============================
Find good test images for the qualitative figure (analysis/qualitative_maps.py).

    python -m analysis.select_qualitative --out paper_results/qualitative_select

A good image, with the criteria fixed before looking at any feature map:
    1. polyp class (Kvasir-v2 label "polyps");
    2. no position box in the bottom-left corner (the green ScopeGuide inset);
    3. clear: well exposed, few saturated pixels; ranked by sharpness.

The polyp itself has no mask in Kvasir-v2. When the image also appears,
unchanged, in Kvasir-SEG (exact thumbnail match), its lesion mask is used to
report how much of the lesion the corruption covers. The final choice among
the best candidates is made by eye on the contact sheet, which shows the
corruption mask of the figure (same shape, ratio and seed), before any
feature map is computed.

Only the test split of --split_seed is searched, so the models of the figure
never saw the chosen image. Pass the chosen positions to qualitative_maps.py
with --indices.

Outputs (in --out): candidates.csv (every polyp test image and its scores),
candidates.pdf/.png (contact sheet of the best ones, with mask and lesion contours).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import common as ac
from analysis.qualitative_maps import corrupt
from shared.DatasetRegistry import build_dataset
from shared.paths import split_file_dir
from shared.splits import load_or_make_split

SIZE = 224


# ----------------------------------------------------------------------------
# Image scores
# ----------------------------------------------------------------------------
def green_box_fraction(img: np.ndarray, corner: float = 0.35) -> float:
    """Share of strongly green pixels in the bottom-left corner (the inset)."""
    h, w = img.shape[:2]
    patch = img[int(h * (1 - corner)):, :int(w * corner)].astype(np.int16)
    r, g, b = patch[..., 0], patch[..., 1], patch[..., 2]
    green = (g > 90) & (g > r + 40) & (g > b + 40)
    return float(green.mean())


def clarity(img: np.ndarray) -> dict:
    """Exposure, contrast and sharpness of the central part of the image.

    The sharpness is the variance of the Laplacian divided by the squared
    contrast, so that a noisy image (high Laplacian variance from noise alone)
    does not rank above a clean, detailed one."""
    gray = img.astype(np.float32).mean(-1) / 255.0
    h, w = gray.shape
    centre = gray[h // 6: -h // 6, w // 6: -w // 6]
    lap = (-4 * centre[1:-1, 1:-1] + centre[:-2, 1:-1] + centre[2:, 1:-1]
           + centre[1:-1, :-2] + centre[1:-1, 2:])
    contrast = float(centre.std())
    return {"brightness": float(centre.mean()),
            "contrast": contrast,
            "saturated": float((img.max(-1) >= 250).mean()),
            "sharpness": float(lap.var()) / max(contrast ** 2, 1e-8)}


def thumbnail(img: np.ndarray, side: int = 32) -> np.ndarray:
    """Grey thumbnail in [0, 1], used to match Kvasir-v2 to Kvasir-SEG."""
    gray = img.astype(np.float32).mean(-1) / 255.0
    k = gray.shape[0] // side
    return gray[: k * side, : k * side].reshape(side, k, side, k).mean((1, 3)).ravel()


# ----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Rank test images for the qualitative figure.")
    p.add_argument("--out", default="paper_results/qualitative_select")
    p.add_argument("--dataset", default="kvasir_v2")
    p.add_argument("--polyp_class", default="polyps")
    p.add_argument("--masks_from", default="kvasir_seg",
                   help="segmentation dataset holding the lesion masks.")
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--split_dir", default="splits")
    p.add_argument("--shape", default="superpixel")
    p.add_argument("--ratio", type=float, default=0.3)
    p.add_argument("--severity", type=int, default=3)
    p.add_argument("--corruption_seed", type=int, default=29)
    p.add_argument("--max_green", type=float, default=0.01,
                   help="max share of green pixels in the bottom-left corner.")
    p.add_argument("--brightness", type=float, nargs=2, default=[0.30, 0.75])
    p.add_argument("--min_contrast", type=float, default=0.12,
                   help="min standard deviation of the grey levels in the centre "
                        "(low values are washed-out or hazy images).")
    p.add_argument("--max_saturated", type=float, default=0.05,
                   help="max share of saturated pixels (specular reflections).")
    p.add_argument("--match_tol", type=float, default=0.01,
                   help="max thumbnail difference for an exact Kvasir-SEG copy.")
    p.add_argument("--top", type=int, default=24, help="images in the contact sheet.")
    args = p.parse_args()

    base = build_dataset(args.dataset, image_size=(SIZE, SIZE))
    test = load_or_make_split(base, split_file_dir(args.split_dir, args.dataset),
                              args.split_seed)["test"]
    polyp = base.class_to_idx[args.polyp_class]
    positions = [k for k, i in enumerate(test) if int(base.labels[i]) == polyp]
    print(f"  {len(positions)} polyp images in the test split of {args.dataset}")

    # Lesion masks: thumbnails of every Kvasir-SEG image.
    seg = build_dataset(args.masks_from, image_size=(SIZE, SIZE))
    seg_thumbs = np.stack([thumbnail(seg.raw(j)[0]) for j in range(len(seg))])
    print(f"  {len(seg)} {args.masks_from} images indexed for matching")

    rows = []
    for k in positions:
        index = int(test[k])
        img = base.raw(index)[0]
        dist = np.abs(seg_thumbs - thumbnail(img)).mean(1)
        j = int(dist.argmin())
        rows.append({"position": k, "index": index, "seg_index": j,
                     "match_dist": float(dist[j]), "green": green_box_fraction(img),
                     **clarity(img)})
    df = pd.DataFrame(rows)
    df["matched"] = df.match_dist <= args.match_tol

    # Fixed filters, reported one by one.
    no_box = df.green <= args.max_green
    exposed = df.brightness.between(*args.brightness)
    not_saturated = df.saturated <= args.max_saturated
    contrasted = df.contrast >= args.min_contrast
    df["selected"] = no_box & exposed & not_saturated & contrasted
    print(f"  without the corner box : {int(no_box.sum())}")
    print(f"  + well exposed         : {int((no_box & exposed).sum())}")
    print(f"  + few saturated pixels : {int((no_box & exposed & not_saturated).sum())}")
    print(f"  + enough contrast      : {int(df.selected.sum())}")
    print(f"  exact Kvasir-SEG copies: {int(df.matched.sum())} (lesion mask available)")

    # Ranking: normalised sharpness, so that clean and detailed images come first.
    df = df.sort_values(["selected", "sharpness"], ascending=[False, False])
    best = df[df.selected].head(args.top).copy()

    # Corruption mask of the figure, and lesion cover when the mask exists.
    panels = {}
    for r in best.itertuples():
        img = base.raw(r.index)[0]
        _, _, cmask = corrupt(base, r.index, "Gaussian Noise", args.shape,
                              args.ratio, args.severity, args.corruption_seed)
        corrupted = cmask[0].numpy().astype(bool)
        lesion = seg.raw(r.seg_index)[1].astype(bool) if r.matched else None
        if lesion is not None:
            df.loc[r.Index, "area"] = float(lesion.mean())
            df.loc[r.Index, "cover"] = float((lesion & corrupted).sum() / max(lesion.sum(), 1))
        panels[r.position] = (img, corrupted, lesion)

    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "candidates.csv"), index=False)
    if best.empty:
        print("  nothing passed: loosen --brightness or --max_saturated")
        return
    print("  candidates by normalised sharpness:", " ".join(str(int(x)) for x in best.position))

    # Contact sheet: white = corruption mask, green = lesion (when known).
    plt = ac.setup_style()
    cols = min(6, len(best))
    rows_n = (len(best) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(1.9 * cols, 2.0 * rows_n), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, best.itertuples()):
        img, corrupted, lesion = panels[r.position]
        ax.imshow(img)
        ax.contour(corrupted, levels=[0.5], colors="white", linewidths=0.4)
        if lesion is not None:
            ax.contour(lesion, levels=[0.5], colors="lime", linewidths=0.8)
        ax.set_title(f"pos {r.position}", fontsize=7)
    ac.save_fig(fig, os.path.join(args.out, "candidates.pdf"))

if __name__ == "__main__":
    main()
