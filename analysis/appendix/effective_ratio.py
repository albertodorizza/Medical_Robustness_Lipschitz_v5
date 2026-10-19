"""
analysis/appendix/effective_ratio.py
==============================
Measure the fraction of pixels each corruption configuration actually
corrupts, and compare it with the nominal ``perturbation_ratio``.

Run it after changing anything in ``corruptions/``, and keep the output in the
repository history: it is the evidence that the ratio axis of the results
means what it says.

    python -m analysis.appendix.effective_ratio --size 224

What the numbers used to be, before the area-based radius was introduced in
``circular`` and before ``squared_size`` was made a divisor of the image size:

    circular    r=0.3  -> 0.615      radius set on the linear size, not the area
    circular    r=0.6  -> 0.874
    circular    r=0.9  -> 0.992
    squared50   r=0.3  -> 0.249      224/50 = 4.48, so the grid covers 80% only
    squared50   r=0.9  -> 0.747      unreachable

Averaging over shapes at a fixed nominal ratio therefore mixed very different
amounts of actual corruption.  With the current code every shape lands within
about one superpixel of the requested value.

A note on the synthetic images used here
----------------------------------------
The images must have spatial STRUCTURE.  SLIC collapses uniform noise into a
single segment, so measuring `superpixel` on white noise reports 1.000 at every
ratio -- an artefact of the test, not of the corruption.  Pass --images to run
on real data whenever possible; the synthetic fallback is a smooth pattern with
enough gradient for SLIC to segment.
"""

import argparse
import os
import sys

import numpy as np

# Allow running as a plain script as well as with -m
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from corruptions.GlobalCorruptions import global_corr
from corruptions.LocalCorruptions import circular, squared, superpixel


def _synthetic_images(n: int, size: int) -> np.ndarray:
    """
    Smooth synthetic images with enough structure for SLIC to segment.

    Uniform noise is NOT usable here: SLIC returns a single segment for it, so
    `superpixel` would appear to corrupt the whole image at every ratio.
    """
    rng = np.random.default_rng(0)
    xx, yy = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    out = []
    for _ in range(n):
        fx, fy, ph = rng.uniform(3, 9), rng.uniform(3, 9), rng.uniform(0, 6.3)
        r = np.sin(fx * xx + ph) * 0.5 + 0.5
        g = np.cos(fy * yy + ph) * 0.5 + 0.5
        b = (xx * yy + 0.2 * np.sin(fx * yy)) % 1.0
        img = np.stack([r, g, b], -1) * 255.0
        out.append(img.astype(np.uint8))
    return np.stack(out)


def _load_images(folder: str | None, n: int, size: int) -> np.ndarray:
    """Load `n` real images from `folder`, or fall back to synthetic ones."""
    if not folder:
        print("[images] using synthetic structured images "
              "(pass --images for real data)")
        return _synthetic_images(n, size)

    import cv2
    paths = sorted(
        os.path.join(dp, f)
        for dp, _, fns in os.walk(folder) for f in fns
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )[:n]
    if not paths:
        raise FileNotFoundError(f"No images found under {folder}")

    imgs = []
    for path in paths:
        im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        imgs.append(cv2.resize(im, (size, size)))
    print(f"[images] using {len(imgs)} real images from {folder}")
    return np.stack(imgs)


def _report(label, fn, imgs):
    """Apply `fn` to every image and print the mean corrupted fraction."""
    fracs = []
    for im in imgs:
        try:
            _, mask = fn(im)
        except Exception as exc:
            print(f"  {label:<26s} ERROR: {exc}")
            return
        fracs.append(float(np.asarray(mask, dtype=bool).mean()))
    f = np.asarray(fracs)
    print(f"  {label:<26s} effective = {f.mean():.3f}  (std {f.std():.3f})")


def main():
    p = argparse.ArgumentParser(
        description="Measure the actual corrupted area of each configuration."
    )
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--n", type=int, default=20,
                   help="Images to average over (superpixel is stochastic).")
    p.add_argument("--ratios", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    p.add_argument("--squared_sizes", type=int, nargs="+", default=[32, 16, 50])
    p.add_argument("--ptype", type=str, default="Gaussian Noise")
    p.add_argument("--severity", type=int, default=1)
    p.add_argument("--images", type=str, default=None,
                   help="Folder of real images to measure on.  Strongly "
                        "preferred over the synthetic fallback, especially "
                        "for the superpixel shape.")
    args = p.parse_args()

    S = args.size
    imgs = _load_images(args.images, args.n, S)

    print(f"\nimage {S}x{S}, perturbation '{args.ptype}', severity {args.severity}\n")

    print("global")
    _report("global", lambda im: global_corr(im, [args.ptype], args.severity), imgs)

    print("\ncircular (invert=True, corrupts outside the disk)")
    for r in args.ratios:
        _report(f"circular r={r}",
                lambda im, r=r: circular(im, args.ptype, r, args.severity,
                                         invert=True), imgs)

    print("\ncircular (invert=False, corrupts inside the disk)")
    for r in args.ratios:
        _report(f"circular-in r={r}",
                lambda im, r=r: circular(im, args.ptype, r, args.severity,
                                         invert=False), imgs)

    print("\nsuperpixel")
    for r in args.ratios:
        _report(f"superpixel r={r}",
                lambda im, r=r: superpixel(im, args.ptype, r, args.severity), imgs)

    print("\nsquared")
    for ps in args.squared_sizes:
        nh, nw = S // ps, S // ps
        cover = nh * nw * ps * ps / (S * S)
        flag = "" if S % ps == 0 else "   <-- does not divide the image size"
        print(f"  [size={ps}: grid {nh}x{nw}, max coverage {cover:.3f}]{flag}")
        for r in args.ratios:
            _report(f"squared{ps} r={r}",
                    lambda im, r=r, ps=ps: squared(im, [args.ptype], r,
                                                   args.severity, ps), imgs)

    divisors = [k for k in range(8, 65) if S % k == 0]
    print(f"\nsquared_size values that tile a {S} px image: {divisors}")


if __name__ == "__main__":
    main()
