"""
analysis/appendix/corruption_audit.py
=====================================
Appendix C: the corruptions do what the paper says they do.

    python -m analysis.appendix.corruption_audit --cache .cache_analysis \
        --out paper_results/appendix_corruption [--dataset kvasir_seg]

Two checks, one table (one row per mask shape and nominal ratio):

    effective ratio   fraction of pixels actually corrupted, read from the
                      corruption sweep (mean and range over all runs)
    max change out    largest absolute change of a pixel OUTSIDE the mask,
                      over every corruption type and severity, measured by
                      re-applying the sweep's own corruption code. Must be 0.

The second check runs on CPU. With --dataset it uses real test images of that
dataset, otherwise synthetic images with enough structure for SLIC.

Outputs (in --out): corruption_audit.tex / .csv
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from box import Box

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis import common as ac
from analysis import io as aio
from analysis.appendix.effective_ratio import _synthetic_images
from corruption_experiments.corruption_common import apply_corruption

SHAPES = ["circular", "squared", "superpixel"]
SHAPE_LABEL = {"circular": "circular", "squared": "patch-based", "superpixel": "superpixel"}
PTYPES = ["Gaussian Noise", "Defocus Blur", "Brightness", "Contrast"]
SEVERITIES = [1, 3, 5]
RATIOS = [0.3, 0.6, 0.9]


def effective_area(root: str, cache: str | None, refresh: bool) -> pd.DataFrame:
    """Effective corrupted fraction per (shape, ratio), over every sweep run."""
    frames = aio.load_all(root, cache_dir=cache, refresh=refresh,
                          points_per_run=0, progress=True)
    df = frames["corruption"]
    df = df[df["shape"].isin(SHAPES)].dropna(subset=["eff_ratio"])
    keys = ["task", "arch", "setting", "eval_dataset", "seed", "shape", "ratio",
            "severity", "ptype"]
    df = df.drop_duplicates(subset=[k for k in keys if k in df.columns])
    df = df.assign(ratio=df.ratio.round(3))
    g = df.groupby(["shape", "ratio"]).eff_ratio
    return pd.DataFrame({"eff_mean": g.mean(), "eff_min": g.min(), "eff_max": g.max()})


def load_images(dataset: str | None, n: int, size: int = 224) -> np.ndarray:
    """Real test images of ``dataset`` (uint8 HWC), or synthetic ones."""
    if not dataset:
        print("  [images] synthetic (pass --dataset for real images)")
        return _synthetic_images(n, size)
    from shared.DatasetRegistry import build_dataset
    from shared.paths import split_file_dir
    from shared.splits import load_or_make_split
    base = build_dataset(dataset, image_size=(size, size))
    test = load_or_make_split(base, split_file_dir("splits", dataset), 0)["test"]
    print(f"  [images] {min(n, len(test))} test images of {dataset}")
    return np.stack([base.raw(int(i))[0] for i in test[:n]])


def confinement(images: np.ndarray, squared_size: int, seed: int) -> pd.Series:
    """Largest absolute change of a pixel outside the mask, per (shape, ratio)."""
    out = {}
    for shape in SHAPES:
        for ratio in RATIOS:
            worst = 0
            for ptype in PTYPES:
                for sev in SEVERITIES:
                    cfg = Box({"corruption": shape, "perturbation_type": ptype,
                               "perturbation_ratio": ratio, "severity_level": sev,
                               "squared_size": squared_size, "seed": seed})
                    for k, img in enumerate(images):
                        corrupted, mask = apply_corruption(img, cfg, seed + k)
                        outside = ~np.asarray(mask, dtype=bool)
                        if outside.ndim == 3:
                            outside = outside.all(-1)
                        diff = np.abs(np.asarray(corrupted, dtype=np.int16)
                                      - img.astype(np.int16))
                        if outside.any():
                            worst = max(worst, int(diff[outside].max()))
            out[(shape, ratio)] = worst
            print(f"  {shape:<11} r={ratio}: max change outside the mask = {worst}")
    idx = pd.MultiIndex.from_tuples(out.keys(), names=["shape", "ratio"])
    return pd.Series(list(out.values()), index=idx, name="max_out")


def main() -> None:
    p = ac.base_parser("Appendix C: corruption audit.")
    p.add_argument("--dataset", default=None,
                   help="dataset whose test images are used for the confinement "
                        "check. Default: synthetic images.")
    p.add_argument("--n_images", type=int, default=20)
    p.add_argument("--squared_size", type=int, default=32,
                   help="patch size of the sweep (its --squared_size). Default 32.")
    p.add_argument("--seed", type=int, default=29,
                   help="base seed of the sweep (its --seed). Default 29.")
    args = p.parse_args()

    area = effective_area(args.results_root, args.cache, args.refresh)
    images = load_images(args.dataset, args.n_images)
    conf = confinement(images, args.squared_size, args.seed)
    df = area.join(conf, how="outer").reset_index()

    ok = (df.max_out.fillna(0) == 0).all()
    print(f"\n  confinement check: {'PASS' if ok else 'FAIL'}")

    os.makedirs(args.out, exist_ok=True)
    rows = [{
        "Mask shape": SHAPE_LABEL[r.shape],
        "$r$": f"{r.ratio:g}",
        "effective ratio": f"{r.eff_mean:.3f}",
        "range": f"[{r.eff_min:.3f}, {r.eff_max:.3f}]",
        "max change outside": f"{int(r.max_out)}",
    } for r in df.itertuples()]
    source = (f"{len(images)} test images of \\texttt{{{args.dataset.replace('_', chr(92) + '_')}}}"
              if args.dataset else f"{len(images)} synthetic images")
    ac.write_table(
        pd.DataFrame(rows), os.path.join(args.out, "corruption_audit.tex"),
        caption=(r"Corruption audit. \emph{effective ratio}: fraction of pixels "
                 r"actually corrupted, mean and range over all the runs of the "
                 r"corruption sweep. \emph{max change outside}: largest absolute "
                 r"change of a pixel outside the mask (8-bit scale), over the four "
                 r"corruption types and the three severities, measured on "
                 + source + r"."),
        label="tab:corruption_audit", column_format="lrrrr", pos="!htbp")
    if not ok:
        raise SystemExit("a corruption modifies pixels outside its mask")


if __name__ == "__main__":
    main()
