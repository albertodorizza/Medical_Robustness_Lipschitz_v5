"""
analysis/appendix/perturbation_magnitude.py
=====================================
Measure, in INPUT space, how strong each corruption actually is inside its own
mask -- separately for every (shape, perturbation, severity, ratio).

Why
---
A corruption table says "severity 3" as if that named a fixed amount of damage.
It does not: severity indexes a parameter of an ImageNet-C function, and how
much the pixels actually move depends on the image and on how much of it is
being perturbed.  Before reading a shape or ratio effect out of the Dice drop,
it is worth knowing whether the corruption hit equally hard in each case.

This script measures that directly, in input space, restricted to the mask.  If
``delta_rms_in`` is flat across shapes at a fixed (perturbation, severity), the
cross-shape comparison is measuring propagation and can be made as is.  If it
is not, the shape effect has to be reported at matched ``delta_rms_in``.

It is also the check that caught the bounding-box bug: ``corruptions/utility.py``
used to perturb the tight bounding box of the mask rather than the whole image,
which made Contrast and Brightness weaker for ``squared`` than for ``circular``
at the same declared severity, by more than a factor of two.  That is fixed, and
rerunning this audit is how you confirm it stayed fixed.

It runs on CPU, touches no model and no checkpoint, and reuses
``CorruptedDataset`` so the masks and the noise realisations are bit-identical
to the ones the sweep used.

Usage
-----
    python -m analysis.appendix.perturbation_magnitude \
        --dataset kvasir_seg --out paper_assets/perturbation_audit.csv

    # then read the per-shape spread printed at the end
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from box import Box

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from corruption_experiments.corruption_common import (
    CorruptedDataset, build_experiment_cfgs,
)
from shared.DatasetRegistry import build_dataset, get_spec
from shared.paths import split_file_dir
from shared.seed import reset_seed
from shared.splits import load_or_make_split

FIELDS = [
    "dataset", "task", "shape", "ptype", "family", "severity",
    "ratio", "eff_ratio", "n_images",
    # Input-space magnitude, in 0-255 grey levels, restricted to the mask.
    "delta_rms_in", "delta_rms_in_sd", "delta_mae_in", "delta_p95_in",
    # Outside the mask: must be exactly 0, otherwise the corruption escaped its
    # spatial support and the whole IN/OUT decomposition is meaningless.
    "delta_max_out",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="Registry key, see shared/DatasetRegistry.py.")
    p.add_argument("--out", default="perturbation_audit.csv")
    p.add_argument("--n_images", type=int, default=100,
                   help="Images sampled from the evaluation split.  The "
                        "quantities here are input-space statistics and "
                        "stabilise quickly; 100 is plenty for a table.")
    p.add_argument("--input_size", type=int, nargs=2, default=[224, 224])
    p.add_argument("--corruption_types", nargs="+",
                   default=["circular", "superpixel", "squared"])
    p.add_argument("--perturbation_ratios", type=float, nargs="+",
                   default=[0.3, 0.6, 0.9])
    p.add_argument("--severity_levels", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--perturbation_types", type=str,
                   default='{"NOISE": ["Gaussian Noise"], "BLUR": ["Defocus Blur"],'
                           ' "COLOR": ["Brightness", "Contrast"]}')
    p.add_argument("--squared_sizes", type=int, nargs="+", default=[32])
    p.add_argument("--seed", type=int, default=29,
                   help="Must match the sweep's --seed for the masks to be "
                        "the same ones.")
    p.add_argument("--split_dir", default="splits")
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--eval_split", default="test", choices=["test", "all"])
    a = p.parse_args()
    a.perturbation_types = json.loads(a.perturbation_types)
    return a


def main():
    args = parse_args()
    reset_seed(args.seed)

    spec = get_spec(args.dataset)
    base_ds = build_dataset(args.dataset, image_size=tuple(args.input_size))

    if args.eval_split == "test":
        split = load_or_make_split(base_ds,
                                   split_file_dir(args.split_dir, args.dataset),
                                   args.split_seed)
        idx = split["test"]
    else:
        idx = list(range(len(base_ds)))

    # A fixed prefix rather than a random draw: the point is comparability
    # across configurations, and every configuration must see the same images.
    idx = idx[:args.n_images]

    cfg = Box(dict(
        corruption_types=args.corruption_types,
        perturbation_ratios=args.perturbation_ratios,
        severity_levels=args.severity_levels,
        perturbation_types=args.perturbation_types,
        squared_sizes=args.squared_sizes,
        seed=args.seed,
        normalize_features=False,
        soft_masking=True,
    ))
    exp_cfgs = build_experiment_cfgs(cfg, spec.task)
    print(f"{len(exp_cfgs)} configurations x {len(idx)} images")

    rows = []
    for n, exp in enumerate(exp_cfgs, 1):
        # One config at a time: the whole point is per-config statistics, and
        # this keeps memory flat.
        ds = CorruptedDataset(base_ds, [exp], indices=idx)

        rms, mae, p95, eff = [], [], [], []
        max_out = 0.0
        for i in range(len(ds)):
            clean, corr, cmask, _ = ds[i]
            c = clean.numpy().astype(np.float32)          # (3, H, W) 0-255
            k = corr[0].numpy().astype(np.float32)
            m = cmask[0].numpy().astype(bool)             # (1, H, W)

            d = np.abs(k - c)                             # (3, H, W)
            inside = np.broadcast_to(m, d.shape)

            if inside.any():
                di = d[inside]
                rms.append(float(np.sqrt((di ** 2).mean())))
                mae.append(float(di.mean()))
                p95.append(float(np.percentile(di, 95)))
            # Leakage check: a non-zero value here invalidates IN/OUT entirely.
            if (~inside).any():
                max_out = max(max_out, float(d[~inside].max()))
            eff.append(float(m.mean()))

        rows.append({
            "dataset":        args.dataset,
            "task":           spec.task,
            "shape":          exp.corruption,
            "ptype":          exp.perturbation_type,
            "family":         exp.perturbation_family,
            "severity":       exp.severity_level,
            "ratio":          exp.perturbation_ratio,
            "eff_ratio":      float(np.mean(eff)) if eff else float("nan"),
            "n_images":       len(ds),
            "delta_rms_in":   float(np.mean(rms)) if rms else float("nan"),
            "delta_rms_in_sd": float(np.std(rms)) if rms else float("nan"),
            "delta_mae_in":   float(np.mean(mae)) if mae else float("nan"),
            "delta_p95_in":   float(np.mean(p95)) if p95 else float("nan"),
            "delta_max_out":  max_out,
        })
        print(f"  [{n}/{len(exp_cfgs)}] {exp.corruption:11s} "
              f"{exp.perturbation_type:15s} s={exp.severity_level} "
              f"r={exp.perturbation_ratio}  "
              f"rms_in={rows[-1]['delta_rms_in']:7.3f}  "
              f"max_out={max_out:g}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {args.out}")

    leaked = [r for r in rows if r["delta_max_out"] > 0]
    if leaked:
        print(f"\nWARNING: {len(leaked)} configuration(s) changed pixels "
              "OUTSIDE their mask. The IN/OUT decomposition is not valid for "
              "those; investigate before reporting.")
        for r in leaked[:10]:
            print(f"  {r['shape']}/{r['ptype']} s={r['severity']} "
                  f"r={r['ratio']}: max_out={r['delta_max_out']:g}")

    # The headline of the audit: spread across shapes at fixed (ptype, severity).
    print("\nSpread of delta_rms_in across shapes, at fixed perturbation and "
          "severity (a large spread means the cross-shape comparison is "
          "confounded by the ROI convention):")
    keys = sorted({(r["ptype"], r["severity"]) for r in rows})
    for ptype, sev in keys:
        vals = {r["shape"]: r["delta_rms_in"] for r in rows
                if r["ptype"] == ptype and r["severity"] == sev
                and r["ratio"] == args.perturbation_ratios[0]}
        if len(vals) < 2:
            continue
        lo, hi = min(vals.values()), max(vals.values())
        rel = (hi - lo) / hi * 100 if hi > 0 else 0.0
        detail = "  ".join(f"{k}={v:.2f}" for k, v in sorted(vals.items()))
        print(f"  {ptype:15s} s={sev}  spread={rel:5.1f}%   {detail}")


if __name__ == "__main__":
    main()
