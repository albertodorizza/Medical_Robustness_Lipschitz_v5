"""
fullres_certificate/figure.py
=============================
Segmentation row of the certificate figure (script 6) drawn from the
full-resolution runs instead of the feature-grid ones.

    python -m fullres_certificate.figure \
        --root /home/a.dorizza/OUTPUTS_MEDICAL/fullres_certificate/segmentation \
        --out paper_results/fullres_certificate --perturbation gaussian \
        --row_label segmentation

The drawing is ``figure()`` of analysis/6_certificate_needs_sigma.py, called
unchanged, so the row stacks with the classification row of that script.
``--resolution grid`` draws the same runs on the feature grid: it must give
back the segmentation row of the paper, which checks that the two sources are
read the same way.

Outputs (in --out): certificate_fullres_segmentation_<pert>[_grid].pdf/.png,
the same stem with _summary.csv (mean and seed spread per backbone and
setting) and a VERIFICATION block with the violation counts.
"""

import argparse
import glob
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import common as ac
from analysis import io as aio

cert = importlib.import_module("analysis.6_certificate_needs_sigma")


def load(root: str, resolution: str, perturbation: str) -> pd.DataFrame:
    """One row per run, with the column names script 6 works with."""
    rows = []
    for path in glob.glob(os.path.join(root, "**", "*_fullres.json"), recursive=True):
        with open(path) as f:
            b = json.load(f)
        if b["perturbation"] != perturbation:
            continue
        s, r = b["summary"], resolution
        rows.append({
            "task": "segmentation",
            "arch": b["run"]["arch"],
            "setting": aio.setting_label(b["run"]),
            "seed": b["run"]["seed"],
            "train_dataset": b["train_dataset"],
            "eval_dataset": b["eval_dataset"],
            "perturbation": perturbation,
            "sigma_max": s["sigma_max"],
            "gamma": s["gamma"],
            "metric_clean": s[f"dice_clean_{r}"],
            "true_drop": s[f"true_drop_{r}"],
            "cert_drop": s[f"cert_drop_{r}"],
            "certified_flip_rate": s[f"at_risk_{r}"],
            "flip_rate": s[f"flip_{r}"],
            "violations_dice": s[f"violations_dice_{r}"],
            "violations_pixel": s[f"violations_pixel_{r}"],
        })
    if not rows:
        sys.exit(f"no {perturbation} *_fullres.json under {root}")
    df = pd.DataFrame(rows)
    df = df[df.train_dataset == df.eval_dataset].copy()  # in distribution, as script 6
    df["arch"] = pd.Categorical(df.arch, aio.ARCH_ORDER, ordered=True)
    df["setting"] = pd.Categorical(df.setting, aio.SETTING_ORDER, ordered=True)
    df["gamma_frac"] = df.gamma / (np.sqrt(2) * df.sigma_max)
    df["cert_rel"] = df.cert_drop / df.metric_clean
    df["excess"] = df.cert_drop - df.true_drop
    df["unit"] = "pixel"
    return df


def verify(df: pd.DataFrame, resolution: str) -> None:
    ck = ac.Checks(f"certificate, segmentation, {resolution} resolution")
    for col, name in (("violations_dice", "Dice bound"),
                      ("violations_pixel", "pixel safe condition")):
        n_bad = int(df[col].sum())
        ck.add(f"{name} is never violated", n_bad == 0,
               f"{n_bad} violations over {len(df)} runs")
    ck.report()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    p.add_argument("--root", required=True,
                   help="fullres_certificate/segmentation under OUTPUTS_MEDICAL")
    p.add_argument("--out", default="paper_results/fullres_certificate")
    p.add_argument("--perturbation", default="gaussian", choices=["gaussian", "Contrast"])
    p.add_argument("--resolution", default="full", choices=["full", "grid"])
    p.add_argument("--no_legend", action="store_true")
    p.add_argument("--row_label", default=None)
    a = p.parse_args()
    ac.apply_figure_flags(a)

    df = load(a.root, a.resolution, a.perturbation)
    stem = f"certificate_fullres_segmentation_{a.perturbation}"
    if a.resolution == "grid":
        stem += "_grid"
    os.makedirs(a.out, exist_ok=True)

    verify(df, a.resolution)
    s = cert.summary(df, ["arch", "setting"])
    s = s.sort_values(["arch", "setting"], key=lambda c: c.map(
        {x: i for i, x in enumerate(aio.ARCH_ORDER + aio.SETTING_ORDER)}))
    s["arch"] = s.arch.map(lambda x: aio.ARCH_LABEL[str(x)])
    s.to_csv(os.path.join(a.out, f"{stem}_summary.csv"), index=False)
    print(s[["arch", "setting", "certified_flip_rate", "cert_drop", "cert_rel",
             "true_drop", "n"]].to_string(index=False, float_format="%.3f"))
    cert.figure(df, "segmentation", os.path.join(a.out, f"{stem}.pdf"))


if __name__ == "__main__":
    main()
