"""
fullres_certificate/summarize.py
================================
Grid against full-resolution certificate, per perturbation, backbone and
setting, averaged over the runs (datasets x seeds) exactly as the certificate
table of the paper averages them.

    python -m fullres_certificate.summarize \
        --root /home/a.dorizza/OUTPUTS_MEDICAL/fullres_certificate/segmentation \
        --out paper_results/fullres_certificate

The ``*_grid`` columns must reproduce the segmentation certificate table of
the paper (e.g. certified drop 0.135 on average for H+L+SN under Gaussian
noise); if they do not, the two runs are not comparable and the full-resolution
columns should not be read.
"""

import argparse
import glob
import json
import os

import pandas as pd

ARCH_LABEL = {"resnet": "ResNet-34", "vit": "ViT-B/16", "dinov2": "DINOv2"}
ARCH_ORDER = ["ResNet-34", "ViT-B/16", "DINOv2"]
SETTING_ORDER = ["H", "H+SN", "H+L", "H+L+SN"]

COLUMNS = [  # (summary key, column name)
    ("dice_clean_grid", "Dice grid"), ("dice_clean_full", "Dice full"),
    ("at_risk_grid", "at-risk grid"), ("at_risk_full", "at-risk full"),
    ("risk_boundary_only", "risk bdry-only"),
    ("cert_drop_grid", "cert grid"), ("cert_drop_full", "cert full"),
    ("true_drop_grid", "obs grid"), ("true_drop_full", "obs full"),
    ("flip_full", "flip full"), ("flip_unseen_by_grid", "flip unseen"),
]


def setting_of(run: dict) -> str:
    lam = run.get("lambda_sn")
    return (("H+L" if run.get("freeze") == "head+last" else "H")
            + ("+SN" if lam else ""))


def load(root: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(root, "**", "*_fullres.json"), recursive=True):
        with open(path) as f:
            b = json.load(f)
        rows.append({"perturbation": b["tag"],
                     "dataset": b["eval_dataset"],
                     "backbone": ARCH_LABEL.get(b["run"]["arch"], b["run"]["arch"]),
                     "setting": setting_of(b["run"]),
                     "seed": b["run"]["seed"], **b["summary"]})
    if not rows:
        raise SystemExit(f"no *_fullres.json under {root}")
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="paper_results/fullres_certificate")
    a = p.parse_args()

    df = load(a.root)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "fullres_runs.csv"), index=False)

    keys = [k for k, _ in COLUMNS]
    tab = (df.groupby(["perturbation", "backbone", "setting"])[keys].mean()
             .rename(columns=dict(COLUMNS)))
    tab["cert/Dice full"] = tab["cert full"] / tab["Dice full"]
    tab["n runs"] = df.groupby(["perturbation", "backbone", "setting"]).size()
    tab = tab.reset_index()
    tab["backbone"] = pd.Categorical(tab.backbone, ARCH_ORDER, ordered=True)
    tab["setting"] = pd.Categorical(tab.setting, SETTING_ORDER, ordered=True)
    tab = tab.sort_values(["perturbation", "backbone", "setting"])
    tab.to_csv(os.path.join(a.out, "fullres_summary.csv"), index=False)

    pd.set_option("display.width", 250)
    for pert, g in tab.groupby("perturbation", sort=False):
        print(f"\n=== {pert} ===")
        print(g.drop(columns="perturbation").to_string(index=False, float_format="%.3f"))
        hl = df[(df.perturbation == pert) & (df.setting == "H+L+SN")]
        print(f"H+L+SN mean: cert grid {hl.cert_drop_grid.mean():.3f}  "
              f"cert full {hl.cert_drop_full.mean():.3f}  "
              f"obs full {hl.true_drop_full.mean():.4f}")

    print("\nVERIFICATION (all must be 0)")
    for k in ("violations_dice_grid", "violations_pixel_grid",
              "violations_dice_full", "violations_pixel_full"):
        print(f"  {k:24s} {int(df[k].sum())}")
    print(f"\nruns: {len(df)}   -> {a.out}/fullres_summary.csv, fullres_runs.csv")


if __name__ == "__main__":
    main()
