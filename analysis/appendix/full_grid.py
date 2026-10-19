"""
analysis/appendix/full_grid.py
==============================
Appendix G: retained fraction m^c/m over the full corruption grid.

    python -m analysis.appendix.full_grid --cache .cache_analysis
    python -m analysis.appendix.full_grid --settings H+L+SN

One table per task. Rows: backbone x corruption type x severity. Columns:
mask shape x corrupted ratio. Each cell is the mean retained fraction over the
in-distribution datasets, the selected settings and the seeds.

Outputs (in --out): full_grid_<task>[_<settings>].tex/.csv
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis import common as ac
from analysis import io as aio

STEM = "full_grid"
SHAPES = ["circular", "squared", "superpixel"]
SHAPE_SHORT = {"circular": "circ.", "squared": "patch", "superpixel": "superp."}
PTYPE_SHORT = {"Gaussian Noise": "Noise", "Defocus Blur": "Blur",
               "Brightness": "Brightness", "Contrast": "Contrast"}


def prepare(root: str, cache: str | None, refresh: bool,
            settings: list[str] | None) -> pd.DataFrame:
    frames = aio.load_all(root, cache_dir=cache, refresh=refresh,
                          points_per_run=0, progress=True)
    df = ac.at_head(frames["corruption"])
    df = df[df.in_distribution].copy()
    if settings:
        df = df[df.setting.isin(settings)]
    df["retained"] = df.metric_corr / df.metric_clean
    return df


def table(df: pd.DataFrame, task: str, path: str, settings: list[str] | None) -> None:
    sub = df[df.task == task]
    if sub.empty:
        return
    grid = (sub.groupby(["arch", "ptype", "severity", "shape", "ratio"], observed=True)
               .retained.mean().unstack(["shape", "ratio"]))
    cols = [(sh, r) for sh in SHAPES for r in sorted(sub.ratio.round(3).unique())
            if (sh, r) in grid.columns]
    rows = []
    for (arch, ptype, sev), vals in grid.iterrows():
        row = {"Backbone": aio.ARCH_LABEL[str(arch)], "Corruption": PTYPE_SHORT.get(ptype, ptype), "$s$": f"{sev}"}
        for sh, r in cols:
            row[f"{SHAPE_SHORT[sh]} {r:g}"] = f"{vals[(sh, r)]:.2f}"
        rows.append(row)
    which = "all settings" if not settings else ", ".join(settings)
    ac.write_table(
        pd.DataFrame(rows), path,
        caption=(f"Retained fraction $m^c/m$ over the full corruption grid, {task}. "
                 f"Columns give the mask shape and the corrupted ratio $r$. Noise and "
                 f"Blur denote Gaussian Noise and Defocus Blur. Mean "
                 f"over {sub.eval_dataset.nunique()} dataset(s), {which} and "
                 f"{sub.seed.nunique()} seeds."),
        label=f"tab:{STEM}_{task}",
        column_format="llr" + "r" * len(cols), env="table*", pos="!tp",
        size=r"\footnotesize\setlength{\tabcolsep}{4pt}")


def main() -> None:
    p = ac.base_parser("Appendix G: full corruption grid.")
    p.add_argument("--settings", nargs="+", default=None, choices=aio.SETTING_ORDER)
    args = p.parse_args()
    df = prepare(args.results_root, args.cache, args.refresh, args.settings)
    os.makedirs(args.out, exist_ok=True)
    tag = "" if not args.settings else "_" + "-".join(args.settings)
    for task in ("classification", "segmentation"):
        table(df, task, os.path.join(args.out, f"{STEM}_{task}{tag}.tex"), args.settings)


if __name__ == "__main__":
    main()
