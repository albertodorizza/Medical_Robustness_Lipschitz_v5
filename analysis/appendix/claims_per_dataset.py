"""
analysis/appendix/claims_per_dataset.py
=======================================
Appendix: the main claims of the paper, checked on each dataset separately.

Reads the outputs already written by the other scripts (no cache needed):

    2_drift/drift_predicts_damage_r0.3-0.6-0.9_cells.csv     (script 2)
    4_geometry/geometry_beats_area_both_cells.csv            (script 4)
    appendix_per_dataset/per_dataset.csv                     (per_dataset.py)
    6_certificate/certificate_needs_sigma_<task>_gaussian_runs.csv  (script 6)

    python -m analysis.appendix.claims_per_dataset --out paper_results

One row per dataset, averaged over backbones (and settings where stated):
    out wins      share of cells where the outside drift is the better
                  predictor at r = 0.3 (segmentation: lesion untouched)
    scat./enc.    median damage ratio, scattered over encircling, r <= 0.6
    retained      m^c/m for H and for H+L+SN
    certified     certified and observed drop for H+L+SN, Gaussian noise

Outputs: <out>/appendix_per_dataset/claims_per_dataset.tex/.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis import common as ac
from analysis import io as aio

TASKS = ["classification", "segmentation"]


def out_wins(path: str) -> pd.Series:
    c = pd.read_csv(path)
    c = c[(c["shape"] == "all") & np.isclose(c.ratio, 0.3)]
    c = c[((c.task == "classification") & (c.stratum == "overall"))
          | ((c.task == "segmentation") & (c.stratum == "none"))]
    c = c.dropna(subset=["rs_in", "rs_out"])
    return (c.rs_out > c.rs_in).groupby([c.task, c.eval_dataset]).mean()


def geometry(path: str) -> pd.DataFrame:
    g = pd.read_csv(path)
    g = g[g.ratio <= 0.6 + 1e-9]
    return g.groupby(["task", "eval_dataset"]).agg(
        geo_ratio=("damage_ratio", "median"),
        geo_win=("damage_scat", lambda s: (s > g.loc[s.index, "damage_enc"]).mean()))


def retained(path: str) -> pd.DataFrame:
    p = pd.read_csv(path)
    r = p.groupby(["task", "eval_dataset", "setting"]).retained.mean().unstack("setting")
    return r[["H", "H+L+SN"]].rename(columns={"H": "ret_H", "H+L+SN": "ret_HLSN"})


def certificate(root: str) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        path = os.path.join(root, f"certificate_needs_sigma_{task}_gaussian_runs.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        d = d[(d.setting == "H+L+SN") & (d.train_dataset == d.eval_dataset)]
        agg = d.groupby("eval_dataset")[["cert_drop", "true_drop"]].mean()
        agg["task"] = task
        rows.append(agg.reset_index())
    return pd.concat(rows).set_index(["task", "eval_dataset"])


def main() -> None:
    p = argparse.ArgumentParser(description="Main claims per dataset.")
    p.add_argument("--out", default="paper_results",
                   help="root of the analysis outputs. Default paper_results.")
    args = p.parse_args()
    o = args.out

    df = pd.concat([
        out_wins(os.path.join(o, "2_drift", "drift_predicts_damage_r0.3-0.6-0.9_cells.csv")).rename("out_wins"),
        geometry(os.path.join(o, "4_geometry", "geometry_beats_area_both_cells.csv")),
        retained(os.path.join(o, "appendix_per_dataset", "per_dataset.csv")),
        certificate(os.path.join(o, "6_certificate")),
    ], axis=1).reset_index().rename(columns={"level_0": "task", "level_1": "eval_dataset"})

    order = {d: i for i, d in enumerate(aio.DATASET_LABEL)}
    df["o"] = df.eval_dataset.map(order)
    df = df.sort_values(["task", "o"]).drop(columns="o")
    dest = os.path.join(o, "appendix_per_dataset")
    os.makedirs(dest, exist_ok=True)
    df.to_csv(os.path.join(dest, "claims_per_dataset.csv"), index=False)

    rows = [{
        "Task": "cls" if r.task == "classification" else "seg",
        "Dataset": aio.DATASET_LABEL.get(r.eval_dataset, r.eval_dataset),
        "out wins": f"{r.out_wins:.2f}",
        "scat./enc.": f"{r.geo_ratio:.2f}",
        "scat. worse": f"{r.geo_win:.2f}",
        "ret.\\ H": f"{r.ret_H:.2f}",
        "ret.\\ H+L+SN": f"{r.ret_HLSN:.2f}",
        "certified": f"{r.cert_drop:.3f}",
        "observed": f"{r.true_drop:.3f}",
    } for r in df.itertuples()]
    ac.write_table(
        pd.DataFrame(rows), os.path.join(dest, "claims_per_dataset.tex"),
        caption=(r"Main results on each dataset, averaged over the three backbones. "
                 r"\emph{out wins}: share of cells where the drift outside the "
                 r"corrupted region predicts the damage better than the drift "
                 r"inside it, at $r=0.3$ (segmentation: images where the "
                 r"corruption does not overlap the lesion). \emph{scat./enc.}: "
                 r"median ratio between the damage of scattered and encircling "
                 r"masks, and \emph{scat.\ worse}: share of comparisons where the "
                 r"scattered mask is more harmful, both at $r \le 0.6$. "
                 r"\emph{ret.}: retained fraction $m^c/m$. \emph{certified} and "
                 r"\emph{observed}: certified and observed drop for H+L+SN under "
                 r"Gaussian noise."),
        label="tab:claims_per_dataset",
        column_format="ll" + "r" * 7, env="table*", pos="!tp")


if __name__ == "__main__":
    main()
