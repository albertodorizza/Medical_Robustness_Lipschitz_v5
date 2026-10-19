"""
analysis/appendix/per_dataset.py
================================
Appendix F: the body results disaggregated by dataset.

    python -m analysis.appendix.per_dataset --cache .cache_analysis

For every (task, backbone) one table, one row per (dataset, setting):
clean metric m, corrupted metric m^c, retained fraction m^c/m, angular drift
inside and outside the mask and the leakage ratio at the head input. The
corruption grid is averaged inside each seed, then mean +/- sd over seeds.
A figure shows the retained fraction per dataset.

Outputs (in --out): per_dataset_<task>_<arch>.tex/.csv, per_dataset_retained.pdf/.png
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis import common as ac
from analysis import io as aio

STEM = "per_dataset"
TASKS = ["classification", "segmentation"]
METRIC = {"classification": "accuracy", "segmentation": "Dice"}
VALUES = ["metric_clean", "metric_corr", "retained", "d_in", "d_out", "leak"]


def prepare(root: str, cache: str | None, refresh: bool, ratio: float | None) -> pd.DataFrame:
    """In-distribution corruption rows at the head input, one value per seed."""
    frames = aio.load_all(root, cache_dir=cache, refresh=refresh,
                          points_per_run=0, progress=True)
    df = ac.at_head(frames["corruption"])
    df = df[df.in_distribution].copy()
    if ratio is not None:
        df = df[df.ratio.round(3) == round(ratio, 3)]
    df["retained"] = df.metric_corr / df.metric_clean
    df["d_in"] = 1.0 - df.cos_in
    df["d_out"] = 1.0 - df.cos_out
    df["leak"] = df.d_out / df.d_in
    key = ["task", "arch", "eval_dataset", "setting", "seed"]
    return df.groupby(key, observed=True)[VALUES].mean().reset_index()


def summary(unit: pd.DataFrame) -> pd.DataFrame:
    """Mean and sd over seeds of every value."""
    by = ["task", "arch", "eval_dataset", "setting"]
    out = None
    for v in VALUES:
        agg = ac.agg_seeds(unit, v, by).rename(columns={"mean": v, "sd": f"{v}_sd"})
        agg = agg[by + [v, f"{v}_sd"]]
        out = agg if out is None else out.merge(agg, on=by)
    return out


def tables(s: pd.DataFrame, out_dir: str, ratio: float | None) -> None:
    """One table per (task, backbone)."""
    where = "averaged over the corruption grid" if ratio is None else \
        f"at corrupted ratio $r={ratio:g}$, averaged over the rest of the grid"
    order = {x: i for i, x in enumerate(aio.SETTING_ORDER)}
    for task in TASKS:
        for arch in aio.ARCH_ORDER:
            sub = s[(s.task == task) & (s.arch == arch)].copy()
            if sub.empty:
                continue
            sub["o"] = sub.setting.astype(str).map(order)
            sub = sub.sort_values(["eval_dataset", "o"])
            rows = [{
                "Dataset": aio.DATASET_LABEL.get(str(r.eval_dataset), r.eval_dataset),
                "Setting": str(r.setting),
                "$m$": ac.fmt_pm(r.metric_clean, r.metric_clean_sd),
                "$m^c$": ac.fmt_pm(r.metric_corr, r.metric_corr_sd),
                "$m^c/m$": ac.fmt_pm(r.retained, r.retained_sd),
                r"$1-\cos_{\mathrm{in}}$": ac.fmt_pm(r.d_in, r.d_in_sd),
                r"$1-\cos_{\mathrm{out}}$": ac.fmt_pm(r.d_out, r.d_out_sd),
                r"$R_{\mathrm{leak}}$": ac.fmt_pm(r.leak, r.leak_sd, digits=2),
            } for r in sub.itertuples()]
            ac.write_table(
                pd.DataFrame(rows),
                os.path.join(out_dir, f"{STEM}_{task}_{arch}.tex"),
                caption=(f"Per-dataset results for {aio.ARCH_LABEL[arch]}, {task}. "
                         f"$m$ and $m^c$ are the clean and corrupted {METRIC[task]}, "
                         f"{where}. Drift and $R_{{\\mathrm{{leak}}}}$ are measured at "
                         f"the head input. Mean $\\pm$ s.d.\\ over seeds."),
                label=f"tab:{STEM}_{task}_{arch}",
                column_format="ll" + "r" * 6, env="table*")


def figure(s: pd.DataFrame, path: str) -> None:
    """Retained fraction per dataset: rows = task, columns = backbone."""
    plt = ac.setup_style()
    tasks = [t for t in TASKS if (s.task == t).any()]
    archs = [a for a in aio.ARCH_ORDER if (s.arch == a).any()]
    settings = [x for x in aio.SETTING_ORDER if (s.setting == x).any()]
    fig, axes = plt.subplots(len(tasks), len(archs), sharey=True, squeeze=False,
                             figsize=(7.0, 1.9 * len(tasks) + 0.5))
    width = 0.8 / len(settings)
    for i, task in enumerate(tasks):
        for j, arch in enumerate(archs):
            ax = axes[i][j]
            sub = s[(s.task == task) & (s.arch == arch)]
            dsets = sorted(sub.eval_dataset.astype(str).unique())
            x = np.arange(len(dsets))
            for k, setting in enumerate(settings):
                g = sub[sub.setting == setting].set_index("eval_dataset")
                ax.bar(x + (k - (len(settings) - 1) / 2) * width,
                       [float(g.retained.get(d, np.nan)) for d in dsets],
                       yerr=[float(g.retained_sd.get(d, np.nan)) for d in dsets],
                       width=width * 0.9, color=ac.SETTING_COLOR[setting],
                       error_kw={"lw": 0.5, "capsize": 1}, label=setting)
            ax.set_xticks(x)
            ax.set_xticklabels([aio.DATASET_LABEL.get(d, d) for d in dsets],
                               rotation=30, ha="right", fontsize=6.5)
            ax.set_ylim(0, 1.05)
            if i == 0:
                ax.set_title(aio.ARCH_LABEL[arch], fontsize=8)
            if j == 0:
                ax.set_ylabel(r"$m^c/m$")
                ax.annotate(task, xy=(0, 0.5), xycoords="axes fraction",
                            xytext=(-30, 0), textcoords="offset points",
                            rotation=90, ha="right", va="center",
                            fontsize=9, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=ac.SETTING_COLOR[x], label=x)
               for x in settings]
    fig.legend(handles=handles, frameon=False, ncol=len(settings),
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    ac.save_fig(fig, path)


def main() -> None:
    p = ac.base_parser("Appendix F: results per dataset.")
    p.add_argument("--ratio", type=float, default=None,
                   help="restrict to one corrupted ratio. Default: all.")
    args = p.parse_args()
    unit = prepare(args.results_root, args.cache, args.refresh, args.ratio)
    s = summary(unit)
    os.makedirs(args.out, exist_ok=True)
    s.to_csv(os.path.join(args.out, f"{STEM}.csv"), index=False)
    tables(s, args.out, args.ratio)
    figure(s, os.path.join(args.out, f"{STEM}_retained.pdf"))


if __name__ == "__main__":
    main()
