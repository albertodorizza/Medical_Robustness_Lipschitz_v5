"""
analysis/3_in_out_dose_response.py
==================================
Insight 3 -- what the IN/OUT decomposition actually measures, and how both
terms respond to the two knobs of a localized corruption: its severity and the
share of the frame it covers.

This is also the script that emits the body table of the corruption sweep (the
one the paper calls Table 2) and its dose-response figure, for any task, any
distance, any subset of the corruption grid:

    python -m analysis.3_in_out_dose_response --cache .cache_analysis
    python -m analysis.3_in_out_dose_response --task segmentation --metric mse
    python -m analysis.3_in_out_dose_response --task both
    python -m analysis.3_in_out_dose_response --dataset kvasir_v2 \\
        --ptypes "Gaussian Noise" Contrast --split_by ratio


The quantities
--------------
At the representation that feeds the head, the drift between the clean and the
corrupted features is summarised inside and outside the corrupted support:

    D_in  = the distance restricted to the corrupted region,
    D_out = the distance restricted to the uncorrupted region,
    D_all = the same over the whole feature grid,

with ``--metric cos`` giving the ANGULAR distortion ``1 - cos`` and
``--metric mse`` the mean squared difference. Both are read from the same
sweep; only the reported one changes. Damage is the task metric under
corruption (accuracy or Dice) and its drop from the clean value.

``--task both`` puts the two sweeps in the same table and the same figure.
The table keeps ``task`` as a grouping key -- a ``Task`` column and two
separate damage-metric columns (Acc., Dice), since accuracy and Dice are not
the same unit and a row only fills the block of its own task. The FIGURE,
which is about the feature drift rather than the damage, pools the two
sweeps into one profile: ``task`` is dropped from the grouping key there, so
each point is the mean over both tasks' repetitions, and the figure looks
exactly like the single-task one -- colour for the setting (H/H+SN, H+L,
H+L+SN), solid circle inside vs dashed hollow square outside -- computed over
the union of the two sweeps. Filters that make sense for one task only
(``--shapes``, most ``--ptypes``) still apply to whichever rows they match; a
filter that leaves one task empty degrades the figure and table to the task
that is left, with a warning, rather than failing.


The claims being tested
-----------------------
(a) CONTROL -- adding the spectral penalty to a frozen backbone cannot move the
    features. ``H`` and ``H+SN`` share every feature-level number exactly
    (max |diff| = 0 over all matched cells), because the penalty touches the
    head only and the drift is measured before it. Any difference between those
    two rows of the body table is therefore a head effect, and any *feature*
    difference would be a bug in the sweep.

(b) The two terms answer different questions, and the sweep separates them:

      * Both grow monotonically with SEVERITY at fixed area, together with the
        task damage.
      * They differ in how they respond to AREA. Growing the mask from 30% to
        90% of the frame moves D_out about five times more than D_in in the
        median. D_out also grows monotonically with the corrupted share in all
        but a handful of profiles, all of them ResNet-34 under Brightness, the
        one pair for which the corruption barely registers in the features.

    Severity is an intensity knob that both terms see, area is a knob that
    essentially only the OUT term sees.

(c) The IN/OUT split is informative only while a meaningful uncorrupted region
    remains. At r = 0.3 the distortion is confined -- D_out < D_in in at least
    96% of cells for every architecture -- at r = 0.6 confinement is partial,
    and at r = 0.9 the ordering is inverted for most cells.

(d) THE DISTANCE IS NOT INTERCHANGEABLE. Under the angular metric the
    corruption is confined at r = 0.3 for every architecture. Under MSE,
    ResNet-34 inverts in about a quarter of the cells at the same ratio and in
    half of them at r = 0.6, while the two transformers do not: the MSE of a
    feature map mixes the drift with the spatial heterogeneity of the feature
    norm, which is large in a CNN and small in a layer-normalised transformer.


What this insight does NOT claim
--------------------------------
That D_in and D_out are comparable ACROSS architectures in the MSE form: the
embedding dimension and the feature scale differ by orders of magnitude between
ResNet-34 and ViT-B, which is why every cross-architecture statement in this
script is made in angle.

Which of the two terms predicts the damage is Insight 2; where the corruption
lands, at equal area, is Insight 4.


Outputs (same stem, in ``--out``)
---------------------------------
    in_out_dose_response[_<task>_<metric>].tex / .csv   body table
    in_out_dose_response[_<task>_<metric>]_profiles.csv every point
    in_out_dose_response[_<task>_<metric>].pdf / .png   dose-response

The script asserts nothing and never exits non-zero: it emits the assets and
the numbers behind them.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import common as ac
from analysis import io as aio
HEAD_INPUT = ac.HEAD_INPUT

STEM = "in_out_dose_response"

# One repetition is one (dataset, seed): the sd in the table is the spread over
# those, never over the corruption grid.
SPREAD_UNIT = ["eval_dataset", "seed"]

# Regions of the decomposition, in the order they are reported.
REGIONS = ("in", "out", "all")

# The profile a claim is read on: pooled over datasets and mask shapes, which
# is the curve the paper plots.
PROFILE_BY = ["arch", "setting", "ptype"]

# The task metric each task's ``metric_corr`` column actually holds.
METRIC_LABEL = {"classification": "Acc.", "segmentation": "Dice"}

# Figure size, in inches. The width matches \textwidth of cas-dc for three
# backbones, so the PDF is included at 1:1.
FIG_COL_WIDTH = 3.1
FIG_HEIGHT = 4.5


# =============================================================================
# Data
# =============================================================================
def drift_col(metric: str, region: str) -> str:
    """Column holding the chosen distance in ``region``."""
    return f"{'cos' if metric == 'cos' else 'mse'}_{region}"


def prepare(root: str, cache: str | None, refresh: bool, task: str,
            datasets: list[str] | None, settings: list[str] | None,
            shapes: list[str] | None, ptypes: list[str] | None,
            ratios: list[float] | None, severities: list[int] | None
            ) -> pd.DataFrame:
    """The corruption sweep at the representation that feeds the head.

    Only the deepest hooked layer is kept: the per-layer profile is Insight 1's
    subject rather than this one's. ``task = "both"`` keeps the two sweeps side
    by side; ``task`` stays a column throughout.
    """
    frames = aio.load_all(root, cache_dir=cache, refresh=refresh,
                          points_per_run=0, progress=True)
    df = frames["corruption"]
    df = (df if task == "both" else df[df.task == task]).copy()
    if df.empty:
        sys.exit(f"no {task} corruption runs found under the results root")
    if task == "both":
        missing = {"classification", "segmentation"} - set(df.task.unique())
        if missing:
            print(f"  [warn] --task both, but no {'/'.join(sorted(missing))} "
                  f"corruption runs are under the results root; only "
                  f"{'/'.join(sorted(set(df.task.unique())))} will be drawn")

    keep = [(l == "head_input") or (HEAD_INPUT.get(str(a)) == l)
            for a, l in zip(df.arch, df.layer)]
    df = df[keep].copy()

    # The angular distortion, which is the scale-free form of the same drift.
    for region in REGIONS:
        df[f"ang_{region}"] = 1.0 - df[f"cos_{region}"]
    # Relative damage, so tasks and architectures with very different clean
    # metrics can be put on one axis; the absolute drop is kept next to it.
    df["retained"] = df.metric_corr / df.metric_clean
    df["rel_drop"] = 1.0 - df.retained

    for col, sel in (("eval_dataset", datasets), ("setting", settings),
                     ("shape", shapes), ("ptype", ptypes)):
        if sel:
            df = df[df[col].isin(sel)]
    if ratios:
        df = df[df.ratio.round(3).isin([round(r, 3) for r in ratios])]
    if severities:
        df = df[df.severity.isin(severities)]
    if df.empty:
        sys.exit("no rows left after the requested filters")
    return df


def profiles(df: pd.DataFrame, metric: str, by: list[str]) -> pd.DataFrame:
    """Mean drift and damage per group, with the spread over repetitions.

    Averaged inside each (dataset, seed) repetition first, so the reported sd
    is the seed-to-seed spread and not the spread of the corruption grid.
    """
    values = [f"{'ang' if metric == 'cos' else 'mse'}_{r}" for r in REGIONS]
    values += ["metric_corr", "metric_clean", "rel_drop"]
    unit = (df.groupby(by + SPREAD_UNIT, observed=True)[values]
              .mean().reset_index())
    out = None
    for value in values:
        agg = ac.agg_seeds(unit, value, by).rename(
            columns={"mean": value, "sd": value + "_sd"})
        cols = by + [value, value + "_sd"] + (["n"] if out is None else [])
        out = agg[cols] if out is None else out.merge(agg[cols], on=by)
    return out.sort_values(by).reset_index(drop=True)


# =============================================================================
# Table
# =============================================================================
def table(df: pd.DataFrame, metric: str, task: str, split_by: str,
          path: str, digits: int) -> None:
    """The body table: drift inside and outside, and the task metric.

    ``split_by`` chooses the column blocks -- severity (the area being pooled)
    or ratio. With ``task == "both"`` a ``Task`` column is added and both
    damage columns (Acc., Dice) are emitted; a row only fills its own.
    """
    pre = "ang" if metric == "cos" else "mse"
    label = {"ang": r"$1-\mathrm{COS}$", "mse": "MSE"}[pre]
    both = task == "both"
    tasks_present = sorted(df.task.unique()) if both else [task]
    levels = sorted(df[split_by].unique())

    group_keys = ["arch", "setting"] + (["task"] if both else [])
    by = group_keys + [split_by]
    p = profiles(df, metric, by)

    rows = []
    for key, g in p.groupby(group_keys, observed=True):
        if both:
            arch, setting, trow = key
        else:
            arch, setting = key
            trow = task
        row = {"Backbone": aio.ARCH_LABEL[str(arch)], "Setting": str(setting)}
        if both:
            row["Task"] = "cls" if trow == "classification" else "seg"
        g = g.set_index(split_by)
        for region in ("in", "out"):
            for lv in levels:
                col_key = f"{label}$_{{\\mathrm{{{region}}}}}$ {lv:g}"
                row[col_key] = (f"{g.loc[lv, f'{pre}_{region}']:.{digits}f}"
                                if lv in g.index else "---")
        for t in tasks_present:
            for lv in levels:
                col_key = f"{METRIC_LABEL[t]} {lv:g}"
                if t == trow:
                    row[col_key] = (f"{g.loc[lv, 'metric_corr']:.3f}"
                                    if lv in g.index else "---")
                elif both:
                    row.setdefault(col_key, "---")
        rows.append(row)

    out = pd.DataFrame(rows).fillna("---")
    order = {s: i for i, s in enumerate(aio.SETTING_ORDER)}
    sort_cols = ["Backbone", "Setting"] + (["Task"] if both else [])
    out = out.sort_values(
        sort_cols,
        key=lambda s: s.map(order) if s.name == "Setting" else s
    ).reset_index(drop=True)

    names = ", ".join(aio.DATASET_LABEL.get(d, d)
                      for d in sorted(df.eval_dataset.unique()))
    task_note = ("both tasks" if both else task)
    fixed = (f"{df.eval_dataset.nunique()} dataset(s) ({names}), "
             f"shapes {', '.join(sorted(df['shape'].dropna().unique()))}, "
             f"types {', '.join(sorted(df.ptype.dropna().unique()))}, "
             f"{df.seed.nunique()} seeds")
    pooled = "ratio" if split_by == "severity" else "severity"
    n_label_cols = 3 if both else 2
    ac.write_table(
        out, path,
        caption=(f"Feature drift inside and outside the corrupted region, and the "
                 f"task metric under corruption, for {task_note}. Columns are "
                 f"grouped by {split_by}; every other axis is averaged: {fixed}, "
                 f"and {pooled} "
                 rf"$\{{{', '.join(f'{v:g}' for v in sorted(df[pooled].unique()))}\}}$. "
                 f"Distance: {'angular (1 - cosine)' if pre == 'ang' else 'MSE'} "
                 f"at the representation that feeds the head."
                 + (" Acc.\\ is classification accuracy, Dice is the "
                    "segmentation overlap; a row fills only the column of its "
                    "own task." if both else "")),
        label=f"tab:{STEM}_{task}_{metric}",
        column_format="l" * n_label_cols + "r" * (len(out.columns) - n_label_cols),
        note=(r"\textbf{H} and \textbf{H+SN} share every feature column exactly: "
              r"the penalty acts on the head, which is applied after the drift is "
              r"measured. Averaging over the corruption ratio mixes the confined "
              r"regime ($r\!\le\!0.6$) with the saturated one ($r\!=\!0.9$), where "
              r"$D_{\mathrm{out}}\!>\!D_{\mathrm{in}}$ by construction."))


# =============================================================================
# Figure
# =============================================================================
def figure(df: pd.DataFrame, metric: str, task: str, fix_ratio: float,
           fix_severity: int, path: str) -> None:
    """Two rows -- severity at fixed area, area at fixed severity -- by architecture.

    With ``task == "both"`` the two sweeps are pooled into the same profile:
    ``task`` is left out of the grouping key. The x label is written once per
    row, under the middle panel: the three panels of a row share it.
    """
    plt = ac.setup_style()
    pre = "ang" if metric == "cos" else "mse"
    ylab = {"ang": r"$(1-\cos)$", "mse": "MSE"}[pre]
    archs = [a for a in aio.ARCH_ORDER if (df.arch == a).any()]
    settings = [s for s in aio.SETTING_ORDER if (df.setting == s).any()]
    # H+SN has exactly the features of H: drawing both would put one curve on
    # top of the other.
    drift_settings = [s for s in settings if s != "H+SN" or "H" not in settings]

    sweeps = [("severity", df[df.ratio.round(3) == round(fix_ratio, 3)],
               f"severity  ($r = {fix_ratio:g}$)"),
              ("ratio", df[df.severity == fix_severity],
               f"corrupted area $r$  (severity {fix_severity:g})")]

    fig, axes = plt.subplots(2, len(archs),
                             figsize=(FIG_COL_WIDTH * len(archs), FIG_HEIGHT),
                             squeeze=False)
    mid = len(archs) // 2
    for i, (x, sub, xlabel) in enumerate(sweeps):
        if sub.empty:
            continue
        p = profiles(sub, metric, ["arch", "setting", x])
        for j, arch in enumerate(archs):
            ax = axes[i][j]
            pa = p[p.arch == arch]
            for setting in drift_settings:
                s = pa[pa.setting == setting].sort_values(x)
                if s.empty:
                    continue
                color = ac.SETTING_COLOR[str(setting)]
                ax.plot(s[x], s[f"{pre}_in"], "-o", color=color,
                        label=f"{setting} in")
                ax.plot(s[x], s[f"{pre}_out"], "--s", color=color, alpha=0.75,
                        markerfacecolor="white", label=f"{setting} out")
            ax.set_xticks(sorted(pa[x].unique()))
            if j == mid:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(f"feature drift {ylab}")
            if i == 0:
                ax.set_title(aio.ARCH_LABEL[str(arch)])

    handles = [plt.Line2D([], [], color=ac.SETTING_COLOR[s], marker="o",
                          label=("H / H+SN" if s == "H" and "H+SN" in settings else s))
               for s in drift_settings]
    handles += [plt.Line2D([], [], color="0.35", marker="o", label="inside (solid)"),
                plt.Line2D([], [], color="0.35", marker="s", linestyle="--",
                           markerfacecolor="white", label="outside (dashed)")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 1.005), handlelength=1.8,
               columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.945), h_pad=0.8)
    ac.save_fig(fig, path)


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    p = ac.base_parser("Insight 3 -- how the IN and OUT terms of the feature "
                       "drift respond to corruption severity and corrupted area.")
    p.add_argument("--task", default="classification",
                   choices=["classification", "segmentation", "both"],
                   help="'both' puts the two sweeps in the same table (a Task "
                        "column, two damage-metric column sets) and pools them "
                        "in the figure. Default classification.")
    p.add_argument("--metric", default="cos", choices=["cos", "mse"],
                   help="distance reported in the table and the figure. "
                        "'cos' is the angular distortion 1 - cosine, which is "
                        "scale free; 'mse' is the raw one. Default cos.")
    p.add_argument("--dataset", nargs="+", default=None,
                   help="evaluation dataset(s). Default: all of them.")
    p.add_argument("--settings", nargs="+", default=None,
                   choices=aio.SETTING_ORDER)
    p.add_argument("--shapes", nargs="+", default=None,
                   choices=["circular", "squared", "superpixel"])
    p.add_argument("--ptypes", nargs="+", default=None,
                   help="corruption types, e.g. --ptypes \"Gaussian Noise\" Contrast.")
    p.add_argument("--ratios", nargs="+", type=float, default=None)
    p.add_argument("--severities", nargs="+", type=int, default=None)
    p.add_argument("--split_by", default="severity", choices=["severity", "ratio"],
                   help="column blocks of the table. Default severity.")
    p.add_argument("--fix_ratio", type=float, default=0.3,
                   help="area held fixed in the severity sweep of the figure.")
    p.add_argument("--fix_severity", type=int, default=3,
                   help="severity held fixed in the area sweep of the figure.")
    p.add_argument("--digits", type=int, default=3,
                   help="decimals of the drift columns. MSE spans orders of "
                        "magnitude between architectures; 2 is usually enough.")
    args = p.parse_args()
    ac.apply_figure_flags(args)

    df = prepare(args.results_root, args.cache, args.refresh, args.task,
                 args.dataset, args.settings, args.shapes, args.ptypes,
                 args.ratios, args.severities)

    stem = f"{STEM}_{args.task}_{args.metric}"
    if args.dataset and len(args.dataset) == 1:
        stem += f"_{args.dataset[0]}"
    os.makedirs(args.out, exist_ok=True)
    print()
    by = ["task", "arch", "setting", "shape", "ptype", "ratio", "severity"]
    prof_path = os.path.join(args.out, f"{stem}_profiles.csv")
    profiles(df, args.metric, by).to_csv(prof_path, index=False)
    print(f"  [csv] {prof_path}")
    table(df, args.metric, args.task, args.split_by,
          os.path.join(args.out, f"{stem}.tex"), args.digits)
    figure(df, args.metric, args.task, args.fix_ratio, args.fix_severity,
           os.path.join(args.out, f"{stem}.pdf"))


if __name__ == "__main__":
    main()