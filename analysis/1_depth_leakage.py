"""
analysis/1_depth_leakage.py
===========================
Insight 1 -- how far a localized corruption spreads beyond its own spatial
support, as a function of depth in the backbone.

One figure: a grid with the corrupted ratio across the columns and the
architecture down the rows. Everything else the sweep varies -- corruption
type, severity, mask shape, dataset and seed -- is averaged inside each panel,
and each of those is a filter, so the same figure shows how the profiles change
when only one of them is kept.

``--task segmentation`` switches to the segmentation sweep. If that sweep hooks
only the representation that feeds the head, there is no depth axis to plot and
the figure becomes a grouped bar chart of the same quantity, one bar per
architecture and fine-tuning setting, still one panel per corrupted ratio.

``--task both`` draws the two sweeps in the same panels. The two are never
averaged together -- they are different models trained on different data -- so
the task is a THIRD encoding channel, not a grouping key: the classification
profile is a filled line and the segmentation measurement is a hollow marker.
Every aggregation in the script carries ``task`` as a key, so nothing
downstream -- the profile, the shallow normalisation or the mask-geometry
control -- ever mixes the two.

``--task both --layout stacked`` is the figure the paper prints: the two tasks
down the rows, the corrupted ratio across the columns, one shared x axis
(relative depth) and one shared y axis (R_leak is dimensionless), one key
underneath. It replaces the two-PDF composite of ``stack_tasks.py`` for this
insight: with a shared axis the x label and the tick row are drawn once, the
two rows sit on the same scale, and the figure is about a third shorter.

    python -m analysis.1_depth_leakage --cache .cache_analysis
    python -m analysis.1_depth_leakage --task segmentation
    python -m analysis.1_depth_leakage --task both --merge_arch
    python -m analysis.1_depth_leakage --task both --layout stacked
    python -m analysis.1_depth_leakage --shapes superpixel
    python -m analysis.1_depth_leakage --ptypes "Gaussian Noise" Contrast
    python -m analysis.1_depth_leakage --merge_arch --agg median

The quantity
------------
Propagation is measured by the **angular leakage ratio**

    R_leak = (1 - cos_out) / (1 - cos_in)   in [0, +inf)

the angular distortion measured outside the corrupted region, relative to the
one measured inside it.

    0   the corruption is perfectly confined to its spatial support
    1   the uncorrupted region is as distorted as the corrupted one
    >1  it is MORE distorted -- possible, and measured; see below

It is a ratio of two averages taken over two disjoint regions, and nothing
bounds it above.

It is both dimensionless and scale-free, which the MSE ratio is not: the
feature energy of a backbone changes by orders of magnitude between adjacent
layers -- it jumps by a factor of about 1700 at ResNet's ``layer4`` -- so a
ratio of raw MSEs mixes propagation with the spatial heterogeneity of the
feature norm and cannot be compared across depths. Only the cosine form is
reported here.


What the figure shows, and what the claim is
-------------------------------------------
The claim is about DEPTH, not about the level of the ratio: in every backbone
and at every corrupted area the deepest probe -- the representation that feeds
the head -- leaks more than the shallowest one. At r = 0.9 the profile is not
monotone: it peaks at mid-depth and comes down towards the head, so the
amplification is a statement about the two ends of the profile, not about
every step along it.

The LEVEL of the ratio does depend on r: the representation that feeds the
head sits at about 0.5 of the in-mask distortion at r = 0.3, at 0.8 at r = 0.6
and above 1 at r = 0.9. ``--normalize shallow`` removes the level from the
picture by dividing each profile by its own shallowest probe; what is left is
the amplification with depth.

``H`` and ``H+SN`` are one curve, not two: the spectral penalty acts on the
head, the drift is measured before it, and the two share every feature-level
number exactly.


Why a curve can go above 1, and why that is not an artefact
----------------------------------------------------------
Numerator and denominator are averages over two DIFFERENT sets of feature
locations, of very different sizes, and no conservation law relates them:

* **What is left to average over.** At r = 0.9 the uncorrupted region is a thin
  strip wrapped around the mask, so every token in it is adjacent to corrupted
  content, while the IN average runs over 90% of the frame and includes tokens
  far from any boundary.

* **A ratio has a heavy right tail.** A cell whose IN distortion happens to be
  small produces a large ratio and the mean feels it. ``--agg median`` is the
  check that a curve is not carried by a handful of cells.

* **It is NOT the length of the mask boundary.** At a fixed corrupted area the
  three mask geometries differ several-fold in perimeter, and R_leak at the
  head input moves by about 20% between them while depth moves it by far more.
  ``*_shape_control.csv`` reports this comparison for every architecture and
  ratio.


Shared definitions
------------------
``HEAD_INPUT``, ``SHAPE_CLASS``, ``FT_FIRST_LAYER`` and :func:`layer_depth`
live in :mod:`analysis.common` and are re-exported here: a module whose name
begins with a digit is not a valid target of an ``import`` statement, so no
script of the series can import from another.


Outputs (stem carries the filters, in ``--out``)
------------------------------------------------
    depth_leakage_<task>[_r<ratios>][_merged|_row|_stacked][_median][_norm].pdf/.png
    depth_leakage_<task>[...].csv                  plotted points
    depth_leakage_<task>[...]_shape_control.csv    R_leak at the head, per shape
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import common as ac
from analysis import io as aio

STEM = "depth_leakage"

HEAD_INPUT = ac.HEAD_INPUT
SHAPE_CLASS = ac.SHAPE_CLASS
FT_FIRST_LAYER = ac.FT_FIRST_LAYER
layer_depth = ac.layer_depth


# =============================================================================
# Data
# =============================================================================
def prepare(root: str, cache: str | None, refresh: bool, task: str,
            ratios: list[float] | None, shapes: list[str] | None,
            ptypes: list[str] | None, datasets: list[str] | None,
            metric: str = "cos") -> pd.DataFrame:
    """Corruption sweep for one task, or for both, at every layer hooked.

    ``task = "both"`` keeps the two sweeps side by side rather than pooling
    them; everything downstream groups by ``task`` as well, so the two never
    enter the same average.
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

    df["depth"] = [layer_depth(a, l) for a, l in zip(df.arch.astype(str), df.layer)]
    df["rel_depth"] = df.groupby("arch", observed=True).depth.transform(
        lambda s: s / s.max())

    # (1 - cos) is the angular distortion; its out/in ratio is the propagation.
    # An empty region yields cos = NaN upstream, never 0, so NaN propagates here
    # and is dropped by the aggregation instead of counting as "no leakage".
    if metric == "mse":
        df["leak_ratio"] = df.mse_out / df.mse_in
    else:
        df["leak_ratio"] = (1.0 - df.cos_out) / (1.0 - df.cos_in)

    if ratios:
        df = df[df.ratio.round(3).isin([round(r, 3) for r in ratios])]
    for col, sel in (("shape", shapes), ("ptype", ptypes),
                     ("eval_dataset", datasets)):
        if sel:
            df = df[df[col].isin(sel)]
    if df.empty:
        sys.exit("no rows left after the requested filters")
    return df


def profile(df: pd.DataFrame, agg: str = "mean") -> pd.DataFrame:
    """Depth profile of ``leak_ratio``, aggregated over everything else.

    ``task`` is a grouping key and not an axis to average over: with
    ``--task both`` the two sweeps share an architecture, a setting, a ratio
    AND a depth, so leaving it out would silently merge them.
    """
    return (df.groupby(["task", "arch", "setting", "ratio", "depth", "rel_depth"],
                       observed=True)
              .leak_ratio.agg(agg).reset_index()
              .sort_values(["task", "arch", "setting", "ratio", "depth"]))


def normalize_shallow(p: pd.DataFrame) -> pd.DataFrame:
    """Divide each profile by its own shallowest probe.

    The shallowest probe carries the same geometric factor as every deeper
    probe of the same profile, so dividing by it cancels that factor to first
    order and leaves the amplification with depth.
    """
    key = ["task", "arch", "setting", "ratio"]
    base = (p.sort_values("depth").groupby(key, observed=True)
             .leak_ratio.transform("first"))
    out = p.copy()
    out["leak_ratio"] = p.leak_ratio / base
    return out


def shape_control(df: pd.DataFrame) -> pd.DataFrame:
    """``R_leak`` at the head input, per mask geometry, at each corrupted area.

    If the ratio were driven by boundary adjacency it would track the mask
    geometry, whose perimeter differs several-fold at equal area.
    """
    at_head = [ac.HEAD_INPUT.get(str(a)) == l or l == "head_input"
               for a, l in zip(df.arch, df.layer)]
    h = df[at_head]
    if h.empty:
        return pd.DataFrame()
    return (h.groupby(["task", "arch", "ratio", "shape"], observed=True)
             .leak_ratio.mean().reset_index())


# =============================================================================
# Figure
# =============================================================================
ARCH_COLOR = ac.ARCH_COLOR
SETTING_STYLE = {"H": "-", "H+SN": "-", "H+L": "--", "H+L+SN": ":"}

TASK_ORDER = ["classification", "segmentation"]
TASK_SHORT = {"classification": "cls", "segmentation": "seg"}
TASK_HATCH = {"classification": "", "segmentation": "//"}

YLABEL = r"$R_{\mathrm{leak}}$"
NORM_YLABEL = (r"$R_{\mathrm{leak}}(d)\,/\,R_{\mathrm{leak}}"
               r"(d_{\mathrm{first}})$   amplification with depth")
XLABEL = "relative depth in the backbone"

# Width of the stacked figure, in inches: the \textwidth of cas-dc, so that the
# PDF is included at 1:1 and the 7-8 pt fonts stay 7-8 pt on the page.
STACKED_WIDTH = 7.0
STACKED_ROW_HEIGHT = 1.45
# Distance of the task label from the left spine, in points: clears the tick
# labels and the R_leak axis label.
ROW_LABEL_OFFSET = 22


def figure(df: pd.DataFrame, path: str, merge_arch: bool = False,
           agg: str = "mean", normalize: str = "none",
           layout: str = "grid") -> None:
    """A grid of depth profiles: one column per corrupted ratio, one row per
    architecture (``grid``), or one row with the architectures overlaid
    (``row`` / ``merge_arch``).

    Architectures are never averaged together: ResNet-34 is probed at four
    depths and the transformers at thirteen. With both tasks present the task
    is carried by the marker fill: filled for classification, hollow for
    segmentation.
    """
    merge_arch = merge_arch or layout == "row"
    plt = ac.setup_style()
    p = profile(df, agg)
    if normalize == "shallow":
        p = normalize_shallow(p)
    archs = [a for a in aio.ARCH_ORDER if (p.arch == a).any()]
    settings = [s for s in aio.SETTING_ORDER if (p.setting == s).any()]
    # H+SN has the features of H exactly: plotting both would draw one curve
    # on top of the other.
    drift_settings = [s for s in settings if s != "H+SN" or "H" not in settings]
    ratios = sorted(p.ratio.round(3).unique())
    tasks = [t for t in TASK_ORDER if (p.task == t).any()]

    # One probe only: no profile to draw, so a panel becomes a grouped bar
    # chart over architectures. Asked per task, not globally.
    single = int(p.groupby(["task", "arch"], observed=True)
                  .depth.nunique().max()) == 1
    rows = [None] if (merge_arch or single) else archs

    xcol = "rel_depth" if (merge_arch or single) else "depth"
    fig, axes = plt.subplots(len(rows), len(ratios),
                             figsize=(2.35 * len(ratios) + 0.9,
                                      (2.8 if single else 1.95 * len(rows) + 0.75)),
                             sharex="col" if (merge_arch and not single) else False,
                             sharey=True, squeeze=False)

    ymax = 0.0
    width = 0.8 / max(len(drift_settings), 1)
    for i, row in enumerate(rows):
        for j, ratio in enumerate(ratios):
            ax = axes[i][j]
            sub = p[p.ratio.round(3) == ratio]
            if single:
                cols = [(t, a) for t in tasks for a in archs]
                pos = np.arange(len(cols))
                for k, setting in enumerate(drift_settings):
                    vals = [float(sub[(sub.task == t) & (sub.arch == a)
                                      & (sub.setting == setting)]
                                  .leak_ratio.mean()) for t, a in cols]
                    bars = ax.bar(pos + (k - (len(drift_settings) - 1) / 2) * width,
                                  vals, width=width * 0.9,
                                  color=ac.SETTING_COLOR[str(setting)],
                                  label=str(setting))
                    if len(tasks) > 1:
                        for patch, (t, _) in zip(bars, cols):
                            patch.set_hatch(TASK_HATCH[str(t)])
                            patch.set_edgecolor("0.35")
                            patch.set_linewidth(0.4)
                    if np.isfinite(np.nanmax(vals)):
                        ymax = max(ymax, float(np.nanmax(vals)))
                ax.set_xticks(pos)
                ax.set_xticklabels(
                    [aio.ARCH_LABEL[str(a)] if len(tasks) == 1
                     else f"{aio.ARCH_LABEL[str(a)]}\n{TASK_SHORT[str(t)]}"
                     for t, a in cols], fontsize=7.5,
                    rotation=15 if len(tasks) == 1 else 0)
            else:
                for arch in (archs if row is None else [row]):
                    for tsk in tasks:
                        for setting in drift_settings:
                            s = sub[(sub.task == tsk) & (sub.arch == arch)
                                    & (sub.setting == setting)].sort_values(xcol)
                            if s.empty:
                                continue
                            color = (ARCH_COLOR[str(arch)] if merge_arch
                                     else ac.SETTING_COLOR[str(setting)])
                            lone = len(s) == 1
                            hollow = tsk == "segmentation" and len(tasks) > 1
                            step = 0.016 if xcol == "rel_depth" else 0.13
                            off = ((archs.index(arch) - (len(archs) - 1) / 2)
                                   * step if lone else 0.0)
                            ax.plot(s[xcol] + off, s.leak_ratio,
                                    marker=ac.ARCH_MARKER[str(arch)],
                                    markersize=6.0 if hollow else 4.0,
                                    markerfacecolor="white" if hollow else color,
                                    markeredgecolor=color,
                                    markeredgewidth=1.3 if hollow else 0,
                                    ls=("" if lone else
                                        (SETTING_STYLE[str(setting)] if merge_arch
                                         else "-")),
                                    color=color)
                            ymax = max(ymax, float(np.nanmax(s.leak_ratio)))

            ax.axhline(1.0, color="#c8442a", lw=0.7, ls=":", zorder=0)

            if i == 0:
                ax.set_title(f"$r = {ratio:g}$")
            if j == 0 and row is not None:
                ax.set_ylabel(aio.ARCH_LABEL[str(row)], fontsize=9)
            if i == len(rows) - 1 and not single:
                ax.set_xlabel(XLABEL if merge_arch else "layer")
            if single:
                continue
            if merge_arch:
                ax.set_xlim(0, 1.04)
                ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
                continue

            # Real indices: shade the blocks that head+last trains.
            if row is not None:
                d = sorted(sub[sub.arch == row].depth.unique())
            else:
                d = sorted(sub.depth.unique())
            if not d:
                continue
            ax.set_xlim(min(d) - 0.5, max(d) + 0.5)
            ax.set_xticks(d if len(d) <= 6 else d[::2])
            first = FT_FIRST_LAYER.get(str(row))
            if first is not None and max(d) >= first:
                ax.axvspan(first - 0.5, max(d) + 0.5, color="0.85", alpha=0.45,
                           zorder=0, linewidth=0)

    if merge_arch and not single:
        handles = [plt.Line2D([], [], color=ARCH_COLOR[str(a)],
                              marker=ac.ARCH_MARKER[str(a)], markeredgewidth=0,
                              label=aio.ARCH_LABEL[str(a)]) for a in archs]
        handles += [plt.Line2D([], [], color="0.35", ls=SETTING_STYLE[s],
                               label=("H / H+SN" if s == "H" and "H+SN" in settings
                                      else s))
                    for s in drift_settings]
    else:
        handles = [plt.Line2D([], [], color=ac.SETTING_COLOR[s],
                              marker="s" if single else "o",
                              ls="" if single else "-", markeredgewidth=0,
                              label=("H / H+SN" if s == "H" and "H+SN" in settings
                                     else s))
                   for s in drift_settings]
        if single and len(tasks) > 1:
            handles += [plt.Line2D([], [], ls="", marker="s", markersize=8,
                                   markerfacecolor="white",
                                   markeredgecolor="0.35",
                                   label=TASK_SHORT[str(t)]) for t in tasks]
    if len(tasks) > 1 and not single:
        handles += [plt.Line2D([], [], color="0.35", ls="-", marker="o",
                               markerfacecolor="0.35", markeredgecolor="0.35",
                               markeredgewidth=0, label="classification"),
                    plt.Line2D([], [], color="0.35", ls="", marker="o",
                               markersize=6, markerfacecolor="white",
                               markeredgecolor="0.35", markeredgewidth=1.3,
                               label="segmentation (head input)")]
    handles.append(plt.Line2D([], [], color="#c8442a", ls=":",
                              label=(r"shallowest probe" if normalize == "shallow"
                                     else r"$R_{\mathrm{leak}} = 1$")))
    fig.legend(handles=handles, frameon=False, ncol=len(handles),
               loc="lower center", bbox_to_anchor=(0.5, -0.02), handlelength=1.8)

    axes[0][0].set_ylim(0, max(1.1, ymax * 1.08))
    fig.supylabel((NORM_YLABEL if normalize == "shallow" else YLABEL)
                  + ("  (median over the grid)" if agg == "median" else ""),
                  fontsize=9)
    fig.tight_layout(rect=(0.015, 0.03, 1, 1))
    ac.save_fig(fig, path)


def figure_stacked(df: pd.DataFrame, path: str, agg: str = "mean",
                   normalize: str = "none") -> None:
    """Tasks down the rows, corrupted ratio across the columns, one figure.

    Both sweeps probe the full depth, so the rows share the x axis (relative
    depth) and the y axis (R_leak is dimensionless): one x label, one tick
    row, one scale. The architecture owns the colour and the marker, the
    fine-tuning setting owns the line style. The full definition of R_leak is
    left to the caption; the axis carries only the symbol.
    """
    plt = ac.setup_style()
    p = profile(df, agg)
    if normalize == "shallow":
        p = normalize_shallow(p)
    tasks = [t for t in TASK_ORDER if (p.task == t).any()]
    archs = [a for a in aio.ARCH_ORDER if (p.arch == a).any()]
    settings = [s for s in aio.SETTING_ORDER if (p.setting == s).any()]
    drift_settings = [s for s in settings if s != "H+SN" or "H" not in settings]
    ratios = sorted(p.ratio.round(3).unique())

    ysym = (r"$R_{\mathrm{leak}}/R_{\mathrm{leak}}^{\mathrm{first}}$"
            if normalize == "shallow" else r"$R_{\mathrm{leak}}$")

    fig, axes = plt.subplots(len(tasks), len(ratios),
                             figsize=(STACKED_WIDTH,
                                      STACKED_ROW_HEIGHT * len(tasks) + 0.55),
                             sharex=True, sharey=True, squeeze=False)
    ymax = 0.0
    for i, tsk in enumerate(tasks):
        for j, ratio in enumerate(ratios):
            ax = axes[i][j]
            sub = p[(p.task == tsk) & (p.ratio.round(3) == ratio)]
            for arch in archs:
                for setting in drift_settings:
                    s = sub[(sub.arch == arch) & (sub.setting == setting)
                            ].sort_values("rel_depth")
                    if s.empty:
                        continue
                    ax.plot(s.rel_depth, s.leak_ratio,
                            color=ARCH_COLOR[str(arch)],
                            ls=SETTING_STYLE[str(setting)], lw=1.1,
                            marker=ac.ARCH_MARKER[str(arch)],
                            markersize=2.8, markeredgewidth=0)
                    if np.isfinite(np.nanmax(s.leak_ratio)):
                        ymax = max(ymax, float(np.nanmax(s.leak_ratio)))
            ax.axhline(1.0, color="#c8442a", lw=0.7, ls=":", zorder=0)
            ax.grid(alpha=0.3, lw=0.4)
            ax.tick_params(labelsize=7, length=2, pad=1.5)
            if i == 0:
                ax.set_title(f"$r = {ratio:g}$", fontsize=8, pad=3)
            if j == 0:
                # The axis carries the quantity, the row label carries the
                # task: lowercase and bold, as in the other composite figures.
                ax.set_ylabel(ysym, fontsize=8, labelpad=2)
                ax.annotate(tsk, xy=(0, 0.5), xycoords="axes fraction",
                            xytext=(-ROW_LABEL_OFFSET, 0),
                            textcoords="offset points", rotation=90,
                            ha="right", va="center", fontsize=9,
                            fontweight="bold")
            if i == len(tasks) - 1 and j == len(ratios) // 2:
                ax.set_xlabel(XLABEL, fontsize=8)

    ax0 = axes[0][0]
    ax0.set_xlim(0, 1.03)
    ax0.set_xticks([0, 0.5, 1])
    top = max(1.1, ymax * 1.05)
    ax0.set_ylim(0, top)
    ax0.set_yticks([0, 1, 2] if top > 2 else [0, 0.5, 1, 1.5])

    handles = [plt.Line2D([], [], color=ARCH_COLOR[str(a)],
                          marker=ac.ARCH_MARKER[str(a)], markersize=3.5,
                          markeredgewidth=0, label=aio.ARCH_LABEL[str(a)])
               for a in archs]
    handles += [plt.Line2D([], [], color="0.35", ls=SETTING_STYLE[s],
                           label=("H / H+SN" if s == "H" and "H+SN" in settings
                                  else s))
                for s in drift_settings]
    handles.append(plt.Line2D([], [], color="#c8442a", ls=":",
                              label=(r"shallowest probe" if normalize == "shallow"
                                     else r"$R_{\mathrm{leak}} = 1$")))
    fig.legend(handles=handles, frameon=False, ncol=len(handles), fontsize=7,
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.4, w_pad=0.6)
    ac.save_fig(fig, path)


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    p = ac.base_parser("Insight 1 -- spatial propagation of a localized "
                       "corruption as a function of depth.")
    p.add_argument("--task", default="classification",
                   choices=["classification", "segmentation", "both"],
                   help="which sweep to draw. 'both' keeps the two side by "
                        "side and never averages them together; it is required "
                        "by --layout stacked. Default classification.")
    p.add_argument("--ratio", nargs="+", default=["all"],
                   help="corruption ratio(s), one column each, or 'all' "
                        "(default). The ratio is never averaged over.")
    p.add_argument("--shapes", nargs="+", default=None,
                   choices=["circular", "squared", "superpixel"],
                   help="mask shapes to include. Default: all of them.")
    p.add_argument("--ptypes", nargs="+", default=None,
                   help="corruption types. Default: all of them.")
    p.add_argument("--dataset", nargs="+", default=None,
                   help="evaluation dataset(s). Default: all of them.")
    p.add_argument("--merge_arch", action="store_true",
                   help="collapse the rows: one panel per ratio with the "
                        "architectures overlaid (colour by architecture, line "
                        "style by setting).")
    p.add_argument("--layout", default="grid",
                   choices=["grid", "row", "stacked"],
                   help="'grid': one row per architecture. 'row': one row of "
                        "panels, one per ratio, for the stack_tasks composite. "
                        "'stacked': the paper figure -- the two tasks down the "
                        "rows with shared axes; needs --task both. "
                        "Default grid.")
    p.add_argument("--normalize", default="none", choices=["none", "shallow"],
                   help="'shallow' divides every profile by its own shallowest "
                        "probe, leaving the amplification with depth. "
                        "Default none.")
    p.add_argument("--agg", default="mean", choices=["mean", "median"],
                   help="how the ratio is aggregated over the grid; 'median' "
                        "is the check that a curve is not carried by a few "
                        "cells. Default mean.")
    p.add_argument("--metric", default="cos", choices=["cos", "mse"],
                   help="feature distance: angular 1 - cos (default, body) or "
                        "MSE (appendix). MSE is compared within an "
                        "architecture only.")

    args = p.parse_args()
    ac.apply_figure_flags(args)

    if args.layout == "stacked" and args.task != "both":
        sys.exit("--layout stacked needs --task both")

    ratios = (None if [str(r).lower() for r in args.ratio] == ["all"]
              else sorted(float(r) for r in args.ratio))

    df = prepare(args.results_root, args.cache, args.refresh, args.task, ratios,
                 args.shapes, args.ptypes, args.dataset, args.metric)
    print(f"\n  {len(df):,} rows | ratios {sorted(df.ratio.round(3).unique())} | "
          f"shapes {sorted(df['shape'].dropna().unique())} | "
          f"types {sorted(df.ptype.dropna().unique())} | "
          f"{df.eval_dataset.nunique()} datasets | {df.seed.nunique()} seeds | "
          f"layers per arch {df.groupby('arch', observed=True).layer.nunique().to_dict()}")

    layout_tag = {"grid": "_merged" if args.merge_arch else "",
                  "row": "_row", "stacked": "_stacked"}[args.layout]
    stem = (STEM + f"_{args.task}"
            + ("" if ratios is None
               else "_r" + "-".join(f"{r:g}" for r in ratios))
            + layout_tag
            + ("" if args.agg == "mean" else f"_{args.agg}")
            + ("_norm" if args.normalize == "shallow" else "")
            + ("_mse" if args.metric == "mse" else ""))

    os.makedirs(args.out, exist_ok=True)
    print()
    csv_path = os.path.join(args.out, f"{stem}.csv")
    prof = profile(df, args.agg)
    (normalize_shallow(prof) if args.normalize == "shallow" else prof
     ).to_csv(csv_path, index=False)
    print(f"  [csv] {csv_path}")
    shape_path = os.path.join(args.out, f"{stem}_shape_control.csv")
    shape_control(df).to_csv(shape_path, index=False)
    print(f"  [csv] {shape_path}")

    pdf_path = os.path.join(args.out, f"{stem}.pdf")
    if args.layout == "stacked":
        figure_stacked(df, pdf_path, args.agg, args.normalize)
    else:
        figure(df, pdf_path, args.merge_arch, args.agg, args.normalize,
               args.layout)


if __name__ == "__main__":
    main()