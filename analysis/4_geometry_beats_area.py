"""
analysis/4_geometry_beats_area.py
=================================
Insight 4 -- at equal corrupted area, WHERE the corruption lands matters more
than HOW MUCH of the frame it covers.

    python -m analysis.4_geometry_beats_area --cache .cache_analysis
    python -m analysis.4_geometry_beats_area --task both            # paper figure
    python -m analysis.4_geometry_beats_area --task segmentation
    python -m analysis.4_geometry_beats_area --layout col
    python -m analysis.4_geometry_beats_area --ptypes "Gaussian Noise" \\
        --ratios 0.3 --damage abs


The comparison
--------------
The sweep applies every corruption under three mask geometries that are matched
in corrupted area to within 2% of the frame:

    ENCIRCLING (``circular``)  the outer ring is corrupted and the CENTRE of
                              the frame is left clean.  This is the geometry of
                              vignetting, of a dirty or wet lens rim, of the
                              black border of a capsule or fibroscope image.
    SCATTERED (``squared``,    the corrupted area is broken up and distributed
    ``superpixel``)            over the frame, so it overlaps the content.
                              This is the geometry of specular highlights,
                              bubbles, debris, instrument shadows and local
                              staining.

Everything else is held fixed: the comparison is made INSIDE a cell of
(architecture, fine-tuning setting, dataset, seed, corruption type, severity,
corrupted ratio), so the two geometries saw the same model, the same images and
the same nominal corrupted area. Damage is the relative drop of the task metric,
``1 - metric_corrupted / metric_clean``, so that architectures and datasets with
very different clean metrics can be pooled; ``--damage abs`` switches to the
absolute drop.


The claims being tested
-----------------------
(a) The encircling geometry is less damaging than the scattered one at the
    same corrupted area, in every task and for every architecture: the median
    damage ratio is 1.3x to 5.5x, and the encircling mask is the milder of the
    two in 67% to 100% of the matched cells.

(b) The effect is worth at least one full severity level, and in segmentation
    more than the entire severity range of the benchmark: an encircling mask at
    severity 5 does less damage than a scattered mask at severity 1, for all
    three backbones.

(b') In segmentation the sweep records the Dice drop stratified by how much the
    corrupted region overlaps the ground-truth lesion (``none`` / ``partial`` /
    ``full``). Restricting the comparison to ``none`` tests whether the
    geometry effect survives when neither mask touches the lesion. In
    classification there is no lesion mask and no such control: the claim
    there is about the frame, not about the finding.

(c) The effect has an operating range: at r = 0.9 there is almost no frame left
    for the geometry to differ over and the median damage ratio collapses to
    0.83-1.15. The saturated ratio is reported rather than averaged in.


What this insight does NOT claim
--------------------------------
That the geometry effect is visible in the feature drift for every backbone
(for ViT-B the OUT term reproduces the damage ordering in only 58-69% of cells);
that the encircling mask spares the finding (it spares the rim of the frame);
that the two scattered geometries are interchangeable (32 px ``squared`` masks
align with the ViT-B token grid and are milder for that backbone alone).


Outputs (same stem, in ``--out``)
---------------------------------
    geometry_beats_area[_<task>].tex / .csv           per-ratio table
    geometry_beats_area[_<task>]_severity.tex / .csv  severity equivalence
    geometry_beats_area[_<task>]_cells.csv            one row per matched cell
    geometry_beats_area[_<task>].pdf / .png           figure

The script asserts nothing and never exits non-zero.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import common as ac
from analysis import io as aio
HEAD_INPUT = ac.HEAD_INPUT
SHAPE_CLASS = ac.SHAPE_CLASS

STEM = "geometry_beats_area"

# A matched cell: the two geometries differ in nothing else.
CELL = ["task", "arch", "setting", "eval_dataset", "seed",
        "ratio", "severity", "ptype"]

SATURATED_RATIO = 0.9   # outside the operating range of the comparison

# Overlap strata recorded by the segmentation sweep. ``no_lesion`` is excluded:
# an image with no foreground has no Dice to drop.
OVERLAP = ["none", "partial", "full"]


# =============================================================================
# Data
# =============================================================================
def metric_name(task: str) -> str:
    """What the task metric is called on an axis or in a caption."""
    return {"classification": "accuracy", "segmentation": "Dice",
            "both": "metric (accuracy / Dice)"}[task]


def prepare(root: str, cache: str | None, refresh: bool, task: str,
            datasets: list[str] | None, settings: list[str] | None,
            ptypes: list[str] | None, ratios: list[float] | None,
            severities: list[int] | None, damage: str,
            min_clean: float = 0.0) -> pd.DataFrame:
    """Corruption sweep at the representation that feeds the head."""
    frames = aio.load_all(root, cache_dir=cache, refresh=refresh,
                          points_per_run=0, progress=True)
    df = frames["corruption"]
    if task != "both":
        df = df[df.task == task]
    df = df.copy()
    if df.empty:
        sys.exit(f"no {task} corruption runs found under the results root")

    keep = [(l == "head_input") or (HEAD_INPUT.get(str(a)) == l)
            for a, l in zip(df.arch, df.layer)]
    df = df[keep].copy()

    df["geometry"] = df["shape"].map(SHAPE_CLASS)
    df["ang_in"] = 1.0 - df.cos_in
    df["ang_out"] = 1.0 - df.cos_out
    df["damage"] = (df.metric_drop if damage == "abs"
                    else 1.0 - df.metric_corr / df.metric_clean)

    for col, sel in (("eval_dataset", datasets), ("setting", settings),
                     ("ptype", ptypes)):
        if sel:
            df = df[df[col].isin(sel)]
    if ratios:
        df = df[df.ratio.round(3).isin([round(r, 3) for r in ratios])]
    if severities:
        df = df[df.severity.isin(severities)]

    # A ratio of two small numbers is not a robustness measurement. The unit
    # dropped is the whole repetition -- (dataset, architecture, seed) across
    # every setting -- so both arms of a paired comparison stay or go together.
    if min_clean > 0:
        key = ["eval_dataset", "arch", "seed"]
        worst = df.groupby(key, observed=True).metric_clean.transform("min")
        dead = df[worst < min_clean]
        if not dead.empty:
            names = (dead[dead.metric_clean < min_clean]
                     .groupby(["eval_dataset", "arch", "setting"], observed=True)
                     .size().index)
            print(f"  [drop] {dead.groupby(key, observed=True).ngroups} "
                  f"repetition(s), {len(dead):,} rows, whose clean metric falls "
                  f"below {min_clean:g} in at least one setting: "
                  + ", ".join(f"{aio.DATASET_LABEL.get(str(d), d)}/"
                              f"{aio.ARCH_LABEL.get(str(a), a)}/{g}"
                              for (d, a, g) in names))
        df = df[worst >= min_clean]
    if df.empty:
        sys.exit("no rows left after the requested filters")
    return df


def cells(df: pd.DataFrame) -> pd.DataFrame:
    """One row per matched cell: encircling against the mean of the scattered masks.

    The scattered geometries are averaged inside the cell, so a cell contributes
    one paired observation.
    """
    values = ["damage", "ang_in", "ang_out", "eff_ratio", "metric_corr"]
    wide = (df.pivot_table(index=CELL, columns="geometry", values=values,
                           observed=True)
              .dropna())
    if wide.empty:
        sys.exit("no cell contains both an encircling and a scattered mask")
    out = pd.DataFrame(index=wide.index)
    for v in values:
        out[f"{v}_enc"] = wide[(v, "encircling")]
        out[f"{v}_scat"] = wide[(v, "scattered")]
    out["damage_ratio"] = out.damage_scat / out.damage_enc.replace(0.0, np.nan)
    out["area_gap"] = (out.eff_ratio_scat - out.eff_ratio_enc).abs()
    out = out.reset_index()

    ov = overlap_cells(df)
    if not ov.empty:
        out = out.merge(ov, on=CELL, how="left")
    return out


def overlap_cells(df: pd.DataFrame) -> pd.DataFrame:
    """The same matched cells, stratified by lesion overlap, where recorded.

    Kept in its own pivot because these columns exist only in the segmentation
    sweep: a joint ``dropna`` would silently empty every classification cell.
    """
    cols = [f"drop_{k}" for k in OVERLAP if f"drop_{k}" in df]
    if not cols:
        return pd.DataFrame()
    sub = df.dropna(subset=cols, how="all")
    if sub.empty:
        return pd.DataFrame()
    wide = sub.pivot_table(index=CELL, columns="geometry", values=cols,
                           observed=True)
    out = pd.DataFrame(index=wide.index)
    for v in cols:
        for geom, suffix in (("encircling", "enc"), ("scattered", "scat")):
            if (v, geom) in wide.columns:
                out[f"{v}_{suffix}"] = wide[(v, geom)]
    return out.reset_index()


def _summary(c: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return (c.groupby(by, observed=True)
             .apply(lambda g: pd.Series({
                 "damage_enc": g.damage_enc.mean(),
                 "damage_scat": g.damage_scat.mean(),
                 "ratio_med": float(np.nanmedian(g.damage_ratio)),
                 "win": float((g.damage_enc < g.damage_scat).mean()),
                 "n": len(g)}), include_groups=False)
             .reset_index())


def severity_equivalence(c: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """How many severity levels the geometry change is worth.

    The encircling damage at the highest severity is read back through the
    severity profile of the scattered masks, so the answer is in severity units.
    """
    rows = []
    for keys, g in c.groupby(by, observed=True):
        prof = g.groupby("severity", observed=True)[["damage_enc", "damage_scat"]].mean()
        prof = prof.sort_index()
        if len(prof) < 2:
            continue
        sev = prof.index.to_numpy(dtype=float)
        top = float(sev.max())
        d_enc = float(prof.damage_enc.iloc[-1])
        scat = prof.damage_scat.to_numpy(dtype=float)
        # np.interp clamps outside the range, which is the conservative answer.
        equiv = float(np.interp(d_enc, scat, sev))
        rows.append(dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
                    | {"severity_top": top, "damage_enc_top": d_enc,
                       "severity_equiv": equiv, "gain": top - equiv,
                       "below_mildest": bool(d_enc <= scat.min())})
    return pd.DataFrame(rows)


# =============================================================================
# Tables
# =============================================================================
def table(df: pd.DataFrame, c: pd.DataFrame, task: str, path: str) -> None:
    """Damage per geometry and per ratio, with the paired ratio and win rate."""
    per_shape = (df.groupby(["arch", "ratio", "shape"], observed=True)
                   .damage.mean().unstack("shape"))
    summ = _summary(c, ["arch", "ratio"]).set_index(["arch", "ratio"])

    rows = []
    for (arch, ratio), r in summ.iterrows():
        row = {"Backbone": aio.ARCH_LABEL[str(arch)], "$r$": f"{ratio:g}"}
        for shape in ("circular", "squared", "superpixel"):
            v = per_shape.get(shape)
            row[shape] = (f"{v.loc[(arch, ratio)]:.3f}"
                          if v is not None and (arch, ratio) in v.index else "---")
        row["scat./enc."] = f"{r.ratio_med:.2f}"
        row["win"] = f"{r.win:.2f}"
        row["$n$"] = f"{int(r.n)}"
        rows.append(row)
    out = pd.DataFrame(rows)

    metric = metric_name(task)
    ac.write_table(
        out, path,
        caption=(f"Relative {metric} damage $1 - m^c/m$ under the three mask "
                 f"geometries at matched corrupted area, for {task}. "
                 f"\\emph{{circular}} is encircling (the rim is corrupted, the centre "
                 f"is not); \\emph{{squared}} and \\emph{{superpixel}} are scattered. "
                 f"\\emph{{scat./enc.}} is the median damage ratio inside a matched "
                 f"cell and \\emph{{win}} the fraction of cells in which the encircling "
                 f"mask is the milder one. Averaged over "
                 f"{', '.join(sorted(df.ptype.dropna().unique()))}, severities "
                 rf"$\{{{', '.join(f'{v:g}' for v in sorted(df.severity.unique()))}\}}$, "
                 f"{df.eval_dataset.nunique()} dataset(s), "
                 f"{df.setting.nunique()} fine-tuning settings, "
                 f"{df.seed.nunique()} seeds."),
        label=f"tab:{STEM}_{task}",
        column_format="ll" + "r" * (len(out.columns) - 2),
        note=(rf"At $r=0.9$ the uncorrupted region is a thin strip and the "
              rf"comparison saturates: the ratio returns to $\approx 1$ and the "
              rf"win rate to chance."))


def severity_table(c: pd.DataFrame, task: str, path: str) -> None:
    """The geometry effect expressed in severity levels."""
    live = c[~np.isclose(c.ratio, SATURATED_RATIO)]
    eq = severity_equivalence(live, ["arch"])
    if eq.empty:
        return
    out = pd.DataFrame({
        "Backbone": [aio.ARCH_LABEL[str(a)] for a in eq.arch],
        "encircling, $s$": [f"{v:g}" for v in eq.severity_top],
        "damage": [f"{v:.3f}" for v in eq.damage_enc_top],
        "equivalent scattered $s$": [f"{v:.1f}" + ("$^*$" if b else "")
                                     for v, b in zip(eq.severity_equiv, eq.below_mildest)],
        "gain (levels)": [f"{v:.1f}" for v in eq.gain],
    })
    metric = metric_name(task)
    ac.write_table(
        out, path,
        caption=(f"The mask geometry expressed in severity levels, for {task}. "
                 f"The relative {metric} damage of an encircling mask at the "
                 f"highest severity is read back through the severity profile of "
                 f"the scattered masks at the same corrupted area. "
                 f"Restricted to $r \\le 0.6$, where the comparison is meaningful."),
        label=f"tab:{STEM}_{task}_severity",
        column_format="lrrrr",
        note=(r"$^*$ the encircling damage is below the mildest scattered "
              r"corruption measured, so the equivalent severity is off the grid "
              r"and the reported gain is a lower bound."))


def overlap_table(c: pd.DataFrame, path: str) -> None:
    """Segmentation only: the geometry comparison on images whose lesion is
    never touched by the mask (``none`` stratum, absolute Dice drop).

    Each arm uses its own ``none`` images, because the sweep stores the drop
    per stratum and not per image."""
    if not {"drop_none_enc", "drop_none_scat"} <= set(c.columns):
        return
    live = c[~np.isclose(c.ratio, SATURATED_RATIO)].dropna(
        subset=["drop_none_enc", "drop_none_scat"])
    if live.empty:
        return
    rows = []
    for (arch, ratio), g in live.groupby(["arch", "ratio"], observed=True):
        rows.append({
            "Backbone": aio.ARCH_LABEL[str(arch)],
            "$r$": f"{ratio:g}",
            "encircling": f"{g.drop_none_enc.mean():.3f}",
            "scattered": f"{g.drop_none_scat.mean():.3f}",
            "win": f"{(g.drop_none_enc < g.drop_none_scat).mean():.2f}",
            "$n$": f"{len(g)}",
        })
    ac.write_table(
        pd.DataFrame(rows), path,
        caption=(r"Segmentation, Dice drop on the images where the corrupted "
                 r"region does not overlap the lesion, under encircling and "
                 r"scattered masks at matched corrupted area. \emph{win} is the "
                 r"fraction of matched cells in which the encircling mask is the "
                 r"milder one. Restricted to $r \le 0.6$."),
        label=f"tab:{STEM}_segmentation_overlap",
        column_format="lrrrrr")


# =============================================================================
# Figure
# =============================================================================
DAMAGE = r"1 - m^{c}/m"
DAMAGE_YLABEL = rf"${DAMAGE}$"
RATIO_YLABEL = (rf"$\left({DAMAGE}\right)_{{\mathrm{{scat}}}}"
                rf"\,/\,\left({DAMAGE}\right)_{{\mathrm{{enc}}}}$")
# The full ratio does not fit on a 2-inch axis; in the compact layout the
# caption carries its definition and the axis the short form.
RATIO_YLABEL_SHORT = r"damage ratio scat./enc."

# Per-layout geometry of the figure, in inches and points.
#   compact : single-column float (\columnwidth of cas-dc), panels side by side
#   col     : single-column float, panels stacked (taller)
#   row     : full-width float, panels side by side
LAYOUTS = {
    "compact": dict(nrows=1, ncols=2, figsize=(3.45, 2.35),
                    gridspec={"width_ratios": [1.0, 1.0]},
                    fs=7.0, ms=4.0, ms_line=3.0, lw=1.1,
                    legend_ncol=3, short_ylabel=True),
    "col": dict(nrows=2, ncols=1, figsize=(3.6, 5.4),
                gridspec={"height_ratios": [1.25, 1.0]},
                fs=None, ms=6.0, ms_line=None, lw=1.6,
                legend_ncol=2, short_ylabel=False),
    "row": dict(nrows=1, ncols=2, figsize=(7.4, 3.3),
                gridspec={"width_ratios": [1.25, 1.0]},
                fs=None, ms=6.0, ms_line=None, lw=1.6,
                legend_ncol=None, short_ylabel=False),
}


def figure(df: pd.DataFrame, c: pd.DataFrame, task: str, path: str,
           layout: str = "compact") -> None:
    """Two panels, both read across the architectures.

    First panel: damage against severity for the two geometries, restricted to
    the ratios where the comparison is meaningful (r < 0.9), on a log axis so
    that a constant ratio between the geometries is a constant vertical offset.
    Colour is the backbone; solid is scattered, dashed is encircling.

    Second panel: the paired median damage ratio against corrupted area. The
    saturated ratio is kept, since its collapse is part of the result.

    ``compact`` (default) is the single-column form of the paper; ``col`` and
    ``row`` are the stacked and the full-width forms.
    """
    L = LAYOUTS[layout]
    plt = ac.setup_style()
    archs = [a for a in aio.ARCH_ORDER if (c.arch == a).any()]
    ratios = sorted(c.ratio.round(3).unique())
    live = c[~np.isclose(c.ratio, SATURATED_RATIO)]

    fig, axes = plt.subplots(L["nrows"], L["ncols"], figsize=L["figsize"],
                             gridspec_kw=L["gridspec"], squeeze=False)
    axes = axes.ravel()
    fs = L["fs"]
    label_kw = {"fontsize": fs} if fs else {}

    # -- first panel: the two geometries, three backbones, one axes ---------
    ax = axes[0]
    line_kw = {"markersize": L["ms_line"]} if L["ms_line"] else {}
    for arch in archs:
        g = live[live.arch == arch]
        if g.empty:
            continue
        color = ac.ARCH_COLOR[str(arch)]
        prof = (g.groupby("severity", observed=True)[["damage_enc", "damage_scat"]]
                 .mean().sort_index())
        ax.plot(prof.index, prof.damage_scat, "-", color=color, lw=L["lw"],
                marker=ac.ARCH_MARKER[str(arch)], markeredgewidth=0, **line_kw)
        ax.plot(prof.index, prof.damage_enc, "--", color=color, lw=L["lw"],
                marker=ac.ARCH_MARKER[str(arch)], markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=0.9, **line_kw)
    ax.set_yscale("log")
    ac.log_ticks(ax.yaxis)
    ax.set_xticks(sorted(live.severity.unique()))
    ax.set_xlabel("severity $s$", **label_kw)
    ax.set_ylabel(DAMAGE_YLABEL, **label_kw)
    ax.set_xlim(min(live.severity) - 0.55, max(live.severity) + 0.25)

    # -- second panel: where the effect lives, and where it dies ------------
    ax = axes[1]
    dodge = 0.022
    for k, arch in enumerate(archs):
        cc = c[c.arch == arch]
        color = ac.ARCH_COLOR[str(arch)]
        xs = [r + (k - (len(archs) - 1) / 2) * dodge for r in ratios]
        med = [float(np.nanmedian(cc[np.isclose(cc.ratio, r)].damage_ratio))
               for r in ratios]
        ax.plot(xs, med, "-", color=color, lw=L["lw"], zorder=2)
        ax.plot(xs, med, ls="", marker=ac.ARCH_MARKER[str(arch)], color=color,
                markeredgewidth=0, markersize=L["ms"], zorder=3)
    ax.axhline(1.0, color="0.3", lw=0.7, ls="--", zorder=1)
    ax.set_xticks(ratios)
    ax.set_xticklabels([f"{r:g}" for r in ratios])
    ax.set_xlim(min(ratios) - 0.12, max(ratios) + 0.12)
    ax.set_xlabel("corrupted area $r$", **label_kw)
    ax.set_ylabel(RATIO_YLABEL_SHORT if L["short_ylabel"] else RATIO_YLABEL,
                  **label_kw)

    if fs:
        for a in axes:
            a.tick_params(labelsize=fs - 0.5, length=2, pad=1.5)
            a.tick_params(which="minor", labelsize=fs - 0.5)

    hms = L["ms_line"] or 6.0
    arch_handles = [plt.Line2D([], [], color=ac.ARCH_COLOR[str(a)], lw=L["lw"],
                               marker=ac.ARCH_MARKER[str(a)], markeredgewidth=0,
                               markersize=hms, label=aio.ARCH_LABEL[str(a)])
                    for a in archs]
    geom_handles = [plt.Line2D([], [], color="0.35", ls="-", marker="o",
                               lw=L["lw"], markeredgewidth=0, markersize=hms,
                               label="scattered"),
                    plt.Line2D([], [], color="0.35", ls="--", marker="o",
                               lw=L["lw"], markerfacecolor="white",
                               markeredgecolor="0.35", markersize=hms,
                               label="encircling")]
    handles = arch_handles + geom_handles
    ncol = L["legend_ncol"] or len(handles)
    if layout == "compact":
        # Two centred rows above the panels: the backbones on the first, the
        # geometries under them. Two legends rather than one wrapped legend, so each row is
        # centred on its own instead of being aligned to a column grid.
        common = dict(loc="lower center", frameon=False, handlelength=1.8,
                      columnspacing=1.2, fontsize=fs - 0.5, borderaxespad=0.1,
                      handletextpad=0.5)
        common["loc"] = "upper center"
        fig.legend(handles=arch_handles, ncol=len(arch_handles),
                   bbox_to_anchor=(0.5, 1.0), **common)
        fig.legend(handles=geom_handles, ncol=len(geom_handles),
                   bbox_to_anchor=(0.5, 0.935), **common)
        fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=0.8)
    else:
        legend_y = -0.10 if layout == "col" else -0.08
        fig.legend(handles=handles, loc="lower center", ncol=ncol,
                   frameon=False, bbox_to_anchor=(0.5, legend_y),
                   handlelength=1.8, columnspacing=1.4, fontsize=7.5)
        fig.tight_layout()
    ac.save_fig(fig, path)


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    p = ac.base_parser("Insight 4 -- at equal corrupted area, where the "
                       "corruption lands dominates how much of the frame it covers.")
    p.add_argument("--task", default="classification",
                   choices=["classification", "segmentation", "both"],
                   help="'both' pools the two tasks on the relative damage "
                        "1 - m^c/m; --damage abs is not poolable and is refused. "
                        "Default classification.")
    p.add_argument("--layout", default="compact",
                   choices=["compact", "col", "row"],
                   help="'compact': single-column float, panels side by side "
                        "(default, the paper figure). 'col': single column, "
                        "panels stacked. 'row': full-width float.")
    p.add_argument("--damage", default="rel", choices=["rel", "abs"],
                   help="relative drop 1 - m^c/m (default) or the absolute drop.")
    p.add_argument("--dataset", nargs="+", default=None)
    p.add_argument("--settings", nargs="+", default=None, choices=aio.SETTING_ORDER)
    p.add_argument("--ptypes", nargs="+", default=None)
    p.add_argument("--ratios", nargs="+", type=float, default=None)
    p.add_argument("--severities", nargs="+", type=int, default=None)
    p.add_argument("--min_clean", type=float, default=0.0,
                   help="drop any (dataset, architecture, seed) whose clean "
                        "metric is below this in any setting. Default 0.")
    args = p.parse_args()
    ac.apply_figure_flags(args)
    if args.task == "both" and args.damage == "abs":
        sys.exit("--task both pools an accuracy drop with a Dice drop, which "
                 "is only meaningful for the relative damage; use --damage rel")

    df = prepare(args.results_root, args.cache, args.refresh, args.task,
                 args.dataset, args.settings, args.ptypes, args.ratios,
                 args.severities, args.damage, args.min_clean)
    c = cells(df)

    stem = (f"{STEM}_{args.task}" + ("" if args.damage == "rel" else "_abs")
            + ("" if not args.dataset else "_" + "-".join(sorted(args.dataset)))
            + ("" if args.min_clean <= 0 else f"_min{args.min_clean:g}")
            + ("" if args.layout == "compact" else f"_{args.layout}"))
    os.makedirs(args.out, exist_ok=True)
    print()
    cells_path = os.path.join(args.out, f"{stem}_cells.csv")
    c.to_csv(cells_path, index=False)
    print(f"  [csv] {cells_path}")
    table(df, c, args.task, os.path.join(args.out, f"{stem}.tex"))
    severity_table(c, args.task, os.path.join(args.out, f"{stem}_severity.tex"))
    if args.task == "segmentation":
        overlap_table(c, os.path.join(args.out, f"{stem}_overlap.tex"))
    figure(df, c, args.task, os.path.join(args.out, f"{stem}.pdf"),
           layout=args.layout)


if __name__ == "__main__":
    main()