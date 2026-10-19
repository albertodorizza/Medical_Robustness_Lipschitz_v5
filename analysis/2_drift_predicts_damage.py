"""
analysis/2_drift_predicts_damage.py
===================================
Which part of the feature drift, inside or outside the corrupted region,
predicts the task damage.

    python -m analysis.2_drift_predicts_damage --task both --ratio all
    python -m analysis.2_drift_predicts_damage --task both --ratio all --metric mse

Inside one model cell (architecture, dataset, setting, seed, ratio) the
corruption grid (shape x severity x type) is swept and the Spearman rank
correlation r_s is taken between the drift of each region and the damage.
The ratio is never pooled. Comparisons between regions are paired in the cell.

In segmentation the Dice drop can be restricted to an overlap stratum between
the corruption mask and the lesion (``none`` / ``partial`` / ``full``). The
figure uses ``--seg_stratum`` (default ``none``: the lesion is never touched).
The feature drift is not stratified, because the sweep stores it per image
set only.

Outputs (in --out): <stem>.csv, <stem>_cells.csv, <stem>.tex (overall),
<stem>_overlap.tex (segmentation strata), <stem>.pdf/.png, and a VERIFICATION
block. The script exits non-zero if a check fails.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import common as ac
from analysis import io as aio

# HEAD_INPUT and SHAPE_CLASS must agree with Insight 1 -- a different choice of
# "the layer that feeds the head" would make the two insights talk about
# different representations -- and Checks is the verification block both print.
# Imported rather than copied so there is one definition of each.
Checks = ac.Checks
HEAD_INPUT = ac.HEAD_INPUT
SHAPE_CLASS = ac.SHAPE_CLASS

STEM = "drift_predicts_damage"

# A cell is one model at one corruption ratio; the grid swept inside it is
# (shape x severity x perturbation type).
CELL = ["task", "arch", "eval_dataset", "setting", "ratio"]
PAIR_ON = ["eval_dataset", "setting", "seed"]
REGIONS = ("in", "out", "all")

# Segmentation strata, in the order of decreasing overlap between the
# corruption mask and the lesion.
STRATA = ["none", "partial", "full"]

# The strong form of claim (a) is about the whole corruption grid, geometry
# included, so it is asserted only when every geometry is present.
SHAPES_FULL = frozenset(SHAPE_CLASS)

# Thresholds of the claims, stated here rather than buried in the checks.
WIN_STRONG = 0.90      # (a) per architecture, over the full grid
WIN_BREAKDOWN = 0.80   # (a) holding within each dataset and each setting
WIN_WITHIN_SHAPE = 0.60  # (a) at fixed mask geometry, where the advantage is smaller
WIN_CLEAR = 0.65       # (c) the 'none' stratum, whose correlations are noisier
CHANCE = 0.50
OUT_MAJORITY = 0.50    # share of the frame left uncorrupted for (a) to be asserted
MIN_CELLS = 30         # below this a stratum is not measurable at that ratio
TIE_BAND = 0.10        # |win - 1/2| inside this band is a tie, for the 'partial' stratum


# Feature distance used for the drift: "cos" -> 1 - cos, "mse" -> MSE.
METRIC = "cos"


def drift(g: pd.DataFrame, region: str) -> np.ndarray:
    """Drift of one region under the selected feature distance."""
    if METRIC == "mse":
        return g[f"mse_{region}"].to_numpy(dtype=float)
    return (1.0 - g[f"cos_{region}"]).to_numpy(dtype=float)


# =============================================================================
# Correlations inside a cell
# =============================================================================
def spearman(x: np.ndarray, y: np.ndarray, min_points: int) -> float:
    """Spearman rank correlation r_s, or NaN when the sweep cannot support one.

    NaN rather than 0 when a region is empty, a stratum holds too few images or
    either variable is constant: 0 would read as "no relation measured here",
    which is a claim, and this is the absence of a measurement.
    """
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < min_points:
        return float("nan")
    if np.unique(x[m]).size < 3 or np.unique(y[m]).size < 3:
        return float("nan")
    r = stats.spearmanr(x[m], y[m]).statistic
    return float(r) if np.isfinite(r) else float("nan")


def cells(df: pd.DataFrame, outcome: str, stratum: str, min_points: int,
          count_col: str | None = None, min_images: int = 0,
          by_shape: bool = False) -> pd.DataFrame:
    """One row per model cell: Spearman r_s of each region against ``outcome``.

    Args:
        df         : corruption rows, already restricted to the layer that
                     feeds the head.
        outcome    : the damage column, positive meaning "worse".
        stratum    : name recorded in the output, e.g. ``"overall"``.
        min_points : grid points a correlation needs before it is computed.
        count_col  : image count backing ``outcome``, for the stratified Dice
                     drops.  Grid points measured on fewer than ``min_images``
                     images are dropped before correlating, because a Dice drop
                     averaged over two images is mostly sampling noise.
        min_images : that threshold.
        by_shape   : split the cell by mask geometry as well, leaving only
                     severity and perturbation type in the swept grid.  This is
                     the harder version of the comparison: with the geometry
                     axis removed the two terms are much closer, and reporting
                     only the pooled number would hide that the geometry is
                     where most of the advantage comes from.

    The corruption ratio is always a cell axis, never part of the swept grid.
    Pooling ratios would let the correlation be carried by "more area
    corrupted, more damage" rather than by the drift, which is a different and
    much easier claim.
    """
    keys = CELL + (["shape"] if by_shape else []) + ["seed"]
    rows = []
    for k, g in df.groupby(keys, observed=True):
        if count_col is not None:
            g = g[g[count_col].fillna(0).astype(float) >= min_images]
        row = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        row.update({"stratum": stratum, "n_points": int(len(g)),
                    "out_share": float(1.0 - g.eff_ratio.mean()) if len(g) else np.nan})
        if not by_shape:
            row["shape"] = "all"
        y = g[outcome].to_numpy(dtype=float)
        for reg in REGIONS:
            row[f"rs_{reg}"] = spearman(drift(g, reg), y, min_points)
        rows.append(row)
    out = pd.DataFrame(rows)
    # A cell with no rs_in or no rs_out cannot take part in a paired
    # comparison, and keeping it would make the denominators of the mean and of
    # the win rate disagree.
    return out.dropna(subset=["rs_in", "rs_out"]).reset_index(drop=True)


def to_long(cell_df: pd.DataFrame) -> pd.DataFrame:
    """Cell frame -> one row per (cell, region), the shape ``paired_delta`` wants."""
    ids = [c for c in cell_df.columns if not c.startswith("rs_")]
    long = cell_df.melt(id_vars=ids, value_vars=[f"rs_{r}" for r in REGIONS],
                        var_name="region", value_name="rs")
    long["region"] = [s[len("rs_"):] for s in long["region"]]
    return long


def paired(cell_df: pd.DataFrame, by: list[str], base: str = "in",
           other: str = "out") -> pd.DataFrame:
    """Within-cell ``rs_other - rs_base``, with the fraction of cells favouring it."""
    return ac.paired_delta(to_long(cell_df), "rs", "region", base, other,
                           by=by, pair_on=PAIR_ON)


def win_rate(cell_df: pd.DataFrame, base: str = "in",
             other: str = "out") -> tuple[float, int]:
    """``(fraction of cells where other beats base, number of cells)``."""
    d = (cell_df[f"rs_{other}"] - cell_df[f"rs_{base}"]).dropna()
    if d.empty:
        return float("nan"), 0
    return float((d > 0).mean()), int(len(d))


def _means(cell_df: pd.DataFrame) -> str:
    return (f"rs_in {cell_df.rs_in.mean():.3f}, rs_out {cell_df.rs_out.mean():.3f} "
            f"(delta {(cell_df.rs_out - cell_df.rs_in).mean():+.3f})")


# =============================================================================
# Data
# =============================================================================
def prepare(root: str, cache: str | None, refresh: bool,
            ratios: list[float], shapes: list[str] | None,
            datasets: list[str] | None = None) -> pd.DataFrame:
    """Corruption sweep at the representation that feeds the head, both tasks."""
    frames = aio.load_all(root, cache_dir=cache, refresh=refresh,
                          points_per_run=0, progress=True)
    df = frames["corruption"]

    # Classification hooks every block; only the deepest one is the
    # representation the head sees, and it is the one Insight 1 reports.
    # Segmentation writes that layer directly, under the name ``head_input``.
    at_head = pd.Series([HEAD_INPUT.get(str(a)) == l for a, l in zip(df.arch, df.layer)],
                        index=df.index)
    df = df[(df.task == "segmentation") | at_head].copy()

    df = df[df.ratio.round(3).isin([round(r, 3) for r in ratios])]
    if shapes:
        df = df[df["shape"].isin(shapes)]
    if datasets:
        df = df[df.eval_dataset.isin(datasets)]
    if df.empty:
        sys.exit(f"no rows left for ratio={ratios} shape={shapes} "
                 f"dataset={datasets}")
    df["shape_class"] = df["shape"].map(SHAPE_CLASS)
    return df


def all_cells(df: pd.DataFrame, min_points: int, min_images: int) -> pd.DataFrame:
    """Every cell frame this script uses: both tasks, the Dice strata, and each
    of them once over the full grid (``shape == "all"``) and once per geometry."""
    blocks = []
    for by_shape in (False, True):
        cls = df[df.task == "classification"]
        if not cls.empty:
            blocks.append(cells(cls, "metric_drop", "overall", min_points,
                                by_shape=by_shape))
        seg = df[df.task == "segmentation"]
        if not seg.empty:
            blocks.append(cells(seg, "metric_drop", "overall", min_points,
                                by_shape=by_shape))
            for s in STRATA:
                blocks.append(cells(seg, f"drop_{s}", s, min_points,
                                    count_col=f"n_{s}", min_images=min_images,
                                    by_shape=by_shape))
    if not blocks:
        sys.exit("no corruption rows left after selecting the head-input layer")
    return pd.concat(blocks, ignore_index=True)


# =============================================================================
# Verification
# =============================================================================
def _check_setup(df: pd.DataFrame, cell_df: pd.DataFrame, ck: Checks,
                 min_images: int) -> None:
    """What the rank statistic rests on, before any claim rests on it."""
    # Within a cell the clean metric is fixed, so ranking by the absolute drop
    # and ranking by the relative drop are the same ranking.
    keys = [c for c in CELL if c != "ratio"] + ["seed"]
    grouped = df.groupby(keys, observed=True)
    n_varying = int((grouped["metric_clean"].nunique() > 1).sum())
    ck.add("the clean metric is constant inside a cell", n_varying == 0,
           f"{n_varying} of {grouped.ngroups} cells vary: the rank statistic is "
           f"therefore identical for an absolute and for a relative drop")

    overall = cell_df[cell_df.stratum == "overall"]
    grid = overall.loc[overall["shape"] == "all", "n_points"]
    fixed = overall.loc[overall["shape"] != "all", "n_points"]
    ck.note("grid swept inside a cell",
            f"{int(grid.min())}-{int(grid.max())} points (shape x severity x "
            f"perturbation), median {int(grid.median())}"
            + (f"; {int(fixed.median())} at fixed geometry" if len(fixed) else ""))

    # If the IN term correlated less well simply because it varies less over
    # the grid, the comparison would be an artefact of dynamic range.  It is a
    # control for claim (a), so it is asserted at the ratios where (a) is
    # asserted and reported at the others: at r = 0.9 the thin uncorrupted
    # strip is the more variable of the two, which is part of why the ordering
    # inverts there rather than a problem with the claim.
    for ratio, t_all in df.groupby("ratio", observed=True):
        hard = float(1.0 - t_all.eff_ratio.mean()) > OUT_MAJORITY
        for task in ("classification", "segmentation"):
            t = t_all[t_all.task == task]
            if t.empty:
                continue
            iqr = {r: float(np.subtract(*np.nanpercentile(drift(t, r), [75, 25])))
                   for r in ("in", "out")}
            detail = (f"IQR of the drift is {iqr['in']:.3f} inside the support "
                      f"and {iqr['out']:.3f} outside it")
            name = f"[r={ratio:g}, {task}] dynamic range does not separate the terms"
            (ck.add(name, iqr["in"] >= iqr["out"],
                    detail + " -- the weaker predictor is the one that varies more, so a "
                             "range artefact would work against the claim")
             if hard else ck.note(name, detail))

    seg = df[df.task == "segmentation"]
    if not seg.empty:
        n = seg[[f"n_{s}" for s in STRATA]].fillna(0)
        share = n.div(n.sum(axis=1).replace(0, np.nan), axis=0).mean()
        ck.note("image share of the overlap strata",
                ", ".join(f"{s} {share[f'n_{s}']:.0%}" for s in STRATA)
                + f"; a grid point enters a stratified correlation only with "
                  f">= {min_images} images behind it")


def _check_primary(cell_df: pd.DataFrame, task: str, ratio: float, ck: Checks,
                   hard: bool) -> None:
    """(a) OUT beats IN, per architecture and inside every dataset and setting."""
    sub = cell_df[(cell_df.task == task) & (cell_df.stratum == "overall")
                  & (cell_df["shape"] == "all")
                  & (cell_df.ratio.round(3) == round(ratio, 3))]
    if sub.empty:
        return
    tag = f"r={ratio:g}, {task}"
    label = "accuracy drop" if task == "classification" else "Dice drop"

    for arch in aio.ARCH_ORDER:
        a = sub[sub.arch == arch]
        if a.empty:
            continue
        w, n = win_rate(a)
        detail = f"{int(round(w * n))}/{n} cells ({w:.0%}); {_means(a)}"
        (ck.add(f"[{tag}] OUT drift predicts the {label} better ({aio.ARCH_LABEL[arch]})",
                w >= WIN_STRONG, detail) if hard else
         ck.note(f"[{tag}] OUT vs IN ({aio.ARCH_LABEL[arch]})", detail))

    if not hard:
        return
    for level, col in (("dataset", "eval_dataset"), ("setting", "setting")):
        worst, who, n_worst = 1.1, "", 0
        for key, g in sub.groupby(col, observed=True):
            w, n = win_rate(g)
            if n and w < worst:
                worst, who, n_worst = w, str(key), n
        ck.add(f"[{tag}] it holds within every {level}", worst >= WIN_BREAKDOWN,
               f"weakest is {aio.DATASET_LABEL.get(who, who)} at {worst:.0%} of "
               f"{n_worst} cells (threshold {WIN_BREAKDOWN:.0%})")


def _check_geometry(cell_df: pd.DataFrame, task: str, ratio: float,
                    ck: Checks) -> None:
    """(a, harder) the advantage survives at fixed mask geometry, but shrinks.

    With the geometry axis removed from the swept grid the comparison is
    between corruptions of the same shape, and the size of the drop in the win
    rate is the measurement of how much of claim (a) the geometry carries.
    """
    sub = cell_df[(cell_df.task == task) & (cell_df.stratum == "overall")
                  & (cell_df.ratio.round(3) == round(ratio, 3))]
    pooled, _ = win_rate(sub[sub["shape"] == "all"])
    shapes = sorted(sub.loc[sub["shape"] != "all", "shape"].dropna().unique())
    if not shapes:
        ck.note(f"[r={ratio:g}, {task}] the fixed-geometry version of claim (a)",
                "not computed: a single geometry leaves fewer grid points per cell than "
                "--min_points requires")
        return
    for shape in shapes:
        g = sub[sub["shape"] == shape]
        w, n = win_rate(g)
        ck.add(f"[r={ratio:g}, {task}] it survives at fixed geometry ('{shape}')",
               w >= WIN_WITHIN_SHAPE,
               f"{int(round(w * n))}/{n} cells ({w:.0%}, against {pooled:.0%} over the "
               f"full grid); {_means(g)} -- most of the advantage of the OUT term is its "
               f"sensitivity to the geometry of the mask, which the IN term barely sees")


def _check_area(cell_df: pd.DataFrame, task: str, ratios: list[float],
                ck: Checks) -> None:
    """(b) the advantage of OUT tracks the share of the frame it covers."""
    sub = cell_df[(cell_df.task == task) & (cell_df.stratum == "overall")
                  & (cell_df["shape"] == "all")]
    if sub.empty:
        return
    rows = []
    for r in ratios:
        g = sub[sub.ratio.round(3) == round(r, 3)]
        if g.empty:
            continue
        w, n = win_rate(g)
        rows.append((r, float(g.out_share.mean()), w, n))
    if not rows:
        return
    detail = "; ".join(f"r={r:g}: OUT covers {s:.0%} of the frame and wins {w:.0%} of {n}"
                       for r, s, w, n in rows)
    if len(rows) < 3:
        ck.note(f"[{task}] the advantage of OUT tracks the area it covers",
                detail + " -- asserted only when three or more ratios are analysed")
        return
    rs_area = float(stats.spearmanr([s for _, s, _, _ in rows],
                                    [w for _, _, w, _ in rows]).statistic)
    below = [f"r={r:g}" for r, _, w, _ in rows if w < CHANCE]
    ck.add(f"[{task}] the advantage of OUT tracks the area it covers",
           rs_area >= 0.999,
           detail + f" (Spearman between the two = {rs_area:+.3f})"
           + (f"; the IN term takes over at {', '.join(below)}" if below else ""))


def _check_overlap(cell_df: pd.DataFrame, ratios: list[float], ck: Checks) -> None:
    """(c) in segmentation the predictive term is the one containing the lesion."""
    seg = cell_df[(cell_df.task == "segmentation") & (cell_df["shape"] == "all")]
    per_shape = cell_df[(cell_df.task == "segmentation") & (cell_df["shape"] != "all")]
    if seg.empty:
        return
    measurable = {}
    for r in ratios:
        for s in STRATA:
            g = seg[(seg.stratum == s) & (seg.ratio.round(3) == round(r, 3))]
            w, n = win_rate(g)
            if n >= MIN_CELLS:
                measurable[(r, s)] = (w, n, g)

    for (r, s), (w, n, g) in sorted(measurable.items()):
        tag = f"r={r:g}, lesion overlap = {s}"
        if s == "none":
            ck.add(f"[{tag}] OUT drift predicts the drop on untouched lesions",
                   w >= WIN_CLEAR,
                   f"{int(round(w * n))}/{n} cells ({w:.0%}); {_means(g)} -- the damage "
                   f"to lesions the corruption never overlapped is carried by the drift "
                   f"outside the mask")
            # As in claim (a), part of this is the geometry axis; what survives
            # at fixed geometry is reported rather than asserted, because the
            # strata thin out quickly once the grid is split.
            shapes = []
            for shape in sorted(per_shape.loc[per_shape["shape"] != "all", "shape"]
                                .dropna().unique()):
                h = per_shape[(per_shape.stratum == s)
                              & (per_shape.ratio.round(3) == round(r, 3))
                              & (per_shape["shape"] == shape)]
                wh, nh = win_rate(h)
                if nh >= MIN_CELLS:
                    shapes.append(f"{shape} {wh:.0%} of {nh}")
            if shapes:
                ck.note(f"[{tag}] at fixed geometry", "; ".join(shapes))
        elif s == "full":
            ck.add(f"[{tag}] IN drift predicts the drop on fully covered lesions",
                   w <= 1.0 - WIN_CLEAR,
                   f"OUT wins only {int(round(w * n))}/{n} cells ({w:.0%}); {_means(g)}")
        else:
            ck.note(f"[{tag}] the two terms tie",
                    f"OUT wins {w:.0%} of {n} cells"
                    + (" (inside the tie band)" if abs(w - CHANCE) <= TIE_BAND else
                       f" (outside the tie band, leaning to "
                       f"{'OUT' if w > CHANCE else 'IN'})"))

    # The ordering across strata is the falsifiable part: it must be monotone
    # wherever two or three strata are measurable at the same ratio.
    for r in ratios:
        present = [s for s in STRATA if (r, s) in measurable]
        if len(present) < 2:
            if present:
                ck.note(f"[r={r:g}] overlap strata measurable",
                        f"only '{present[0]}' -- a lesion is fully covered only at a "
                        f"large ratio, and rarely left untouched at one")
            continue
        wins = [measurable[(r, s)][0] for s in present]
        ck.add(f"[r={r:g}] the win rate is monotone in the mask/lesion overlap",
               all(a > b for a, b in zip(wins, wins[1:])),
               " > ".join(f"{s} {w:.0%}" for s, w in zip(present, wins))
               + " -- which region's drift predicts the damage follows where the lesion "
                 "is, not which region is intrinsically more informative")

    for r in ratios:
        g = seg[(seg.stratum == "overall") & (seg.ratio.round(3) == round(r, 3))]
        w, n = win_rate(g)
        if n:
            ck.note(f"[r={r:g}] pooled Dice drop, for contrast",
                    f"OUT wins {w:.0%} of {n} cells; {_means(g)} -- the pooled drop is "
                    f"dominated by the 'partial' stratum and inherits its tie")


def _note_global(cell_df: pd.DataFrame, ratios: list[float], ck: Checks) -> None:
    """The whole-frame drift, which is the alternative to using the decomposition.

    Reported per ratio rather than pooled, because the answer changes with the
    ratio and a pooled number would hide that: at r = 0.3 the OUT term is the
    best of the three, and the whole-frame drift only takes over once the
    corrupted region is large enough to dominate it.
    """
    for task in ("classification", "segmentation"):
        for r in ratios:
            sub = cell_df[(cell_df.task == task) & (cell_df.stratum == "overall")
                          & (cell_df["shape"] == "all")
                          & (cell_df.ratio.round(3) == round(r, 3))]
            if sub.empty:
                continue
            w, n = win_rate(sub, base="out", other="all")
            ck.note(f"[r={r:g}, {task}] the whole-frame drift, for comparison",
                    f"rs_all {sub.rs_all.mean():.3f} against rs_out "
                    f"{sub.rs_out.mean():.3f}, higher in {int(round(w * n))}/{n} cells "
                    + ("-- the OUT term is the best of the three here" if w < CHANCE else
                       "-- pooling the frame is at least as good a global predictor here; "
                       "what the split buys is knowing where the damage came from"))


def verify(df: pd.DataFrame, cell_df: pd.DataFrame, ratios: list[float],
           min_images: int) -> bool:
    ck = Checks("Insight 2 -- which region's feature drift predicts the task damage "
                f"(within-cell Spearman; r in {[f'{r:g}' for r in ratios]}; "
                f"shapes {sorted(df['shape'].dropna().unique())})")
    _check_setup(df, cell_df, ck, min_images)

    # The strong form of claim (a) is a statement about the whole corruption
    # grid: restricting --shapes changes what is being claimed, so it is
    # reported instead of asserted rather than tested against a grid it was not
    # stated over.
    full_grid = set(df["shape"].dropna().unique()) >= set(SHAPES_FULL)
    if not full_grid:
        ck.note("mask geometries analysed",
                f"{sorted(df['shape'].dropna().unique())} of {sorted(SHAPES_FULL)} -- "
                f"claim (a) is about the full grid and is reported, not asserted, here")

    for r in ratios:
        sub = cell_df[(cell_df.stratum == "overall") & (cell_df["shape"] == "all")
                      & (cell_df.ratio.round(3) == round(r, 3))]
        share = float(sub.out_share.mean()) if not sub.empty else float("nan")
        # One decision per ratio, taken before anything is asserted: claim (a)
        # is about the term that covers most of the frame, so at a ratio where
        # the uncorrupted region is the minority it is reported, not asserted.
        hard = bool(share > OUT_MAJORITY)
        ck.note(f"[r={r:g}] uncorrupted share of the frame",
                f"{share:.0%}"
                + ("; OUT is the majority region, so claim (a) is asserted here"
                   if hard else
                   "; OUT is the minority region, so claim (a) is reported but not "
                   "asserted at this ratio"))
        for task in ("classification", "segmentation"):
            _check_primary(cell_df, task, r, ck,
                           hard and full_grid and task == "classification")
        if hard and full_grid:
            _check_geometry(cell_df, "classification", r, ck)

    for task in ("classification", "segmentation"):
        _check_area(cell_df, task, ratios, ck)
    _check_overlap(cell_df, ratios, ck)
    _note_global(cell_df, ratios, ck)
    return ck.report()


# =============================================================================
# Assets
# =============================================================================
def summary(cell_df: pd.DataFrame) -> pd.DataFrame:
    """Mean r_s per region, the paired delta and the win rate, per cell group.

    ``shape == "all"`` are the rows computed over the full corruption grid, the
    ones the tables report; the per-geometry rows are kept alongside because
    they are where most of the advantage of the OUT term turns out to live.
    """
    by = ["task", "stratum", "shape", "ratio", "arch"]
    out = None
    for reg in REGIONS:
        block = (ac.agg_seeds(cell_df, f"rs_{reg}", by)
                 .rename(columns={"mean": f"rs_{reg}", "sd": f"rs_{reg}_sd"})
                 [by + [f"rs_{reg}", f"rs_{reg}_sd"]])
        out = block if out is None else out.merge(block, on=by, how="outer")

    delta = paired(cell_df, by).rename(columns={"delta": "delta_out_in",
                                                "delta_sd": "delta_out_in_sd",
                                                "n_pairs": "n_cells",
                                                "frac_improved": "win_out_over_in"})
    out = out.merge(delta, on=by, how="outer")
    share = (cell_df.groupby(by, observed=True)["out_share"].mean()
             .rename("out_share").reset_index())
    return out.merge(share, on=by, how="left").sort_values(by).reset_index(drop=True)


def table(summ: pd.DataFrame, path: str) -> None:
    rows = []
    for _, r in summ[(summ.stratum == "overall") & (summ["shape"] == "all")].iterrows():
        rows.append({
            "Task": "cls" if r.task == "classification" else "seg",
            "$r$": f"{r.ratio:g}",
            "Backbone": aio.ARCH_LABEL.get(str(r.arch), str(r.arch)),
            r"$r_s^{\mathrm{in}}$": ac.fmt_pm(r.rs_in, r.rs_in_sd),
            r"$r_s^{\mathrm{out}}$": ac.fmt_pm(r.rs_out, r.rs_out_sd),
            r"$r_s^{\mathrm{all}}$": ac.fmt_pm(r.rs_all, r.rs_all_sd),
            r"$\Delta$": ac.fmt_pm(r.delta_out_in, r.delta_out_in_sd),
            "cells favouring out": f"{int(round(r.win_out_over_in * r.n_cells))}/{int(r.n_cells)}",
        })
    ac.write_table(
        pd.DataFrame(rows), path,
        caption=(r"Spearman rank correlation between the feature drift and the "
                 r"task damage, computed inside each model cell over the corruption grid "
                 r"(mask shape $\times$ severity $\times$ perturbation type) and averaged "
                 r"over cells. A cell is one (dataset, fine-tuning setting, seed) at the "
                 r"stated ratio $r$; $\Delta$ and the last column are paired within the "
                 r"cell. Mean $\pm$ s.d.\ over cells."),
        label="tab:drift-predicts", column_format="llrrrrrr",
        )


def overlap_table(summ: pd.DataFrame, path: str) -> None:
    sub = summ[(summ.task == "segmentation") & (summ.stratum != "overall")
               & (summ["shape"] == "all")]
    sub = sub[sub.n_cells.fillna(0) > 0]
    if sub.empty:
        return
    rows = []
    for _, r in sub.iterrows():
        rows.append({
            "$r$": f"{r.ratio:g}",
            "Mask/lesion overlap": r.stratum,
            "Backbone": aio.ARCH_LABEL.get(str(r.arch), str(r.arch)),
            r"$r_s^{\mathrm{in}}$": ac.fmt_pm(r.rs_in, r.rs_in_sd),
            r"$r_s^{\mathrm{out}}$": ac.fmt_pm(r.rs_out, r.rs_out_sd),
            r"$\Delta$": ac.fmt_pm(r.delta_out_in, r.delta_out_in_sd),
            "cells favouring out": f"{int(round(r.win_out_over_in * r.n_cells))}/{int(r.n_cells)}",
        })
    ac.write_table(
        pd.DataFrame(rows), path,
        caption=(r"Segmentation, Dice drop stratified by the overlap between the "
                 r"corruption mask and the lesion. \emph{none}: the lesion was never "
                 r"touched; \emph{full}: it lies entirely inside the corrupted region. "
                 r"Only the Dice drop is stratified, the drift is averaged over all images."),
        label="tab:drift-predicts-overlap", column_format="lllrrrr",
        note=(r"Grid points backed by too few images are dropped before correlating, and "
              r"a stratum is reported only where enough cells survive: this is why "
              r"\emph{full} appears only at large $r$ and \emph{none} mostly at small "
              r"$r$."))


def figure(cell_df: pd.DataFrame, ratios: list[float], path: str,
           tasks: list[str], seg_stratum: str = "none",
           row_label: str | None = None) -> None:
    """One row per task, one column per ratio: r_s^in against r_s^out.

    Each point is a cell; the number in the corner is the share of cells above
    the diagonal. The segmentation row uses the Dice drop of ``seg_stratum``.
    """
    plt = ac.setup_style()
    full = cell_df[cell_df["shape"] == "all"]
    stratum = {"classification": "overall", "segmentation": seg_stratum}
    rows = [t for t in tasks if not full[(full.task == t)
                                         & (full.stratum == stratum[t])].empty]
    if not rows:
        print("  [fig] nothing to draw")
        return

    fig, axes = plt.subplots(len(rows), len(ratios),
                             figsize=(2.2 * len(ratios) + 0.6, 2.3 * len(rows) + 0.4),
                             squeeze=False)
    for i, task in enumerate(rows):
        sub_t = full[(full.task == task) & (full.stratum == stratum[task])]
        for j, r in enumerate(ratios):
            ax = axes[i][j]
            sub = sub_t[sub_t.ratio.round(3) == round(r, 3)]
            if i == 0:
                ax.set_title(f"$r = {r:g}$")
            if sub.empty:
                ax.text(0.5, 0.5, "not measurable", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7, color="0.5")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            lo = float(min(sub.rs_in.min(), sub.rs_out.min())) - 0.03
            ax.plot([lo, 1.0], [lo, 1.0], color="0.55", lw=0.7, ls=":", zorder=0)
            for arch in aio.ARCH_ORDER:
                a = sub[sub.arch == arch]
                if not a.empty:
                    ax.scatter(a.rs_in, a.rs_out, s=9, linewidths=0,
                               marker=ac.ARCH_MARKER[arch],
                               color=ac.ARCH_COLOR[arch])
            w, _ = win_rate(sub)
            ax.text(0.04, 0.96, f"{w:.0%}", transform=ax.transAxes,
                    va="top", ha="left", fontsize=7.5)
            ax.set_xlim(lo, 1.0)
            ax.set_ylim(lo, 1.0)
            ax.set_aspect("equal", adjustable="box")
            if i == len(rows) - 1 and j == len(ratios) // 2:
                ax.set_xlabel(r"$r_s^{\mathrm{in}}$")
        axes[i][0].set_ylabel(r"$r_s^{\mathrm{out}}$")
        label = row_label if (row_label and len(rows) == 1) else task
        if len(rows) > 1 or row_label:
            axes[i][0].annotate(label, xy=(0, 0.5), xycoords="axes fraction",
                                xytext=(-46, 0), textcoords="offset points",
                                rotation=90, ha="right", va="center",
                                fontsize=9, fontweight="bold")

    handles = [plt.Line2D([], [], ls="", marker=ac.ARCH_MARKER[a],
                          color=ac.ARCH_COLOR[a], label=aio.ARCH_LABEL[a])
               for a in aio.ARCH_ORDER if (full.arch == a).any()]
    fig.legend(handles=handles, frameon=False, loc="lower center",
               ncol=len(handles), bbox_to_anchor=(0.5, 0.0), handlelength=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    ac.save_fig(fig, path)


# =============================================================================
def main() -> None:
    global METRIC
    p = ac.base_parser("Which region's feature drift predicts the task damage.")
    p.add_argument("--task", default="both",
                   choices=["both", "classification", "segmentation"],
                   help="rows of the figure. Tables and checks always cover "
                        "every task in the results tree.")
    p.add_argument("--seg_stratum", default="none",
                   choices=["overall", "none", "partial", "full"],
                   help="Dice drop used in the segmentation row: all images, "
                        "or only one mask/lesion overlap stratum. Default none.")
    p.add_argument("--metric", default="cos", choices=["cos", "mse"],
                   help="feature distance of the drift. Default cos.")
    p.add_argument("--ratio", nargs="+", default=["all"],
                   help="corruption ratio(s), or 'all'. Default all.")
    p.add_argument("--shapes", nargs="+", default=None,
                   choices=["circular", "squared", "superpixel"])
    p.add_argument("--dataset", nargs="+", default=None)
    p.add_argument("--min_images", type=int, default=5,
                   help="images a stratified Dice drop needs to enter a "
                        "correlation. Default 5.")
    p.add_argument("--min_points", type=int, default=12,
                   help="grid points a correlation needs. Default 12.")
    args = p.parse_args()
    ac.apply_figure_flags(args)
    METRIC = args.metric

    ratios = ([0.3, 0.6, 0.9] if [str(r).lower() for r in args.ratio] == ["all"]
              else sorted(float(r) for r in args.ratio))
    df = prepare(args.results_root, args.cache, args.refresh, ratios,
                 args.shapes, args.dataset)
    cell_df = all_cells(df, args.min_points, args.min_images)
    print(f"\n  {len(df):,} rows | ratios {[f'{r:g}' for r in ratios]} | "
          f"{df.eval_dataset.nunique()} datasets | {df.seed.nunique()} seeds | "
          f"{len(cell_df):,} cell correlations | metric {METRIC}")

    ok = verify(df, cell_df, ratios, args.min_images)

    tag = ("_r" + "-".join(f"{r:g}" for r in ratios)
           + ("" if not args.dataset else "_" + "-".join(sorted(args.dataset)))
           + ("_mse" if METRIC == "mse" else ""))
    stem_data = STEM + tag
    stem_fig = (STEM + ("" if args.task == "both" else f"_{args.task}") + tag
                + ("" if args.seg_stratum == "overall" else f"_seg-{args.seg_stratum}"))
    os.makedirs(args.out, exist_ok=True)
    print()
    summ = summary(cell_df)
    for name, frame in ((f"{stem_data}.csv", summ), (f"{stem_data}_cells.csv", cell_df)):
        path = os.path.join(args.out, name)
        frame.to_csv(path, index=False)
        print(f"  [csv] {path}")
    table(summ, os.path.join(args.out, f"{stem_data}.tex"))
    overlap_table(summ, os.path.join(args.out, f"{stem_data}_overlap.tex"))
    tasks = ["classification", "segmentation"] if args.task == "both" else [args.task]
    figure(cell_df, ratios, os.path.join(args.out, f"{stem_fig}.pdf"), tasks,
           args.seg_stratum, args.row_label)

    if not ok:
        raise SystemExit("a verification check failed: do not use these assets")


if __name__ == "__main__":
    main()
