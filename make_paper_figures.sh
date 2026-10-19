"""
analysis/common.py
==================
Statistics, LaTeX emission and plot styling shared by every analysis script.

Three decisions worth stating, because they show up in every table
------------------------------------------------------------------
**Seeds are repetitions, not samples of a population.**  Five seeds is too few
for a normal-theory interval to mean much, so the spread is reported as
mean +/- sd with the number of repetitions attached, and where an interval is
genuinely needed it is a BCa-free percentile bootstrap over seeds.  A t-test on
n=5 is not reported; a paired comparison is.

**Comparisons between settings are paired and must stay paired.**  Seed k of
``H+L`` and seed k of ``H+L+SN`` share the split and the initialisation, and
every corruption sees the identical mask and noise realisation.  Differencing
within a seed removes the split variance, which is the dominant term; averaging
first and subtracting afterwards throws that away.  :func:`paired_delta` is the
only correct way to compute a delta here.

**NaN means "not measured", never zero.**  It comes from an empty region -- no
uncorrupted area left at ratio 0.9, no foreground to score.  Every aggregation
below skips them and reports how many were skipped, because a mean over 150 of
200 images is a different claim from a mean over 200.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# =============================================================================
# Definitions shared by every insight script
# =============================================================================
# The representation that feeds the head, per architecture: everything the
# Lipschitz analysis says is a statement about this layer.
HEAD_INPUT = {"resnet": "layer4", "vit": "encoder.ln", "dinov2": "backbone.norm"}

# Mask geometries, grouped by what they leave uncorrupted.  Not used by this
# figure, which averages over all of them, but used by the insights that
# compare the two regimes.
SHAPE_CLASS = {"squared": "scattered", "superpixel": "scattered",
               "circular": "encircling"}

# First probe that ``head+last`` makes trainable, as a 1-based layer index.
# Read off the code (finetuning_cls.py: ``layer4`` for ResNet-34, the last FOUR
# encoder blocks plus the final norm for the transformers) and confirmed in the
# data: every shallower probe is bit-identical between H and H+L.
FT_FIRST_LAYER = {"resnet": 4, "vit": 9, "dinov2": 9}



def at_head(df: pd.DataFrame) -> pd.DataFrame:
    """Rows measured at the representation that feeds the head."""
    keep = [(l == "head_input") or (HEAD_INPUT.get(str(a)) == l)
            for a, l in zip(df.arch, df.layer)]
    return df[keep]


def layer_depth(arch: str, layer: str) -> float:
    """1-based forward index of a hooked layer, the final norm counted last.

    Parsed from the layer name rather than taken from the order of the
    ``layer_metrics`` dict, so a file written with the hooks registered in a
    different order cannot silently reverse the x axis.

    The segmentation sweep hooks a single point -- the representation that
    feeds the head -- and writes it as ``head_input``; it is the deepest probe
    by construction, so it takes the last position and the figure degrades to
    one value per model rather than a profile.
    """
    # The segmentation wrappers nest the backbone, so the hooked names carry a
    # prefix (``backbone.layer2``, ``backbone.vit.encoder.layers...``); only the
    # last component identifies the block.
    tail = layer.rsplit(".", 1)[-1]
    deepest = 4.0 if arch.startswith("resnet") else 13.0
    if layer == "head_input":
        return deepest
    if arch.startswith("resnet"):
        return float(tail.replace("layer", ""))
    if arch.startswith("vit"):
        return deepest if tail.endswith("ln") else float(tail.split("_")[-1]) + 1
    if arch.startswith("dinov2"):
        return deepest if tail.endswith("norm") else float(tail) + 1
    raise ValueError(f"cannot infer depth for {arch!r} / {layer!r}")


class Checks:
    """PASS / FAIL / NOTE lines printed as one block; any FAIL is fatal.

    Defined here because this is the first script of the series; the insights
    that assert claims import it so the package has one verification format.
    """

    def __init__(self, title: str):
        self.title = title
        self.rows: list[tuple[str, str, str]] = []

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.rows.append((name, detail, "PASS" if ok else "FAIL"))

    def note(self, name: str, detail: str) -> None:
        """A diagnostic that reports a number without asserting anything."""
        self.rows.append((name, detail, "NOTE"))

    def report(self) -> bool:
        width = max(len(r[0]) for r in self.rows) + 2
        print("\n" + "=" * 110)
        print(f"VERIFICATION -- {self.title}")
        print("=" * 110)
        for name, detail, status in self.rows:
            print(f"  [{status}] {name:<{width}} {detail}")
        n_fail = sum(1 for r in self.rows if r[2] == "FAIL")
        n_check = sum(1 for r in self.rows if r[2] != "NOTE")
        print("-" * 110)
        print(f"  {n_check - n_fail}/{n_check} checks passed"
              + (f", {n_fail} FAILED" if n_fail else ""))
        print("=" * 110)
        return n_fail == 0


# =============================================================================
# Aggregation over seeds
# =============================================================================
def agg_seeds(df: pd.DataFrame, value: str, by: list[str],
              n_boot: int = 0, seed: int = 0) -> pd.DataFrame:
    """
    Mean, sd and repetition count of ``value`` within each group of ``by``.

    Args:
        df     : long frame, one row per seed within each group.
        value  : column to aggregate.
        by     : grouping columns.
        n_boot : if > 0, add a percentile bootstrap CI over seeds.
        seed   : RNG seed of the bootstrap.

    Returns:
        One row per group with ``mean``, ``sd``, ``n`` (seeds contributing) and
        ``n_missing`` (seeds present but NaN), plus ``lo`` / ``hi`` when
        bootstrapping.
    """
    rng = np.random.default_rng(seed)
    out = []
    for keys, g in df.groupby(by, dropna=False, observed=True):
        v = g[value].to_numpy(dtype=float)
        ok = v[~np.isnan(v)]
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        row.update({
            "mean": float(ok.mean()) if ok.size else np.nan,
            # ddof=1: these are repetitions, and the n-1 denominator is what
            # every paper means by "+/- sd over seeds".
            "sd": float(ok.std(ddof=1)) if ok.size > 1 else np.nan,
            "n": int(ok.size), "n_missing": int(v.size - ok.size),
        })
        if n_boot and ok.size > 1:
            draws = rng.choice(ok, size=(n_boot, ok.size), replace=True).mean(axis=1)
            row["lo"], row["hi"] = np.percentile(draws, [2.5, 97.5])
        out.append(row)
    return pd.DataFrame(out)


def paired_delta(df: pd.DataFrame, value: str, contrast: str,
                 base: str, other: str, by: list[str],
                 pair_on: list[str] | None = None) -> pd.DataFrame:
    """
    Within-seed difference ``other - base``, then averaged.

    Args:
        df       : long frame containing both levels of ``contrast``.
        value    : column to difference.
        contrast : column holding the two levels, e.g. ``"setting"``.
        base     : level to subtract.
        other    : level to subtract from.
        by       : grouping columns of the reported table.
        pair_on  : columns that identify a matched pair inside a group.
                   Defaults to ``["seed"]``; add the corruption axes when the
                   comparison is per (shape, ratio, severity, ptype).

    Returns:
        One row per group with the mean paired delta, its sd, the number of
        pairs, and the fraction of pairs where the delta favours ``other``.
        That last column is the non-parametric statement the paper can make
        with five seeds: "better on 5/5 pairs" is honest, "p < 0.05" is not.
    """
    pair_on = pair_on or ["seed"]
    keys = list(by) + list(pair_on)
    a = df[df[contrast] == base].set_index(keys)[value]
    b = df[df[contrast] == other].set_index(keys)[value]
    joined = pd.concat({"base": a, "other": b}, axis=1).dropna()
    joined["delta"] = joined["other"] - joined["base"]

    rows = []
    for k, g in joined.groupby(level=list(range(len(by))), observed=True):
        d = g["delta"].to_numpy(dtype=float)
        row = dict(zip(by, k if isinstance(k, tuple) else (k,)))
        row.update({
            "delta": float(d.mean()),
            "delta_sd": float(d.std(ddof=1)) if d.size > 1 else np.nan,
            "n_pairs": int(d.size),
            "frac_improved": float((d > 0).mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_pm(mean: float, sd: float, digits: int = 3, n: int | None = None) -> str:
    """``0.912 +/- 0.006`` as LaTeX, with an em dash for a missing value."""
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "---"
    txt = f"{mean:.{digits}f}"
    if sd is not None and not (isinstance(sd, float) and np.isnan(sd)):
        txt += rf" $\pm$ {sd:.{digits}f}"
    if n is not None:
        txt += rf" \tiny{{({n})}}"
    return txt


# =============================================================================
# LaTeX
# =============================================================================
def write_table(df: pd.DataFrame, path: str, caption: str, label: str,
                column_format: str | None = None, note: str | None = None,
                index: bool = False, env: str = "table",
                size: str = r"\small") -> None:
    """
    Write a booktabs table, plus the same content as CSV next to it.

    The CSV is not an afterthought: it is what makes a number in the paper
    traceable back to the frame that produced it without re-running anything.

    Args:
        df            : already formatted for display (strings are fine).
        path          : ``.tex`` output path; ``.csv`` is written alongside.
        caption       : table caption.  State here every dimension the table
                        FIXES, not only the ones it varies -- a reader cannot
                        otherwise tell whether a number is over one seed or 60.
        label         : LaTeX label.
        column_format : e.g. ``"llrrr"``; inferred when omitted.
        note          : small-print line under the table.
        index         : write the DataFrame index as a column.
        env           : ``"table"`` or ``"table*"`` (full width).
        size          : font size (and spacing) commands for the table body.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    df.to_csv(path.replace(".tex", ".csv"), index=index)

    ncols = len(df.columns) + (1 if index else 0)
    column_format = column_format or ("l" * min(2, ncols) + "r" * max(0, ncols - 2))

    body = df.to_latex(index=index, escape=False, column_format=column_format,
                       na_rep="---")
    body = body.replace(r"\begin{tabular}", r"\begin{tabular}")
    lines = [rf"\begin{{{env}}}[t]", r"\centering", size,
             rf"\caption{{{caption}}}", rf"\label{{{label}}}", body]
    if note:
        lines.append(rf"\vspace{{2pt}}{{\footnotesize {note}}}")
    lines.append(rf"\end{{{env}}}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  [tex] {path}")


# =============================================================================
# Plotting
# =============================================================================
# One palette for the whole package.  Architecture owns the COLOUR and
# fine-tuning setting owns the MARKER in every figure that shows both, so a
# colour means the same thing in Insight 1 and in Insight 6 and the reader
# carries one key between figures.  ``analysis/finetuning/generate_figures.py``
# imports these too.
ARCH_COLOR = {"resnet": "#84B4D5", "resnet34": "#84B4D5",
              "vit": "#FFB87A", "vit_b_16": "#FFB87A", "vitb16b": "#FFB87A",
              "dinov2": "#8BCA8B"}
ARCH_MARKER = {"resnet": "o", "vit": "s", "dinov2": "^"}

# The setting is a four-level ordinal (two interventions, on or off), so it is
# carried by a marker when the colour is already spent on the architecture and
# by these colours when it is not.
SETTING_COLOR = {"H": "#9aa0a6", "H+SN": "#4c8bf5",
                 "H+L": "#e8843c", "H+L+SN": "#1a7f37"}
SETTING_MARKER = {"H": "o", "H+SN": "v", "H+L": "s", "H+L+SN": "D"}


def shade(color: str, factor: float = 0.55) -> tuple:
    """Darken ``color`` towards black by ``factor``.

    The architecture palette is deliberately pale, which is right for a large
    filled area and too faint for a 0.5 pt outline or a marker edge; this gives
    the matching stroke colour instead of a second hand-picked palette that can
    drift out of step with the first.
    """
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


# =============================================================================
# Figure flags, for the stacked classification / segmentation composites
# =============================================================================
# The paper puts the two tasks in ONE figure -- classification on top,
# segmentation underneath, a single key for both -- and the two rows come from
# two runs of the same script.  Two things therefore have to be switchable from
# the command line without every figure function growing a pair of arguments:
# whether the legend is drawn at all (it must appear once, on the bottom row),
# and whether a row is labelled with the task it belongs to.
#
# They are module state read at draw time rather than parameters threaded
# through seven ``figure`` functions, because the legends of those seven live
# in different places -- some on an axes, some on the figure, some three to a
# panel -- and a flag that has to be honoured at each of those sites is a flag
# that will be missed at one of them.  ``setup_style`` patches the two legend
# entry points once instead, so a script that grows a new legend tomorrow obeys
# the flag without being told to.
LEGEND = True
ROW_LABEL = None
_PATCHED = False


def apply_figure_flags(args) -> None:
    """Read ``--no_legend`` and ``--row_label`` off a parsed argument set."""
    global LEGEND, ROW_LABEL
    LEGEND = not getattr(args, "no_legend", False)
    ROW_LABEL = getattr(args, "row_label", None) or None


def setup_style():
    """
    Matplotlib defaults for figures that survive being shrunk to one column.

    Agg, because these scripts run over SSH; vector output, because a raster
    scatter of 600k points is both huge and unreadable in print; and a font
    size chosen so a two-column figure is legible at 3.3 inches wide.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.4, "lines.markersize": 4,
        "figure.autolayout": False,
        # Type 42 keeps the text selectable and satisfies most camera-ready
        # checkers, which reject Type 3.
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    _patch_legends(plt)
    return plt


def _patch_legends(plt) -> None:
    """Make every ``ax.legend`` / ``fig.legend`` obey :data:`LEGEND`.

    Applied once per process.  The wrapper returns ``None`` when legends are
    off, which is what the un-drawn legend would be worth anyway; no script in
    the package uses the return value.
    """
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    for cls in (plt.Axes, plt.Figure):
        original = cls.legend

        def wrapper(self, *a, __original=original, **kw):
            if not LEGEND:
                return None
            return __original(self, *a, **kw)
        cls.legend = wrapper


def log_ticks(axis, subs=(1.0, 2.0, 5.0)) -> None:
    """Decades plus a couple of subdivisions, labelled as plain numbers.

    Matplotlib's default log axis labels every minor step it can fit, in
    scientific notation, so a range of one decade comes out as a row of
    overlapping ``2 x 10^-1``, ``3 x 10^-1``, ``4 x 10^-1``.  These figures
    span one or two decades at most, where the exponent carries no information
    the reader needs and the collisions do the damage.

    Args:
        axis : ``ax.xaxis`` or ``ax.yaxis``, already set to a log scale.
        subs : which subdivisions of each decade get a tick.
    """
    import matplotlib.ticker as mticker
    axis.set_major_locator(mticker.LogLocator(base=10.0, subs=subs, numticks=15))
    axis.set_minor_locator(mticker.NullLocator())
    axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))


def save_fig(fig, path: str) -> None:
    """Save as both PDF (for the paper) and PNG (to look at over SSH)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if ROW_LABEL:
        # Written on the figure rather than added by the stacker afterwards:
        # at this point the layout is known and the label can sit outside it,
        # where ``bbox_inches="tight"`` will make room for it instead of
        # overprinting the leftmost y axis.
        fig.text(-0.005, 0.5, ROW_LABEL, rotation=90, va="center", ha="right",
                 fontsize=9.5, fontweight="bold", color="0.25")
    fig.savefig(path if path.endswith(".pdf") else path + ".pdf")
    fig.savefig((path[:-4] if path.endswith(".pdf") else path) + ".png")
    print(f"  [fig] {path}")
    import matplotlib.pyplot as plt
    plt.close(fig)


# =============================================================================
# Shared CLI
# =============================================================================
def base_parser(description: str):
    """
    Argument parser shared by every analysis script.

    Every script takes the same four arguments, so they can be run
    individually while debugging and in a loop from ``make_paper_figures.sh``
    without any of them growing its own conventions.
    """
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--results_root",
                   default=os.environ.get("RESULTS_ROOT",
                                          "/home/a.dorizza/OUTPUTS_MEDICAL"),
                   help="OUTPUTS_MEDICAL root. Defaults to $RESULTS_ROOT.")
    p.add_argument("--out", default="paper_results",
                   help="Directory for the .tex / .csv / .pdf outputs.")
    p.add_argument("--cache", default=None,
                   help="Parquet cache directory. Strongly recommended: "
                        "parsing the drift files takes minutes.")
    p.add_argument("--refresh", action="store_true",
                   help="Re-read the results tree even if the cache exists.")
    p.add_argument("--appendix", action="store_true",
                   help="Also emit the fully disaggregated versions.")
    p.add_argument("--no_legend", action="store_true",
                   help="Draw the figure without its key. Used for the upper "
                        "row of a stacked classification / segmentation "
                        "composite, where the key is drawn once under the "
                        "lower row. See analysis/stack_tasks.py.")
    p.add_argument("--row_label", default=None,
                   help="Write this down the left-hand side of the figure, "
                        "e.g. 'classification'. Used to name the rows of a "
                        "stacked composite.")
    return p
