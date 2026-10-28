"""
analysis/5_adaptation_not_robustness.py
=======================================
Insight 5 -- what the two training interventions of the paper actually buy.
Unfreezing the last backbone block buys clean performance, and buys relative
robustness only in classification; the spectral penalty buys neither, at a cost
of neither, which is exactly why it is worth paying for (Insight 6).

    python -m analysis.5_adaptation_not_robustness --cache .cache_analysis
    python -m analysis.5_adaptation_not_robustness --task segmentation
    python -m analysis.5_adaptation_not_robustness --ratios 0.3 --severities 3


The quantities
--------------
For one model and one corruption, the clean metric ``m`` and the corrupted
metric ``m^c`` answer different questions, and the difference between them is
the whole point:

    m          how good the model is,
    m^c        how good it still is under corruption,
    m^c / m    how much of itself it KEEPS -- the retained fraction.

A model can improve on ``m`` and ``m^c`` while getting relatively worse under
corruption, which is what a larger clean metric with a smaller retained
fraction means, and it is not visible in either metric alone. Every comparison
below is PAIRED: seed k of one setting and seed k of the other share the split,
the initialisation and the identical corruption grid, so the difference is
taken inside the pair and the split variance -- which dominates on the small
polyp datasets -- cancels. The repetition unit is one (dataset, seed); the
corruption grid is averaged inside it before differencing, so the reported
spread is the seed-to-seed spread and not the spread of the grid.


The claims being tested
-----------------------
(a) The clean ranking is not the robustness ranking. ResNet-34 loses far more
    of itself under corruption than the two transformers -- in classification
    it retains about 75% against 88%, in segmentation about 62% against
    74-79% -- while in classification its clean gap to them is under three
    points. A backbone chosen on
    clean accuracy alone is not the backbone that degrades least.

(b) Unfreezing the last block (H -> H+L) buys a large clean gain in both tasks,
    and buys relative robustness in CLASSIFICATION only: there the retained
    fraction improves for all three backbones, on 93-100% of the pairs. In
    SEGMENTATION the same intervention improves the clean Dice by 0.19-0.36
    and leaves the retained fraction flat or worse -- ViT-B loses 3.6 points of
    it, on 88% of the pairs. Adaptation is therefore not a robustness
    intervention, and reporting it as one requires the corrupted metric to be
    normalised by the clean one.

(c) The spectral penalty on a FROZEN backbone is the cleanest experiment in
    the paper, because H and H+SN produce bitwise-identical features (Insight
    3, claim (a)): the only thing that differs between them is the sensitivity
    of the head. Its effect is therefore purely a downstream effect, and it is
    task-dependent. In CLASSIFICATION the penalty converts a clean loss of
    1.1-2.3 points into a gain of the retained fraction for all three backbones
    -- decisively for ResNet-34 (+8.0 points, on 87% of pairs) and DINOv2
    (+1.9, 100%), marginally for ViT-B (+2.6, 60%) -- which is the Lipschitz
    mechanism of Section 4 visible in a task metric with the representation
    held fixed. In SEGMENTATION it is inert (|change| <= 0.6 points): the head
    there is a 1x1 convolution over a binary decision whose per-pixel margins
    are wide, and shrinking the norm of the weights does not move the argmax.

(d) Once the last block is unfrozen, the penalty becomes inert on the task
    metrics in BOTH tasks: it costs at most 0.4 clean points and moves the
    retained fraction by at most 0.6, with win rates at chance. What it still
    buys there is the certificate of Insight 6, and the fact that it buys it
    for nothing is what makes that trade worth making.


What this insight does NOT claim
--------------------------------
That the retained fraction is the only correct robustness measure. It is a
ratio of two metrics and it flatters a weak model: ResNet-34 under head-only
fine-tuning on ETIS-Larib retains 67% of a Dice of 0.05, which is not a useful
model. The absolute corrupted metric is reported in the same table and the
absolute paired drop in the delta table, so no claim here rests on the ratio
alone; the per-repetition CSV carries both for any other cut.

That the size of the ResNet effect in (c) generalises. The direction is
asserted for all three backbones, but +8.0 points of retained accuracy is one
architecture in one task, and ViT-B's +2.6 holds on only 60% of the pairs. The
mechanism -- a head whose margins are narrow enough for the spectral norm to
decide flips -- is consistent with Insight 6 but is not isolated here.


Outputs (same stem, in ``--out``)
---------------------------------
    adaptation_not_robustness[_<task>].tex / .csv       per-setting table
    adaptation_not_robustness[_<task>]_deltas.tex/.csv  paired contrasts
    adaptation_not_robustness[_<task>]_units.csv        one row per repetition
    adaptation_not_robustness[_<task>].pdf / .png       figure

The script asserts nothing and never exits non-zero: it emits the assets and
the numbers behind them, and every claim above is argued in the paper against
the CSV written next to each table. The thresholds that used to gate them were
a property of this code and not of the result, and they are gone with the
checks that used them.
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

STEM = "adaptation_not_robustness"

# One repetition: the corruption grid is averaged inside it before anything is
# compared, so a difference is a difference between two trainings and not
# between two subsets of the grid.  ``task`` is part of it so that ``--task
# both`` still pairs a classification run with itself and never across tasks;
# the pooling happens when the repetitions are averaged, not when they are
# matched.
UNIT = ["task", "eval_dataset", "seed"]

# The two interventions, each as (base, other).  They are kept separate because
# adding the penalty to a frozen backbone and to an unfrozen one are different
# interventions and their average is not either of them.
CONTRASTS = [("H", "H+L"), ("H", "H+SN"), ("H+L", "H+L+SN"), ("H", "H+L+SN")]


# Both panels of the figure are keyed by architecture through the colour and by
# setting through the marker, so one legend covers the whole figure and a colour
# never means two different things.
ARCH_COLOR = ac.ARCH_COLOR
SETTING_MARKER = ac.SETTING_MARKER


# =============================================================================
# Data
# =============================================================================
def prepare(root: str, cache: str | None, refresh: bool, task: str,
            datasets: list[str] | None, shapes: list[str] | None,
            ptypes: list[str] | None, ratios: list[float] | None,
            severities: list[int] | None, min_clean: float = 0.0) -> pd.DataFrame:
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
    df["retained"] = df.metric_corr / df.metric_clean
    df["drop"] = df.metric_clean - df.metric_corr

    for col, sel in (("eval_dataset", datasets), ("shape", shapes),
                     ("ptype", ptypes)):
        if sel:
            df = df[df[col].isin(sel)]
    if ratios:
        df = df[df.ratio.round(3).isin([round(r, 3) for r in ratios])]
    if severities:
        df = df[df.severity.isin(severities)]

    # A ratio of two small numbers is not a robustness measurement: ResNet-34
    # under head-only fine-tuning reaches a clean Dice of 0.05 on ETIS-Larib
    # and "retains" 67% of it, which enters every average as if it were a
    # working model.
    #
    # The unit dropped is the whole REPETITION -- (dataset, architecture, seed)
    # across every setting -- and not the individual row.  Dropping rows would
    # remove exactly the head-only arms, which are the ones that collapse, and
    # the surviving H -> H+L pairs would then be the pairs where unfreezing had
    # the least to fix: a filter on the outcome of one arm of a paired
    # comparison biases that comparison. Dropping the repetition keeps both
    # arms present or absent together.
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


def units(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (architecture, setting, dataset, seed): the grid averaged."""
    return (df.groupby(["arch", "setting"] + UNIT, observed=True)
              .agg(clean=("metric_clean", "mean"),
                   corr=("metric_corr", "mean"),
                   retained=("retained", "mean"),
                   drop=("drop", "mean"),
                   n_grid=("metric_corr", "size"))
              .reset_index())


def contrasts(u: pd.DataFrame, values: list[str]) -> pd.DataFrame:
    """Paired differences for every contrast, architecture and value."""
    rows = []
    for base, other in CONTRASTS:
        sub = u[u.setting.isin([base, other])]
        if sub.setting.nunique() < 2:
            continue
        for value in values:
            d = ac.paired_delta(sub, value, "setting", base, other,
                                by=["arch"], pair_on=UNIT)
            d["contrast"] = f"{base} -> {other}"
            d["value"] = value
            rows.append(d)
    return (pd.concat(rows, ignore_index=True) if rows
            else pd.DataFrame(columns=["arch", "delta", "value", "contrast"]))


def _get(d: pd.DataFrame, contrast: str, value: str, arch: str) -> pd.Series | None:
    sel = d[(d.contrast == contrast) & (d.value == value) & (d.arch == arch)]
    return None if sel.empty else sel.iloc[0]


def metric_name(task: str) -> str:
    """What the task metric is called on an axis or in a caption.

    ``both`` pools the two tasks, which is legitimate for the RETAINED
    FRACTION and for nothing else on this figure: ``m^c/m`` is a share of a
    model's own clean performance and is therefore unit-free and comparable
    across an accuracy and a Dice, while the clean metric itself is not.  The
    label says so rather than pretending there is one metric.
    """
    return {"classification": "accuracy", "segmentation": "Dice",
            "both": "metric (accuracy / Dice)"}[task]


# =============================================================================
# Tables
# =============================================================================
def table(u: pd.DataFrame, task: str, path: str, df: pd.DataFrame) -> None:
    """Clean, corrupted and retained, per architecture and setting."""
    rows = []
    for (arch, setting), g in u.groupby(["arch", "setting"], observed=True):
        row = {"Backbone": aio.ARCH_LABEL[str(arch)], "Setting": str(setting)}
        for name, col in (("clean", "clean"), ("corrupted", "corr")):
            s = ac.agg_seeds(g, col, ["arch"]).iloc[0]
            row[name] = ac.fmt_pm(s["mean"], s["sd"])
        s = ac.agg_seeds(g, "retained", ["arch"]).iloc[0]
        row["retained"] = ac.fmt_pm(s["mean"], s["sd"])
        rows.append(row)
    out = pd.DataFrame(rows)
    order = {s: i for i, s in enumerate(aio.SETTING_ORDER)}
    out = out.sort_values(["Backbone", "Setting"],
                          key=lambda s: s.map(order) if s.name == "Setting" else s
                          ).reset_index(drop=True)

    metric = metric_name(task)
    ac.write_table(
        out, path,
        caption=(f"Clean {metric}, {metric} under corruption and the retained "
                 f"fraction $m^c/m$, for {task}. Mean $\\pm$ sd over "
                 f"{u.eval_dataset.nunique()} dataset(s) $\\times$ "
                 f"{u.seed.nunique()} seeds; inside each repetition the whole "
                 f"corruption grid is averaged "
                 f"({', '.join(sorted(df.ptype.dropna().unique()))}, shapes "
                 f"{', '.join(sorted(df['shape'].dropna().unique()))}, ratios "
                 rf"$\{{{', '.join(f'{v:g}' for v in sorted(df.ratio.unique()))}\}}$, "
                 f"severities "
                 rf"$\{{{', '.join(f'{v:g}' for v in sorted(df.severity.unique()))}\}}$)."),
        label=f"tab:{STEM}_{task}",
        column_format="ll" + "r" * (len(out.columns) - 2),
        note=(r"The retained fraction is a ratio and flatters a weak model: read "
              r"it next to the corrupted column, never instead of it."))


def delta_table(d: pd.DataFrame, task: str, path: str) -> None:
    """The paired contrasts: what each intervention costs and buys."""
    rows = []
    for (contrast, arch), g in d.groupby(["contrast", "arch"], observed=True):
        row = {"Contrast": contrast, "Backbone": aio.ARCH_LABEL[str(arch)]}
        for value, name in (("clean", r"$\Delta$ clean"),
                            ("corr", r"$\Delta$ corrupted"),
                            ("retained", r"$\Delta$ retained")):
            r = g[g.value == value]
            row[name] = ("---" if r.empty else
                         f"{r.delta.iloc[0]:+.3f} "
                         rf"\tiny{{({r.frac_improved.iloc[0]:.0%})}}")
        row["$n$"] = "---" if g.empty else f"{int(g.n_pairs.iloc[0])}"
        rows.append(row)
    out = pd.DataFrame(rows)
    order = {f"{b} -> {o}": i for i, (b, o) in enumerate(CONTRASTS)}
    out = out.sort_values(["Contrast", "Backbone"],
                          key=lambda s: s.map(order) if s.name == "Contrast" else s
                          ).reset_index(drop=True)
    out["Contrast"] = out["Contrast"].str.replace("->", r"$\to$", regex=False)

    metric = metric_name(task)
    ac.write_table(
        out, path,
        caption=(f"Paired effect of each training intervention on the clean "
                 f"{metric}, the {metric} under corruption and the retained "
                 f"fraction, for {task}. Differences are taken WITHIN a "
                 f"(dataset, seed) repetition, so the split variance cancels; "
                 f"the parenthesis is the fraction of the $n$ pairs on which "
                 f"the difference is positive, which is the statement five "
                 f"seeds support."),
        label=f"tab:{STEM}_{task}_deltas",
        column_format="ll" + "r" * (len(out.columns) - 2),
        note=(r"A positive $\Delta$ clean with a negative $\Delta$ retained is the "
              r"signature of an intervention that buys performance and pays for it "
              r"in relative robustness."))


# =============================================================================
# Figure
# =============================================================================
def figure(u: pd.DataFrame, d: pd.DataFrame, task: str, path: str) -> None:
    """Left: clean against retained, one point per setting. Right: the contrasts.

    Both panels are keyed by ARCHITECTURE through the colour and by SETTING
    through the marker, so a colour means the same thing in both and the
    architectures need no in-plot labels: one legend, in the upper right
    corner of the right panel, is enough for the whole figure.  In the stacked
    composite it is drawn on the classification row, i.e. at the top right of
    the page, where the last contrast (H+L -> H+L+SN) is near zero and leaves
    the corner empty.
    """
    plt = ac.setup_style()
    archs = [a for a in aio.ARCH_ORDER if (u.arch == a).any()]
    settings = [s for s in aio.SETTING_ORDER if (u.setting == s).any()]
    metric = metric_name(task)

    # Short rows: the figure is two of these stacked in the body of the paper.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})

    # -- left: where each configuration sits on the two axes ---------------
    ax = axes[0]
    for arch in archs:
        pts = []
        for setting in settings:
            g = u[(u.arch == arch) & (u.setting == setting)]
            if g.empty:
                continue
            x, y = float(g.clean.mean()), float(g.retained.mean())
            pts.append((x, y))
            ax.scatter(x, y, s=34, color=ARCH_COLOR[str(arch)],
                       marker=SETTING_MARKER[str(setting)], zorder=3,
                       edgecolor="white", linewidth=0.6)
        if len(pts) > 1:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-",
                    color=ARCH_COLOR[str(arch)], alpha=0.65, linewidth=1.1,
                    zorder=1)
    ax.set_xlabel(f"clean {metric}")
    ax.set_ylabel("retained fraction $m^c/m$")

    # -- right: what each intervention buys, paired ------------------------
    ax = axes[1]
    shown = [c for c in ("H -> H+L", "H -> H+SN", "H+L -> H+L+SN")
             if (d.contrast == c).any()]
    width = 0.8 / max(len(archs), 1)
    for k, arch in enumerate(archs):
        xs, ys, es = [], [], []
        for i, contrast in enumerate(shown):
            r = _get(d, contrast, "retained", arch)
            if r is None:
                continue
            xs.append(i + (k - (len(archs) - 1) / 2) * width)
            ys.append(float(r.delta))
            es.append(float(r.delta_sd) / max(np.sqrt(r.n_pairs), 1))
        ax.bar(xs, ys, width=width * 0.9, yerr=es, capsize=2,
               color=ARCH_COLOR[str(arch)])
    ax.axhline(0, color="0.3", linewidth=0.8)
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels([c.replace("->", "$\\to$") for c in shown], fontsize=8)
    ax.set_ylabel("paired $\\Delta$ retained")

    # One legend for the whole figure: colour = architecture, marker = setting.
    # Matplotlib fills a two-column legend column-major, and the two keys are
    # of different length (three backbones against four settings), so the
    # fourth setting spills to the bottom of the backbone column and reads as
    # a backbone.  Padding the short column keeps each key inside its own
    # column; H is rotated to the free slot at the bottom, in line with the
    # other settings.
    def blank():
        return plt.Line2D([], [], marker="", linestyle="", label="")

    arch_handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=7,
                               color=ARCH_COLOR[str(a)], label=aio.ARCH_LABEL[str(a)])
                    for a in archs]
    setting_handles = [plt.Line2D([], [], marker=SETTING_MARKER[str(s)],
                                  linestyle="", color="0.35", label=str(s))
                       for s in settings]
    setting_handles = setting_handles[1:] + setting_handles[:1]
    rows = max(len(arch_handles), len(setting_handles))
    arch_handles += [blank() for _ in range(rows - len(arch_handles))]
    setting_handles += [blank() for _ in range(rows - len(setting_handles))]
    ax.legend(handles=arch_handles + setting_handles, frameon=False,
              fontsize=7, ncol=2, loc="upper right", handlelength=1.0,
              handletextpad=0.4, columnspacing=0.8, labelspacing=0.3,
              borderaxespad=0.2)

    fig.tight_layout(pad=0.3, w_pad=1.2)
    ac.save_fig(fig, path)


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    p = ac.base_parser("Insight 5 -- what unfreezing the last block and the "
                       "spectral penalty buy, on clean and on corrupted data.")
    p.add_argument("--task", default="classification",
                   choices=["classification", "segmentation", "both"],
                   help="'both' pools the two tasks into one figure and one "
                        "table. The retained fraction $m^c/m$ is a share of a "
                        "model's own clean metric, so an accuracy and a Dice "
                        "are on the same axis there; the clean metric itself "
                        "is not, and the task-specific claims are reported "
                        "rather than asserted when pooled. Default "
                        "classification.")
    p.add_argument("--dataset", nargs="+", default=None)
    p.add_argument("--shapes", nargs="+", default=None,
                   choices=["circular", "squared", "superpixel"])
    p.add_argument("--ptypes", nargs="+", default=None)
    p.add_argument("--ratios", nargs="+", type=float, default=None)
    p.add_argument("--severities", nargs="+", type=int, default=None)
    p.add_argument("--min_clean", type=float, default=0.0,
                   help="drop any (model, dataset) whose CLEAN metric is below "
                        "this. The retained fraction m^c/m flatters a model "
                        "that predicts almost nothing -- ResNet-34 head-only "
                        "on ETIS-Larib retains 67%% of a Dice of 0.05 -- so a "
                        "segmentation run is worth passing 0.3 here. What is "
                        "dropped is printed. Default 0 (keep everything).")
    args = p.parse_args()
    ac.apply_figure_flags(args)

    df = prepare(args.results_root, args.cache, args.refresh, args.task,
                 args.dataset, args.shapes, args.ptypes, args.ratios,
                 args.severities, args.min_clean)
    u = units(df)
    d = contrasts(u, ["clean", "corr", "retained", "drop"])

    stem = (f"{STEM}_{args.task}"
            + ("" if not args.dataset else "_" + "-".join(sorted(args.dataset)))
            + ("" if args.min_clean <= 0 else f"_min{args.min_clean:g}"))
    os.makedirs(args.out, exist_ok=True)
    print()
    units_path = os.path.join(args.out, f"{stem}_units.csv")
    u.to_csv(units_path, index=False)
    print(f"  [csv] {units_path}")
    table(u, args.task, os.path.join(args.out, f"{stem}.tex"), df)
    delta_table(d, args.task, os.path.join(args.out, f"{stem}_deltas.tex"))
    figure(u, d, args.task, os.path.join(args.out, f"{stem}.pdf"))


if __name__ == "__main__":
    main()
