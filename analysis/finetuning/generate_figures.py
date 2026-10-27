"""
Publication figure: head spectral norm across fine-tuning strategies.

Layout
------
    rows    : task        (classification / segmentation)
    columns : backbone    (ResNet-34 / ViT-B/16 / DINOv2)
    x axis  : fine-tuning setting, ordered so that each baseline is
              immediately followed by its +SN counterpart
    lines   : datasets
    y       : head spectral norm, mean +/- std over seeds

Each row has its own horizontal dataset legend right below it, so the
panels can use the full text width; the task name stays on the y axis.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# PATHS
# ============================================================

ROOT = Path(os.environ.get("RESULTS_ROOT", "/home/a.dorizza/OUTPUTS_MEDICAL")) / "finetuning"
CSV_PATH = ROOT / "all_results.csv"

FIG_DIR = Path(os.environ.get("OUT_DIR", "paper_results")) / "finetuning"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

# Paired order: baseline immediately followed by its +SN version.
SETTING_ORDER = [
    "head",
    "head+SN",
    "head+last",
    "head+last+SN",
]

SETTING_LABELS = {
    "head": "H",
    "head+SN": "H+SN",
    "head+last": "H+L",
    "head+last+SN": "H+L+SN",
}

# Columns (in SETTING_ORDER index space) that use the SN penalty.
SN_COLUMNS = [1, 3]

BACKBONE_ORDER = [
    "resnet34",
    "vitb16b",
    "dinov2",
]

BACKBONE_LABELS = {
    "resnet34": "ResNet-34",
    "vitb16b": "ViT-B/16",
    "dinov2": "DINOv2",
}

CLASSIFICATION_DATASETS = [
    "kvasirv2",
    "hyperkvasir",
    "kvasir_capsule",
]

SEGMENTATION_DATASETS = [
    "kvasirSEG",
    "cvc clinic",
    "cvc colon",
    "etis larib",
    "aircare_all",
]

DATASET_LABELS = {
    "kvasirv2": "KvasirV2",
    "hyperkvasir": "HyperKvasir",
    "kvasir_capsule": "Kvasir-Capsule",
    "kvasirSEG": "KvasirSEG",
    "cvc clinic": "CVC-ClinicDB",
    "cvc colon": "CVC-ColonDB",
    "etis larib": "ETIS-Larib",
    "aircare_all": "AirCare",
}

# Colour-blind safe (Okabe-Ito), one colour per dataset.
DATASET_COLORS = {
    "kvasirv2": "#0072B2",
    "hyperkvasir": "#D55E00",
    "kvasir_capsule": "#009E73",
    "kvasirSEG": "#0072B2",
    "cvc clinic": "#D55E00",
    "cvc colon": "#009E73",
    "etis larib": "#CC79A7",
    "aircare_all": "#8C6D1F",
}

DATASET_MARKERS = {
    "kvasirv2": "o",
    "hyperkvasir": "s",
    "kvasir_capsule": "^",
    "kvasirSEG": "o",
    "cvc clinic": "s",
    "cvc colon": "^",
    "etis larib": "D",
    "aircare_all": "v",
}

TASKS = [
    ("classification", CLASSIFICATION_DATASETS, "Classification"),
    ("segmentation", SEGMENTATION_DATASETS, "Segmentation"),
]

# Figure geometry (inches). Full two-column text width, compact height.
FIG_WIDTH = 7.2
FIG_HEIGHT = 3.2


# ============================================================
# STYLE
# ============================================================

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8.5,
    "axes.linewidth": 0.7,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,   # editable text in the PDF
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


# ============================================================
# LOAD DATA
# ============================================================

def load_data(csv_path=CSV_PATH):

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    required_columns = [
        "dataset",
        "task",
        "backbone",
        "setting",
        "head_spectral_norm",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {csv_path.name}: {missing}"
        )

    return df


# ============================================================
# VALIDATION
# ============================================================

EXPECTED_SEEDS = 5


def validate(df, strict=True):
    """
    Sanity checks on all_results.csv before plotting.

    Verifies that every (task, dataset, backbone, setting) cell
    exists, has the expected number of seeds, no duplicated seeds,
    no missing spectral norms and a single lambda_sn per task.

    Returns the list of problems found; raises if strict.
    """

    problems = []
    notes = []

    # --- unknown labels ------------------------------------

    known_datasets = set(CLASSIFICATION_DATASETS + SEGMENTATION_DATASETS)

    unknown = sorted(set(df["dataset"]) - known_datasets)
    if unknown:
        problems.append(f"unknown dataset(s) in CSV: {unknown}")

    unknown = sorted(set(df["backbone"]) - set(BACKBONE_ORDER))
    if unknown:
        problems.append(f"unknown backbone(s) in CSV: {unknown}")

    unknown = sorted(set(df["setting"]) - set(SETTING_ORDER))
    if unknown:
        problems.append(f"unknown setting(s) in CSV: {unknown}")

    unknown = sorted(set(df["task"]) - {t for t, _, _ in TASKS})
    if unknown:
        problems.append(f"unknown task(s) in CSV: {unknown}")

    # --- completeness of every cell ------------------------

    n_cells = 0

    for task, datasets, _ in TASKS:
        for dataset in datasets:
            for backbone in BACKBONE_ORDER:
                for setting in SETTING_ORDER:

                    cell = df[
                        (df["task"] == task)
                        & (df["dataset"] == dataset)
                        & (df["backbone"] == backbone)
                        & (df["setting"] == setting)
                    ]

                    tag = f"{task}/{dataset}/{backbone}/{setting}"
                    n_cells += 1

                    if len(cell) == 0:
                        problems.append(f"MISSING cell: {tag}")
                        continue

                    if len(cell) != EXPECTED_SEEDS:
                        problems.append(
                            f"{tag}: {len(cell)} run(s), "
                            f"expected {EXPECTED_SEEDS}"
                        )

                    if "seed" in cell.columns:
                        seeds = cell["seed"].tolist()
                        if len(set(seeds)) != len(seeds):
                            problems.append(
                                f"{tag}: duplicated seeds {sorted(seeds)}"
                            )

                    n_nan = cell["head_spectral_norm"].isna().sum()
                    if n_nan:
                        problems.append(
                            f"{tag}: {n_nan} missing head_spectral_norm"
                        )

    # --- one lambda_sn per task/setting --------------------

    if "lambda_sn" in df.columns:
        for (task, setting), sub in df.groupby(["task", "setting"]):
            lambdas = sorted(sub["lambda_sn"].unique())
            if len(lambdas) > 1:
                problems.append(
                    f"{task}/{setting}: mixed lambda_sn {lambdas}"
                )
            elif setting.endswith("SN"):
                notes.append(
                    f"{task}: lambda_sn = {lambdas[0]} for {setting}"
                )

    # --- negative / implausible values ---------------------

    bad = df[df["head_spectral_norm"] < 0]
    if len(bad):
        problems.append(
            f"{len(bad)} negative head_spectral_norm value(s)"
        )

    # --- report --------------------------------------------

    print(f"Rows read          : {len(df)}")
    print(f"Cells checked      : {n_cells}")
    print(f"Runs expected      : {n_cells * EXPECTED_SEEDS}")

    for note in notes:
        print(f"  note: {note}")

    if problems:
        print(f"\nPROBLEMS ({len(problems)}):")
        for problem in problems:
            print(f"  - {problem}")
        if strict:
            raise ValueError(
                f"Validation failed with {len(problems)} problem(s). "
                "Set strict=False to plot anyway."
            )
    else:
        print("Validation          : OK, all cells complete.")

    return problems


# ============================================================
# AGGREGATION
# ============================================================

def mean_std(df, task, dataset, backbone, setting):
    """Mean and sample std (ddof=1) of the head spectral norm."""

    values = df[
        (df["task"] == task)
        & (df["dataset"] == dataset)
        & (df["backbone"] == backbone)
        & (df["setting"] == setting)
    ]["head_spectral_norm"].dropna()

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        return float(values.iloc[0]), 0.0

    return float(values.mean()), float(values.std(ddof=1))


# ============================================================
# LEGEND
# ============================================================

def dataset_handle(dataset):
    return Line2D(
        [], [],
        color=DATASET_COLORS[dataset],
        marker=DATASET_MARKERS[dataset],
        markersize=3.6,
        markeredgecolor="white",
        markeredgewidth=0.5,
        linewidth=1.2,
        label=DATASET_LABELS[dataset],
    )


def add_row_legend(fig, axes_row, datasets, gap_in):
    """
    Horizontal dataset legend centred under one row of panels.

    gap_in is the distance, in inches, between the bottom of the panels
    and the top of the legend: larger on the last row, where the x tick
    labels sit in between.
    """

    left = axes_row[0].get_position().x0
    right = axes_row[-1].get_position().x1
    bottom = axes_row[0].get_position().y0

    return fig.legend(
        handles=[dataset_handle(d) for d in datasets],
        loc="upper center",
        bbox_to_anchor=(0.5 * (left + right), bottom - gap_in / FIG_HEIGHT),
        ncol=len(datasets),
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.4,
        borderaxespad=0.0,
        borderpad=0.0,
    )


# ============================================================
# FIGURE
# ============================================================

def make_spectral_norm_figure(df):

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    x = np.arange(len(SETTING_ORDER))

    for row, (task, datasets, task_label) in enumerate(TASKS):

        # ----------------------------------------------------
        # Shared y range within the row
        # ----------------------------------------------------

        row_hi = 0.0

        for backbone in BACKBONE_ORDER:
            for dataset in datasets:
                for setting in SETTING_ORDER:
                    m, s = mean_std(
                        df, task, dataset, backbone, setting
                    )
                    if not np.isnan(m):
                        row_hi = max(row_hi, m + (0.0 if np.isnan(s) else s))

        y_upper = row_hi * 1.06
        y_lower = 0.0

        for col, backbone in enumerate(BACKBONE_ORDER):

            ax = axes[row, col]

            # ------------------------------------------------
            # Light band behind the SN columns
            # ------------------------------------------------

            for c in SN_COLUMNS:
                ax.axvspan(
                    c - 0.5,
                    c + 0.5,
                    color="#000000",
                    alpha=0.045,
                    linewidth=0,
                    zorder=0,
                )

            # ------------------------------------------------
            # One line per dataset
            # ------------------------------------------------

            for dataset in datasets:

                means, stds = [], []

                for setting in SETTING_ORDER:
                    m, s = mean_std(
                        df, task, dataset, backbone, setting
                    )
                    means.append(m)
                    stds.append(s)

                means = np.asarray(means, dtype=float)
                stds = np.asarray(stds, dtype=float)

                ax.fill_between(
                    x,
                    means - stds,
                    means + stds,
                    color=DATASET_COLORS[dataset],
                    alpha=0.16,
                    linewidth=0,
                    zorder=2,
                )

                ax.plot(
                    x,
                    means,
                    color=DATASET_COLORS[dataset],
                    marker=DATASET_MARKERS[dataset],
                    markersize=3.6,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    linewidth=1.2,
                    zorder=3,
                    label=DATASET_LABELS[dataset],
                )

            # ------------------------------------------------
            # Axes cosmetics
            # ------------------------------------------------

            ax.set_xlim(-0.5, len(SETTING_ORDER) - 0.5)
            ax.set_ylim(y_lower, y_upper)

            ax.set_xticks(x)
            ax.set_xticklabels([SETTING_LABELS[s] for s in SETTING_ORDER])
            ax.tick_params(axis="x", pad=1.5)
            ax.tick_params(axis="y", pad=1.5)

            ax.grid(axis="y", color="0.88", linewidth=0.5, zorder=0)
            ax.set_axisbelow(True)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row == 0:
                ax.set_title(
                    BACKBONE_LABELS[backbone],
                    #fontweight="bold",
                    pad=3,
                )

            if col == 0:
                # Two-line y label: task name (bold) over the quantity.
                # Replaces the separate rotated row label and saves the
                # left margin it needed.
                ax.set_ylabel(
                    r"$\bf{" + task_label + r"}$" + "\n" + r"head $\sigma$",
                    linespacing=1.3,
                    labelpad=3,
                )

            # x tick labels only under the bottom row (sharex hides the
            # top ones), so the gap between rows can be small.
            if row < len(TASKS) - 1:
                ax.tick_params(axis="x", length=0)

    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        bottom=0.16,
        top=0.93,
        wspace=0.07,
        hspace=0.32,
    )
    fig.align_ylabels(axes[:, 0])

    # One legend per row, below it. The first one sits in the gap between
    # the rows (no tick labels there); the last one has to clear the x tick
    # labels.
    for row, (_, datasets, _) in enumerate(TASKS):
        last = row == len(TASKS) - 1
        add_row_legend(
            fig,
            axes[row],
            datasets,
            gap_in=0.22 if last else 0.05,
        )

    # ========================================================
    # Save
    # ========================================================

    for ext, kwargs in [
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 400}),
    ]:
        path = FIG_DIR / f"head_spectral_norm.{ext}"
        fig.savefig(path, **kwargs)
        print(f"  saved {path}")

    plt.close(fig)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    data = load_data()

    print("=" * 60)
    print("VALIDATION")
    print("=" * 60)

    validate(data, strict=True)

    print("\nSaving figures:")
    make_spectral_norm_figure(data)
    print("Done.")