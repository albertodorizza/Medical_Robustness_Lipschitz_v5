"""
LaTeX + CSV result tables, with validation of the input file
so that an incomplete or inconsistent CSV cannot be turned
into a table silently.
"""

import os
from pathlib import Path
import sys

import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT = (
    Path(os.environ.get("RESULTS_ROOT", "/home/a.dorizza/OUTPUTS_MEDICAL")) / "finetuning" / "all_results.csv"
)

OUTPUT_DIR = Path(os.environ.get("OUT_DIR", "paper_results")) / "finetuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_SEEDS = 5

# Stop instead of writing tables when validation finds problems.
STRICT = True


BACKBONES = [
    "resnet34",
    "vitb16b",
    "dinov2",
]

SETTINGS = [
    "head",
    "head+last",
    "head+SN",
    "head+last+SN",
]

SETTING_LABELS = {
    "head": "H",
    "head+last": "H+L",
    "head+SN": "H+SN",
    "head+last+SN": "H+L+SN",
}

BACKBONE_LABELS = {
    "resnet34": "ResNet-34",
    "vitb16b": "ViT-B/16",
    "dinov2": "DINOv2",
}


# ============================================================
# DATASETS
# ============================================================

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


PROBLEMS = []


def problem(message):
    PROBLEMS.append(message)


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(f"Input CSV not found: {INPUT}")

df = pd.read_csv(INPUT)

if df.empty:
    raise ValueError(f"Input CSV is empty: {INPUT}")

REQUIRED_COLUMNS = [
    "task",
    "dataset",
    "backbone",
    "freeze_mode",
    "lambda_sn",
    "test_acc",
    "test_dice",
]

missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]

if missing_columns:
    raise ValueError(
        f"Missing column(s) in {INPUT.name}: {missing_columns}"
    )

print(f"Input              : {INPUT}")
print(f"Rows loaded        : {len(df)}")


# ============================================================
# NORMALIZE SETTING
# ============================================================

def normalize_setting(row):

    freeze_mode = str(row["freeze_mode"]).strip().lower()

    try:
        lambda_sn = float(row["lambda_sn"])
    except (ValueError, TypeError):
        problem(
            f"non-numeric lambda_sn '{row['lambda_sn']}' "
            f"(row {row.name})"
        )
        lambda_sn = 0.0

    has_sn = lambda_sn > 0

    if freeze_mode == "head":
        return "head+SN" if has_sn else "head"

    if freeze_mode == "head+last":
        return "head+last+SN" if has_sn else "head+last"

    problem(f"unknown freeze_mode '{freeze_mode}' (row {row.name})")

    return freeze_mode


df["setting"] = df.apply(normalize_setting, axis=1)


# ============================================================
# NORMALIZE BACKBONE
# ============================================================

def normalize_backbone(x):

    x = str(x).strip().lower()

    if x in {"resnet", "resnet34"}:
        return "resnet34"

    if x in {"vit", "vitb16b", "vit-b/16"}:
        return "vitb16b"

    if x == "dinov2":
        return "dinov2"

    problem(f"unknown backbone '{x}'")

    return x


df["backbone"] = df["backbone"].apply(normalize_backbone)


# ============================================================
# VALIDATION
# ============================================================

def validate(dataframe, datasets, metric, task):
    """
    Check that every cell of the table is backed by the expected
    number of runs, with no duplicated seeds and no missing metric.
    """

    subset = dataframe[dataframe["dataset"].isin(datasets)]

    unexpected_task = subset[subset["task"] != task]

    if len(unexpected_task):
        problem(
            f"{len(unexpected_task)} row(s) with task != '{task}' "
            f"among the {task} datasets"
        )

    for dataset in datasets:

        for backbone in BACKBONES:

            for setting in SETTINGS:

                cell = subset[
                    (subset["dataset"] == dataset)
                    & (subset["backbone"] == backbone)
                    & (subset["setting"] == setting)
                ]

                tag = f"{dataset} / {backbone} / {setting}"

                if len(cell) == 0:
                    problem(f"MISSING cell: {tag}")
                    continue

                if len(cell) != EXPECTED_SEEDS:
                    problem(
                        f"{tag}: {len(cell)} run(s), "
                        f"expected {EXPECTED_SEEDS}"
                    )

                if "seed" in cell.columns:
                    seeds = cell["seed"].tolist()
                    if len(set(seeds)) != len(seeds):
                        problem(f"{tag}: duplicated seeds {sorted(seeds)}")

                n_nan = cell[metric].isna().sum()

                if n_nan:
                    problem(f"{tag}: {n_nan} missing {metric}")

                values = cell[metric].dropna()

                if len(values) and (values.min() < 0 or values.max() > 1):
                    problem(
                        f"{tag}: {metric} outside [0, 1] "
                        f"(min {values.min():.4f}, max {values.max():.4f})"
                    )

        # one lambda per setting, reported for the caption

    for setting in ["head+SN", "head+last+SN"]:

        lambdas = sorted(pd.unique(
            subset.loc[subset["setting"] == setting, "lambda_sn"]
        ))

        if len(lambdas) > 1:
            problem(f"{task} / {setting}: mixed lambda_sn {lambdas}")
        elif lambdas:
            print(f"lambda_sn          : {task} / {setting} = {lambdas[0]}")

    # datasets present in the CSV but not in the table
    extra = sorted(
        set(dataframe.loc[dataframe["task"] == task, "dataset"]) - set(datasets)
    )

    if extra:
        problem(f"{task}: dataset(s) in the CSV but not in the table: {extra}")


# ============================================================
# COMPUTE MEAN +/- STD
# ============================================================

def compute_results(dataframe, datasets, metric):

    results = {}

    subset = dataframe[dataframe["dataset"].isin(datasets)].copy()

    for dataset in datasets:

        dataset_df = subset[subset["dataset"] == dataset]

        results[dataset] = {}

        for backbone in BACKBONES:

            backbone_df = dataset_df[dataset_df["backbone"] == backbone]

            results[dataset][backbone] = {}

            for setting in SETTINGS:

                values = backbone_df[
                    backbone_df["setting"] == setting
                ][metric].dropna()

                if len(values) == 0:
                    mean, std = np.nan, np.nan
                else:
                    mean = values.mean()
                    std = values.std(ddof=1) if len(values) > 1 else 0.0

                results[dataset][backbone][setting] = (mean, std)

    return results


# ============================================================
# LATEX TABLE
# ============================================================

def make_latex_table(results, datasets, caption, label, output_file):

    lines = []

    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")

    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")

    lines.append(r"\begin{tabular}{l l c c c c}")
    lines.append(r"\toprule")

    lines.append(
        r"\textbf{Dataset} & "
        r"\textbf{Backbone} & "
        r"\textbf{H} & "
        r"\textbf{H+L} & "
        r"\textbf{H+SN} & "
        r"\textbf{H+L+SN} \\"
    )

    lines.append(r"\midrule")

    for dataset_idx, dataset in enumerate(datasets):

        dataset_label = DATASET_LABELS.get(dataset, dataset)

        for backbone_idx, backbone in enumerate(BACKBONES):

            row = []

            if backbone_idx == 0:
                row.append(rf"\multirow{{3}}{{*}}{{{dataset_label}}}")
            else:
                row.append("")

            row.append(BACKBONE_LABELS.get(backbone, backbone))

            for setting in SETTINGS:

                mean, std = results[dataset][backbone][setting]

                if np.isnan(mean):
                    cell = "--"
                else:
                    cell = f"{mean:.3f} $\\pm$ {std:.3f}"

                row.append(cell)

            lines.append(" & ".join(row) + r" \\")

        if dataset_idx < len(datasets) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    output_file.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved LaTeX        : {output_file}")


# ============================================================
# CSV TABLE
# ============================================================

def make_csv_table(results, datasets, output_file):

    rows = []

    for dataset in datasets:

        for backbone in BACKBONES:

            row = {
                "Dataset": DATASET_LABELS.get(dataset, dataset),
                "Backbone": BACKBONE_LABELS.get(backbone, backbone),
            }

            for setting in SETTINGS:

                mean, std = results[dataset][backbone][setting]

                row[SETTING_LABELS[setting]] = (
                    "" if np.isnan(mean) else f"{mean:.3f} ± {std:.3f}"
                )

            rows.append(row)

    pd.DataFrame(rows).to_csv(output_file, index=False)

    print(f"Saved CSV          : {output_file}")


# ============================================================
# RUN VALIDATION FIRST
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

validate(df, CLASSIFICATION_DATASETS, "test_acc", "classification")
validate(df, SEGMENTATION_DATASETS, "test_dice", "segmentation")

if PROBLEMS:

    print(f"\nPROBLEMS ({len(PROBLEMS)}):")

    for message in PROBLEMS:
        print(f"  - {message}")

    if STRICT:
        print("\nNothing was written. Set STRICT = False to build anyway.")
        sys.exit(1)

else:
    print("Validation         : OK, every cell has "
          f"{EXPECTED_SEEDS} complete runs.")


# ============================================================
# CLASSIFICATION
# ============================================================

print("\n" + "=" * 70)
print("TABLES")
print("=" * 70)

classification_results = compute_results(
    dataframe=df,
    datasets=CLASSIFICATION_DATASETS,
    metric="test_acc",
)

make_latex_table(
    results=classification_results,
    datasets=CLASSIFICATION_DATASETS,
    caption=(
        "Test classification accuracy (mean $\\pm$ standard "
        "deviation over five random seeds) for the different "
        "fine-tuning configurations."
    ),
    label="tab:classification_test_accuracy",
    output_file=OUTPUT_DIR / "classification_test_accuracy_table.tex",
)

make_csv_table(
    results=classification_results,
    datasets=CLASSIFICATION_DATASETS,
    output_file=OUTPUT_DIR / "classification_test_accuracy_table.csv",
)


# ============================================================
# SEGMENTATION
# ============================================================

segmentation_results = compute_results(
    dataframe=df,
    datasets=SEGMENTATION_DATASETS,
    metric="test_dice",
)

make_latex_table(
    results=segmentation_results,
    datasets=SEGMENTATION_DATASETS,
    caption=(
        "Test Dice score (mean $\\pm$ standard deviation "
        "over five random seeds) for the different "
        "fine-tuning configurations."
    ),
    label="tab:segmentation_test_dice",
    output_file=OUTPUT_DIR / "segmentation_test_dice_table.tex",
)

make_csv_table(
    results=segmentation_results,
    datasets=SEGMENTATION_DATASETS,
    output_file=OUTPUT_DIR / "segmentation_test_dice_table.csv",
)


# ============================================================
# PAPER TABLE (both tasks, Table 3)
# ============================================================

# Names and order used in the paper.
PAPER_DATASET_LABELS = {
    "kvasirv2": "Kvasir-v2",
    "hyperkvasir": "Hyper-Kvasir",
    "kvasir_capsule": "Kvasir-Capsule",
    "kvasirSEG": "Kvasir-SEG",
    "cvc clinic": "CVC-ClinicDB",
    "cvc colon": "CVC-ColonDB",
    "etis larib": "ETIS-Larib",
    "aircare_all": "AIRCARE",
}
PAPER_SETTINGS = ["head", "head+SN", "head+last", "head+last+SN"]


def make_paper_table(cls_results, seg_results, output_file):
    """Clean test metrics of both tasks in percent, as in the paper (needs the \\ms macro)."""
    def cell(mean, std):
        return "--" if np.isnan(mean) else rf"\ms{{{100 * mean:.1f}}}{{{100 * std:.1f}}}"

    cls_labels = [PAPER_DATASET_LABELS[d] for d in CLASSIFICATION_DATASETS]
    seg_labels = [PAPER_DATASET_LABELS[d] for d in SEGMENTATION_DATASETS]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Clean test accuracy (classification) and Dice score "
        r"(segmentation), in percent, as mean $\pm$ standard deviation over five "
        r"seeds. Configurations: H (head only), H+SN (head with spectral "
        r"regularization), H+L (head and last backbone block), H+L+SN (both).}",
        r"\label{tab:clean_metrics}",
        r"\begin{tabular}{l l ccc ccccc}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{\textbf{Classification (accuracy)}} & "
        r"\multicolumn{5}{c}{\textbf{Segmentation (Dice)}} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-10}",
        r"\textbf{Backbone} & \textbf{Config.} & " + " & ".join(cls_labels + seg_labels) + r" \\",
        r"\midrule",
    ]
    for b_idx, backbone in enumerate(BACKBONES):
        lines.append(rf"\multirow{{4}}{{*}}{{{BACKBONE_LABELS[backbone]}}}")
        for setting in PAPER_SETTINGS:
            cells = [cell(*cls_results[d][backbone][setting]) for d in CLASSIFICATION_DATASETS]
            cells += [cell(*seg_results[d][backbone][setting]) for d in SEGMENTATION_DATASETS]
            lines.append(f" & {SETTING_LABELS[setting]:<6} & " + " & ".join(cells) + r" \\")
        if b_idx < len(BACKBONES) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    output_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved LaTeX        : {output_file}")


make_paper_table(classification_results, segmentation_results,
                 OUTPUT_DIR / "clean_metrics.tex")


# ============================================================
# DONE
# ============================================================

print()
print("Done.")
print(f"Tables saved in: {OUTPUT_DIR}")
