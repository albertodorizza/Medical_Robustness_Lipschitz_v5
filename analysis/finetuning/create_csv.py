"""
Collect every run JSON into a single table (CSV + XLSX),
with validation so that silent parsing problems cannot pass.
"""

from pathlib import Path
import json
import os
import sys

import pandas as pd
import numpy as np


# ============================================================
# PATH
# ============================================================

ROOT = (
    Path(os.environ.get("RESULTS_ROOT", "/home/a.dorizza/OUTPUTS_MEDICAL")) / "finetuning"
)

OUTPUT_CSV = ROOT / "all_results.csv"
OUTPUT_XLSX = ROOT / "all_results.xlsx"

# Set to 0 or 1 to force the convention; "auto" infers it from
# the smallest best_epoch found across all runs.
EPOCH_INDEX_BASE = "auto"

# Expected number of seeds per configuration.
EXPECTED_SEEDS = 5

# Stop with a non-zero exit code if validation finds problems.
STRICT = True


# ============================================================
# NAMES
# ============================================================

DATASET_NAMES = {
    "kvasir_v2": "kvasirv2",
    "hyperkvasir": "hyperkvasir",
    "kvasir_capsule": "kvasir_capsule",

    "kvasir_seg": "kvasirSEG",
    "cvc_clinicdb": "cvc clinic",
    "cvc_colondb": "cvc colon",
    "etis_larib": "etis larib",
    "aircare_all": "aircare_all",
}

BACKBONE_NAMES = {
    "resnet": "resnet34",
    "vit": "vitb16b",
    "dinov2": "dinov2",
}

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

DATASET_ORDER = CLASSIFICATION_DATASETS + SEGMENTATION_DATASETS

TASK_OF_DATASET = (
    {d: "classification" for d in CLASSIFICATION_DATASETS}
    | {d: "segmentation" for d in SEGMENTATION_DATASETS}
)


# Collected during parsing, printed at the end.
PROBLEMS = []
WARNINGS = []


def problem(message):
    PROBLEMS.append(message)


def warn(message):
    WARNINGS.append(message)


# ============================================================
# EPOCH CONVENTION
# ============================================================

def detect_epoch_base(json_files):
    """
    Infer whether best_epoch is 0-based or 1-based.

    If any run reports best_epoch == 0 the numbering starts at 0;
    if the smallest best_epoch over all runs is 1 and at least one
    run reports best_epoch == len(curve), it starts at 1.
    """

    if EPOCH_INDEX_BASE in (0, 1):
        return EPOCH_INDEX_BASE

    min_epoch = None
    max_epoch = None
    curve_lengths = set()

    for path in json_files:

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        best_epoch = data.get("selection", {}).get("best_epoch", None)

        if best_epoch is None:
            continue

        best_epoch = int(best_epoch)

        min_epoch = best_epoch if min_epoch is None else min(min_epoch, best_epoch)
        max_epoch = best_epoch if max_epoch is None else max(max_epoch, best_epoch)

        curve = data.get("training_curves", {}).get("spectral_norm", None)

        if isinstance(curve, list):
            curve_lengths.add(len(curve))

    if min_epoch is None:
        warn("could not infer the epoch base: no best_epoch found")
        return 0

    base = 0 if min_epoch == 0 else 1

    print(
        f"Epoch numbering    : best_epoch in "
        f"[{min_epoch}, {max_epoch}], curve length(s) "
        f"{sorted(curve_lengths) if curve_lengths else 'n/a'} "
        f"-> assuming {base}-based"
    )

    if base == 1 and max_epoch is not None and curve_lengths:
        if max_epoch > max(curve_lengths):
            problem(
                f"best_epoch max ({max_epoch}) exceeds the longest "
                f"spectral_norm curve ({max(curve_lengths)})"
            )

    return base


# ============================================================
# SPECTRAL NORM AT BEST EPOCH
# ============================================================

def get_spectral_norm_at_best_epoch(data, best_epoch, epoch_base, path):

    curves = data.get("training_curves", {})
    spectral_norm = curves.get("spectral_norm", None)

    if spectral_norm is None:
        problem(f"{path}: no training_curves.spectral_norm")
        return np.nan

    if best_epoch is None or (
        isinstance(best_epoch, float) and np.isnan(best_epoch)
    ):
        problem(f"{path}: no best_epoch")
        return np.nan

    best_epoch = int(best_epoch)

    # -------- list -----------------------------------------

    if isinstance(spectral_norm, list):

        idx = best_epoch - epoch_base

        if not (0 <= idx < len(spectral_norm)):
            problem(
                f"{path}: best_epoch {best_epoch} out of range for a "
                f"curve of length {len(spectral_norm)} "
                f"(base {epoch_base})"
            )
            return np.nan

        return spectral_norm[idx]

    # -------- dict -----------------------------------------

    if isinstance(spectral_norm, dict):

        for key in (best_epoch, str(best_epoch)):
            if key in spectral_norm:
                return spectral_norm[key]

        problem(
            f"{path}: best_epoch {best_epoch} not a key of "
            f"spectral_norm dict"
        )
        return np.nan

    problem(
        f"{path}: unexpected spectral_norm type "
        f"{type(spectral_norm).__name__}"
    )
    return np.nan


# ============================================================
# PARSE ONE JSON
# ============================================================

REQUIRED_KEYS = ["task", "dataset", "model_type", "freeze_mode", "selection"]


def parse_json(path, epoch_base):

    with open(path, "r") as f:
        data = json.load(f)

    missing_keys = [k for k in REQUIRED_KEYS if k not in data]

    if missing_keys:
        problem(f"{path}: missing key(s) {missing_keys}")

    task = data.get("task", "").lower().strip()
    dataset_raw = data.get("dataset", "").lower().strip()
    backbone_raw = data.get("model_type", "").lower().strip()
    freeze_mode = data.get("freeze_mode", "").lower().strip()

    # --------------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------------

    hp = data.get("hyperparameters", {})

    lambda_sn = hp.get("lambda_sn", 0)

    seed = hp.get("split_seed", hp.get("seed", np.nan))

    if seed is None or (isinstance(seed, float) and np.isnan(seed)):
        problem(f"{path}: no seed / split_seed")

    # --------------------------------------------------------
    # Normalize, flagging anything unexpected
    # --------------------------------------------------------

    if dataset_raw not in DATASET_NAMES:
        problem(f"{path}: unknown dataset '{dataset_raw}'")

    if backbone_raw not in BACKBONE_NAMES:
        problem(f"{path}: unknown model_type '{backbone_raw}'")

    if freeze_mode not in {"head", "head+last"}:
        problem(f"{path}: unknown freeze_mode '{freeze_mode}'")

    dataset = DATASET_NAMES.get(dataset_raw, dataset_raw)
    backbone = BACKBONE_NAMES.get(backbone_raw, backbone_raw)

    if dataset in TASK_OF_DATASET and task != TASK_OF_DATASET[dataset]:
        problem(
            f"{path}: task '{task}' does not match dataset "
            f"'{dataset}' (expected {TASK_OF_DATASET[dataset]})"
        )

    # --------------------------------------------------------
    # Setting
    # --------------------------------------------------------

    try:
        has_sn = float(lambda_sn) > 0
    except (TypeError, ValueError):
        problem(f"{path}: lambda_sn '{lambda_sn}' is not numeric")
        has_sn = False

    if freeze_mode == "head":
        setting = "head+SN" if has_sn else "head"
    elif freeze_mode == "head+last":
        setting = "head+last+SN" if has_sn else "head+last"
    else:
        setting = freeze_mode

    # --------------------------------------------------------
    # Selection
    # --------------------------------------------------------

    selection = data.get("selection", {})

    best_epoch = selection.get("best_epoch", np.nan)

    val_acc = test_acc = val_dice = test_dice = np.nan

    if task == "classification":

        val_acc = selection.get("best_val_acc", np.nan)
        test_acc = selection.get("test_acc", np.nan)

        if pd.isna(test_acc):
            problem(f"{path}: missing selection.test_acc")

    elif task == "segmentation":

        val_dice = selection.get("best_val_dice", np.nan)
        test_dice = selection.get("test_dice", np.nan)

        if pd.isna(test_dice):
            problem(f"{path}: missing selection.test_dice")

    else:
        problem(f"{path}: unknown task '{task}'")

    # --------------------------------------------------------
    # Spectral norm
    # --------------------------------------------------------

    spectral_norm = get_spectral_norm_at_best_epoch(
        data,
        best_epoch,
        epoch_base,
        path,
    )

    # --------------------------------------------------------
    # Output row
    # --------------------------------------------------------

    return {
        "task": task,
        "dataset": dataset,
        "dataset_raw": dataset_raw,

        "backbone": backbone,
        "backbone_raw": backbone_raw,

        "setting": setting,
        "freeze_mode": freeze_mode,

        "lambda_sn": lambda_sn,
        "seed": seed,

        "best_epoch": best_epoch,

        "val_acc": val_acc,
        "test_acc": test_acc,

        "val_dice": val_dice,
        "test_dice": test_dice,

        "head_spectral_norm": spectral_norm,

        "experiment": path.parent.name,
        "json_path": str(path),
    }


# ============================================================
# SCAN ALL JSON
# ============================================================

if not ROOT.exists():
    raise FileNotFoundError(f"ROOT does not exist: {ROOT}")

json_files = sorted(ROOT.rglob("*.json"))

# The output files themselves must not be re-read as runs.
json_files = [
    p for p in json_files
    if p.name not in {OUTPUT_CSV.name, OUTPUT_XLSX.name}
]

print(f"Root               : {ROOT}")
print(f"JSON files found   : {len(json_files)}")

if len(json_files) == 0:
    raise FileNotFoundError(f"No JSON files under {ROOT}")

EPOCH_BASE = detect_epoch_base(json_files)


rows = []
failed = []

for path in json_files:

    try:
        rows.append(parse_json(path, EPOCH_BASE))

    except Exception as exc:
        failed.append((path, exc))
        problem(f"{path}: parsing failed ({exc})")


print(f"Parsed             : {len(rows)} / {len(json_files)}")

if failed:
    print("\nFAILED FILES:")
    for path, exc in failed:
        print(f"  {path}\n    {exc}")


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(rows)

df["backbone"] = pd.Categorical(
    df["backbone"], categories=BACKBONES, ordered=True
)

df["setting"] = pd.Categorical(
    df["setting"], categories=SETTINGS, ordered=True
)

df["dataset"] = pd.Categorical(
    df["dataset"], categories=DATASET_ORDER, ordered=True
)

df = df.sort_values(
    ["task", "dataset", "backbone", "setting", "seed"]
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

# --- rows that fell outside the known categories -----------

for column in ["dataset", "backbone", "setting"]:
    n_bad = df[column].isna().sum()
    if n_bad:
        problem(
            f"{n_bad} row(s) with an unrecognised {column} "
            "(dropped from the categorical order)"
        )

# --- duplicated runs ---------------------------------------

key = ["task", "dataset", "backbone", "setting", "seed"]

duplicated = df[df.duplicated(key, keep=False)]

if len(duplicated):
    problem(f"{len(duplicated)} duplicated run(s) for the same {key}")
    print("\nDUPLICATES:")
    print(
        duplicated[key + ["json_path"]]
        .sort_values(key)
        .to_string(index=False)
    )

# --- completeness ------------------------------------------

counts = (
    df.groupby(["dataset", "backbone", "setting"], observed=True)
    .size()
    .to_dict()
)

missing = []
wrong_count = []

for dataset in DATASET_ORDER:
    for backbone in BACKBONES:
        for setting in SETTINGS:

            n = counts.get((dataset, backbone, setting), 0)

            if n == 0:
                missing.append((dataset, backbone, setting))
            elif n != EXPECTED_SEEDS:
                wrong_count.append((dataset, backbone, setting, n))

if missing:
    problem(f"{len(missing)} missing configuration(s)")
    print("\nMISSING COMBINATIONS:")
    for x in missing:
        print(f"  {x}")
else:
    print("Combinations       : all present")

if wrong_count:
    problem(f"{len(wrong_count)} configuration(s) without "
            f"{EXPECTED_SEEDS} seeds")
    print("\nWRONG SEED COUNT:")
    for dataset, backbone, setting, n in wrong_count:
        print(f"  {dataset} / {backbone} / {setting}: {n}")
else:
    print(f"Seeds per config   : {EXPECTED_SEEDS} everywhere")

# --- one lambda per task and setting ------------------------

for (task, setting), sub in df.groupby(["task", "setting"], observed=True):

    lambdas = sorted(pd.unique(sub["lambda_sn"]))

    if len(lambdas) > 1:
        problem(f"{task} / {setting}: mixed lambda_sn {lambdas}")
    elif str(setting).endswith("SN"):
        print(f"lambda_sn          : {task} / {setting} = {lambdas[0]}")

# --- metric coverage ---------------------------------------

print("\nMETRIC COVERAGE (non-null / expected):")

coverage = [
    ("val_acc", df["task"] == "classification"),
    ("test_acc", df["task"] == "classification"),
    ("val_dice", df["task"] == "segmentation"),
    ("test_dice", df["task"] == "segmentation"),
    ("head_spectral_norm", pd.Series(True, index=df.index)),
]

for column, mask in coverage:

    expected = int(mask.sum())
    got = int(df.loc[mask, column].notna().sum())

    flag = "" if got == expected else "   <-- INCOMPLETE"

    print(f"  {column:20s}: {got} / {expected}{flag}")

    if got != expected:
        problem(f"{column}: {expected - got} missing value(s)")

# --- plausible ranges --------------------------------------

for column in ["val_acc", "test_acc", "val_dice", "test_dice"]:

    values = df[column].dropna()

    if len(values) and (values.min() < 0 or values.max() > 1):
        problem(
            f"{column} outside [0, 1]: "
            f"min {values.min():.4f}, max {values.max():.4f}"
        )

sn = df["head_spectral_norm"].dropna()

if len(sn) and sn.min() < 0:
    problem("negative head_spectral_norm value(s)")

print(
    f"\nhead_spectral_norm : min {sn.min():.4f}, "
    f"max {sn.max():.4f}, mean {sn.mean():.4f}"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL DATASET")
print("=" * 80)

print("Rows:", len(df))

for column in ["task", "dataset", "backbone", "setting"]:
    print(f"\n{column}:")
    print(df[column].value_counts().to_string())


# ============================================================
# SAVE
# ============================================================

if PROBLEMS and STRICT:

    print("\n" + "=" * 80)
    print(f"PROBLEMS ({len(PROBLEMS)}) - NOTHING WAS SAVED")
    print("=" * 80)

    for message in PROBLEMS:
        print(f"  - {message}")

    sys.exit(1)

df.to_csv(OUTPUT_CSV, index=False)
df.to_excel(OUTPUT_XLSX, index=False)

print("\nSaved:")
print(f"  {OUTPUT_CSV}")
print(f"  {OUTPUT_XLSX}")

if WARNINGS:
    print("\nWARNINGS:")
    for message in WARNINGS:
        print(f"  - {message}")

if PROBLEMS:
    print(f"\nPROBLEMS ({len(PROBLEMS)}) - files saved anyway (STRICT=False):")
    for message in PROBLEMS:
        print(f"  - {message}")
else:
    print("\nValidation: OK")
