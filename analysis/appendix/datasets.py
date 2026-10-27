#!/usr/bin/env python3
"""
analysis/appendix/datasets.py
==========================
Everything Appendix A ("Dataset details and splits") states, emitted as LaTeX
and CSV so that no number in it is typed by hand.

    MEDICAL_ROOT=/home/datasets/medical \
    python -m analysis.appendix.datasets --out paper_results/appendix_datasets

Writes, in ``--out``:

    datasets_overview.tex / .csv      the body Table 1, regenerated from disk
    classes_<dataset>.tex / .csv      per-class counts, classification only
    classes_dropped.tex / .csv        classes removed by min_per_class
    aircare_sites.tex / .csv          AIRCARE per-site breakdown
    splits.tex / .csv                 sizes and fingerprint, per dataset and seed

It reads the datasets through the same registry the training scripts use, so a
count that disagrees with a run is a real disagreement and not a different way
of counting. The split files are read, never regenerated: a missing one is
reported rather than created, because creating it here with a different code
path is exactly how a fingerprint mismatch gets hidden.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

from shared.DatasetRegistry import (
    DATASETS, IMG_EXT, FolderClassificationDataset, SegmentationDataset,
    get_spec, list_datasets,
)

MIN_PER_CLASS = 20          # the value every reported run used
SEEDS = (0, 1, 2, 3, 4)

PRETTY = {
    "kvasir_v2": "Kvasir-v2", "hyperkvasir": "Hyper-Kvasir",
    "kvasir_capsule": "Kvasir-Capsule", "kvasir_seg": "Kvasir-SEG",
    "cvc_clinicdb": "CVC-ClinicDB", "cvc_colondb": "CVC-ColonDB",
    "etis_larib": "ETIS-Larib", "aircare_all": "AIRCARE",
}


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def tex(x) -> str:
    """Escape the characters LaTeX treats as special in a table cell."""
    s = str(x)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("_", r"\_"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def write(df: pd.DataFrame, out: str, stem: str, caption: str, label: str,
          index: bool = False, escape_cols: tuple = ("Class", "Dataset", "Site")) -> None:
    """One table, two formats, so every LaTeX number has a machine-readable twin.

    The LaTeX is a plain booktabs table placed with [!htbp], never a longtable,
    so it works in the two-column layout of the paper."""
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"{stem}.csv"), index=index)
    shown = df.copy()
    for c in shown.columns:
        if c in escape_cols:
            shown[c] = shown[c].map(tex)
    body = shown.to_latex(index=index, escape=False, float_format="%.2f")
    with open(os.path.join(out, f"{stem}.tex"), "w") as fh:
        fh.write("\n".join([r"\begin{table}[!htbp]", r"\centering", r"\footnotesize",
                            rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                            body, r"\end{table}", ""]))
    print(f"  wrote {stem}.tex / .csv  ({len(df)} rows)")


def raw_class_counts(root: str) -> dict[str, int]:
    """
    Images per leaf folder, BEFORE any filtering.

    Re-walks the tree with the same rule the dataset class uses (a folder is a
    class when it directly contains images), which is what makes the "dropped"
    table a difference of two things counted the same way.
    """
    counts: dict[str, int] = {}
    for dirpath, _, filenames in os.walk(root):
        n = sum(1 for f in filenames if f.lower().endswith(IMG_EXT))
        if n:
            counts[os.path.relpath(dirpath, root).replace(os.sep, "/")] = n
    return counts


# -----------------------------------------------------------------------------
# per-dataset collection
# -----------------------------------------------------------------------------
TRACT = {"lower-gi-tract": "lower GI", "upper-gi-tract": "upper GI"}
SITE_FIX = {"orophaynx": "oropharynx"}


def split_class(name: str) -> tuple[str, str]:
    """Hyper-Kvasir classes are folder paths (tract/group/class): keep the tract
    and the class, which is unique on its own. Flat names have no tract."""
    parts = name.split("/")
    if len(parts) == 1:
        return "", name
    return TRACT.get(parts[0], parts[0]), parts[-1]


def site_label(site: str) -> str:
    """Clean an AIRCARE site folder name: drop counts in brackets, fix typos."""
    s = re.sub(r"\(.*?\)", "", site.replace("_", " ")).strip().lower()
    s = SITE_FIX.get(s, s)
    return s[:1].upper() + s[1:]


def classification_tables(out: str) -> list[dict]:
    rows = []
    dropped = []
    for key in list_datasets("classification"):
        spec = get_spec(key)
        before = raw_class_counts(spec.root)
        ds = FolderClassificationDataset(spec, min_per_class=MIN_PER_CLASS,
                                         max_per_class=None)
        kept = Counter(ds.labels.tolist())

        tracts, names = zip(*(split_class(c) for c in ds.classes))
        per_class = pd.DataFrame({
            "Tract": tracts, "Class": names,
            "Images": [kept[i] for i in range(len(ds.classes))],
        })
        if not any(tracts):
            per_class = per_class.drop(columns="Tract")
        per_class["Share (\\%)"] = (100 * per_class["Images"]
                                    / per_class["Images"].sum()).round(2)
        write(per_class, out, f"classes_{key}",
              caption=(f"{PRETTY.get(key, key)}: images per class, after removing "
                       f"the classes with fewer than {MIN_PER_CLASS} images."
                       + (" Classes are grouped by gastrointestinal tract." if any(tracts) else "")),
              label=f"tab:classes-{key.replace('_', '-')}")

        for cls, n in sorted(before.items()):
            if cls not in ds.class_to_idx:
                dropped.append({"Dataset": PRETTY.get(key, key),
                                "Class": split_class(cls)[1], "Images": n})

        rows.append({
            "Dataset": PRETTY.get(key, key), "Task": "Classification",
            "Images": len(ds), "Classes": ds.num_classes,
            "Imbalance (max/min)": round(max(kept.values()) / min(kept.values()), 1),
        })

    write(pd.DataFrame(dropped), out, "classes_dropped",
          caption=("Classes removed because they hold fewer than "
                   f"{MIN_PER_CLASS} images, and therefore cannot be split "
                   "70/10/20 in any meaningful way."),
          label="tab:classes-dropped")
    return rows


def segmentation_tables(out: str) -> list[dict]:
    rows, sites = [], []
    for key in list_datasets("segmentation"):
        if key.startswith("aircare_") and key != "aircare_all":
            continue                        # pooled into aircare_all
        spec = get_spec(key)
        ds = SegmentationDataset(spec)

        # Lesion area is what the segmentation split stratifies on, so report it.
        areas = np.array([ds.raw(i)[1].mean() for i in range(len(ds))])

        if spec.sources:
            for site, n in sorted(Counter(ds.sites).items()):
                sel = np.array([s == site for s in ds.sites])
                sites.append({"Site": site_label(site), "Images": n,
                              "Median lesion area (\\%)":
                                  round(100 * float(np.median(areas[sel])), 2)})

        rows.append({
            "Dataset": PRETTY.get(key, key), "Task": "Segmentation",
            "Images": len(ds), "Classes": 2,
            "Median lesion area (\\%)": round(100 * float(np.median(areas)), 2),
        })

    if sites:
        write(pd.DataFrame(sites), out, "aircare_sites",
              caption=("AIRCARE, per anatomical site. The four sites are pooled "
                       "into one site-agnostic lesion/background dataset because "
                       "individually they are too small for a 70/10/20 split."),
              label="tab:aircare-sites")
    return rows


def split_table(out: str, split_root: str) -> None:
    rows = []
    for key in sorted(DATASETS):
        if key.startswith("aircare_") and key != "aircare_all":
            continue
        for seed in SEEDS:
            path = os.path.join(split_root, key, f"split_seed{seed}.json")
            if not os.path.isfile(path):
                print(f"  [missing] {path}")
                continue
            d = json.load(open(path))
            rows.append({
                "Dataset": PRETTY.get(key, key), "Seed": seed,
                "Images": d["n_samples"], "Train": len(d["train"]),
                "Val.": len(d["val"]), "Test": len(d["test"]),
                "Fingerprint": f"\\texttt{{{d['fingerprint']}}}",
            })
    write(pd.DataFrame(rows), out, "splits",
          caption=("Stratified 70/10/20 partitions, one per dataset and seed. "
                   "The fingerprint is computed from the number of samples and "
                   "the stratification targets: the corruption and drift stages "
                   "load the split and fail loudly on a mismatch, so a change of "
                   "the data on disk cannot silently evaluate on images seen "
                   "during training."),
          label="tab:splits")


REFERENCE = {
    "Kvasir-v2": r"\citet{pogorelov2017kvasir}",
    "Hyper-Kvasir": r"\citet{borgli2020hyperkvasir}",
    "Kvasir-Capsule": r"\citet{smedsrud2021kvasircapsule}",
    "Kvasir-SEG": r"\citet{jha2020kvasirseg}",
    "CVC-ClinicDB": r"\citet{bernal2015wm}",
    "CVC-ColonDB": r"\citet{bernal2012towards}",
    "ETIS-Larib": r"\citet{silva2014toward}",
    "AIRCARE": "private (AIRCARE)",
}


def paper_table(rows: list[dict], out: str, split_root: str) -> None:
    """Table 1 of the paper: images and classes from disk, split sizes from the
    split files (seed 0; a different size for another seed is reported)."""
    key_of = {v: k for k, v in PRETTY.items()}
    by_name = {r["Dataset"]: r for r in rows}
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{6pt}",
        r"\caption{Endoscopic datasets used in this work. \emph{Images} is the "
        r"number of samples retained after dropping classes with fewer than "
        rf"{MIN_PER_CLASS} images (\Cref{{sec:setup_datasets}}). For segmentation, "
        r"the two classes are lesion and background. \emph{Train}, "
        r"\emph{Validation} and \emph{Test} report the sizes of the stratified "
        r"70/10/20 split, which is redrawn for each of the five seeds. AIRCARE is a "
        r"private collection pooling four head and neck sites into a single "
        r"lesion/background task.}",
        r"\label{tab:datasets}",
        r"\begin{tabular}{l l r r r r r l}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Task} & \textbf{Images} & \textbf{Classes} & "
        r"\textbf{Train} & \textbf{Validation} & \textbf{Test} & \textbf{Reference} \\",
        r"\midrule",
    ]
    fmt = lambda n: f"{n:,}".replace(",", r"\,")
    previous_task = None
    for name in PRETTY.values():
        if name not in by_name:
            continue
        r = by_name[name]
        sizes = []
        for seed in SEEDS:
            path = os.path.join(split_root, key_of[name], f"split_seed{seed}.json")
            if os.path.isfile(path):
                d = json.load(open(path))
                sizes.append((len(d["train"]) if isinstance(d["train"], list) else d["train"],
                              len(d["val"]) if isinstance(d["val"], list) else d["val"],
                              len(d["test"]) if isinstance(d["test"], list) else d["test"]))
        if not sizes:
            print(f"  [missing] splits of {name}")
            continue
        if len(set(sizes)) > 1:
            print(f"  [note] {name}: split sizes differ across seeds {sorted(set(sizes))}")
        train, val, test = sizes[0]
        if previous_task and r["Task"] != previous_task:
            lines.append(r"\midrule")
        previous_task = r["Task"]
        lines.append(f"{name:<15} & {r['Task']} & {fmt(r['Images'])} & {r['Classes']} & "
                     f"{fmt(train)} & {fmt(val)} & {fmt(test)} & {REFERENCE[name]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    with open(os.path.join(out, "datasets.tex"), "w") as fh:
        fh.write("\n".join(lines))
    print("  wrote datasets.tex (Table 1 of the paper)")


# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="paper_results/appendix_datasets")
    p.add_argument("--split_root",
                   default=os.environ.get("SPLIT_ROOT", "splits"))
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = classification_tables(args.out) + segmentation_tables(args.out)

    write(pd.DataFrame(rows), args.out, "datasets_overview",
          caption="Datasets, as read from disk through the registry.",
          label="tab:datasets-overview")
    split_table(args.out, args.split_root)
    paper_table(rows, args.out, args.split_root)
    print(f"\nDataset tables in {args.out}")


if __name__ == "__main__":
    main()
