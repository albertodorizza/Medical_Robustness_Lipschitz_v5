"""
analysis/pack_results.py
===============
Compress the whole OUTPUTS_MEDICAL tree into a handful of small, tidy CSVs.

Why this exists
---------------
The raw JSONs are large for one reason only: the drift files carry per-pixel and
per-image arrays (``upper_bounds``, ``drift_values``, ``ratios``, ``margins``,
``pixels``, ``per_image``), and the corruption files repeat a nested dict for
every one of the ~108 experiments in a sweep.  None of that needs to travel:
every table and every figure of the paper is a function of a few scalars per
(run, experiment) plus, for two figures, a point cloud that a few thousand
samples represent just as well as a few million.

So this script walks the tree once and writes:

    results_pack/
      finetuning.csv.gz     one row per (task, dataset, arch, setting, seed)
      ablation.csv.gz       same, for the lambda ablation tree
      corruption.csv.gz     one row per (run, experiment, feature layer)
      drift.csv.gz          one row per (run, eval dataset, perturbation)
      drift_points.csv.gz   subsample of the drift point clouds (for scatters)
      manifest.json         what was found, what was skipped, and the sizes

Usage
-----
    python -m analysis.pack_results --results_root /home/a.dorizza/OUTPUTS_MEDICAL \
                           --out results_pack
    tar czf results_pack.tgz results_pack

Options worth knowing:
    --last-layer-only     keep only the deepest hooked layer of each
                          classification sweep (drops ~3/4 of the rows)
    --points-per-run N    how many drift samples to keep per run (default 300,
                          0 disables drift_points.csv.gz entirely)
    --no-quantiles        skip the quantile summary of the drift arrays

Only the standard library plus numpy is used, so it runs on the cluster with no
extra install.
"""

import argparse
import csv
import glob
import gzip
import json
import os
import sys
import time
from datetime import datetime

try:
    import numpy as np
except ImportError:                                    # pragma: no cover
    print("numpy is required (quantiles and subsampling).", file=sys.stderr)
    raise


# =============================================================================
# Run-name parsing
# =============================================================================
def parse_run_name(name: str) -> dict:
    """
    Recover ``arch``, ``freeze``, ``lambda_sn`` and ``seed`` from a run folder.

    Mirrors ``shared.paths.parse_run_name`` but is duplicated here on purpose:
    this script has to be runnable on its own, copied next to the results, with
    no import path set up.
    """
    seed, base = 0, name
    if "_seed" in base:
        base, _, tail = base.rpartition("_seed")
        if tail.isdigit():
            seed = int(tail)
        else:
            base = name

    if "_lambda" not in base:
        return {"arch": name, "freeze": "?", "lambda_sn": None, "seed": seed}

    head, _, lam = base.rpartition("_lambda")
    arch, _, freeze = head.partition("_")
    try:
        lam_val = float(lam)
    except ValueError:
        lam_val = None
    return {"arch": arch, "freeze": freeze, "lambda_sn": lam_val, "seed": seed}


def setting_label(info: dict) -> str:
    """'H', 'H+SN', 'H+L' or 'H+L+SN' from a parsed run name."""
    return ("H+L" if info["freeze"] == "head+last" else "H") + \
           ("+SN" if info["lambda_sn"] else "")


def split_corruption_folder(folder: str) -> tuple[str, str, str]:
    """
    Split a corruption run folder into ``(run_name, eval_dataset, variant)``.

    Folders look like ``<run>_rawfeat[_softmasking]`` in distribution and
    ``<run>_on_<evalds>_rawfeat[_softmasking]`` when evaluated on another
    dataset.  The eval dataset is empty when it equals the training one; the
    caller fills it in from the metadata, which is authoritative.
    """
    base = folder
    variant = ""
    for suffix in ("_rawfeat_softmasking", "_rawfeat_norm_softmasking",
                   "_rawfeat", "_rawfeat_norm"):
        if base.endswith(suffix):
            variant = suffix.lstrip("_")
            base = base[: -len(suffix)]
            break
    eval_ds = ""
    if "_on_" in base:
        base, _, eval_ds = base.rpartition("_on_")
    return base, eval_ds, variant


def first(d: dict, *keys, default=None):
    """
    First key of ``keys`` that ``d`` actually has, else ``default``.

    The drift scripts have been through a rename: the summary block used to
    spell the segmentation fields ``pixel_flip_rate`` / ``true_dice_drop`` /
    ``mean_dice_bound`` and now spells them ``true_flip_rate`` / ``true_drop``
    / ``predicted_drop``.  Reading only one spelling silently produced a column
    of NaN for whichever half of the results tree was written by the other
    version, with nothing in the manifest to show for it.  Listing both means
    the packer can be pointed at a tree containing runs from either era.

    ``None`` counts as absent, so a key present but null falls through to the
    next alias rather than poisoning the column.
    """
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def f(x):
    """JSON value -> float, with None and non-numbers becoming NaN."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# =============================================================================
# Writer
# =============================================================================
class CsvGz:
    """Gzipped CSV with a fixed header, written incrementally."""

    def __init__(self, path, fieldnames):
        self.path = path
        self.fh = gzip.open(path, "wt", newline="", compresslevel=9)
        self.w = csv.DictWriter(self.fh, fieldnames=fieldnames,
                                extrasaction="ignore")
        self.w.writeheader()
        self.n = 0

    def write(self, row):
        self.w.writerow(row)
        self.n += 1

    def close(self):
        self.fh.close()
        return {"rows": self.n,
                "bytes": os.path.getsize(self.path) if os.path.exists(self.path) else 0}


def load_json(path, errors):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:                            # truncated / running job
        errors.append({"path": path, "error": repr(exc)})
        return None


# =============================================================================
# Stage 1: fine-tuning logs (and the ablation tree, same schema)
# =============================================================================
FT_FIELDS = ["stage", "task", "dataset", "arch", "freeze", "lambda_sn",
             "setting", "seed", "best_epoch", "val", "test", "test_iou",
             "sigma", "sigma_final", "n_epochs"]


def pack_finetuning(root, stage, out, errors, skipped):
    """
    One row per run.  The training curves are dropped except for the spectral
    norm at the selected epoch and at the end, which is what the lambda
    analysis needs -- and which is also the quantity that reveals whether the
    head is shrinking its logits rather than becoming more stable.
    """
    n = 0
    for task in ("classification", "segmentation"):
        base = os.path.join(root, stage, task)
        for path in sorted(glob.glob(os.path.join(base, "**", "*_log.json"),
                                     recursive=True)):
            log = load_json(path, errors)
            if log is None:
                continue
            folder = os.path.dirname(path)
            info = parse_run_name(os.path.basename(folder))
            sel = log.get("selection", {})
            if not sel:
                skipped.append({"path": path, "reason": "no 'selection' block"})
                continue
            curves = log.get("training_curves", {})
            sn = curves.get("spectral_norm", []) or []
            best = sel.get("best_epoch", -1)
            out.write({
                "stage":     stage,
                "task":      task,
                "dataset":   log.get("dataset") or os.path.basename(os.path.dirname(folder)),
                "arch":      info["arch"],
                "freeze":    info["freeze"],
                "lambda_sn": info["lambda_sn"],
                "setting":   setting_label(info),
                "seed":      info["seed"],
                "best_epoch": best,
                "val":       f(sel.get("best_val_dice", sel.get("best_val_acc"))),
                "test":      f(sel.get("test_dice", sel.get("test_acc"))),
                "test_iou":  f(sel.get("test_iou")),
                "sigma":     f(sn[best]) if 0 <= best < len(sn) else float("nan"),
                "sigma_final": f(sn[-1]) if sn else float("nan"),
                "n_epochs":  len(sn),
            })
            n += 1
    return n


# =============================================================================
# Stage 2: corruption sweeps
# =============================================================================
CORR_FIELDS = [
    "task", "train_dataset", "eval_dataset", "eval_split", "arch", "freeze",
    "lambda_sn", "setting", "seed", "variant", "layer",
    "shape", "ratio", "eff_ratio", "severity", "family", "ptype", "n_samples",
    # output level, name-neutral so both tasks share the schema
    "metric_clean", "metric_corr", "metric_drop",
    # feature level
    "mse_in", "mse_out", "mse_all", "cos_in", "cos_out", "cos_all",
    # segmentation only
    "iou_clean", "iou_corr", "dice_in", "dice_out",
    "drop_none", "drop_partial", "drop_full", "drop_no_lesion",
    "n_none", "n_partial", "n_full", "n_no_lesion",
]

REGION_KEY = {"in": "corrupted", "out": "non_corrupted", "all": "global"}


def _feature_cols(region_dict):
    """Flatten ``{region: {metric: value}}`` into mse_in/out/all + cos_*."""
    cols = {}
    for short, key in REGION_KEY.items():
        block = region_dict.get(key, {}) or {}
        cols[f"mse_{short}"] = f(block.get("mse"))
        cols[f"cos_{short}"] = f(block.get("cos"))
    return cols


def pack_corruption(root, out, errors, skipped, last_layer_only):
    n = 0
    for task in ("classification", "segmentation"):
        base = os.path.join(root, "corruption_experiments", task)
        paths = sorted(glob.glob(os.path.join(base, "**", "metrics.json"),
                                 recursive=True))
        if paths:
            print(f"  corruption/{task}: {len(paths)} files", flush=True)
        t0 = time.time()
        for i, path in enumerate(paths, 1):
            if i % 100 == 0 or i == len(paths):
                done = time.time() - t0
                print(f"    {i}/{len(paths)}  {done:.0f}s elapsed, "
                      f"~{done / i * (len(paths) - i):.0f}s left", flush=True)
            blob = load_json(path, errors)
            if blob is None:
                continue
            meta = blob.get("metadata", {})
            cfg = blob.get("config", {})
            folder = os.path.basename(os.path.dirname(path))
            run, eval_from_folder, variant = split_corruption_folder(folder)
            info = parse_run_name(run)

            # The training dataset is the directory under the task; the eval
            # dataset comes from the metadata, which is what was actually
            # loaded -- the folder name is only a fallback.
            rel = os.path.relpath(os.path.dirname(path), base)
            train_ds = rel.split(os.sep)[0]
            eval_ds = meta.get("dataset") or eval_from_folder or train_ds

            common = {
                "task":          task,
                "train_dataset": train_ds,
                "eval_dataset":  eval_ds,
                "eval_split":    meta.get("eval_split", ""),
                "arch":          info["arch"],
                "freeze":        info["freeze"],
                "lambda_sn":     info["lambda_sn"],
                "setting":       setting_label(info),
                "seed":          cfg.get("split_seed", info["seed"]),
                "variant":       variant,
            }

            exps = blob.get("all_experiments", [])
            if not exps:
                skipped.append({"path": path, "reason": "no experiments"})
                continue

            for exp in exps:
                row = dict(common)
                row.update({
                    "shape":     exp.get("corruption_type"),
                    "ratio":     f(exp.get("perturbation_ratio")),
                    "eff_ratio": f(exp.get("effective_ratio")),
                    "severity":  exp.get("severity_level"),
                    "family":    exp.get("perturbation_family"),
                    "ptype":     exp.get("perturbation_name"),
                    "n_samples": exp.get("n_samples"),
                })

                if task == "segmentation":
                    ov = exp.get("dice_drop_by_overlap", {}) or {}
                    nov = exp.get("n_by_overlap", {}) or {}
                    row.update({
                        "metric_clean": f(exp.get("dice_clean")),
                        "metric_corr":  f(exp.get("dice_corrupted")),
                        "metric_drop":  f(exp.get("dice_drop")),
                        "iou_clean":    f(exp.get("iou_clean")),
                        "iou_corr":     f(exp.get("iou_corrupted")),
                        "dice_in":      f(exp.get("dice_corrupted_in")),
                        "dice_out":     f(exp.get("dice_corrupted_out")),
                        "drop_none":       f(ov.get("none")),
                        "drop_partial":    f(ov.get("partial")),
                        "drop_full":       f(ov.get("full")),
                        "drop_no_lesion":  f(ov.get("no_lesion")),
                        "n_none":      nov.get("none"),
                        "n_partial":   nov.get("partial"),
                        "n_full":      nov.get("full"),
                        "n_no_lesion": nov.get("no_lesion"),
                        "layer":       "head_input",
                    })
                    row.update(_feature_cols(exp.get("feature_metrics", {}) or {}))
                    out.write(row)
                    n += 1
                else:
                    row.update({
                        "metric_clean": f(exp.get("accuracy_clean")),
                        "metric_corr":  f(exp.get("accuracy_corrupted")),
                        "metric_drop":  f(exp.get("accuracy_drop")),
                    })
                    lm = exp.get("layer_metrics", {}) or {}
                    layers = list(lm)
                    if last_layer_only and layers:
                        layers = layers[-1:]
                    for layer in layers:
                        lrow = dict(row)
                        lrow["layer"] = layer
                        lrow.update(_feature_cols(lm.get(layer, {}) or {}))
                        out.write(lrow)
                        n += 1
    return n


# =============================================================================
# Stage 3: drift analysis
# =============================================================================
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

DRIFT_FIELDS = [
    "task", "train_dataset", "eval_dataset", "arch", "freeze", "lambda_sn",
    "setting", "seed", "perturbation", "severity", "eps",
    "sigma_max", "gamma", "mean_dz_norm", "mean_dlogit_norm",
    "mean_ratio", "std_ratio",
    "flip_rate", "certified_flip_rate",
    "metric_clean", "metric_corr", "true_drop", "predicted_drop",
    "bound_violations", "bound_violation_rate", "mean_tightness",
    "n_units", "n_images",
] + [f"{arr}_q{int(q * 100)}" for arr in ("ratio", "dz", "dlogit", "margin")
     for q in QUANTILES]

POINT_FIELDS = ["task", "train_dataset", "eval_dataset", "arch", "setting",
                "seed", "perturbation", "dz", "dlogit", "ratio", "margin"]


def _as_numpy(arrays):
    """
    Convert each raw JSON list to a float array exactly once.

    Doing this per element, inside the subsampling loop, is what made an
    earlier version of this script appear to hang: with 200k saved pixels per
    drift run and four arrays, re-converting for every kept index is tens of
    millions of conversions per file.
    """
    out = {}
    for name, values in arrays.items():
        if values is None:
            continue
        arr = np.asarray(values, dtype=float)
        if arr.size:
            out[name] = arr
    return out


def _quantile_cols(arrays_np, names, enabled):
    """Five quantiles per array; NaN-safe and empty-safe."""
    cols = {}
    for name in names:
        arr = arrays_np.get(name)
        for q in QUANTILES:
            key = f"{name}_q{int(q * 100)}"
            if not enabled or arr is None or arr.size == 0 or np.all(np.isnan(arr)):
                cols[key] = float("nan")
            else:
                cols[key] = float(np.nanquantile(arr, q))
    return cols


def _subsample(rng, arrays_np, k):
    """Take the same k random indices from every array, or all if shorter."""
    if k <= 0 or not arrays_np:
        return []
    n = min(arr.shape[0] for arr in arrays_np.values())
    if n == 0:
        return []
    idx = rng.choice(n, size=min(k, n), replace=False)
    picked = {name: arr[idx] for name, arr in arrays_np.items()}
    return [{name: float(vals[j]) for name, vals in picked.items()}
            for j in range(len(idx))]


def pack_drift(root, out, points_out, errors, skipped, points_per_run,
               quantiles_enabled, seed=0):
    """
    One row per drift run.  The heavy arrays are replaced by five quantiles
    each, plus an optional subsample written to a separate file, which is
    enough to redraw the observed-vs-bound scatter and the tightness histogram.
    """
    rng = np.random.default_rng(seed)
    n = 0
    for task in ("classification", "segmentation"):
        base = os.path.join(root, "drift_analysis", task)
        paths = sorted(glob.glob(os.path.join(base, "**", "*_results.json"),
                                 recursive=True))
        if paths:
            print(f"  drift/{task}: {len(paths)} files", flush=True)
        t0 = time.time()
        for i, path in enumerate(paths, 1):
            if i % 25 == 0 or i == len(paths):
                done = time.time() - t0
                eta = done / i * (len(paths) - i)
                print(f"    {i}/{len(paths)}  {done:.0f}s elapsed, "
                      f"~{eta:.0f}s left", flush=True)
            blob = load_json(path, errors)
            if blob is None:
                continue
            folder = os.path.dirname(path)
            rel = os.path.relpath(folder, base)
            parts = rel.split(os.sep)
            train_ds = parts[0] if parts else ""
            # .../<train_ds>/<drift_tag>/<run_name>/
            run = parts[-1] if len(parts) >= 3 else os.path.basename(folder)
            info = parse_run_name(run)

            if task == "segmentation":
                cfg = blob.get("config", {}) or {}
                s = blob.get("summary", {}) or {}
                arrays = {
                    "ratio":  (blob.get("pixels", {}) or {}).get("ratio"),
                    "dz":     (blob.get("pixels", {}) or {}).get("dz_norm"),
                    "dlogit": (blob.get("pixels", {}) or {}).get("dlogit_norm"),
                    "margin": (blob.get("pixels", {}) or {}).get("margin"),
                }
                row = {
                    "task": task, "train_dataset": train_ds,
                    "eval_dataset": cfg.get("dataset", ""),
                    "arch": info["arch"], "freeze": info["freeze"],
                    "lambda_sn": info["lambda_sn"], "setting": setting_label(info),
                    "seed": cfg.get("split_seed", info["seed"]),
                    "perturbation": cfg.get("perturbation"),
                    "severity": cfg.get("severity"), "eps": cfg.get("eps"),
                    "sigma_max": f(s.get("sigma_max")), "gamma": f(s.get("gamma")),
                    "mean_dz_norm": f(s.get("mean_dz_norm")),
                    "mean_dlogit_norm": f(s.get("mean_dlogit_norm")),
                    "mean_ratio": f(s.get("mean_ratio")),
                    "std_ratio": f(s.get("std_ratio")),
                    "flip_rate": f(first(s, "true_flip_rate", "pixel_flip_rate")),
                    "certified_flip_rate": f(s.get("certified_flip_rate")),
                    "metric_clean": f(s.get("mean_dice_clean")),
                    "metric_corr": f(s.get("mean_dice_corrupted")),
                    "true_drop": f(first(s, "true_drop", "true_dice_drop")),
                    "predicted_drop": f(first(s, "predicted_drop",
                                              "mean_dice_bound")),
                    "bound_violations": s.get("bound_violations"),
                    "bound_violation_rate": f(s.get("bound_violation_rate")),
                    "mean_tightness": f(s.get("mean_tightness")),
                    "n_units": s.get("n_pixels"), "n_images": s.get("n_images"),
                }
            else:
                # Two layouts in the wild.  Current drift_analysis_cls.py writes a
                # 'summary' block plus 'per_sample' arrays and a 'config' block
                # (the same skeleton as the segmentation script).  Older runs
                # wrote a 'margin_bound' block with top-level 'ratios' /
                # 'drift_values' / 'margins' and the scalars spelled
                # true_acc_drop / predicted_acc_drop / tightness_flip.
                #
                # The old reader looked only for 'margin_bound' and skipped
                # everything else, which meant every classification drift run
                # produced by the current code was dropped -- 360 files, the
                # whole stage, reported only as a line in the manifest.  Merging
                # the two blocks and listing both spellings reads either.
                s = blob.get("summary", {}) or {}
                mb = blob.get("margin_bound", {}) or {}
                per = blob.get("per_sample", {}) or {}
                cfg = blob.get("config", {}) or {}
                if not s and not mb:
                    skipped.append({"path": path,
                                    "reason": "neither 'summary' nor "
                                              "'margin_bound' block"})
                    continue
                src = {**mb, **s}          # summary wins where both are present

                arrays = {
                    "ratio":  first(per, "ratio", default=blob.get("ratios")),
                    # 'drift' is the logit-space drift in the current layout;
                    # the old 'drift_values' held the same quantity.
                    "dz":     blob.get("drift_values"),
                    "dlogit": first(per, "drift", default=None),
                    "margin": first(per, "margin", default=blob.get("margins")),
                }
                row = {
                    "task": task, "train_dataset": train_ds,
                    "eval_dataset": first(cfg, "dataset",
                                          default=blob.get("dataset", "")),
                    "arch": info["arch"], "freeze": info["freeze"],
                    "lambda_sn": info["lambda_sn"], "setting": setting_label(info),
                    "seed": first(cfg, "split_seed",
                                  default=blob.get("split_seed", info["seed"])),
                    "perturbation": first(cfg, "perturbation",
                                          default=blob.get("perturbation")),
                    "severity": first(cfg, "severity",
                                      default=blob.get("severity")),
                    "eps": first(cfg, "eps", default=blob.get("eps")),
                    "sigma_max": f(first(src, "sigma_max", "spectral_norm_W",
                                         default=blob.get("spectral_norm_W"))),
                    "gamma": f(src.get("gamma")),
                    "mean_dz_norm": f(src.get("mean_upper")),
                    "mean_dlogit_norm": f(src.get("mean_drift")),
                    "mean_ratio": f(src.get("mean_ratio")),
                    "std_ratio": f(src.get("std_ratio")),
                    "flip_rate": f(src.get("true_flip_rate")),
                    "certified_flip_rate": f(src.get("certified_flip_rate")),
                    "metric_clean": f(src.get("clean_acc")),
                    "true_drop": f(first(src, "true_drop", "true_acc_drop")),
                    "predicted_drop": f(first(src, "predicted_drop",
                                              "predicted_acc_drop")),
                    "bound_violations": src.get("bound_violations"),
                    "bound_violation_rate": f(src.get("bound_violation_rate")),
                    "mean_tightness": f(first(src, "mean_tightness",
                                              "tightness_flip")),
                    "n_units": first(src, "n_samples",
                                     default=blob.get("num_samples")),
                    "n_images": first(src, "n_samples",
                                      default=blob.get("num_samples")),
                }

            arrays_np = (_as_numpy(arrays)
                         if (quantiles_enabled or points_out is not None)
                         else {})
            row.update(_quantile_cols(arrays_np, ("ratio", "dz", "dlogit", "margin"),
                                      quantiles_enabled))
            out.write(row)
            n += 1

            if points_out is not None:
                for pt in _subsample(rng, arrays_np, points_per_run):
                    pt.update({
                        "task": row["task"],
                        "train_dataset": row["train_dataset"],
                        "eval_dataset": row["eval_dataset"],
                        "arch": row["arch"], "setting": row["setting"],
                        "seed": row["seed"], "perturbation": row["perturbation"],
                    })
                    points_out.write(pt)
    return n


# =============================================================================
# Main
# =============================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_root", required=True,
                   help="OUTPUTS_MEDICAL root.")
    p.add_argument("--out", default="results_pack",
                   help="Output directory (created if missing).")
    p.add_argument("--last-layer-only", action="store_true",
                   help="Keep only the deepest hooked layer per classification "
                        "sweep instead of every hooked layer.")
    p.add_argument("--points-per-run", type=int, default=300,
                   help="Drift samples kept per run for the scatter plots; "
                        "0 disables drift_points.csv.gz.")
    p.add_argument("--no-quantiles", action="store_true",
                   help="Skip the quantile summary of the drift arrays.")
    args = p.parse_args()

    root = os.path.abspath(args.results_root)
    os.makedirs(args.out, exist_ok=True)
    errors, skipped = [], []

    writers = {
        "finetuning": CsvGz(os.path.join(args.out, "finetuning.csv.gz"), FT_FIELDS),
        "ablation":   CsvGz(os.path.join(args.out, "ablation.csv.gz"), FT_FIELDS),
        "corruption": CsvGz(os.path.join(args.out, "corruption.csv.gz"), CORR_FIELDS),
        "drift":      CsvGz(os.path.join(args.out, "drift.csv.gz"), DRIFT_FIELDS),
    }
    points = (CsvGz(os.path.join(args.out, "drift_points.csv.gz"), POINT_FIELDS)
              if args.points_per_run > 0 else None)

    print(f"Reading {root}")
    counts = {}
    counts["finetuning"] = pack_finetuning(root, "finetuning",
                                           writers["finetuning"], errors, skipped)
    print(f"  finetuning : {counts['finetuning']} runs")
    counts["ablation"] = pack_finetuning(root, "ablation",
                                         writers["ablation"], errors, skipped)
    print(f"  ablation   : {counts['ablation']} runs")
    counts["corruption"] = pack_corruption(root, writers["corruption"], errors,
                                           skipped, args.last_layer_only)
    print(f"  corruption : {counts['corruption']} rows")
    counts["drift"] = pack_drift(root, writers["drift"], points, errors, skipped,
                                 args.points_per_run, not args.no_quantiles)
    print(f"  drift      : {counts['drift']} runs")

    stats = {name: w.close() for name, w in writers.items()}
    if points is not None:
        stats["drift_points"] = points.close()

    manifest = {
        "results_root": root,
        "created": datetime.now().isoformat(),
        "options": {"last_layer_only": args.last_layer_only,
                    "points_per_run": args.points_per_run,
                    "quantiles": not args.no_quantiles},
        "files": stats,
        "skipped": skipped[:200],
        "n_skipped": len(skipped),
        "errors": errors[:200],
        "n_errors": len(errors),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    total = sum(v["bytes"] for v in stats.values())
    print("\nWrote:")
    for name, v in stats.items():
        print(f"  {name:14s} {v['rows']:>9,} rows  {v['bytes'] / 1e6:8.2f} MB")
    print(f"  {'TOTAL':14s} {'':>9}       {total / 1e6:8.2f} MB")
    if skipped:
        print(f"\n{len(skipped)} file(s) skipped, see manifest.json")
    if errors:
        print(f"{len(errors)} file(s) failed to parse, see manifest.json")


if __name__ == "__main__":
    main()