"""
analysis/io.py
==============
Load the whole ``OUTPUTS_MEDICAL`` tree into four tidy pandas DataFrames.

This is the only place in the analysis layer that knows what a result file
looks like.  Every table and every figure downstream is a groupby on one of
these frames, so a change in the on-disk format is a change in one file.

Two on-disk formats, on purpose
-------------------------------
The drift stage has been through a rename and the tree contains both eras:

===================  ==========================  =========================
                     current                     legacy
===================  ==========================  =========================
classification       ``summary`` + ``per_sample`` ``margin_bound`` + top-level
                     + ``config`` + ``metadata``  ``ratios`` / ``margins``
segmentation         ``true_flip_rate``,          ``pixel_flip_rate``,
                     ``true_drop``,               ``true_dice_drop``,
                     ``predicted_drop``,          ``mean_dice_bound``,
                     with a ``metadata`` block    no ``metadata`` block
===================  ==========================  =========================

Reading only one spelling is what silently produced a column of NaN, or an
empty task, with nothing to show for it but a line in a manifest.  Every field
here is resolved through :func:`_first`, which takes the first spelling that is
actually present, and :func:`load_drift` records which layout each file used in
a ``format`` column so a mixed tree is visible rather than merely averaged.

Caching
-------
Walking the tree costs minutes, mostly in the drift files, which carry hundreds
of thousands of per-pixel values each.  ``--cache`` writes the parsed frames
next to the analysis outputs and reuses them until ``--refresh``.  Parquet is
used when ``pyarrow`` is installed and pickle otherwise, because the cluster
environment cannot be assumed to have it and a missing optional dependency
should not stop a table from being produced.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# Run-name parsing
# =============================================================================
def parse_run_name(name: str) -> dict:
    """
    Recover ``arch`` / ``freeze`` / ``lambda_sn`` / ``seed`` from a run folder.

    Mirrors :func:`shared.paths.parse_run_name`.  Duplicated rather than
    imported so the analysis layer can be pointed at a results tree copied off
    the cluster, with no project on the path.
    """
    seed, base = 0, name
    if "_seed" in base:
        base, _, tail = base.rpartition("_seed")
        if tail.isdigit():
            seed = int(tail)
        else:
            base = name

    if "_lambda" not in base:
        return {"arch": name, "freeze": "?", "lambda_sn": np.nan, "seed": seed}

    head, _, lam = base.rpartition("_lambda")
    arch, _, freeze = head.partition("_")
    try:
        lam_val = float(lam)
    except ValueError:
        lam_val = np.nan
    return {"arch": arch, "freeze": freeze, "lambda_sn": lam_val, "seed": seed}


def setting_label(info: dict) -> str:
    """``H``, ``H+SN``, ``H+L`` or ``H+L+SN`` from a parsed run name."""
    lam = info.get("lambda_sn")
    has_sn = bool(lam) and not (isinstance(lam, float) and np.isnan(lam))
    return ("H+L" if info["freeze"] == "head+last" else "H") + ("+SN" if has_sn else "")


ARCH_LABEL = {"resnet": "ResNet-34", "resnet34": "ResNet-34",
              "vit": "ViT-B/16", "vit_b_16": "ViT-B/16",
              "dinov2": "DINOv2"}

DATASET_LABEL = {
    "kvasir_v2": "Kvasir-v2", "hyperkvasir": "Hyper-Kvasir",
    "kvasir_capsule": "Kvasir-Capsule", "kvasir_seg": "Kvasir-SEG",
    "cvc_clinicdb": "CVC-ClinicDB", "cvc_colondb": "CVC-ColonDB",
    "etis_larib": "ETIS-Larib", "aircare_all": "AIRCARE (pooled)",
}

SETTING_ORDER = ["H", "H+SN", "H+L", "H+L+SN"]
ARCH_ORDER = ["resnet", "vit", "dinov2"]


# =============================================================================
# Helpers
# =============================================================================
def _first(d: dict, *keys, default=None):
    """First key of ``keys`` present and non-None in ``d``, else ``default``."""
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


def _f(x) -> float:
    """Anything -> float, with None and non-numbers becoming NaN."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _load(path: str, errors: list):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:
        errors.append({"path": path, "error": repr(exc)})
        return None


def _split_corruption_folder(folder: str) -> tuple[str, str, str]:
    """``<run>[_on_<evalds>]_<variant>`` -> ``(run, eval_ds, variant)``."""
    base, variant = folder, ""
    for suffix in ("_rawfeat_softmasking", "_rawfeat_norm_softmasking",
                   "_rawfeat", "_rawfeat_norm",
                   "_normfeat_softmasking", "_normfeat"):
        if base.endswith(suffix):
            variant = suffix.lstrip("_")
            base = base[: -len(suffix)]
            break
    eval_ds = ""
    if "_on_" in base:
        base, _, eval_ds = base.rpartition("_on_")
    return base, eval_ds, variant


def _decorate(rows: list[dict]) -> pd.DataFrame:
    """List of row dicts -> DataFrame with the ordered categorical labels."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["arch_label"] = df["arch"].map(lambda a: ARCH_LABEL.get(a, a))
    if "setting" in df:
        df["setting"] = pd.Categorical(df["setting"], SETTING_ORDER, ordered=True)
    df["arch"] = pd.Categorical(df["arch"], ARCH_ORDER, ordered=True)
    return df


# =============================================================================
# Stage 1 -- fine-tuning and the lambda ablation (same schema)
# =============================================================================
def load_finetuning(root: str, stage: str = "finetuning") -> pd.DataFrame:
    """
    One row per training run.

    Args:
        root  : ``OUTPUTS_MEDICAL``.
        stage : ``"finetuning"`` or ``"ablation"``.

    Returns:
        Columns: identity (task, dataset, arch, freeze, lambda_sn, setting,
        seed), the selected epoch, the validation and test metric under
        task-neutral names (``val`` / ``test``), ``test_iou`` where it exists,
        the spectral norm at the selected epoch and at the end, and the number
        of images the segmentation Dice was actually averaged over
        (``n_test_dice`` / ``n_test_images``; NaN for older logs and for
        classification).
    """
    rows, errors = [], []
    for task in ("classification", "segmentation"):
        base = os.path.join(root, stage, task)
        for path in sorted(glob.glob(os.path.join(base, "**", "*_log.json"),
                                     recursive=True)):
            log = _load(path, errors)
            if log is None:
                continue
            sel = log.get("selection") or {}
            if not sel:
                errors.append({"path": path, "error": "no 'selection' block"})
                continue
            folder = os.path.dirname(path)
            info = parse_run_name(os.path.basename(folder))
            curves = log.get("training_curves") or {}
            sn = curves.get("spectral_norm") or []
            best = int(sel.get("best_epoch", -1))
            rows.append({
                "stage": stage, "task": task,
                "dataset": log.get("dataset") or os.path.basename(os.path.dirname(folder)),
                **info, "setting": setting_label(info),
                "best_epoch": best,
                "val":  _f(_first(sel, "best_val_dice", "best_val_acc")),
                "test": _f(_first(sel, "test_dice", "test_acc")),
                "test_iou": _f(sel.get("test_iou")),
                # Present only in logs written after the Dice estimator was
                # unified; NaN elsewhere, which is itself informative.
                "n_test_dice":   _f(sel.get("n_test_dice")),
                "n_test_images": _f(sel.get("n_test_images")),
                "sigma": _f(sn[best]) if 0 <= best < len(sn) else float("nan"),
                "sigma_final": _f(sn[-1]) if sn else float("nan"),
                "n_epochs": len(sn),
                "path": path,
            })
    df = _decorate(rows)
    df.attrs["errors"] = errors
    return df


# =============================================================================
# Stage 2 -- corruption sweeps
# =============================================================================
_REGION = {"in": "corrupted", "out": "non_corrupted", "all": "global"}


def _feature_cols(region_dict: dict) -> dict:
    """``{region: {metric: value}}`` -> ``mse_in`` / ``cos_in`` / ... columns."""
    cols = {}
    for short, key in _REGION.items():
        block = region_dict.get(key) or {}
        cols[f"mse_{short}"] = _f(block.get("mse"))
        cols[f"cos_{short}"] = _f(block.get("cos"))
    return cols


def load_corruption(root: str, last_layer_only: bool = False,
                    progress: bool = True) -> pd.DataFrame:
    """
    One row per (run, experiment) for segmentation, per (run, experiment,
    hooked layer) for classification.

    Args:
        root            : ``OUTPUTS_MEDICAL``.
        last_layer_only : keep only the deepest hooked layer of each
                          classification sweep.  Drops about three quarters of
                          the rows and is what every body table wants; the
                          per-layer appendix figure needs them all.
        progress        : print a line per file batch.

    Returns:
        Task-neutral output columns (``metric_clean`` / ``metric_corr`` /
        ``metric_drop``) so both tasks share the schema, plus the segmentation
        extras (IoU, in/out Dice, the overlap strata and their counts).
    """
    rows, errors = [], []
    for task in ("classification", "segmentation"):
        base = os.path.join(root, "corruption_experiments", task)
        paths = sorted(glob.glob(os.path.join(base, "**", "metrics.json"),
                                 recursive=True))
        if progress and paths:
            print(f"  corruption/{task}: {len(paths)} files", flush=True)
        for i, path in enumerate(paths, 1):
            if progress and (i % 100 == 0 or i == len(paths)):
                print(f"    {i}/{len(paths)}", flush=True)
            blob = _load(path, errors)
            if blob is None:
                continue
            meta = blob.get("metadata") or {}
            cfg = blob.get("config") or {}
            folder = os.path.basename(os.path.dirname(path))
            run, eval_from_folder, variant = _split_corruption_folder(folder)
            info = parse_run_name(run)

            rel = os.path.relpath(os.path.dirname(path), base)
            train_ds = rel.split(os.sep)[0]
            # The metadata is authoritative: it names the dataset that was
            # actually loaded.  The folder is only a fallback for runs written
            # before the metadata block existed.
            eval_ds = meta.get("dataset") or eval_from_folder or train_ds

            common = {
                "task": task, "train_dataset": train_ds, "eval_dataset": eval_ds,
                "in_distribution": train_ds == eval_ds,
                "eval_split": meta.get("eval_split", ""),
                **info, "setting": setting_label(info),
                "seed": cfg.get("split_seed", info["seed"]),
                "variant": variant, "path": path,
            }

            exps = blob.get("all_experiments") or []
            if not exps:
                errors.append({"path": path, "error": "no experiments"})
                continue

            for exp in exps:
                row = dict(common)
                row.update({
                    "shape": exp.get("corruption_type"),
                    "ratio": _f(exp.get("perturbation_ratio")),
                    "eff_ratio": _f(exp.get("effective_ratio")),
                    "severity": exp.get("severity_level"),
                    "family": exp.get("perturbation_family"),
                    "ptype": exp.get("perturbation_name"),
                    "n_samples": exp.get("n_samples"),
                })

                if task == "segmentation":
                    ov = exp.get("dice_drop_by_overlap") or {}
                    nov = exp.get("n_by_overlap") or {}
                    row.update({
                        "metric_clean": _f(exp.get("dice_clean")),
                        "metric_corr":  _f(exp.get("dice_corrupted")),
                        "metric_drop":  _f(exp.get("dice_drop")),
                        "iou_clean": _f(exp.get("iou_clean")),
                        "iou_corr":  _f(exp.get("iou_corrupted")),
                        "dice_in":   _f(exp.get("dice_corrupted_in")),
                        "dice_out":  _f(exp.get("dice_corrupted_out")),
                        **{f"drop_{k}": _f(ov.get(k))
                           for k in ("none", "partial", "full", "no_lesion")},
                        **{f"n_{k}": nov.get(k)
                           for k in ("none", "partial", "full", "no_lesion")},
                        # Written only after the Dice estimator was unified.
                        "n_dice": exp.get("n_dice"),
                        "n_dice_out": exp.get("n_dice_out"),
                        "layer": "head_input",
                    })
                    head = dict(row)
                    head.update(_feature_cols(exp.get("feature_metrics") or {}))
                    rows.append(head)

                    # Optional depth study (``--hook_depth`` in the sweep): one
                    # extra row per hooked block.  The deepest probe is not in
                    # here -- it is the ``head_input`` row above -- so a layer
                    # never appears twice.  Absent in single-probe files, which
                    # then behave exactly as before.
                    for layer, block in (exp.get("layer_metrics") or {}).items():
                        lrow = dict(row)
                        lrow["layer"] = layer
                        lrow.update(_feature_cols(block or {}))
                        rows.append(lrow)
                else:
                    row.update({
                        "metric_clean": _f(exp.get("accuracy_clean")),
                        "metric_corr":  _f(exp.get("accuracy_corrupted")),
                        "metric_drop":  _f(exp.get("accuracy_drop")),
                    })
                    lm = exp.get("layer_metrics") or {}
                    layers = list(lm)
                    if last_layer_only and layers:
                        layers = layers[-1:]
                    for li, layer in enumerate(layers):
                        lrow = dict(row)
                        lrow["layer"] = layer
                        # Depth as a fraction, so ResNet's 4 hooks and the
                        # transformers' 13 can share an x axis.
                        lrow["layer_depth"] = (li / (len(lm) - 1)
                                               if len(lm) > 1 else 1.0)
                        lrow["layer_index"] = list(lm).index(layer)
                        lrow.update(_feature_cols(lm.get(layer) or {}))
                        rows.append(lrow)
    df = _decorate(rows)
    df.attrs["errors"] = errors
    return df


# =============================================================================
# Stage 3 -- drift / Lipschitz bound
# =============================================================================
_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _quantiles(arr, name: str) -> dict:
    out = {}
    a = None if arr is None else np.asarray(arr, dtype=float)
    for q in _QUANTILES:
        key = f"{name}_q{int(q * 100)}"
        if a is None or a.size == 0 or np.all(np.isnan(a)):
            out[key] = float("nan")
        else:
            out[key] = float(np.nanquantile(a, q))
    return out


def load_drift(root: str, points_per_run: int = 0, seed: int = 0,
               progress: bool = True):
    """
    One row per (checkpoint, evaluation dataset, perturbation).

    Reads both on-disk layouts (see the module docstring) and records which one
    each file used in a ``format`` column, so a tree containing runs from two
    eras is visible in the data rather than silently averaged.

    Args:
        root           : ``OUTPUTS_MEDICAL``.
        points_per_run : if > 0, also return a second frame with this many
                         per-unit samples per run, for the scatter plots.  The
                         per-pixel arrays are hundreds of thousands of values
                         per file, so keeping them all is neither necessary nor
                         possible.
        seed           : RNG seed of that subsample.
        progress       : print a line per file batch.

    Returns:
        ``summary_df`` or ``(summary_df, points_df)`` when ``points_per_run``.
    """
    rng = np.random.default_rng(seed)
    rows, points, errors = [], [], []

    for task in ("classification", "segmentation"):
        base = os.path.join(root, "drift_analysis", task)
        paths = sorted(glob.glob(os.path.join(base, "**", "*_results.json"),
                                 recursive=True))
        if progress and paths:
            print(f"  drift/{task}: {len(paths)} files", flush=True)
        for i, path in enumerate(paths, 1):
            if progress and (i % 200 == 0 or i == len(paths)):
                print(f"    {i}/{len(paths)}", flush=True)
            blob = _load(path, errors)
            if blob is None:
                continue

            folder = os.path.dirname(path)
            parts = os.path.relpath(folder, base).split(os.sep)
            train_ds = parts[0] if parts else ""
            # .../<train_ds>/<drift_tag>/<run_name>/
            run = parts[-1] if len(parts) >= 3 else os.path.basename(folder)
            info = parse_run_name(run)

            cfg = blob.get("config") or {}
            summary = blob.get("summary") or {}
            legacy_mb = blob.get("margin_bound") or {}
            src = {**legacy_mb, **summary}          # summary wins on conflict
            if not src:
                errors.append({"path": path,
                               "error": "neither 'summary' nor 'margin_bound'"})
                continue

            if task == "segmentation":
                px = blob.get("pixels") or {}
                arrays = {"ratio": px.get("ratio"), "dz": px.get("dz_norm"),
                          "dlogit": px.get("dlogit_norm"), "margin": px.get("margin")}
                fmt = ("legacy" if "pixel_flip_rate" in summary else "current")
                extra = {
                    "metric_clean": _f(src.get("mean_dice_clean")),
                    "metric_corr":  _f(src.get("mean_dice_corrupted")),
                    "mean_dz_norm": _f(src.get("mean_dz_norm")),
                    "mean_dlogit_norm": _f(src.get("mean_dlogit_norm")),
                    "n_units":  src.get("n_pixels"),
                    "n_images": src.get("n_images"),
                    # The drift Dice lives on the feature grid, not at full
                    # resolution: never put it in the same column as the
                    # corruption or baseline Dice.
                    "dice_resolution": src.get("dice_resolution", "feature_grid"),
                }
            else:
                per = blob.get("per_sample") or {}
                # The classification script saves ``upper = sigma * ||dz||``
                # rather than ||dz|| itself.  Dividing it back out recovers the
                # feature drift exactly (sigma is a constant of the run), which
                # is what puts the two tasks on the same axis: the segmentation
                # script saves ``dz_norm`` directly.
                sigma_run = _f(_first(src, "sigma_max", "spectral_norm_W"))
                upper = _first(per, "upper", default=None)
                dz = (list(np.asarray(upper, dtype=float) / sigma_run)
                      if upper is not None and sigma_run > 0
                      else blob.get("drift_values"))
                arrays = {
                    "ratio": _first(per, "ratio", default=blob.get("ratios")),
                    "dz": dz,
                    "dlogit": _first(per, "drift", default=None),
                    "margin": _first(per, "margin", default=blob.get("margins")),
                }
                fmt = "legacy" if legacy_mb and not summary else "current"
                extra = {
                    "metric_clean": _f(src.get("clean_acc")),
                    "metric_corr":  float("nan"),
                    "mean_dz_norm": _f(src.get("mean_upper")),
                    "mean_dlogit_norm": _f(src.get("mean_drift")),
                    "n_units":  _first(src, "n_samples", default=blob.get("num_samples")),
                    "n_images": _first(src, "n_samples", default=blob.get("num_samples")),
                    "dice_resolution": "",
                }

            row = {
                "task": task, "train_dataset": train_ds,
                "eval_dataset": _first(cfg, "dataset", default=blob.get("dataset", "")),
                **info, "setting": setting_label(info),
                "seed": _first(cfg, "split_seed", default=info["seed"]),
                "perturbation": _first(cfg, "perturbation",
                                       default=blob.get("perturbation")),
                "severity": _first(cfg, "severity", default=blob.get("severity")),
                "eps": _first(cfg, "eps", default=blob.get("eps")),
                "format": fmt,
                "sigma_max": _f(_first(src, "sigma_max", "spectral_norm_W")),
                "gamma": _f(src.get("gamma")),
                "mean_ratio": _f(src.get("mean_ratio")),
                "std_ratio": _f(src.get("std_ratio")),
                # Both spellings of the three renamed fields.
                "flip_rate": _f(_first(src, "true_flip_rate", "pixel_flip_rate")),
                "certified_flip_rate": _f(src.get("certified_flip_rate")),
                "true_drop": _f(_first(src, "true_drop", "true_dice_drop",
                                       "true_acc_drop")),
                "predicted_drop": _f(_first(src, "predicted_drop",
                                            "mean_dice_bound",
                                            "predicted_acc_drop")),
                "bound_violations": src.get("bound_violations"),
                "bound_violation_rate": _f(src.get("bound_violation_rate")),
                # Exact worst-case Dice bound; absent for classification and
                # for segmentation runs written before it existed.
                "predicted_drop_exact": _f(src.get("predicted_drop_exact")),
                "bound_violations_exact": src.get("bound_violations_exact"),
                "bound_violation_rate_exact": _f(src.get("bound_violation_rate_exact")),
                "mean_tightness": _f(_first(src, "mean_tightness", "tightness_flip")),
                **extra,
                "path": path,
            }
            # sqrt(2)*sigma is the a-priori bound on Gamma; how far below it
            # Gamma actually sits is a result, so it is materialised here.
            row["gamma_over_sqrt2_sigma"] = (
                row["gamma"] / (np.sqrt(2) * row["sigma_max"])
                if row["sigma_max"] > 0 else float("nan"))
            for name in ("ratio", "dz", "dlogit", "margin"):
                row.update(_quantiles(arrays.get(name), name))
            rows.append(row)

            if points_per_run > 0:
                arr = {k: np.asarray(v, dtype=float)
                       for k, v in arrays.items() if v is not None and len(v)}
                if arr:
                    n = min(a.shape[0] for a in arr.values())
                    k = min(points_per_run, n)
                    if k > 0:
                        sel = rng.choice(n, size=k, replace=False)
                        block = pd.DataFrame({name: a[sel] for name, a in arr.items()})
                        for col in ("task", "train_dataset", "eval_dataset", "arch",
                                    "setting", "seed", "perturbation"):
                            block[col] = row[col]
                        points.append(block)

    df = _decorate(rows)
    df.attrs["errors"] = errors
    if points_per_run <= 0:
        return df
    pts = pd.concat(points, ignore_index=True) if points else pd.DataFrame()
    return df, pts


# =============================================================================
# The portable results pack
# =============================================================================
PACK_FILES = ("finetuning", "ablation", "corruption", "drift", "drift_points")


def is_pack(root: str) -> bool:
    """True when ``root`` is a ``results_pack`` directory rather than a tree.

    ``analysis/pack_results.py`` reduces the whole ``OUTPUTS_MEDICAL`` tree to a
    handful of gzipped CSVs, which is what travels off the cluster.  Every
    script takes ``--results_root``, so recognising the pack here means the
    figures are reproducible from the pack with no other change.
    """
    return (os.path.isdir(root)
            and os.path.exists(os.path.join(root, "manifest.json"))
            and any(os.path.exists(os.path.join(root, f"{n}.csv.gz"))
                    for n in PACK_FILES))


def load_pack(root: str, progress: bool = True) -> dict:
    """Read a ``results_pack`` directory into the same frames as the tree.

    The pack is written from the parsed frames, so the columns already match;
    what it does not carry is the decorations the rest of the package relies on
    -- the ordered categoricals, ``in_distribution``, the derived ratio of
    Gamma to its a-priori bound -- and those are restored here rather than at
    every call site.  Two columns cannot be restored and are simply absent:
    ``path``, and the ``format`` diagnostic that records which of the two
    on-disk drift layouts a run was read from.  Anything that wants them has to
    say so rather than assume them.
    """
    if progress:
        print(f"Reading the results pack in {root}")
    out = {}
    for name in PACK_FILES:
        path = os.path.join(root, f"{name}.csv.gz")
        if not os.path.exists(path):
            out[name] = pd.DataFrame()
            continue
        df = pd.read_csv(path, compression="gzip", low_memory=False)
        if not df.empty:
            if {"train_dataset", "eval_dataset"} <= set(df.columns):
                df["in_distribution"] = df.train_dataset == df.eval_dataset
            # Derived in :func:`load_drift` and not written to the pack, which
            # carries what was measured.  Recomputed rather than dropped, so a
            # table does not have to know where its frame came from.
            if {"gamma", "sigma_max"} <= set(df.columns):
                df["gamma_over_sqrt2_sigma"] = np.where(
                    df.sigma_max > 0, df.gamma / (np.sqrt(2) * df.sigma_max),
                    np.nan)
            if "arch" in df:
                df["arch_label"] = df["arch"].map(lambda a: ARCH_LABEL.get(a, a))
                df["arch"] = pd.Categorical(df["arch"], ARCH_ORDER, ordered=True)
            if "setting" in df:
                df["setting"] = pd.Categorical(df["setting"], SETTING_ORDER,
                                               ordered=True)
        out[name] = df
        if progress:
            print(f"  {name:14s} {len(df):>8,} rows")
    return out


# =============================================================================
# One-call loader, with a cache
# =============================================================================

def _cache_backend():
    """
    ``(extension, writer, reader)`` for the cache.

    Parquet is smaller and faster but needs ``pyarrow``, which is not part of
    ``requirements.txt`` and is often missing on a cluster login node.  Pickle
    is always available and round-trips a DataFrame exactly, so it is the
    fallback rather than an error.
    """
    try:
        import pyarrow  # noqa: F401
        return (".parquet",
                lambda df, p: df.to_parquet(p, index=False),
                pd.read_parquet)
    except ImportError:
        return (".pkl",
                lambda df, p: df.to_pickle(p),
                pd.read_pickle)


def load_all(root: str, cache_dir: str | None = None, refresh: bool = False,
             points_per_run: int = 300, progress: bool = True) -> dict:
    """
    Every frame the analysis scripts need, parsed once and cached.

    Args:
        root           : ``OUTPUTS_MEDICAL``.
        cache_dir      : where to write the Parquet cache; ``None`` disables it.
        refresh        : re-read the tree even when the cache is present.
        points_per_run : drift samples kept per run for the scatter plots.
        progress       : print progress while walking the tree.

    Returns:
        Dict with keys ``finetuning``, ``ablation``, ``corruption``, ``drift``,
        ``drift_points``.
    """
    if is_pack(root):
        return load_pack(root, progress=progress)

    names = ("finetuning", "ablation", "corruption", "drift", "drift_points")
    ext, writer, reader = _cache_backend()

    if cache_dir and not refresh:
        paths = {n: os.path.join(cache_dir, f"{n}{ext}") for n in names}
        if all(os.path.exists(p) for p in paths.values()):
            if progress:
                print(f"Reading the cache in {cache_dir} (use --refresh to rebuild)")
            return {n: reader(p) for n, p in paths.items()}

    # Every script shares one cache directory, so the cache has to be written
    # complete: a script that wants no scatter points (``points_per_run=0``,
    # which is what the insight scripts pass) would otherwise leave an empty
    # ``drift_points`` behind for the next script to read as if it were the
    # data.  When a cache is being built, the per-unit samples are collected
    # regardless; the extra cost is a subsample of arrays that have already
    # been parsed.
    if cache_dir and points_per_run <= 0:
        points_per_run = 300

    if progress:
        print(f"Reading {root}")
    out = {
        "finetuning": load_finetuning(root, "finetuning"),
        "ablation":   load_finetuning(root, "ablation"),
        "corruption": load_corruption(root, progress=progress),
    }
    if points_per_run > 0:
        out["drift"], out["drift_points"] = load_drift(
            root, points_per_run=points_per_run, progress=progress)
    else:
        # load_drift returns the summary alone in this case, not a pair.
        out["drift"] = load_drift(root, points_per_run=0, progress=progress)
        out["drift_points"] = pd.DataFrame()

    n_err = sum(len(df.attrs.get("errors", [])) for df in out.values()
                if hasattr(df, "attrs"))
    if progress:
        for name, df in out.items():
            print(f"  {name:14s} {len(df):>8,} rows")
        if n_err:
            print(f"  {n_err} file(s) could not be parsed; see .attrs['errors']")

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        for name, df in out.items():
            # The ordered categoricals survive either backend; the ``attrs``
            # do not, which is fine since they are parse diagnostics and are
            # printed above.
            writer(df, os.path.join(cache_dir, f"{name}{ext}"))
        if progress:
            print(f"  cached to {cache_dir} ({ext})")
    return out
