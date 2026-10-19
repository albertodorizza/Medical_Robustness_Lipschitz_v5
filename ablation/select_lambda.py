#!/usr/bin/env python3
"""
ablation/select_lambda.py
================
Pick lambda* from the ablation runs, on VALIDATION, for both tasks.

Reads the JSON logs written by ``finetuning_cls.py`` / ``finetuning_seg.py`` under
the ablation tree::

    <ablation_root>/<task>/<dataset>/<run_name>/<model>_<freeze>_log.json

Selection rule (fixed before looking at the numbers9:

1. For every (architecture, freeze mode) and every lambda, the validation
   metric -- accuracy for classification, Dice for segmentation -- is compared
   to lambda = 0 **seed by seed**, and the per-seed differences are averaged.
   Pairing removes the split-to-split variance, which is much larger than the
   effect being measured.
2. The noise level is estimated by POOLING the paired differences of every
   (architecture, freeze mode) at that lambda, after centring each combination
   on its own mean.  A per-combination standard deviation over a handful of
   seeds has too few degrees of freedom to threshold on; pooling six
   combinations multiplies them by six.  The tolerance is ``--tol-sigma``
   standard errors of that pooled estimate, floored at ``--min-tol``:

       tol(lambda) = max( tol_sigma * sd_pooled(lambda) / sqrt(n_seeds),
                          min_tol )

   It is deliberately soft: the rule is applied at every lambda to every
   combination, so a tighter threshold would reject an admissible lambda by
   chance somewhere in the grid and the contiguity requirement below would
   propagate it.  Because the first term grows where the runs are noisy, the
   tolerance is NOT constant across the grid -- report it as a column, not as
   a single number.
3. A lambda is *admissible* if, for EVERY combination, the mean paired
   difference is no worse than minus that tolerance, and every smaller lambda
   on the grid is admissible too (so a fluke past the collapse cannot be
   picked).  The floor ``--min-tol`` is an equivalence margin -- two metric
   points by default -- and it, not the noise estimate, is what usually binds:
   a purely random dip of one point is common with few seeds, while the
   collapse this rule has to catch is tens of points.
4. lambda* is the largest admissible lambda.  ``--margin`` optionally backs
   off by that many grid steps, for a finer grid where sitting one step inside
   the plateau costs little; on the default half-decade grid a step is a factor
   of ~3, so the default is 0.

Robustness reporting (this is what makes the choice quotable):

* the BINDING constraint -- which combination rejects the first non-admissible
  lambda, and by how much;
* the SLACK at lambda*, as a fraction of the tolerance.  A selection whose
  worst combination sits a few percent of the tolerance away from rejection is
  decided by seed noise, not by the data: it is flagged as FRAGILE, and
  ``--margin 1`` (or more seeds) is the fix;
* a SENSITIVITY sweep of lambda* over several values of ``--min-tol``, so the
  paper can show the choice does not hang on the equivalence margin;
* the seed count per combination, with a warning when it is not uniform (the
  paired deltas silently use the intersection of the seeds available).

The test split is never read here.

Outputs: a console report, a LaTeX table per task, a tolerance-sensitivity
LaTeX table, a 2x2 figure per task (validation metric and head spectral norm,
one column per freeze mode) and a machine-readable JSON for run_all.sh.

Usage:
    python3 ablation/select_lambda.py /home/a.dorizza/OUTPUTS_MEDICAL/ablation
    python3 ablation/select_lambda.py <root> --outdir paper_assets --margin 1
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

METRIC = {
    "classification": ("best_val_acc", "Validation accuracy"),
    "segmentation":   ("best_val_dice", "Validation Dice"),
}
MODEL_LABEL = {"resnet": "ResNet", "vit": "ViT", "dinov2": "DINOv2"}

# Curve names that, if present in the logs, let the report quote the relative
# weight of the penalty against the data term -- the quantity that makes two
# different lambda* values comparable across tasks.  Unknown schemas are simply
# skipped: nothing here is required.
PENALTY_KEYS = ("reg_loss", "penalty", "sn_loss", "spectral_loss",
                "lambda_penalty", "reg_term")
DATA_KEYS = ("train_loss_data", "data_loss", "task_loss", "train_loss")


# =============================================================================
# Loading
# =============================================================================
def load_runs(root: Path) -> tuple[dict, dict, dict]:
    """Return {task: {(model, freeze): {lambda: {seed: (metric, sigma)}}}}."""
    runs: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    datasets: dict = {}
    curve_keys: dict = defaultdict(set)
    ratios: dict = defaultdict(lambda: defaultdict(dict))
    n_files = 0

    for path in sorted(root.glob("*/*/*/*_log.json")):
        with open(path) as f:
            log = json.load(f)

        task = log["task"]
        if task not in METRIC:
            print(f"  ! unknown task '{task}' in {path}, skipped")
            continue

        metric_key = METRIC[task][0]
        hp = log["hyperparameters"]
        sel = log["selection"]

        if metric_key not in sel:
            print(f"  ! {metric_key} missing in {path}, skipped")
            continue

        best_epoch = int(sel["best_epoch"])
        curves = log["training_curves"]
        curve = curves["spectral_norm"]
        sigma = curve[best_epoch] if 0 <= best_epoch < len(curve) else float("nan")

        key = (log["model_type"], log["freeze_mode"])
        lam = float(hp["lambda_sn"])
        seed = int(hp["split_seed"])

        runs[task][key][lam][seed] = (float(sel[metric_key]), float(sigma))
        datasets.setdefault(task, set()).add(log["dataset"])
        curve_keys[task].update(curves.keys())

        # Opportunistic: penalty / data-term ratio at the selected epoch.
        pen = _curve_at(curves, PENALTY_KEYS, best_epoch)
        dat = _curve_at(curves, DATA_KEYS, best_epoch)
        if pen is not None and dat:
            ratios[task][key].setdefault(lam, []).append(abs(pen) / abs(dat))

        n_files += 1

    print(f"Loaded {n_files} run logs from {root}")
    for task, names in datasets.items():
        print(f"  {task:<15} dev dataset(s): {', '.join(sorted(names))}")
    return runs, datasets, {"curve_keys": curve_keys, "ratios": ratios}


def _curve_at(curves: dict, candidates: tuple, epoch: int):
    """First candidate curve present in `curves`, sampled at `epoch`."""
    for name in candidates:
        seq = curves.get(name)
        if isinstance(seq, list) and 0 <= epoch < len(seq):
            try:
                return float(seq[epoch])
            except (TypeError, ValueError):
                continue
    return None


# =============================================================================
# Selection
# =============================================================================
def aggregate(by_lambda: dict) -> dict:
    """{lambda: {seed: (metric, sigma)}} -> {lambda: (mean, std, n, sigma_mean)}."""
    out = {}
    for lam, per_seed in by_lambda.items():
        metrics = [m for m, _ in per_seed.values()]
        sigmas = [s for _, s in per_seed.values()]
        std = statistics.stdev(metrics) if len(metrics) > 1 else 0.0
        out[lam] = (statistics.fmean(metrics), std, len(metrics),
                    statistics.fmean(sigmas))
    return out


def paired_deltas(by_lambda: dict, lam: float):
    """Per-seed differences vs lambda=0, over the seeds both runs share."""
    base = by_lambda.get(0.0, {})
    cur = by_lambda.get(lam, {})
    seeds = sorted(set(base) & set(cur))
    if not seeds:
        return None
    return [cur[s][0] - base[s][0] for s in seeds]


def pooled_tolerance(all_deltas: list[list[float]], min_tol: float,
                     tol_sigma: float) -> float:
    """Tolerance from the within-combination spread, pooled over combinations.

    Each combination is centred on its own mean first, so a real effect in one
    of them does not inflate the noise estimate.
    """
    residuals, dof, n_min = [], 0, None
    for deltas in all_deltas:
        if len(deltas) < 2:
            continue
        m = statistics.fmean(deltas)
        residuals += [d - m for d in deltas]
        dof += len(deltas) - 1
        n_min = len(deltas) if n_min is None else min(n_min, len(deltas))
    if dof == 0 or not n_min:
        return min_tol
    pooled_sd = math.sqrt(sum(r * r for r in residuals) / dof)
    return max(tol_sigma * pooled_sd / math.sqrt(n_min), min_tol)


def compute_deltas(task_runs: dict, grid: list, min_tol: float,
                   tol_sigma: float) -> dict:
    """deltas[key][lam] = (mean paired difference, tolerance, n pairs)."""
    raw = {key: {lam: paired_deltas(by_lambda, lam) for lam in grid}
           for key, by_lambda in task_runs.items()}

    deltas = {key: {} for key in task_runs}
    for lam in grid:
        per_combo = [raw[key][lam] for key in task_runs if raw[key][lam]]
        tol = pooled_tolerance(per_combo, min_tol, tol_sigma)
        for key in task_runs:
            d = raw[key][lam]
            if d:
                deltas[key][lam] = (statistics.fmean(d), tol, len(d))
    return deltas


def worst_at(task_runs: dict, deltas: dict, lam: float):
    """Worst combination at `lam`.

    Returns (slack, key, mean_delta, tol) where slack = mean_delta + tol, so
    slack < 0 means the lambda is rejected.  Returns (None, key, ...) when some
    combination has no run at this lambda -- a partially evaluated lambda is
    never selectable, so that a crashed run cannot be mistaken for a pass.
    """
    rows = []
    for key in sorted(task_runs):
        if lam not in deltas[key]:
            return (None, key, None, None)
        mean_delta, tol, _ = deltas[key][lam]
        rows.append((mean_delta + tol, key, mean_delta, tol))
    return min(rows)


def largest_admissible(task_runs: dict, grid: list, deltas: dict):
    """Walk the sorted grid until the first rejection.  Returns (lam, binding)."""
    largest_ok = 0.0
    binding = None
    for lam in grid:
        slack, key, mean_delta, tol = worst_at(task_runs, deltas, lam)
        if slack is None:
            binding = (lam, key, None, None, "not evaluated for this combination")
            break
        if slack < 0:
            binding = (lam, key, mean_delta, tol, "below tolerance")
            break
        largest_ok = lam
    return largest_ok, binding


def select(task_runs: dict, min_tol: float, tol_sigma: float, margin: int):
    """Apply the selection rule and everything needed to defend it."""
    grid = sorted({lam for combo in task_runs.values() for lam in combo})
    if not grid or grid[0] != 0.0:
        raise SystemExit("The grid must contain lambda = 0 as the baseline.")

    agg = {key: aggregate(by_lambda) for key, by_lambda in task_runs.items()}
    deltas = compute_deltas(task_runs, grid, min_tol, tol_sigma)
    largest_ok, binding = largest_admissible(task_runs, grid, deltas)

    idx = grid.index(largest_ok)
    lambda_star = grid[max(idx - margin, 0)]
    backed_off = grid[max(idx - 1, 0)]

    # How much room the worst combination has at lambda*, in units of the
    # tolerance.  Small => the choice is decided by seed noise.
    slack, worst_key, worst_delta, worst_tol = worst_at(task_runs, deltas, lambda_star)
    slack_frac = (slack / worst_tol) if (slack is not None and worst_tol) else None

    return {
        "lambda_star": lambda_star,
        "largest_ok": largest_ok,
        "backed_off": backed_off,
        "grid": grid,
        "agg": agg,
        "deltas": deltas,
        "binding": binding,
        "worst": (slack, worst_key, worst_delta, worst_tol),
        "slack_frac": slack_frac,
    }


def tol_sensitivity(task_runs: dict, tol_grid: list, tol_sigma: float,
                    margin: int) -> list:
    """lambda* as a function of the equivalence margin --min-tol."""
    grid = sorted({lam for combo in task_runs.values() for lam in combo})
    out = []
    for min_tol in tol_grid:
        deltas = compute_deltas(task_runs, grid, min_tol, tol_sigma)
        largest_ok, _ = largest_admissible(task_runs, grid, deltas)
        idx = grid.index(largest_ok)
        out.append((min_tol, grid[max(idx - margin, 0)]))
    return out


def seed_census(task_runs: dict) -> dict:
    """{(model, freeze): {lambda: n_seeds}} plus the min/max over the grid."""
    counts = {key: {lam: len(seeds) for lam, seeds in by_lambda.items()}
              for key, by_lambda in task_runs.items()}
    flat = [n for per_lam in counts.values() for n in per_lam.values()]
    return {"per_combo": counts,
            "min": min(flat) if flat else 0,
            "max": max(flat) if flat else 0}


# =============================================================================
# Reporting
# =============================================================================
def report(task: str, res: dict, census: dict, sens: list, ratios: dict,
           min_tol: float, tol_sigma: float, margin: int) -> None:
    agg, grid, deltas = res["agg"], res["grid"], res["deltas"]
    lambda_star = res["lambda_star"]
    _, metric_label = METRIC[task]

    n_lo, n_hi = census["min"], census["max"]
    seeds_txt = f"{n_lo}" if n_lo == n_hi else f"{n_lo}-{n_hi}"

    print(f"\n{'=' * 78}\n{task.upper()}  --  {metric_label} "
          f"(mean +- sd over seeds; delta is paired vs lambda=0)\n{'=' * 78}")
    print(f"  rule: keep the largest lambda whose every (arch, freeze) stays "
          f"within tolerance of lambda=0")
    print(f"  tol(lambda) = max({tol_sigma:g} * sd_pooled / sqrt(n), "
          f"{min_tol:g})   [not constant across the grid]")

    if n_lo != n_hi:
        print(f"  ! seed count is NOT uniform ({seeds_txt} seeds): paired "
              f"deltas use the intersection of the available seeds.")
        for key in sorted(census["per_combo"]):
            per_lam = census["per_combo"][key]
            odd = {lam: n for lam, n in sorted(per_lam.items()) if n != n_hi}
            if odd:
                print(f"      {MODEL_LABEL.get(key[0], key[0]):<8} {key[1]:<10} "
                      + ", ".join(f"lambda={lam:g}:n={n}" for lam, n in odd.items()))

    for key in sorted(agg):
        model, freeze = key
        n = agg[key][0.0][2]
        print(f"\n  {MODEL_LABEL.get(model, model)}  |  {freeze}   (n={n} seeds)")
        print(f"    {'lambda':>10}  {'metric':>16}  {'sigma':>8}  "
              f"{'delta':>9}  {'tol':>7}")
        for lam in grid:
            if lam not in agg[key]:
                print(f"    {lam:>10.6g}  {'missing':>16}")
                continue
            mean, std, _, sigma = agg[key][lam]
            if lam in deltas[key]:
                mean_delta, tol, _ = deltas[key][lam]
                flag = "" if mean_delta >= -tol else "  <-- below tolerance"
                extra = f"  {mean_delta:>+9.4f}  {tol:>7.4f}{flag}"
            else:
                extra = ""
            print(f"    {lam:>10.6g}  {mean:>8.4f} +- {std:<5.4f}  "
                  f"{sigma:>8.4f}{extra}")

    print(f"\n  Largest admissible lambda : {res['largest_ok']:g}")
    print(f"  Margin                    : {margin} grid step(s)")
    print(f"  ==> lambda* ({task})      : {lambda_star:g}")

    for key in sorted(agg):
        if lambda_star in agg[key]:
            mean, _, _, sigma = agg[key][lambda_star]
            base_sigma = agg[key][0.0][3]
            d = deltas[key].get(lambda_star, (float("nan"),))[0]
            drop = (1 - sigma / base_sigma) * 100 if base_sigma else float("nan")
            print(f"       {MODEL_LABEL.get(key[0], key[0]):<8} {key[1]:<10} "
                  f"metric {mean:.4f} ({d:+.4f})   "
                  f"sigma {sigma:.4f} (was {base_sigma:.4f}, {drop:+.0f}%)")

    # ---- what actually decided the choice -----------------------------------
    print(f"\n  {'-' * 74}\n  Robustness of the selection\n  {'-' * 74}")
    if res["binding"]:
        lam, key, mean_delta, tol, why = res["binding"]
        label = f"{MODEL_LABEL.get(key[0], key[0])} / {key[1]}"
        if mean_delta is None:
            print(f"  Rejected at lambda={lam:g} by {label}: {why}.")
        else:
            print(f"  Rejected at lambda={lam:g} by {label}: "
                  f"delta {mean_delta:+.4f} vs tol {tol:.4f} ({why}).")
    else:
        print("  No lambda on the grid was rejected: lambda* is at the top edge "
              "and the plateau is not bracketed.")

    slack, worst_key, worst_delta, worst_tol = res["worst"]
    if slack is not None:
        label = f"{MODEL_LABEL.get(worst_key[0], worst_key[0])} / {worst_key[1]}"
        print(f"  Tightest combination at lambda*: {label}  "
              f"delta {worst_delta:+.4f}, tol {worst_tol:.4f}  "
              f"=> slack {slack:+.4f} ({100 * res['slack_frac']:.0f}% of tol)")
        if res["slack_frac"] < FRAGILE_FRAC:
            print(f"  !! FRAGILE: the worst combination clears the threshold by "
                  f"less than {100 * FRAGILE_FRAC:.0f}% of the tolerance.")
            print(f"     A different draw of seeds can flip lambda* to "
                  f"{res['backed_off']:g}.  Add seeds, or take --margin 1.")
        else:
            print("  Selection is not decided by noise "
                  f"(slack >= {100 * FRAGILE_FRAC:.0f}% of tol).")

    print(f"  lambda* with --margin 1   : {res['backed_off']:g}")

    print("\n  Sensitivity to the equivalence margin (--min-tol):")
    print(f"    {'min_tol':>9}  {'lambda*':>9}")
    for tol_value, lam in sens:
        mark = "   <-- default" if abs(tol_value - min_tol) < 1e-12 else ""
        print(f"    {tol_value:>9.3f}  {lam:>9g}{mark}")

    # ---- penalty / data term, if the logs carry it ---------------------------
    if ratios:
        print("\n  Penalty / data-term ratio at the selected epoch "
              "(makes lambda* comparable across tasks):")
        for key in sorted(ratios):
            per_lam = ratios[key]
            if lambda_star in per_lam:
                r = statistics.fmean(per_lam[lambda_star])
                print(f"       {MODEL_LABEL.get(key[0], key[0]):<8} {key[1]:<10} "
                      f"{r:.4f}")


FRAGILE_FRAC = 0.25


def tex_escape(s: str) -> str:
    """Escape the characters that appear in dataset keys (underscores)."""
    return s.replace("\\", r"\textbackslash{}").replace("_", r"\_")


def latex_table(task: str, agg: dict, grid: list, deltas: dict,
                lambda_star: float, dataset: str, n_seeds: str,
                out: Path) -> None:
    _, metric_label = METRIC[task]
    short = {"classification": r"Val.\ Acc.", "segmentation": "Dice"}[task]
    dataset = ", ".join(rf"\texttt{{{tex_escape(d.strip())}}}" for d in dataset.split(","))
    keys = sorted(agg)
    lines = [
        r"\begin{table*}[!tp]", r"  \centering", r"  \small",
        rf"  \caption{{{metric_label} and head spectral norm $\sigma$ as a function of "
        rf"the spectral-norm coefficient $\lambda$ on {dataset} "
        rf"(mean $\pm$ s.d. over {n_seeds} seeds). $\lambda$ is selected on "
        rf"validation only. The last column is the admissibility tolerance. "
        rf"The selected value $\lambda^\star={lambda_star:g}$ is in bold.}}",
        rf"  \label{{tab:ablation-lambda-{task}}}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{l" + "cc" * len(keys) + "c}",
        r"    \toprule",
        "    $\\lambda$ & " + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{{MODEL_LABEL.get(m, m)} / {f}}}" for m, f in keys)
        + r" & \\",
        "    " + " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(keys))),
        "     & " + " & ".join(rf"{short} & $\sigma$" for _ in keys) + r" & tol \\",
        r"    \midrule",
    ]
    for lam in grid:
        cells = []
        tol_cell = "--"
        for key in keys:
            if lam in agg[key]:
                mean, std, _, sigma = agg[key][lam]
                cells += [rf"${mean:.3f}\pm{std:.3f}$", rf"${sigma:.3f}$"]
            else:
                cells += ["--", "--"]
            if lam in deltas[key]:
                tol_cell = rf"${deltas[key][lam][1]:.3f}$"
        label = rf"\textbf{{{lam:g}}}" if lam == lambda_star else f"{lam:g}"
        lines.append(f"    {label} & " + " & ".join(cells) + f" & {tol_cell}" + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}%", r"  }", r"\end{table*}", ""]

    out.write_text("\n".join(lines))
    print(f"  [LaTeX] {out}")


def latex_sensitivity(sens_by_task: dict, out: Path) -> None:
    """One small table: lambda* per task as a function of the equivalence margin."""
    tasks = sorted(sens_by_task)
    tol_values = [t for t, _ in sens_by_task[tasks[0]]]
    lines = [
        r"\begin{table}[!htbp]", r"  \centering", r"  \small",
        r"  \caption{Sensitivity of the selected coefficient $\lambda^\star$ to the "
        r"equivalence margin $\varepsilon$ used by the selection rule. The choice is "
        r"stable over an order of magnitude of the margin, and the two tasks never "
        r"coincide.}",
        r"  \label{tab:ablation-lambda-tol-sensitivity}",
        r"  \begin{tabular}{l" + "c" * len(tasks) + "}",
        r"    \toprule",
        r"    Equivalence margin $\varepsilon$ & " + " & ".join(t.capitalize() for t in tasks) + r" \\",
        r"    \midrule",
    ]
    for i, tol_value in enumerate(tol_values):
        cells = [f"{sens_by_task[t][i][1]:g}" for t in tasks]
        lines.append(rf"    {tol_value:.3f} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    out.write_text("\n".join(lines))
    print(f"\n  [LaTeX] {out}")


def figure(task: str, agg: dict, grid: list, lambda_star: float,
           dataset: str, n_seeds: str, out_stem: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [figure] matplotlib not available, skipped")
        return

    _, metric_label = METRIC[task]
    freezes = sorted({f for _, f in agg})
    models = sorted({m for m, _ in agg})
    x = list(range(len(grid)))
    labels = [f"{lam:g}" for lam in grid]
    star_x = grid.index(lambda_star)

    fig, axes = plt.subplots(2, len(freezes), figsize=(5.2 * len(freezes), 7.2),
                             sharex=True)
    axes = axes.reshape(2, len(freezes))

    for col, freeze in enumerate(freezes):
        for row, what in enumerate(("metric", "sigma")):
            ax = axes[row][col]
            for model in models:
                key = (model, freeze)
                if key not in agg:
                    continue
                xs, ys, es = [], [], []
                for i, lam in enumerate(grid):
                    if lam not in agg[key]:
                        continue
                    mean, std, _, sigma = agg[key][lam]
                    xs.append(i)
                    ys.append(mean if what == "metric" else sigma)
                    es.append(std if what == "metric" else 0.0)
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                            label=MODEL_LABEL.get(model, model))
            ax.axvline(star_x, ls="--", color="tab:red", lw=1)
            ax.annotate(rf"$\lambda^\star={lambda_star:g}$",
                        xy=(star_x, 0.97), xycoords=("data", "axes fraction"),
                        ha="left", va="top", fontsize=8, color="tab:red")
            ax.grid(alpha=0.3, ls=":")
            ax.set_ylabel(metric_label if what == "metric" else r"Spectral norm $\sigma$")
            if row == 0:
                ax.set_title(f"{freeze}")
                ax.legend(fontsize=8)
            else:
                ax.set_xlabel(r"$\lambda$")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha="right")

    fig.suptitle(f"{task.capitalize()} -- $\\lambda$ ablation on {dataset} "
                 f"(validation, {n_seeds} seeds)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {out_stem}.pdf / .png")


# =============================================================================
# Entry point
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path,
                    help="Ablation root, e.g. <RESULTS_ROOT>/ablation")
    ap.add_argument("--outdir", type=Path, default=Path("paper_assets"))
    ap.add_argument("--margin", type=int, default=0,
                    help="Grid steps to back off from the largest admissible "
                         "lambda.  0 (the default) takes it as is, which is "
                         "what a half-decade grid wants: one step there is a "
                         "factor of ~3 and would discard most of the "
                         "regularization.  Use 1 when the report flags the "
                         "selection as FRAGILE and more seeds are not an option.")
    ap.add_argument("--min-tol", type=float, default=0.02,
                    help="Equivalence margin: the loss in absolute metric "
                         "units a configuration may take before a lambda is "
                         "rejected (0.02 = two accuracy/Dice points).  Acts as "
                         "a floor under the noise-based tolerance.")
    ap.add_argument("--tol-sigma", type=float, default=3.0,
                    help="Standard errors of the paired difference allowed "
                         "before a lambda is rejected.")
    ap.add_argument("--tol-grid", type=float, nargs="+",
                    default=[0.005, 0.01, 0.02, 0.05],
                    help="Equivalence margins for the sensitivity table.")
    ap.add_argument("--show-curve-keys", action="store_true",
                    help="List the training_curves keys found in the logs, to "
                         "check whether the loss components needed for the "
                         "penalty/data ratio are recorded.")
    args = ap.parse_args()

    runs, datasets, extra = load_runs(args.root)
    if not runs:
        raise SystemExit("No logs found. Check the path.")

    if args.show_curve_keys:
        print("\n  training_curves keys per task:")
        for task, keys in extra["curve_keys"].items():
            print(f"    {task:<15} {', '.join(sorted(keys))}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    chosen, payload, sens_by_task = {}, {}, {}

    for task in sorted(runs):
        res = select(runs[task], args.min_tol, args.tol_sigma, args.margin)
        census = seed_census(runs[task])
        sens = tol_sensitivity(runs[task], args.tol_grid, args.tol_sigma,
                               args.margin)
        sens_by_task[task] = sens

        n_lo, n_hi = census["min"], census["max"]
        n_seeds = f"{n_lo}" if n_lo == n_hi else f"{n_lo}--{n_hi}"

        report(task, res, census, sens, extra["ratios"].get(task, {}),
               args.min_tol, args.tol_sigma, args.margin)

        dataset = ", ".join(sorted(datasets[task]))
        latex_table(task, res["agg"], res["grid"], res["deltas"],
                    res["lambda_star"], dataset, n_seeds,
                    args.outdir / f"ablation_lambda_{task}.tex")
        figure(task, res["agg"], res["grid"], res["lambda_star"], dataset,
               n_seeds, args.outdir / f"ablation_lambda_{task}")

        chosen[task] = res["lambda_star"]
        grid = res["grid"]
        slack, worst_key, worst_delta, worst_tol = res["worst"]
        binding = res["binding"]
        payload[task] = {
            "lambda_star": res["lambda_star"],
            "at_grid_edge": bool(res["lambda_star"] in (grid[0], grid[-1])),
            "grid": grid,
            "largest_admissible": res["largest_ok"],
            "lambda_star_margin1": res["backed_off"],
            "n_seeds_min": n_lo,
            "n_seeds_max": n_hi,
            "rule": {"min_tol": args.min_tol, "tol_sigma": args.tol_sigma,
                     "margin": args.margin},
            "tightest_combination": {
                "model": worst_key[0] if worst_key else None,
                "freeze_mode": worst_key[1] if worst_key else None,
                "delta": worst_delta, "tol": worst_tol, "slack": slack,
                "slack_frac_of_tol": res["slack_frac"],
            },
            "binding_constraint": ({
                "lambda": binding[0], "model": binding[1][0],
                "freeze_mode": binding[1][1], "delta": binding[2],
                "tol": binding[3], "reason": binding[4],
            } if binding else None),
            "fragile": bool(res["slack_frac"] is not None
                            and res["slack_frac"] < FRAGILE_FRAC),
            "tol_sensitivity": [{"min_tol": t, "lambda_star": l} for t, l in sens],
        }

    latex_sensitivity(sens_by_task,
                      args.outdir / "ablation_lambda_tol_sensitivity.tex")

    (args.outdir / "selected_lambda.json").write_text(json.dumps(payload, indent=2))
    print(f"  [json] {args.outdir / 'selected_lambda.json'}")

    print(f"\n{'=' * 78}\nSelected coefficients\n{'=' * 78}")
    for task, lam in chosen.items():
        notes = []
        if payload[task]["at_grid_edge"]:
            notes.append("ON A GRID EDGE, ablation inconclusive")
        if payload[task]["fragile"]:
            notes.append(f"FRAGILE, flips to "
                         f"{payload[task]['lambda_star_margin1']:g} under noise")
        suffix = ("   <-- " + "; ".join(notes)) if notes else ""
        print(f"  {task:<15} lambda* = {lam:g}{suffix}")

    if len(set(chosen.values())) == 1:
        lam = next(iter(chosen.values()))
        print(f"\n  Both tasks agree: set LAMBDA_VALUES=(0 {lam:g}) in both sweeps.")
    else:
        print("\n  The two tasks disagree: use each task's value in its own sweep,")
        print("  and say so explicitly in the paper.  lambda weights the penalty")
        print("  against data terms with different gradient scales, so the bare")
        print("  coefficient is not comparable between the two -- quote the")
        print("  penalty/data ratio (or the sigma reduction) instead.")


if __name__ == "__main__":
    main()
