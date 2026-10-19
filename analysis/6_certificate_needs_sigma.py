"""
analysis/6_certificate_needs_sigma.py
=====================================
Margin certificate of Section 4.3 against what the models actually do.

    python -m analysis.6_certificate_needs_sigma --task classification
    python -m analysis.6_certificate_needs_sigma --task segmentation --perturbation Contrast

For every drift run (one checkpoint, one evaluation dataset, one perturbation):

    sigma = ||W||_2                    spectral norm of the head
    Gamma = max_kl ||w_k - w_l||_2     margin sensitivity of the head
    at-risk rate                       fraction of units (images, or pixels in
                                       segmentation) with rho(z) <= Gamma ||dz||_2
    certified drop                     accuracy: fraction of correct and at-risk
                                       images; Dice: exact worst-case bound
    true drop                          what actually happened

The feature drift ||dz||_2 is the Euclidean norm at the head input, so the
certificate is an L2 statement, unlike the angular metrics of scripts 1-4.
For segmentation the loose bound 3n/(S-n) is reported next to the exact one,
and the exact one also as a fraction of the clean Dice. Only runs evaluated on
their training dataset are used, unless --external is given.

Outputs (in --out): <stem>.tex/.csv (main table), <stem>_gamma.tex,
<stem>_runs.csv (one row per run), <stem>.pdf/.png, and a VERIFICATION block
with the violation counts, which must be zero.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import common as ac
from analysis import io as aio

STEM = "certificate_needs_sigma"

# One repetition is one (dataset, seed).
SPREAD_UNIT = ["eval_dataset", "seed"]

MILD = "gaussian"       # the perturbation the usability claim is asserted on


# =============================================================================
# Data
# =============================================================================
def prepare(root: str, cache: str | None, refresh: bool, task: str,
            perturbation: str | None, datasets: list[str] | None,
            settings: list[str] | None, external: bool) -> pd.DataFrame:
    """Drift runs, with the derived certificate quantities."""
    frames = aio.load_all(root, cache_dir=cache, refresh=refresh,
                          points_per_run=0, progress=True)
    df = frames["drift"]
    df = df[df.task == task].copy()
    if df.empty:
        sys.exit(f"no {task} drift runs found under the results root")

    if perturbation:
        df = df[df.perturbation == perturbation]
    if datasets:
        df = df[df.eval_dataset.isin(datasets)]
    if settings:
        df = df[df.setting.isin(settings)]
    # In distribution only, as in the rest of the paper, unless asked.
    if not external:
        df = df[df.train_dataset == df.eval_dataset]
    if df.empty:
        sys.exit("no rows left after the requested filters")

    df["gamma_frac"] = df.gamma / (np.sqrt(2) * df.sigma_max)
    # Certified drop shown in the paper: the exact Dice bound in segmentation
    # (falls back to the loose one for runs written before it existed).
    df["cert_drop_loose"] = df.predicted_drop
    exact = df.get("predicted_drop_exact")
    df["cert_drop"] = (exact.fillna(df.predicted_drop)
                       if task == "segmentation" and exact is not None
                       else df.predicted_drop)
    df["excess"] = df.cert_drop - df.true_drop
    # The drop can never exceed the clean metric: share of it that is certified.
    df["cert_rel"] = df.cert_drop / df.metric_clean
    # The classification stage stores sigma*||dz|| as its drift, the
    # segmentation stage ||dz||: bring both back to ||dz||_2.
    df["dz"] = (df.mean_dz_norm / df.sigma_max if task == "classification"
                else df.mean_dz_norm)
    # Tightness of the logit bound, kept in the runs CSV only.
    df["lip_tightness"] = df.mean_dlogit_norm / (df.sigma_max * df.dz)
    # The two sides of the safe condition rho(z) > Gamma ||dz||_2.
    df["pert_size"] = df.gamma * df.dz
    df["safety"] = df.margin_q50 / df.pert_size
    df["unit"] = "image" if task == "classification" else "pixel"
    return df


def summary(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean and seed-to-seed spread of every certificate quantity."""
    values = [v for v in ("sigma_max", "gamma", "gamma_frac", "flip_rate",
                          "certified_flip_rate", "true_drop", "cert_drop", "cert_rel",
                          "cert_drop_loose", "excess", "safety")
              if v in df and df[v].notna().any()]
    unit = (df.groupby(by + SPREAD_UNIT, observed=True)[values]
              .mean().reset_index())
    out = None
    for value in values:
        agg = ac.agg_seeds(unit, value, by).rename(
            columns={"mean": value, "sd": value + "_sd"})
        cols = by + [value, value + "_sd"] + (["n"] if out is None else [])
        out = agg[cols] if out is None else out.merge(agg[cols], on=by)
    return out.sort_values(by).reset_index(drop=True)


def verify(df: pd.DataFrame, task: str) -> None:
    """Violation counts of every bound, which must all be zero."""
    ck = ac.Checks(f"certificate, {task}")
    bounds = [("logit/margin bound", "bound_violations")]
    if task == "segmentation":
        bounds = [("loose Dice bound 3n/(S-n)", "bound_violations"),
                  ("exact Dice bound", "bound_violations_exact")]
    for name, col in bounds:
        v = pd.to_numeric(df.get(col), errors="coerce") if col in df else None
        if v is None or v.isna().all():
            ck.note(name, "not recorded in these runs (rerun the drift stage)")
            continue
        n_bad = int(v.fillna(0).sum())
        ck.add(f"{name} is never violated", n_bad == 0,
               f"{n_bad} violating images over {int(v.notna().sum())} runs")
    for (arch, setting), g in df.groupby(["arch", "setting"], observed=True):
        ck.note(f"{aio.ARCH_LABEL[str(arch)]} {setting}",
                f"sigma {g.sigma_max.mean():.3f}, Gamma {g.gamma.mean():.3f}, "
                f"at-risk {g.certified_flip_rate.mean():.3f}, "
                f"certified {g.cert_drop.mean():.3f}, true {g.true_drop.mean():.4f}, "
                f"median safety ratio {g.safety.median():.2f}")
    ck.report()


# =============================================================================
# Tables
# =============================================================================
def table(df: pd.DataFrame, task: str, path: str) -> None:
    """One row per architecture and setting: the two sides of the guarantee."""
    s = summary(df, ["arch", "setting"])
    unit = df.unit.iloc[0]
    metric = {"classification": "accuracy", "segmentation": "Dice"}[task]
    rows = []
    for _, r in s.iterrows():
        row = {
            "Backbone": aio.ARCH_LABEL[str(r.arch)],
            "Setting": str(r.setting),
            r"$\sigma$": f"{r.sigma_max:.3f}",
            r"$\Gamma$": f"{r.gamma:.3f}",
            f"at-risk {unit}s": f"{r.certified_flip_rate:.3f}",
            f"flipped {unit}s": f"{r.flip_rate:.3f}",
            "certified drop": f"{r.cert_drop:.3f}",
        }
        if task == "segmentation":
            row["certified / clean"] = f"{r.cert_rel:.2f}"
            row["loose bound"] = f"{r.cert_drop_loose:.3f}"
        row["true drop"] = f"{r.true_drop:.3f}"
        rows.append(row)
    out = pd.DataFrame(rows)
    order = {x: i for i, x in enumerate(aio.SETTING_ORDER)}
    out = out.sort_values(["Backbone", "Setting"],
                          key=lambda c: c.map(order) if c.name == "Setting" else c
                          ).reset_index(drop=True)

    perts = ", ".join(sorted(df.perturbation.dropna().unique()))
    ac.write_table(
        out, path,
        caption=(f"Margin certificate and observed degradation, {task}, "
                 f"{perts}. $\\sigma=\\|\\mathbf{{W}}\\|_2$ and $\\Gamma$ are the "
                 f"spectral norm and the margin sensitivity of the head. "
                 f"\\emph{{at-risk}} is the fraction of {unit}s with "
                 f"$\\rho(z) \\leq \\Gamma\\|\\Delta z\\|_2$, where "
                 f"$\\|\\Delta z\\|_2$ is the Euclidean norm of the feature "
                 f"drift at the head input. \\emph{{certified drop}} is the "
                 f"bound on the {metric} drop"
                 + (" (exact worst case), \\emph{certified / clean} the same "
                    "bound as a fraction of the clean Dice, \\emph{loose bound} "
                    "the earlier $3n/(S-n)$" if task == "segmentation" else "")
                 + f", \\emph{{true drop}} the observed one. Mean over "
                 f"{df.eval_dataset.nunique()} dataset(s) and "
                 f"{df.seed.nunique()} seeds."),
        label=f"tab:{STEM}_{task}",
        column_format="ll" + "r" * (len(out.columns) - 2))


def gamma_table(df: pd.DataFrame, task: str, path: str) -> None:
    """How much the exact Gamma buys over sqrt(2) sigma, per architecture."""
    g = (df.groupby("arch", observed=True)
           .agg(sigma=("sigma_max", "mean"), gamma=("gamma", "mean"),
                frac=("gamma_frac", "mean"), lo=("gamma_frac", "min"),
                hi=("gamma_frac", "max"), n=("gamma_frac", "size"))
           .reset_index())
    if g.empty:
        return
    out = pd.DataFrame({
        "Backbone": [aio.ARCH_LABEL[str(a)] for a in g.arch],
        r"$\sigma$": [f"{v:.3f}" for v in g.sigma],
        r"$\sqrt{2}\,\sigma$": [f"{v:.3f}" for v in np.sqrt(2) * g.sigma],
        r"$\Gamma$": [f"{v:.3f}" for v in g.gamma],
        r"$\Gamma / (\sqrt{2}\sigma)$": [f"{v:.3f}" for v in g.frac],
        "range": [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(g.lo, g.hi)],
        "$n$": [f"{int(v)}" for v in g.n],
    })
    ac.write_table(
        out, path,
        caption=(f"Margin sensitivity $\\Gamma$ against its bound "
                 f"$\\sqrt{{2}}\\sigma$, {task}."),
        label=f"tab:{STEM}_{task}_gamma",
        column_format="lrrrrrr",
        note=(r"For a two-class head the two rows of $\mathbf{W}$ are "
              r"near-antisymmetric, so $\Gamma \approx \sqrt{2}\,\sigma$."))


# =============================================================================
# Figure
# =============================================================================
def figure(df: pd.DataFrame, task: str, path: str) -> None:
    """Left: certified and true drop. Middle: certified drop against sigma.
    Right: at-risk and flipped units. Hatch/marker = backbone, colour = setting."""
    plt = ac.setup_style()
    archs = [a for a in aio.ARCH_ORDER if (df.arch == a).any()]
    settings = [s for s in aio.SETTING_ORDER if (df.setting == s).any()]
    metric = {"classification": "accuracy", "segmentation": "Dice"}[task]
    unit = df.unit.iloc[0]
    s_all = summary(df, ["arch", "setting"])
    hatch = {"resnet": "", "vit": "//", "dinov2": ".."}

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))
    width = 0.8 / max(len(archs), 1)
    pos = np.arange(len(settings))

    def bars(ax, bound_col, observed_col, observed_color):
        """Grey = what the certificate promises, colour = what happened."""
        for k, arch in enumerate(archs):
            xs = pos + (k - (len(archs) - 1) / 2) * width
            sa = s_all[s_all.arch == arch].set_index("setting")
            cert = [float(sa[bound_col].get(x, np.nan)) for x in settings]
            true = [float(sa[observed_col].get(x, np.nan)) for x in settings]
            ax.bar(xs, cert, width=width * 0.9, color="#c9ccd1",
                   edgecolor="0.45", linewidth=0.5, hatch=hatch[str(arch)])
            ax.bar(xs, true, width=width * 0.55, color=observed_color,
                   edgecolor="0.25", linewidth=0.4, hatch=hatch[str(arch)])
        ax.set_xticks(pos)
        ax.set_xticklabels(settings, fontsize=8)

    # -- the guarantee against reality -------------------------------------
    ax = axes[0]
    bars(ax, "cert_drop", "true_drop", ac.SETTING_COLOR["H+L+SN"])
    ax.axhline(1.0, color="#c8442a", linewidth=0.8, linestyle="--")
    ax.set_ylabel(f"{metric} drop")
    ax.set_ylim(0, 1.15)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor="#c9ccd1",
                                     edgecolor="0.45", label="certified"),
                       plt.Rectangle((0, 0), 1, 1,
                                     facecolor=ac.SETTING_COLOR["H+L+SN"],
                                     edgecolor="0.25", label="observed")],
              frameon=False, fontsize=7.5, loc="center right")

    # -- and what moves it --------------------------------------------------
    ax = axes[1]
    for setting in settings:
        for arch in archs:
            sub = df[(df.setting == setting) & (df.arch == arch)]
            if sub.empty:
                continue
            ax.scatter(sub.sigma_max, sub.cert_drop, s=14, alpha=0.6,
                       color=ac.SETTING_COLOR[str(setting)],
                       marker=ac.ARCH_MARKER[str(arch)], edgecolor="none")
    ax.set_xscale("log")
    ac.log_ticks(ax.xaxis)
    ax.set_xlabel(r"head spectral norm $\sigma$")
    ax.set_ylabel("certified drop")
    handles = [plt.Line2D([], [], ls="", marker="s", color=ac.SETTING_COLOR[str(x)],
                          label=str(x)) for x in settings]
    handles += [plt.Line2D([], [], ls="", marker=ac.ARCH_MARKER[str(a)],
                           color="0.35", label=aio.ARCH_LABEL[str(a)])
                for a in archs]
    ax.legend(handles=handles, frameon=False, fontsize=7, ncol=2,
              loc="best", handlelength=1.0, columnspacing=0.8)

    # -- the at-risk rate ---------------------------------------------------
    ax = axes[2]
    bars(ax, "certified_flip_rate", "flip_rate", ac.SETTING_COLOR["H+L"])
    ax.set_ylabel(f"fraction of {unit}s")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor="white",
                                     edgecolor="0.45", hatch=hatch[str(a)],
                                     label=aio.ARCH_LABEL[str(a)]) for a in archs],
              frameon=False, fontsize=7.5, loc="best")

    fig.tight_layout()
    ac.save_fig(fig, path)


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    p = ac.base_parser("Margin certificate against the observed degradation.")
    p.add_argument("--task", default="classification",
                   choices=["classification", "segmentation"])
    p.add_argument("--perturbation", default=MILD,
                   help="drift perturbation: 'gaussian' (eps = 0.01, the mild "
                        "one the usability claim is asserted on), 'Contrast' "
                        "(severity 3), or 'all'. Default gaussian.")
    p.add_argument("--dataset", nargs="+", default=None,
                   help="evaluation dataset(s). Default: all of them.")
    p.add_argument("--settings", nargs="+", default=None, choices=aio.SETTING_ORDER)
    p.add_argument("--external", action="store_true",
                   help="also keep the runs evaluated on a dataset other than "
                        "the training one (segmentation evaluates across the "
                        "polyp datasets). Default: in distribution only.")
    args = p.parse_args()
    ac.apply_figure_flags(args)

    pert = None if str(args.perturbation).lower() == "all" else args.perturbation
    df = prepare(args.results_root, args.cache, args.refresh, args.task,
                 pert, args.dataset, args.settings, args.external)

    stem = f"{STEM}_{args.task}_{args.perturbation}"
    if args.external:
        stem += "_external"
    os.makedirs(args.out, exist_ok=True)
    print()
    runs_path = os.path.join(args.out, f"{stem}_runs.csv")
    df.drop(columns=[c for c in ("path",) if c in df]).to_csv(runs_path, index=False)
    print(f"  [csv] {runs_path}")
    verify(df, args.task)
    table(df, args.task, os.path.join(args.out, f"{stem}.tex"))
    gamma_table(df, args.task, os.path.join(args.out, f"{stem}_gamma.tex"))
    figure(df, args.task, os.path.join(args.out, f"{stem}.pdf"))


if __name__ == "__main__":
    main()
