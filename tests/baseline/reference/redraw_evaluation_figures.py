"""Plot-only re-export of the evaluation figures.

The stored Monte Carlo summaries are unchanged.  The layout uses clean,
offset point intervals without pairwise connector lines, together with a
compact tuning display.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts" / "Evaluation_regularized_balancing_estimators"
FIGURE_DIR = ARTIFACT_DIR / "figures"
PART2_DIR = ARTIFACT_DIR / "Exp part 2"
for directory in (FIGURE_DIR, PART2_DIR):
    directory.mkdir(parents=True, exist_ok=True)

FAMILIES = [
    "Uniform", "IPW", "Trimmed IPW", "Entropy balancing", "SBW", "CBPS",
    "Overlap weights", "Bruns-Smith ABW",
]
ORDER = FAMILIES + ["ARB"]
LABELS = {
    "Uniform": "Uniform", "IPW": "IPW", "Trimmed IPW": "Trimmed IPW",
    "Entropy balancing": "Entropy balancing", "SBW": "SBW", "CBPS": "CBPS",
    "Overlap weights": "Overlap weights", "Bruns-Smith ABW": "Bruns-Smith ABW",
    "ARB": "ARB",
}
SHORT_LABELS = {
    **LABELS,
    "Trimmed IPW": "Trimmed", "Entropy balancing": "Entropy",
    "Overlap weights": "Overlap", "Bruns-Smith ABW": "BS-ABW",
}
TICK_LABELS = {
    "Uniform": "Uniform", "IPW": "IPW", "Trimmed IPW": "Trimmed\nIPW",
    "Entropy balancing": "Entropy\nbalancing", "SBW": "SBW", "CBPS": "CBPS",
    "Overlap weights": "Overlap\nweights", "Bruns-Smith ABW": "Bruns-Smith\nABW",
}
OVERLAPS = ["Strong overlap", "Intermediate weak overlap", "Severe weak overlap"]
STAGES = ["Base", "Augmented (OOD)", "Full ARB"]
COLORS = {"Base": "#6B7280", "Augmented (OOD)": "#0072B2", "Full ARB": "#D55E00"}
MARKERS = {"Base": "s", "Augmented (OOD)": "o", "Full ARB": "^"}
Y_OFFSETS = {"Base": 0.13, "Augmented (OOD)": -0.13, "Full ARB": 0.0}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14.0,
    "axes.labelsize": 15.5, "axes.titlesize": 16.0,
    "xtick.labelsize": 13.0, "ytick.labelsize": 13.0,
    "legend.fontsize": 13.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.17, "grid.linewidth": 0.75,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _save(fig, canonical: str, alias: str) -> None:
    for directory in (FIGURE_DIR, PART2_DIR):
        fig.savefig(directory / alias, bbox_inches="tight", pad_inches=0.22, facecolor="white")
    fig.savefig(FIGURE_DIR / canonical, bbox_inches="tight", pad_inches=0.22, facecolor="white")
    fig.savefig(
        FIGURE_DIR / canonical.replace(".pdf", ".png"), dpi=240,
        bbox_inches="tight", pad_inches=0.22, facecolor="white",
    )
    plt.close(fig)


def _legend_handles():
    return [
        Line2D([0], [0], color=COLORS[s], marker=MARKERS[s], linestyle="none",
               markersize=7.2, label=label)
        for s, label in zip(
            STAGES,
            ["Base estimator", "Our method (target-aware tuning)", "Full ARB"],
        )
    ]


def render_estimator_figures(summary: pd.DataFrame | None = None) -> None:
    if summary is None:
        summary = pd.read_csv(ARTIFACT_DIR / "arb_estimator_summary.csv")
    specs = [
        (0.5, "01_ARB_estimator_comparison_phi0_0p5.pdf", "Evaluation0.5.pdf"),
        (1.25, "01_ARB_estimator_comparison_phi0_1p25.pdf", "Evaluation01.25.pdf"),
    ]
    y = np.arange(len(ORDER))[::-1].astype(float)
    for phi0, canonical, alias in specs:
        fig, axes = plt.subplots(1, 3, figsize=(20.0, 8.4), sharex=True, sharey=True)
        for col, (ax, overlap) in enumerate(zip(axes, OVERLAPS)):
            panel = summary[
                summary["phi0"].eq(phi0) & summary["overlap"].eq(overlap)
            ]
            for stage in STAGES:
                values = panel[panel["stage"].eq(stage)].set_index("family").reindex(ORDER)
                valid = values["empirical_predictive_rmse"].notna().to_numpy()
                ax.errorbar(
                    values.loc[valid, "empirical_predictive_rmse"],
                    y[valid] + Y_OFFSETS[stage],
                    xerr=2 * values.loc[valid, "rmse_mcse"],
                    color=COLORS[stage], marker=MARKERS[stage], linestyle="none",
                    markersize=8.2, capsize=3.2, elinewidth=1.35,
                    markeredgecolor="white", markeredgewidth=0.75, zorder=3,
                )
            ax.set_title(overlap, pad=10)
            ax.set_xlabel("Counterfactual prediction RMSE")
            ax.set_yticks(y)
            ax.set_yticklabels([LABELS[x] for x in ORDER])
            ax.tick_params(axis="y", labelleft=(col == 0), pad=6)
            ax.grid(axis="x", alpha=.19); ax.grid(axis="y", alpha=.07)
            ax.margins(x=.11, y=.09)
        fig.legend(handles=_legend_handles(), loc="upper center", bbox_to_anchor=(.5, .995),
                   ncol=3, frameon=False, columnspacing=2.2)
        fig.subplots_adjust(left=.135, right=.985, bottom=.12, top=.87, wspace=.13)
        _save(fig, canonical, alias)


METRIC_SPECS = [
    ("signal_risk", "signal_risk_mcse", r"Bias  $B_n(\widehat{\lambda})$"),
    ("noise_risk", "noise_risk_mcse", r"Variance  $V_n(\widehat{\lambda})$"),
    ("exact_risk", "exact_risk_mcse", r"Total risk  $R_n(\widehat{\lambda})$"),
    ("weight_norm_squared", "weight_norm_squared_mcse", r"Weight norm  $\|\widehat{\gamma}\|_2^2$"),
    ("heldout_imbalance_squared", "heldout_imbalance_squared_mcse", r"Held-out imbalance  $\|\Delta_E\|_2^2$"),
]


def render_metric_figures(summary: pd.DataFrame | None = None) -> None:
    if summary is None:
        summary = pd.read_csv(ARTIFACT_DIR / "primary_five_metric_summary.csv")
    specs = [
        (0.5, "02_required_metrics_phi0_0p5.pdf", "Metrics0.5.pdf"),
        (1.25, "02_required_metrics_phi0_1p25.pdf", "Metrics1.25.pdf"),
    ]
    y = np.arange(len(ORDER))[::-1].astype(float)
    for phi0, canonical, alias in specs:
        fig, axes = plt.subplots(3, 5, figsize=(23.2, 13.8), sharey=True, squeeze=False)
        for row, overlap in enumerate(OVERLAPS):
            panel = summary[
                summary["phi0"].eq(phi0) & summary["overlap"].eq(overlap)
            ]
            for col, (metric, mcse, title) in enumerate(METRIC_SPECS):
                ax = axes[row, col]
                if row == 0:
                    ax.set_title(title, pad=10)
                for stage in STAGES:
                    values = panel[panel["stage"].eq(stage)].set_index("family").reindex(ORDER)
                    valid = values[metric].notna().to_numpy()
                    ax.errorbar(
                        values.loc[valid, metric], y[valid] + Y_OFFSETS[stage],
                        xerr=2 * values.loc[valid, mcse], color=COLORS[stage],
                        marker=MARKERS[stage], linestyle="none", markersize=7.0,
                        capsize=2.8, elinewidth=1.2, markeredgecolor="white",
                        markeredgewidth=.65, zorder=3,
                    )
                ax.set_yticks(y)
                ax.set_yticklabels([SHORT_LABELS[x] for x in ORDER])
                ax.tick_params(axis="y", labelleft=(col == 0), pad=5)
                ax.grid(axis="x", alpha=.19); ax.grid(axis="y", alpha=.06)
                ax.margins(x=.15, y=.09)
                if col == 0:
                    ax.set_ylabel(overlap, labelpad=16)
        handles = _legend_handles()
        handles[-1].set_label("ARB residual-weight diagnostic")
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .995),
                   ncol=3, frameon=False, columnspacing=2.2)
        fig.subplots_adjust(left=.105, right=.99, bottom=.075, top=.89, hspace=.30, wspace=.25)
        _save(fig, canonical, alias)


RULES = ["Spectral target-aware", "Two-moment diagnostic", "Source GCV", "Infeasible oracle"]
RULE_COLORS = {
    "Spectral target-aware": "#0072B2", "Two-moment diagnostic": "#CC79A7",
    "Source GCV": "#D55E00", "Infeasible oracle": "#6B7280",
}
RULE_MARKERS = {
    "Spectral target-aware": "o", "Two-moment diagnostic": "s",
    "Source GCV": "D", "Infeasible oracle": "^",
}
RULE_OFFSETS = {
    "Spectral target-aware": -.27, "Two-moment diagnostic": -.09,
    "Source GCV": .09, "Infeasible oracle": .27,
}


def render_tuning_figures(
    penalty: pd.DataFrame | None = None,
    excess: pd.DataFrame | None = None,
) -> None:
    if penalty is None:
        penalty = pd.read_csv(ARTIFACT_DIR / "tuning_selection_summary.csv")
    if excess is None:
        excess = pd.read_csv(ARTIFACT_DIR / "tuning_excess_exact_risk.csv")
    specs = [
        (0.5, "03_tuning_rules_phi0_0p5.pdf", "Tuning0.5.pdf"),
        (1.25, "03_tuning_rules_phi0_1p25.pdf", "Tuning1.25.pdf"),
    ]
    x = np.arange(len(FAMILIES), dtype=float)
    for phi0, canonical, alias in specs:
        fig, axes = plt.subplots(2, 3, figsize=(20.8, 10.0), sharex="col", squeeze=False)
        for col, overlap in enumerate(OVERLAPS):
            penalty_ax, risk_ax = axes[0, col], axes[1, col]
            penalty_ax.set_title(overlap, pad=10)
            pp = penalty[penalty["phi0"].eq(phi0) & penalty["overlap"].eq(overlap)]
            for rule in RULES:
                values = pp[pp["rule"].eq(rule)].set_index("family").reindex(FAMILIES)
                center = values["median"].to_numpy(float)
                penalty_ax.errorbar(
                    x + RULE_OFFSETS[rule], center,
                    yerr=np.vstack((center - values["lower"].to_numpy(float),
                                    values["upper"].to_numpy(float) - center)),
                    color=RULE_COLORS[rule], marker=RULE_MARKERS[rule], linestyle="none",
                    markersize=7.8, capsize=3.0, elinewidth=1.25,
                    markeredgecolor="white", markeredgewidth=.65,
                )
            penalty_ax.set_yscale("log")
            penalty_ax.grid(axis="y", which="both", alpha=.20)
            penalty_ax.grid(axis="x", alpha=.09)
            penalty_ax.margins(x=.055, y=.14)
            if col == 0:
                penalty_ax.set_ylabel(r"Selected $\lambda$")

            ep = excess[excess["phi0"].eq(phi0) & excess["overlap"].eq(overlap)]
            for rule in RULES[:-1]:
                values = ep[ep["rule"].eq(rule)].set_index("family").reindex(FAMILIES)
                risk_ax.errorbar(
                    x + RULE_OFFSETS[rule], values["excess_exact_risk"],
                    yerr=2 * values["excess_exact_risk_mcse"],
                    color=RULE_COLORS[rule], marker=RULE_MARKERS[rule], linestyle="none",
                    markersize=7.8, capsize=3.0, elinewidth=1.25,
                    markeredgecolor="white", markeredgewidth=.65,
                )
            risk_ax.axhline(0, color=".45", linewidth=1.0, linestyle="--")
            risk_ax.grid(axis="y", alpha=.20); risk_ax.grid(axis="x", alpha=.09)
            risk_ax.margins(x=.055, y=.14)
            if col == 0:
                risk_ax.set_ylabel("Excess exact risk\nrelative to oracle")
            risk_ax.set_xticks(x)
            risk_ax.set_xticklabels(
                [SHORT_LABELS[f] for f in FAMILIES], fontsize=12.2,
                rotation=18, ha="right", rotation_mode="anchor",
            )
        handles = [
            Line2D([0], [0], color=RULE_COLORS[r], marker=RULE_MARKERS[r], linestyle="none",
                   markersize=7.0, label=r)
            for r in RULES
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .995),
                   ncol=4, frameon=False, columnspacing=1.9)
        fig.subplots_adjust(left=.08, right=.99, bottom=.145, top=.87, hspace=.29, wspace=.21)
        _save(fig, canonical, alias)


def render_all() -> None:
    render_estimator_figures()
    render_metric_figures()
    render_tuning_figures()


if __name__ == "__main__":
    render_all()
    print("Re-exported all evaluation figures from the stored Monte Carlo summaries.")
