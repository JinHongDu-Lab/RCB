"""Paper-aligned experiments requested in the red numerical-organization block.

The module deliberately separates two theoretical settings from the paper:

1. Theorem 3 uses normalized design-independent base weights.  These weights
   are constructed so that the two coupled-extrapolation scaling identities
   hold exactly at every finite sample size.
2. Theorem 7 and Corollary D.6 use an honest pilot/evaluation target split and
   the ridge base weights in Appendix D.2.  The evaluation fold is not used to
   construct the base weights.

All reported Monte Carlo draws are retained.  No draw is removed because of
an inconvenient estimate, boundary solution, selected penalty, or coverage
result.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import seaborn as sns

try:
    from .spectral_quasi_score import (
        DEFAULT_THETA_BOUNDS,
        equation_41_estimates_many,
        spectral_quasi_score_from_feature_eigen,
    )
except ImportError:  # pragma: no cover - supports direct notebook execution
    from spectral_quasi_score import (
        DEFAULT_THETA_BOUNDS,
        equation_41_estimates_many,
        spectral_quasi_score_from_feature_eigen,
    )


ARTIFACT_DIR = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "Paper_aligned_red_requirements_rechecked"
)
FIGURE_DIR = ARTIFACT_DIR / "figures"

COLORS = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "gold": "#E69F00",
    "gray": "#6B7280",
    "black": "#111827",
}

COMPONENT_COLORS = {
    "Signal": COLORS["blue"],
    "Residual noise": COLORS["vermillion"],
    "Total": COLORS["green"],
}

# Keep the saved-data terminology stable, but use shorter paper-facing labels
# in the uniform-error figure.  In the manuscript caption, "Bias" is defined
# as the integrated squared conditional bias B_n and "Variance" as the
# conditional residual-noise variance V_n.
COMPONENT_DISPLAY_LABELS = {
    "Signal": "Bias",
    "Residual noise": "Variance",
    "Total": "Total risk",
}

SCENARIOS = {
    "Population shift only": (1.0, 0.0),
    "Weight concentration only": (0.0, 1.0),
    "Coupled": (1.0, 1.0),
}


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 20260802
    r2: float = 1.0
    sigma2: float = 0.5
    # The red deterministic-equivalent block requests exactly these exponents.
    etas: tuple[float, ...] = (0.0, 0.5, 1.0)
    # The paper's predictive-calibration setup on pages 22--23 also includes 0.75.
    calibration_etas: tuple[float, ...] = (0.0, 0.5, 0.75, 1.0)
    lambda_min: float = 0.02
    lambda_max: float = 30.0
    lambda_count: int = 25
    phi0: float = 0.75
    phi1: float = 0.75
    # Exact dimension grid in the paper's displayed calibration setup.
    dimensions: tuple[int, ...] = (80, 160, 320, 640)
    deterministic_designs: int = 24
    # Ex 2 uses a larger dimension to control the finite-sample empirical-mean
    # terms at the extreme aspect ratios (in particular, phi1=10).
    aspect_dimension: int = 1280
    aspect_designs: int = 24
    # Keep the structured-covariance supplement at the original dimension; it
    # is logically separate from the Ex 2 finite-sample refinement.
    orientation_dimension: int = 640
    inference_designs: int = 36
    outcomes_per_design: int = 100
    base_ridge_penalty: float = 1.0
    nuisance_designs: int = 24
    nuisance_outcomes: int = 50
    split_designs: int = 18
    split_outcomes: int = 40
    calibration_shift_rho2: float = 0.5
    sparse_shift_fraction: float = 0.1
    pilot_fraction: float = 0.5
    outcome_shift_constant: float = 100.0

    @property
    def lambdas(self) -> np.ndarray:
        return np.geomspace(self.lambda_min, self.lambda_max, self.lambda_count)


DEFAULT_CONFIG = ExperimentConfig()


def configure_style() -> None:
    """Use one font family and one size hierarchy for every figure."""
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12.5,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.4,
            "lines.markersize": 6.5,
            "savefig.dpi": 240,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.22,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def prepare_output(config: ExperimentConfig = DEFAULT_CONFIG) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    config_row = asdict(config)
    config_row["etas"] = ";".join(f"{value:g}" for value in config.etas)
    config_row["calibration_etas"] = ";".join(
        f"{value:g}" for value in config.calibration_etas
    )
    config_row["dimensions"] = ";".join(str(value) for value in config.dimensions)
    pd.DataFrame([config_row]).to_csv(ARTIFACT_DIR / "configuration.csv", index=False)
    package_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": mpl.__version__,
        "seaborn": sns.__version__,
    }
    (ARTIFACT_DIR / "package_versions.json").write_text(
        json.dumps(package_versions, indent=2), encoding="utf-8"
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Apply the paper-wide typography and outer-padding rules."""
    if fig._suptitle is not None:
        fig._suptitle.remove()
        fig._suptitle = None
    for ax in fig.axes:
        ax.title.set_size(14)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        ax.tick_params(axis="both", which="both", labelsize=12)
        if ax.legend_ is not None:
            for text_item in ax.legend_.get_texts():
                text_item.set_fontsize(12.5)
    for legend in fig.legends:
        for text_item in legend.get_texts():
            text_item.set_fontsize(12.5)
    fig.savefig(FIGURE_DIR / f"{stem}.png")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf")
    plt.close(fig)


def _centered_unit_vector(length: int) -> np.ndarray:
    values = np.arange(length, dtype=float) - (length - 1.0) / 2.0
    values /= np.linalg.norm(values)
    if abs(values.sum()) > 1.0e-12:
        raise RuntimeError("The deterministic weight direction is not centered.")
    return values


def external_base_weights(n0: int, eta: float, varrho2: float) -> np.ndarray:
    """Normalized external weights with n0**eta ||C0 gamma||^2 = varrho2."""
    gamma = np.full(n0, 1.0 / n0)
    if varrho2 > 0:
        gamma = gamma + np.sqrt(varrho2 / n0**eta) * _centered_unit_vector(n0)
    return gamma


def mean_shift(
    p: int,
    n0: int,
    eta: float,
    rho2: float,
    *,
    orientation: str = "flat",
    sparse_fraction: float = 0.1,
) -> np.ndarray:
    """Mean shift with n0**eta ||delta_nu||^2 / p = rho2 exactly."""
    if rho2 <= 0:
        return np.zeros(p)
    if orientation == "low":
        direction = np.eye(1, p, 0).ravel()
    elif orientation == "high":
        direction = np.eye(1, p, p - 1).ravel()
    elif orientation == "flat":
        direction = np.full(p, 1.0 / np.sqrt(p))
    elif orientation == "sparse":
        k = max(1, int(round(sparse_fraction * p)))
        direction = np.zeros(p)
        direction[:k] = 1.0 / np.sqrt(k)
    else:
        raise ValueError(f"Unknown shift orientation: {orientation}")
    return np.sqrt(rho2 * p / n0**eta) * direction


def design_eigendecomposition(X0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xbar0 = X0.mean(axis=0)
    X0c = X0 - xbar0
    S0c = X0c.T @ X0c / len(X0)
    eigenvalues, eigenvectors = np.linalg.eigh(S0c)
    return xbar0, X0c, eigenvalues, eigenvectors


def ridge_augmented_path(
    X0: np.ndarray,
    X0c: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    xbar_target: np.ndarray,
    gamma_base: np.ndarray,
    lambdas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta = xbar_target - X0.T @ gamma_base
    projected = eigenvectors.T @ delta
    coefficients = projected[:, None] / (eigenvalues[:, None] + lambdas[None, :])
    directions = eigenvectors @ coefficients
    gamma_path = gamma_base[:, None] + X0c @ directions / len(X0)
    normalization_error = float(np.max(np.abs(gamma_path.sum(axis=0) - 1.0)))
    if normalization_error > 2.0e-10:
        raise RuntimeError(f"Affine normalization failed: {normalization_error}")
    return gamma_path, delta


def exact_risk_components(
    X0: np.ndarray,
    gamma_path: np.ndarray,
    nu1: np.ndarray,
    r2: float,
    sigma2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    imbalance = X0.T @ gamma_path - nu1[:, None]
    signal = (r2 / X0.shape[1]) * np.sum(imbalance**2, axis=0)
    residual = sigma2 * np.sum(gamma_path**2, axis=0)
    return signal, residual, signal + residual


def deterministic_equivalent_diagonal(
    lambdas: np.ndarray,
    *,
    phi0: float,
    phi1: float,
    eta: float,
    rho2: float,
    varrho2: float,
    r2: float,
    sigma2: float,
    sigma0_eigenvalues: np.ndarray | None = None,
    sigma1_diagonal: np.ndarray | None = None,
    shift_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Theorem 3 specialized to simultaneously diagonal Sigma0 and Sigma1."""
    if sigma0_eigenvalues is None:
        sigma0_eigenvalues = np.ones(1)
    s = np.asarray(sigma0_eigenvalues, dtype=float)
    if sigma1_diagonal is None:
        sigma1_diagonal = np.ones_like(s)
    t = np.asarray(sigma1_diagonal, dtype=float)
    if shift_weights is None:
        shift_weights = np.full(len(s), 1.0 / len(s))
    g = np.asarray(shift_weights, dtype=float)
    g = g / g.sum()

    signal_rows: list[float] = []
    residual_rows: list[float] = []
    for lam in np.asarray(lambdas, dtype=float):
        lower, upper = 1.0e-12, 1.0 / lam
        for _ in range(100):
            v = 0.5 * (lower + upper)
            equation = 1.0 / v - lam - phi0 * np.mean(s / (1.0 + v * s))
            if equation > 0:
                lower = v
            else:
                upper = v
        v = 0.5 * (lower + upper)
        vev = 1.0 / (v**-2 - phi0 * np.mean(s**2 / (1.0 + v * s) ** 2))

        def veb(a: np.ndarray) -> float:
            return float(
                phi0 * vev * np.mean(a * s / (1.0 + v * s) ** 2)
            )

        def k1(a: np.ndarray) -> float:
            return float(np.mean(a / (1.0 + v * s)))

        def k2(a: np.ndarray) -> float:
            vb = veb(a)
            return float(np.mean((vb * s + a) / (1.0 + v * s) ** 2))

        vb_identity = veb(np.ones_like(s))
        shift_signal = float(np.sum(g * (vb_identity * s + 1.0) / (1.0 + v * s) ** 2))
        shift_residual = float(
            np.sum(g * (v - vb_identity) * s / (1.0 + v * s) ** 2)
        )

        b = (
            lam**2 * r2 * varrho2 / phi0 * (v - lam * vev)
            + r2 * rho2 * shift_signal
        )
        w = (
            sigma2 * varrho2 * lam**2 * vev
            + sigma2 * phi0 * rho2 / lam * shift_residual
        )
        if eta == 1.0:
            target_kernel = float(np.mean(t)) - 2.0 * k1(t) + k2(t)
            b += r2 * phi1 / phi0 * target_kernel + r2 * k2(s)
            w += sigma2 * phi1 / lam * (k1(t) - k2(t))
            w += sigma2 * (1.0 + phi0 / lam * (k1(s) - k2(s)))
        signal_rows.append(b)
        residual_rows.append(w)

    signal = np.asarray(signal_rows)
    residual = np.asarray(residual_rows)
    return signal, residual, signal + residual


def deterministic_equivalent_identity_closed_form(
    lambdas: np.ndarray,
    *,
    phi0: float,
    phi1: float,
    eta: float,
    rho2: float,
    varrho2: float,
    r2: float,
    sigma2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent identity-covariance specialization of Theorem 3.

    This implementation solves the scalar Marchenko-Pastur fixed point by its
    positive quadratic root rather than by the bisection and spectral averages
    used in ``deterministic_equivalent_diagonal``.  It is used only as a direct
    formula audit for Experiment 2.
    """
    signal_rows: list[float] = []
    residual_rows: list[float] = []
    for lam in np.asarray(lambdas, dtype=float):
        linear = lam + phi0 - 1.0
        v = (-linear + np.sqrt(linear**2 + 4.0 * lam)) / (2.0 * lam)
        vev = 1.0 / (v**-2 - phi0 / (1.0 + v) ** 2)
        veb = phi0 * vev / (1.0 + v) ** 2
        k1 = 1.0 / (1.0 + v)
        k2 = (veb + 1.0) / (1.0 + v) ** 2
        shift_residual = (v - veb) / (1.0 + v) ** 2

        b = (
            lam**2 * r2 * varrho2 / phi0 * (v - lam * vev)
            + r2 * rho2 * k2
        )
        w = (
            sigma2 * varrho2 * lam**2 * vev
            + sigma2 * phi0 * rho2 / lam * shift_residual
        )
        if eta == 1.0:
            b += r2 * phi1 / phi0 * (1.0 - 2.0 * k1 + k2)
            b += r2 * k2
            w += sigma2 * phi1 / lam * (k1 - k2)
            w += sigma2 * (1.0 + phi0 / lam * (k1 - k2))
        signal_rows.append(b)
        residual_rows.append(w)

    signal = np.asarray(signal_rows)
    residual = np.asarray(residual_rows)
    return signal, residual, signal + residual


def _summary_two_se(frame: pd.DataFrame, value: str, groups: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(groups, as_index=False)[value]
        .agg(mean="mean", sd="std", n="count")
        .assign(se=lambda x: x["sd"] / np.sqrt(x["n"]), two_se=lambda x: 2.0 * x["se"])
    )


def run_deterministic_equivalent_experiment(
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, pd.DataFrame]:
    """Red requirement 1: centered exact risks and Theorem 3 equivalents."""
    rng = np.random.default_rng(config.seed + 101)
    lambdas = config.lambdas
    path_rows: list[dict[str, float | int | str]] = []
    uniform_rows: list[dict[str, float | int | str]] = []
    scaling_rows: list[dict[str, float | int | str]] = []

    for p in config.dimensions:
        n0 = int(round(p / config.phi0))
        n1 = int(round(p / config.phi1))
        phi0_n, phi1_n = p / n0, p / n1
        for eta in config.etas:
            for design in range(config.deterministic_designs):
                X0 = rng.normal(size=(n0, p))
                target_noise = rng.normal(size=(n1, p))
                _, X0c, eigenvalues, eigenvectors = design_eigendecomposition(X0)
                for scenario, (rho2, varrho2) in SCENARIOS.items():
                    nu1 = mean_shift(p, n0, eta, rho2)
                    X1 = target_noise + nu1
                    gamma_base = external_base_weights(n0, eta, varrho2)
                    gamma_path, _ = ridge_augmented_path(
                        X0,
                        X0c,
                        eigenvalues,
                        eigenvectors,
                        X1.mean(axis=0),
                        gamma_base,
                        lambdas,
                    )
                    exact = exact_risk_components(
                        X0, gamma_path, nu1, config.r2, config.sigma2
                    )
                    equivalent = deterministic_equivalent_diagonal(
                        lambdas,
                        phi0=phi0_n,
                        phi1=phi1_n,
                        eta=eta,
                        rho2=rho2,
                        varrho2=varrho2,
                        r2=config.r2,
                        sigma2=config.sigma2,
                    )
                    scaled_exact = tuple(n0**eta * component for component in exact)
                    for component, exact_values, de_values in zip(
                        ("Signal", "Residual noise", "Total"),
                        scaled_exact,
                        equivalent,
                    ):
                        for lam, exact_value, de_value in zip(
                            lambdas, exact_values, de_values
                        ):
                            path_rows.append(
                                {
                                    "p": p,
                                    "n0": n0,
                                    "n1": n1,
                                    "eta": eta,
                                    "scenario": scenario,
                                    "design": design,
                                    "component": component,
                                    "lambda": lam,
                                    "scaled_exact": exact_value,
                                    "deterministic_equivalent": de_value,
                                }
                            )
                        uniform_rows.append(
                            {
                                "p": p,
                                "eta": eta,
                                "scenario": scenario,
                                "design": design,
                                "component": component,
                                "uniform_absolute_error": float(
                                    np.max(np.abs(exact_values - de_values))
                                ),
                            }
                        )
                    centered = gamma_base - 1.0 / n0
                    nonzero_shift = np.abs(nu1[np.abs(nu1) > 0])
                    scaling_rows.append(
                        {
                            "p": p,
                            "n0": n0,
                            "eta": eta,
                            "scenario": scenario,
                            "design": design,
                            "rho2_requested": rho2,
                            "rho2_realized": n0**eta * np.dot(nu1, nu1) / p,
                            "varrho2_requested": varrho2,
                            "varrho2_realized": n0**eta * np.dot(centered, centered),
                            "shift_squared_norm": float(np.dot(nu1, nu1)),
                            "expected_shift_squared_norm": float(
                                rho2 * p / n0**eta
                            ),
                            "centered_weight_squared_norm": float(
                                np.dot(centered, centered)
                            ),
                            "expected_centered_weight_squared_norm": float(
                                varrho2 / n0**eta
                            ),
                            "nonzero_shift_coordinate_abs": float(
                                nonzero_shift[0] if len(nonzero_shift) else 0.0
                            ),
                            "vector_norm_exponent": -eta / 2.0,
                            "squared_norm_exponent": -eta,
                            "omitted_empirical_mean_scaled_order": float(
                                n0 ** (eta - 1.0) if eta < 1.0 else 0.0
                            ),
                            "base_weight_sum": gamma_base.sum(),
                            "base_depends_on_design": False,
                        }
                    )

    paths = pd.DataFrame(path_rows)
    uniform = pd.DataFrame(uniform_rows)
    scaling = pd.DataFrame(scaling_rows)
    paths.to_csv(ARTIFACT_DIR / "deterministic_equivalent_path_draws.csv.gz", index=False)
    uniform.to_csv(ARTIFACT_DIR / "deterministic_equivalent_uniform_errors.csv", index=False)
    scaling.to_csv(ARTIFACT_DIR / "external_weight_scaling_audit.csv", index=False)
    eta_half_audit = (
        scaling[scaling["eta"].eq(0.5)]
        .drop_duplicates(["p", "scenario"])
        .sort_values(["scenario", "p"])
    )
    eta_half_audit.to_csv(ARTIFACT_DIR / "eta_half_scaling_audit.csv", index=False)

    eta_half_error = (
        uniform[
            uniform["eta"].eq(0.5)
            & uniform["scenario"].eq("Coupled")
            & uniform["component"].eq("Total")
        ]
        .groupby("p", as_index=False)
        .agg(
            median_uniform_error=("uniform_absolute_error", "median"),
            mean_uniform_error=("uniform_absolute_error", "mean"),
        )
    )
    eta_half_scale = eta_half_audit[
        eta_half_audit["scenario"].eq("Coupled")
    ][["p", "n0", "omitted_empirical_mean_scaled_order"]]
    eta_half_diagnostic = eta_half_error.merge(eta_half_scale, on="p")
    eta_half_diagnostic.to_csv(
        ARTIFACT_DIR / "eta_half_finite_sample_diagnostic.csv", index=False
    )

    largest = paths[paths["p"].eq(max(config.dimensions))]
    mean_paths = (
        largest.groupby(
            ["p", "eta", "scenario", "component", "lambda"], as_index=False
        )[["scaled_exact", "deterministic_equivalent"]]
        .mean()
    )
    mean_paths.to_csv(ARTIFACT_DIR / "deterministic_equivalent_path_summary.csv", index=False)

    fig, axes = plt.subplots(3, 3, figsize=(18.5, 13.5), sharex=True)
    for row, eta in enumerate(config.etas):
        panel = mean_paths[mean_paths["eta"].eq(eta)]
        for col, scenario in enumerate(SCENARIOS):
            ax = axes[row, col]
            data = panel[panel["scenario"].eq(scenario)]
            for component in ("Signal", "Residual noise", "Total"):
                line = data[data["component"].eq(component)]
                color = COMPONENT_COLORS[component]
                ax.plot(
                    line["lambda"], line["scaled_exact"],
                    color=color, label=f"{component} - exact"
                )
                ax.plot(
                    line["lambda"], line["deterministic_equivalent"],
                    color=color, linestyle="--", label=f"{component} - equivalent"
                )
            ax.set_xscale("log")
            if row == 0:
                ax.set_title(scenario, pad=10)
            if col == 0:
                ax.annotate(
                    fr"$\eta={eta:g}$",
                    xy=(-0.18, 0.5),
                    xycoords="axes fraction",
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=14,
                )
    fig.supxlabel(r"$\lambda$", y=0.075, fontsize=14)
    fig.supylabel("Scaled risk", x=0.018, fontsize=14)
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=3, frameon=False, columnspacing=1.8, handlelength=2.5
    )
    fig.subplots_adjust(
        left=0.105, right=0.985, bottom=0.155, top=0.96,
        hspace=0.26, wspace=0.24
    )
    save_figure(fig, "01_centered_risk_all_eta")

    fig, axes = plt.subplots(3, 3, figsize=(20, 16), sharex=True)
    for row, eta in enumerate(config.etas):
        for col, component in enumerate(("Signal", "Residual noise", "Total")):
            ax = axes[row, col]
            panel = uniform[
                uniform["eta"].eq(eta) & uniform["component"].eq(component)
            ].copy()
            panel["plot_error"] = np.maximum(
                panel["uniform_absolute_error"], 1.0e-8
            )
            sns.boxplot(
                data=panel,
                x="p",
                y="plot_error",
                hue="scenario",
                hue_order=list(SCENARIOS),
                palette=[COLORS["blue"], COLORS["vermillion"], COLORS["green"]],
                showfliers=True,
                ax=ax,
            )
            ax.set_yscale("log")
            ax.set_xlabel("")
            ax.set_ylabel(fr"$\eta={eta:g}$" if col == 0 else "")
            if row == 0:
                ax.set_title(COMPONENT_DISPLAY_LABELS[component], pad=10)
            if ax.legend_ is not None:
                ax.legend_.remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.supxlabel(r"$p$", y=0.065, fontsize=14)
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=3, frameon=False, columnspacing=2.0
    )
    fig.subplots_adjust(
        left=0.085, right=0.985, bottom=0.14, top=0.96,
        hspace=0.28, wspace=0.24
    )
    save_figure(fig, "02_uniform_error_distributions")
    return {"paths": paths, "uniform_errors": uniform, "scaling_audit": scaling}


def run_aspect_ratio_experiment(
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Red requirement 2: source and target aspect-ratio effects.

    The source-ratio figure emphasizes the mechanism requested in the paper:
    smooth deterministic-equivalent curves for five signal-to-weight ratios,
    decomposed into integrated signal, residual noise, and total risk.  Exact
    finite-design means at rho2/varrho2=1 are overlaid only as validation.

    The target-ratio figure separates finite-dimensional convergence from the
    continuous deterministic equivalent.  This avoids treating five simulated
    aspect ratios as though they were the theoretical curve itself.
    """
    rng = np.random.default_rng(config.seed + 202)
    selected_lambda = 0.316
    phi_scan = (0.50, 1.00, 2.00, 5.00, 10.00)
    phi_curve = np.geomspace(0.50, 1000.0, 321)
    ratio_levels = (0.5, 1.0, 5.0, 10.0, 20.0)
    aspect_etas = (0.0, 1.0)
    rows: list[dict[str, float | int | str]] = []
    p = config.aspect_dimension

    # Finite-design validation at the baseline rho2/varrho2=1.  The larger
    # dimension requested in the revision is retained at every aspect ratio.
    for eta in aspect_etas:
        for varied in ("Source", "Target"):
            for phi in phi_scan:
                phi0 = phi if varied == "Source" else config.phi0
                phi1 = phi if varied == "Target" else config.phi1
                n0, n1 = int(round(p / phi0)), int(round(p / phi1))
                phi0_n, phi1_n = p / n0, p / n1
                nu1 = mean_shift(p, n0, eta, 1.0)
                gamma_base = external_base_weights(n0, eta, 1.0)
                de_signal, de_residual, de_total = deterministic_equivalent_diagonal(
                    np.array([selected_lambda]),
                    phi0=phi0_n,
                    phi1=phi1_n,
                    eta=eta,
                    rho2=1.0,
                    varrho2=1.0,
                    r2=config.r2,
                    sigma2=config.sigma2,
                )
                for design in range(config.aspect_designs):
                    X0 = rng.normal(size=(n0, p))
                    X1 = rng.normal(size=(n1, p)) + nu1
                    _, X0c, eigenvalues, eigenvectors = design_eigendecomposition(X0)
                    gamma_path, _ = ridge_augmented_path(
                        X0, X0c, eigenvalues, eigenvectors,
                        X1.mean(axis=0), gamma_base, np.array([selected_lambda])
                    )
                    exact_signal, exact_residual, exact_total = exact_risk_components(
                        X0, gamma_path, nu1, config.r2, config.sigma2
                    )
                    scale = n0**eta
                    rows.append(
                        {
                            "eta": eta,
                            "varied_aspect_ratio": varied,
                            "phi": phi,
                            "phi0_realized": phi0_n,
                            "phi1_realized": phi1_n,
                            "p": p,
                            "n0": n0,
                            "n1": n1,
                            "design": design,
                            "lambda": selected_lambda,
                            "rho2": 1.0,
                            "varrho2": 1.0,
                            "rho2_over_varrho2": 1.0,
                            "scaled_exact_signal": scale * exact_signal[0],
                            "scaled_exact_residual": scale * exact_residual[0],
                            "scaled_exact_total_risk": scale * exact_total[0],
                            "deterministic_signal": de_signal[0],
                            "deterministic_residual": de_residual[0],
                            "deterministic_equivalent": de_total[0],
                        }
                    )
    draws = pd.DataFrame(rows)
    draws.to_csv(ARTIFACT_DIR / "aspect_ratio_draws.csv", index=False)
    summary = (
        draws.groupby(
            ["eta", "varied_aspect_ratio", "phi", "lambda"], as_index=False
        )
        .agg(
            exact_signal_mean=("scaled_exact_signal", "mean"),
            exact_signal_sd=("scaled_exact_signal", "std"),
            exact_residual_mean=("scaled_exact_residual", "mean"),
            exact_residual_sd=("scaled_exact_residual", "std"),
            exact_mean=("scaled_exact_total_risk", "mean"),
            exact_sd=("scaled_exact_total_risk", "std"),
            designs=("design", "nunique"),
            deterministic_signal=("deterministic_signal", "mean"),
            deterministic_residual=("deterministic_residual", "mean"),
            deterministic_equivalent=("deterministic_equivalent", "mean"),
        )
    )
    summary["exact_signal_two_se"] = (
        2.0 * summary["exact_signal_sd"] / np.sqrt(summary["designs"])
    )
    summary["exact_residual_two_se"] = (
        2.0 * summary["exact_residual_sd"] / np.sqrt(summary["designs"])
    )
    summary["exact_two_se"] = 2.0 * summary["exact_sd"] / np.sqrt(summary["designs"])
    summary.to_csv(ARTIFACT_DIR / "aspect_ratio_summary.csv", index=False)

    curve_rows: list[dict[str, float | str]] = []
    for eta in aspect_etas:
        for varied in ("Source", "Target"):
            for ratio in ratio_levels:
                for phi in phi_curve:
                    phi0 = float(phi) if varied == "Source" else config.phi0
                    phi1 = float(phi) if varied == "Target" else config.phi1
                    signal, residual, total = deterministic_equivalent_identity_closed_form(
                        np.array([selected_lambda]),
                        phi0=phi0,
                        phi1=phi1,
                        eta=eta,
                        rho2=ratio,
                        varrho2=1.0,
                        r2=config.r2,
                        sigma2=config.sigma2,
                    )
                    curve_rows.append(
                        {
                            "eta": eta,
                            "varied_aspect_ratio": varied,
                            "phi": float(phi),
                            "lambda": selected_lambda,
                            "rho2": ratio,
                            "varrho2": 1.0,
                            "rho2_over_varrho2": ratio,
                            "deterministic_signal": signal[0],
                            "deterministic_residual": residual[0],
                            "deterministic_equivalent": total[0],
                        }
                    )
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(
        ARTIFACT_DIR / "aspect_ratio_deterministic_curves.csv", index=False
    )

    ratio_colors = {
        0.5: "#0047FF",
        1.0: "#6F4CFF",
        5.0: "#E43D3D",
        10.0: "#F28E2B",
        20.0: "#167A3A",
    }
    phi_colors = {
        0.5: "#4C78FF",
        1.0: "#20C997",
        2.0: "#9DDC3A",
        5.0: "#E69F00",
        10.0: "#D55E00",
    }

    # Source aspect ratio: the compact 3x2 mechanism plot requested by the
    # revision.  Lines are direct formula evaluations; colored points and
    # two-MCSE bars validate the baseline ratio at the simulated phi values.
    component_rows = (
        ("deterministic_equivalent", "exact_mean", "exact_two_se", "Scaled risk"),
        ("deterministic_signal", "exact_signal_mean", "exact_signal_two_se", "Scaled signal"),
        ("deterministic_residual", "exact_residual_mean", "exact_residual_two_se", "Scaled residual variance"),
    )
    fig, axes = plt.subplots(
        3, 2, figsize=(13.8, 9.6), sharex=True, sharey="row"
    )
    for col, eta in enumerate(aspect_etas):
        exact_panel = summary[
            summary["varied_aspect_ratio"].eq("Source")
            & summary["eta"].eq(eta)
        ].sort_values("phi")
        for row, (curve_column, exact_column, se_column, ylabel) in enumerate(component_rows):
            ax = axes[row, col]
            for ratio in ratio_levels:
                panel = curves[
                    curves["varied_aspect_ratio"].eq("Source")
                    & curves["eta"].eq(eta)
                    & curves["rho2_over_varrho2"].eq(ratio)
                ].sort_values("phi")
                ax.plot(
                    panel["phi"], panel[curve_column],
                    color=ratio_colors[ratio],
                    label=fr"$\rho_\eta^2/\varrho_\eta^2={ratio:g}$",
                )
            ax.errorbar(
                exact_panel["phi"], exact_panel[exact_column],
                yerr=exact_panel[se_column], fmt="o", ms=4.8,
                color=ratio_colors[1.0], ecolor=ratio_colors[1.0],
                elinewidth=1.05, capsize=2.2, zorder=5,
            )
            ax.set_xscale("log")
            ax.set_xlim(0.45, 1100.0)
            ax.grid(True, which="major", alpha=0.24)
            ax.grid(True, which="minor", alpha=0.07)
            if row == 0:
                ax.set_title(fr"$\eta={eta:g}$", pad=10)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 2:
                ax.set_xlabel(r"$\phi_0$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=5, frameon=False, columnspacing=1.55, handlelength=2.5,
    )
    fig.subplots_adjust(
        left=0.095, right=0.985, bottom=0.085, top=0.895,
        hspace=0.12, wspace=0.14,
    )
    save_figure(fig, "03a_source_aspect_ratio_curves")

    # Target aspect ratio: show finite-dimensional convergence on the left and
    # the continuous direct formula on the right, matching the requested visual
    # logic while retaining the larger p=1280 endpoint.
    growth_p_grid = tuple(sorted(set(config.dimensions + (config.aspect_dimension,))))
    growth_designs = max(12, config.aspect_designs // 2)
    growth_rows: list[dict[str, float | int]] = []
    for eta in aspect_etas:
        for phi in phi_scan:
            for growth_p in growth_p_grid:
                if growth_p == config.aspect_dimension:
                    existing = draws[
                        draws["varied_aspect_ratio"].eq("Target")
                        & draws["eta"].eq(eta)
                        & draws["phi"].eq(phi)
                    ]
                    for record in existing.itertuples(index=False):
                        growth_rows.append(
                            {
                                "eta": eta,
                                "phi": phi,
                                "p": growth_p,
                                "design": int(record.design),
                                "scaled_exact_total_risk": float(record.scaled_exact_total_risk),
                            }
                        )
                    continue
                phi0 = config.phi0
                n0, n1 = int(round(growth_p / phi0)), int(round(growth_p / phi))
                nu1 = mean_shift(growth_p, n0, eta, 1.0)
                gamma_base = external_base_weights(n0, eta, 1.0)
                for design in range(growth_designs):
                    X0 = rng.normal(size=(n0, growth_p))
                    X1 = rng.normal(size=(n1, growth_p)) + nu1
                    _, X0c, eigenvalues, eigenvectors = design_eigendecomposition(X0)
                    gamma_path, _ = ridge_augmented_path(
                        X0, X0c, eigenvalues, eigenvectors,
                        X1.mean(axis=0), gamma_base, np.array([selected_lambda]),
                    )
                    exact_total = exact_risk_components(
                        X0, gamma_path, nu1, config.r2, config.sigma2
                    )[2][0]
                    growth_rows.append(
                        {
                            "eta": eta,
                            "phi": phi,
                            "p": growth_p,
                            "design": design,
                            "scaled_exact_total_risk": n0**eta * exact_total,
                        }
                    )
    growth_draws = pd.DataFrame(growth_rows)
    growth_draws.to_csv(ARTIFACT_DIR / "aspect_ratio_growth_draws.csv", index=False)
    growth_summary = (
        growth_draws.groupby(["eta", "phi", "p"], as_index=False)
        .agg(
            exact_mean=("scaled_exact_total_risk", "mean"),
            exact_sd=("scaled_exact_total_risk", "std"),
            designs=("design", "nunique"),
        )
    )
    growth_summary["exact_two_se"] = (
        2.0 * growth_summary["exact_sd"] / np.sqrt(growth_summary["designs"])
    )
    growth_summary.to_csv(
        ARTIFACT_DIR / "aspect_ratio_growth_summary.csv", index=False
    )

    fig, axes = plt.subplots(
        2, 2, figsize=(13.8, 8.1),
        gridspec_kw={"width_ratios": (2.35, 1.25)},
    )
    target_etas = (1.0, 0.0)
    for row, eta in enumerate(target_etas):
        finite_ax, theory_ax = axes[row]
        baseline_curve = curves[
            curves["varied_aspect_ratio"].eq("Target")
            & curves["eta"].eq(eta)
            & curves["rho2_over_varrho2"].eq(1.0)
        ].sort_values("phi")
        theory_ax.plot(
            baseline_curve["phi"], baseline_curve["deterministic_equivalent"],
            color=COLORS["blue"], lw=2.6,
        )
        theory_ax.set_xscale("log")
        theory_ax.set_xlim(0.45, 1100.0)
        for phi in phi_scan:
            color = phi_colors[phi]
            panel = growth_summary[
                growth_summary["eta"].eq(eta)
                & growth_summary["phi"].eq(phi)
            ].sort_values("p")
            finite_ax.errorbar(
                panel["p"], panel["exact_mean"], yerr=panel["exact_two_se"],
                color=color, marker="o", ms=4.4, lw=1.45,
                elinewidth=0.95, capsize=2.0,
            )
            signal, residual, total = deterministic_equivalent_identity_closed_form(
                np.array([selected_lambda]),
                phi0=config.phi0,
                phi1=phi,
                eta=eta,
                rho2=1.0,
                varrho2=1.0,
                r2=config.r2,
                sigma2=config.sigma2,
            )
            level = total[0]
            finite_ax.axhline(level, color=color, lw=1.05, ls="--", alpha=0.70)
            theory_ax.scatter(
                [phi], [level], s=40, color=color,
                edgecolor="white", linewidth=0.65, zorder=5,
            )
        finite_ax.set_ylabel(fr"$\eta={eta:g}$" + "\nScaled risk")
        finite_ax.grid(True, which="major", alpha=0.24)
        theory_ax.grid(True, which="major", alpha=0.24)
        theory_ax.grid(True, which="minor", alpha=0.07)
        if row == 0:
            finite_ax.set_title("Finite-dimensional convergence", pad=10)
            theory_ax.set_title("Deterministic equivalent", pad=10)
        if row == 1:
            finite_ax.set_xlabel(r"$p$")
            theory_ax.set_xlabel(r"$\phi_1$")

    from matplotlib.lines import Line2D

    phi_handles = [
        Line2D(
            [0], [0], color=phi_colors[phi], marker="o", lw=1.6,
            label=fr"$\phi_1={phi:g}$",
        )
        for phi in phi_scan
    ]
    guide_handles = [
        Line2D(
            [0], [0], color=COLORS["blue"], lw=2.6,
            label="Continuous deterministic equivalent",
        ),
        Line2D(
            [0], [0], color=COLORS["gray"], lw=1.1, ls="--",
            label="Asymptotic level at selected aspect ratio",
        ),
    ]
    fig.legend(
        handles=phi_handles,
        loc="lower center", bbox_to_anchor=(0.5, 0.055),
        ncol=5, frameon=False, columnspacing=1.7, handlelength=2.3,
    )
    fig.legend(
        handles=guide_handles,
        loc="lower center", bbox_to_anchor=(0.5, 0.002),
        ncol=2, frameon=False, columnspacing=2.1, handlelength=2.8,
    )
    fig.subplots_adjust(
        left=0.09, right=0.985, bottom=0.19, top=0.92,
        hspace=0.28, wspace=0.22,
    )
    save_figure(fig, "03b_target_aspect_ratio_curves")

    pd.DataFrame(
        [
            {
                "selected_lambda": selected_lambda,
                "fixed_phi0": config.phi0,
                "fixed_phi1": config.phi1,
                "eta_values": "0;1",
                "rho2_over_varrho2_levels": ";".join(f"{x:g}" for x in ratio_levels),
                "simulated_phi_values": ";".join(f"{x:g}" for x in phi_scan),
                "continuous_phi_min": min(phi_curve),
                "continuous_phi_max": max(phi_curve),
                "finite_dimension_grid": ";".join(str(x) for x in growth_p_grid),
                "finite_designs_per_small_dimension": growth_designs,
                "finite_designs_at_largest_dimension": config.aspect_designs,
            }
        ]
    ).to_csv(ARTIFACT_DIR / "aspect_ratio_plot_manifest.csv", index=False)
    return draws


def honest_ridge_base_weights(
    X0: np.ndarray,
    X0c: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    xbar0: np.ndarray,
    xbar_pilot: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Appendix D.2, equation (D.2), using only the target pilot fold."""
    d_pilot = xbar_pilot - xbar0
    projected = eigenvectors.T @ d_pilot
    direction = eigenvectors @ (projected / (eigenvalues + alpha))
    gamma = np.full(len(X0), 1.0 / len(X0)) + X0c @ direction / len(X0)
    if abs(gamma.sum() - 1.0) > 2.0e-10:
        raise RuntimeError("Honest ridge base weights are not normalized.")
    return gamma


def feasible_risk_geometry(
    X1_evaluation: np.ndarray,
    delta: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    gamma_path: np.ndarray,
    lambdas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Outcome-free factors in the plug-in risk estimator (4.8)."""
    projected_delta = eigenvectors.T @ delta
    centered_target = X1_evaluation - X1_evaluation.mean(axis=0)
    rotated_target = centered_target @ eigenvectors
    target_variances = np.sum(rotated_target**2, axis=0) / (len(X1_evaluation) - 1)
    quadratic = (lambdas**2 / len(eigenvalues)) * np.sum(
        projected_delta[:, None] ** 2
        / (eigenvalues[:, None] + lambdas[None, :]) ** 2,
        axis=0,
    )
    trace_correction = np.sum(
        (
            1.0
            - 2.0
            * lambdas[None, :]
            / (eigenvalues[:, None] + lambdas[None, :])
        )
        * target_variances[:, None],
        axis=0,
    # The paper uses normalized trace tr(A) = Tr(A) / p.
    ) / (len(eigenvalues) * len(X1_evaluation))
    signal_factor = np.maximum(quadratic + trace_correction, 0.0)
    weight_norm = np.sum(gamma_path**2, axis=0)
    return signal_factor, weight_norm


def source_gcv_curves(
    X0c: np.ndarray,
    Y0: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    lambdas: np.ndarray,
) -> np.ndarray:
    """Ordinary source-distribution GCV with an unpenalized intercept."""
    n0 = len(X0c)
    Y0c = Y0 - Y0.mean(axis=0, keepdims=True)
    threshold = max(float(eigenvalues.max()), 1.0) * 1.0e-10
    positive = eigenvalues > threshold
    s = eigenvalues[positive]
    U = eigenvectors[:, positive]
    left_scores = U.T @ (X0c.T @ Y0c)
    left_scores /= np.sqrt(n0 * s)[:, None]
    z2 = left_scores**2
    null_energy = np.maximum(np.sum(Y0c**2, axis=0) - np.sum(z2, axis=0), 0.0)
    residual_factors = (lambdas[None, :] / (s[:, None] + lambdas[None, :])) ** 2
    sse = null_energy[None, :] + residual_factors.T @ z2
    degrees_freedom = 1.0 + np.sum(
        s[:, None] / (s[:, None] + lambdas[None, :]), axis=0
    )
    return (sse / n0) / (1.0 - degrees_freedom[:, None] / n0) ** 2


def _inference_design(
    *,
    rng: np.random.Generator,
    config: ExperimentConfig,
    p: int,
    eta: float,
    phi0: float,
    phi1_total: float,
    pilot_fraction: float,
    outcomes: int,
    design: int,
    experiment_label: str,
) -> list[dict[str, float | int | bool | str]]:
    n0 = int(round(p / phi0))
    n1_total = int(round(p / phi1_total))
    n_pilot = int(round(pilot_fraction * n1_total))
    n_pilot = min(max(n_pilot, 2), n1_total - 2)
    n_evaluation = n1_total - n_pilot
    nu1 = mean_shift(
        p,
        n0,
        eta,
        config.calibration_shift_rho2,
        orientation="sparse",
        sparse_fraction=config.sparse_shift_fraction,
    )
    X0 = rng.normal(size=(n0, p))
    X1 = rng.normal(size=(n1_total, p)) + nu1
    X1_pilot = X1[:n_pilot]
    X1_evaluation = X1[n_pilot:]
    xbar0, X0c, eigenvalues, eigenvectors = design_eigendecomposition(X0)
    gamma_base = honest_ridge_base_weights(
        X0,
        X0c,
        eigenvalues,
        eigenvectors,
        xbar0,
        X1_pilot.mean(axis=0),
        config.base_ridge_penalty,
    )
    eps_pilot = X1_pilot.mean(axis=0) - nu1
    eps_evaluation = X1_evaluation.mean(axis=0) - nu1
    d_pilot = X1_pilot.mean(axis=0) - xbar0
    projected_d_pilot = eigenvectors.T @ d_pilot
    d3_right = (
        eps_evaluation
        - eps_pilot
        + config.base_ridge_penalty
        * (
            eigenvectors
            @ (
                projected_d_pilot
                / (eigenvalues + config.base_ridge_penalty)
            )
        )
    )
    d3_left = X1_evaluation.mean(axis=0) - X0.T @ gamma_base
    d3_error = float(np.linalg.norm(d3_left - d3_right))
    gamma_path, delta = ridge_augmented_path(
        X0,
        X0c,
        eigenvalues,
        eigenvectors,
        X1_evaluation.mean(axis=0),
        gamma_base,
        config.lambdas,
    )
    exact_signal, exact_residual, exact_total = exact_risk_components(
        X0, gamma_path, nu1, config.r2, config.sigma2
    )
    signal_factor, weight_norm = feasible_risk_geometry(
        X1_evaluation,
        delta,
        eigenvalues,
        eigenvectors,
        gamma_path,
        config.lambdas,
    )
    oracle_nuisance_feasible = (
        config.r2 * signal_factor + config.sigma2 * weight_norm
    )

    beta = rng.normal(scale=np.sqrt(config.r2 / p), size=(p, outcomes))
    epsilon = rng.normal(scale=np.sqrt(config.sigma2), size=(n0, outcomes))
    Y0 = X0 @ beta + epsilon
    target_epsilon = rng.normal(
        scale=np.sqrt(config.sigma2), size=(n1_total, outcomes)
    )
    treatment_effect = 1.0
    Y1 = X1 @ beta + target_epsilon + treatment_effect
    mom_r2, mom_sigma2 = equation_41_estimates_many(X0, Y0)
    sq_r2, sq_sigma2, sq_diagnostics = spectral_quasi_score_from_feature_eigen(
        X0,
        Y0,
        eigenvalues,
        eigenvectors,
        theta_bounds=DEFAULT_THETA_BOUNDS,
        pilot_r2=mom_r2,
        pilot_sigma2=mom_sigma2,
        grid_size=41,
    )
    feasible = (
        sq_r2[:, None] * signal_factor[None, :]
        + sq_sigma2[:, None] * weight_norm[None, :]
    )
    selected_ood = np.argmin(feasible, axis=1)
    gcv = source_gcv_curves(
        X0c, Y0, eigenvalues, eigenvectors, config.lambdas
    )
    selected_gcv = np.argmin(gcv, axis=0)
    selected_oracle = int(np.argmin(exact_total))
    outcome_index = np.arange(outcomes)
    selected_risk_hat = feasible[outcome_index, selected_ood]
    selected_exact_ood = exact_total[selected_ood]
    selected_exact_gcv = exact_total[selected_gcv]
    oracle_exact = float(exact_total[selected_oracle])

    estimates_by_lambda = gamma_path.T @ Y0
    mu_hat = estimates_by_lambda[selected_ood, outcome_index]
    mu_true = nu1 @ beta
    prediction_error = mu_hat - mu_true
    studentized = prediction_error / np.sqrt(selected_risk_hat)
    critical = stats.norm.ppf(0.975)
    covered = np.abs(prediction_error) <= critical * np.sqrt(selected_risk_hat)

    # The paper explicitly requires an outcome-location invariance rerun for
    # every covariate design.  Re-estimate both nuisance procedures and both
    # tuning criteria after adding the same large constant to every outcome.
    shift_constant = config.outcome_shift_constant
    Y0_shifted = Y0 + shift_constant
    Y1_shifted = Y1 + shift_constant
    shifted_mom_r2, shifted_mom_sigma2 = equation_41_estimates_many(
        X0, Y0_shifted
    )
    shifted_sq_r2, shifted_sq_sigma2, _ = spectral_quasi_score_from_feature_eigen(
        X0,
        Y0_shifted,
        eigenvalues,
        eigenvectors,
        theta_bounds=DEFAULT_THETA_BOUNDS,
        pilot_r2=shifted_mom_r2,
        pilot_sigma2=shifted_mom_sigma2,
        grid_size=41,
    )
    shifted_feasible = (
        shifted_sq_r2[:, None] * signal_factor[None, :]
        + shifted_sq_sigma2[:, None] * weight_norm[None, :]
    )
    shifted_selected_ood = np.argmin(shifted_feasible, axis=1)
    shifted_gcv = source_gcv_curves(
        X0c, Y0_shifted, eigenvalues, eigenvectors, config.lambdas
    )
    shifted_selected_gcv = np.argmin(shifted_gcv, axis=0)
    shifted_estimates_by_lambda = gamma_path.T @ Y0_shifted
    shifted_mu_hat = shifted_estimates_by_lambda[
        shifted_selected_ood, outcome_index
    ]
    shifted_prediction_error = shifted_mu_hat - (mu_true + shift_constant)
    treated_mean = Y1.mean(axis=0)
    shifted_treated_mean = Y1_shifted.mean(axis=0)
    att_hat = treated_mean - mu_hat
    shifted_att_hat = shifted_treated_mean - shifted_mu_hat

    scaled = n0**eta
    risk_sup_error = scaled * np.max(
        np.abs(feasible - exact_total[None, :]), axis=1
    )
    oracle_nuisance_risk_sup_error = float(
        scaled * np.max(np.abs(oracle_nuisance_feasible - exact_total))
    )
    rows: list[dict[str, float | int | bool | str]] = []
    for draw in range(outcomes):
        rows.append(
            {
                "experiment": experiment_label,
                "p": p,
                "eta": eta,
                "phi0": p / n0,
                "phi1_total": p / n1_total,
                "phi_pilot": p / n_pilot,
                "phi_evaluation": p / n_evaluation,
                "n0": n0,
                "n1_total": n1_total,
                "n_pilot": n_pilot,
                "n_evaluation": n_evaluation,
                "pilot_fraction": n_pilot / n1_total,
                "calibration_rho2_realized": float(
                    n0**eta * np.dot(nu1, nu1) / p
                ),
                "sparse_shift_nonzero_coordinates": int(np.count_nonzero(nu1)),
                "design": design,
                "draw": draw,
                "rhat2_sq": sq_r2[draw],
                "sigmahat2_sq": sq_sigma2[draw],
                "rhat2_two_moment": mom_r2[draw],
                "sigmahat2_two_moment": mom_sigma2[draw],
                "sq_boundary": bool(sq_diagnostics["boundary"][draw]),
                "scaled_uniform_risk_error": risk_sup_error[draw],
                "scaled_uniform_risk_error_oracle_nuisance": oracle_nuisance_risk_sup_error,
                "lambda_ood": config.lambdas[selected_ood[draw]],
                "lambda_gcv": config.lambdas[selected_gcv[draw]],
                "lambda_oracle": config.lambdas[selected_oracle],
                "scaled_regret_ood": scaled * (selected_exact_ood[draw] - oracle_exact),
                "scaled_regret_gcv": scaled * (selected_exact_gcv[draw] - oracle_exact),
                "scaled_regret_oracle": 0.0,
                "exact_signal_at_ood": exact_signal[selected_ood[draw]],
                "exact_residual_at_ood": exact_residual[selected_ood[draw]],
                "exact_total_at_ood": selected_exact_ood[draw],
                "exact_signal_at_gcv": exact_signal[selected_gcv[draw]],
                "exact_residual_at_gcv": exact_residual[selected_gcv[draw]],
                "exact_total_at_gcv": selected_exact_gcv[draw],
                "exact_signal_at_oracle": exact_signal[selected_oracle],
                "exact_residual_at_oracle": exact_residual[selected_oracle],
                "exact_total_at_oracle": oracle_exact,
                "risk_hat_at_ood": selected_risk_hat[draw],
                "risk_hat_to_exact": selected_risk_hat[draw] / selected_exact_ood[draw],
                "prediction_error": prediction_error[draw],
                "studentized_error": studentized[draw],
                "covered_95": bool(covered[draw]),
                "honest_d3_decomposition_error": d3_error,
                "outcome_shift_constant": shift_constant,
                "shift_rhat2_absolute_error": abs(
                    shifted_sq_r2[draw] - sq_r2[draw]
                ),
                "shift_sigmahat2_absolute_error": abs(
                    shifted_sq_sigma2[draw] - sq_sigma2[draw]
                ),
                "shift_lambda_ood_invariant": bool(
                    shifted_selected_ood[draw] == selected_ood[draw]
                ),
                "shift_lambda_gcv_invariant": bool(
                    shifted_selected_gcv[draw] == selected_gcv[draw]
                ),
                "shift_prediction_error_absolute_error": abs(
                    shifted_prediction_error[draw] - prediction_error[draw]
                ),
                "shift_att_absolute_error": abs(
                    shifted_att_hat[draw] - att_hat[draw]
                ),
                "base_weight_sum": gamma_base.sum(),
                "max_augmented_weight_sum_error": float(
                    np.max(np.abs(gamma_path.sum(axis=0) - 1.0))
                ),
            }
        )
    return rows


def _clustered_inference_summary(draws: pd.DataFrame) -> pd.DataFrame:
    by_design = (
        draws.groupby(["p", "eta", "design"], as_index=False)
        .agg(
            coverage=("covered_95", "mean"),
            mean_t=("studentized_error", "mean"),
            variance_t=("studentized_error", "var"),
            mean_scaled_uniform_error=("scaled_uniform_risk_error", "mean"),
        )
    )
    summary = (
        by_design.groupby(["p", "eta"], as_index=False)
        .agg(
            coverage=("coverage", "mean"),
            coverage_design_sd=("coverage", "std"),
            mean_t=("mean_t", "mean"),
            mean_t_design_sd=("mean_t", "std"),
            variance_t=("variance_t", "mean"),
            variance_t_design_sd=("variance_t", "std"),
            scaled_uniform_error=("mean_scaled_uniform_error", "mean"),
            scaled_uniform_error_design_sd=("mean_scaled_uniform_error", "std"),
            designs=("design", "nunique"),
        )
    )
    root_designs = np.sqrt(summary["designs"])
    summary["coverage_clustered_mcse"] = summary["coverage_design_sd"] / root_designs
    summary["mean_t_clustered_mcse"] = summary["mean_t_design_sd"] / root_designs
    summary["variance_t_clustered_mcse"] = summary["variance_t_design_sd"] / root_designs
    summary["scaled_uniform_error_clustered_mcse"] = (
        summary["scaled_uniform_error_design_sd"] / root_designs
    )
    return summary


def run_risk_tuning_calibration_experiment(
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, pd.DataFrame]:
    """Red requirements 3-4 with honest splitting and outcome-adaptive tuning."""
    rng = np.random.default_rng(config.seed + 303)
    rows: list[dict[str, float | int | bool | str]] = []
    for p in config.dimensions:
        for eta in config.calibration_etas:
            for design in range(config.inference_designs):
                rows.extend(
                    _inference_design(
                        rng=rng,
                        config=config,
                        p=p,
                        eta=eta,
                        phi0=config.phi0,
                        phi1_total=config.phi1,
                        pilot_fraction=config.pilot_fraction,
                        outcomes=config.outcomes_per_design,
                        design=design,
                        experiment_label="main",
                    )
                )
    draws = pd.DataFrame(rows)
    draws.to_csv(ARTIFACT_DIR / "risk_tuning_calibration_draws.csv.gz", index=False)
    calibration = _clustered_inference_summary(draws)
    calibration.to_csv(ARTIFACT_DIR / "predictive_calibration_summary.csv", index=False)

    risk_design_parts: list[pd.DataFrame] = []
    for estimator, column in (
        ("SQ plug-in", "scaled_uniform_risk_error"),
        ("Known nuisance (infeasible)", "scaled_uniform_risk_error_oracle_nuisance"),
    ):
        part = (
            draws.groupby(["p", "eta", "design"], as_index=False)[column]
            .mean()
            .rename(columns={column: "scaled_uniform_risk_error"})
        )
        part["nuisance_input"] = estimator
        risk_design_parts.append(part)
    risk_by_design_all = pd.concat(risk_design_parts, ignore_index=True)
    risk_by_design = risk_by_design_all[
        risk_by_design_all["nuisance_input"].eq("SQ plug-in")
    ].copy()
    risk_by_design.to_csv(
        ARTIFACT_DIR / "uniform_risk_estimation_by_design.csv", index=False
    )
    risk_by_design_all[
        risk_by_design_all["nuisance_input"].ne("SQ plug-in")
    ].to_csv(
        ARTIFACT_DIR / "known_nuisance_risk_diagnostic_by_design.csv", index=False
    )
    risk_summary = _summary_two_se(
        risk_by_design,
        "scaled_uniform_risk_error",
        ["p", "eta", "nuisance_input"],
    )
    risk_summary.to_csv(ARTIFACT_DIR / "uniform_risk_estimation_summary.csv", index=False)
    known_nuisance_summary = _summary_two_se(
        risk_by_design_all[
            risk_by_design_all["nuisance_input"].ne("SQ plug-in")
        ],
        "scaled_uniform_risk_error",
        ["p", "eta", "nuisance_input"],
    )
    known_nuisance_summary.to_csv(
        ARTIFACT_DIR / "known_nuisance_risk_diagnostic_summary.csv", index=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), sharey=False, sharex=True)
    for ax, eta in zip(axes.ravel(), config.calibration_etas):
        panel = risk_by_design[risk_by_design["eta"].eq(eta)]
        sns.boxplot(
            data=panel,
            x="p",
            y="scaled_uniform_risk_error",
            color=COLORS["blue"],
            width=0.58,
            showfliers=True,
            ax=ax,
        )
        ax.set_title(fr"$\eta={eta:g}$", pad=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.supxlabel(r"$p$", y=0.035, fontsize=14)
    fig.supylabel("Scaled uniform error", x=0.02, fontsize=14)
    fig.subplots_adjust(
        left=0.10, right=0.985, bottom=0.10, top=0.95,
        hspace=0.30, wspace=0.24
    )
    save_figure(fig, "04_uniform_risk_estimation")

    tuning_long_rows: list[pd.DataFrame] = []
    for method, lambda_col, regret_col in (
        ("Target-aware", "lambda_ood", "scaled_regret_ood"),
        ("Source GCV", "lambda_gcv", "scaled_regret_gcv"),
        ("Exact-risk oracle", "lambda_oracle", "scaled_regret_oracle"),
    ):
        part = draws[["p", "eta", "design", "draw", lambda_col, regret_col]].copy()
        part.columns = ["p", "eta", "design", "draw", "selected_lambda", "scaled_regret"]
        part["method"] = method
        tuning_long_rows.append(part)
    tuning_long = pd.concat(tuning_long_rows, ignore_index=True)
    tuning_long.to_csv(ARTIFACT_DIR / "tuning_draws_long.csv.gz", index=False)
    tuning_long["log_selected_lambda"] = np.log(tuning_long["selected_lambda"])
    design_tuning = (
        tuning_long.groupby(["p", "eta", "design", "method"], as_index=False)
        .agg(
            scaled_regret=("scaled_regret", "mean"),
            log_selected_lambda=("log_selected_lambda", "mean"),
        )
    )
    design_tuning["selected_lambda"] = np.exp(
        design_tuning["log_selected_lambda"]
    )
    design_tuning.to_csv(ARTIFACT_DIR / "tuning_by_design.csv", index=False)
    tuning_summary = (
        design_tuning.groupby(["p", "eta", "method"], as_index=False)
        .agg(
            regret_mean=("scaled_regret", "mean"),
            regret_sd=("scaled_regret", "std"),
            lambda_mean=("selected_lambda", "mean"),
            lambda_sd=("selected_lambda", "std"),
            lambda_log_mean=("log_selected_lambda", "mean"),
            lambda_log_sd=("log_selected_lambda", "std"),
            designs=("design", "nunique"),
        )
    )
    tuning_summary["regret_two_se"] = 2.0 * tuning_summary["regret_sd"] / np.sqrt(tuning_summary["designs"])
    tuning_summary["lambda_two_se"] = 2.0 * tuning_summary["lambda_sd"] / np.sqrt(tuning_summary["designs"])
    tuning_summary["lambda_geometric_mean"] = np.exp(
        tuning_summary["lambda_log_mean"]
    )
    lambda_log_two_se = (
        2.0
        * tuning_summary["lambda_log_sd"]
        / np.sqrt(tuning_summary["designs"])
    )
    tuning_summary["lambda_geometric_lower"] = np.exp(
        tuning_summary["lambda_log_mean"] - lambda_log_two_se
    )
    tuning_summary["lambda_geometric_upper"] = np.exp(
        tuning_summary["lambda_log_mean"] + lambda_log_two_se
    )
    tuning_summary.to_csv(ARTIFACT_DIR / "tuning_summary.csv", index=False)

    method_order = ("Target-aware", "Source GCV", "Exact-risk oracle")
    palette = {
        "Target-aware": COLORS["blue"],
        "Source GCV": COLORS["vermillion"],
        "Exact-risk oracle": COLORS["gray"],
    }
    method_markers = {
        "Target-aware": "o",
        "Source GCV": "s",
        "Exact-risk oracle": "D",
    }
    method_offsets = {
        "Target-aware": -0.18,
        "Source GCV": 0.0,
        "Exact-risk oracle": 0.18,
    }
    p_positions = np.arange(len(config.dimensions), dtype=float)
    fig, axes = plt.subplots(4, 2, figsize=(14.2, 15.2), sharex=True)
    for row, eta in enumerate(config.calibration_etas):
        panel = design_tuning[design_tuning["eta"].eq(eta)]
        sns.boxplot(
            data=panel,
            x="p",
            y="scaled_regret",
            hue="method",
            hue_order=method_order,
            palette=palette,
            showfliers=True,
            width=0.68,
            linewidth=1.0,
            saturation=0.88,
            flierprops={
                "marker": "o", "markersize": 3.2,
                "markerfacecolor": "white", "markeredgecolor": ".28",
                "markeredgewidth": 0.75,
            },
            ax=axes[row, 0],
        )
        summary_panel = tuning_summary[tuning_summary["eta"].eq(eta)]
        for method in method_order:
            method_panel = (
                summary_panel[summary_panel["method"].eq(method)]
                .set_index("p")
                .reindex(config.dimensions)
            )
            center = method_panel["lambda_geometric_mean"].to_numpy(dtype=float)
            lower = method_panel["lambda_geometric_lower"].to_numpy(dtype=float)
            upper = method_panel["lambda_geometric_upper"].to_numpy(dtype=float)
            axes[row, 1].errorbar(
                p_positions + method_offsets[method],
                center,
                yerr=np.vstack((center - lower, upper - center)),
                fmt=method_markers[method],
                color=palette[method],
                markeredgecolor="white",
                markeredgewidth=0.7,
                markersize=7.6,
                capsize=4.0,
                elinewidth=1.6,
                capthick=1.4,
                linestyle="none",
                label=method,
            )
        axes[row, 0].set_ylabel(fr"$\eta={eta:g}$", rotation=0, labelpad=27, va="center")
        axes[row, 1].set_ylabel("")
        axes[row, 1].set_yscale("log")
        axes[row, 1].set_xticks(
            p_positions, labels=[str(p) for p in config.dimensions]
        )
        for col in range(2):
            if axes[row, col].legend_ is not None:
                axes[row, col].legend_.remove()
            axes[row, col].set_xlabel("")
            axes[row, col].grid(axis="x", alpha=0.10)
            axes[row, col].grid(axis="y", alpha=0.20)
    axes[0, 0].set_title("Scaled regret", pad=10)
    axes[0, 1].set_title(r"Selected $\lambda$", pad=10)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.supxlabel(r"Dimension $p$", y=0.052, fontsize=14)
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=3, frameon=False, columnspacing=2.0
    )
    fig.subplots_adjust(
        left=0.085, right=0.985, bottom=0.115, top=0.95,
        hspace=0.28, wspace=0.20
    )
    save_figure(fig, "05_tuning_regret_and_penalties")

    probabilities = np.linspace(0.01, 0.99, 99)
    normal_quantiles = stats.norm.ppf(probabilities)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 12), sharex=True, sharey=True)
    dimension_colors = (
        COLORS["blue"], COLORS["green"], COLORS["gold"], COLORS["vermillion"]
    )
    for ax, eta in zip(axes.ravel(), config.calibration_etas):
        for index, p in enumerate(config.dimensions):
            values = draws[
                draws["eta"].eq(eta) & draws["p"].eq(p)
            ]["studentized_error"].to_numpy()
            empirical = np.quantile(values, probabilities)
            ax.plot(
                normal_quantiles,
                empirical,
                color=dimension_colors[index],
                label=f"p={p}",
            )
        ax.plot([-2.6, 2.6], [-2.6, 2.6], color=COLORS["black"], linestyle="--", label="N(0,1)")
        ax.set_title(fr"$\eta={eta:g}$", pad=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
    handles, labels = axes[-1, -1].get_legend_handles_labels()
    fig.supxlabel("Standard Gaussian quantile", y=0.075, fontsize=14)
    fig.supylabel("Empirical studentized-error quantile", x=0.025, fontsize=14)
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=5, frameon=False, columnspacing=1.5
    )
    fig.subplots_adjust(
        left=0.105, right=0.985, bottom=0.15, top=0.95,
        hspace=0.27, wspace=0.20
    )
    save_figure(fig, "06_selected_penalty_qq")
    return {
        "draws": draws,
        "calibration": calibration,
        "risk_summary": risk_summary,
        "tuning_summary": tuning_summary,
    }


def run_orientation_supplement(
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Alternative target-shift orientations relative to a structured Sigma0."""
    rng = np.random.default_rng(config.seed + 404)
    p = config.orientation_dimension
    eta = 0.5
    n0 = int(round(p / config.phi0))
    n1 = int(round(p / config.phi1))
    sigma0 = np.linspace(0.5, 2.0, p)
    sigma1 = sigma0.copy()
    rows: list[dict[str, float | int | str]] = []
    for orientation in ("low", "high"):
        nu1 = mean_shift(p, n0, eta, 1.0, orientation=orientation)
        gamma_base = external_base_weights(n0, eta, 1.0)
        shift_weights = np.zeros(p)
        shift_weights[0 if orientation == "low" else p - 1] = 1.0
        de = deterministic_equivalent_diagonal(
            config.lambdas,
            phi0=p / n0,
            phi1=p / n1,
            eta=eta,
            rho2=1.0,
            varrho2=1.0,
            r2=config.r2,
            sigma2=config.sigma2,
            sigma0_eigenvalues=sigma0,
            sigma1_diagonal=sigma1,
            shift_weights=shift_weights,
        )[2]
        for design in range(config.deterministic_designs):
            X0 = rng.normal(size=(n0, p)) * np.sqrt(sigma0)
            X1 = rng.normal(size=(n1, p)) * np.sqrt(sigma1) + nu1
            _, X0c, eigenvalues, eigenvectors = design_eigendecomposition(X0)
            gamma_path, _ = ridge_augmented_path(
                X0,
                X0c,
                eigenvalues,
                eigenvectors,
                X1.mean(axis=0),
                gamma_base,
                config.lambdas,
            )
            exact_total = exact_risk_components(
                X0, gamma_path, nu1, config.r2, config.sigma2
            )[2]
            for lam, exact_value, de_value in zip(
                config.lambdas, n0**eta * exact_total, de
            ):
                rows.append(
                    {
                        "orientation": orientation,
                        "design": design,
                        "p": p,
                        "eta": eta,
                        "lambda": lam,
                        "scaled_exact_total_risk": exact_value,
                        "deterministic_equivalent": de_value,
                        "sigma0_eigenvalue_at_shift": sigma0[
                            0 if orientation == "low" else p - 1
                        ],
                    }
                )
    draws = pd.DataFrame(rows)
    draws.to_csv(ARTIFACT_DIR / "orientation_draws.csv", index=False)
    summary = (
        draws.groupby(["orientation", "lambda"], as_index=False)
        .agg(
            exact_mean=("scaled_exact_total_risk", "mean"),
            deterministic_equivalent=("deterministic_equivalent", "mean"),
        )
    )
    summary.to_csv(ARTIFACT_DIR / "orientation_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    for ax, orientation in zip(axes, ("low", "high")):
        panel = summary[summary["orientation"].eq(orientation)]
        ax.plot(
            panel["lambda"], panel["exact_mean"],
            color=COLORS["blue"], label="Exact risk mean"
        )
        ax.plot(
            panel["lambda"], panel["deterministic_equivalent"],
            color=COLORS["vermillion"], linestyle="--",
            label="Deterministic equivalent"
        )
        ax.set_xscale("log")
        label = "Low-variance direction" if orientation == "low" else "High-variance direction"
        ax.set_title(label, pad=9)
        ax.set_xlabel("")
    axes[0].set_ylabel("Scaled risk")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.supxlabel(r"$\lambda$", y=0.10, fontsize=14)
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=2, frameon=False, columnspacing=2.0
    )
    fig.subplots_adjust(
        left=0.09, right=0.985, bottom=0.23, top=0.92, wspace=0.18
    )
    save_figure(fig, "07_structured_covariance_orientations")
    return draws


def run_nuisance_diagnostics(
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Spectral quasi-likelihood versus the optional two-moment estimator."""
    rng = np.random.default_rng(config.seed + 505)
    rows: list[dict[str, float | int | bool | str]] = []
    for phi0 in (0.5, 1.25):
        for p in config.dimensions:
            n0 = int(round(p / phi0))
            for design in range(config.nuisance_designs):
                X0 = rng.normal(size=(n0, p))
                _, _, eigenvalues, eigenvectors = design_eigendecomposition(X0)
                beta = rng.normal(
                    scale=np.sqrt(config.r2 / p),
                    size=(p, config.nuisance_outcomes),
                )
                epsilon = rng.normal(
                    scale=np.sqrt(config.sigma2),
                    size=(n0, config.nuisance_outcomes),
                )
                Y0 = X0 @ beta + epsilon
                mm_r2, mm_sigma2 = equation_41_estimates_many(X0, Y0)
                sq_r2, sq_sigma2, sq_diag = spectral_quasi_score_from_feature_eigen(
                    X0,
                    Y0,
                    eigenvalues,
                    eigenvectors,
                    theta_bounds=DEFAULT_THETA_BOUNDS,
                    pilot_r2=mm_r2,
                    pilot_sigma2=mm_sigma2,
                    grid_size=41,
                )
                for draw in range(config.nuisance_outcomes):
                    for estimator, rhat, shat, boundary in (
                        (
                            "Spectral quasi-likelihood",
                            sq_r2[draw],
                            sq_sigma2[draw],
                            bool(sq_diag["boundary"][draw]),
                        ),
                        (
                            "Two-moment diagnostic",
                            mm_r2[draw],
                            mm_sigma2[draw],
                            bool(mm_r2[draw] <= 1.0e-12 or mm_sigma2[draw] <= 1.0e-10),
                        ),
                    ):
                        rows.append(
                            {
                                "phi0_requested": phi0,
                                "phi0": p / n0,
                                "p": p,
                                "design": design,
                                "draw": draw,
                                "estimator": estimator,
                                "rhat2": rhat,
                                "sigmahat2": shat,
                                "boundary": boundary,
                            }
                        )
    draws = pd.DataFrame(rows)
    draws.to_csv(ARTIFACT_DIR / "nuisance_diagnostic_draws.csv.gz", index=False)
    long = pd.concat(
        [
            draws.assign(parameter="Signal variance", estimate=draws["rhat2"], truth=config.r2),
            draws.assign(parameter="Residual variance", estimate=draws["sigmahat2"], truth=config.sigma2),
        ],
        ignore_index=True,
    )
    long["error"] = long["estimate"] - long["truth"]
    summary = (
        long.groupby(
            ["phi0_requested", "phi0", "p", "estimator", "parameter"],
            as_index=False,
        )
        .agg(
            bias=("error", "mean"),
            rmse=("error", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
            boundary_frequency=("boundary", "mean"),
            draws=("draw", "count"),
        )
    )
    summary.to_csv(ARTIFACT_DIR / "nuisance_diagnostic_summary.csv", index=False)

    styles = {
        ("Spectral quasi-likelihood", "Signal variance"): (COLORS["blue"], "o", -0.18),
        ("Spectral quasi-likelihood", "Residual variance"): (COLORS["blue"], "s", -0.06),
        ("Two-moment diagnostic", "Signal variance"): (COLORS["vermillion"], "o", 0.06),
        ("Two-moment diagnostic", "Residual variance"): (COLORS["vermillion"], "s", 0.18),
    }
    phi_values = sorted(summary["phi0_requested"].unique())
    fig, axes = plt.subplots(3, 2, figsize=(15.5, 15), sharex=True)
    for col, phi0 in enumerate(phi_values):
        panel = summary[np.isclose(summary["phi0_requested"], phi0)]
        x = np.arange(len(config.dimensions), dtype=float)
        for key, (color, marker, offset) in styles.items():
            estimator, parameter = key
            line = panel[
                panel["estimator"].eq(estimator)
                & panel["parameter"].eq(parameter)
            ]
            label = f"{estimator}; {parameter}"
            line = line.set_index("p").loc[list(config.dimensions)]
            axes[0, col].scatter(
                x + offset, line["bias"], color=color, marker=marker, label=label
            )
            axes[1, col].scatter(
                x + offset, line["rmse"], color=color, marker=marker, label=label
            )
        for estimator, color in (
            ("Spectral quasi-likelihood", COLORS["blue"]),
            ("Two-moment diagnostic", COLORS["vermillion"]),
        ):
            line = panel[
                panel["estimator"].eq(estimator)
                & panel["parameter"].eq("Signal variance")
            ]
            line = line.set_index("p").loc[list(config.dimensions)]
            boundary_offset = -0.08 if estimator == "Spectral quasi-likelihood" else 0.08
            axes[2, col].scatter(
                x + boundary_offset,
                line["boundary_frequency"],
                color=color,
                marker="o",
                label=estimator,
            )
        axes[0, col].axhline(0.0, color=COLORS["gray"], linestyle=":")
        axes[0, col].set_title(fr"$\phi_0={phi0:g}$", pad=9)
        axes[2, col].set_xlabel("")
        for row in range(3):
            axes[row, col].set_xticks(
                x, labels=[str(value) for value in config.dimensions]
            )
    axes[0, 0].set_ylabel("Bias")
    axes[1, 0].set_ylabel("Root mean-squared error")
    axes[2, 0].set_ylabel("Boundary-selection frequency")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.supxlabel(r"$p$", y=0.055, fontsize=14)
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=2, frameon=False, columnspacing=1.8
    )
    fig.subplots_adjust(
        left=0.09, right=0.985, bottom=0.13, top=0.95,
        hspace=0.24, wspace=0.22
    )
    save_figure(fig, "08_variance_component_diagnostics")
    return summary


def run_exact_risk_verification(
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Proposition 2 checked against repeated conditional outcome draws."""
    rng = np.random.default_rng(config.seed + 606)
    p = 80
    n0, n1 = int(round(p / config.phi0)), int(round(p / config.phi1))
    requested = np.array([0.1, 1.0, 10.0])
    lambdas = np.array([config.lambdas[np.argmin(abs(config.lambdas - x))] for x in requested])
    rows: list[dict[str, float | int | str]] = []
    conditional_draws = 2000
    for eta in (0.5, 1.0):
        nu1 = mean_shift(p, n0, eta, 1.0)
        gamma_base = external_base_weights(n0, eta, 1.0)
        for design in range(12):
            X0 = rng.normal(size=(n0, p))
            X1 = rng.normal(size=(n1, p)) + nu1
            _, X0c, eigenvalues, eigenvectors = design_eigendecomposition(X0)
            gamma_path, _ = ridge_augmented_path(
                X0, X0c, eigenvalues, eigenvectors,
                X1.mean(axis=0), gamma_base, lambdas
            )
            exact = exact_risk_components(
                X0, gamma_path, nu1, config.r2, config.sigma2
            )
            imbalance = X0.T @ gamma_path - nu1[:, None]
            beta = rng.normal(scale=np.sqrt(config.r2 / p), size=(p, conditional_draws))
            epsilon = rng.normal(scale=np.sqrt(config.sigma2), size=(n0, conditional_draws))
            signal_error = imbalance.T @ beta
            residual_error = gamma_path.T @ epsilon
            empirical = (
                np.mean(signal_error**2, axis=1),
                np.mean(residual_error**2, axis=1),
                np.mean((signal_error + residual_error) ** 2, axis=1),
            )
            for component, exact_values, empirical_values in zip(
                ("Signal", "Residual noise", "Total"), exact, empirical
            ):
                for lam, exact_value, empirical_value in zip(
                    lambdas, exact_values, empirical_values
                ):
                    rows.append(
                        {
                            "eta": eta,
                            "design": design,
                            "lambda": lam,
                            "component": component,
                            "exact_risk": exact_value,
                            "empirical_mean_squared_error": empirical_value,
                            "empirical_to_exact_ratio": empirical_value / exact_value,
                            "conditional_draws": conditional_draws,
                        }
                    )
    draws = pd.DataFrame(rows)
    draws.to_csv(ARTIFACT_DIR / "exact_risk_verification_draws.csv", index=False)
    summary = (
        draws.groupby(["eta", "lambda", "component"], as_index=False)
        .agg(
            exact_risk=("exact_risk", "mean"),
            empirical_mse=("empirical_mean_squared_error", "mean"),
            empirical_to_exact_ratio=("empirical_to_exact_ratio", "mean"),
        )
    )
    summary.to_csv(ARTIFACT_DIR / "exact_risk_verification_summary.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    for ax, component in zip(axes, ("Signal", "Residual noise", "Total")):
        panel = draws[draws["component"].eq(component)]
        for eta, color, marker in (
            (0.5, COLORS["blue"], "o"),
            (1.0, COLORS["vermillion"], "s"),
        ):
            line = panel[panel["eta"].eq(eta)]
            ax.scatter(
                line["exact_risk"], line["empirical_mean_squared_error"],
                color=color, marker=marker, alpha=0.65, label=fr"$\eta={eta:g}$"
            )
        limits = np.array([
            min(panel["exact_risk"].min(), panel["empirical_mean_squared_error"].min()),
            max(panel["exact_risk"].max(), panel["empirical_mean_squared_error"].max()),
        ])
        ax.plot(limits, limits, color=COLORS["black"], linestyle="--")
        ax.set_title(component, pad=9)
        ax.set_xlabel("")
    axes[0].set_ylabel("")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.supxlabel("Exact finite-design risk", y=0.10, fontsize=14)
    fig.supylabel("Repeated-draw mean squared error", x=0.02, fontsize=14)
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=2, frameon=False, columnspacing=2.0
    )
    fig.subplots_adjust(
        left=0.10, right=0.985, bottom=0.23, top=0.92, wspace=0.24
    )
    save_figure(fig, "09_exact_risk_verification")
    return summary


def run_split_sensitivity(
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Under/overparameterized designs and target-split sensitivity."""
    rng = np.random.default_rng(config.seed + 707)
    p = 160
    split_configs = {
        "Balanced 50/50": 0.50,
        "Pilot-light 33/67": 1.0 / 3.0,
        "Evaluation-light 67/33": 2.0 / 3.0,
    }
    rows: list[dict[str, float | int | bool | str]] = []
    for phi0 in (0.5, 1.25):
        for eta in (0.5, 1.0):
            for split_label, pilot_fraction in split_configs.items():
                for design in range(config.split_designs):
                    part = _inference_design(
                        rng=rng,
                        config=config,
                        p=p,
                        eta=eta,
                        phi0=phi0,
                        phi1_total=config.phi1,
                        pilot_fraction=pilot_fraction,
                        outcomes=config.split_outcomes,
                        design=design,
                        experiment_label="split_sensitivity",
                    )
                    for row in part:
                        row["split"] = split_label
                    rows.extend(part)
    draws = pd.DataFrame(rows)
    draws.to_csv(ARTIFACT_DIR / "split_sensitivity_draws.csv.gz", index=False)
    long_rows: list[pd.DataFrame] = []
    for method, suffix in (
        ("Target-aware", "ood"),
        ("Source GCV", "gcv"),
        ("Exact-risk oracle", "oracle"),
    ):
        part = draws[
            ["phi0", "eta", "split", "design", "draw",
             f"exact_signal_at_{suffix}", f"exact_residual_at_{suffix}", f"exact_total_at_{suffix}"]
        ].copy()
        part.columns = [
            "phi0", "eta", "split", "design", "draw",
            "integrated_signal", "residual_noise", "total_risk"
        ]
        part["method"] = method
        long_rows.append(part)
    long = pd.concat(long_rows, ignore_index=True)
    long["predictive_rmse"] = np.sqrt(long["total_risk"])
    design_summary = (
        long.groupby(
            ["phi0", "eta", "split", "method", "design"], as_index=False
        )
        .agg(
            integrated_signal=("integrated_signal", "mean"),
            residual_noise=("residual_noise", "mean"),
            predictive_rmse=("predictive_rmse", "mean"),
        )
    )
    design_summary.to_csv(
        ARTIFACT_DIR / "split_sensitivity_by_design.csv", index=False
    )
    summary = (
        design_summary.groupby(
            ["phi0", "eta", "split", "method"], as_index=False
        )
        .agg(
            integrated_signal=("integrated_signal", "mean"),
            integrated_signal_sd=("integrated_signal", "std"),
            residual_noise=("residual_noise", "mean"),
            residual_noise_sd=("residual_noise", "std"),
            predictive_rmse=("predictive_rmse", "mean"),
            predictive_rmse_sd=("predictive_rmse", "std"),
            designs=("design", "nunique"),
        )
    )
    for metric in ("integrated_signal", "residual_noise", "predictive_rmse"):
        summary[f"{metric}_two_se"] = (
            2.0 * summary[f"{metric}_sd"] / np.sqrt(summary["designs"])
        )
    summary.to_csv(ARTIFACT_DIR / "split_sensitivity_summary.csv", index=False)

    palette = {
        "Target-aware": COLORS["blue"],
        "Source GCV": COLORS["vermillion"],
        "Exact-risk oracle": COLORS["gray"],
    }
    split_order = list(split_configs)
    split_tick_labels = ["Balanced", "Pilot light", "Evaluation light"]
    for eta in (0.5, 1.0):
        fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
        for col, phi0 in enumerate((0.5, 1.25)):
            panel = summary[
                np.isclose(summary["phi0"], phi0) & summary["eta"].eq(eta)
            ]
            x = np.arange(len(split_order), dtype=float)
            offsets = {
                "Target-aware": -0.18,
                "Source GCV": 0.0,
                "Exact-risk oracle": 0.18,
            }
            markers = {
                "Target-aware": "o",
                "Source GCV": "s",
                "Exact-risk oracle": "^",
            }
            for method, color in palette.items():
                points = (
                    panel[panel["method"].eq(method)]
                    .set_index("split")
                    .loc[split_order]
                )
                axes[0, col].errorbar(
                    x + offsets[method],
                    points["integrated_signal"],
                    yerr=points["integrated_signal_two_se"],
                    color=color,
                    marker=markers[method],
                    linestyle="none",
                    capsize=3,
                    label=method,
                )
                axes[1, col].errorbar(
                    x + offsets[method],
                    points["residual_noise"],
                    yerr=points["residual_noise_two_se"],
                    color=color,
                    marker=markers[method],
                    linestyle="none",
                    capsize=3,
                    label=method,
                )
            axes[0, col].set_title(fr"$\phi_0={phi0:g}$", pad=9)
            for row in range(2):
                axes[row, col].set_xticks(x, split_tick_labels, rotation=0)
                axes[row, col].set_xlabel("")
        axes[0, 0].set_ylabel("Integrated signal component")
        axes[1, 0].set_ylabel("Residual-noise component")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.text(0.015, 0.965, fr"$\eta={eta:g}$", ha="left", va="top", fontsize=14)
        fig.supxlabel("Target split", y=0.075, fontsize=14)
        fig.legend(
            handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
            ncol=3, frameon=False, columnspacing=1.8
        )
        fig.subplots_adjust(
            left=0.09, right=0.985, bottom=0.16, top=0.94,
            hspace=0.25, wspace=0.22
        )
        save_figure(fig, f"10_split_sensitivity_eta_{str(eta).replace('.', 'p')}")
    return summary


def run_formula_implementation_audit(
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Cross-check spectral implementations against the paper's matrix formulas."""
    rng = np.random.default_rng(config.seed + 808)
    p = 24
    n0 = 32
    n1_total = 32
    n_pilot = n1_total // 2
    eta = 0.5
    alpha = config.base_ridge_penalty
    lambdas = config.lambdas
    nu1 = mean_shift(
        p,
        n0,
        eta,
        config.calibration_shift_rho2,
        orientation="sparse",
        sparse_fraction=config.sparse_shift_fraction,
    )
    X0 = rng.normal(size=(n0, p))
    X1 = rng.normal(size=(n1_total, p)) + nu1
    X1_pilot = X1[:n_pilot]
    X1_evaluation = X1[n_pilot:]
    xbar0, X0c, eigenvalues, eigenvectors = design_eigendecomposition(X0)
    S0c = X0c.T @ X0c / n0

    gamma_spectral = honest_ridge_base_weights(
        X0,
        X0c,
        eigenvalues,
        eigenvectors,
        xbar0,
        X1_pilot.mean(axis=0),
        alpha,
    )
    d_pilot = X1_pilot.mean(axis=0) - xbar0
    gamma_direct = (
        np.full(n0, 1.0 / n0)
        + X0c @ np.linalg.solve(S0c + alpha * np.eye(p), d_pilot) / n0
    )

    gamma_path_spectral, delta = ridge_augmented_path(
        X0,
        X0c,
        eigenvalues,
        eigenvectors,
        X1_evaluation.mean(axis=0),
        gamma_spectral,
        lambdas,
    )
    gamma_path_direct = np.column_stack(
        [
            gamma_direct
            + X0c
            @ np.linalg.solve(S0c + lam * np.eye(p), delta)
            / n0
            for lam in lambdas
        ]
    )

    signal_spectral, weight_norm_spectral = feasible_risk_geometry(
        X1_evaluation,
        delta,
        eigenvalues,
        eigenvectors,
        gamma_path_spectral,
        lambdas,
    )
    sigmahat1 = np.cov(X1_evaluation, rowvar=False, ddof=1)
    signal_direct: list[float] = []
    for lam in lambdas:
        M = np.linalg.inv(S0c + lam * np.eye(p))
        raw_signal = (
            lam**2 / p * delta @ (M @ M) @ delta
            + np.trace((np.eye(p) - 2.0 * lam * M) @ sigmahat1)
            / (p * len(X1_evaluation))
        )
        signal_direct.append(max(float(raw_signal), 0.0))
    signal_direct_array = np.asarray(signal_direct)
    weight_norm_direct = np.sum(gamma_path_direct**2, axis=0)

    beta = rng.normal(scale=np.sqrt(config.r2 / p), size=(p, 5))
    epsilon = rng.normal(scale=np.sqrt(config.sigma2), size=(n0, 5))
    Y0 = X0 @ beta + epsilon
    gcv_spectral = source_gcv_curves(
        X0c, Y0, eigenvalues, eigenvectors, lambdas
    )
    Y0c = Y0 - Y0.mean(axis=0, keepdims=True)
    gcv_direct = []
    for lam in lambdas:
        H = X0c @ np.linalg.solve(
            X0c.T @ X0c + n0 * lam * np.eye(p), X0c.T
        )
        residual = (np.eye(n0) - H) @ Y0c
        df = 1.0 + np.trace(H)
        gcv_direct.append(
            np.sum(residual**2, axis=0) / n0 / (1.0 - df / n0) ** 2
        )
    gcv_direct_array = np.asarray(gcv_direct)

    theorem3_identity_differences: list[float] = []
    for audit_eta in (0.0, 0.5, 1.0):
        for varied in ("Source", "Target"):
            for phi in (0.5, 1.0, 2.0, 5.0, 10.0):
                phi0 = phi if varied == "Source" else config.phi0
                phi1 = phi if varied == "Target" else config.phi1
                general = deterministic_equivalent_diagonal(
                    lambdas,
                    phi0=phi0,
                    phi1=phi1,
                    eta=audit_eta,
                    rho2=1.0,
                    varrho2=1.0,
                    r2=config.r2,
                    sigma2=config.sigma2,
                )
                closed = deterministic_equivalent_identity_closed_form(
                    lambdas,
                    phi0=phi0,
                    phi1=phi1,
                    eta=audit_eta,
                    rho2=1.0,
                    varrho2=1.0,
                    r2=config.r2,
                    sigma2=config.sigma2,
                )
                theorem3_identity_differences.extend(
                    float(np.max(np.abs(a - b)))
                    for a, b in zip(general, closed)
                )

    audit = pd.DataFrame(
        [
            (
                "Appendix D.2 base weights",
                float(np.max(np.abs(gamma_spectral - gamma_direct))),
                1.0e-10,
            ),
            (
                "Centered augmentation path",
                float(np.max(np.abs(gamma_path_spectral - gamma_path_direct))),
                1.0e-10,
            ),
            (
                "Equation (4.8) signal geometry with normalized trace",
                float(np.max(np.abs(signal_spectral - signal_direct_array))),
                1.0e-10,
            ),
            (
                "Equation (4.8) weight norm",
                float(np.max(np.abs(weight_norm_spectral - weight_norm_direct))),
                1.0e-10,
            ),
            (
                "Source GCV with unpenalized intercept",
                float(np.max(np.abs(gcv_spectral - gcv_direct_array))),
                1.0e-9,
            ),
            (
                "Theorem 3 identity-covariance formula for Experiment 2",
                float(max(theorem3_identity_differences)),
                1.0e-10,
            ),
        ],
        columns=["formula", "max_absolute_difference", "tolerance"],
    )
    audit["passed"] = audit["max_absolute_difference"] < audit["tolerance"]
    audit.to_csv(ARTIFACT_DIR / "formula_implementation_audit.csv", index=False)
    return audit


def write_validation_and_audit(
    config: ExperimentConfig,
    deterministic: dict[str, pd.DataFrame],
    inference: dict[str, pd.DataFrame],
    exact_verification: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, bool | float | int]]:
    scaling = deterministic["scaling_audit"]
    inference_draws = inference["draws"]
    formula_audit = run_formula_implementation_audit(config)
    expected_inference_rows = (
        len(config.dimensions)
        * len(config.calibration_etas)
        * config.inference_designs
        * config.outcomes_per_design
    )
    validation: dict[str, bool | float | int] = {
        "external_weights_are_design_independent": bool(
            (~scaling["base_depends_on_design"]).all()
        ),
        "max_external_weight_sum_error": float(
            np.max(np.abs(scaling["base_weight_sum"] - 1.0))
        ),
        "max_rho2_scaling_error": float(
            np.max(np.abs(scaling["rho2_realized"] - scaling["rho2_requested"]))
        ),
        "max_varrho2_scaling_error": float(
            np.max(
                np.abs(
                    scaling["varrho2_realized"] - scaling["varrho2_requested"]
                )
            )
        ),
        "inference_rows_expected": expected_inference_rows,
        "inference_rows_observed": int(len(inference_draws)),
        "all_inference_draws_retained": bool(
            len(inference_draws) == expected_inference_rows
        ),
        "all_sq_estimates_finite": bool(
            np.isfinite(
                inference_draws[["rhat2_sq", "sigmahat2_sq"]].to_numpy()
            ).all()
        ),
        "all_selected_risk_estimates_positive": bool(
            (inference_draws["risk_hat_at_ood"] > 0).all()
        ),
        "max_honest_base_weight_sum_error": float(
            np.max(np.abs(inference_draws["base_weight_sum"] - 1.0))
        ),
        "max_augmented_weight_sum_error": float(
            inference_draws["max_augmented_weight_sum_error"].max()
        ),
        "paper_dimension_grid_exact": bool(
            tuple(sorted(inference_draws["p"].unique())) == config.dimensions
            == (80, 160, 320, 640)
        ),
        "paper_calibration_eta_grid_exact": bool(
            tuple(sorted(inference_draws["eta"].unique()))
            == config.calibration_etas
            == (0.0, 0.5, 0.75, 1.0)
        ),
        "paper_sparse_shift_rho2_exact": bool(
            config.calibration_shift_rho2 == 0.5
            and config.sparse_shift_fraction == 0.1
            and np.max(
                np.abs(
                    inference_draws["calibration_rho2_realized"]
                    - config.calibration_shift_rho2
                )
            )
            < 1.0e-12
            and (
                inference_draws["sparse_shift_nonzero_coordinates"]
                == inference_draws["p"].map(
                    lambda value: max(
                        1, int(round(config.sparse_shift_fraction * value))
                    )
                )
            ).all()
        ),
        "target_total_sample_size_held_fixed_before_split": bool(
            (
                inference_draws["n_pilot"]
                + inference_draws["n_evaluation"]
                == inference_draws["n1_total"]
            ).all()
            and np.allclose(
                inference_draws["phi1_total"],
                inference_draws["p"] / inference_draws["n1_total"],
            )
        ),
        "max_honest_d3_decomposition_error": float(
            inference_draws["honest_d3_decomposition_error"].max()
        ),
        "outcome_shift_all_ood_penalties_invariant": bool(
            inference_draws["shift_lambda_ood_invariant"].all()
        ),
        "outcome_shift_all_gcv_penalties_invariant": bool(
            inference_draws["shift_lambda_gcv_invariant"].all()
        ),
        "outcome_shift_max_rhat2_error": float(
            inference_draws["shift_rhat2_absolute_error"].max()
        ),
        "outcome_shift_max_sigmahat2_error": float(
            inference_draws["shift_sigmahat2_absolute_error"].max()
        ),
        "outcome_shift_max_prediction_error": float(
            inference_draws["shift_prediction_error_absolute_error"].max()
        ),
        "outcome_shift_max_att_error": float(
            inference_draws["shift_att_absolute_error"].max()
        ),
        "standard_normal_critical_value_used": True,
        "monte_carlo_draws_never_set_critical_values": True,
        "exact_risk_verification_max_mean_ratio_error": float(
            np.max(np.abs(exact_verification["empirical_to_exact_ratio"] - 1.0))
        ),
        "single_font_family": True,
        "shared_12_14_point_hierarchy": True,
        "all_direct_formula_cross_checks_pass": bool(
            formula_audit["passed"].all()
        ),
        "max_direct_formula_cross_check_difference": float(
            formula_audit["max_absolute_difference"].max()
        ),
    }
    validation["all_core_checks_pass"] = bool(
        validation["external_weights_are_design_independent"]
        and validation["max_external_weight_sum_error"] < 1.0e-10
        and validation["max_rho2_scaling_error"] < 1.0e-10
        and validation["max_varrho2_scaling_error"] < 1.0e-10
        and validation["all_inference_draws_retained"]
        and validation["all_sq_estimates_finite"]
        and validation["all_selected_risk_estimates_positive"]
        and validation["max_honest_base_weight_sum_error"] < 1.0e-10
        and validation["max_augmented_weight_sum_error"] < 1.0e-10
        and validation["paper_dimension_grid_exact"]
        and validation["paper_calibration_eta_grid_exact"]
        and validation["paper_sparse_shift_rho2_exact"]
        and validation["target_total_sample_size_held_fixed_before_split"]
        and validation["max_honest_d3_decomposition_error"] < 1.0e-10
        and validation["outcome_shift_all_ood_penalties_invariant"]
        and validation["outcome_shift_all_gcv_penalties_invariant"]
        and validation["outcome_shift_max_rhat2_error"] < 5.0e-7
        and validation["outcome_shift_max_sigmahat2_error"] < 5.0e-7
        and validation["outcome_shift_max_prediction_error"] < 1.0e-10
        and validation["outcome_shift_max_att_error"] < 1.0e-10
        and validation["all_direct_formula_cross_checks_pass"]
    )
    (ARTIFACT_DIR / "validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )

    audit_rows = [
        (
            "Assumptions 2 and 10",
            "Gaussian beta and source errors are independent of all covariate arrays",
            True,
        ),
        (
            "Assumption 3",
            "Independent Gaussian source and target coordinates; bounded covariance spectra; proportional aspect ratios",
            True,
        ),
        (
            "Assumption 4",
            "Theorem 3 experiment uses normalized design-independent weights with exact coupled scaling",
            bool(
                validation["external_weights_are_design_independent"]
                and validation["max_rho2_scaling_error"] < 1.0e-10
                and validation["max_varrho2_scaling_error"] < 1.0e-10
            ),
        ),
        (
            "Assumption 5",
            "Gaussian coordinates satisfy the paper's uniform sub-Gaussian sufficient condition",
            True,
        ),
        (
            "Assumption 6",
            "Fixed aspect ratios, exact scaling constants, and fixed diagonal spectral laws",
            True,
        ),
        (
            "Assumption 7",
            "Independent Gaussian random coefficients and source errors with finite moments",
            True,
        ),
        (
            "Assumption 8",
            "A fixed total target sample is split: ridge base weights use only the pilot fold and risk uses the disjoint evaluation fold",
            bool(validation["target_total_sample_size_held_fixed_before_split"]),
        ),
        (
            "Assumption 9 / Corollary D.6",
            "The honest ridge base is exactly the Appendix D.2 construction covered by the paper",
            True,
        ),
        (
            "Direct formula cross-check",
            "Theorem 3's identity-covariance specialization, Appendix D.2 weights, centered augmentation, equation (4.8) with tr(A)=Tr(A)/p, and intercept-GCV agree with independent calculations",
            bool(validation["all_direct_formula_cross_checks_pass"]),
        ),
        (
            "Predictive target",
            "Coverage concerns mu_0,beta, not fixed-response confidence coverage or full ATT coverage",
            True,
        ),
        (
            "Monte Carlo reporting",
            "All generated draws, including boundary selections and poor finite-sample cases, are retained",
            bool(validation["all_inference_draws_retained"]),
        ),
        (
            "Paper pages 22--23 calibration DGP",
            "Dimensions 80/160/320/640, eta 0/0.5/0.75/1, sparse 10% shift, scaled shift energy 1/2",
            bool(
                validation["paper_dimension_grid_exact"]
                and validation["paper_calibration_eta_grid_exact"]
                and validation["paper_sparse_shift_rho2_exact"]
            ),
        ),
        (
            "Required centered-rerun location invariance",
            "Every design is rerun after adding 100 to source and target outcomes; penalties, prediction error, and ATT are unchanged",
            bool(
                validation["outcome_shift_all_ood_penalties_invariant"]
                and validation["outcome_shift_all_gcv_penalties_invariant"]
                and validation["outcome_shift_max_prediction_error"] < 1.0e-10
                and validation["outcome_shift_max_att_error"] < 1.0e-10
            ),
        ),
    ]
    audit = pd.DataFrame(audit_rows, columns=["paper_item", "implementation", "verified"])
    audit.to_csv(ARTIFACT_DIR / "assumption_audit.csv", index=False)
    return audit, validation


def write_results_summary(
    config: ExperimentConfig,
    deterministic: dict[str, pd.DataFrame],
    aspect: pd.DataFrame,
    inference: dict[str, pd.DataFrame],
    nuisance: pd.DataFrame,
    exact_verification: pd.DataFrame,
    split: pd.DataFrame,
    validation: dict[str, bool | float | int],
) -> str:
    calibration = inference["calibration"].copy()
    calibration["absolute_coverage_error"] = abs(calibration["coverage"] - 0.95)
    risk_errors = deterministic["uniform_errors"]
    smallest_p = min(config.dimensions)
    largest_p = max(config.dimensions)
    error_by_p = risk_errors.groupby("p")["uniform_absolute_error"].median()
    nuisance_worst = nuisance.loc[nuisance["rmse"].idxmax()]
    exact_worst = exact_verification.loc[
        np.argmax(abs(exact_verification["empirical_to_exact_ratio"] - 1.0))
    ]
    split_best = split.loc[split["predictive_rmse"].idxmin()]
    eta_half_total = (
        risk_errors[
            risk_errors["eta"].eq(0.5)
            & risk_errors["scenario"].eq("Coupled")
            & risk_errors["component"].eq("Total")
        ]
        .groupby("p")["uniform_absolute_error"]
        .median()
    )
    eta_half_audit = pd.read_csv(ARTIFACT_DIR / "eta_half_scaling_audit.csv")
    eta_half_max_shift_scaling_error = float(
        np.max(
            np.abs(
                eta_half_audit["rho2_realized"]
                - eta_half_audit["rho2_requested"]
            )
        )
    )
    eta_half_max_weight_scaling_error = float(
        np.max(
            np.abs(
                eta_half_audit["varrho2_realized"]
                - eta_half_audit["varrho2_requested"]
            )
        )
    )
    lines = [
        "# Paper-aligned experiment: factual result summary",
        "",
        "This file is generated from the saved Monte Carlo draws. No result was manually edited.",
        "",
        "## Headline numerical facts",
        "",
        f"- Median uniform deterministic-equivalent error: {error_by_p.loc[smallest_p]:.4g} at p={smallest_p} and {error_by_p.loc[largest_p]:.4g} at p={largest_p}.",
        f"- Predictive coverage ranged from {calibration['coverage'].min():.3f} to {calibration['coverage'].max():.3f} across the reported p and eta cells.",
        f"- The largest absolute 95% coverage error was {calibration['absolute_coverage_error'].max():.3f}.",
        f"- Mean studentized errors ranged from {calibration['mean_t'].min():.3f} to {calibration['mean_t'].max():.3f}; design-averaged studentized variances ranged from {calibration['variance_t'].min():.3f} to {calibration['variance_t'].max():.3f}.",
        f"- The largest nuisance RMSE was {nuisance_worst['rmse']:.3f} for {nuisance_worst['estimator']} ({nuisance_worst['parameter']}, phi0={nuisance_worst['phi0']:.2f}, p={int(nuisance_worst['p'])}).",
        f"- In the exact-risk verification, the worst mean empirical/exact ratio was {exact_worst['empirical_to_exact_ratio']:.3f} ({exact_worst['component']}, eta={exact_worst['eta']:g}).",
        f"- The smallest split-sensitivity predictive RMSE cell was {split_best['predictive_rmse']:.3f} ({split_best['method']}, {split_best['split']}, phi0={split_best['phi0']:.2f}, eta={split_best['eta']:g}).",
        f"- For eta=1/2 in the coupled design, the median pathwise uniform total-risk error changed from {eta_half_total.loc[smallest_p]:.4g} at p={smallest_p} to {eta_half_total.loc[largest_p]:.4g} at p={largest_p}.",
        "",
        "## Interpretation boundary",
        "",
        "Coverage is conditional random-effects predictive coverage for the counterfactual target mu_0,beta. It is not a confidence interval for a fixed regression surface and not an interval for the full ATT.",
        "",
        f"Core validation status: {validation['all_core_checks_pass']}.",
    ]
    text = "\n".join(lines) + "\n"
    (ARTIFACT_DIR / "RESULTS_SUMMARY.md").write_text(text, encoding="utf-8")
    correction_lines = [
        "# Recheck findings and corrections",
        "",
        "## Eta = 1/2 audit",
        "",
        "The paper scales squared norms: n0^eta ||delta_nu||^2 / p and n0^eta ||C0 gamma||^2. Therefore eta=1/2 implies n0^(-1/2) squared norms and n0^(-1/4) vector norms. The generator implements this distinction.",
        "",
        f"Maximum realized/requested shift-scaling discrepancy: {eta_half_max_shift_scaling_error:.3e}.",
        f"Maximum realized/requested centered-weight-scaling discrepancy: {eta_half_max_weight_scaling_error:.3e}.",
        "",
        "For eta<1, the paper's leading deterministic equivalent omits empirical source/target mean terms. Their scaled finite-sample order is n0^(eta-1), which is n0^(-1/2) at eta=1/2 but n0^(-1) at eta=0. This explains why the intermediate exponent can converge visibly more slowly even when the DGP exponent is correct.",
        "",
        "## Corrected setup errors",
        "",
        "1. The pilot and evaluation folds now partition one fixed total target sample n1=round(p/0.75); they no longer each receive a full n1.",
        "2. Equation (4.8) uses the paper's normalized-trace convention tr(A)=Tr(A)/p. The NumPy implementation therefore divides the ordinary matrix trace by p*n1,E; the direct-matrix audit explicitly checks this convention.",
        "3. Predictive calibration now uses p in {80,160,320,640}, eta in {0,0.5,0.75,1}, a 10%-sparse mean shift, and scaled shift energy 1/2, exactly as stated on paper pages 22--23.",
        "4. Every covariate design is rerun after adding 100 to all outcomes. The variance estimates, OOD/GCV selections, counterfactual prediction error, and ATT estimate are checked for invariance.",
        "5. Appendix D.2 equation (D.3) is checked numerically for every honest-split design, and all spectral implementations are cross-checked against direct matrix solves.",
        "",
        "## Figure-choice audit",
        "",
        "Penalty paths, aspect-ratio scans, and Q-Q maps use continuous lines over numeric horizontal coordinates. Discrete dimension, tuning-method, and split comparisons use boxplots or unconnected point-and-interval displays.",
        "The Experiment 2 deterministic equivalent is independently cross-checked against the closed-form identity-covariance specialization of Theorem 3.",
    ]
    (ARTIFACT_DIR / "RECHECK_FINDINGS.md").write_text(
        "\n".join(correction_lines) + "\n", encoding="utf-8"
    )
    return text


def run_all(config: ExperimentConfig = DEFAULT_CONFIG) -> dict[str, object]:
    configure_style()
    prepare_output(config)
    deterministic = run_deterministic_equivalent_experiment(config)
    aspect = run_aspect_ratio_experiment(config)
    inference = run_risk_tuning_calibration_experiment(config)
    orientation = run_orientation_supplement(config)
    nuisance = run_nuisance_diagnostics(config)
    exact_verification = run_exact_risk_verification(config)
    split = run_split_sensitivity(config)
    audit, validation = write_validation_and_audit(
        config, deterministic, inference, exact_verification
    )
    summary_text = write_results_summary(
        config,
        deterministic,
        aspect,
        inference,
        nuisance,
        exact_verification,
        split,
        validation,
    )
    return {
        "deterministic": deterministic,
        "aspect": aspect,
        "inference": inference,
        "orientation": orientation,
        "nuisance": nuisance,
        "exact_verification": exact_verification,
        "split": split,
        "audit": audit,
        "validation": validation,
        "summary_text": summary_text,
    }


if __name__ == "__main__":
    results = run_all()
    print(results["summary_text"])
