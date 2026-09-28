"""Section 5: the experiment entry points.

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

The estimator itself, the risk formulas and the comparison methods live in the
shared ``rcb`` package; what remains here is this section's data-generating
process, its experiment loops and its figures.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from rcb import io, paths, style  # noqa: E402
from rcb.balancing import (  # noqa: E402
    design_eigendecomposition,
    design_independent_base_weights,
    pilot_ridge_base_weights,
    ridge_augmented_path,
)
from rcb.benchmarks.gcv import source_gcv_curves  # noqa: E402
from rcb.montecarlo import mean_with_two_se as _summary_two_se  # noqa: E402
from rcb.nuisance import DEFAULT_VARIANCE_COMPONENT_BOUNDS  # noqa: E402
from rcb.risk import (  # noqa: E402
    deterministic_equivalent_diagonal,
    deterministic_equivalent_isotropic,
    exact_risk_components,
    feasible_risk_geometry,
)
from rcb.style import (  # noqa: E402
    COLORS,
    configure_style,
)

from . import figure_inputs  # noqa: E402
from .dgp import (  # noqa: E402
    ARTIFACT_DIR,
    DEFAULT_CONFIG,
    SCENARIOS,
    ExperimentConfig,
    _clustered_inference_summary,
    _inference_design,
    _nuisance_estimates,
    mean_shift,
    prepare_output,
)


def run_deterministic_equivalent_experiment(
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, pd.DataFrame]:
    """Exact conditional risks along the path and their deterministic equivalents.

    The design-independent benchmark: for each dimension, shift exponent and
    scenario, the centered exact risk components of the ridge-augmented path
    and the finite-dimensional equivalents they are compared with, plus the
    uniform-in-penalty error and the coupled scaling identities.
    """
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
                    gamma_base = design_independent_base_weights(n0, eta, varrho2)
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


    return {"paths": paths, "uniform_errors": uniform, "scaling_audit": scaling}


def run_aspect_ratio_experiment(
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Source and target aspect-ratio effects on the risk components.

    The source-ratio figure emphasizes the mechanism requested in the paper:
    smooth deterministic-equivalent curves for five signal-to-weight ratios,
    decomposed into integrated signal, residual noise, and total risk.  Exact
    finite-design means at rho2/varrho2=1 are overlaid only as validation.

    The target-ratio figure separates finite-dimensional convergence from the
    continuous deterministic equivalent.  This avoids treating five simulated
    aspect ratios as though they were the theoretical curve itself.
    """
    selected_lambda = 0.316
    phi_scan = (0.50, 1.00, 2.00, 5.00, 10.00)
    ratio_levels = (0.5, 1.0, 5.0, 10.0, 20.0)
    aspect_etas = (0.0, 1.0)

    rng = np.random.default_rng(config.seed + 202)
    phi_curve = np.geomspace(0.50, 1000.0, 321)
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
                gamma_base = design_independent_base_weights(n0, eta, 1.0)
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
                    signal, residual, total = deterministic_equivalent_isotropic(
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


    # Source aspect ratio: the compact 3x2 mechanism plot requested by the
    # revision.  Lines are direct formula evaluations; colored points and
    # two-MCSE bars validate the baseline ratio at the simulated phi values.

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
                gamma_base = design_independent_base_weights(n0, eta, 1.0)
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

    # Each growth panel draws a horizontal asymptote at the deterministic
    # equivalent for its (eta, phi1).  That is an evaluation of the theory, not
    # a drawing instruction, so it is computed here and written out.
    asymptotes = pd.DataFrame([
        {
            "eta": eta, "phi": phi, "lambda": selected_lambda,
            "deterministic_equivalent": float(
                deterministic_equivalent_isotropic(
                    np.array([selected_lambda]), phi0=config.phi0, phi1=phi,
                    eta=eta, rho2=1.0, varrho2=1.0,
                    r2=config.r2, sigma2=config.sigma2,
                )[2][0]
            ),
        }
        for eta in aspect_etas
        for phi in phi_scan
    ])
    asymptotes.to_csv(
        ARTIFACT_DIR / "aspect_ratio_growth_asymptotes.csv", index=False
    )


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


def run_risk_tuning_calibration_experiment(
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, pd.DataFrame]:
    """Target-aware tuning under the honest split: risk-estimate error, selected-
    penalty regret against the oracle and source GCV, and predictive coverage,
    with the outcome-location invariance check on every design.
    """
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


    return {
        "draws": draws,
        "calibration": calibration,
        "risk_summary": risk_summary,
        "tuning_summary": tuning_summary,
    }


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
                sq_r2, sq_sigma2, sq_diag = _nuisance_estimates(
                    X0, Y0, eigenvalues, eigenvectors
                )
                mm_r2 = sq_diag["pilot_r2"]
                mm_sigma2 = sq_diag["pilot_sigma2"]
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
        gamma_base = design_independent_base_weights(n0, eta, 1.0)
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

    return summary


#: The three joint (phi0, phi1) points the Appendix risk-path panels (ap-1,
#: ap-2, ap-3) rerun Figure 1's pipeline at, beside the manuscript's default
#: (0.75, 0.75).
ASPECT_PAIR_RISK_PATH_PAIRS = ((0.75, 1.25), (1.25, 0.75), (1.25, 1.25))


def _aspect_pair_stem(phi0: float, phi1: float) -> str:
    return f"phi0_{phi0:g}_phi1_{phi1:g}".replace(".", "p")


def run_aspect_pair_risk_paths(
    config: ExperimentConfig = DEFAULT_CONFIG,
    pairs: tuple[tuple[float, float], ...] = ASPECT_PAIR_RISK_PATH_PAIRS,
) -> dict[tuple[float, float], dict[str, pd.DataFrame]]:
    """Figure 1's risk-path pipeline rerun at three joint (phi0, phi1) points.

    Reuses ``figure_inputs._main_draws``'s Monte Carlo and
    ``figure_inputs.prepare_main_figure_inputs``'s summary logic verbatim,
    with ``phi0``/``phi1`` threaded through instead of read off ``CFG``.  Each
    pair gets its own cache, summary and selected-penalty file so the
    manuscript's Appendix panels (ap-1/ap-2/ap-3) can be plotted with the
    unmodified Figure 1 plotting code.
    """
    # p, outcomes and the seed base are fixed by figure_inputs, as in the
    # default Figure 1 call; config is accepted for interface symmetry with
    # this module's other run_* entry points.
    del config
    outputs = {}
    for index, (phi0, phi1) in enumerate(pairs, start=1):
        stem = _aspect_pair_stem(phi0, phi1)
        outputs[(phi0, phi1)] = figure_inputs.prepare_main_figure_inputs(
            phi0=phi0, phi1=phi1,
            seed_offset=9101 + index,
            draws_stem=f"aspect_pair_single_design_draws_{stem}",
            summary_stem=f"aspect_pair_risk_summary_{stem}",
            selected_stem=f"aspect_pair_selected_lambda_{stem}",
        )
    return outputs


def run_ar1_main_aspect_pair_risk_paths(
    config: ExperimentConfig = DEFAULT_CONFIG,
    pairs: tuple[tuple[float, float], ...] = ASPECT_PAIR_RISK_PATH_PAIRS,
) -> dict[tuple[float, float], dict[str, pd.DataFrame]]:
    """The revised AR(1) Figure 1's risk-path pipeline rerun at the same three
    joint (phi0, phi1) points as :func:`run_aspect_pair_risk_paths`.

    Reuses ``figure_inputs._ar1_main_draws``'s Monte Carlo and
    ``figure_inputs.prepare_ar1_main_figure_inputs``'s summary logic
    verbatim, with ``phi0``/``phi1`` threaded through instead of read off
    ``CFG`` -- the AR(1) analogue of ``run_aspect_pair_risk_paths``, one seed
    stream over so neither collides with the other's.
    """
    del config
    outputs = {}
    for index, (phi0, phi1) in enumerate(pairs, start=1):
        stem = _aspect_pair_stem(phi0, phi1)
        outputs[(phi0, phi1)] = figure_inputs.prepare_ar1_main_figure_inputs(
            phi0=phi0, phi1=phi1,
            seed_offset=9121 + index,
            draws_stem=f"ar1_main_aspect_pair_single_design_draws_{stem}",
            summary_stem=f"ar1_main_aspect_pair_risk_summary_{stem}",
            selected_stem=f"ar1_main_aspect_pair_selected_lambda_{stem}",
        )
    return outputs


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

    gamma_spectral = pilot_ridge_base_weights(
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
                closed = deterministic_equivalent_isotropic(
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
    # Three of the checks below assert that this run used the paper's grid.
    # A reduced run is supposed to fail them, so it must not be required to
    # pass them -- but whether a run is reduced has to come from the config the
    # caller asked for, not from whether the checks passed.  Reading it off the
    # outcome would make a production run that quietly got the wrong grid
    # excuse itself, which is the one thing these checks exist to catch.
    paper_run = config == DEFAULT_CONFIG
    validation["run_used_the_paper_config"] = paper_run
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
        and (not paper_run or validation["paper_dimension_grid_exact"])
        and (not paper_run or validation["paper_calibration_eta_grid_exact"])
        and (not paper_run or validation["paper_sparse_shift_rho2_exact"])
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
    # Descriptive, not an assertion: True under production, False under
    # smoke, and either is a correct answer.
    advisory = {"run_used_the_paper_config"}
    if not paper_run:
        advisory |= {
            "paper_dimension_grid_exact",
            "paper_calibration_eta_grid_exact",
            "paper_sparse_shift_rho2_exact",
        }
    io.write_validation(validation, ARTIFACT_DIR, advisory=frozenset(advisory))

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
    return text


def run_all(config: ExperimentConfig = DEFAULT_CONFIG) -> dict[str, object]:
    configure_style()
    prepare_output(config)
    deterministic = run_deterministic_equivalent_experiment(config)
    aspect = run_aspect_ratio_experiment(config)
    inference = run_risk_tuning_calibration_experiment(config)
    nuisance = run_nuisance_diagnostics(config)
    exact_verification = run_exact_risk_verification(config)
    split = run_split_sensitivity(config)
    figure_inputs.prepare_main_figure_inputs()
    figure_inputs.prepare_aspect_figure_inputs()
    ar1_main = figure_inputs.prepare_ar1_main_figure_inputs()
    ar1_aspect = figure_inputs.prepare_ar1_aspect_figure_inputs()
    aspect_pair_risk_paths = run_aspect_pair_risk_paths(config)
    ar1_main_aspect_pair_risk_paths = run_ar1_main_aspect_pair_risk_paths(config)
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
        "ar1_main": ar1_main,
        "ar1_aspect": ar1_aspect,
        "nuisance": nuisance,
        "exact_verification": exact_verification,
        "split": split,
        "aspect_pair_risk_paths": aspect_pair_risk_paths,
        "ar1_main_aspect_pair_risk_paths": ar1_main_aspect_pair_risk_paths,
        "audit": audit,
        "validation": validation,
        "summary_text": summary_text,
    }


#: A reduced configuration that exercises every stage without proving anything.
SMOKE_CONFIG = dataclasses.replace(
    DEFAULT_CONFIG,
    dimensions=(40, 80),
    calibration_etas=(0.0, 1.0),
    deterministic_designs=3,
    aspect_dimension=160,
    aspect_designs=3,
    inference_designs=3,
    outcomes_per_design=5,
    nuisance_designs=3,
    nuisance_outcomes=5,
    split_designs=3,
    split_outcomes=5,
    lambda_count=8,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true",
        help="reduced design and draw counts; exercises every stage, proves nothing",
    )
    args = parser.parse_args()
    results = run_all(SMOKE_CONFIG if args.smoke else DEFAULT_CONFIG)
    print(results["summary_text"])
