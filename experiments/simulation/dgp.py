"""Section 5: the data-generating process and the honest-split design.

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

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from rcb import io, paths, style  # noqa: E402
from rcb.balancing import pilot_ridge_base_weights  # noqa: E402
from rcb.dgp import spectral_shift_direction  # noqa: E402
from rcb.estimator import fit_rcb, prepare_geometry, reference_penalty_rules  # noqa: E402
from rcb.nuisance import DEFAULT_VARIANCE_COMPONENT_BOUNDS  # noqa: E402
from rcb.risk import (  # noqa: E402
    deterministic_equivalent_diagonal,
    estimate_variance_components,
    feasible_risk_path,
)

STUDY = "simulation"


ARTIFACT_DIR = paths.results(STUDY)


FIGURE_DIR = paths.figures(STUDY)


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
    # AR(1) coefficient for the anisotropic source covariance (Sigma0)_{jk} =
    # rho_ar**|j-k|, used by the spectral-orientation experiment (Figure 1)
    # and the AR(1) aspect-ratio experiment (Figure 2).  Condition number
    # (1+rho_ar)/(1-rho_ar) is about 9 at rho_ar=0.5, chosen so the anisotropy
    # is visible without becoming an ill-conditioning demonstration.
    rho_ar: float = 0.5
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


def prepare_output(config: ExperimentConfig = DEFAULT_CONFIG) -> None:
    config_row = asdict(config)
    config_row["etas"] = ";".join(f"{value:g}" for value in config.etas)
    config_row["calibration_etas"] = ";".join(
        f"{value:g}" for value in config.calibration_etas
    )
    config_row["dimensions"] = ";".join(str(value) for value in config.dimensions)
    io.save_table(pd.DataFrame([config_row]), ARTIFACT_DIR, "configuration.csv")
    io.save_json(io.package_versions(), ARTIFACT_DIR, "package_versions.json")


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Write one figure into this section's figure directory."""
    style.save_figure(fig, stem, FIGURE_DIR)


def _nuisance_estimates(X0, Y0, eigenvalues, eigenvectors):
    """This section's nuisance settings: a 41-point refinement grid.

    Returns the spectral quasi-score estimates and the diagnostics, which carry
    the two-moment pilot under ``pilot_r2`` and ``pilot_sigma2``.
    """
    return estimate_variance_components(
        X0, Y0, eigenvalues, eigenvectors,
        theta_bounds=DEFAULT_VARIANCE_COMPONENT_BOUNDS, grid_size=41,
    )


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


def spectral_shift_weights(p: int, orientation: str) -> np.ndarray | None:
    """``G_{delta,p}`` for :func:`spectral_shift_direction`'s three orientations.

    "low"/"high" are one-hot at the smallest/largest-eigenvalue index; the
    diffuse direction's weights are uniform, i.e. ``None``, which is
    :func:`rcb.risk.deterministic_equivalent_diagonal`'s own default.
    """
    if orientation == "low":
        weights = np.zeros(p)
        weights[0] = 1.0
        return weights
    elif orientation == "high":
        weights = np.zeros(p)
        weights[-1] = 1.0
        return weights
    elif orientation == "diffuse":
        return None
    else:
        raise ValueError(f"Unknown spectral orientation: {orientation}")


def spectral_mean_shift(
    p: int, n0: int, eta: float, rho2: float, eigenvectors: np.ndarray, orientation: str,
) -> np.ndarray:
    """Eigenbasis-relative analogue of :func:`mean_shift`.

    Same scaling identity, ``n0**eta ||delta_nu||^2 / p = rho2``, applied to
    :func:`spectral_shift_direction` instead of a coordinate direction.
    """
    if rho2 <= 0:
        return np.zeros(p)
    direction = spectral_shift_direction(eigenvectors, orientation)
    return np.sqrt(rho2 * p / n0**eta) * direction


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
    geometry = prepare_geometry(X0)
    xbar0, X0c = geometry["xbar0"], geometry["centered"]
    eigenvalues, eigenvectors = geometry["eigenvalues"], geometry["eigenvectors"]
    gamma_base = pilot_ridge_base_weights(
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

    beta = rng.normal(scale=np.sqrt(config.r2 / p), size=(p, outcomes))
    epsilon = rng.normal(scale=np.sqrt(config.sigma2), size=(n0, outcomes))
    Y0 = X0 @ beta + epsilon
    target_epsilon = rng.normal(
        scale=np.sqrt(config.sigma2), size=(n1_total, outcomes)
    )
    treatment_effect = 1.0
    Y1 = X1 @ beta + target_epsilon + treatment_effect

    # The estimator on every outcome draw at once, with this section's
    # settings (the shared search box on a 41-point profile grid), and the
    # reference rules -- source GCV and the exact-risk oracle, which needs the
    # true shift and components and evaluates only.
    fit = fit_rcb(
        X0, Y0, X1_evaluation, gamma_base, config.lambdas,
        theta_bounds=DEFAULT_VARIANCE_COMPONENT_BOUNDS, grid_size=41, geometry=geometry,
    )
    rules = reference_penalty_rules(
        fit, Y0, nu1=nu1, true_r2=config.r2, true_sigma2=config.sigma2, X0=X0,
    )
    gamma_path = fit.weights_path
    exact_signal = rules["exact_signal"]
    exact_residual = rules["exact_residual"]
    exact_total = rules["exact_total"]
    # The feasible criterion with the true components in place of estimates:
    # a diagnostic of the risk formula alone, not a selectable rule.
    oracle_nuisance_feasible = feasible_risk_path(
        [config.r2], [config.sigma2], fit.qhat_path, fit.norm_squared_path
    )[0]
    sq_r2, sq_sigma2, sq_diagnostics = fit.r2, fit.sigma2, fit.nuisance_diagnostics
    mom_r2 = sq_diagnostics["pilot_r2"]
    mom_sigma2 = sq_diagnostics["pilot_sigma2"]
    feasible = fit.risk_path
    selected_ood = fit.selected_index
    selected_gcv = rules["gcv"]
    selected_oracle = rules["oracle"]
    outcome_index = np.arange(outcomes)
    selected_risk_hat = feasible[outcome_index, selected_ood]
    selected_exact_ood = exact_total[selected_ood]
    selected_exact_gcv = exact_total[selected_gcv]
    oracle_exact = float(exact_total[selected_oracle])

    estimates_by_lambda = fit.estimate_path
    mu_hat = fit.muhat0
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
    shifted_fit = fit_rcb(
        X0, Y0_shifted, X1_evaluation, gamma_base, config.lambdas,
        theta_bounds=DEFAULT_VARIANCE_COMPONENT_BOUNDS, grid_size=41, geometry=geometry,
    )
    shifted_sq_r2, shifted_sq_sigma2 = shifted_fit.r2, shifted_fit.sigma2
    shifted_selected_ood = shifted_fit.selected_index
    shifted_selected_gcv = reference_penalty_rules(shifted_fit, Y0_shifted)["gcv"]
    shifted_mu_hat = shifted_fit.muhat0
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
