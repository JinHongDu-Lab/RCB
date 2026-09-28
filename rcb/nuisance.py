"""Spectral quasi-score estimation for the two source variance components.

The implementation follows the paper-ready definition in
``work/revision/section_4_1_spectral_quasi_score_insert.tex``:

    theta_hat in argmin_theta L_m(theta),

where the criterion is the Gaussian working likelihood evaluated in the
eigenbasis of the centered source Gram matrix.  The word ``quasi`` is
important: identification uses only

    E(z_i**2 | X_0) = r**2 d_i + sigma_0**2,

not Gaussianity.  A global one-dimensional profile search is used instead
of selecting an arbitrary root of the finite-sample score equations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar


DEFAULT_VARIANCE_COMPONENT_BOUNDS = ((1.0e-4, 25.0), (1.0e-4, 10.0))


def _as_outcome_matrix(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        return y[:, None]
    if y.ndim != 2:
        raise ValueError("Y0 must be a vector or a two-dimensional matrix.")
    return y


def two_moment_estimates(
    x0: np.ndarray,
    y0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Centered Equation (4.1) estimator for one or many outcome columns."""
    y0 = _as_outcome_matrix(y0)
    n0, p = x0.shape
    x0c = x0 - x0.mean(axis=0, keepdims=True)
    y0c = y0 - y0.mean(axis=0, keepdims=True)
    m = n0 - 1

    w0c = x0c.T @ x0c / m
    a1 = float(np.trace(w0c) / p)
    phi0c = p / m
    d0 = float(np.sum(w0c * w0c) / p - phi0c * a1**2)
    t1 = np.sum(y0c**2, axis=0) / m
    xty = x0c.T @ y0c
    t2 = np.sum(xty**2, axis=0) / m**2

    if d0 <= m ** (-0.25):
        rhat2 = np.zeros(y0.shape[1])
    else:
        rhat2 = np.maximum((t2 - phi0c * a1 * t1) / d0, 0.0)
    sigmahat2 = np.maximum(t1 - a1 * rhat2, 1.0e-12)
    return np.asarray(rhat2), np.asarray(sigmahat2)


def _centered_spectral_representation(
    x0: np.ndarray,
    y0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return positive ``d_i``, squared spectral outcomes, null energy, and m."""
    y0 = _as_outcome_matrix(y0)
    n0, p = x0.shape
    x0c = x0 - x0.mean(axis=0, keepdims=True)
    y0c = y0 - y0.mean(axis=0, keepdims=True)
    m = n0 - 1

    left, singular_values, _ = np.linalg.svd(x0c, full_matrices=False)
    if singular_values.size == 0:
        raise ValueError("The centered source design has no singular values.")
    tolerance = max(float(singular_values[0]) * 1.0e-10, 1.0e-12)
    positive = singular_values > tolerance
    d = singular_values[positive] ** 2 / p
    z = left[:, positive].T @ y0c
    z2 = z**2
    total_energy = np.sum(y0c**2, axis=0)
    residual_energy = np.maximum(total_energy - np.sum(z2, axis=0), 0.0)
    return d, z2, residual_energy, m


def _spectral_representation_from_feature_eigen(
    x0: np.ndarray,
    y0: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Reuse an eigendecomposition of ``X0c.T @ X0c / n0``.

    This avoids an additional SVD in experiments that already diagonalize the
    centered feature covariance to construct the ridge path.
    """
    y0 = _as_outcome_matrix(y0)
    n0, p = x0.shape
    x0c = x0 - x0.mean(axis=0, keepdims=True)
    y0c = y0 - y0.mean(axis=0, keepdims=True)
    m = n0 - 1

    maximum = max(float(np.max(eigenvalues)), 1.0)
    positive = np.asarray(eigenvalues) > maximum * 1.0e-10
    positive_values = np.asarray(eigenvalues)[positive]
    positive_vectors = np.asarray(eigenvectors)[:, positive]
    d = (n0 / p) * positive_values

    singular_values = np.sqrt(n0 * positive_values)
    cross = positive_vectors.T @ (x0c.T @ y0c)
    z = cross / singular_values[:, None]
    z2 = z**2
    total_energy = np.sum(y0c**2, axis=0)
    residual_energy = np.maximum(total_energy - np.sum(z2, axis=0), 0.0)
    return d, z2, residual_energy, m


def _profile_geometry(
    rho: float,
    theta_bounds: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float]:
    (r2_low, r2_high), (sigma2_low, sigma2_high) = theta_bounds
    lower = max(sigma2_low, r2_low / rho)
    upper = min(sigma2_high, r2_high / rho)
    if not lower <= upper:
        if lower <= upper * (1.0 + 1.0e-10):
            common = 0.5 * (lower + upper)
            lower = common
            upper = common
        else:
            raise ValueError("rho lies outside the ratio range induced by Theta.")
    return float(lower), float(upper)


def _profile_geometry_grid(
    rho_grid: np.ndarray,
    theta_bounds: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized :func:`_profile_geometry`, evaluated at every grid point.

    Same branches as the scalar version, elementwise: the "nearly equal"
    merge only fires (if at all) at the grid endpoints, where ``rho`` sits
    exactly at ``rho_low``/``rho_high`` and the interval collapses to a point
    up to floating-point roundoff.
    """
    (r2_low, r2_high), (sigma2_low, sigma2_high) = theta_bounds
    lower = np.maximum(sigma2_low, r2_low / rho_grid)
    upper = np.minimum(sigma2_high, r2_high / rho_grid)
    bad = lower > upper
    if np.any(bad):
        mergeable = lower <= upper * (1.0 + 1.0e-10)
        if np.any(bad & ~mergeable):
            raise ValueError("rho lies outside the ratio range induced by Theta.")
        common = 0.5 * (lower + upper)
        lower = np.where(bad, common, lower)
        upper = np.where(bad, common, upper)
    return lower, upper


def _objective_grid_all_draws(
    d: np.ndarray,
    z2: np.ndarray,
    residual_energy: np.ndarray,
    m: int,
    theta_bounds: tuple[tuple[float, float], tuple[float, float]],
    log_grid: np.ndarray,
) -> np.ndarray:
    """The coarse log-rho objective on ``log_grid``, for every draw at once.

    This is :func:`_profile_one`'s inner ``objective_log_rho``, evaluated at
    every grid point for every outcome column simultaneously via matrix
    products instead of the ``grid_size * draws`` sequential scalar calls a
    Python loop over both would make -- the outcome-heavy call sites profile
    hundreds of draws, and this grid pass (not the handful of per-draw local
    refinements after it) is what dominates the cost.  Returns shape
    ``(grid_size, draws)``.
    """
    rho_grid = np.exp(log_grid)
    inflation = 1.0 + rho_grid[:, None] * d[None, :]
    log_det_term = np.sum(np.log1p(rho_grid[:, None] * d[None, :]), axis=1) / m
    weighted = (1.0 / inflation) @ z2
    unconstrained = np.maximum((weighted + residual_energy[None, :]) / m, 1.0e-15)
    scale_low, scale_high = _profile_geometry_grid(rho_grid, theta_bounds)
    constrained = np.clip(unconstrained, scale_low[:, None], scale_high[:, None])
    return np.log(constrained) + log_det_term[:, None] + unconstrained / constrained


def _profile_one(
    d: np.ndarray,
    z2: np.ndarray,
    residual_energy: float,
    m: int,
    theta_bounds: tuple[tuple[float, float], tuple[float, float]],
    pilot: tuple[float, float] | None,
    grid_size: int,
    *,
    log_grid: np.ndarray,
    objective_grid: np.ndarray,
) -> dict[str, float | bool]:
    """Refine one draw's already-computed coarse grid to its profile estimate.

    ``log_grid`` and ``objective_grid`` (this draw's column of
    :func:`_objective_grid_all_draws`'s output) are supplied by the caller,
    which computes them once for every draw at once; only the local-minimum
    refinement below is inherently per-draw.
    """
    (r2_low, r2_high), (sigma2_low, sigma2_high) = theta_bounds
    if not (0 < r2_low < r2_high and 0 < sigma2_low < sigma2_high):
        raise ValueError("Theta bounds must define a positive compact rectangle.")

    rho_low = r2_low / sigma2_high
    rho_high = r2_high / sigma2_low
    log_low, log_high = np.log(rho_low), np.log(rho_high)

    def objective_log_rho(log_rho: float) -> float:
        rho = float(np.exp(np.clip(log_rho, log_low, log_high)))
        inflation = 1.0 + rho * d
        unconstrained_scale = (
            float(np.sum(z2 / inflation)) + float(residual_energy)
        ) / m
        unconstrained_scale = max(unconstrained_scale, 1.0e-15)
        scale_low, scale_high = _profile_geometry(rho, theta_bounds)
        constrained_scale = float(
            np.clip(unconstrained_scale, scale_low, scale_high)
        )
        return float(
            np.log(constrained_scale)
            + np.sum(np.log1p(rho * d)) / m
            + unconstrained_scale / constrained_scale
        )

    candidate_logs = [float(log_grid[0]), float(log_grid[-1])]

    local_indices = np.flatnonzero(
        (objective_grid[1:-1] <= objective_grid[:-2])
        & (objective_grid[1:-1] <= objective_grid[2:])
    ) + 1
    for index in local_indices:
        result = minimize_scalar(
            objective_log_rho,
            bounds=(float(log_grid[index - 1]), float(log_grid[index + 1])),
            method="bounded",
            options={"xatol": 1.0e-10, "maxiter": 250},
        )
        if result.success and np.isfinite(result.fun):
            candidate_logs.append(float(result.x))

    if pilot is not None:
        pilot_rho = max(float(pilot[0]), r2_low) / max(
            float(pilot[1]), sigma2_low
        )
        candidate_logs.append(float(np.log(np.clip(pilot_rho, rho_low, rho_high))))

    candidates = [
        (objective_log_rho(log_rho), float(log_rho))
        for log_rho in candidate_logs
    ]
    objective_value, best_log_rho = min(candidates, key=lambda item: (item[0], item[1]))
    rhohat = float(np.exp(best_log_rho))

    inflation = 1.0 + rhohat * d
    unconstrained_scale = (
        float(np.sum(z2 / inflation)) + float(residual_energy)
    ) / m
    scale_low, scale_high = _profile_geometry(rhohat, theta_bounds)
    sigmahat2 = float(
        np.clip(max(unconstrained_scale, 1.0e-15), scale_low, scale_high)
    )
    rhat2 = float(rhohat * sigmahat2)

    relative_tolerance = 2.0e-4
    boundary = bool(
        rhat2 <= r2_low * (1 + relative_tolerance)
        or rhat2 >= r2_high * (1 - relative_tolerance)
        or sigmahat2 <= sigma2_low * (1 + relative_tolerance)
        or sigmahat2 >= sigma2_high * (1 - relative_tolerance)
    )

    pilot_objective = np.nan
    if pilot is not None:
        pilot_rho = max(float(pilot[0]), r2_low) / max(
            float(pilot[1]), sigma2_low
        )
        pilot_objective = objective_log_rho(
            float(np.log(np.clip(pilot_rho, rho_low, rho_high)))
        )

    return {
        "rhat2": rhat2,
        "sigmahat2": sigmahat2,
        "rhohat": rhohat,
        "objective": float(objective_value),
        "pilot_objective": float(pilot_objective),
        "objective_improvement": float(pilot_objective - objective_value),
        "boundary": boundary,
        "global_grid_minimum_index": int(np.argmin(objective_grid)),
        "local_minima_considered": int(len(local_indices)),
        "success": bool(np.isfinite(objective_value)),
    }


def _spectral_quasi_score_from_representation(
    d: np.ndarray,
    z2: np.ndarray,
    residual_energy: np.ndarray,
    m: int,
    *,
    theta_bounds: tuple[tuple[float, float], tuple[float, float]] = DEFAULT_VARIANCE_COMPONENT_BOUNDS,
    pilot_r2: np.ndarray | None = None,
    pilot_sigma2: np.ndarray | None = None,
    grid_size: int = 81,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Estimate all outcome columns from a precomputed spectral representation."""
    z2 = np.asarray(z2, dtype=float)
    if z2.ndim == 1:
        z2 = z2[:, None]
    residual_energy = np.atleast_1d(np.asarray(residual_energy, dtype=float))
    draws = z2.shape[1]
    if residual_energy.shape != (draws,):
        raise ValueError("Residual energy must have one value per outcome column.")

    if pilot_r2 is not None:
        pilot_r2 = np.atleast_1d(np.asarray(pilot_r2, dtype=float))
        pilot_sigma2 = np.atleast_1d(np.asarray(pilot_sigma2, dtype=float))
        if pilot_r2.shape != (draws,) or pilot_sigma2.shape != (draws,):
            raise ValueError("Pilot arrays must have one value per outcome column.")

    (r2_low, r2_high), (sigma2_low, sigma2_high) = theta_bounds
    log_grid = np.linspace(
        np.log(r2_low / sigma2_high), np.log(r2_high / sigma2_low), int(grid_size)
    )
    objective_grid_all = _objective_grid_all_draws(
        d, z2, residual_energy, m, theta_bounds, log_grid
    )

    results: list[dict[str, Any]] = []
    for draw in range(draws):
        pilot = None
        if pilot_r2 is not None:
            pilot = (float(pilot_r2[draw]), float(pilot_sigma2[draw]))
        results.append(
            _profile_one(
                d,
                z2[:, draw],
                float(residual_energy[draw]),
                m,
                theta_bounds,
                pilot,
                grid_size,
                log_grid=log_grid,
                objective_grid=objective_grid_all[:, draw],
            )
        )

    rhat2 = np.array([row["rhat2"] for row in results], dtype=float)
    sigmahat2 = np.array([row["sigmahat2"] for row in results], dtype=float)
    diagnostics = {
        key: np.array([row[key] for row in results])
        for key in [
            "rhohat",
            "objective",
            "pilot_objective",
            "objective_improvement",
            "boundary",
            "global_grid_minimum_index",
            "local_minima_considered",
            "success",
        ]
    }
    return rhat2, sigmahat2, diagnostics


def _spectral_quasi_score_over_draws(
    x0: np.ndarray,
    y0: np.ndarray,
    *,
    theta_bounds: tuple[tuple[float, float], tuple[float, float]] = DEFAULT_VARIANCE_COMPONENT_BOUNDS,
    pilot_r2: np.ndarray | None = None,
    pilot_sigma2: np.ndarray | None = None,
    grid_size: int = 81,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    d, z2, residual_energy, m = _centered_spectral_representation(x0, y0)
    return _spectral_quasi_score_from_representation(
        d,
        z2,
        residual_energy,
        m,
        theta_bounds=theta_bounds,
        pilot_r2=pilot_r2,
        pilot_sigma2=pilot_sigma2,
        grid_size=grid_size,
    )


def _gaussian_covariance_from_representation(
    d: np.ndarray,
    rhat2: np.ndarray,
    sigmahat2: np.ndarray,
    m: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plug-in covariance for the spectral estimator under Gaussianity.

    The theorem gives

        sqrt(m) (theta_hat - theta_0) -> N(0, 2 J^{-1}),

    where

        J = m^{-1} sum_i q(d_i) q(d_i)' / v_i(theta_0)^2,
        q(d) = (d, 1)'.

    This function returns the covariance of ``theta_hat`` itself,
    ``2 J_hat^{-1} / m``.  Zero eigenvalues omitted from ``d`` are included
    analytically in the intercept-intercept entry of ``J_hat``.

    For a non-Gaussian DGP, ``2 J_hat`` must be replaced by a consistent
    estimator of the fourth-moment sandwich middle matrix.  The Gaussian
    formula should not be relabelled as distribution-free.
    """
    d = np.asarray(d, dtype=float).reshape(-1)
    rhat2 = np.atleast_1d(np.asarray(rhat2, dtype=float))
    sigmahat2 = np.atleast_1d(np.asarray(sigmahat2, dtype=float))
    if rhat2.shape != sigmahat2.shape:
        raise ValueError("rhat2 and sigmahat2 must have matching shapes.")
    if np.any(rhat2 <= 0) or np.any(sigmahat2 <= 0):
        raise ValueError("Log-scale covariance requires positive estimates.")
    if not 0 <= len(d) <= m:
        raise ValueError("The number of positive eigenvalues cannot exceed m.")

    variance = d[:, None] * rhat2[None, :] + sigmahat2[None, :]
    inverse_variance_squared = 1.0 / variance**2
    j11 = np.sum(d[:, None] ** 2 * inverse_variance_squared, axis=0) / m
    j12 = np.sum(d[:, None] * inverse_variance_squared, axis=0) / m
    null_multiplicity = m - len(d)
    j22 = (
        np.sum(inverse_variance_squared, axis=0)
        + null_multiplicity / sigmahat2**2
    ) / m

    determinant = j11 * j22 - j12**2
    if np.any(~np.isfinite(determinant)) or np.any(determinant <= 0):
        raise np.linalg.LinAlgError(
            "The plug-in sensitivity matrix is not positive definite."
        )

    draws = len(rhat2)
    sensitivity = np.empty((draws, 2, 2), dtype=float)
    sensitivity[:, 0, 0] = j11
    sensitivity[:, 0, 1] = j12
    sensitivity[:, 1, 0] = j12
    sensitivity[:, 1, 1] = j22

    inverse_sensitivity = np.empty_like(sensitivity)
    inverse_sensitivity[:, 0, 0] = j22 / determinant
    inverse_sensitivity[:, 0, 1] = -j12 / determinant
    inverse_sensitivity[:, 1, 0] = -j12 / determinant
    inverse_sensitivity[:, 1, 1] = j11 / determinant
    covariance = (2.0 / m) * inverse_sensitivity
    condition_number = np.linalg.cond(sensitivity)
    return covariance, sensitivity, condition_number


def _kurtosis_sandwich_covariance_from_svd(
    left_singular_vectors: np.ndarray,
    singular_values: np.ndarray,
    right_singular_vectors_transpose: np.ndarray,
    rhat2: np.ndarray,
    sigmahat2: np.ndarray,
    m: int,
    *,
    beta_excess_kurtosis: float,
    error_excess_kurtosis: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Plug-in sandwich covariance when the two excess kurtoses are known.

    This implements the finite-sample fourth-moment formula in the paper for
    iid standardized random-coefficient coordinates and iid standardized
    source errors.  It is useful as a simulation benchmark: the two supplied
    excess kurtoses are properties of the DGP, not quantities inferred by this
    function from one observed outcome vector.

    The inputs are the positive part of the thin SVD of the centered design,

        X0c = left_singular_vectors @ diag(singular_values)
              @ right_singular_vectors_transpose.

    The returned covariance is for ``theta_hat`` itself:

        J_hat^{-1} Omega_hat J_hat^{-1} / m.

    Setting both excess kurtoses to zero recovers the Gaussian-information
    covariance ``2 J_hat^{-1} / m``.
    """
    left = np.asarray(left_singular_vectors, dtype=float)
    singular_values = np.asarray(singular_values, dtype=float).reshape(-1)
    right_t = np.asarray(right_singular_vectors_transpose, dtype=float)
    rhat2 = np.atleast_1d(np.asarray(rhat2, dtype=float))
    sigmahat2 = np.atleast_1d(np.asarray(sigmahat2, dtype=float))

    if left.ndim != 2 or right_t.ndim != 2:
        raise ValueError("The SVD factors must be two-dimensional arrays.")
    rank = len(singular_values)
    if left.shape[1] != rank or right_t.shape[0] != rank:
        raise ValueError("The SVD factors have incompatible shapes.")
    if rhat2.shape != sigmahat2.shape:
        raise ValueError("rhat2 and sigmahat2 must have matching shapes.")
    if np.any(rhat2 <= 0) or np.any(sigmahat2 <= 0):
        raise ValueError("Sandwich covariance requires positive estimates.")
    n0, p = left.shape[0], right_t.shape[1]
    if m != n0 - 1 or rank > m:
        raise ValueError("The centered SVD rank is inconsistent with m.")

    d = singular_values**2 / p
    left_squared = left**2
    centered_null_diagonal = np.maximum(
        1.0 - 1.0 / n0 - np.sum(left_squared, axis=1),
        0.0,
    )
    projected_design_squared = (
        singular_values[:, None] * right_t
    ) ** 2

    draws = len(rhat2)
    covariance = np.empty((draws, 2, 2), dtype=float)
    sensitivity = np.empty_like(covariance)
    middle = np.empty_like(covariance)
    condition_number = np.empty(draws, dtype=float)

    for draw in range(draws):
        a = float(rhat2[draw])
        b = float(sigmahat2[draw])
        variance = a * d + b
        inverse_variance_squared = 1.0 / variance**2
        weights = np.vstack(
            [d * inverse_variance_squared, inverse_variance_squared]
        )

        j11 = float(np.sum(d**2 * inverse_variance_squared) / m)
        j12 = float(np.sum(d * inverse_variance_squared) / m)
        j22 = float(
            (
                np.sum(inverse_variance_squared)
                + (m - rank) / b**2
            )
            / m
        )
        jhat = np.array([[j11, j12], [j12, j22]], dtype=float)
        inverse_jhat = np.linalg.inv(jhat)

        beta_diagonal = (a / p) * (
            weights @ projected_design_squared
        )
        error_diagonal = b * (weights @ left_squared.T)
        error_diagonal[1] += centered_null_diagonal / b

        omega = (
            2.0 * jhat
            + float(beta_excess_kurtosis)
            * (beta_diagonal @ beta_diagonal.T)
            / m
            + float(error_excess_kurtosis)
            * (error_diagonal @ error_diagonal.T)
            / m
        )
        covariance[draw] = inverse_jhat @ omega @ inverse_jhat / m
        sensitivity[draw] = jhat
        middle[draw] = omega
        condition_number[draw] = np.linalg.cond(jhat)

    return covariance, sensitivity, middle, condition_number


def spectral_quasi_score_estimates(
    x0: np.ndarray,
    y0: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    *,
    theta_bounds: tuple[tuple[float, float], tuple[float, float]] = DEFAULT_VARIANCE_COMPONENT_BOUNDS,
    pilot_r2: np.ndarray | None = None,
    pilot_sigma2: np.ndarray | None = None,
    grid_size: int = 81,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    d, z2, residual_energy, m = _spectral_representation_from_feature_eigen(
        x0, y0, eigenvalues, eigenvectors
    )
    return _spectral_quasi_score_from_representation(
        d,
        z2,
        residual_energy,
        m,
        theta_bounds=theta_bounds,
        pilot_r2=pilot_r2,
        pilot_sigma2=pilot_sigma2,
        grid_size=grid_size,
    )
