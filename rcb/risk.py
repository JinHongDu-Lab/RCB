"""Risk formulas for the balancing-weight path.

Three quantities are kept strictly distinct here, as they are in the manuscript
and in the section notebooks:

*theoretical*
    the finite-dimensional deterministic equivalent --
    :func:`deterministic_equivalent_diagonal` and its identity-covariance
    special case :func:`deterministic_equivalent_isotropic`;

*exact*
    the exact conditional risk for a fixed source design, given the true
    variance components -- :func:`exact_risk_components`;

*risk estimate*
    the feasible plug-in risk under repeated outcomes -- the outcome-free
    factors from :func:`feasible_risk_geometry` combined with estimated
    components by :func:`feasible_risk_path`.

Only the third is ever allowed to select a penalty.  The first is what the
theory predicts, the second is what evaluation compares against; using either
to tune would make the comparison circular.

:func:`fixed_weight_risk_proxy` is a fourth, coarser thing again: it scores one
already-chosen weight vector so that estimators from different families can be
put on a common axis.  It has no penalty path and is descriptive only.
"""

from __future__ import annotations

import numpy as np

from .nuisance import (
    DEFAULT_VARIANCE_COMPONENT_BOUNDS,
    two_moment_estimates,
    spectral_quasi_score_estimates,
)

__all__ = [
    "deterministic_equivalent_diagonal",
    "deterministic_equivalent_isotropic",
    "estimate_variance_components",
    "exact_risk_components",
    "feasible_risk_geometry",
    "feasible_risk_path",
    "fixed_weight_risk_proxy",
]


# ---------------------------------------------------------------------------
# Exact conditional risk
# ---------------------------------------------------------------------------
def exact_risk_components(
    X0: np.ndarray,
    gamma_path: np.ndarray,
    nu1: np.ndarray,
    r2: float,
    sigma2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact conditional risk of a weight path, given the true components.

    Returns ``(signal, residual, total)``, the bias and variance terms
    ``B_n(gamma) = (r2/p) ||X0' gamma - nu1||^2`` and
    ``V_n(gamma) = sigma2 ||gamma||^2``, evaluated along the path.
    """
    imbalance = X0.T @ gamma_path - nu1[:, None]
    signal = (r2 / X0.shape[1]) * np.sum(imbalance**2, axis=0)
    residual = sigma2 * np.sum(gamma_path**2, axis=0)
    return signal, residual, signal + residual


# ---------------------------------------------------------------------------
# Deterministic equivalents
# ---------------------------------------------------------------------------
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

    lam = np.asarray(lambdas, dtype=float)

    # The fixed point is solved for the whole penalty grid at once.  The
    # bisection runs a fixed 100 iterations with no early exit, so every
    # penalty takes the same path and the loop was pure overhead.
    lower = np.full_like(lam, 1.0e-12)
    upper = 1.0 / lam
    for _ in range(100):
        v = 0.5 * (lower + upper)
        equation = 1.0 / v - lam - phi0 * np.mean(
            s / (1.0 + v[:, None] * s), axis=1
        )
        positive = equation > 0
        lower = np.where(positive, v, lower)
        upper = np.where(positive, upper, v)
    v = 0.5 * (lower + upper)

    denominator = 1.0 + v[:, None] * s
    vev = 1.0 / (v**-2 - phi0 * np.mean(s**2 / denominator**2, axis=1))

    def veb(a: np.ndarray) -> np.ndarray:
        return phi0 * vev * np.mean(a * s / denominator**2, axis=1)

    def k1(a: np.ndarray) -> np.ndarray:
        return np.mean(a / denominator, axis=1)

    def k2(a: np.ndarray) -> np.ndarray:
        return np.mean((veb(a)[:, None] * s + a) / denominator**2, axis=1)

    vb_identity = veb(np.ones_like(s))
    shift_signal = np.sum(
        g * (vb_identity[:, None] * s + 1.0) / denominator**2, axis=1
    )
    shift_residual = np.sum(
        g * (v[:, None] - vb_identity[:, None]) * s / denominator**2, axis=1
    )

    signal = (
        lam**2 * r2 * varrho2 / phi0 * (v - lam * vev)
        + r2 * rho2 * shift_signal
    )
    residual = (
        sigma2 * varrho2 * lam**2 * vev
        + sigma2 * phi0 * rho2 / lam * shift_residual
    )
    if eta == 1.0:
        target_kernel = np.mean(t) - 2.0 * k1(t) + k2(t)
        signal = signal + r2 * phi1 / phi0 * target_kernel + r2 * k2(s)
        residual = residual + sigma2 * phi1 / lam * (k1(t) - k2(t))
        residual = residual + sigma2 * (
            1.0 + phi0 / lam * (k1(s) - k2(s))
        )
    return signal, residual, signal + residual


def deterministic_equivalent_isotropic(
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
    lam = np.asarray(lambdas, dtype=float)
    linear = lam + phi0 - 1.0
    v = (-linear + np.sqrt(linear**2 + 4.0 * lam)) / (2.0 * lam)
    vev = 1.0 / (v**-2 - phi0 / (1.0 + v) ** 2)
    veb = phi0 * vev / (1.0 + v) ** 2
    k1 = 1.0 / (1.0 + v)
    k2 = (veb + 1.0) / (1.0 + v) ** 2
    shift_residual = (v - veb) / (1.0 + v) ** 2

    signal = (
        lam**2 * r2 * varrho2 / phi0 * (v - lam * vev)
        + r2 * rho2 * k2
    )
    residual = (
        sigma2 * varrho2 * lam**2 * vev
        + sigma2 * phi0 * rho2 / lam * shift_residual
    )
    if eta == 1.0:
        signal = signal + r2 * phi1 / phi0 * (1.0 - 2.0 * k1 + k2)
        signal = signal + r2 * k2
        residual = residual + sigma2 * phi1 / lam * (k1 - k2)
        residual = residual + sigma2 * (1.0 + phi0 / lam * (k1 - k2))
    return signal, residual, signal + residual


# ---------------------------------------------------------------------------
# Feasible plug-in risk
# ---------------------------------------------------------------------------
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


def feasible_risk_path(
    r2: np.ndarray,
    sigma2: np.ndarray,
    signal_factor: np.ndarray,
    weight_norm: np.ndarray,
) -> np.ndarray:
    """Combine estimated components with the outcome-free geometry.

    With ``r2`` and ``sigma2`` given per outcome draw, the result is one risk
    curve per draw, shaped ``(draws, penalties)``.  This is the only criterion
    permitted to select a penalty.
    """
    return (
        np.asarray(r2)[:, None] * np.asarray(signal_factor)[None, :]
        + np.asarray(sigma2)[:, None] * np.asarray(weight_norm)[None, :]
    )


def fixed_weight_risk_proxy(X0, Xt, w, r2, sigma2) -> float:
    """Score one fixed weight vector, for cross-family comparison only.

    Unlike :func:`feasible_risk_path` this has no penalty path and no target
    covariance correction; it exists so that weights produced by unrelated
    estimators can be placed on a common axis, and it never tunes anything.
    """
    imbalance = Xt.mean(axis=0) - X0.T @ w
    return float(r2 / X0.shape[1] * (imbalance @ imbalance) + sigma2 * (w @ w))


# ---------------------------------------------------------------------------
# Variance components feeding the plug-in risk
# ---------------------------------------------------------------------------
def estimate_variance_components(
    X0: np.ndarray,
    Y0: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    *,
    theta_bounds: tuple = DEFAULT_VARIANCE_COMPONENT_BOUNDS,
    grid_size: int = 81,
    standardize_outcomes: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Estimate ``(r^2, sigma_0^2)`` by spectral quasi-score, warm-started.

    The two-moment estimator of Equation (4.1) supplies the pilot value that
    starts the profile search, which is how every call site in the manuscript
    uses it.

    ``theta_bounds`` and ``grid_size`` are arguments rather than constants
    because the sections genuinely disagree: Sections 5 and 6 search
    ``((1e-4, 25), (1e-4, 10))``, Section 7 searches ``((1e-6, 25), (1e-6, 10))``,
    and Section 5 refines on a 41-point grid where the others take the default
    81.  Folding those into one default would silently move published numbers.

    ``standardize_outcomes`` centers and scales ``Y0`` before estimation and
    rescales the components afterwards, as the real-data section does, where
    the outcome is measured in dollars and the raw scale would sit far outside
    the search box.
    """
    if standardize_outcomes:
        y_scale = float(np.std(Y0, ddof=1))
        values = (Y0 - Y0.mean()) / max(y_scale, 1e-12)
    else:
        values = Y0

    pilot_r2, pilot_sigma2 = two_moment_estimates(X0, values)
    r2, sigma2, diagnostics = spectral_quasi_score_estimates(
        X0,
        values,
        eigenvalues,
        eigenvectors,
        theta_bounds=theta_bounds,
        pilot_r2=pilot_r2,
        pilot_sigma2=pilot_sigma2,
        grid_size=grid_size,
    )
    if standardize_outcomes:
        r2 = r2 * y_scale**2
        sigma2 = sigma2 * y_scale**2
    # The pilot travels with the diagnostics so callers that report both the
    # two-moment and the refined estimate need not compute the pilot twice.
    # Under ``standardize_outcomes`` it is on the standardized scale.
    diagnostics = {
        **diagnostics, "pilot_r2": pilot_r2, "pilot_sigma2": pilot_sigma2
    }
    return r2, sigma2, diagnostics
