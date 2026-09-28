"""The pieces of the proposed risk-calibrated ridge balancing method.

:mod:`rcb.estimator` sequences them; this module holds them:

1. **Geometry and path.**  :func:`design_eigendecomposition` diagonalizes the
   centered source second-moment matrix once; :func:`ridge_augmented_path` then
   sweeps a penalty grid, moving an arbitrary affine-normalized base weight
   vector toward the target mean.
2. **Base weights the theory is stated for.**  :func:`design_independent_base_weights`
   are the design-independent weights of the random-matrix results;
   :func:`pilot_ridge_base_weights` are the pilot-fold ridge base of the
   tuning theorem's verification.

Penalty selection -- minimizing the feasible plug-in risk of :mod:`rcb.risk`
over the grid, and nothing else -- lives in :func:`rcb.estimator.fit_rcb`,
together with the exact ``lambda = infinity`` endpoint the real-data study puts
on its grid so that "no adjustment at all" is a real option rather than a
numerical limit.  The honest pilot/evaluation split is the caller's: the base
is built from the pilot fold, the criterion sees only the evaluation fold, and
the folds are never swapped or averaged, since the guarantee is stated for a
single split and does not transfer to cross-fitting.
"""

from __future__ import annotations

import numpy as np

from .risk import feasible_risk_geometry

__all__ = [
    "design_eigendecomposition",
    "design_independent_base_weights",
    "pilot_ridge_base_weights",
    "ridge_augmented_path",
]


# ---------------------------------------------------------------------------
# Design geometry
# ---------------------------------------------------------------------------
def design_eigendecomposition(
    X0: np.ndarray, *, clip: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(xbar0, X0c, eigenvalues, eigenvectors)`` for the source design.

    ``clip`` floors small negative eigenvalues at zero, which ``eigh`` can
    return for a rank-deficient design.  It defaults on; the one caller that
    turns it off is documented where it does, since that is a deliberate
    per-section policy rather than a numerical default.
    """
    xbar0 = X0.mean(axis=0)
    X0c = X0 - xbar0
    S0c = X0c.T @ X0c / len(X0)
    eigenvalues, eigenvectors = np.linalg.eigh(S0c)
    if clip:
        eigenvalues = np.maximum(eigenvalues, 0.0)
    return xbar0, X0c, eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# Base weights
# ---------------------------------------------------------------------------
def _centered_unit_vector(length: int) -> np.ndarray:
    values = np.arange(length, dtype=float) - (length - 1.0) / 2.0
    values /= np.linalg.norm(values)
    if abs(values.sum()) > 1.0e-12:
        raise RuntimeError("The deterministic weight direction is not centered.")
    return values


def design_independent_base_weights(n0: int, eta: float, varrho2: float) -> np.ndarray:
    """Normalized external weights with n0**eta ||C0 gamma||^2 = varrho2."""
    gamma = np.full(n0, 1.0 / n0)
    if varrho2 > 0:
        gamma = gamma + np.sqrt(varrho2 / n0**eta) * _centered_unit_vector(n0)
    return gamma


def pilot_ridge_base_weights(
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
    if abs(gamma.sum() - 1.0) > 1.0e-9:
        raise RuntimeError("Honest ridge base weights are not normalized.")
    return gamma


# ---------------------------------------------------------------------------
# The ridge-augmented path
# ---------------------------------------------------------------------------
def ridge_augmented_path(
    X0: np.ndarray,
    X0c: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    xbar_target: np.ndarray,
    gamma_base: np.ndarray,
    lambdas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep one base weight vector along the penalty grid.

    Returns the path, shaped ``(n0, len(lambdas))``, and the base residual
    imbalance ``delta``.  Affine normalization is asserted rather than repaired:
    a path that does not sum to one is a bug in the geometry, not something to
    renormalize away.
    """
    delta = xbar_target - X0.T @ gamma_base
    projected = eigenvectors.T @ delta
    coefficients = projected[:, None] / (eigenvalues[:, None] + lambdas[None, :])
    directions = eigenvectors @ coefficients
    gamma_path = gamma_base[:, None] + X0c @ directions / len(X0)
    normalization_error = float(np.max(np.abs(gamma_path.sum(axis=0) - 1.0)))
    if normalization_error > 1.0e-9:
        raise RuntimeError(f"Affine normalization failed: {normalization_error}")
    return gamma_path, delta


