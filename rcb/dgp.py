"""Shared source-covariance and shift-direction construction.

Sections 5 and 6 both draw source and target covariates from a common Gaussian
AR(1) family, ``(Sigma0)_{jk} = rho_AR**|j-k|``, and both need shift directions
expressed relative to that covariance's own eigenbasis rather than the
coordinate basis. This module holds the one implementation they share; each
section still chooses its own shift construction on top of it (Section 5's
``low``/``high``/``diffuse`` scenario grid lives in
``experiments/simulation/dgp.py``, Section 6's Mahalanobis-normalized diffuse
shift is built directly from :func:`spectral_shift_direction` and
:func:`mahalanobis_rescale`).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ar1_covariance",
    "symmetric_sqrt",
    "spectral_shift_direction",
    "mahalanobis_rescale",
]


def ar1_covariance(p: int, rho_ar: float) -> tuple[np.ndarray, np.ndarray]:
    """Eigensystem of the AR(1) covariance (Sigma0)_{jk} = rho_ar**|j-k|.

    Returns ``(s, W)`` with ascending eigenvalues ``s`` and orthonormal
    eigenvectors ``W``, matching the manuscript's ``s_1 <= ... <= s_p``
    eigensystem convention (``main.tex``, deterministic-equivalent notation).
    """
    index = np.arange(p)
    sigma0 = rho_ar ** np.abs(index[:, None] - index[None, :])
    eigenvalues, eigenvectors = np.linalg.eigh(sigma0)
    return eigenvalues, eigenvectors


def symmetric_sqrt(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    """Sigma**(1/2) = W diag(sqrt(s)) W^T from an existing eigendecomposition.

    Uses ``.dot`` rather than ``@``: on this machine's Accelerate-backed
    BLAS, ``@`` on this exact shape/value combination raises spurious
    divide-by-zero/overflow/invalid-value ``RuntimeWarning``s despite a
    numerically correct, finite result (checked directly); ``.dot`` does not.
    """
    return eigenvectors.dot(np.sqrt(eigenvalues)[:, None] * eigenvectors.T)


def spectral_shift_direction(eigenvectors: np.ndarray, orientation: str) -> np.ndarray:
    """Unit shift direction relative to a covariance's own eigenbasis.

    Eigenvectors are assumed ascending by eigenvalue, as returned by
    :func:`ar1_covariance`.  "low"/"high" are the smallest/largest-eigenvalue
    eigenvectors; "diffuse" is ``(1/sqrt(p)) sum_j w_j``, unit norm by
    orthonormality, whose squared eigenbasis projection is uniform across
    every index -- i.e. it realizes ``G_{delta,p} = H_{0,p}``.
    """
    p = eigenvectors.shape[1]
    if orientation == "low":
        return eigenvectors[:, 0]
    elif orientation == "high":
        return eigenvectors[:, -1]
    elif orientation == "diffuse":
        return eigenvectors.sum(axis=1) / np.sqrt(p)
    else:
        raise ValueError(f"Unknown spectral orientation: {orientation}")


def mahalanobis_rescale(
    delta: float,
    direction: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Rescale ``direction`` so its Mahalanobis norm under Sigma equals ``delta``.

    Solves ``nu^T Sigma^{-1} nu = delta**2`` for ``nu = c * direction`` with
    ``Sigma = W diag(s) W^T``, giving
    ``c = delta / sqrt(direction^T Sigma^{-1} direction)`` from the
    eigendecomposition directly, with no explicit matrix inverse.

    Used to build Section 6's diffuse shift so that
    ``nu_delta^T Sigma0^{-1} nu_delta = delta**2`` holds exactly under the
    AR(1) covariance, keeping the population effective-sample-size fraction
    ``exp(-delta**2)`` unchanged from the isotropic design.
    """
    projected = eigenvectors.T @ direction
    mahalanobis_sq = float(np.sum(projected**2 / eigenvalues))
    return delta / np.sqrt(mahalanobis_sq) * direction
