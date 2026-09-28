"""Generalized cross-validation on the source distribution.

The comparison baseline that ignores covariate shift entirely: it tunes the
ridge outcome model to predict well where the data came from, not where the
estimate is wanted.  Section 5 reports it alongside the target-aware criterion
precisely to show what that omission costs as the shift grows.

The intercept is left unpenalized, matching the ridge model the rest of the
comparison uses.
"""

from __future__ import annotations

import numpy as np

__all__ = ["source_gcv_curves"]


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
