"""Weight normalization, standardization, and balance diagnostics.

These are the small shared helpers that every estimator in :mod:`rcb.benchmarks`
and :mod:`rcb.balancing` needs.  Two of them carry a genuine disagreement
between the sections, preserved here as an argument rather than resolved:

``normalize_weights``
    Section 6 repairs non-finite input by zeroing it and falls back to uniform
    weights when nothing is left; Section 7 refuses to normalize it at all.
    Both are defensible -- Section 6 sweeps thousands of Monte Carlo cells where
    one degenerate solve should not abort the run, Section 7 reports a single
    real-data estimate where a silent fallback would be a fabricated number.

``standardize``
    Section 6 treats a standard deviation as degenerate below ``1e-8``,
    Section 7 below ``1e-10``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "effective_sample_size",
    "negative_weight_fraction",
    "normalize_weights",
    "standardize",
    "standardized_balance_problem",
    "weighted_smd",
]


def normalize_weights(raw: np.ndarray, *, on_invalid: str = "raise") -> np.ndarray:
    """Scale weights to sum to one.

    ``on_invalid="raise"`` refuses non-finite or non-positive totals;
    ``on_invalid="repair"`` zeroes non-finite entries and falls back to uniform
    weights when the remaining total is negligible.  See the module docstring
    for why both behaviors are kept.
    """
    if on_invalid == "repair":
        raw = np.asarray(raw, dtype=float)
        raw = np.where(np.isfinite(raw), raw, 0.0)
        total = float(raw.sum())
        if total <= 1e-14:
            return np.full(len(raw), 1.0 / len(raw))
        return raw / total

    if on_invalid != "raise":
        raise ValueError(f"Unknown on_invalid policy: {on_invalid}")
    raw = np.asarray(raw, dtype=float)
    total = float(np.sum(raw))
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Cannot normalize non-positive or non-finite weights.")
    return raw / total


def standardize(X: np.ndarray, *, floor: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Column center and scale, with degenerate columns left unscaled."""
    center = X.mean(axis=0)
    scale = X.std(axis=0, ddof=1)
    scale = np.where(scale > floor, scale, 1.0)
    return center, scale


def standardized_balance_problem(
    X0: np.ndarray, target: np.ndarray, *, floor: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Put design and target mean on the source's standardized scale."""
    center, scale = standardize(X0, floor=floor)
    return (X0 - center) / scale, (target - center) / scale


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, float)
    return float(np.sum(w) ** 2 / np.sum(w**2))


def weighted_smd(
    source_X: np.ndarray, target_X: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Absolute standardized mean differences after weighting the source arm."""
    weighted_mean = weights @ source_X
    target_mean = target_X.mean(axis=0)
    pooled_sd = np.sqrt(
        0.5 * (source_X.var(axis=0, ddof=1) + target_X.var(axis=0, ddof=1))
    )
    pooled_sd = np.where(pooled_sd > 1e-12, pooled_sd, 1.0)
    return np.abs(weighted_mean - target_mean) / pooled_sd


def negative_weight_fraction(weights: np.ndarray) -> float:
    w = np.asarray(weights, float)
    return float(np.mean(w < 0))


