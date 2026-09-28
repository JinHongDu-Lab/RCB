"""The proposed estimator, assembled in one place.

Risk-calibrated balancing (RCB) takes any normalized base weight vector,
moves it along the ridge-augmented path of Section 2, and selects the penalty
by minimizing the feasible plug-in risk of Section 4 on the evaluation fold of
the target sample.  Every ingredient already lives in :mod:`rcb.balancing`,
:mod:`rcb.risk` and :mod:`rcb.nuisance`; this module only sequences them, so a
reader who wants to run RCB on top of some weighting rule -- the ABW balancing
component, ARB's residual-balancing weights, entropy balancing, anything that
sums to one and was built without the control outcomes -- calls one function::

    fit = fit_rcb(X0, y0, X1_evaluation, base_weights, penalties,
                  theta_bounds=((1e-4, 25.0), (1e-4, 10.0)))
    fit.muhat0, fit.selected_lambda, fit.weights

Every setting on which the manuscript's three studies differ is an explicit
argument with no shared default -- the variance-component search box, the
profile grid, outcome standardization, and whether the exact ``lambda =
infinity`` endpoint (no adjustment at all) is on the grid -- because a
default shared across studies would silently move published numbers.

The feasible criterion is the only thing that selects a penalty here.  The
reference rules the studies report alongside it -- source GCV, the two-moment
nuisance variant, a fixed penalty, and the exact-risk oracle that needs the
true shift -- are in :func:`reference_penalty_rules`, kept apart so that the
oracle can never leak into a reported selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .balancing import design_eigendecomposition, ridge_augmented_path
from .benchmarks.gcv import source_gcv_curves
from .risk import (
    estimate_variance_components,
    exact_risk_components,
    feasible_risk_geometry,
    feasible_risk_path,
)

__all__ = ["RCBFit", "fit_rcb", "prepare_geometry", "reference_penalty_rules"]


def prepare_geometry(X0: np.ndarray, *, clip_eigenvalues: bool = True) -> dict:
    """Center the source design and diagonalize its second-moment matrix once.

    Returns ``{"xbar0", "centered", "eigenvalues", "eigenvectors"}``.  Pass the
    result to :func:`fit_rcb` as ``geometry`` when several bases or outcome
    draws share one design, which is the common case in a Monte Carlo sweep.
    """
    xbar0, centered, eigenvalues, eigenvectors = design_eigendecomposition(
        X0, clip=clip_eigenvalues
    )
    return {
        "xbar0": xbar0,
        "centered": centered,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
    }


@dataclass
class RCBFit:
    """One risk-calibrated fit: the selected estimator and the whole path.

    Shapes follow ``y0``.  For a single outcome vector, ``weights`` is
    ``(n0,)``, ``muhat0`` and the selected quantities are scalars, and
    ``risk_path`` is ``(len(penalty_grid),)``.  For an outcome matrix
    ``(n0, draws)`` -- repeated outcome draws on one design -- ``weights`` is
    ``(n0, draws)``, ``muhat0`` is ``(draws,)``, and ``risk_path`` is
    ``(draws, len(penalty_grid))``.  ``weights_path``, ``qhat_path`` and
    ``norm_squared_path`` are outcome-free and never carry a draws axis.
    """

    weights: np.ndarray
    muhat0: float | np.ndarray
    selected_lambda: float | np.ndarray
    selected_index: int | np.ndarray
    penalty_grid: np.ndarray
    risk_path: np.ndarray
    weights_path: np.ndarray
    estimate_path: np.ndarray
    qhat_path: np.ndarray
    norm_squared_path: np.ndarray
    r2: float | np.ndarray
    sigma2: float | np.ndarray
    nuisance_diagnostics: dict
    base_weights: np.ndarray
    base_residual_imbalance: np.ndarray
    geometry: dict = field(repr=False)

    @property
    def lambda_at_boundary(self):
        """Selected the smallest finite penalty (most aggressive adjustment)."""
        return self.selected_index == 0

    @property
    def selected_no_adjustment_endpoint(self):
        """Selected the exact ``lambda = infinity`` endpoint, if it was on the grid."""
        if not np.isinf(self.penalty_grid[-1]):
            return np.zeros_like(self.selected_index, dtype=bool) if np.ndim(self.selected_index) else False
        return self.selected_index == len(self.penalty_grid) - 1

    @property
    def normalization_error(self) -> float:
        """Largest departure of the selected weights from summing to one."""
        return float(np.max(np.abs(np.sum(self.weights, axis=0) - 1.0)))

    @property
    def normalization_error_along_path(self) -> float:
        """The same, over every penalty on the grid."""
        return float(np.max(np.abs(self.weights_path.sum(axis=0) - 1.0)))


def fit_rcb(
    X0: np.ndarray,
    y0: np.ndarray,
    X1_evaluation: np.ndarray,
    base_weights: np.ndarray,
    penalties: np.ndarray,
    *,
    theta_bounds: tuple[tuple[float, float], tuple[float, float]],
    grid_size: int = 81,
    standardize_outcomes: bool = False,
    include_infinite_endpoint: bool = False,
    geometry: dict | None = None,
    variance_components: tuple[float, float] | None = None,
) -> RCBFit:
    """Risk-calibrated ridge augmentation of one base weight vector.

    ``base_weights`` must sum to one and must have been constructed without
    the control outcomes ``y0``; the pilot fold of the target sample is the
    usual source of its target information, and ``X1_evaluation`` must be the
    disjoint evaluation fold, which is the only target data the criterion sees.

    ``penalties`` is the finite grid.  With ``include_infinite_endpoint`` the
    exact ``lambda = infinity`` point -- the base weights unchanged -- is
    appended, so that "no adjustment" is a real option rather than a numerical
    limit; the real-data study uses this and the simulation studies do not.

    ``theta_bounds`` is the search box for ``(r^2, sigma_0^2)``; it has no
    default because the studies disagree on it (see
    :func:`rcb.risk.estimate_variance_components`).  ``grid_size`` and
    ``standardize_outcomes`` are passed through to the same function.

    ``variance_components`` short-circuits that estimation with an already
    estimated ``(r2, sigma2)`` pair, for callers that fit several bases to one
    source sample and outcome vector: the components depend on ``(X0, y0)``
    only, so estimating them once per base would repeat identical work.  The
    diagnostics are then empty, since nothing was estimated here.
    """
    penalties = np.asarray(penalties, dtype=float)
    if geometry is None:
        geometry = prepare_geometry(X0)
    centered = geometry["centered"]
    eigenvalues = geometry["eigenvalues"]
    eigenvectors = geometry["eigenvectors"]
    base_weights = np.asarray(base_weights, dtype=float)
    target_mean = X1_evaluation.mean(axis=0)

    weights_path, delta = ridge_augmented_path(
        X0, centered, eigenvalues, eigenvectors, target_mean, base_weights,
        penalties,
    )
    qhat_path, norm_squared_path = feasible_risk_geometry(
        X1_evaluation, delta, eigenvalues, eigenvectors, weights_path, penalties,
    )
    penalty_grid = penalties
    if include_infinite_endpoint:
        # The same arithmetic as rcb.balancing.prepare_adjustment_paths: the
        # trace of the target covariance is basis-free, so the endpoint's
        # plug-in signal factor needs no eigenbasis of its own.
        p = X0.shape[1]
        target_variance_trace = float(
            np.sum((X1_evaluation - X1_evaluation.mean(axis=0)) ** 2)
            / (len(X1_evaluation) - 1)
        )
        qhat_infinity = max(
            float(delta @ delta) / p
            - target_variance_trace / (p * len(X1_evaluation)),
            0.0,
        )
        weights_path = np.column_stack([weights_path, base_weights])
        qhat_path = np.r_[qhat_path, qhat_infinity]
        norm_squared_path = np.r_[norm_squared_path, base_weights @ base_weights]
        penalty_grid = np.r_[penalties, np.inf]

    if variance_components is None:
        r2, sigma2, diagnostics = estimate_variance_components(
            X0, y0, eigenvalues, eigenvectors,
            theta_bounds=theta_bounds, grid_size=grid_size,
            standardize_outcomes=standardize_outcomes,
        )
    else:
        r2, sigma2 = (np.atleast_1d(np.asarray(c, dtype=float)) for c in variance_components)
        diagnostics = {}
    risk = feasible_risk_path(r2, sigma2, qhat_path, norm_squared_path)
    selected = np.argmin(risk, axis=1)

    single_outcome = np.ndim(y0) == 1
    if single_outcome:
        # A 1-D dot product of the selected column with y0, as the evaluation
        # and real-data studies compute it; the matrix product below can round
        # its last bit differently through BLAS, so the two are kept apart.
        index = int(selected[0])
        weights = weights_path[:, index]
        estimate_path = weights_path.T @ y0
        return RCBFit(
            weights=weights,
            muhat0=float(weights @ y0),
            selected_lambda=float(penalty_grid[index]),
            selected_index=index,
            penalty_grid=penalty_grid,
            risk_path=risk[0],
            weights_path=weights_path,
            estimate_path=estimate_path,
            qhat_path=qhat_path,
            norm_squared_path=norm_squared_path,
            r2=float(r2[0]),
            sigma2=float(sigma2[0]),
            nuisance_diagnostics=diagnostics,
            base_weights=base_weights,
            base_residual_imbalance=delta,
            geometry=geometry,
        )

    # Repeated outcome draws on one design: the estimates along the path are
    # one matrix product, indexed at each draw's selection, as the simulation
    # study computes them.
    draws = np.arange(y0.shape[1])
    estimate_path = weights_path.T @ y0
    return RCBFit(
        weights=weights_path[:, selected],
        muhat0=estimate_path[selected, draws],
        selected_lambda=penalty_grid[selected],
        selected_index=selected,
        penalty_grid=penalty_grid,
        risk_path=risk,
        weights_path=weights_path,
        estimate_path=estimate_path,
        qhat_path=qhat_path,
        norm_squared_path=norm_squared_path,
        r2=r2,
        sigma2=sigma2,
        nuisance_diagnostics=diagnostics,
        base_weights=base_weights,
        base_residual_imbalance=delta,
        geometry=geometry,
    )


def reference_penalty_rules(
    fit: RCBFit,
    y0: np.ndarray,
    *,
    fixed_lambda: float | None = None,
    nu1: np.ndarray | None = None,
    true_r2: float | None = None,
    true_sigma2: float | None = None,
    X0: np.ndarray | None = None,
) -> dict:
    """The comparison rules reported next to the feasible criterion.

    Returns a dict of selected indices (and the curves behind them) for:

    ``two_moment``
        the feasible criterion with the two-moment pilot estimates of
        ``(r^2, sigma_0^2)`` in place of the spectral quasi-score refinement
        (on the standardized scale if the fit standardized outcomes);
    ``gcv``
        ordinary source-distribution GCV over the finite grid, the
        shift-blind baseline;
    ``fixed``
        the grid point nearest ``fixed_lambda``, when given;
    ``oracle``
        the minimizer of the exact conditional risk, which needs the true
        shift ``nu1`` and the true components -- available in simulation only,
        and never allowed to select a reported estimate.

    Index conventions follow ``fit``: scalars for a single outcome vector,
    one entry per draw otherwise.  ``X0`` is needed only for the oracle.
    """
    diagnostics = fit.nuisance_diagnostics
    single = np.ndim(y0) == 1
    grid_finite = fit.penalty_grid[np.isfinite(fit.penalty_grid)]
    rules: dict = {}

    two_moment_risk = feasible_risk_path(
        diagnostics["pilot_r2"], diagnostics["pilot_sigma2"],
        fit.qhat_path, fit.norm_squared_path,
    )
    two_moment_index = np.argmin(two_moment_risk, axis=1)
    rules["two_moment_risk_path"] = two_moment_risk[0] if single else two_moment_risk
    rules["two_moment"] = int(two_moment_index[0]) if single else two_moment_index

    Y = y0[:, None] if single else y0
    gcv = source_gcv_curves(
        fit.geometry["centered"], Y, fit.geometry["eigenvalues"],
        fit.geometry["eigenvectors"], grid_finite,
    )
    gcv_index = np.argmin(gcv, axis=0)
    rules["gcv_curve"] = gcv[:, 0] if single else gcv
    rules["gcv"] = int(gcv_index[0]) if single else gcv_index

    if fixed_lambda is not None:
        rules["fixed"] = int(np.argmin(np.abs(grid_finite - fixed_lambda)))

    if nu1 is not None:
        if X0 is None or true_r2 is None or true_sigma2 is None:
            raise ValueError("The oracle needs X0, nu1, true_r2 and true_sigma2.")
        finite_path = fit.weights_path[:, : len(grid_finite)]
        signal, residual, total = exact_risk_components(
            X0, finite_path, nu1, true_r2, true_sigma2
        )
        rules["exact_signal"], rules["exact_residual"], rules["exact_total"] = (
            signal, residual, total
        )
        rules["oracle"] = int(np.argmin(total))
    return rules
