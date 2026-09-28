"""Approximate residual balancing, the two-stage comparator.

ARB pairs capped balancing weights with an elastic-net outcome model and
corrects the fitted target mean by the weighted source residual.  Sections 6 and
7 carried the same implementation twice, differing only in whether the target
was passed as a matrix or as its mean, and in the seed.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler

from ..diagnostics import standardize

__all__ = ["arb_elastic_net_fit", "arb_residual_weights", "arb_estimate"]


def arb_residual_weights(
    source_X: np.ndarray,
    target_mean: np.ndarray,
    *,
    zeta: float = 0.5,
    maxiter: int = 400,
    ftol: float = 1e-10,
    scale_floor: float = 1e-8,
    on_invalid: str = "raise",
) -> tuple[np.ndarray, dict]:
    """Procedure-1 approximate residual-balancing weights.

    Minimizes ``(1 - zeta) ||gamma||^2 + zeta * slack^2`` subject to the
    standardized balance constraints holding within ``slack``, the weights
    summing to one, and each weight capped at ``n0**(-2/3)``.  The cap is what
    keeps the residual correction from concentrating on a handful of units.
    """
    n_source, p = source_X.shape
    center, scale = standardize(source_X, floor=scale_floor)
    Z = (source_X - center) / scale
    target = (target_mean - center) / scale
    cap = n_source ** (-2 / 3)
    uniform = np.full(n_source, 1.0 / n_source)
    initial_imbalance = target - Z.T @ uniform
    x0 = np.r_[uniform, np.max(np.abs(initial_imbalance)) + 1e-8]

    def objective(x):
        gamma, slack = x[:-1], x[-1]
        return (1 - zeta) * (gamma @ gamma) + zeta * slack**2

    def objective_gradient(x):
        gamma, slack = x[:-1], x[-1]
        return np.r_[2 * (1 - zeta) * gamma, 2 * zeta * slack]

    def balance_constraints(x):
        gamma, slack = x[:-1], x[-1]
        imbalance = target - Z.T @ gamma
        return np.r_[slack - imbalance, slack + imbalance]

    balance_jacobian = np.zeros((2 * p, n_source + 1))
    balance_jacobian[:p, :n_source] = Z.T
    balance_jacobian[p:, :n_source] = -Z.T
    balance_jacobian[:, -1] = 1.0
    result = minimize(
        objective, x0, jac=objective_gradient, method="SLSQP",
        bounds=[(0.0, cap)] * n_source + [(0.0, None)],
        constraints=[
            {
                "type": "eq",
                "fun": lambda x: np.sum(x[:-1]) - 1.0,
                "jac": lambda x: np.r_[np.ones(n_source), 0.0],
            },
            {
                "type": "ineq",
                "fun": balance_constraints,
                "jac": lambda x: balance_jacobian,
            },
        ],
        options={"maxiter": maxiter, "ftol": ftol, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"ARB weight program failed: {result.message}")
    weights = result.x[:-1]
    imbalance = target - Z.T @ weights
    normalization_error = float(abs(weights.sum() - 1.0))
    maximum_balance_violation = float(
        np.max(np.abs(imbalance)) - result.x[-1]
    )
    cap_violation = float(weights.max() - cap)
    minimum_weight = float(weights.min())
    if (
        normalization_error > 5e-8
        or maximum_balance_violation > 5e-8
        or cap_violation > 5e-8
        or minimum_weight < -5e-8
    ):
        raise RuntimeError(
            "ARB solution failed its declared constraints: "
            f"normalization_error={normalization_error:.3g}; "
            f"balance_violation={maximum_balance_violation:.3g}; "
            f"cap_violation={cap_violation:.3g}; "
            f"minimum_weight={minimum_weight:.3g}"
        )
    return weights, {
        "success": bool(result.success),
        "iterations": int(result.nit),
        "slack": float(result.x[-1]),
        "cap": float(cap),
        "normalization_error": normalization_error,
        "max_balance_constraint_violation": maximum_balance_violation,
        "cap_violation": cap_violation,
        "minimum_weight": minimum_weight,
    }


def arb_elastic_net_fit(
    source_X: np.ndarray,
    source_y: np.ndarray,
    *,
    seed: int,
    l1_ratio: float = 0.9,
    cv: int = 5,
    n_alphas: int = 60,
    max_iter: int = 10000,
):
    """Elastic net outcome regression with a glmnet-style one-standard-error rule.

    Returns ``(scaler, model, alpha_1se)``.  The one-SE rule takes the largest
    penalty whose mean CV loss is within one standard error of the minimum,
    which is the sparser and more stable choice the ARB proposal calls for.
    """
    scaler = StandardScaler().fit(source_X)
    Z = scaler.transform(source_X)
    cv_model = ElasticNetCV(
        l1_ratio=l1_ratio, cv=cv, n_alphas=n_alphas, max_iter=max_iter,
        random_state=seed, n_jobs=1,
    ).fit(Z, source_y)
    mean_loss = cv_model.mse_path_.mean(axis=1)
    se_loss = cv_model.mse_path_.std(axis=1, ddof=1) / np.sqrt(
        cv_model.mse_path_.shape[1]
    )
    best = int(np.argmin(mean_loss))
    eligible = np.flatnonzero(mean_loss <= mean_loss[best] + se_loss[best])
    alpha_1se = float(np.max(cv_model.alphas_[eligible]))
    model = ElasticNet(
        alpha=alpha_1se, l1_ratio=l1_ratio, max_iter=max_iter
    ).fit(Z, source_y)
    return scaler, model, alpha_1se


def arb_estimate(
    source_X: np.ndarray,
    target_X: np.ndarray,
    source_y: np.ndarray,
    *,
    seed: int,
    residual_weights: np.ndarray | None = None,
    **outcome_options,
) -> dict:
    """Complete two-stage ARB estimate of the source-arm target mean."""
    if residual_weights is None:
        residual_weights, weight_diagnostics = arb_residual_weights(
            source_X, target_X.mean(axis=0)
        )
    else:
        weight_diagnostics = {"precomputed_residual_weights": True}

    scaler, model, alpha_1se = arb_elastic_net_fit(
        source_X, source_y, seed=seed, **outcome_options
    )
    fitted_source = model.predict(scaler.transform(source_X))
    fitted_target = float(model.predict(scaler.transform(target_X)).mean())
    residual_correction = float(residual_weights @ (source_y - fitted_source))
    return {
        "muhat0": fitted_target + residual_correction,
        "weights": residual_weights,
        "elastic_net_alpha_1se": alpha_1se,
        "fitted_target_component": fitted_target,
        "residual_correction": residual_correction,
        "weight_diagnostics": weight_diagnostics,
    }
