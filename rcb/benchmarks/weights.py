"""Competing balancing-weight families.

Sections 6 and 7 compare the proposed estimator against the same set of base
weight families, and until now each carried its own copy -- Section 7's
docstrings even name their origin ("used in the evaluation notebook", "verbatim
from repo").  The copies had drifted in their numerical settings, so every such
setting is an explicit argument here and each caller passes its own value.  The
defaults are Section 7's, which is the more conservative choice on real data.

Where the two versions differ, the difference is documented at the argument that
carries it.  Nothing has been averaged or split the difference on.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, linprog, minimize
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..balancing import design_eigendecomposition, ridge_augmented_path
from ..diagnostics import normalize_weights, standardize, standardized_balance_problem

__all__ = [
    "l2_balancing_weights_spectral",
    "cbps_weights",
    "entropy_balancing_weights",
    "fit_propensity_model",
    "l2_balancing_weights",
    "predict_propensity",
    "propensity_weight_families",
    "l2_residual_imbalance",
    "sbw_weights",
]


# ---------------------------------------------------------------------------
# Propensity-based families
# ---------------------------------------------------------------------------
def fit_propensity_model(
    source_X: np.ndarray,
    target_X: np.ndarray,
    *,
    seed: int,
) -> object:
    """Fit the regularized source-versus-target propensity model."""
    X = np.vstack([source_X, target_X])
    arm = np.r_[np.zeros(len(source_X)), np.ones(len(target_X))]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=800, random_state=seed),
    )
    return model.fit(X, arm)


def predict_propensity(model: object, X: np.ndarray) -> np.ndarray:
    """Predict target-group propensities without altering the fitted values."""
    return model.predict_proba(X)[:, 1]


def propensity_weight_families(
    source_X: np.ndarray,
    target_X: np.ndarray,
    *,
    seed: int,
    trim: tuple[float, float] = (0.05, 0.95),
    on_invalid: str = "raise",
) -> tuple[dict, dict]:
    """IPW, sample-trimmed IPW, and overlap weights from one classifier.

    The overlap base uses the source propensity itself, so before augmentation
    it does not target the treated-sample estimand.  Returns the three weight
    vectors and a diagnostic record of the fitted propensity range.  Trimmed
    IPW gives zero weight to source observations outside ``trim``; it does not
    winsorize their propensities to the interval endpoints.
    """
    model = fit_propensity_model(source_X, target_X, seed=seed)
    propensity = predict_propensity(model, source_X)
    retained = (propensity >= trim[0]) & (propensity <= trim[1])
    if not retained.any():
        raise RuntimeError("Trimmed IPW retained no source observations")
    # Numerical bounding remains part of the untrimmed IPW implementation, but
    # it does not define the trimming indicator or alter retained observations.
    bounded_propensity = np.clip(propensity, 1e-4, 1 - 1e-4)
    odds = bounded_propensity / (1 - bounded_propensity)
    weights = {
        "IPW": normalize_weights(odds, on_invalid=on_invalid),
        "Trimmed IPW": normalize_weights(
            np.where(retained, odds, 0.0), on_invalid=on_invalid
        ),
        "Overlap weights": normalize_weights(
            bounded_propensity, on_invalid=on_invalid
        ),
    }
    return weights, {
        "min_source_propensity": float(propensity.min()),
        "max_source_propensity": float(propensity.max()),
        "trim_interval": list(trim),
        "trimmed_source_count": int((~retained).sum()),
        "retained_source_count": int(retained.sum()),
        "retained_source_fraction": float(retained.mean()),
    }


def cbps_weights(
    source_X: np.ndarray,
    target_X: np.ndarray,
    *,
    max_nfev: int = 400,
    tol: float = 1e-10,
    moment_tolerance: float | None = 1e-5,
    ridge: float = 1e-6,
    on_invalid: str = "raise",
) -> tuple[np.ndarray, dict]:
    """Covariate-balancing propensity score (Imai and Ratkovic, 2014), just-
    identified ATT form.

    Solves the balancing moment condition ``E[a Z] = E[(1-a) Z exp(Z'theta)]``
    for the log-odds coefficients ``theta``: the source arm, reweighted by its
    odds, matches the target arm on every (standardized) covariate.  This is
    the condition CBPS is named for, and it is not the logistic-MLE score
    equation ``E[Z(a - e)] = 0`` -- the two coincide only when every source
    weight is exactly its own propensity odds, which is what is being solved
    for, not something already true of the score equations.  ``ridge``
    regularizes the just-identified solve, needed because the plain balancing
    system is unstable in overparameterized designs (``p`` close to or above
    ``n0``), where it is otherwise nearly rank deficient.

    ``moment_tolerance`` sets what counts as failure.  Section 7 accepts a
    solve that did not formally converge as long as the largest balance moment
    is below ``1e-5``; Section 6 accepts nothing but formal convergence, which
    is ``moment_tolerance=None``.  The optimizer budget differs too -- Section
    6 allows 500 evaluations at ``1e-9``, Section 7 allows 400 at ``1e-10``.
    The balancing condition needs more evaluations than the logistic score it
    replaced (not the gradient of a convex objective, so no fast quadratic
    convergence near the root), which is why Section 6's budget is the larger
    of the two despite its tighter formal-convergence requirement.
    """
    n_source = len(source_X)
    X = np.vstack([source_X, target_X])
    arm = np.r_[np.zeros(n_source), np.ones(len(target_X))]
    control = arm == 0
    scaler = StandardScaler().fit(X)
    Z = np.column_stack([np.ones(len(X)), scaler.transform(X)])

    # The balancing condition is not the gradient of a convex objective (unlike
    # the logistic score), so it is only solved reliably from a good starting
    # point.  The logistic-MLE fit on the same design is that starting point --
    # standard CBPS practice, and what keeps the least_squares solve from
    # reporting a small-step "success" nowhere near the balancing condition.
    theta0 = LogisticRegression(
        penalty="l2", C=1.0, fit_intercept=False, solver="lbfgs", max_iter=1000,
    ).fit(Z, arm).coef_.ravel()

    def balance(theta):
        log_odds = np.clip(Z @ theta, -30, 30)
        odds = np.exp(log_odds)
        return Z.T @ (arm - (1 - arm) * odds) / len(Z)

    def balance_jacobian(theta):
        log_odds = np.clip(Z[control] @ theta, -30, 30)
        odds = np.exp(log_odds)
        return -(Z[control].T * odds) @ Z[control] / len(Z)

    def residuals(theta):
        return np.concatenate([balance(theta), np.sqrt(ridge) * theta])

    def residuals_jacobian(theta):
        return np.vstack(
            [balance_jacobian(theta), np.sqrt(ridge) * np.eye(len(theta))]
        )

    result = least_squares(
        residuals, theta0, jac=residuals_jacobian,
        max_nfev=max_nfev, xtol=tol, ftol=tol, gtol=tol,
    )
    max_moment = float(np.max(np.abs(balance(result.x))))
    if not result.success and (
        moment_tolerance is None or max_moment > moment_tolerance
    ):
        raise RuntimeError(
            f"CBPS failed: {result.message}; max moment={max_moment:.3g}"
        )
    weights = normalize_weights(
        np.exp(np.clip(Z[:n_source] @ result.x, -20, 20)),
        on_invalid=on_invalid,
    )
    return weights, {
        "success": bool(result.success),
        "iterations": int(result.nfev),
        "max_abs_moment": max_moment,
        "message": str(result.message),
    }


# ---------------------------------------------------------------------------
# Moment-balancing families
# ---------------------------------------------------------------------------
def entropy_balancing_weights(
    source_X: np.ndarray,
    target_mean: np.ndarray,
    *,
    ridge: float = 0.0,
    maxiter: int = 5000,
    ftol: float = 1e-14,
    gtol: float = 1e-10,
    require_success: bool = False,
    scale_floor: float = 1e-10,
) -> tuple[np.ndarray, dict]:
    """Exponential-tilting (entropy balancing) weights, optionally stabilized.

    ``ridge`` is the stabilization the manuscript uses for the regularized
    entropy base; see :func:`entropy_stabilization_penalty` for the value.  The
    optimizer settings differ between sections -- Section 6 stops at 250
    iterations and insists on convergence, Section 7 runs to 5000 and reports
    the outcome instead -- so both are arguments.
    """
    center, scale = standardize(source_X, floor=scale_floor)
    Z = (source_X - center) / scale
    target = (target_mean - center) / scale

    def objective(theta):
        score = Z @ theta
        weights = np.exp(score - logsumexp(score))
        value = logsumexp(score) - np.log(len(score)) - target @ theta
        value += 0.5 * ridge * float(theta @ theta)
        gradient = Z.T @ weights - target + ridge * theta
        return float(value), gradient

    result = minimize(
        fun=lambda th: objective(th)[0], jac=lambda th: objective(th)[1],
        x0=np.zeros(source_X.shape[1]), method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol},
    )
    if require_success and not result.success:
        raise RuntimeError(f"Entropy balancing failed: {result.message}")
    score = Z @ result.x
    weights = np.exp(score - logsumexp(score))
    return weights, {
        "success": bool(result.success),
        "max_abs_std_imbalance": float(np.max(np.abs(Z.T @ weights - target))),
        "ridge": float(ridge),
    }


def entropy_stabilization_penalty(p: int, n_target: int) -> float:
    """The manuscript's stabilization ``sqrt(log(p + 1) / n_target)``."""
    return float(np.sqrt(np.log(p + 1) / n_target))


def sbw_weights(
    source_X: np.ndarray,
    target_mean: np.ndarray,
    *,
    balance_tolerance: float | np.ndarray = 0.5,
    maxiter: int = 2000,
    solver_tolerance: float = 1e-10,
    scale_floor: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """Canonical stable balancing weights with explicit balance constraints.

    Minimize dispersion around uniform weights subject to exact normalization,
    nonnegativity, and coordinatewise standardized imbalance no larger than
    ``balance_tolerance``.  A linear program first finds a feasible point; the
    quadratic objective is then solved by SLSQP without a fallback or implicit
    relaxation of the declared tolerance.
    """
    n_source = len(source_X)
    Z, target = standardized_balance_problem(
        source_X, target_mean, floor=scale_floor
    )
    uniform = np.full(n_source, 1.0 / n_source)
    tolerance = np.broadcast_to(
        np.asarray(balance_tolerance, dtype=float), target.shape
    ).copy()
    if np.any(tolerance < 0) or not np.isfinite(tolerance).all():
        raise ValueError("SBW balance tolerances must be finite and nonnegative")

    A_ub = np.vstack([Z.T, -Z.T])
    b_ub = np.r_[target + tolerance, -target + tolerance]
    feasible = linprog(
        np.zeros(n_source), A_ub=A_ub, b_ub=b_ub,
        A_eq=np.ones((1, n_source)), b_eq=np.ones(1),
        bounds=(0.0, None), method="highs",
    )
    if not feasible.success:
        raise RuntimeError(
            "SBW constraints are infeasible at the declared tolerance: "
            f"{feasible.message}"
        )

    def objective(weights):
        centered = weights - uniform
        return float(centered @ centered)

    def gradient(weights):
        return 2.0 * (weights - uniform)

    def balance_constraints(weights):
        imbalance = Z.T @ weights - target
        return np.r_[tolerance - imbalance, tolerance + imbalance]

    balance_jacobian = np.vstack([-Z.T, Z.T])
    result = minimize(
        objective, feasible.x, jac=gradient, method="SLSQP",
        bounds=[(0.0, None)] * n_source,
        constraints=[
            {
                "type": "eq",
                "fun": lambda weights: np.sum(weights) - 1.0,
                "jac": lambda weights: np.ones(n_source),
            },
            {
                "type": "ineq",
                "fun": balance_constraints,
                "jac": lambda weights: balance_jacobian,
            },
        ],
        options={"maxiter": maxiter, "ftol": solver_tolerance, "disp": False},
    )
    imbalance = Z.T @ result.x - target
    max_violation = float(np.max(np.abs(imbalance) - tolerance))
    normalization_error = float(abs(result.x.sum() - 1.0))
    minimum_weight = float(result.x.min())
    if (
        not result.success
        or max_violation > 5e-8
        or normalization_error > 5e-10
        or minimum_weight < -5e-10
    ):
        raise RuntimeError(
            "SBW optimization failed its constraints: "
            f"{result.message}; max_violation={max_violation:.3g}; "
            f"normalization_error={normalization_error:.3g}; "
            f"minimum_weight={minimum_weight:.3g}"
        )
    weights = result.x
    return weights, {
        "success": bool(result.success),
        "iterations": int(result.nit),
        "objective": objective(weights),
        "balance_tolerance": tolerance,
        "max_abs_standardized_imbalance": float(
            np.max(np.abs(imbalance))
        ),
        "max_balance_constraint_violation": max_violation,
        "normalization_error": normalization_error,
        "minimum_weight": minimum_weight,
    }


# ---------------------------------------------------------------------------
# Ridge-balancing families
# ---------------------------------------------------------------------------
def l2_balancing_weights(Xc: np.ndarray, t: np.ndarray, delta: float) -> np.ndarray:
    """Affinely-normalized l2 (ridge) balancing weights of the source arm.

    w = 1/n0 + (1/n0) Xc M_delta t,  M_delta = (S + delta I)^{-1},
    S = Xc^T Xc / n0,  t = xbar1 - xbar0 (the target shift to balance).

    delta -> 0 gives exact balance (residual imbalance -> 0); delta -> infinity
    gives uniform weights.  These are exactly the paper's augmented weights with
    a uniform base, so the resulting estimate equals the ridge outcome plug-in.
    """
    n0 = Xc.shape[0]
    S = Xc.T @ Xc / n0
    a = np.linalg.solve(S + delta * np.eye(Xc.shape[1]), t)
    return np.full(n0, 1.0 / n0) + (Xc @ a) / n0


def l2_residual_imbalance(Xc: np.ndarray, t: np.ndarray, delta: float) -> np.ndarray:
    """Delta(delta) = xbar1 - X0^T w(delta) = delta (S + delta I)^{-1} t.

    Evaluated by the closed form rather than by forming the weights and
    subtracting.  The two agree in exact arithmetic but not in floating point,
    and the closed form is what the published numbers were computed with.
    """
    n0 = Xc.shape[0]
    S = Xc.T @ Xc / n0
    return delta * np.linalg.solve(S + delta * np.eye(Xc.shape[1]), t)


def l2_balancing_weights_spectral(
    source_X: np.ndarray,
    target_mean: np.ndarray,
    *,
    alpha: float = 1.0,
    clip_eigenvalues: bool = True,
    normalization_tolerance: float = 1e-10,
) -> np.ndarray:
    """Ridge-balancing component of the Bruns-Smith augmented balancing family.

    This is the proposed method's path evaluated at a single fixed penalty from
    a uniform base, which is exactly what makes it the natural benchmark: it
    isolates what risk-calibrated penalty selection buys over a fixed choice.

    ``clip_eigenvalues`` floors the source eigenvalues at zero, which Section 7
    does and Section 6 does not.  It only matters when the design is rank
    deficient enough for ``eigh`` to return small negative values.

    This is :func:`rcb.balancing.ridge_augmented_path` evaluated at the single
    penalty ``alpha`` from a uniform base -- the same ridge solve the proposed
    estimator's path uses, not a second derivation of it.
    """
    n_source = len(source_X)
    _, centered, eigenvalues, eigenvectors = design_eigendecomposition(
        source_X, clip=clip_eigenvalues
    )
    base = np.full(n_source, 1.0 / n_source)
    gamma_path, _ = ridge_augmented_path(
        source_X, centered, eigenvalues, eigenvectors, target_mean, base,
        np.array([alpha]),
    )
    weights = gamma_path[:, 0]
    if abs(weights.sum() - 1.0) > normalization_tolerance:
        raise RuntimeError("Bruns-Smith ridge base is not affine normalized")
    return weights
