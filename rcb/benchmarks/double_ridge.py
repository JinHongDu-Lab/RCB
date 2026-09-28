"""Double ridge and the outcome-model cross-validation rules it is tuned by.

The Bruns-Smith augmented l2 balancing estimator -- "double ridge" -- is the
manuscript's principal comparator, and the point of comparison is the tuning
rule rather than the estimator: the same weight path is walked, and only the
selected point differs.  Three rules appear here, cross-validating the outcome
model, the held-out imbalance, and the Riesz loss respectively.

Two outcome-CV entry points are deliberately kept apart and must not be merged.
:func:`outcome_cv_fixed_folds` searches a public log-spaced grid; :func:`outcome_cv_fixed_folds_raw_grid`
reproduces the original authors' Figure 3 exactly, on their raw
``linspace(1e-10, 300, 1000)`` grid.  Collapsing them would silently change
which of the two a given published number came from.

``Xc`` is centered by the caller on the full control sample throughout.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from .weights import l2_balancing_weights, l2_residual_imbalance

__all__ = [
    "double_ridge_estimate",
    "ols_plugin_estimate",
    "outcome_ridge_slope",
    "outcome_cv_fixed_folds_raw_grid",
    "design_only_balance_penalty_cv",
    "outcome_cv_fixed_folds",
    "outcome_cv_fixed_folds_prepared",
    "outcome_cv_shuffled_kfold",
    "prepare_outcome_cv_fixed_folds",
    "ridge_plugin_weights",
]

#: The original authors' raw penalty grid, reproduced exactly.
ABW_AUTHORS_OUTCOME_GRID = np.linspace(1e-10, 300.0, 1000)

# The penalty grid travels with the code from Section 7, the only section that
# uses double ridge; no other section holds a competing value, so keeping it as
# a default unifies nothing.  A study with its own grid should pass it.
DEFAULT_PENALTY_GRID = np.logspace(-4, 3, 141)


def outcome_cv_shuffled_kfold(
    X: np.ndarray,
    y: np.ndarray,
    *,
    repeats: int = 5,
    seed: int,
) -> tuple[float, dict]:
    """Outcome-ridge CV from the authors' simulation implementation.

    Each repeat uses shuffled five-fold CV, centers the training and held-out
    samples at the training-fold means, minimizes held-out mean squared error
    continuously over ``[0, 100000]`` from ``0.01``, and returns the median
    selected penalty.  The public implementation leaves fold shuffling
    unseeded; this implementation supplies consecutive seeds so the simulation
    is reproducible while otherwise retaining its tuning rule.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")

    selected = np.empty(repeats, dtype=float)
    objectives = np.empty(repeats, dtype=float)
    successes = np.empty(repeats, dtype=bool)
    messages: list[str] = []
    for repeat in range(repeats):
        prepared = []
        splitter = KFold(
            n_splits=5, shuffle=True, random_state=seed + repeat
        )
        for train, test in splitter.split(X):
            x_mean = X[train].mean(axis=0)
            X_train = X[train] - x_mean
            X_test = X[test] - x_mean
            y_mean = y[train].mean()
            y_train = y[train] - y_mean
            y_test = y[test] - y_mean
            covariance = X_train.T @ X_train / len(train)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 0.0)
            score = eigenvectors.T @ (X_train.T @ y_train / len(train))
            prepared.append(
                (X_test @ eigenvectors, y_test, eigenvalues, score)
            )

        def loss(value: np.ndarray) -> float:
            lam = float(value[0])
            mse = 0.0
            for X_test_rotated, y_test, eigenvalues, score in prepared:
                beta_rotated = np.divide(
                    score,
                    eigenvalues + lam,
                    out=np.zeros_like(score),
                    where=(eigenvalues + lam) > 1e-12,
                )
                error = X_test_rotated @ beta_rotated - y_test
                mse += float(np.mean(error**2)) / 5.0
            return mse

        result = minimize(
            loss, x0=np.array([0.01]), bounds=[(0.0, 100000.0)],
            tol=1e-12,
        )
        selected[repeat] = float(result.x[0])
        objectives[repeat] = float(result.fun)
        successes[repeat] = bool(result.success)
        messages.append(str(result.message))

    return float(np.median(selected)), {
        "repeat_penalties": selected,
        "repeat_objectives": objectives,
        "repeat_successes": successes,
        "optimizer_messages": messages,
        "folds": 5,
        "repeats": repeats,
        "seed": seed,
        "aggregation": "median",
    }


def outcome_ridge_slope(Xc: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge slope on centered features Xc for centered y (penalty lam on S+lamI).

    Returns beta with muhat0-contribution t @ beta, on the *raw* feature scale,
    matching the paper's beta(lambda) = (1/n0) M_lambda (X0^c)^T y0.
    """
    n0 = Xc.shape[0]
    S = Xc.T @ Xc / n0
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = eigenvectors.T @ (Xc.T @ y / n0)
    denominator = eigenvalues + lam
    if lam > 1e-10:
        return eigenvectors @ (projected / denominator)
    inverse = np.divide(
        1.0, denominator, out=np.zeros_like(denominator),
        where=denominator > 1e-10,
    )
    return eigenvectors @ (projected * inverse)


def double_ridge_estimate(
    Xc: np.ndarray, y_control: np.ndarray, t: np.ndarray,
    delta: float, lam: float,
) -> dict:
    """Bruns-Smith augmented l2 balancing (double ridge) estimate of mu0.

    muhat0 = w(delta)^T y0 + Delta(delta)^T beta(lam),
    where w(delta) are l2 balancing weights and beta(lam) is the ridge outcome
    slope.  At delta = 0 the augmentation term vanishes (exact balance) and this
    collapses to the OLS/ridge plug-in; Bruns-Smith set delta = lam.
    """
    n0 = Xc.shape[0]
    S = Xc.T @ Xc / n0
    if delta > 1e-10:
        w = l2_balancing_weights(Xc, t, delta)
        delta_vec = l2_residual_imbalance(Xc, t, delta)
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        projected_shift = eigenvectors.T @ t
        denominator = eigenvalues + delta
        inverse = np.divide(
            1.0, denominator, out=np.zeros_like(denominator),
            where=denominator > 1e-10,
        )
        direction = eigenvectors @ (projected_shift * inverse)
        w = np.full(n0, 1.0 / n0) + Xc @ direction / n0
        w += (1.0 - w.sum()) / n0
        delta_vec = t - Xc.T @ w
    yc = y_control - y_control.mean()
    beta = outcome_ridge_slope(Xc, yc, lam)
    # The augmented estimator is itself linear in ``y_control``.  Keep the
    # original l2-balancing weights and expose the final implied weights
    # separately; balance/ESS diagnostics for double ridge must use the latter.
    if lam > 1e-10:
        correction = Xc @ np.linalg.solve(
            S + lam * np.eye(Xc.shape[1]), delta_vec
        ) / n0
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        denominator = eigenvalues + lam
        inverse = np.divide(
            1.0, denominator, out=np.zeros_like(denominator),
            where=denominator > 1e-10,
        )
        correction_direction = eigenvectors @ (
            (eigenvectors.T @ delta_vec) * inverse
        )
        correction = Xc @ correction_direction / n0
    augmented_weights = w + correction
    if lam <= 1e-10 or delta <= 1e-10:
        formula_muhat0 = float(
            y_control.mean() + w @ yc + delta_vec @ beta
        )
    else:
        formula_muhat0 = float(w @ y_control + delta_vec @ beta)
    if lam <= 1e-10 or delta <= 1e-10:
        augmented_weights += (1.0 - augmented_weights.sum()) / n0
        muhat0 = float(augmented_weights @ y_control)
    else:
        muhat0 = formula_muhat0
    return {
        "muhat0": muhat0,
        "weights": w,
        "base_weights": w,
        "augmented_weights": augmented_weights,
        "delta": delta,
        "lam": lam,
        "residual_imbalance_norm": float(np.linalg.norm(delta_vec)),
        "linear_representation_error": float(
            abs(augmented_weights @ y_control - muhat0)
        ),
        "formula_representation_error": float(abs(formula_muhat0 - muhat0)),
    }


def ols_plugin_estimate(Xc: np.ndarray, y_control: np.ndarray, t: np.ndarray) -> float:
    """OLS regression-adjustment plug-in: ybar0 + t^T beta_ols (delta,lam -> 0)."""
    yc = y_control - y_control.mean()
    beta = np.linalg.lstsq(Xc, yc, rcond=None)[0]
    return float(y_control.mean() + t @ beta)


def design_only_balance_penalty_cv(
    X: np.ndarray,
    target_shift: np.ndarray,
    balance_grid: np.ndarray | None = None,
    riesz_grid: np.ndarray | None = None,
) -> dict:
    """Reproduce the design-only delta tuners in the authors' public code.

    This intentionally keeps their different validation designs: two shuffled
    folds (seed 6401433) for held-out imbalance and ten shuffled folds (seed 1)
    for the held-out Riesz loss.  ``X`` is the raw control feature matrix; each
    training fold supplies its own centering, exactly as in ``hyperparam.py``
    from ``balance-equiv-jrssb``.
    """
    if balance_grid is None:
        balance_grid = np.linspace(0.0, 20.0, 1000)
    if riesz_grid is None:
        riesz_grid = np.linspace(0.0, 1.0, 1000)

    def curve(grid: np.ndarray, folds: int, seed: int, loss: str) -> np.ndarray:
        values = np.zeros(len(grid), dtype=float)
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        for train, test in splitter.split(X):
            Xtr = X[train] - X[train].mean(axis=0)
            Xte = X[test] - X[train].mean(axis=0)
            covariance = Xtr.T @ Xtr / len(train)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 0.0)
            projected = eigenvectors.T @ target_shift
            inverse = np.divide(
                1.0,
                eigenvalues[:, None] + grid[None, :],
                out=np.zeros((len(eigenvalues), len(grid))),
                where=(eigenvalues[:, None] + grid[None, :]) > 1e-12,
            )
            theta = eigenvectors @ (projected[:, None] * inverse)
            heldout_covariance = Xte.T @ Xte / len(test)
            if loss == "imbalance":
                heldout_moment = heldout_covariance @ theta
                values += 0.5 * np.sum(
                    (heldout_moment - target_shift[:, None]) ** 2, axis=0
                )
            elif loss == "riesz":
                values += (
                    np.einsum("ig,ij,jg->g", theta, heldout_covariance, theta)
                    - 2.0 * target_shift @ theta
                )
            else:
                raise ValueError(loss)
        return values / folds

    balance_curve = curve(balance_grid, folds=2, seed=6401433, loss="imbalance")
    riesz_curve = curve(riesz_grid, folds=10, seed=1, loss="riesz")
    balance_index = int(np.argmin(balance_curve))
    riesz_index = int(np.argmin(riesz_curve))
    return {
        "delta_cv_imbalance": float(balance_grid[balance_index]),
        "delta_cv_riesz": float(riesz_grid[riesz_index]),
        "balance_grid": np.asarray(balance_grid, dtype=float),
        "balance_curve": balance_curve,
        "riesz_grid": np.asarray(riesz_grid, dtype=float),
        "riesz_curve": riesz_curve,
        "balance_folds": 2,
        "balance_seed": 6401433,
        "riesz_folds": 10,
        "riesz_seed": 1,
    }


def outcome_cv_fixed_folds(
    X: np.ndarray, y: np.ndarray, grid: np.ndarray = DEFAULT_PENALTY_GRID
) -> tuple[float, np.ndarray]:
    """Paper-aligned 3-fold CV R2, with deterministic unshuffled folds.

    ``grid`` is on the normalized covariance scale, S + lambda I.  The final
    raw ridge penalty shown in the paper is n0 * lambda.
    """
    fold_scores = np.zeros((3, len(grid)))
    splitter = KFold(n_splits=3, shuffle=False)
    for k, (train, test) in enumerate(splitter.split(X)):
        Xtr_raw, Xte_raw = X[train], X[test]
        ytr, yte = y[train], y[test]
        xbar, ybar = Xtr_raw.mean(axis=0), ytr.mean()
        Xtr, Xte = Xtr_raw - xbar, Xte_raw - xbar
        yc = ytr - ybar
        S = Xtr.T @ Xtr / len(train)
        eig, V = np.linalg.eigh(S)
        eig = np.maximum(eig, 0.0)
        score = V.T @ (Xtr.T @ yc / len(train))
        beta_path = V @ (score[:, None] / (eig[:, None] + grid[None, :]))
        pred = ybar + Xte @ beta_path
        sse = np.sum((yte[:, None] - pred) ** 2, axis=0)
        denom = np.sum((yte - yte.mean()) ** 2)
        fold_scores[k] = 1.0 - sse / max(denom, 1e-12)
    cv_r2 = fold_scores.mean(axis=0)
    return float(grid[int(np.argmax(cv_r2))]), cv_r2


def prepare_outcome_cv_fixed_folds(
    X: np.ndarray, grid: np.ndarray = DEFAULT_PENALTY_GRID
) -> list[dict]:
    """Cache design-only linear algebra for repeated outcome-CV draws."""
    prepared = []
    splitter = KFold(n_splits=3, shuffle=False)
    for train, test in splitter.split(X):
        Xtr_raw, Xte_raw = X[train], X[test]
        xbar = Xtr_raw.mean(axis=0)
        Xtr, Xte = Xtr_raw - xbar, Xte_raw - xbar
        S = Xtr.T @ Xtr / len(train)
        eig, V = np.linalg.eigh(S)
        eig = np.maximum(eig, 0.0)
        prepared.append({
            "train": train,
            "test": test,
            "Xtr": Xtr,
            "XteV": Xte @ V,
            "V": V,
            "eig": eig,
            "denominator": eig[:, None] + grid[None, :],
        })
    return prepared


def outcome_cv_fixed_folds_prepared(
    prepared: list[dict], y: np.ndarray, grid: np.ndarray = DEFAULT_PENALTY_GRID
) -> tuple[float, np.ndarray]:
    """Evaluate deterministic outcome CV using cached design decompositions."""
    fold_scores = np.zeros((len(prepared), len(grid)))
    for k, fold in enumerate(prepared):
        ytr, yte = y[fold["train"]], y[fold["test"]]
        ybar = ytr.mean()
        yc = ytr - ybar
        score = fold["V"].T @ (
            fold["Xtr"].T @ yc / len(fold["train"])
        )
        pred = ybar + fold["XteV"] @ (
            score[:, None] / fold["denominator"]
        )
        sse = np.sum((yte[:, None] - pred) ** 2, axis=0)
        denom = np.sum((yte - yte.mean()) ** 2)
        fold_scores[k] = 1.0 - sse / max(denom, 1e-12)
    cv_r2 = fold_scores.mean(axis=0)
    return float(grid[int(np.argmax(cv_r2))]), cv_r2


def outcome_cv_fixed_folds_raw_grid(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Exact outcome-ridge CV used by the authors' Figure 3 notebook.

    Their public code globally centers X and y, applies unshuffled three-fold
    CV to raw Ridge penalties on ``linspace(1e-10, 300, 1000)``, and scores R2.
    The returned penalty is normalized only for the final full-sample fit.
    """
    Xg = X - X.mean(axis=0)
    yg = y - y.mean()
    scores = np.zeros((3, len(ABW_AUTHORS_OUTCOME_GRID)))
    splitter = KFold(n_splits=3, shuffle=False)
    for k, (train, test) in enumerate(splitter.split(Xg)):
        Xtr, Xte = Xg[train], Xg[test]
        ytr, yte = yg[train], yg[test]
        eigenvalues, eigenvectors = np.linalg.eigh(Xtr.T @ Xtr)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        projected = eigenvectors.T @ (Xtr.T @ ytr)
        beta_path = eigenvectors @ (
            projected[:, None]
            / (eigenvalues[:, None] + ABW_AUTHORS_OUTCOME_GRID[None, :])
        )
        prediction = Xte @ beta_path
        sse = np.sum((yte[:, None] - prediction) ** 2, axis=0)
        denominator = np.sum((yte - yte.mean()) ** 2)
        scores[k] = 1.0 - sse / max(denominator, 1e-12)
    mean_score = scores.mean(axis=0)
    raw_alpha = float(ABW_AUTHORS_OUTCOME_GRID[int(np.argmax(mean_score))])
    return raw_alpha / len(X), mean_score


def ridge_plugin_weights(Xc: np.ndarray, shift: np.ndarray, lam: float) -> np.ndarray:
    n = len(Xc)
    S = Xc.T @ Xc / n
    # The exact 171-column design has numerical rank 167.  Express the OLS
    # plug-in as a minimum-norm weight correction, imposing affine
    # normalization in the same linear system.  This remains stable when a
    # bootstrap sample contains repeated rows.  Positive ridge penalties use
    # the usual solve.
    if lam <= 0:
        system = np.vstack([Xc.T, np.ones((1, n))])
        target = np.r_[shift, 0.0]
        correction, *_ = np.linalg.lstsq(system, target, rcond=None)
        return np.full(n, 1.0 / n) + correction
    direction = np.linalg.solve(
        S + lam * np.eye(Xc.shape[1]), shift
    )
    return np.full(n, 1.0 / n) + Xc @ direction / n
