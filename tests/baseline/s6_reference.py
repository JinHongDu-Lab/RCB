"""Verbatim reference copy of Section 6's inline notebook code.

Section 6 defines its experiment inside notebook cells rather than in a module,
so there is nothing on disk to compare the extracted ``rcb`` package against.
This file lifts the definitions out of ``Section 6.ipynb`` cells 6, 8, 10, 12
and 16 without altering a character of them, so the equivalence tests have a
fixed pre-refactor reference.

Only the cells' *definitions* are kept.  Their side effects -- writing
``configuration.csv``, the ``joblib`` Monte Carlo sweeps, the summary frames and
the figure calls -- are left out, because reproducing them is the job of the
Section 6 driver, not of a test fixture.

Do not edit.  Regenerate from the notebook if the notebook changes.
"""


import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image, Markdown, display
from joblib import Parallel, delayed
from scipy.optimize import least_squares, linprog, minimize, nnls
from scipy.special import expit, logsumexp
from sklearn.linear_model import ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from spectral_quasi_score import (
        equation_41_estimates_many,
        spectral_quasi_score_estimates_many,
        spectral_quasi_score_from_feature_eigen,
    )
except ModuleNotFoundError:
    from aligned_experiments.spectral_quasi_score import (
        equation_41_estimates_many,
        spectral_quasi_score_estimates_many,
        spectral_quasi_score_from_feature_eigen,
    )


warnings.filterwarnings("ignore", category=RuntimeWarning)
sns.set_theme(style="whitegrid", context="notebook", font="DejaVu Sans", font_scale=1.55)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 17, "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15, "figure.titlesize": 19})

SEED = 20260714
P = 50
PHI1 = 0.50
PHI0_SETTINGS = [0.50, 1.25]
N1 = int(round(P / PHI1))
PILOT_FRACTION = 0.50
R2 = 4.0
SIGMA02 = 0.50
TAU = 1.0
LAMBDAS = np.logspace(-2, 2, 41)  # fixed compact subset of (0, infinity)
FIXED_LAMBDA = 1.0
SQ_THETA_BOUNDS = ((1e-4, 25.0), (1e-4, 10.0))

SMOKE_TEST = os.getenv("CAUSAL_EXTRAPOLATION_SMOKE", "0") == "1"
N_REPETITIONS = 4 if SMOKE_TEST else 120
N_JOBS = int(os.getenv("CAUSAL_EXTRAPOLATION_JOBS", "1" if SMOKE_TEST else "6"))

OVERLAP_SETTINGS = [
    ("Strong overlap", 0.5),
    ("Intermediate weak overlap", 1.0),
    ("Severe weak overlap", np.sqrt(3.0)),
]
FAMILY_ORDER = [
    "Uniform", "IPW", "Trimmed IPW", "Entropy balancing", "SBW", "CBPS",
    "Overlap weights", "Bruns-Smith ABW",
]
HEADLINE_STAGES = ["Base", "Augmented (OOD)"]
COMPARISON_STAGES = HEADLINE_STAGES + ["Full ARB"]
DIAGNOSTIC_STAGES = [
    "Augmented (OOD, Method of moments)", "Augmented (GCV)",
    "Augmented (fixed lambda=1)", "Augmented (oracle)",
]
STAGE_COLORS = {"Base": "#6B7280", "Augmented (OOD)": "#0072B2", "Full ARB": "#D55E00"}
STAGE_MARKERS = {"Base": "s", "Augmented (OOD)": "o", "Full ARB": "^"}
overlap_order = [x[0] for x in OVERLAP_SETTINGS]
stage_offsets = {"Base": -0.12, "Augmented (OOD)": 0.12, "Full ARB": 0.00}
DISPLAY_LABELS = {
    "Uniform": "Uniform", "IPW": "IPW", "Trimmed IPW": "Trimmed",
    "Entropy balancing": "Entropy", "SBW": "SBW", "CBPS": "CBPS",
    "Overlap weights": "Overlap", "Bruns-Smith ABW": "BS-ABW", "ARB": "ARB",
}


SHIFT_DIRECTION = np.ones(P) / np.sqrt(P)
SIGMA_EIGENVALUES = np.ones(P)


def target_mean(a: float) -> np.ndarray:
    return a * SHIFT_DIRECTION


def draw_replication(repetition: int, phi0: float, a: float) -> dict:
    rng = np.random.default_rng(np.random.SeedSequence([SEED, repetition, int(100 * phi0), int(1000 * a)]))
    n0 = int(round(P / phi0))
    nu1 = target_mean(a)
    X0 = rng.normal(size=(n0, P))
    X1 = nu1 + rng.normal(size=(N1, P))

    order = rng.permutation(N1)
    n_pilot = int(round(PILOT_FRACTION * N1))
    I_P = np.sort(order[:n_pilot])
    I_E = np.sort(order[n_pilot:])

    beta = np.sqrt(R2 / P) * rng.normal(size=P)
    epsilon0 = rng.normal(scale=np.sqrt(SIGMA02), size=n0)
    epsilon1 = rng.normal(scale=np.sqrt(SIGMA02), size=N1)
    Y0 = X0 @ beta + epsilon0
    Y1 = X1 @ beta + TAU + epsilon1
    return {
        "X0": X0, "X1": X1, "X1P": X1[I_P], "X1E": X1[I_E],
        "Y0": Y0, "Y1": Y1, "beta": beta, "nu1": nu1,
        "I_P": I_P, "I_E": I_E,
        "mu0_true": float(nu1 @ beta),
        "beta_shift_cos2": float((beta @ nu1) ** 2 / max((beta @ beta) * (nu1 @ nu1), 1e-15)),
    }


def normalize_weights(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    total = float(raw.sum())
    if total <= 1e-14:
        return np.full(len(raw), 1.0 / len(raw))
    return raw / total


def standardized_balance_problem(X0: np.ndarray, target: np.ndarray):
    mean0 = X0.mean(axis=0)
    scale0 = X0.std(axis=0, ddof=1)
    scale0 = np.where(scale0 > 1e-8, scale0, 1.0)
    return (X0 - mean0) / scale0, (target - mean0) / scale0


def propensity_weights(X0: np.ndarray, X1P: np.ndarray) -> dict:
    X = np.vstack([X0, X1P])
    A = np.r_[np.zeros(len(X0)), np.ones(len(X1P))]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=800, random_state=SEED),
    )
    model.fit(X, A)
    e0 = np.clip(model.predict_proba(X0)[:, 1], 1e-4, 1 - 1e-4)
    odds = e0 / (1 - e0)
    e0_trimmed = np.clip(e0, 0.05, 0.95)
    trimmed_odds = e0_trimmed / (1 - e0_trimmed)
    return {
        "IPW": normalize_weights(odds),
        "Trimmed IPW": normalize_weights(trimmed_odds),
        "Overlap weights": normalize_weights(e0),
    }


def entropy_balancing_weights(X0: np.ndarray, target: np.ndarray, n_pilot: int) -> np.ndarray:
    Z, t = standardized_balance_problem(X0, target)
    stabilization = float(np.sqrt(np.log(P + 1) / n_pilot))

    def objective(theta):
        scores = Z @ theta
        w = np.exp(scores - logsumexp(scores))
        value = logsumexp(scores) - np.log(len(scores)) - t @ theta
        value += 0.5 * stabilization * (theta @ theta)
        gradient = Z.T @ w - t + stabilization * theta
        return float(value), gradient

    result = minimize(
        fun=lambda x: objective(x)[0], jac=lambda x: objective(x)[1],
        x0=np.zeros(P), method="L-BFGS-B",
        options={"maxiter": 250, "ftol": 1e-11, "gtol": 1e-7},
    )
    if not result.success:
        raise RuntimeError(f"Entropy balancing failed: {result.message}")
    scores = Z @ result.x
    return np.exp(scores - logsumexp(scores))


def sbw_weights(X0: np.ndarray, target: np.ndarray) -> np.ndarray:
    n0 = len(X0)
    Z0, target_z = standardized_balance_problem(X0, target)
    uniform = np.full(n0, 1 / n0)
    alpha = 0.25
    sum_penalty = 100.0
    A = np.vstack([
        Z0.T,
        np.sqrt(alpha) * np.eye(n0),
        np.sqrt(sum_penalty) * np.ones((1, n0)),
    ])
    b = np.r_[target_z, np.sqrt(alpha) * uniform, np.sqrt(sum_penalty)]
    gamma, _ = nnls(A, b, maxiter=5 * n0)
    return normalize_weights(gamma)


def cbps_weights(X0: np.ndarray, X1P: np.ndarray) -> np.ndarray:
    X = np.vstack([X0, X1P])
    scaler = StandardScaler().fit(X)
    Z = np.column_stack([np.ones(len(X)), scaler.transform(X)])
    A = np.r_[np.zeros(len(X0)), np.ones(len(X1P))]

    def moments(theta):
        return Z.T @ (A - expit(np.clip(Z @ theta, -30, 30))) / len(Z)

    def jacobian(theta):
        e = expit(np.clip(Z @ theta, -30, 30))
        return -(Z.T * (e * (1 - e))) @ Z / len(Z)

    try:
        result = least_squares(
            moments, np.zeros(Z.shape[1]), jac=jacobian,
            max_nfev=200, xtol=1e-9, ftol=1e-9, gtol=1e-9,
        )
        if not result.success:
            raise RuntimeError(f"CBPS solve failed: {result.message}")
        return normalize_weights(np.exp(np.clip(Z[:len(X0)] @ result.x, -20, 20)))
    except Exception as exc:
        raise RuntimeError("CBPS failed; no silent fallback is permitted") from exc



def bruns_smith_ridge_balance_weights(X0: np.ndarray, target: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Ridge balancing component of ridge-ridge augmented balancing weights."""
    n0 = len(X0)
    xbar0 = X0.mean(axis=0)
    X0c = X0 - xbar0
    S0 = X0c.T @ X0c / n0
    evals, U = np.linalg.eigh(S0)
    direction = U @ ((U.T @ (target - xbar0)) / (evals + alpha))
    gamma = np.full(n0, 1.0 / n0) + X0c @ direction / n0
    if abs(gamma.sum() - 1.0) > 1e-10:
        raise RuntimeError("Bruns-Smith ridge-balancing weights are not affine normalized")
    return gamma

def all_base_weights(X0: np.ndarray, X1P: np.ndarray) -> dict:
    prop = propensity_weights(X0, X1P)
    target = X1P.mean(axis=0)
    return {
        "Uniform": np.full(len(X0), 1.0 / len(X0)),
        "IPW": prop["IPW"],
        "Trimmed IPW": prop["Trimmed IPW"],
        "Entropy balancing": entropy_balancing_weights(X0, target, len(X1P)),
        "SBW": sbw_weights(X0, target),
        "CBPS": cbps_weights(X0, X1P),
        "Overlap weights": prop["Overlap weights"],
        "Bruns-Smith ABW": bruns_smith_ridge_balance_weights(X0, target, alpha=1.0),
    }



def variance_component_estimates(X0: np.ndarray, y0: np.ndarray) -> tuple[float, float]:
    rhat2, sigmahat2 = equation_41_estimates_many(X0, y0)
    return float(rhat2[0]), float(sigmahat2[0])


def augmentation_path(X0, X1E, y0, gamma_base, nu1):
    n0, p = X0.shape
    X0c = X0 - X0.mean(axis=0, keepdims=True)
    y0c = y0 - y0.mean()
    delta = X1E.mean(axis=0) - X0.T @ gamma_base
    S0 = X0c.T @ X0c / n0
    evals, U = np.linalg.eigh(S0)
    utd = U.T @ delta
    sigma1hat = np.cov(X1E, rowvar=False, ddof=1)
    sigma1_diag = np.einsum("ij,ij->j", U, sigma1hat @ U)

    r_eq, s_eq = equation_41_estimates_many(X0, y0)
    r_sq, s_sq, sq_diagnostics = spectral_quasi_score_from_feature_eigen(
        X0, y0, evals, U,
        theta_bounds=SQ_THETA_BOUNDS,
        pilot_r2=r_eq, pilot_sigma2=s_eq,
    )
    r_eq, s_eq = float(r_eq[0]), float(s_eq[0])
    r_sq, s_sq = float(r_sq[0]), float(s_sq[0])

    gamma_path = np.empty((n0, len(LAMBDAS)))
    rood_mom = np.empty(len(LAMBDAS))
    rood_sq = np.empty(len(LAMBDAS))
    exact = np.empty(len(LAMBDAS))
    qhat_path = np.empty(len(LAMBDAS))
    for j, lam in enumerate(LAMBDAS):
        mdelta = U @ (utd / (evals + lam))
        gamma = gamma_base + X0c @ mdelta / n0
        gamma_path[:, j] = gamma
        q_signal = (lam**2 / p) * np.sum(utd**2 / (evals + lam) ** 2)
        q_correction = np.sum(
            (1 - 2 * lam / (evals + lam)) * sigma1_diag
        ) / (p * len(X1E))
        qhat = max(float(q_signal + q_correction), 0.0)
        qhat_path[j] = qhat
        weight_norm2 = float(gamma @ gamma)
        rood_mom[j] = r_eq * qhat + s_eq * weight_norm2
        rood_sq[j] = r_sq * qhat + s_sq * weight_norm2
        imbalance = X0.T @ gamma - nu1
        exact[j] = (R2 / p) * (imbalance @ imbalance) + SIGMA02 * weight_norm2

    # Ordinary source GCV with an unpenalized intercept.
    uty = U.T @ (X0c.T @ y0c / n0)
    gcv = np.empty(len(LAMBDAS))
    for j, lam in enumerate(LAMBDAS):
        beta_gcv = U @ (uty / (evals + lam))
        residual = y0c - X0c @ beta_gcv
        effective_df = 1.0 + np.sum(evals / (evals + lam))
        gcv[j] = np.mean(residual**2) / max((1 - effective_df / n0) ** 2, 1e-12)

    sq_index = int(np.argmin(rood_sq))
    mom_index = int(np.argmin(rood_mom))
    return {
        "gamma_path": gamma_path,
        "rood": rood_sq,
        "rood_sq": rood_sq,
        "rood_mom": rood_mom,
        "exact": exact,
        "qhat": qhat_path,
        "ood_index": sq_index,
        "ood_sq_index": sq_index,
        "ood_mom_index": mom_index,
        "gcv_index": int(np.argmin(gcv)),
        "oracle_index": int(np.argmin(exact)),
        "fixed_index": int(np.argmin(np.abs(LAMBDAS - FIXED_LAMBDA))),
        "rhat2": r_sq,
        "sigmahat2": s_sq,
        "rhat2_sq": r_sq,
        "sigmahat2_sq": s_sq,
        "rhat2_mom": r_eq,
        "sigmahat2_mom": s_eq,
        "rhohat_sq": float(sq_diagnostics["rhohat"][0]),
        "sq_boundary": bool(sq_diagnostics["boundary"][0]),
        "sq_objective_improvement": float(
            sq_diagnostics["objective_improvement"][0]
        ),
        "max_normalization_error": float(
            np.max(np.abs(gamma_path.sum(axis=0) - 1.0))
        ),
    }


def evaluate(family, stage, gamma, gamma_base, data, overlap, d2, phi0, repetition, selected_lambda, selected_rood):
    imbalance = data["X0"].T @ gamma - data["nu1"]
    delta_eval = data["X1E"].mean(axis=0) - data["X0"].T @ gamma
    signal_risk = float((R2 / P) * (imbalance @ imbalance))
    noise_risk = float(SIGMA02 * (gamma @ gamma))
    mu0_hat = float(gamma @ data["Y0"])
    tau_hat = float(data["Y1"].mean() - mu0_hat)
    return {
        "overlap": overlap, "D2": d2, "chi2": float(np.exp(d2) - 1),
        "oracle_ess_fraction": float(np.exp(-d2)),
        "phi0": phi0, "n0": len(data["X0"]), "n1": len(data["X1"]),
        "repetition": repetition, "family": family, "stage": stage,
        "lambda_selected": selected_lambda, "selected_R_OOD": selected_rood,
        "signal_risk": signal_risk, "noise_risk": noise_risk,
        "exact_risk": signal_risk + noise_risk,
        "mu0_error": mu0_hat - data["mu0_true"],
        "att_error": tau_hat - TAU,
        "population_imbalance_norm": float(np.linalg.norm(imbalance)),
        "imbalance_norm": float(np.linalg.norm(imbalance)),
        "delta_norm": float(np.linalg.norm(delta_eval)),
        "gamma_norm": float(np.linalg.norm(gamma)),
        "ess": float(1 / max(gamma @ gamma, 1e-15)),
        "negative_mass": float(np.maximum(-gamma, 0).sum()),
        "weight_sum": float(gamma.sum()),
        "distance_from_base": float(np.linalg.norm(gamma - gamma_base)),
        "beta_shift_cos2": data["beta_shift_cos2"],
    }


def arb_residual_weights(X0: np.ndarray, target: np.ndarray, zeta: float = 0.5) -> np.ndarray:
    """Procedure-1 approximate residual-balancing weights."""
    n0 = len(X0)
    Z0, target_z = standardized_balance_problem(X0, target)
    cap = n0 ** (-2 / 3)
    uniform = np.full(n0, 1 / n0)
    initial_imbalance = target_z - Z0.T @ uniform
    x0 = np.r_[uniform, np.max(np.abs(initial_imbalance)) + 1e-8]

    def objective(x):
        gamma, slack = x[:-1], x[-1]
        return (1 - zeta) * (gamma @ gamma) + zeta * slack**2

    def objective_gradient(x):
        gamma, slack = x[:-1], x[-1]
        return np.r_[2 * (1 - zeta) * gamma, 2 * zeta * slack]

    def balance_constraints(x):
        gamma, slack = x[:-1], x[-1]
        imbalance = target_z - Z0.T @ gamma
        return np.r_[slack - imbalance, slack + imbalance]

    balance_jacobian = np.zeros((2 * P, n0 + 1))
    balance_jacobian[:P, :n0] = Z0.T
    balance_jacobian[P:, :n0] = -Z0.T
    balance_jacobian[:, -1] = 1.0
    result = minimize(
        objective, x0, jac=objective_gradient, method="SLSQP",
        bounds=[(0.0, cap)] * n0 + [(0.0, None)],
        constraints=[
            {
                "type": "eq",
                "fun": lambda x: np.sum(x[:-1]) - 1.0,
                "jac": lambda x: np.r_[np.ones(n0), 0.0],
            },
            {
                "type": "ineq",
                "fun": balance_constraints,
                "jac": lambda x: balance_jacobian,
            },
        ],
        options={"maxiter": 400, "ftol": 1e-10, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"ARB weight program failed: {result.message}")
    return normalize_weights(result.x[:-1])


def arb_outcome_fit(X0: np.ndarray, Y0: np.ndarray):
    """Elastic Net outcome regression with a glmnet-style one-SE penalty."""
    scaler = StandardScaler().fit(X0)
    Z0 = scaler.transform(X0)
    cv_model = ElasticNetCV(
        l1_ratio=0.9, cv=5, n_alphas=60, max_iter=10000,
        random_state=SEED, n_jobs=1,
    ).fit(Z0, Y0)
    mean_loss = cv_model.mse_path_.mean(axis=1)
    se_loss = cv_model.mse_path_.std(axis=1, ddof=1) / np.sqrt(cv_model.mse_path_.shape[1])
    best = int(np.argmin(mean_loss))
    eligible = np.flatnonzero(mean_loss <= mean_loss[best] + se_loss[best])
    alpha_1se = float(np.max(cv_model.alphas_[eligible]))
    model = ElasticNet(alpha=alpha_1se, l1_ratio=0.9, max_iter=10000).fit(Z0, Y0)
    return scaler, model, alpha_1se


def full_arb_estimate(data):
    """Complete ARB: residual-balancing weights plus Elastic Net outcome regression."""
    X0, Y0 = data["X0"], data["Y0"]
    gamma_arb = arb_residual_weights(X0, data["X1P"].mean(axis=0))

    scaler, model, alpha_1se = arb_outcome_fit(X0, Y0)
    fitted_source = model.predict(scaler.transform(X0))
    fitted_target = float(model.predict(scaler.transform(data["X1"])).mean())
    residual_correction = float(gamma_arb @ (Y0 - fitted_source))
    estimate = fitted_target + residual_correction
    imbalance = X0.T @ gamma_arb - data["nu1"]
    delta_eval = data["X1E"].mean(axis=0) - X0.T @ gamma_arb
    return {
        "stage": "Full ARB",
        "mu0_error": float(estimate - data["mu0_true"]),
        "signal_risk": float((R2 / P) * (imbalance @ imbalance)),
        "noise_risk": float(SIGMA02 * (gamma_arb @ gamma_arb)),
        "gamma_norm": float(np.linalg.norm(gamma_arb)),
        "delta_norm": float(np.linalg.norm(delta_eval)),
        "residual_weight_norm": float(np.linalg.norm(gamma_arb)),
        "residual_imbalance_norm": float(
            np.linalg.norm(data["X1"].mean(axis=0) - X0.T @ gamma_arb)
        ),
        "elastic_net_alpha_1se": alpha_1se,
        "fitted_target_component": fitted_target,
        "residual_correction": residual_correction,
        "lambda_OOD": np.nan,
        "uses_centered_ridge_augmentation": False,
    }
