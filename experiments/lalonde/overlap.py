#!/usr/bin/env python3
"""Outcome-free classifier-induced overlap design used by formal R9."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import experiment as E


SEED = 20260816


def normalize_logweights(logweights: np.ndarray) -> np.ndarray:
    return np.exp(logweights - logsumexp(logweights))


def fit_cross_fitted_density_ratio(data: dict) -> dict:
    """Cross-fitted ridge-logistic density-ratio proxy using X and arm only."""
    X0, Xt = data["X0"], data["Xt"]
    X = np.vstack([X0, Xt])
    sample = np.r_[np.zeros(len(X0), dtype=int), np.ones(len(Xt), dtype=int)]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000),
    )
    propensity = cross_val_predict(
        model, X, sample, cv=folds, method="predict_proba"
    )[:, 1]
    propensity = np.clip(propensity, 1e-4, 1.0 - 1e-4)
    source_propensity = propensity[: len(X0)]
    ratio = (
        source_propensity / (1.0 - source_propensity)
        * len(X0) / len(Xt)
    )
    log_ratio = np.clip(np.log(ratio), -5.0, 5.0)
    ratio = np.exp(log_ratio)
    ratio /= ratio.mean()
    return {
        "ratio": ratio,
        "log_ratio": np.log(ratio),
        "target_proxy": ratio / ratio.sum(),
        "propensity_clip": [1e-4, 1.0 - 1e-4],
        "log_ratio_clip": [-5.0, 5.0],
    }


def source_distribution(alpha: float, density: dict) -> np.ndarray:
    return normalize_logweights(alpha * density["log_ratio"])


def classifier_chi2(q: np.ndarray, density: dict) -> float:
    p = density["target_proxy"]
    return float(np.sum(p * p / q) - 1.0)


def design_row(alpha: float, density: dict) -> dict:
    q = source_distribution(alpha, density)
    return {
        "alpha": float(alpha),
        "classifier_chi2": classifier_chi2(q, density),
        "source_sampling_ess": float(1.0 / (q @ q)),
        "maximum_sampling_probability": float(q.max()),
    }


def fit_plasmode_signal(data: dict) -> dict:
    """Fit the single R9 outcome surface before source resampling."""
    X0, Xt, Xc, y0 = data["X0"], data["Xt"], data["Xc"], data["y0"]
    penalty, _ = E.outcome_cv_fixed_folds_raw_grid(Xc, y0)
    covariance = Xc.T @ Xc / len(Xc)
    beta = np.linalg.solve(
        covariance + penalty * np.eye(Xc.shape[1]),
        Xc.T @ (y0 - y0.mean()) / len(Xc),
    )
    fitted_source = y0.mean() + Xc @ beta
    fitted_target = y0.mean() + (Xt - X0.mean(axis=0)) @ beta
    return {
        "m0_source": fitted_source,
        "m0_target": fitted_target,
        "truth_mu0": float(fitted_target.mean()),
        "residual_sigma": float(np.std(y0 - fitted_source, ddof=1)),
        "pilot_lambda": float(penalty),
    }
