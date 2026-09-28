"""Regression tests for the corrected Section 6 benchmark definitions."""

from __future__ import annotations

import numpy as np

from experiments.evaluation import dgp
from rcb.balancing import design_eigendecomposition, ridge_augmented_path
from rcb.benchmarks import weights as W
from rcb.benchmarks.arb import arb_residual_weights
from rcb.benchmarks.double_ridge import (
    double_ridge_estimate,
    design_only_balance_penalty_cv,
    outcome_cv_shuffled_kfold,
)


def test_trimmed_ipw_zeros_instead_of_clipping_excluded_sources() -> None:
    source = np.array([[-4.0], [-1.0], [0.0], [1.0], [4.0]])
    target = np.array([[1.5], [2.0], [2.5], [3.0]])
    model = W.fit_propensity_model(source, target, seed=1)
    propensity = W.predict_propensity(model, source)
    retained = (propensity >= 0.2) & (propensity <= 0.8)
    families, diagnostics = W.propensity_weight_families(
        source, target, seed=1, trim=(0.2, 0.8)
    )
    trimmed = families["Trimmed IPW"]
    assert retained.any() and (~retained).any()
    assert np.all(trimmed[~retained] == 0.0)
    assert np.isclose(trimmed.sum(), 1.0)
    assert diagnostics["retained_source_count"] == int(retained.sum())


def test_sbw_enforces_declared_constraints_and_exact_normalization() -> None:
    source = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    target = np.array([0.75])
    weights, diagnostics = W.sbw_weights(
        source, target, balance_tolerance=0.1
    )
    center = source.mean(axis=0)
    scale = source.std(axis=0, ddof=1)
    standardized_imbalance = (
        ((source - center) / scale).T @ weights
        - (target - center) / scale
    )
    assert np.isclose(weights.sum(), 1.0, atol=5e-10)
    assert weights.min() >= -5e-10
    assert np.max(np.abs(standardized_imbalance)) <= 0.1 + 5e-8
    assert diagnostics["max_balance_constraint_violation"] <= 5e-8


def test_outcome_cv_shuffled_kfold_is_reproducible() -> None:
    rng = np.random.default_rng(21)
    X = rng.normal(size=(35, 5))
    y = X @ np.arange(1.0, 6.0) + rng.normal(size=35)
    first, first_diagnostics = outcome_cv_shuffled_kfold(
        X, y, repeats=2, seed=8
    )
    second, second_diagnostics = outcome_cv_shuffled_kfold(
        X, y, repeats=2, seed=8
    )
    assert first == second
    np.testing.assert_array_equal(
        first_diagnostics["repeat_penalties"],
        second_diagnostics["repeat_penalties"],
    )
    assert first_diagnostics["folds"] == 5
    assert first_diagnostics["aggregation"] == "median"


def test_full_abw_default_is_complete_double_ridge_with_delta_equal_lambda() -> None:
    data = dgp.draw_replication(0, 0.5, 1.0)
    rows = dgp.full_double_ridge_abw(data, repetition=0)
    default = rows[0]
    assert default["stage"] == "Full ABW (outcome CV, delta=lambda)"
    assert default["delta_balance"] == default["lambda_outcome"]
    assert default["uses_target_outcomes"] is False
    assert default["linear_representation_error"] < 1e-10

    center = data["X0"].mean(axis=0)
    direct = double_ridge_estimate(
        data["X0"] - center,
        data["Y0"],
        data["X1"].mean(axis=0) - center,
        default["delta_balance"],
        default["lambda_outcome"],
    )
    assert np.isclose(
        default["mu0_error"], direct["muhat0"] - data["mu0_true"]
    )


def test_trimmed_sensitivity_changes_both_samples_and_estimand() -> None:
    data = dgp.draw_replication(0, 0.5, 1.0)
    result = dgp.sample_trimmed_ipw_estimate(data)
    assert result["source_retained"] < len(data["X0"])
    assert result["target_retained"] < len(data["X1"])
    assert result["estimand"] == "ATT in retained target sample"
    assert np.isclose(result["weight_sum"], 1.0)


def test_arb_weights_satisfy_published_program_constraints() -> None:
    rng = np.random.default_rng(13)
    source = rng.normal(size=(45, 6))
    target = rng.normal(loc=0.25, size=(30, 6)).mean(axis=0)
    weights, diagnostics = arb_residual_weights(source, target, zeta=0.5)
    assert abs(weights.sum() - 1.0) <= 5e-8
    assert weights.min() >= -5e-8
    assert weights.max() <= len(source) ** (-2 / 3) + 5e-8
    assert diagnostics["max_balance_constraint_violation"] <= 5e-8


def test_l2_base_is_ridge_balancing_at_the_design_only_riesz_penalty() -> None:
    """The l2 family is w(delta) from the pilot fold, delta from the Riesz-CV rule."""
    data = dgp.draw_replication(0, 1.25, 1.0)  # overparameterized: n0 = 40 < p
    X0, X1P = data["X0"], data["X1P"]
    diagnostics = {}
    bases = dgp.all_base_weights(X0, X1P, diagnostics=diagnostics)
    delta = diagnostics["L2 balancing"]["delta"]
    shift = X1P.mean(axis=0) - X0.mean(axis=0)
    assert delta == design_only_balance_penalty_cv(X0, shift)["delta_cv_riesz"]
    assert 0.0 < delta <= 1.0
    direct = W.l2_balancing_weights(X0 - X0.mean(axis=0), shift, delta)
    np.testing.assert_allclose(bases["L2 balancing"], direct, rtol=0, atol=1e-12)
    assert abs(bases["L2 balancing"].sum() - 1.0) < 1e-10


def test_arb_base_is_the_full_arb_residual_weight_vector() -> None:
    data = dgp.draw_replication(0, 0.5, 1.0)
    bases = dgp.all_base_weights(data["X0"], data["X1P"])
    full = dgp.arb_estimate(data)
    np.testing.assert_array_equal(
        np.linalg.norm(bases["ARB"]), full["residual_weight_norm"]
    )
    assert bases["ARB"].min() >= -5e-8
    assert bases["ARB"].max() <= len(data["X0"]) ** (-2 / 3) + 5e-8


def test_full_abw_rows_report_the_l2_family() -> None:
    data = dgp.draw_replication(0, 0.5, 1.0)
    rows = dgp.full_double_ridge_abw(data, repetition=0)
    assert {row["family"] for row in rows} == {"L2 balancing"}


def test_double_ridge_implied_weights_are_the_ridge_augmented_path() -> None:
    """ABW's implied weights are the proposed path from w(delta) at lambda.

    This is the identity the manuscript states when it reads the l2 row as a
    comparison of penalty rules; it is checked on a rank-deficient design so
    that the zero eigenvalues are exercised too.
    """
    data = dgp.draw_replication(1, 1.25, 1.0)
    X0 = data["X0"]
    center = X0.mean(axis=0)
    shift = data["X1"].mean(axis=0) - center
    delta, lam = 0.7, 0.3
    abw = double_ridge_estimate(X0 - center, data["Y0"], shift, delta, lam)
    _, X0c, evals, U = design_eigendecomposition(X0, clip=False)
    path, _ = ridge_augmented_path(
        X0, X0c, evals, U, data["X1"].mean(axis=0), abw["base_weights"],
        np.array([lam]),
    )
    np.testing.assert_allclose(path[:, 0], abw["augmented_weights"], rtol=0, atol=1e-12)
    assert np.isclose(path[:, 0] @ data["Y0"], abw["muhat0"])
