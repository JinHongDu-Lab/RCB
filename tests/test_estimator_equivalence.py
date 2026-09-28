"""``rcb.estimator.fit_rcb`` reproduces each study's hand-assembled estimator.

The three studies assemble the same estimator from the same pieces but in
their own code, with their own settings.  This pins bit-for-bit agreement
with ``fit_rcb`` -- ``assert_array_equal``, not ``allclose`` -- on the risk
curve, the selected index, the selected weights and the estimate, so that
switching cannot move a published number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
if str(CODE / "tests" / "baseline") not in sys.path:
    sys.path.insert(0, str(CODE / "tests" / "baseline"))

from rcb.balancing import (  # noqa: E402
    design_eigendecomposition,
    pilot_ridge_base_weights,
    ridge_augmented_path,
)
from rcb.benchmarks.gcv import source_gcv_curves  # noqa: E402
from rcb.estimator import fit_rcb, prepare_geometry, reference_penalty_rules  # noqa: E402
from rcb.nuisance import DEFAULT_VARIANCE_COMPONENT_BOUNDS  # noqa: E402
from rcb.risk import exact_risk_components, feasible_risk_geometry  # noqa: E402

equal = np.testing.assert_array_equal


# ---------------------------------------------------------------------------
# Evaluation study (Section 5.2): experiments/evaluation/dgp.py::augmentation_path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("repetition,phi0,a", [(0, 0.5, 1.0), (1, 1.25, np.sqrt(3.0))])
def test_matches_evaluation_assembly(repetition: int, phi0: float, a: float) -> None:
    from experiments.evaluation import dgp

    data = dgp.draw_replication(repetition, phi0, a)
    X0, Y0, X1E, nu1 = data["X0"], data["Y0"], data["X1E"], data["nu1"]
    bases = dgp.all_base_weights(X0, data["X1P"])
    geometry = prepare_geometry(X0)
    for family, base in bases.items():
        path = dgp.augmentation_path(X0, X1E, Y0, base, nu1)
        fit = fit_rcb(
            X0, Y0, X1E, base, dgp.LAMBDAS,
            theta_bounds=dgp.SQ_THETA_BOUNDS, geometry=geometry,
        )
        rules = reference_penalty_rules(
            fit, Y0, fixed_lambda=dgp.FIXED_LAMBDA, nu1=nu1,
            true_r2=dgp.R2, true_sigma2=dgp.SIGMA02, X0=X0,
        )
        equal(fit.risk_path, path["rood_sq"])
        equal(fit.weights_path, path["gamma_path"])
        equal(fit.qhat_path, path["qhat"])
        assert fit.selected_index == path["ood_sq_index"], family
        assert fit.r2 == path["rhat2_sq"] and fit.sigma2 == path["sigmahat2_sq"]
        equal(rules["two_moment_risk_path"], path["rood_mom"])
        assert rules["two_moment"] == path["ood_mom_index"]
        assert rules["gcv"] == path["gcv_index"]
        assert rules["oracle"] == path["oracle_index"]
        assert rules["fixed"] == path["fixed_index"]
        equal(rules["exact_total"], path["exact"])
        # The study's estimate is the 1-D dot product in dgp.evaluate.
        selected = path["gamma_path"][:, path["ood_sq_index"]]
        equal(fit.weights, selected)
        assert fit.muhat0 == float(selected @ Y0)
        assert fit.selected_lambda == float(dgp.LAMBDAS[path["ood_sq_index"]])


# ---------------------------------------------------------------------------
# Real-data study (Section 6): the study's prepare / estimate / select adapters
# over fit_rcb, with the infinity endpoint and standardized outcomes.  The
# original two-step implementation they replaced is pinned by the recorded
# ``s7`` fixtures in test_equivalence.py; this checks the adapters agree with a
# direct call, field by field.
# ---------------------------------------------------------------------------
def test_matches_lalonde_assembly() -> None:
    import generate  # the fixture module's synthetic design, for a rank-deficient case
    from experiments.lalonde import experiment as E
    from experiments.lalonde import methods as M

    d = generate._synthetic_design(20260806, n0=60, nt=70, p=45)
    X0, Xt, y0 = d["X0"], d["Xt"], d["y0"] * 2500.0 + 4000.0  # dollar scale
    rng = np.random.default_rng(3)
    order = rng.permutation(len(Xt))
    pilot, evaluation = order[:35], order[35:]
    Xc = X0 - X0.mean(axis=0)
    shift_pilot = Xt[pilot].mean(axis=0) - X0.mean(axis=0)
    for name, base in {
        "uniform": np.full(len(X0), 1.0 / len(X0)),
        "l2": M.l2_balancing_weights(Xc, shift_pilot, 1.0),
    }.items():
        prepared = E.prepare_adjustment_paths(X0, Xt[evaluation], {name: base})
        r2, sigma2 = E.estimate_risk_nuisance(X0, y0, prepared["geometry"])
        selected = E.select_prepared_adjustment(
            prepared["paths"][name], y0, r2, sigma2
        )
        fit = fit_rcb(
            X0, y0, Xt[evaluation], base, E.GRID,
            theta_bounds=M.SQ_BOUNDS, standardize_outcomes=True,
            include_infinite_endpoint=True,
        )
        assert fit.r2 == r2 and fit.sigma2 == sigma2
        equal(fit.penalty_grid, prepared["penalty_grid"])
        equal(fit.qhat_path, selected["qhat_path"])
        equal(fit.risk_path, selected["risk"])
        equal(fit.estimate_path, selected["estimate_path"])
        assert fit.selected_index == selected["selected_index"]
        assert fit.selected_lambda == selected["selected_lambda"]
        equal(fit.weights, selected["weights"])
        assert fit.muhat0 == selected["muhat0"]
        assert fit.lambda_at_boundary == selected["lambda_at_boundary"]
        assert (
            fit.selected_no_adjustment_endpoint
            == selected["selected_no_adjustment_endpoint"]
        )
        assert float(np.linalg.norm(fit.base_residual_imbalance)) == (
            selected["base_residual_imbalance_norm"]
        )


# ---------------------------------------------------------------------------
# Simulation study (Section 5.1): the block inside
# experiments/simulation/dgp.py::_inference_design, on an outcome matrix
# ---------------------------------------------------------------------------
def test_matches_simulation_assembly() -> None:
    from experiments.simulation.dgp import ExperimentConfig, _nuisance_estimates, mean_shift

    config = ExperimentConfig()
    p, eta, outcomes = 40, 0.5, 6
    rng = np.random.default_rng(config.seed + 303)
    n0 = int(round(p / config.phi0))
    n1_total = int(round(p / config.phi1))
    n_pilot = int(round(config.pilot_fraction * n1_total))
    nu1 = mean_shift(
        p, n0, eta, config.calibration_shift_rho2,
        orientation="sparse", sparse_fraction=config.sparse_shift_fraction,
    )
    X0 = rng.normal(size=(n0, p))
    X1 = rng.normal(size=(n1_total, p)) + nu1
    X1_pilot, X1_evaluation = X1[:n_pilot], X1[n_pilot:]

    # --- the study's own assembly, verbatim ---
    xbar0, X0c, eigenvalues, eigenvectors = design_eigendecomposition(X0)
    gamma_base = pilot_ridge_base_weights(
        X0, X0c, eigenvalues, eigenvectors, xbar0, X1_pilot.mean(axis=0),
        config.base_ridge_penalty,
    )
    gamma_path, delta = ridge_augmented_path(
        X0, X0c, eigenvalues, eigenvectors, X1_evaluation.mean(axis=0),
        gamma_base, config.lambdas,
    )
    exact_signal, exact_residual, exact_total = exact_risk_components(
        X0, gamma_path, nu1, config.r2, config.sigma2
    )
    signal_factor, weight_norm = feasible_risk_geometry(
        X1_evaluation, delta, eigenvalues, eigenvectors, gamma_path, config.lambdas
    )
    beta = rng.normal(scale=np.sqrt(config.r2 / p), size=(p, outcomes))
    epsilon = rng.normal(scale=np.sqrt(config.sigma2), size=(n0, outcomes))
    Y0 = X0 @ beta + epsilon
    sq_r2, sq_sigma2, sq_diagnostics = _nuisance_estimates(
        X0, Y0, eigenvalues, eigenvectors
    )
    feasible = (
        sq_r2[:, None] * signal_factor[None, :]
        + sq_sigma2[:, None] * weight_norm[None, :]
    )
    selected_ood = np.argmin(feasible, axis=1)
    gcv = source_gcv_curves(X0c, Y0, eigenvalues, eigenvectors, config.lambdas)
    selected_gcv = np.argmin(gcv, axis=0)
    outcome_index = np.arange(outcomes)
    estimates_by_lambda = gamma_path.T @ Y0
    mu_hat = estimates_by_lambda[selected_ood, outcome_index]

    # --- through the entry point ---
    # Section 5.1's settings: the shared default search box on a 41-point grid.
    fit = fit_rcb(
        X0, Y0, X1_evaluation, gamma_base, config.lambdas,
        theta_bounds=DEFAULT_VARIANCE_COMPONENT_BOUNDS, grid_size=41,
    )
    rules = reference_penalty_rules(
        fit, Y0, nu1=nu1, true_r2=config.r2, true_sigma2=config.sigma2, X0=X0,
    )
    equal(fit.risk_path, feasible)
    equal(fit.selected_index, selected_ood)
    equal(fit.weights_path, gamma_path)
    equal(fit.estimate_path, estimates_by_lambda)
    equal(fit.muhat0, mu_hat)
    equal(fit.r2, sq_r2)
    equal(fit.sigma2, sq_sigma2)
    equal(fit.nuisance_diagnostics["pilot_r2"], sq_diagnostics["pilot_r2"])
    equal(rules["gcv"], selected_gcv)
    equal(rules["gcv_curve"], gcv)
    assert rules["oracle"] == int(np.argmin(exact_total))
    equal(rules["exact_signal"], exact_signal)
    equal(rules["exact_residual"], exact_residual)
    equal(fit.weights, gamma_path[:, selected_ood])


def test_simulation_inference_design_runs_end_to_end() -> None:
    """The study's own loop body executes and reports the same selection.

    ``_inference_design`` is what the production run calls; the assembly test
    above re-derives its arithmetic but does not call it, so this guards the
    wiring (imports, names, row keys) that only a run would otherwise exercise.
    """
    from experiments.simulation.dgp import ExperimentConfig, _inference_design

    config = ExperimentConfig()
    rows = _inference_design(
        rng=np.random.default_rng(7), config=config, p=40, eta=0.5,
        phi0=config.phi0, phi1_total=config.phi1,
        pilot_fraction=config.pilot_fraction, outcomes=3, design=0,
        experiment_label="test",
    )
    assert len(rows) == 3
    for row in rows:
        assert row["lambda_ood"] in config.lambdas
        assert row["lambda_gcv"] in config.lambdas
        assert np.isfinite(row["prediction_error"])
        assert row["exact_total_at_ood"] >= row["exact_total_at_oracle"]
        assert isinstance(row["covered_95"], bool)
