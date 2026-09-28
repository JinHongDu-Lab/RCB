"""Fixed-seed cases pinning the behavior of every primitive slated for extraction.

The same case definitions are evaluated twice.  Run as a script, they go through
``reference/`` -- the modules materialized verbatim from the notebooks before any
code moved -- and the results are stored as ``baseline_s{5,6,7}.npz``.  Imported
by ``tests/test_equivalence.py`` with :data:`MODE` set to ``"current"``, they go
through the extracted ``rcb`` package instead, and the two are compared with
``np.testing.assert_array_equal``: exact equality, not a tolerance.

One set of definitions serving both sides is what makes the comparison
meaningful -- a second, separately written harness would only pin whatever it
happened to call.

Every case is driven by a fixed seed and synthetic covariates.  No real data is
read and nothing is written outside ``tests/baseline/``.

A caveat on regenerating: ``reference/`` is frozen, but the Section 7 modules
are live and now delegate into ``rcb``, so reference mode reproduces the
recorded numbers only at the commit where they were first recorded.  After
that, the committed ``.npz`` files are the reference -- regenerating them is a
deliberate act of rebaselining, not a refresh.

Usage::

    python code/tests/baseline/generate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]

#: ``"reference"`` resolves the frozen pre-refactor modules, ``"current"`` the
#: extracted package.  ``tests/test_equivalence.py`` flips this.
MODE = "reference"

# ``CODE`` puts ``rcb`` and ``experiments`` on the path.  ``reference/`` comes
# last so its frozen copies shadow same-named modules elsewhere.
for path in (CODE, HERE, HERE / "reference"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _nuisance_module():
    """The spectral quasi-score implementation under test."""
    if MODE == "reference":
        import spectral_quasi_score as module
    else:
        from rcb import nuisance as module
    return module


class _Namespace:
    """Flat view over several modules, so cases can name functions directly."""

    def __init__(self, *modules):
        self._modules = modules

    def __getattr__(self, name):
        for module in self._modules:
            if hasattr(module, name):
                return getattr(module, name)
        raise AttributeError(name)


def _section5_namespace() -> _Namespace:
    """Section 5's balancing and risk primitives, reference or extracted."""
    if MODE == "reference":
        import paper_requirements_aligned_core as core

        return _Namespace(core)
    from rcb import balancing, risk

    import paper_requirements_aligned_core as core  # mean_shift stays local

    return _Namespace(balancing, risk, core)


def _flatten(prefix: str, value, out: dict) -> None:
    """Store one case result as 0-d/n-d float arrays under dotted keys."""
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}", item, out)
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _flatten(f"{prefix}.{index}", item, out)
        return
    if isinstance(value, (bool, np.bool_)):
        out[prefix] = np.asarray(bool(value))
        return
    if isinstance(value, str):
        out[prefix] = np.asarray(value)
        return
    array = np.asarray(value)
    if array.dtype == object:
        return
    out[prefix] = array


def _record(cases: dict) -> dict:
    recorded: dict = {}
    for name, thunk in cases.items():
        _flatten(name, thunk(), recorded)
    return recorded


# ---------------------------------------------------------------------------
# Shared synthetic inputs
# ---------------------------------------------------------------------------
def _synthetic_design(seed: int, n0: int, nt: int, p: int) -> dict:
    rng = np.random.default_rng(seed)
    shift = np.ones(p) / np.sqrt(p)
    X0 = rng.standard_normal((n0, p))
    Xt = 0.6 * shift + rng.standard_normal((nt, p))
    beta = rng.standard_normal(p) / np.sqrt(p)
    y0 = X0 @ beta + 0.5 * rng.standard_normal(n0)
    return {"X0": X0, "Xt": Xt, "beta": beta, "y0": y0, "shift": shift}


# ---------------------------------------------------------------------------
# Section 5
# ---------------------------------------------------------------------------
def section5_cases() -> dict:
    C = _section5_namespace()
    S = _nuisance_module()

    d = _synthetic_design(20260101, n0=90, nt=60, p=40)
    X0, y0 = d["X0"], d["y0"]
    lambdas = np.geomspace(0.02, 30.0, 25)

    xbar0, X0c, evals, evecs = C.design_eigendecomposition(X0)
    gamma_base = C.design_independent_base_weights(len(X0), eta=0.5, varrho2=1.0)
    xbar_target = d["Xt"].mean(axis=0)
    gamma_path, delta = C.ridge_augmented_path(
        X0, X0c, evals, evecs, xbar_target, gamma_base, lambdas
    )
    nu1 = C.mean_shift(40, len(X0), eta=0.5, rho2=0.5, orientation="flat")
    Y0 = np.column_stack([y0, y0 * 0.5 + 0.1])

    mom = S.two_moment_estimates(X0, Y0)

    return {
        "design_eigendecomposition": lambda: (xbar0, X0c, evals, evecs),
        "external_base_weights_eta0": lambda: C.design_independent_base_weights(90, 0.0, 1.0),
        "external_base_weights_eta1": lambda: C.design_independent_base_weights(90, 1.0, 2.0),
        "mean_shift_flat": lambda: C.mean_shift(40, 90, 0.5, 0.5, orientation="flat"),
        "mean_shift_sparse": lambda: C.mean_shift(
            40, 90, 0.5, 0.5, orientation="sparse", sparse_fraction=0.1
        ),
        "mean_shift_low": lambda: C.mean_shift(40, 90, 0.0, 0.5, orientation="low"),
        "mean_shift_high": lambda: C.mean_shift(40, 90, 1.0, 0.5, orientation="high"),
        "ridge_augmented_path": lambda: (gamma_path, delta),
        "honest_ridge_base_weights": lambda: C.pilot_ridge_base_weights(
            X0, X0c, evals, evecs, xbar0, d["Xt"][:30].mean(axis=0), alpha=1.0
        ),
        "exact_risk_components": lambda: C.exact_risk_components(
            X0, gamma_path, nu1, r2=1.0, sigma2=0.5
        ),
        "feasible_risk_geometry": lambda: C.feasible_risk_geometry(
            d["Xt"][30:], delta, evals, evecs, gamma_path, lambdas
        ),
        "source_gcv_curves": lambda: C.source_gcv_curves(
            X0c, Y0, evals, evecs, lambdas
        ),
        "deterministic_equivalent_diagonal": lambda: C.deterministic_equivalent_diagonal(
            lambdas, phi0=0.75, phi1=0.75, eta=0.5, rho2=0.5,
            varrho2=1.0, r2=1.0, sigma2=0.5,
        ),
        # eta=1 takes the extra target-kernel branch, and a structured spectrum
        # exercises the spectral averages a length-one default cannot.
        "deterministic_equivalent_diagonal_eta1": (
            lambda: C.deterministic_equivalent_diagonal(
                lambdas, phi0=0.75, phi1=0.75, eta=1.0, rho2=0.5,
                varrho2=1.0, r2=1.0, sigma2=0.5,
            )
        ),
        "deterministic_equivalent_diagonal_structured": (
            lambda: C.deterministic_equivalent_diagonal(
                lambdas, phi0=0.75, phi1=0.75, eta=1.0, rho2=0.5,
                varrho2=1.0, r2=1.0, sigma2=0.5,
                sigma0_eigenvalues=np.linspace(0.4, 3.0, 320),
                sigma1_diagonal=np.linspace(0.5, 2.0, 320),
            )
        ),
        "deterministic_equivalent_identity_closed_form": (
            lambda: C.deterministic_equivalent_isotropic(
                lambdas, phi0=0.75, phi1=0.75, eta=0.5, rho2=0.5,
                varrho2=1.0, r2=1.0, sigma2=0.5,
            )
        ),
        "equation_41_estimates_many": lambda: mom,
        "spectral_quasi_score_from_feature_eigen": (
            lambda: S.spectral_quasi_score_estimates(
                X0, Y0, evals, evecs,
                pilot_r2=mom[0], pilot_sigma2=mom[1], grid_size=41,
            )
        ),
        "spectral_quasi_score_estimates_many": (
            lambda: S._spectral_quasi_score_over_draws(X0, Y0)
        ),
        "centered_spectral_representation": (
            lambda: S._centered_spectral_representation(X0, Y0)
        ),
    }


# ---------------------------------------------------------------------------
# Section 6
# ---------------------------------------------------------------------------
def _section6_module():
    """Section 6's experiment code, reference or extracted."""
    if MODE == "reference":
        import s6_reference as module
    else:
        from experiments.evaluation import dgp as module
    return module


def section6_cases() -> dict:
    R = _section6_module()

    data = R.draw_replication(repetition=0, phi0=0.5, a=1.0)
    X0, X1P, X1E = data["X0"], data["X1P"], data["X1E"]
    y0 = data["Y0"]
    target = X1P.mean(axis=0)
    bases = R.all_base_weights(X0, X1P)
    path = R.augmentation_path(X0, X1E, y0, bases["Uniform"], data["nu1"])

    degenerate = np.array([0.4, np.nan, 1.2, -0.3, np.inf])
    if MODE == "reference":
        family_cases = {
            "normalize_weights": lambda: R.normalize_weights(degenerate),
            "propensity_weights": lambda: R.propensity_weight_families(X0, X1P),
            "entropy_balancing_weights": lambda: R.entropy_balancing_weights(
                X0, target, len(X1P)
            ),
            "sbw_weights": lambda: R.sbw_weights(X0, target),
            "cbps_weights": lambda: R.cbps_weights(X0, X1P),
            "bruns_smith_ridge_balance_weights": (
                lambda: R.l2_balancing_weights_spectral(X0, target, alpha=1.0)
            ),
            "arb_residual_weights": lambda: R.arb_residual_weights(
                X0, target, zeta=0.5
            ),
            "arb_outcome_fit": lambda: R.arb_elastic_net_fit(X0, y0),
        }
    else:
        # The per-study wrappers these cases were recorded through are gone;
        # the same vectors now come from rcb.benchmarks.bases driven by the
        # study's BASE_SETTINGS, so the case names stay and the resolution
        # moves.  The recorded values are untouched.
        import dataclasses

        from rcb.benchmarks.arb import arb_elastic_net_fit
        from rcb.benchmarks.bases import base_weight_families
        from rcb.diagnostics import normalize_weights

        def registry(names, **overrides):
            settings = R.BASE_SETTINGS
            if overrides:
                settings = dataclasses.replace(settings, **overrides)
            return base_weight_families(list(names), X0, X1P, settings)[0]

        def propensity_dict():
            w = registry(["ipw", "trimmed_ipw", "overlap"])
            return {
                "IPW": w["ipw"], "Trimmed IPW": w["trimmed_ipw"],
                "Overlap weights": w["overlap"],
            }

        family_cases = {
            "normalize_weights": lambda: normalize_weights(
                degenerate, on_invalid="repair"
            ),
            "propensity_weights": propensity_dict,
            "entropy_balancing_weights": lambda: registry(["entropy"])["entropy"],
            "sbw_weights": lambda: registry(["sbw"])["sbw"],
            "cbps_weights": lambda: registry(["cbps"])["cbps"],
            "bruns_smith_ridge_balance_weights": (
                lambda: registry(["l2"], l2_penalty=1.0)["l2"]
            ),
            "arb_residual_weights": lambda: registry(["arb"])["arb"],
            "arb_outcome_fit": lambda: arb_elastic_net_fit(X0, y0, seed=R.SEED),
        }

    return {
        "draw_replication": lambda: {
            k: v for k, v in data.items() if k not in ("I_P", "I_E")
        },
        "draw_replication_index_P": lambda: data["I_P"],
        "draw_replication_index_E": lambda: data["I_E"],
        **family_cases,
        "standardized_balance_problem": lambda: R.standardized_balance_problem(
            X0, target
        ),
        "all_base_weights": lambda: bases,
        # Recorded through a study helper that has been removed; it returned the
        # two-moment pilot estimates the spectral quasi-score search starts from.
        "variance_component_estimates": (
            (lambda: R.variance_component_estimates(X0, y0))
            if MODE == "reference" else _s6_two_moment_pilot(R, X0, y0)
        ),
        "augmentation_path": lambda: path,
        "evaluate": lambda: {
            key: value
            for key, value in R.score_weights(
                "Uniform", "Augmented (OOD)",
                path["gamma_path"][:, path["ood_sq_index"]], bases["Uniform"],
                data, "Intermediate weak overlap", 1.0, 0.5, 0,
                float(R.LAMBDAS[path["ood_sq_index"]]), float("nan"),
            ).items()
            if not isinstance(value, str)
        },
        "full_arb_estimate": lambda: {
            key: value
            for key, value in R.arb_estimate(data).items()
            if not isinstance(value, str)
        },
    }


# ---------------------------------------------------------------------------
# Section 7
# ---------------------------------------------------------------------------
def section7_cases() -> dict:
    from experiments.lalonde import experiment as E
    from experiments.lalonde import methods as M

    d = _synthetic_design(20260806, n0=140, nt=70, p=45)
    X0, Xt, y0 = d["X0"], d["Xt"], d["y0"]
    center, scale = M._standardize_fit(X0)
    Xc = (X0 - center) / scale
    shift = (Xt.mean(axis=0) - center) / scale
    grid = np.logspace(-2, 2, 25)

    uniform = np.full(len(X0), 1.0 / len(X0))
    geometry = None
    prepared = E.prepare_adjustment_paths(X0, Xt, {"uniform": uniform}, geometry)
    nuisance = E.estimate_risk_nuisance(X0, y0, prepared["geometry"])
    selected = E.select_prepared_adjustment(
        prepared["paths"]["uniform"], y0, nuisance[0], nuisance[1]
    )
    from rcb.estimator import fit_rcb

    uniform_fit = fit_rcb(
        X0, y0, Xt, uniform, E.GRID, theta_bounds=M.SQ_BOUNDS,
        standardize_outcomes=True, include_infinite_endpoint=True,
    )

    return {
        # Section 7's normalize_weights raises on non-finite input where
        # Section 6's silently repairs it; the inputs differ accordingly.
        "normalize_weights": lambda: M.normalize_weights(
            np.array([0.4, 0.9, 1.2, 0.3])
        ),
        "effective_sample_size": lambda: M.effective_sample_size(uniform),
        "weighted_smd": lambda: M.weighted_smd(X0, Xt, uniform),
        "negative_weight_fraction": lambda: M.negative_weight_fraction(
            np.array([0.5, -0.1, 0.6])
        ),
        "_standardize_fit": lambda: M._standardize_fit(X0),
        "outcome_ridge_coef": lambda: M.outcome_ridge_slope(Xc, y0, 1.0),
        "l2_balancing_weights": lambda: M.l2_balancing_weights(Xc, shift, 1.0),
        "residual_imbalance": lambda: M.l2_residual_imbalance(Xc, shift, 1.0),
        "double_ridge_estimate": lambda: M.double_ridge_estimate(
            Xc, y0, shift, 1.0, 1.0
        ),
        "ols_plugin_mu0": lambda: M.ols_plugin_estimate(Xc, y0, shift),
        "paper_author_double_ridge_tuners": lambda: M.design_only_balance_penalty_cv(
            X0, Xt.mean(axis=0) - X0.mean(axis=0), grid, grid
        ),
        # These cases were recorded through per-study wrappers that no longer
        # exist; the same vectors come from rcb.benchmarks.bases driven by the
        # study's BASE_SETTINGS.  Case names and recorded values are untouched.
        "entropy_tilt": lambda: _s7_family(
            M, X0, Xt, "entropy", entropy_stabilized=False
        ),
        "regularized_entropy_base": lambda: M.regularized_entropy_base(X0, Xt),
        "evaluation_propensity_bases": lambda: _s7_propensity(M, X0, Xt),
        "evaluation_sbw_base": lambda: _s7_family(M, X0, Xt, "sbw"),
        "evaluation_cbps_base": lambda: _s7_family(M, X0, Xt, "cbps"),
        "bruns_smith_ridge_balance_base": lambda: _s7_family(M, X0, Xt, "l2")[0],
        "arb_residual_weights": lambda: M.arb_residual_weights(X0, Xt, zeta=0.5),
        "full_arb_estimate": lambda: M.arb_estimate(X0, Xt, y0),
        "paper_cv_ridge": lambda: _s7_outcome_cv_fixed_folds(E, X0, y0, grid),
        "paper_author_cv_ridge": lambda: E.outcome_cv_fixed_folds_raw_grid(X0, y0),
        "ridge_plugin_weights": lambda: E.ridge_plugin_weights(Xc, shift, 1.0),
        "diagnostics": lambda: E.diagnostics(X0, Xt, uniform),
        "plugin_risk_proxy": lambda: E.fixed_weight_risk_proxy(X0, Xt, uniform, 1.0, 0.5),
        # The outcome-free path arrays used to sit on the prepared object;
        # they are now built by rcb.estimator.fit_rcb, whose settings here are
        # exactly the ones E.select_prepared_adjustment passes.
        "prepare_adjustment_paths": lambda: {
            "qhat_path": uniform_fit.qhat_path,
            "norm_squared_path": uniform_fit.norm_squared_path,
            "weights_path": uniform_fit.weights_path,
        },
        "estimate_risk_nuisance": lambda: nuisance,
        "select_prepared_adjustment": lambda: {
            key: value
            for key, value in selected.items()
            if not isinstance(value, str)
        },
    }


def _s6_two_moment_pilot(R, X0, y0):
    """The retired ``variance_component_estimates``: two-moment pilot values."""
    from rcb.balancing import design_eigendecomposition
    from rcb.risk import estimate_variance_components

    def thunk():
        _, _, diagnostics = estimate_variance_components(
            X0, y0, *design_eigendecomposition(X0)[2:],
            theta_bounds=R.SQ_THETA_BOUNDS,
        )
        return (
            float(diagnostics["pilot_r2"][0]),
            float(diagnostics["pilot_sigma2"][0]),
        )
    return thunk


def _s7_family(M, X0, Xt, name, **overrides):
    """One registry family under the LaLonde study's settings: ``(weights, diagnostics)``."""
    import dataclasses

    from rcb.benchmarks.bases import base_weight_families

    settings = M.BASE_SETTINGS
    if overrides:
        settings = dataclasses.replace(settings, **overrides)
    weights, diagnostics = base_weight_families([name], X0, Xt, settings)
    return weights[name], diagnostics[name]


def _s7_propensity(M, X0, Xt):
    """The three propensity families as the retired wrapper returned them."""
    from rcb.benchmarks.bases import base_weight_families

    weights, diagnostics = base_weight_families(
        ["ipw", "trimmed_ipw", "overlap"], X0, Xt, M.BASE_SETTINGS
    )
    return (
        {
            "IPW": weights["ipw"], "Trimmed IPW": weights["trimmed_ipw"],
            "Overlap weights": weights["overlap"],
        },
        diagnostics["ipw"],
    )


def _s7_outcome_cv_fixed_folds(E, X0, y0, grid):
    return E.outcome_cv_fixed_folds(X0, y0, grid)


def main() -> None:
    for name, builder in (
        ("s5", section5_cases),
        ("s6", section6_cases),
        ("s7", section7_cases),
    ):
        recorded = _record(builder())
        destination = HERE / f"baseline_{name}.npz"
        np.savez_compressed(destination, **recorded)
        print(f"{destination.name}: {len(recorded)} arrays")


if __name__ == "__main__":
    main()
