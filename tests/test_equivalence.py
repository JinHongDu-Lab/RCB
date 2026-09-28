"""Assert the extracted package reproduces the pre-refactor numbers exactly.

Replays ``tests/baseline/generate.py``'s cases through ``rcb`` and compares each
result with the recorded baseline using ``assert_array_equal``.  The bar is
exact equality rather than ``allclose`` on purpose: moving arithmetic between
modules can reassociate floating-point operations, and this refactor is supposed
to move code without moving numbers.  A case that cannot clear that bar should
be reported with its difference measured, not relaxed to a tolerance.

Run with::

    python -m pytest code/tests/test_equivalence.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE / "baseline") not in sys.path:
    sys.path.insert(0, str(HERE / "baseline"))

import generate  # noqa: E402  (sys.path is set immediately above)

generate.MODE = "current"

SECTIONS = {
    "s5": generate.section5_cases,
    "s6": generate.section6_cases,
    "s7": generate.section7_cases,
}


def _load(section: str) -> dict:
    path = HERE / "baseline" / f"baseline_{section}.npz"
    if not path.exists():
        pytest.skip(f"{path.name} not recorded yet")
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def _current(section: str) -> dict:
    recorded: dict = {}
    for name, thunk in SECTIONS[section]().items():
        generate._flatten(name, thunk(), recorded)
    return recorded


# Names that must resolve into ``rcb`` once extracted.  Without this the
# equivalence test could pass while still calling the old implementation, which
# would make it worthless.  Extend as each module lands.
# The one place where extraction moved the numbers, listed explicitly rather
# than rebaselined away.
#
# Section 6 computed the ridge path, the plug-in risk geometry and the exact
# risk with hand-written per-penalty loops, specialized to its isotropic design.
# Those now call the shared vectorized implementations, which perform the same
# arithmetic in a different order -- matrix-matrix products where there were
# matrix-vector products, and one whole-array reduction where there was a loop
# over columns.  Floating-point addition is not associative, so the last bits
# move.
#
# Measured across the recorded cases: largest absolute difference anywhere
# 7.1e-15; largest relative difference 1.1e-15, except the weight path at
# 7.4e-13 on entries of magnitude around 1e-16, where the path values
# themselves are around 1e-2.  Every selected penalty index is unchanged, which
# is what ``test_reassociated_keys_are_not_decisive`` pins down.
#
# The second such site: deterministic_equivalent_diagonal used to bisect one
# penalty at a time and accumulate the eta=1 target-kernel terms with +=.  It
# now bisects the whole grid at once, which is 4.8x faster and reassociates
# that accumulation.  Measured: exactly 1 ULP on the signal term, largest
# absolute difference 4.4e-16, largest relative 2.1e-16; the residual term and
# every eta=0 case are bit-identical, as is the identity closed form.
#
# The third: Section 7's prepare_adjustment_paths computed its qhat_path from
# an independent derivation of the same plug-in risk geometry (target moments
# via a cached covariance-eigenbasis product, rather than the target sample
# rotated through the eigenbasis directly).  It now calls
# rcb.risk.feasible_risk_geometry, the same formula Sections 5 and 6 use, so
# the two computed target-variance quantities agree only up to reassociation.
# Measured: largest absolute difference 6.9e-18, largest relative 3.8e-16;
# selected_index is bit-identical (still checked below, exact).
#
# The fourth: bruns_smith_ridge_balance_weights (now l2_balancing_weights_spectral) re-derived its own centering,
# eigendecomposition and ridge direction instead of calling
# rcb.balancing.design_eigendecomposition and ridge_augmented_path -- the same
# ridge solve the proposed estimator's own path uses.  It now calls them (at
# the single penalty alpha, from a uniform base).  Measured: largest absolute
# difference 8.7e-18, largest relative 4.8e-14 (on entries near 1e-4 that
# nearly cancel against the uniform base).
#
# Everything else in all three sections is still held to exact equality.
REASSOCIATED = {
    "s5.deterministic_equivalent_diagonal_eta1.0",
    "s5.deterministic_equivalent_diagonal_eta1.2",
    "s5.deterministic_equivalent_diagonal_structured.0",
    "s5.deterministic_equivalent_diagonal_structured.2",
    "s6.augmentation_path.exact",
    "s6.augmentation_path.gamma_path",
    "s6.augmentation_path.qhat",
    "s6.augmentation_path.rood",
    "s6.augmentation_path.rood_mom",
    "s6.augmentation_path.rood_sq",
    "s6.evaluate.att_error",
    "s6.evaluate.delta_norm",
    "s6.evaluate.distance_from_base",
    "s6.evaluate.ess",
    "s6.evaluate.exact_risk",
    "s6.evaluate.mu0_error",
    "s6.evaluate.negative_mass",
    "s6.evaluate.noise_risk",
    "s6.evaluate.weight_sum",
    "s7.prepare_adjustment_paths.qhat_path",
    "s7.select_prepared_adjustment.risk",
    "s7.select_prepared_adjustment.qhat_path",
    "s6.bruns_smith_ridge_balance_weights",
    "s6.all_base_weights.Bruns-Smith ABW",
    "s7.bruns_smith_ridge_balance_base",
}

# Deliberate benchmark-definition corrections are exempt from the historical
# pre-refactor values, but only at the explicitly listed outputs below.  Their
# defining constraints and invariants are pinned in
# ``test_benchmark_corrections.py``.  In Section 6, the fixed ridge-balancing
# component and source-only pseudo-trimming leave the primary base-family
# sweep; complete double-ridge ABW and matched-sample trimming are tested and
# reported separately.  In Sections 6 and 7, SBW changes from penalized NNLS
# to the canonical constrained dispersion problem.
#
# Two base families joined the Section 6 sweep after the baseline was recorded
# and so have no recorded value to compare against: the l2 balancing component
# of double-ridge ABW at its design-only Riesz-CV penalty (replacing the fixed
# alpha=1 version removed above) and the residual-balancing component of ARB
# (the same vector ``arb_residual_weights`` already records).  They are listed
# separately from the corrections because nothing about them is a correction;
# their invariants are likewise pinned in ``test_benchmark_corrections.py``.
ADDED_SINCE_BASELINE = {
    "s6.all_base_weights.L2 balancing",
    "s6.all_base_weights.ARB",
}
# On the recorded fixtures, the maximum absolute weight changes are 0.0522 for
# Section 6 SBW, 0.0818 for Section 7 SBW, 0.00900 and 0.285 for the former
# source-only trimmed-IPW vectors, and 2.78e-17 for the removal of ARB's
# post-solve renormalization.  These are definition changes, not reassociation.
INTENTIONAL_CORRECTIONS = {
    "s6.all_base_weights.Bruns-Smith ABW",
    "s6.all_base_weights.Trimmed IPW",
    "s6.all_base_weights.SBW",
    "s6.propensity_weights.Trimmed IPW",
    "s6.sbw_weights",
    "s6.arb_residual_weights",
    "s6.full_arb_estimate.mu0_error",
    "s6.full_arb_estimate.signal_risk",
    "s6.full_arb_estimate.noise_risk",
    "s6.full_arb_estimate.gamma_norm",
    "s6.full_arb_estimate.delta_norm",
    "s6.full_arb_estimate.residual_weight_norm",
    "s6.full_arb_estimate.residual_imbalance_norm",
    "s6.full_arb_estimate.residual_correction",
    "s7.evaluation_propensity_bases.0.Trimmed IPW",
    "s7.double_ridge_estimate.formula_representation_error",
    "s7.evaluation_propensity_bases.1.retained_source_count",
    "s7.evaluation_propensity_bases.1.retained_source_fraction",
    "s7.evaluation_propensity_bases.1.trimmed_source_count",
    "s7.evaluation_sbw_base.0",
    "s7.evaluation_sbw_base.1.balance_tolerance",
    "s7.evaluation_sbw_base.1.iterations",
    "s7.evaluation_sbw_base.1.max_abs_standardized_imbalance",
    "s7.evaluation_sbw_base.1.max_balance_constraint_violation",
    "s7.evaluation_sbw_base.1.minimum_weight",
    "s7.evaluation_sbw_base.1.nnls_residual_norm",
    "s7.evaluation_sbw_base.1.normalization_error",
    "s7.evaluation_sbw_base.1.objective",
    "s7.evaluation_sbw_base.1.success",
    "s7.arb_residual_weights.0",
    "s7.arb_residual_weights.1.cap_violation",
    "s7.arb_residual_weights.1.max_balance_constraint_violation",
    "s7.arb_residual_weights.1.minimum_weight",
    "s7.arb_residual_weights.1.normalization_error",
    "s7.full_arb_estimate.muhat0",
    "s7.full_arb_estimate.residual_correction",
    "s7.full_arb_estimate.weights",
    "s7.full_arb_estimate.weight_diagnostics.cap_violation",
    "s7.full_arb_estimate.weight_diagnostics.max_balance_constraint_violation",
    "s7.full_arb_estimate.weight_diagnostics.minimum_weight",
    "s7.full_arb_estimate.weight_diagnostics.normalization_error",
}

EXTRACTED_INTO_RCB = [
    "design_eigendecomposition",
    "deterministic_equivalent_diagonal",
    "deterministic_equivalent_isotropic",
    "exact_risk_components",
    "design_independent_base_weights",
    "feasible_risk_geometry",
    "pilot_ridge_base_weights",
    "ridge_augmented_path",
]


@pytest.mark.parametrize("name", EXTRACTED_INTO_RCB)
def test_section5_cases_call_the_package(name: str) -> None:
    namespace = generate._section5_namespace()
    module = getattr(namespace, name).__module__
    assert module.startswith("rcb."), f"{name} still resolves to {module}"


def test_section7_calls_the_package() -> None:
    from experiments.lalonde import experiment as E

    assert E.prepare_adjustment_paths.__doc__ is not None
    assert "prepare_geometry" in E.prepare_adjustment_paths.__code__.co_names, (
        "Section 7 no longer takes its design geometry from rcb.estimator"
    )
    assert "fit_rcb" in E.select_prepared_adjustment.__code__.co_names, (
        "Section 7 no longer delegates the estimator to rcb.estimator.fit_rcb"
    )
    assert "risk" in E.fixed_weight_risk_proxy.__code__.co_names


@pytest.mark.parametrize("section", sorted(SECTIONS))
def test_extracted_package_reproduces_baseline(section: str) -> None:
    expected = _load(section)
    actual = _current(section)

    exempt = INTENTIONAL_CORRECTIONS | ADDED_SINCE_BASELINE
    actual_unmodified = {
        key for key in actual if f"{section}.{key}" not in exempt
    }
    expected_unmodified = {
        key for key in expected if f"{section}.{key}" not in exempt
    }
    assert actual_unmodified == expected_unmodified, (
        f"case set changed for {section}: "
        f"added {sorted(actual_unmodified - expected_unmodified)}, "
        f"dropped {sorted(expected_unmodified - actual_unmodified)}"
    )

    differing = []
    for key, reference in expected.items():
        full = f"{section}.{key}"
        if full in INTENTIONAL_CORRECTIONS:
            continue
        try:
            if full in REASSOCIATED:
                np.testing.assert_allclose(
                    actual[key], reference, rtol=1e-11, atol=1e-13
                )
            else:
                np.testing.assert_array_equal(actual[key], reference)
        except AssertionError:
            differing.append(key)

    assert not differing, (
        f"{len(differing)} of {len(expected)} recorded values changed in "
        f"{section}: {differing[:10]}"
    )


@pytest.mark.parametrize("section", sorted({key.split(".", 1)[0] for key in REASSOCIATED}))
def test_reassociated_keys_are_not_decisive(section: str) -> None:
    """No penalty selection may depend on the reassociated arithmetic.

    A last-bit difference in a risk curve is harmless; a last-bit difference
    that flips an ``argmin`` is not, because it changes which penalty gets
    reported.  This pins the selection indices to exact equality.
    """
    expected = _load(section)
    actual = _current(section)
    decisive = [k for k in expected if k.endswith("_index") or "boundary" in k]
    assert decisive, "no selection indices recorded"
    for key in decisive:
        assert f"{section}.{key}" not in REASSOCIATED
        np.testing.assert_array_equal(actual[key], expected[key])
