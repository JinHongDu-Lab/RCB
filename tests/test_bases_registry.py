"""``rcb.benchmarks.bases`` reproduces each study's base-weight wrappers exactly.

This pins bit-for-bit agreement between the registry, driven by each
study's ``BaseSettings``, and the per-study base-weight wrappers.
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

from rcb.benchmarks.bases import (  # noqa: E402
    DISPLAY_NAMES,
    FAMILIES,
    BaseSettings,
    base_weight_families,
)

equal = np.testing.assert_array_equal


def evaluation_settings():
    from experiments.evaluation import dgp

    return dgp.BASE_SETTINGS


def lalonde_settings():
    from experiments.lalonde import methods as M

    return BaseSettings(
        seed=M.DEFAULT_SEED, sbw_tolerance=0.5, l2_penalty=1.0,
    )


@pytest.mark.parametrize("repetition,phi0,a", [(0, 0.5, 1.0), (1, 1.25, np.sqrt(3.0))])
def test_registry_matches_evaluation_wrappers(repetition, phi0, a) -> None:
    from experiments.evaluation import dgp

    data = dgp.draw_replication(repetition, phi0, a)
    X0, X1P = data["X0"], data["X1P"]
    wrapper_diagnostics: dict = {}
    wrapped = dgp.all_base_weights(X0, X1P, diagnostics=wrapper_diagnostics)
    names = ["uniform", "ipw", "entropy", "sbw", "cbps", "overlap", "l2", "arb"]
    registry, diagnostics = base_weight_families(names, X0, X1P, evaluation_settings())
    assert [DISPLAY_NAMES[n] for n in names] == list(wrapped)
    for name in names:
        equal(registry[name], wrapped[DISPLAY_NAMES[name]])
    assert diagnostics["l2"]["delta"] == wrapper_diagnostics["L2 balancing"]["delta"]
    assert diagnostics["l2"]["penalty_rule"] == "riesz_cv"
    assert 0.0 < diagnostics["l2"]["delta"] <= 1.0


def test_registry_matches_lalonde_wrappers() -> None:
    import generate
    from experiments.lalonde import methods as M

    d = generate._synthetic_design(20260806, n0=140, nt=70, p=45)
    X0, Xt = d["X0"], d["Xt"]
    wrapped, _ = M.evaluation_style_base_weights(X0, Xt)
    legacy_names = {
        "uniform": "Uniform", "ipw": "IPW", "trimmed_ipw": "Trimmed IPW",
        "entropy": "Entropy balancing", "sbw": "SBW", "cbps": "CBPS",
        "overlap": "Overlap weights", "l2": "Bruns-Smith ABW",
    }
    registry, diagnostics = base_weight_families(
        list(legacy_names) + ["arb"], X0, Xt, lalonde_settings()
    )
    for name, legacy in legacy_names.items():
        equal(registry[name], wrapped[legacy])
    assert diagnostics["l2"] == {"delta": 1.0, "penalty_rule": "fixed"}
    arb_weights, arb_diagnostics = M.arb_residual_weights(X0, Xt)
    equal(registry["arb"], arb_weights)
    assert diagnostics["arb"] == arb_diagnostics


def test_registry_rejects_unknown_family() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(KeyError):
        base_weight_families(["nope"], rng.normal(size=(20, 3)), rng.normal(size=(10, 3)),
                             BaseSettings(seed=1))
    assert set(DISPLAY_NAMES) == set(FAMILIES)
