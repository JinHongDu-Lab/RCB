"""Estimators for the LaLonde (1986) / NSW real-data application.

The proposed estimator, the risk formulas and the comparison methods live in
the shared ``rcb`` package; this module supplies what is specific to the
application:

* this study's numerical policy on the shared estimators (``BASE_SETTINGS``),
  which refuses to repair a degenerate solve rather than falling back silently;
* the base families the exploratory pipeline and the R1--R9 driver use, built
  through ``rcb.benchmarks.bases`` with that policy, and the full ARB
  comparator.

Notation follows the manuscript: source = controls (arm a = 0), target =
treated (a = 1); the counterfactual mean is mu0 = E{y(0) | a = 1} and the ATT
is ybar1 - muhat0.  On the observational data the experimental NSW estimate
(~$1,794 on the DW subsample) is the ground truth, so "better" means closer
to it.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[2]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from rcb import paths  # noqa: E402
from rcb.benchmarks.arb import arb_estimate as _full_arb_estimate  # noqa: E402
from rcb.benchmarks.bases import BaseSettings, base_weight_families  # noqa: E402
from rcb.benchmarks.double_ridge import (  # noqa: E402
    double_ridge_estimate,
    ols_plugin_estimate,
    outcome_ridge_slope,
    design_only_balance_penalty_cv,
)
from rcb.benchmarks.weights import (  # noqa: E402
    l2_balancing_weights,
    l2_residual_imbalance,
)
from rcb.diagnostics import (  # noqa: E402
    effective_sample_size,
    negative_weight_fraction,
    normalize_weights as _normalize_weights,
    standardize,
    weighted_smd,
)
DEFAULT_SEED = 20260806
LAMBDA_GRID = np.logspace(-4, 3, 141)
SQ_BOUNDS = ((1e-6, 25.0), (1e-6, 10.0))

# Dehejia-Wahba experimental benchmark ATT on their subsample (1978 dollars).
EXPERIMENTAL_ATT = 1794.3

DATA_DIR = paths.data("lalonde")


# ---------------------------------------------------------------------------
# This study's numerical policy on the shared estimators
#
# This study refuses to normalize degenerate weights rather than repairing
# them: it reports one real-data estimate, where a silent uniform fallback would
# be a fabricated number.  The evaluation study makes the opposite call for its
# Monte Carlo sweeps.  Every policy constant below is a field of BASE_SETTINGS,
# which ``rcb.benchmarks.bases`` reads; the base families are built there.
# ---------------------------------------------------------------------------
BASE_SETTINGS = BaseSettings(
    seed=DEFAULT_SEED,
    on_invalid="raise",
    # Entropy balancing: run L-BFGS-B to 5000 iterations at tight tolerances
    # and report the outcome rather than insisting on formal convergence.
    entropy_maxiter=5000, entropy_ftol=1e-14, entropy_gtol=1e-10,
    entropy_require_success=False, entropy_scale_floor=1e-10,
    # SBW: a fixed standardized balance tolerance of 0.5.
    sbw_tolerance=0.5, sbw_scale_floor=1e-8,
    # CBPS: a solve that did not formally converge is accepted when its largest
    # balance moment is below 1e-5.
    cbps_max_nfev=400, cbps_tol=1e-10, cbps_moment_tolerance=1e-5,
    cbps_on_invalid="raise",
    # l2 balancing: the fixed penalty 1.0 of the exploratory pipeline's ridge
    # base; source eigenvalues are floored at zero, which matters on this
    # rank-deficient 171-column design.
    l2_penalty=1.0, l2_clip_eigenvalues=True,
    # ARB's residual-balancing program, as published.
    arb_zeta=0.5, arb_scale_floor=1e-8,
)


def normalize_weights(raw: np.ndarray) -> np.ndarray:
    return _normalize_weights(raw, on_invalid="raise")


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return standardize(X, floor=1e-10)


# ---------------------------------------------------------------------------
# Base families, all from rcb.benchmarks.bases with this study's settings
# ---------------------------------------------------------------------------
def regularized_entropy_base(source_X, target_X):
    """Stabilized entropy balancing toward the target sample's mean."""
    weights, diagnostics = base_weight_families(
        ("entropy",), source_X, target_X, BASE_SETTINGS
    )
    return weights["entropy"], diagnostics["entropy"]


#: Registry name -> the label the exploratory pipeline's results files use.
#: The last label is historical: it names a fixed-penalty ridge base, and it
#: stays because it is a key in committed outputs.
EXPLORATORY_FAMILIES = {
    "uniform": "Uniform", "ipw": "IPW", "trimmed_ipw": "Trimmed IPW",
    "entropy": "Entropy balancing", "sbw": "SBW", "cbps": "CBPS",
    "overlap": "Overlap weights", "l2": "Bruns-Smith ABW",
}


def evaluation_style_base_weights(source_X, target_X, seed=DEFAULT_SEED):
    """The eight-family set the exploratory pipeline reports.

    Returns ``(weights, diagnostics)`` keyed as the results files expect.
    """
    settings = BASE_SETTINGS
    if seed != BASE_SETTINGS.seed:
        settings = dataclasses.replace(BASE_SETTINGS, seed=seed)
    weights, records = base_weight_families(
        tuple(EXPLORATORY_FAMILIES), source_X, target_X, settings
    )
    diagnostics = {
        "propensity": records["ipw"], "entropy": records["entropy"],
        "sbw": records["sbw"], "cbps": records["cbps"],
    }
    return (
        {label: weights[name] for name, label in EXPLORATORY_FAMILIES.items()},
        diagnostics,
    )


def arb_residual_weights(source_X, target_X, zeta=0.5):
    """Approximate residual-balancing weights used by the full ARB comparator."""
    settings = BASE_SETTINGS
    if zeta != BASE_SETTINGS.arb_zeta:
        settings = dataclasses.replace(BASE_SETTINGS, arb_zeta=zeta)
    weights, diagnostics = base_weight_families(
        ("arb",), source_X, target_X, settings
    )
    return weights["arb"], diagnostics["arb"]


def arb_estimate(
    source_X, target_X, source_y, residual_weights=None,
    seed=DEFAULT_SEED,
):
    """Complete ARB estimator."""
    return _full_arb_estimate(
        source_X, target_X, source_y, seed=seed,
        residual_weights=residual_weights,
    )
