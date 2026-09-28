"""Shared implementation behind the manuscript's Sections 5, 6 and 7.

The three empirical sections study the same estimator from three angles --
Section 5 on the risk theory, Section 6 against competing balancing estimators,
Section 7 on the LaLonde data -- and each used to carry its own copy of the
machinery.  This package holds the single implementation they now share:

``rcb.estimator``
    Start here.  ``fit_rcb`` runs the proposed estimator end to end on any
    normalized base weight vector, and ``reference_penalty_rules`` gives the
    comparison rules the studies report next to it.

``rcb.balancing``
    The proposed method's pieces: design geometry, base weights, the
    ridge-augmented path, risk-based penalty selection, and the honest
    pilot/evaluation split.

``rcb.risk``
    The deterministic equivalent, the exact conditional risk, and the feasible
    plug-in risk estimate, kept strictly apart.

``rcb.nuisance``
    Spectral quasi-score estimation of the two source variance components
    ``(r^2, sigma_0^2)`` from the centered source design and outcomes.

``rcb.benchmarks``
    The methods being compared against: competing weight families, double ridge
    and its tuning rules, approximate residual balancing, source GCV.

``rcb.dgp``
    The shared Gaussian AR(1) source covariance and eigenbasis-relative shift
    directions Sections 5 and 6 both draw their covariates from.

``rcb.diagnostics``, ``rcb.montecarlo``, ``rcb.config``, ``rcb.io``,
``rcb.paths``, ``rcb.style``
    Weight and balance summaries; Monte Carlo summaries and resampling; seeding
    and grids; artifact writing; output locations; figure style.

Sections reach this package through the ``code/`` directory already on their
import path, so ``import rcb`` needs no installation step.

Terminology is kept distinct throughout, matching the manuscript and the
section notebooks: *theoretical* means the finite-dimensional deterministic
equivalent, *exact* means the exact conditional risk for a fixed design, and
*risk estimate* means the feasible plug-in risk under repeated outcomes.
"""

__all__ = [
    "balancing",
    "benchmarks",
    "config",
    "dgp",
    "diagnostics",
    "estimator",
    "io",
    "montecarlo",
    "nuisance",
    "paths",
    "risk",
    "style",
]
