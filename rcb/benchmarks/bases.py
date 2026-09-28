"""The base weighting rules the proposed estimator is applied on top of.

Every family here is outcome-free and normalized, built from the source design
``X0`` and the *pilot* fold of the target sample, so any of them can be handed
to :func:`rcb.estimator.fit_rcb` as its ``base_weights``.  The registry gives
each rule one short name and one implementation; the numerical policy that
used to be encoded in per-study wrapper functions -- solver budgets,
tolerances, trimming, failure handling -- lives in :class:`BaseSettings`, and
each study passes its own instance.  Two studies never share a default, because
a default shared across studies would silently move published numbers.

Names and what they are:

``uniform``
    the unweighted control mean;
``ipw``, ``trimmed_ipw``, ``overlap``
    the three families one regularized logistic source-versus-pilot classifier
    yields (:func:`rcb.benchmarks.weights.propensity_weight_families`);
``entropy``
    exponential tilting, stabilized by ``sqrt(log(p+1)/n_pilot)`` unless
    ``entropy_stabilized`` is off;
``sbw``
    stable balancing weights with an explicit standardized balance tolerance;
``cbps``
    the just-identified covariate-balancing propensity score;
``l2``
    the ridge (l2) balancing weights that form the balancing component of the
    double-ridge ABW estimator, at a fixed penalty or at the design-only
    Riesz-CV penalty of the ABW authors' code;
``arb``
    the capped residual-balancing weights of approximate residual balancing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import weights as W
from .arb import arb_residual_weights
from .double_ridge import design_only_balance_penalty_cv

__all__ = ["BaseSettings", "DISPLAY_NAMES", "FAMILIES", "base_weight_families"]

#: The positive part of the inherited ABW Riesz grid, ``{j/999: j=1..999}``.
#: The l2 base needs a strictly positive balancing penalty -- the ridge solve
#: is undefined at zero on a rank-deficient design, which the guard below
#: refuses -- so zero is not a candidate.  Dropping it changes no selection:
#: the argmin has never fallen on the zero endpoint in any recorded sweep.
RIESZ_POSITIVE_GRID = np.linspace(0.0, 1.0, 1000)[1:]

FAMILIES = (
    "uniform", "ipw", "trimmed_ipw", "overlap", "entropy", "sbw", "cbps", "l2",
    "arb",
)

#: The labels the studies' results files key their rows by.
DISPLAY_NAMES = {
    "uniform": "Uniform",
    "ipw": "IPW",
    "trimmed_ipw": "Trimmed IPW",
    "overlap": "Overlap weights",
    "entropy": "Entropy balancing",
    "sbw": "SBW",
    "cbps": "CBPS",
    "l2": "L2 balancing",
    "arb": "ARB",
}

_PROPENSITY_KEYS = {
    "ipw": "IPW", "trimmed_ipw": "Trimmed IPW", "overlap": "Overlap weights",
}


@dataclass(frozen=True)
class BaseSettings:
    """One study's numerical policy for every base family.

    Nothing here has a cross-study default that matters: the values below are
    the shared functions' own defaults, and each study overrides what it
    decided differently (documented at the study's instance).
    """

    seed: int
    #: Propensity families: trimming interval and what to do with a degenerate
    #: weight vector (``"raise"`` on real data, ``"repair"`` in Monte Carlo).
    trim_interval: tuple[float, float] = (0.05, 0.95)
    on_invalid: str = "raise"
    #: Entropy balancing: stabilization and L-BFGS-B budget.
    entropy_stabilized: bool = True
    entropy_maxiter: int = 5000
    entropy_ftol: float = 1e-14
    entropy_gtol: float = 1e-10
    entropy_require_success: bool = False
    entropy_scale_floor: float = 1e-10
    #: SBW: a fixed standardized tolerance, or ``None`` for the sampling-error
    #: rule ``multiplier * sqrt(1/n0 + 1/n_pilot)``.
    sbw_tolerance: float | None = None
    sbw_tolerance_multiplier: float = 3.0
    sbw_scale_floor: float = 1e-8
    #: CBPS: evaluation budget and failure policy.
    cbps_max_nfev: int = 400
    cbps_tol: float = 1e-10
    cbps_moment_tolerance: float | None = 1e-5
    cbps_on_invalid: str = "raise"
    #: l2 balancing: a fixed penalty, or ``None`` for the design-only Riesz-CV
    #: rule; whether to floor negative source eigenvalues at zero.
    l2_penalty: float | None = None
    l2_clip_eigenvalues: bool = True
    #: ARB's residual-balancing program.
    arb_zeta: float = 0.5
    arb_scale_floor: float = 1e-8


def base_weight_families(
    names: tuple[str, ...] | list[str],
    X0: np.ndarray,
    X1_pilot: np.ndarray,
    settings: BaseSettings,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Build the named base families from the source design and pilot fold.

    Returns ``(weights, diagnostics)``, both keyed by the registry name.  The
    propensity classifier is fitted once for however many of its three
    families were requested.
    """
    n0, p = X0.shape
    target = X1_pilot.mean(axis=0)
    weights: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict] = {}
    propensity = None

    for name in names:
        if name == "uniform":
            weights[name], diagnostics[name] = np.full(n0, 1.0 / n0), {}
        elif name in _PROPENSITY_KEYS:
            if propensity is None:
                propensity = W.propensity_weight_families(
                    X0, X1_pilot, seed=settings.seed,
                    trim=settings.trim_interval, on_invalid=settings.on_invalid,
                )
            weights[name] = propensity[0][_PROPENSITY_KEYS[name]]
            diagnostics[name] = propensity[1]
        elif name == "entropy":
            ridge = (
                W.entropy_stabilization_penalty(p, len(X1_pilot))
                if settings.entropy_stabilized else 0.0
            )
            weights[name], diagnostics[name] = W.entropy_balancing_weights(
                X0, target, ridge=ridge,
                maxiter=settings.entropy_maxiter, ftol=settings.entropy_ftol,
                gtol=settings.entropy_gtol,
                require_success=settings.entropy_require_success,
                scale_floor=settings.entropy_scale_floor,
            )
        elif name == "sbw":
            tolerance = (
                settings.sbw_tolerance
                if settings.sbw_tolerance is not None
                else settings.sbw_tolerance_multiplier
                * np.sqrt(1.0 / n0 + 1.0 / len(X1_pilot))
            )
            weights[name], diagnostics[name] = W.sbw_weights(
                X0, target, balance_tolerance=tolerance,
                scale_floor=settings.sbw_scale_floor,
            )
        elif name == "cbps":
            weights[name], diagnostics[name] = W.cbps_weights(
                X0, X1_pilot, max_nfev=settings.cbps_max_nfev,
                tol=settings.cbps_tol,
                moment_tolerance=settings.cbps_moment_tolerance,
                on_invalid=settings.cbps_on_invalid,
            )
        elif name == "l2":
            if settings.l2_penalty is None:
                delta = design_only_balance_penalty_cv(
                    X0, target - X0.mean(axis=0),
                    riesz_grid=RIESZ_POSITIVE_GRID,
                )["delta_cv_riesz"]
                if not np.isfinite(delta) or delta <= 0.0:
                    # On a rank-deficient design the ridge solve is undefined at
                    # zero; refusing is safer than flooring silently.
                    raise RuntimeError(
                        f"Riesz-CV balancing penalty is not positive: {delta}"
                    )
                rule = "riesz_cv"
            else:
                delta, rule = float(settings.l2_penalty), "fixed"
            weights[name] = W.l2_balancing_weights_spectral(
                X0, target, alpha=delta,
                clip_eigenvalues=settings.l2_clip_eigenvalues,
            )
            diagnostics[name] = {"delta": float(delta), "penalty_rule": rule}
        elif name == "arb":
            weights[name], diagnostics[name] = arb_residual_weights(
                X0, target, zeta=settings.arb_zeta,
                scale_floor=settings.arb_scale_floor,
                on_invalid=settings.on_invalid,
            )
        else:
            raise KeyError(f"unknown base family {name!r}; known: {FAMILIES}")
    return weights, diagnostics
