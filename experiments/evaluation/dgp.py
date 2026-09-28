"""Section 6: evaluation of regularized balancing estimators.

The design shares Section 5's Gaussian AR(1) source covariance,
``(Sigma0)_{jk} = rho_AR**|j-k|`` with ``Sigma1 = Sigma0``, and its
``r^2=1``, ``sigma_0^2=0.5`` outcome model, so the two sections' numbers are
directly comparable.  The mean shift is diffuse and Mahalanobis-normalized
(``rcb.dgp.mahalanobis_rescale`` applied to the diffuse eigenbasis direction),
so that ``nu_delta^T Sigma0^{-1} nu_delta = delta**2`` holds exactly under the
AR(1) covariance, preserving the population effective-sample-size fraction
``exp(-delta**2)`` this section reports for each overlap regime, and so that
``G_{delta,p} = H_{0,p}`` -- varying overlap severity is not confounded with an
arbitrarily favorable or unfavorable shift orientation.  A fixed 41-point
penalty grid is used throughout.  Eight outcome-free base weighting families
-- uniform, IPW, stabilized entropy balancing, SBW, CBPS, overlap weights, the
l2 balancing component of double-ridge ABW at its design-only Riesz-CV
penalty, and the residual-balancing component of ARB -- are each augmented by
the proposed risk-calibrated ridge correction, and the augmented estimator is
scored against its own base, against the complete two-stage ABW and ARB
estimators built on the last two bases, and against four alternative tuning
rules.

Exactly one rule is the method's: the target-aware criterion with spectral
quasi-score nuisance estimates, reported as stage ``"Augmented (OOD)"``.  The
two-moment variant is a diagnostic, source GCV is the shift-blind baseline,
fixed lambda is a reference point, and the oracle uses the true variance
components and therefore evaluates only -- it never selects.

The estimator, the risk formulas and the competing weight families live in the
shared ``rcb`` package; what is here is this section's data-generating process
and its evaluation protocol.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from rcb import paths  # noqa: E402
from rcb.benchmarks import weights as W  # noqa: E402
from rcb.benchmarks.double_ridge import (  # noqa: E402
    double_ridge_estimate,
    design_only_balance_penalty_cv,
    outcome_cv_shuffled_kfold,
)
from rcb.benchmarks.arb import arb_elastic_net_fit  # noqa: E402
from rcb.benchmarks.bases import (  # noqa: E402
    DISPLAY_NAMES,
    BaseSettings,
    base_weight_families,
)
from rcb.config import replication_rng  # noqa: E402
from rcb.dgp import (  # noqa: E402
    ar1_covariance,
    mahalanobis_rescale,
    spectral_shift_direction,
    symmetric_sqrt,
)
from rcb.diagnostics import (  # noqa: E402
    normalize_weights as _normalize_weights,
    standardized_balance_problem,
)
from rcb.estimator import fit_rcb, reference_penalty_rules  # noqa: E402

STUDY = "evaluation"

SEED = 20260714
P = 50
PHI1 = 0.50
#: The joint (phi0, phi1) appendix panels (app-E7, app-E8) rerun the required-
#: metrics figure at this target aspect ratio instead of the default PHI1.
PHI1_SUPPLEMENT = 1.25
PHI0_SETTINGS = [0.50, 1.25]
N1 = int(round(P / PHI1))
PILOT_FRACTION = 0.50
#: Matches Section 5's AR(1) coefficient (experiments/simulation/dgp.py's
#: ExperimentConfig.rho_ar), so the two sections share one source covariance.
RHO_AR = 0.5
R2 = 1.0
SIGMA02 = 0.50
TAU = 1.0
LAMBDAS = np.logspace(-2, 2, 41)  # fixed compact subset of (0, infinity)
FIXED_LAMBDA = 1.0
TRIM_INTERVAL = (0.05, 0.95)
SBW_STANDARD_ERROR_MULTIPLIER = 3.0
ABW_OUTCOME_CV_REPEATS = 5
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
    "Uniform", "IPW", "Entropy balancing", "SBW", "CBPS",
    "Overlap weights", "L2 balancing", "ARB",
]
HEADLINE_STAGES = ["Base", "Augmented (OOD)"]
#: Figure labels, imported by ``2_evaluation.ipynb`` rather than redefined
#: there: the full names and the short tick labels the manuscript uses.
FAMILY_LABELS = {
    "Uniform": "Uniform", "IPW": "IPW", "Entropy balancing": "Entropy balancing",
    "SBW": "SBW", "CBPS": "CBPS", "Overlap weights": "Overlap weights",
    "L2 balancing": r"$\ell_2$ balancing", "ARB": "ARB",
}
FAMILY_SHORT_LABELS = {
    **FAMILY_LABELS, "Entropy balancing": "SEB", "Overlap weights": "OW",
    "L2 balancing": r"$\ell_2$",
}

#: Source/target covariance eigensystem, shared with Section 5 (Sigma1 = Sigma0).
SIGMA0_EIGENVALUES, SIGMA0_EIGENVECTORS = ar1_covariance(P, RHO_AR)
SIGMA0_SQRT = symmetric_sqrt(SIGMA0_EIGENVALUES, SIGMA0_EIGENVECTORS)
#: Diffuse shift direction, Mahalanobis-rescaled to unit Mahalanobis norm so
#: that ``a * SHIFT_DIRECTION`` has ``nu^T Sigma0^{-1} nu = a**2`` for any
#: overlap parameter ``a`` -- the isotropic ESS identity ``exp(-a**2)``
#: therefore carries over unchanged under the AR(1) covariance.
SHIFT_DIRECTION = mahalanobis_rescale(
    1.0,
    spectral_shift_direction(SIGMA0_EIGENVECTORS, "diffuse"),
    SIGMA0_EIGENVALUES,
    SIGMA0_EIGENVECTORS,
)

#: Appendix isotropic sensitivity (Sigma0=I_p, used before this revision): the
#: Mahalanobis-normalized diffuse shift above specializes to the familiar
#: coordinate all-ones direction, since every eigenvalue equals 1 and every
#: unit vector already has Mahalanobis norm 1 under the identity covariance.
ISOTROPIC_SHIFT_DIRECTION = np.ones(P) / np.sqrt(P)

ARTIFACT_DIR = paths.results(STUDY, create=False)
FIGURE_DIR = paths.figures(STUDY, create=False)

# ---------------------------------------------------------------------------
# Fixed response surfaces, for the response-model robustness appendix
#
# The sweep above draws the outcome from the isotropic random-effects law the
# theory assumes, so the tuning rule is scored under its own generative model.
# These four surfaces replace that law by a fixed, non-random regression
# function -- three linear ones aligned with different parts of the source
# spectrum, and one with a quadratic component the linear model cannot
# represent -- while every other part of the pipeline stays as it is.
#
# ``rcb.dgp.ar1_covariance`` returns ASCENDING eigenvalues, so the largest and
# smallest sit at the last and first index; the appendix states its own
# descending convention and writes these as ``(s_1,w_1)`` and ``(s_p,w_p)``.
# The names here are by role, not by index, so the two cannot be confused.
# ---------------------------------------------------------------------------
#: The overparameterized cell the robustness experiment uses: p=50 over n0=40.
RESPONSE_PHI0 = 1.25
#: First five coordinates, equally weighted: the sparse surface's direction.
RESPONSE_SPARSE_SUPPORT = 5
#: Paired bootstrap for the RMSE ratios, as in experiments/lalonde/chi2_stress.
RESPONSE_BOOTSTRAP_DRAWS = 5000
RESPONSE_BOOTSTRAP_SEED = SEED + 6

S_HIGH = float(SIGMA0_EIGENVALUES[-1])
W_HIGH = SIGMA0_EIGENVECTORS[:, -1]
S_LOW = float(SIGMA0_EIGENVALUES[0])
W_LOW = SIGMA0_EIGENVECTORS[:, 0]


def sigma0_quadratic_form(u: np.ndarray) -> float:
    """``u^T Sigma0 u`` from the stored eigensystem, without forming Sigma0.

    ``.dot`` rather than ``@`` throughout this block, for the reason
    :func:`rcb.dgp.symmetric_sqrt` documents: on an Accelerate-backed BLAS
    ``@`` raises spurious floating-point ``RuntimeWarning``s on these shapes
    despite a correct, finite result.
    """
    projection = SIGMA0_EIGENVECTORS.T.dot(u)
    return float(projection.dot(SIGMA0_EIGENVALUES * projection))


_SPARSE_DIRECTION = np.zeros(P)
_SPARSE_DIRECTION[:RESPONSE_SPARSE_SUPPORT] = 1.0 / np.sqrt(RESPONSE_SPARSE_SUPPORT)

#: Each is Sigma0-normalized, so ``var_{P0}(x^T beta) = 1`` matches ``R2``.
BETA_HIGH = W_HIGH / np.sqrt(S_HIGH)
BETA_LOW = W_LOW / np.sqrt(S_LOW)
BETA_SPARSE = _SPARSE_DIRECTION / np.sqrt(sigma0_quadratic_form(_SPARSE_DIRECTION))


@dataclass(frozen=True)
class ResponseSurface:
    """One fixed regression function and its known population target mean.

    ``evaluate`` is ``m`` applied rowwise to a design matrix.  ``population_
    mean`` is ``mu_m = E_{P1} m(x)`` in closed form, which is available because
    the target law is ``N(nu, Sigma0)`` with ``Sigma1 = Sigma0``; it is what the
    experiment scores against, so it is computed rather than simulated.
    ``source_signal_variance`` is ``var_{P0}{m(x)}``, held at one across the
    four surfaces so that none of them is an easier or harder signal than the
    random-effects outcome model the sweep above uses.
    """

    name: str
    evaluate: Callable[[np.ndarray], np.ndarray]
    population_mean: Callable[[np.ndarray], float]
    source_signal_variance: float


def _linear_surface(name: str, beta: np.ndarray) -> ResponseSurface:
    return ResponseSurface(
        name=name,
        evaluate=lambda X, beta=beta: X.dot(beta),
        population_mean=lambda nu, beta=beta: float(nu.dot(beta)),
        source_signal_variance=sigma0_quadratic_form(beta),
    )


def _nonlinear_evaluate(X: np.ndarray) -> np.ndarray:
    # The quadratic part is centered at its source mean and scaled by its own
    # source standard deviation sqrt(2)*s_high, so the two halves contribute
    # equally and the 2^{-1/2} in front returns the total to unit variance.
    quadratic = (X.dot(W_HIGH) ** 2 - S_HIGH) / (np.sqrt(2.0) * S_HIGH)
    return (X.dot(BETA_SPARSE) + quadratic) / np.sqrt(2.0)


def _nonlinear_population_mean(nu: np.ndarray) -> float:
    # E{(w^T x)^2} = (w^T nu)^2 + w^T Sigma0 w = (w^T nu)^2 + s_high under
    # Sigma1 = Sigma0, so the centering constant cancels exactly.
    return float(
        (nu.dot(BETA_SPARSE) + W_HIGH.dot(nu) ** 2 / (np.sqrt(2.0) * S_HIGH))
        / np.sqrt(2.0)
    )


#: The four surfaces, in the order the appendix figure's rows read.
RESPONSE_SURFACES = (
    _linear_surface("High spectrum", BETA_HIGH),
    _linear_surface("Low spectrum", BETA_LOW),
    _linear_surface("Sparse", BETA_SPARSE),
    ResponseSurface(
        name="Nonlinear",
        evaluate=_nonlinear_evaluate,
        population_mean=_nonlinear_population_mean,
        # var = (1/2){var(x^T beta_S) + var[(w^T x)^2]/(2 s_high^2)} = 1; the
        # cross term vanishes because a centered Gaussian has no third moment.
        source_signal_variance=1.0,
    ),
)
RESPONSE_SURFACE_ORDER = [surface.name for surface in RESPONSE_SURFACES]
#: Comparable signal for every surface is the point of the normalizations
#: above, so it is asserted here rather than left to the reader to check.
assert all(
    abs(surface.source_signal_variance - R2) < 1e-12
    for surface in RESPONSE_SURFACES
)



# ---------------------------------------------------------------------------
# This section's numerical policy on the shared estimators
#
# Monte Carlo sweeps run thousands of cells, so a degenerate weight vector is
# repaired rather than raised on -- the opposite of the real-data study's call,
# which reports a single estimate where a silent fallback would be a fabricated
# number.  The solver budgets are likewise this section's.  Every one of them
# is a field of BASE_SETTINGS, which ``rcb.benchmarks.bases`` reads; the base
# families themselves are built there, not here.
# ---------------------------------------------------------------------------
BASE_SETTINGS = BaseSettings(
    seed=SEED,
    trim_interval=TRIM_INTERVAL,
    on_invalid="repair",
    # Entropy balancing: a 250-iteration budget that must formally converge.
    entropy_maxiter=250, entropy_ftol=1e-11, entropy_gtol=1e-7,
    entropy_require_success=True, entropy_scale_floor=1e-8,
    # SBW: the outcome-free sampling-error tolerance 3*sqrt(1/n0 + 1/n1_pilot).
    sbw_tolerance=None, sbw_tolerance_multiplier=SBW_STANDARD_ERROR_MULTIPLIER,
    # CBPS: no silent fallback is permitted; the solve must formally converge.
    # ``max_nfev`` is 1000, not 400: the just-identified balancing condition is
    # not the gradient of a convex objective the way the (mislabeled) logistic
    # score it replaced was, so it needs more evaluations to reach a stationary
    # point.  One cell of the full production grid (``phi1=1.25`` supplement,
    # strong overlap, ``phi0=0.5``, repetition 9) needed between 500 and 800
    # evaluations; 1000 was chosen for headroom rather than trimmed to it.
    cbps_max_nfev=1000, cbps_tol=1e-9, cbps_moment_tolerance=None,
    cbps_on_invalid="repair",
    # l2 balancing: the design-only Riesz-CV penalty of the ABW authors' code,
    # which sees only the source design and the pilot-fold shift; source
    # eigenvalues are not floored at zero in this section.
    l2_penalty=None, l2_clip_eigenvalues=False,
    # ARB's residual-balancing program, as published.
    arb_zeta=0.5, arb_scale_floor=1e-8,
)

#: The families in the order the results files list them.  The last two are
#: the weighting components of the complete two-stage comparators.
BASE_FAMILY_NAMES = (
    "uniform", "ipw", "entropy", "sbw", "cbps", "overlap", "l2", "arb",
)
assert FAMILY_ORDER == [DISPLAY_NAMES[name] for name in BASE_FAMILY_NAMES]


# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------
def target_mean(a: float) -> np.ndarray:
    return a * SHIFT_DIRECTION


def draw_replication(
    repetition: int, phi0: float, a: float, phi1: float = PHI1,
    *, isotropic: bool = False,
) -> dict:
    # phi1 only joins the seed coordinates when it departs from the default,
    # so the default (phi1=PHI1) call reproduces the original single-stream
    # draw exactly; the phi1=PHI1_SUPPLEMENT appendix sweep gets its own,
    # distinguishable stream instead of colliding with it. ``isotropic`` does
    # NOT join the seed coordinates: the appendix sensitivity sweep is meant
    # to reuse the same underlying standard-Gaussian draws as the AR(1)
    # primary sweep (common random numbers), differing only in the
    # covariance transform applied to them below, for the cleanest possible
    # isotropic-vs-AR(1) contrast.
    coordinates = (repetition, int(100 * phi0), int(1000 * a))
    if phi1 != PHI1:
        coordinates = coordinates + (int(100 * phi1),)
    rng = replication_rng(SEED, *coordinates)
    n0 = int(round(P / phi0))
    n1 = int(round(P / phi1))
    if isotropic:
        nu1 = a * ISOTROPIC_SHIFT_DIRECTION
        X0 = rng.normal(size=(n0, P))
        X1 = nu1 + rng.normal(size=(n1, P))
    else:
        nu1 = target_mean(a)
        X0 = rng.normal(size=(n0, P)).dot(SIGMA0_SQRT)
        X1 = nu1 + rng.normal(size=(n1, P)).dot(SIGMA0_SQRT)

    order = rng.permutation(n1)
    n_pilot = int(round(PILOT_FRACTION * n1))
    I_P = np.sort(order[:n_pilot])
    I_E = np.sort(order[n_pilot:])

    beta = np.sqrt(R2 / P) * rng.normal(size=P)
    epsilon0 = rng.normal(scale=np.sqrt(SIGMA02), size=n0)
    epsilon1 = rng.normal(scale=np.sqrt(SIGMA02), size=n1)
    Y0 = X0 @ beta + epsilon0
    Y1 = X1 @ beta + TAU + epsilon1
    return {
        "X0": X0, "X1": X1, "X1P": X1[I_P], "X1E": X1[I_E],
        "Y0": Y0, "Y1": Y1, "beta": beta, "nu1": nu1,
        "I_P": I_P, "I_E": I_E,
        "mu0_true": float(nu1 @ beta),
        "beta_shift_cos2": float(
            (beta @ nu1) ** 2 / max((beta @ beta) * (nu1 @ nu1), 1e-15)
        ),
    }


def all_base_weights(
    X0: np.ndarray, X1P: np.ndarray, diagnostics: dict | None = None
) -> dict:
    """Every outcome-free base family, built from the source and pilot fold.

    The families and their numerical policy are :mod:`rcb.benchmarks.bases`
    with ``BASE_SETTINGS``; the keys are the display labels of
    ``FAMILY_ORDER``.  The last two are the weighting components of the
    complete two-stage comparators: the ARB base is the very vector
    ``arb_estimate`` corrects with its elastic net, and the l2 base is the
    double-ridge balancing half at its design-only penalty.  ``diagnostics``,
    when supplied, receives each family's diagnostic record under its display
    label (the l2 family's selected penalty is its ``"delta"`` entry).
    """
    weights, records = base_weight_families(
        BASE_FAMILY_NAMES, X0, X1P, BASE_SETTINGS
    )
    if diagnostics is not None:
        diagnostics.update(
            {DISPLAY_NAMES[name]: records[name] for name in BASE_FAMILY_NAMES}
        )
    return {DISPLAY_NAMES[name]: weights[name] for name in BASE_FAMILY_NAMES}


# ---------------------------------------------------------------------------
# The augmented path and the five tuning rules scored against it
# ---------------------------------------------------------------------------
def augmentation_path(X0, X1E, y0, gamma_base, nu1):
    """Build the penalty path and locate each tuning rule's selected point.

    The estimator is :func:`rcb.estimator.fit_rcb` with this study's search
    box and grid; ``rood_sq`` is its criterion.  The comparison rules come from
    :func:`rcb.estimator.reference_penalty_rules`: ``exact`` uses the true
    components and exists only to define the oracle and to score how far each
    feasible rule lands from it; it never selects a reported estimate.
    """
    fit = fit_rcb(X0, y0, X1E, gamma_base, LAMBDAS, theta_bounds=SQ_THETA_BOUNDS)
    rules = reference_penalty_rules(
        fit, y0, fixed_lambda=FIXED_LAMBDA,
        nu1=nu1, true_r2=R2, true_sigma2=SIGMA02, X0=X0,
    )
    sq_diagnostics = fit.nuisance_diagnostics
    return {
        "gamma_path": fit.weights_path,
        "rood": fit.risk_path,
        "rood_sq": fit.risk_path,
        "rood_mom": rules["two_moment_risk_path"],
        "exact": rules["exact_total"],
        "qhat": fit.qhat_path,
        "ood_index": fit.selected_index,
        "ood_sq_index": fit.selected_index,
        "ood_mom_index": rules["two_moment"],
        "gcv_index": rules["gcv"],
        "oracle_index": rules["oracle"],
        "fixed_index": rules["fixed"],
        "rhat2": fit.r2,
        "sigmahat2": fit.sigma2,
        "rhat2_sq": fit.r2,
        "sigmahat2_sq": fit.sigma2,
        "rhat2_mom": float(sq_diagnostics["pilot_r2"][0]),
        "sigmahat2_mom": float(sq_diagnostics["pilot_sigma2"][0]),
        "rhohat_sq": float(sq_diagnostics["rhohat"][0]),
        "sq_boundary": bool(sq_diagnostics["boundary"][0]),
        "sq_objective_improvement": float(
            sq_diagnostics["objective_improvement"][0]
        ),
        "max_normalization_error": fit.normalization_error_along_path,
    }


def score_weights(family, stage, gamma, gamma_base, data, overlap, d2, phi0,
             repetition, selected_lambda, selected_rood):
    """Exact risk and estimand error for one selected weight vector."""
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


def arb_estimate(data):
    """Complete ARB: residual-balancing weights plus an elastic-net outcome fit."""
    X0, Y0 = data["X0"], data["Y0"]
    # The same registry call as the "ARB" base family, so the two are one vector.
    gamma_arb = base_weight_families(("arb",), X0, data["X1P"], BASE_SETTINGS)[0]["arb"]

    scaler, model, alpha_1se = arb_elastic_net_fit(X0, Y0, seed=SEED)
    fitted_source = model.predict(scaler.transform(X0))
    fitted_target = float(model.predict(scaler.transform(data["X1E"])).mean())
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
        "residual_imbalance_norm": float(np.linalg.norm(delta_eval)),
        "elastic_net_alpha_1se": alpha_1se,
        "fitted_target_component": fitted_target,
        "residual_correction": residual_correction,
        "lambda_OOD": np.nan,
        "uses_centered_ridge_augmentation": False,
    }


def full_double_ridge_abw(data, repetition: int) -> list[dict]:
    """Full double-ridge ABW under the original simulation tuning rules.

    The default estimator selects the outcome ridge penalty by the authors'
    shuffled five-fold MSE criterion and sets the balancing penalty equal to
    it.  The imbalance-CV and Riesz-CV variants are retained as diagnostics.
    All three use the full target covariate mean, as in the original ABW
    simulation, but no target outcomes.

    The rows carry ``family = "L2 balancing"`` because the estimator's base is
    the l2 balancing component that family reports alone and augmented: the
    implied weights are :func:`rcb.balancing.ridge_augmented_path` from
    ``w(delta)`` at ``lambda``, so the three markers on that row differ only in
    the penalty rule and in which target mean the shift is measured against.
    """
    X0, Y0 = data["X0"], data["Y0"]
    center = X0.mean(axis=0)
    X0c = X0 - center
    shift = data["X1"].mean(axis=0) - center
    lam, cv_diagnostics = outcome_cv_shuffled_kfold(
        X0, Y0, repeats=ABW_OUTCOME_CV_REPEATS,
        seed=SEED + repetition * ABW_OUTCOME_CV_REPEATS,
    )
    design_tuners = design_only_balance_penalty_cv(X0, shift)
    variants = [
        ("Full ABW (outcome CV, delta=lambda)", lam),
        ("Full ABW (imbalance CV)", design_tuners["delta_cv_imbalance"]),
        ("Full ABW (Riesz CV)", design_tuners["delta_cv_riesz"]),
    ]
    rows = []
    for stage, delta in variants:
        result = double_ridge_estimate(X0c, Y0, shift, delta, lam)
        mu0_hat = float(result["muhat0"])
        rows.append({
            "family": "L2 balancing",
            "stage": stage,
            "mu0_error": mu0_hat - data["mu0_true"],
            "att_error": float(data["Y1"].mean() - mu0_hat - TAU),
            "lambda_outcome": lam,
            "delta_balance": float(delta),
            "weight_sum": float(result["augmented_weights"].sum()),
            "gamma_norm": float(np.linalg.norm(result["augmented_weights"])),
            "negative_mass": float(
                np.maximum(-result["augmented_weights"], 0).sum()
            ),
            "residual_imbalance_norm": result["residual_imbalance_norm"],
            "linear_representation_error": result[
                "linear_representation_error"
            ],
            "formula_representation_error": result[
                "formula_representation_error"
            ],
            "outcome_cv_all_success": bool(
                np.all(cv_diagnostics["repeat_successes"])
            ),
            "outcome_cv_repeat_penalties": ";".join(
                f"{value:.12g}"
                for value in cv_diagnostics["repeat_penalties"]
            ),
            "uses_target_outcomes": False,
            "uses_centered_ridge_augmentation": True,
        })
    return rows


def sample_trimmed_ipw_estimate(data) -> dict:
    """IPW after removing source and target observations outside overlap.

    The propensity model is fit on source controls and the honest pilot target
    split.  Its fixed rule is then applied to the source and the full target
    sample.  The reported error is for the retained target-sample estimand,
    which is distinct from the untrimmed ATT estimand.
    """
    model = W.fit_propensity_model(data["X0"], data["X1P"], seed=SEED)
    source_propensity = W.predict_propensity(model, data["X0"])
    target_propensity = W.predict_propensity(model, data["X1"])
    lower, upper = TRIM_INTERVAL
    source_retained = (
        (source_propensity >= lower) & (source_propensity <= upper)
    )
    target_retained = (
        (target_propensity >= lower) & (target_propensity <= upper)
    )
    if not source_retained.any() or not target_retained.any():
        raise RuntimeError("Sample trimming removed an entire study arm")
    odds = source_propensity[source_retained] / (
        1.0 - source_propensity[source_retained]
    )
    weights = _normalize_weights(odds, on_invalid="raise")
    mu0_hat = float(weights @ data["Y0"][source_retained])
    target_mean = float(data["Y1"][target_retained].mean())
    retained_mu0_truth = float(
        data["X1"][target_retained].mean(axis=0) @ data["beta"]
    )
    return {
        "family": "Sample-trimmed IPW",
        "stage": "Matched-overlap sensitivity",
        "mu0_error": mu0_hat - retained_mu0_truth,
        "att_error": target_mean - mu0_hat - TAU,
        "source_retained": int(source_retained.sum()),
        "target_retained": int(target_retained.sum()),
        "source_retained_fraction": float(source_retained.mean()),
        "target_retained_fraction": float(target_retained.mean()),
        "weight_sum": float(weights.sum()),
        "ess": float(1.0 / (weights @ weights)),
        "estimand": "ATT in retained target sample",
        "trim_lower": lower,
        "trim_upper": upper,
    }
