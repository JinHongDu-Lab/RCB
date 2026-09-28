"""Section 5: the inputs behind the manuscript figures.

The draws and summaries here are experiments -- they run the Monte Carlo and
write CSVs.  The plotting that used to sit beside them lives in
``1_simulation.ipynb``, which reads what this writes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from rcb.balancing import (
    design_eigendecomposition,
    design_independent_base_weights,
    ridge_augmented_path,
)
from rcb.dgp import ar1_covariance, symmetric_sqrt
from rcb.montecarlo import single_design_summary as _single_design_summary
from rcb.nuisance import DEFAULT_VARIANCE_COMPONENT_BOUNDS
from rcb.risk import (
    deterministic_equivalent_diagonal,
    deterministic_equivalent_isotropic,
    estimate_variance_components,
    exact_risk_components,
    feasible_risk_geometry,
)

from .dgp import (
    ARTIFACT_DIR,
    DEFAULT_CONFIG,
    SCENARIOS,
    mean_shift,
    spectral_mean_shift,
    spectral_shift_weights,
)

CFG = DEFAULT_CONFIG
COMPONENTS = ("Signal", "Residual noise", "Total")
COMPONENT_COLUMNS = {
    "Signal": ("exact_signal", "estimate_signal", "theoretical_signal"),
    "Residual noise": ("exact_residual", "estimate_residual", "theoretical_residual"),
    "Total": ("exact_total", "estimate_total", "theoretical_total"),
}
#: Shift direction fixed for the AR(1) main figure (S5-2, revised): the
#: high-spectrum eigenvector of Sigma0, so that G_{delta,p} = delta_{s_p} and
#: the SCENARIOS grid below is directly comparable, column by column, to its
#: isotropic control in Appendix E.
AR1_MAIN_SHIFT_ORIENTATION = "high"


def _figure_nuisance_estimates(X0, Y0, eig, vec):
    """This section's nuisance settings: a 41-point refinement grid."""
    sq_r, sq_s, _ = estimate_variance_components(
        X0, Y0, eig, vec, theta_bounds=DEFAULT_VARIANCE_COMPONENT_BOUNDS, grid_size=41,
    )
    return sq_r, sq_s


def _main_draws(phi0=None, phi1=None, *, seed_offset=9101,
                 cache_stem="revised_main_single_design_draws"):
    """One fixed covariate design with repeated source outcomes.

    ``phi0``/``phi1`` default to ``CFG``'s values, and the default
    ``seed_offset``/``cache_stem`` reproduce Figure 1's original single call
    exactly; :func:`run_aspect_pair_risk_paths` reruns this at other joint
    aspect-ratio points by passing a distinct cache stem and seed offset.
    """
    if phi0 is None: phi0 = CFG.phi0
    if phi1 is None: phi1 = CFG.phi1
    cache = ARTIFACT_DIR / f"{cache_stem}.csv.gz"
    if cache.exists(): return pd.read_csv(cache)
    rng = np.random.default_rng(CFG.seed + seed_offset)
    p, outcomes = 640, 200
    n0, n1 = round(p/phi0), round(p/phi1)
    X0=rng.normal(size=(n0,p)); target_noise=rng.normal(size=(n1,p))
    _,X0c,eig,vec=design_eigendecomposition(X0)
    beta=rng.normal(scale=np.sqrt(CFG.r2/p),size=(p,outcomes))
    eps=rng.normal(scale=np.sqrt(CFG.sigma2),size=(n0,outcomes))
    Y0=X0@beta+eps; rhat,shat=_figure_nuisance_estimates(X0,Y0,eig,vec)
    rows=[]
    for eta in CFG.etas:
        for scenario,(rho2,varrho2) in SCENARIOS.items():
            nu=mean_shift(p,n0,eta,rho2); X1=target_noise+nu
            gamma0=design_independent_base_weights(n0,eta,varrho2)
            gp,delta=ridge_augmented_path(X0,X0c,eig,vec,X1.mean(0),gamma0,CFG.lambdas)
            exact=exact_risk_components(X0,gp,nu,CFG.r2,CFG.sigma2)
            sf,wn=feasible_risk_geometry(X1,delta,eig,vec,gp,CFG.lambdas)
            est_signal=rhat[:,None]*sf; est_residual=shat[:,None]*wn; est_total=est_signal+est_residual
            theory=deterministic_equivalent_diagonal(
                CFG.lambdas,phi0=p/n0,phi1=p/n1,eta=eta,rho2=rho2,varrho2=varrho2,
                r2=CFG.r2,sigma2=CFG.sigma2,
            )
            selected=np.argmin(est_total,axis=1)
            for draw in range(outcomes):
                for j,lam in enumerate(CFG.lambdas):
                    rows.append({"eta":eta,"scenario":scenario,"design":0,"draw":draw,"lambda":lam,
                      "exact_signal":n0**eta*exact[0][j],"exact_residual":n0**eta*exact[1][j],"exact_total":n0**eta*exact[2][j],
                      "estimate_signal":n0**eta*est_signal[draw,j],"estimate_residual":n0**eta*est_residual[draw,j],"estimate_total":n0**eta*est_total[draw,j],
                      "theoretical_signal":theory[0][j],"theoretical_residual":theory[1][j],"theoretical_total":theory[2][j],
                      "selected_lambda":CFG.lambdas[selected[draw]]})
    out=pd.DataFrame(rows); out.to_csv(cache,index=False,compression="gzip"); return out


def _aspect_draws():
    cache=ARTIFACT_DIR/"revised_aspect_single_design_rho_grid_draws.csv.gz"
    if cache.exists(): return pd.read_csv(cache)
    rng=np.random.default_rng(CFG.seed+9202); p=640; outcomes=200; lam=np.array([1.0]); phis=(.5,1.,2.,5.,10.)
    ratios=(.5,1.,5.,10.,20.); varrho2=1.0
    rows=[]
    def record(varied,phi,eta,ratio,design,X0,X1,nu,eig,vec,X0c,rhat,shat,gamma0):
        gp,delta=ridge_augmented_path(X0,X0c,eig,vec,X1.mean(0),gamma0,lam)
        exact=exact_risk_components(X0,gp,nu,CFG.r2,CFG.sigma2); sf,wn=feasible_risk_geometry(X1,delta,eig,vec,gp,lam)
        n0=len(X0); phi0=p/n0; phi1=p/len(X1)
        theory=deterministic_equivalent_isotropic(lam,phi0=phi0,phi1=phi1,eta=eta,rho2=ratio,varrho2=varrho2,r2=CFG.r2,sigma2=CFG.sigma2)
        for draw in range(outcomes):
            es=rhat[draw]*sf[0]; ev=shat[draw]*wn[0]
            rows.append({"varied":varied,"phi":phi,"eta":eta,"rho2_over_varrho2":ratio,"design":design,"draw":draw,
              "exact_signal":n0**eta*exact[0][0],"exact_residual":n0**eta*exact[1][0],"exact_total":n0**eta*exact[2][0],
              "estimate_signal":n0**eta*es,"estimate_residual":n0**eta*ev,"estimate_total":n0**eta*(es+ev),
              "theoretical_signal":theory[0][0],"theoretical_residual":theory[1][0],"theoretical_total":theory[2][0]})
    # Source ratio: one fixed covariate design at each phi, reused across eta.
    for phi in phis:
        n0=round(p/phi); n1=round(p/CFG.phi1)
        X0=rng.normal(size=(n0,p)); _,X0c,eig,vec=design_eigendecomposition(X0)
        beta=rng.normal(scale=np.sqrt(CFG.r2/p),size=(p,outcomes)); eps=rng.normal(scale=np.sqrt(CFG.sigma2),size=(n0,outcomes))
        rhat,shat=_figure_nuisance_estimates(X0,X0@beta+eps,eig,vec); noise=rng.normal(size=(n1,p))
        for eta in (0.,1.):
            for ratio in ratios:
                nu=mean_shift(p,n0,eta,ratio); X1=noise+nu; gamma0=design_independent_base_weights(n0,eta,varrho2)
                record("Source",phi,eta,ratio,0,X0,X1,nu,eig,vec,X0c,rhat,shat,gamma0)
    # Target ratio: one fixed source design reused over all target ratios.
    n0=round(p/CFG.phi0)
    X0=rng.normal(size=(n0,p)); _,X0c,eig,vec=design_eigendecomposition(X0)
    beta=rng.normal(scale=np.sqrt(CFG.r2/p),size=(p,outcomes)); eps=rng.normal(scale=np.sqrt(CFG.sigma2),size=(n0,outcomes))
    rhat,shat=_figure_nuisance_estimates(X0,X0@beta+eps,eig,vec)
    for phi in phis:
        n1=round(p/phi); noise=rng.normal(size=(n1,p))
        for eta in (0.,1.):
            for ratio in ratios:
                nu=mean_shift(p,n0,eta,ratio); X1=noise+nu; gamma0=design_independent_base_weights(n0,eta,varrho2)
                record("Target",phi,eta,ratio,0,X0,X1,nu,eig,vec,X0c,rhat,shat,gamma0)
    out=pd.DataFrame(rows); out.to_csv(cache,index=False,compression="gzip"); return out


def _selected_lambda_summary(draws: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Selected-penalty summary keyed by ``group_cols``: the star's location (S5-9).

    The manuscript marks the grid point nearest the *median* selected penalty
    (previously the geometric mean); the 25th/75th percentiles are stored
    alongside it for a possible later Appendix E display but are not otherwise
    used yet.
    """
    id_cols = group_cols + ["design", "draw", "selected_lambda"]
    grouped = draws[id_cols].drop_duplicates().groupby(group_cols)["selected_lambda"]
    return grouped.agg(
        selected_lambda="median",
        selected_lambda_q25=lambda x: float(np.percentile(x, 25)),
        selected_lambda_q75=lambda x: float(np.percentile(x, 75)),
    ).reset_index()


def prepare_main_figure_inputs(
    phi0=None, phi1=None, *, seed_offset=9101,
    draws_stem="revised_main_single_design_draws",
    summary_stem="revised_main_risk_summary",
    selected_stem="revised_main_selected_lambda",
):
    """Draws and summaries behind Figure 1, written for the notebook to plot.

    Default arguments reproduce Figure 1's original output files exactly;
    :func:`run_aspect_pair_risk_paths` reuses this summary logic verbatim at
    other joint ``(phi0, phi1)`` points by passing distinct stems.
    """
    d = _main_draws(phi0, phi1, seed_offset=seed_offset, cache_stem=draws_stem)
    groups = ["eta", "scenario", "lambda"]
    cols = [x for trio in COMPONENT_COLUMNS.values() for x in trio]
    s = _single_design_summary(d, groups, cols)
    repetitions = int(s["outcome_repetitions"].iloc[0])
    for exact, _, _ in COMPONENT_COLUMNS.values():
        # Under the Gaussian predictive law, a squared centered prediction
        # error with variance R has variance 2 R^2.  This is the analytic SE
        # of the empirical mean of squared errors over the outcome repeats.
        s[f"{exact}_gaussian_se"] = np.sqrt(2.0 / repetitions) * s[exact]
    tune = _selected_lambda_summary(d, ["eta", "scenario"])
    s.to_csv(ARTIFACT_DIR / f"{summary_stem}.csv", index=False)
    tune.to_csv(ARTIFACT_DIR / f"{selected_stem}.csv", index=False)
    return {"draws": d, "summary": s, "selected": tune}


def _ar1_main_draws(phi0=None, phi1=None, *, seed_offset=9111,
                     cache_stem="ar1_main_single_design_draws"):
    """One fixed AR(1) covariate design with repeated source outcomes (S5-2, revised).

    Structurally mirrors :func:`_main_draws`, including its ``phi0``/``phi1``
    passthrough (defaulting to ``CFG``'s values so the default call reproduces
    Figure 1's original single call exactly) and its ``SCENARIOS`` grid,
    unchanged from the isotropic version so the two figures are directly
    comparable column by column.  What changes is the covariance -- AR(1)
    rather than identity, with ``Sigma1 = Sigma0`` -- and the shift direction,
    fixed to :data:`AR1_MAIN_SHIFT_ORIENTATION` (high-spectrum) throughout
    rather than the isotropic flat coordinate direction.
    """
    if phi0 is None: phi0 = CFG.phi0
    if phi1 is None: phi1 = CFG.phi1
    cache = ARTIFACT_DIR / f"{cache_stem}.csv.gz"
    if cache.exists(): return pd.read_csv(cache)
    rng = np.random.default_rng(CFG.seed + seed_offset)
    p, outcomes = 640, 200
    n0, n1 = round(p / phi0), round(p / phi1)
    s, W = ar1_covariance(p, CFG.rho_ar)
    sigma0_sqrt = symmetric_sqrt(s, W)
    X0 = rng.normal(size=(n0, p)).dot(sigma0_sqrt)
    target_noise = rng.normal(size=(n1, p)).dot(sigma0_sqrt)
    _, X0c, eig, vec = design_eigendecomposition(X0)
    beta = rng.normal(scale=np.sqrt(CFG.r2 / p), size=(p, outcomes))
    eps = rng.normal(scale=np.sqrt(CFG.sigma2), size=(n0, outcomes))
    Y0 = X0 @ beta + eps
    rhat, shat = _figure_nuisance_estimates(X0, Y0, eig, vec)
    shift_weights = spectral_shift_weights(p, AR1_MAIN_SHIFT_ORIENTATION)
    rows = []
    for eta in CFG.etas:
        for scenario, (rho2, varrho2) in SCENARIOS.items():
            nu = spectral_mean_shift(p, n0, eta, rho2, W, AR1_MAIN_SHIFT_ORIENTATION)
            X1 = target_noise + nu
            gamma0 = design_independent_base_weights(n0, eta, varrho2)
            gp, delta = ridge_augmented_path(X0, X0c, eig, vec, X1.mean(0), gamma0, CFG.lambdas)
            exact = exact_risk_components(X0, gp, nu, CFG.r2, CFG.sigma2)
            sf, wn = feasible_risk_geometry(X1, delta, eig, vec, gp, CFG.lambdas)
            est_signal = rhat[:, None] * sf
            est_residual = shat[:, None] * wn
            est_total = est_signal + est_residual
            theory = deterministic_equivalent_diagonal(
                CFG.lambdas, phi0=p / n0, phi1=p / n1, eta=eta, rho2=rho2, varrho2=varrho2,
                r2=CFG.r2, sigma2=CFG.sigma2,
                sigma0_eigenvalues=s, sigma1_diagonal=s,
                shift_weights=shift_weights,
            )
            selected = np.argmin(est_total, axis=1)
            for draw in range(outcomes):
                for j, lam in enumerate(CFG.lambdas):
                    rows.append({
                        "eta": eta, "scenario": scenario, "design": 0, "draw": draw, "lambda": lam,
                        "exact_signal": n0**eta * exact[0][j], "exact_residual": n0**eta * exact[1][j], "exact_total": n0**eta * exact[2][j],
                        "estimate_signal": n0**eta * est_signal[draw, j], "estimate_residual": n0**eta * est_residual[draw, j], "estimate_total": n0**eta * est_total[draw, j],
                        "theoretical_signal": theory[0][j], "theoretical_residual": theory[1][j], "theoretical_total": theory[2][j],
                        "selected_lambda": CFG.lambdas[selected[draw]],
                    })
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False, compression="gzip")
    return out


def prepare_ar1_main_figure_inputs(
    phi0=None, phi1=None, *, seed_offset=9111,
    draws_stem="ar1_main_single_design_draws",
    summary_stem="ar1_main_risk_summary",
    selected_stem="ar1_main_selected_lambda",
):
    """Draws and summaries behind the revised AR(1) Figure 1 (S5-2).

    Default arguments reproduce Figure 1's original output files exactly;
    :func:`run.run_ar1_main_aspect_pair_risk_paths` reuses this summary logic
    verbatim at other joint ``(phi0, phi1)`` points by passing distinct
    stems, mirroring :func:`prepare_main_figure_inputs`.
    """
    d = _ar1_main_draws(phi0, phi1, seed_offset=seed_offset, cache_stem=draws_stem)
    groups = ["eta", "scenario", "lambda"]
    cols = [x for trio in COMPONENT_COLUMNS.values() for x in trio]
    s = _single_design_summary(d, groups, cols)
    repetitions = int(s["outcome_repetitions"].iloc[0])
    for exact, _, _ in COMPONENT_COLUMNS.values():
        s[f"{exact}_gaussian_se"] = np.sqrt(2.0 / repetitions) * s[exact]
    tune = _selected_lambda_summary(d, ["eta", "scenario"])
    s.to_csv(ARTIFACT_DIR / f"{summary_stem}.csv", index=False)
    tune.to_csv(ARTIFACT_DIR / f"{selected_stem}.csv", index=False)
    return {"draws": d, "summary": s, "selected": tune}


def prepare_aspect_figure_inputs():
    """Draws and summaries behind Figure 2, written for the notebook to plot."""
    d = _aspect_draws()
    groups = ["varied", "phi", "eta", "rho2_over_varrho2"]
    cols = [x for trio in COMPONENT_COLUMNS.values() for x in trio]
    s = _single_design_summary(d, groups, cols)
    repetitions = int(s["outcome_repetitions"].iloc[0])
    for exact, _, _ in COMPONENT_COLUMNS.values():
        s[f"{exact}_gaussian_se"] = np.sqrt(2.0 / repetitions) * s[exact]
    s.to_csv(ARTIFACT_DIR / "revised_aspect_ratio_summary.csv", index=False)

    # The dashed theoretical curve on each panel is a deterministic-equivalent
    # evaluation over a 240-point aspect-ratio grid.  It is an experiment, not a
    # drawing instruction, so it is computed and written here rather than in the
    # notebook.
    phi_curve = np.geomspace(0.5, 10, 240)
    rows = []
    for varied in ("Source", "Target"):
        phi0s = phi_curve if varied == "Source" else np.full_like(phi_curve, CFG.phi0)
        phi1s = phi_curve if varied == "Target" else np.full_like(phi_curve, CFG.phi1)
        for eta in (0.0, 1.0):
            for ratio in (0.5, 1.0, 5.0, 10.0, 20.0):
                for phi, a, b in zip(phi_curve, phi0s, phi1s):
                    signal, residual, total = (
                        deterministic_equivalent_isotropic(
                            np.array([1.0]), phi0=float(a), phi1=float(b),
                            eta=eta, rho2=ratio, varrho2=1.0,
                            r2=CFG.r2, sigma2=CFG.sigma2,
                        )
                    )
                    rows.append({
                        "varied": varied, "eta": eta,
                        "rho2_over_varrho2": ratio, "phi": float(phi),
                        "Signal": float(signal[0]),
                        "Residual noise": float(residual[0]),
                        "Total": float(total[0]),
                    })
    curves = pd.DataFrame(rows)
    curves.to_csv(
        ARTIFACT_DIR / "revised_aspect_theoretical_curves.csv", index=False
    )
    return {"draws": d, "summary": s, "theoretical_curves": curves}


def _ar1_aspect_draws():
    """One fixed AR(1) covariate design per aspect ratio, diffuse shift only (S5-3).

    Structurally mirrors :func:`_aspect_draws`: ``Sigma0 = Sigma1 =`` the AR(1)
    covariance replaces the identity, and the shift orientation is fixed to
    diffuse (``G_{delta,p} = H_{0,p}``) throughout, avoiding an arbitrary
    favorable/unfavorable orientation while ``phi0``/``phi1`` vary.
    """
    cache = ARTIFACT_DIR / "ar1_aspect_single_design_rho_grid_draws.csv.gz"
    if cache.exists(): return pd.read_csv(cache)
    rng = np.random.default_rng(CFG.seed + 9212)
    p, outcomes, lam = 640, 200, np.array([1.0])
    phis = (.5, 1., 2., 5., 10.)
    ratios = (.5, 1., 5., 10., 20.)
    varrho2 = 1.0
    s, W = ar1_covariance(p, CFG.rho_ar)
    sigma_sqrt = symmetric_sqrt(s, W)
    rows = []

    def record(varied, phi, eta, ratio, design, X0, X1, nu, eig, vec, X0c, rhat, shat, gamma0):
        gp, delta = ridge_augmented_path(X0, X0c, eig, vec, X1.mean(0), gamma0, lam)
        exact = exact_risk_components(X0, gp, nu, CFG.r2, CFG.sigma2)
        sf, wn = feasible_risk_geometry(X1, delta, eig, vec, gp, lam)
        n0 = len(X0); phi0 = p / n0; phi1 = p / len(X1)
        theory = deterministic_equivalent_diagonal(
            lam, phi0=phi0, phi1=phi1, eta=eta, rho2=ratio, varrho2=varrho2,
            r2=CFG.r2, sigma2=CFG.sigma2,
            sigma0_eigenvalues=s, sigma1_diagonal=s, shift_weights=None,
        )
        for draw in range(outcomes):
            es = rhat[draw] * sf[0]; ev = shat[draw] * wn[0]
            rows.append({"varied": varied, "phi": phi, "eta": eta, "rho2_over_varrho2": ratio, "design": design, "draw": draw,
              "exact_signal": n0**eta*exact[0][0], "exact_residual": n0**eta*exact[1][0], "exact_total": n0**eta*exact[2][0],
              "estimate_signal": n0**eta*es, "estimate_residual": n0**eta*ev, "estimate_total": n0**eta*(es+ev),
              "theoretical_signal": theory[0][0], "theoretical_residual": theory[1][0], "theoretical_total": theory[2][0]})

    # Source ratio: one fixed AR(1) covariate design at each phi, reused across eta.
    for phi in phis:
        n0 = round(p / phi); n1 = round(p / CFG.phi1)
        X0 = rng.normal(size=(n0, p)).dot(sigma_sqrt)
        _, X0c, eig, vec = design_eigendecomposition(X0)
        beta = rng.normal(scale=np.sqrt(CFG.r2 / p), size=(p, outcomes))
        eps = rng.normal(scale=np.sqrt(CFG.sigma2), size=(n0, outcomes))
        rhat, shat = _figure_nuisance_estimates(X0, X0 @ beta + eps, eig, vec)
        noise = rng.normal(size=(n1, p)).dot(sigma_sqrt)
        for eta in (0., 1.):
            for ratio in ratios:
                nu = spectral_mean_shift(p, n0, eta, ratio, W, "diffuse")
                X1 = noise + nu
                gamma0 = design_independent_base_weights(n0, eta, varrho2)
                record("Source", phi, eta, ratio, 0, X0, X1, nu, eig, vec, X0c, rhat, shat, gamma0)

    # Target ratio: one fixed AR(1) source design reused over all target ratios.
    n0 = round(p / CFG.phi0)
    X0 = rng.normal(size=(n0, p)).dot(sigma_sqrt)
    _, X0c, eig, vec = design_eigendecomposition(X0)
    beta = rng.normal(scale=np.sqrt(CFG.r2 / p), size=(p, outcomes))
    eps = rng.normal(scale=np.sqrt(CFG.sigma2), size=(n0, outcomes))
    rhat, shat = _figure_nuisance_estimates(X0, X0 @ beta + eps, eig, vec)
    for phi in phis:
        n1 = round(p / phi)
        noise = rng.normal(size=(n1, p)).dot(sigma_sqrt)
        for eta in (0., 1.):
            for ratio in ratios:
                nu = spectral_mean_shift(p, n0, eta, ratio, W, "diffuse")
                X1 = noise + nu
                gamma0 = design_independent_base_weights(n0, eta, varrho2)
                record("Target", phi, eta, ratio, 0, X0, X1, nu, eig, vec, X0c, rhat, shat, gamma0)
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False, compression="gzip")
    return out


def prepare_ar1_aspect_figure_inputs():
    """Draws and summaries behind the new AR(1) Figure 2 (S5-3)."""
    d = _ar1_aspect_draws()
    groups = ["varied", "phi", "eta", "rho2_over_varrho2"]
    cols = [x for trio in COMPONENT_COLUMNS.values() for x in trio]
    s = _single_design_summary(d, groups, cols)
    repetitions = int(s["outcome_repetitions"].iloc[0])
    for exact, _, _ in COMPONENT_COLUMNS.values():
        s[f"{exact}_gaussian_se"] = np.sqrt(2.0 / repetitions) * s[exact]
    s.to_csv(ARTIFACT_DIR / "ar1_aspect_ratio_summary.csv", index=False)

    # The dashed theoretical curve, as in prepare_aspect_figure_inputs, but via
    # the general (bisection-based) deterministic equivalent rather than the
    # identity-covariance closed form, since Sigma0 = Sigma1 != I here.
    sigma_eigenvalues, _ = ar1_covariance(640, CFG.rho_ar)
    phi_curve = np.geomspace(0.5, 10, 240)
    rows = []
    for varied in ("Source", "Target"):
        phi0s = phi_curve if varied == "Source" else np.full_like(phi_curve, CFG.phi0)
        phi1s = phi_curve if varied == "Target" else np.full_like(phi_curve, CFG.phi1)
        for eta in (0.0, 1.0):
            for ratio in (0.5, 1.0, 5.0, 10.0, 20.0):
                for phi, a, b in zip(phi_curve, phi0s, phi1s):
                    signal, residual, total = deterministic_equivalent_diagonal(
                        np.array([1.0]), phi0=float(a), phi1=float(b),
                        eta=eta, rho2=ratio, varrho2=1.0,
                        r2=CFG.r2, sigma2=CFG.sigma2,
                        sigma0_eigenvalues=sigma_eigenvalues,
                        sigma1_diagonal=sigma_eigenvalues, shift_weights=None,
                    )
                    rows.append({
                        "varied": varied, "eta": eta,
                        "rho2_over_varrho2": ratio, "phi": float(phi),
                        "Signal": float(signal[0]),
                        "Residual noise": float(residual[0]),
                        "Total": float(total[0]),
                    })
    curves = pd.DataFrame(rows)
    curves.to_csv(ARTIFACT_DIR / "ar1_aspect_theoretical_curves.csv", index=False)
    return {"draws": d, "summary": s, "theoretical_curves": curves}
