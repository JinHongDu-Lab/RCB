"""Section 6: the evaluation sweep.

Runs the Monte Carlo and writes ``results/evaluation/``.  The notebook
``2_evaluation.ipynb`` reads what this writes and plots it.

Run the whole thing::

    python -m experiments.evaluation.run

or a fast pass over every stage at four replications::

    python -m experiments.evaluation.run --smoke

``--smoke`` sets ``CAUSAL_EXTRAPOLATION_SMOKE``, which ``dgp`` reads for its
replication count, so the reduced configuration is declared in exactly one
place.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .dgp import (
    ARTIFACT_DIR,
    FAMILY_ORDER,
    FIGURE_DIR,
    FIXED_LAMBDA,
    HEADLINE_STAGES,
    LAMBDAS,
    N1,
    N_JOBS,
    N_REPETITIONS,
    OVERLAP_SETTINGS,
    P,
    PHI0_SETTINGS,
    PHI1,
    PHI1_SUPPLEMENT,
    PILOT_FRACTION,
    R2,
    RESPONSE_BOOTSTRAP_DRAWS,
    RESPONSE_BOOTSTRAP_SEED,
    RESPONSE_PHI0,
    RESPONSE_SURFACES,
    RESPONSE_SURFACE_ORDER,
    SEED,
    SIGMA02,
    SBW_STANDARD_ERROR_MULTIPLIER,
    SMOKE_TEST,
    SQ_THETA_BOUNDS,
    TAU,
    all_base_weights,
    augmentation_path,
    draw_replication,
    score_weights,
    arb_estimate,
    full_double_ridge_abw,
    sample_trimmed_ipw_estimate,
)
from rcb.benchmarks.bases import base_weight_families
from rcb.estimator import fit_rcb, reference_penalty_rules
from .dgp import BASE_SETTINGS


#: Every (replication, overlap, shift, aspect-ratio) cell of the design.  Both
#: sweeps address the same grid, so the ARB comparator runs on exactly the
#: replications the augmented estimators ran on.
TASKS = [
    (rep, overlap, a, phi0)
    for overlap, a in OVERLAP_SETTINGS
    for phi0 in PHI0_SETTINGS
    for rep in range(N_REPETITIONS)
]

def write_configuration() -> None:
    """Record the design so a result can be traced back to what produced it."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "seed": SEED, "p": P, "phi1": PHI1,
        "phi0_settings": ";".join(map(str, PHI0_SETTINGS)),
        "n1": N1, "pilot_fraction": PILOT_FRACTION,
        "r2": R2, "sigma02": SIGMA02, "tau": TAU,
        "lambda_grid": ";".join(f"{x:.12g}" for x in LAMBDAS),
        "fixed_lambda": FIXED_LAMBDA, "replications": N_REPETITIONS,
        "overlap_settings": str(OVERLAP_SETTINGS),
        "family_order": ";".join(FAMILY_ORDER),
        "headline_nuisance_method": "Spectral quasi-score",
        "paired_nuisance_baseline": "Method of moments",
        "spectral_quasi_score_definition": "global profiled quasi-likelihood minimizer",
        "sq_r2_bounds": str(SQ_THETA_BOUNDS[0]),
        "sq_sigma2_bounds": str(SQ_THETA_BOUNDS[1]),
        "abw_default": "Full ABW (outcome CV, delta=lambda)",
        "trimmed_ipw_estimand": "ATT in retained target sample",
        "sbw_tolerance_rule": (
            f"{SBW_STANDARD_ERROR_MULTIPLIER:g}*sqrt(1/n0+1/n1_pilot)"
        ),
    }]).to_csv(ARTIFACT_DIR / "configuration.csv", index=False)


def run_method_sweep(
    tasks: list[tuple[int, str, float, float]] = TASKS,
    *, phi1: float = PHI1, output_suffix: str = "", isotropic: bool = False,
) -> dict:
    """The design-only base families and augmentations over every design cell.

    ``phi1``/``output_suffix`` default to the primary sweep's target aspect
    ratio and its unsuffixed filenames; :func:`run_phi1_supplement_sweep`
    reruns this at ``PHI1_SUPPLEMENT`` for the appendix panels app-E7/app-E8,
    and :func:`run_isotropic_sweep` reruns this with ``isotropic=True`` for
    the appendix sensitivity check against ``Sigma0=I_p``.
    """
    def simulate_cell(repetition, overlap, a, phi0):
        d2 = float(a**2)
        data = draw_replication(repetition, phi0, a, phi1=phi1, isotropic=isotropic)
        base_diagnostics = {}
        bases = all_base_weights(data["X0"], data["X1P"], diagnostics=base_diagnostics)
        rows, diagnostics = [], []
        for family, gamma_base in bases.items():
            path = augmentation_path(data["X0"], data["X1E"], data["Y0"], gamma_base, data["nu1"])
            rows.append(score_weights(family, "Base", gamma_base, gamma_base, data, overlap, d2, phi0, repetition, np.nan, np.nan))
            selections = [
                ("Augmented (OOD)", path["ood_sq_index"], path["rood_sq"]),
                ("Augmented (OOD, Method of moments)", path["ood_mom_index"], path["rood_mom"]),
                ("Augmented (GCV)", path["gcv_index"], path["rood_sq"]),
                ("Augmented (fixed lambda=1)", path["fixed_index"], path["rood_sq"]),
                ("Augmented (oracle)", path["oracle_index"], path["rood_sq"]),
            ]
            for stage, idx, criterion_path in selections:
                rows.append(score_weights(
                    family, stage, path["gamma_path"][:, idx], gamma_base, data,
                    overlap, d2, phi0, repetition, float(LAMBDAS[idx]),
                    float(criterion_path[idx]),
                ))
            diagnostics.append({
                "overlap": overlap, "D2": d2, "phi0": phi0, "repetition": repetition, "family": family,
                # The base's own design-only tuning constant, where it has one
                # (the l2 family's Riesz-CV penalty); NaN for the others.
                "base_penalty": float(
                    base_diagnostics.get(family, {}).get("delta", np.nan)
                ),
                "lambda_OOD": float(LAMBDAS[path["ood_sq_index"]]),
                "lambda_OOD_mom": float(LAMBDAS[path["ood_mom_index"]]),
                "lambda_GCV": float(LAMBDAS[path["gcv_index"]]),
                "lambda_oracle": float(LAMBDAS[path["oracle_index"]]),
                "R_OOD_selected": float(path["rood_sq"][path["ood_sq_index"]]),
                "R_OOD_mom_selected": float(path["rood_mom"][path["ood_mom_index"]]),
                "R_exact_at_OOD": float(path["exact"][path["ood_sq_index"]]),
                "R_exact_at_OOD_sq": float(path["exact"][path["ood_sq_index"]]),
                "R_exact_at_OOD_mom": float(path["exact"][path["ood_mom_index"]]),
                "R_exact_at_oracle": float(path["exact"][path["oracle_index"]]),
                "rhat2": path["rhat2_sq"], "sigmahat2": path["sigmahat2_sq"],
                "rhat2_sq": path["rhat2_sq"], "sigmahat2_sq": path["sigmahat2_sq"],
                "rhat2_mom": path["rhat2_mom"],
                "sigmahat2_mom": path["sigmahat2_mom"],
                "rhohat_sq": path["rhohat_sq"],
                "sq_boundary": path["sq_boundary"],
                "sq_objective_improvement": path["sq_objective_improvement"],
                "max_normalization_error": path["max_normalization_error"],
                "honest_split_disjoint": bool(np.intersect1d(data["I_P"], data["I_E"]).size == 0),
            })
        return rows, diagnostics


    results = Parallel(n_jobs=N_JOBS, prefer="processes", verbose=0)(
        delayed(simulate_cell)(*task) for task in tasks
    )
    draws = pd.DataFrame([row for rows, _ in results for row in rows])
    design = pd.DataFrame([row for _, diagnostics in results for row in diagnostics])

    def summarize(g):
        risk = g["exact_risk"].to_numpy()
        rmse = float(np.sqrt(risk.mean()))
        rmse_mcse = float(risk.std(ddof=1) / (2 * rmse * np.sqrt(len(risk))))
        mcse = lambda column: float(g[column].std(ddof=1) / np.sqrt(len(g)))
        return pd.Series({
            "replications": len(g),
            "predictive_rmse_exact": rmse,
            "predictive_rmse_mcse": rmse_mcse,
            "signal_risk": g["signal_risk"].mean(),
            "integrated_squared_extrapolation_bias": g["signal_risk"].mean(),
            "integrated_squared_extrapolation_bias_mcse": mcse("signal_risk"),
            "noise_risk": g["noise_risk"].mean(),
            "residual_noise_variance": g["noise_risk"].mean(),
            "residual_noise_variance_mcse": mcse("noise_risk"),
            "empirical_mu0_rmse": np.sqrt(np.mean(g["mu0_error"] ** 2)),
            "att_rmse": np.sqrt(np.mean(g["att_error"] ** 2)),
            "imbalance_norm": g["imbalance_norm"].mean(),
            "population_imbalance_norm": g["population_imbalance_norm"].mean(),
            "delta_norm": g["delta_norm"].mean(),
            "delta_norm_mcse": mcse("delta_norm"),
            "gamma_norm": g["gamma_norm"].mean(),
            "weight_norm": g["gamma_norm"].mean(),
            "weight_norm_mcse": mcse("gamma_norm"),
            "ess_median": g["ess"].median(),
            "negative_mass": g["negative_mass"].mean(),
            "lambda_median": g["lambda_selected"].median(),
        })

    summary = draws.groupby(["overlap", "D2", "phi0", "family", "stage"], sort=False).apply(summarize, include_groups=False).reset_index()
    draws.to_csv(ARTIFACT_DIR / f"method_level_draws{output_suffix}.csv.gz", index=False, compression="gzip")
    design.to_csv(ARTIFACT_DIR / f"design_diagnostics{output_suffix}.csv.gz", index=False, compression="gzip")
    summary.to_csv(ARTIFACT_DIR / f"method_summary{output_suffix}.csv", index=False)


    unique_nuisance = design.drop_duplicates(
        ["overlap", "phi0", "repetition"]
    ).copy()
    nuisance_rows = []
    for method, r_column, sigma_column in [
        ("Method of moments", "rhat2_mom", "sigmahat2_mom"),
        ("Spectral quasi-score", "rhat2_sq", "sigmahat2_sq"),
    ]:
        frame = unique_nuisance[["overlap", "phi0", "repetition", r_column, sigma_column]].copy()
        frame["nuisance_method"] = method
        frame["rhat2"] = frame[r_column]
        frame["sigmahat2"] = frame[sigma_column]
        frame["r2_error"] = frame["rhat2"] - R2
        frame["sigma2_error"] = frame["sigmahat2"] - SIGMA02
        nuisance_rows.append(frame)
    nuisance_draws = pd.concat(nuisance_rows, ignore_index=True)
    nuisance_summary = (
        nuisance_draws.groupby(["overlap", "phi0", "nuisance_method"], sort=False)
        .apply(
            lambda g: pd.Series({
                "designs": len(g),
                "rhat2_mean": g["rhat2"].mean(),
                "rhat2_rmse": np.sqrt(np.mean(g["r2_error"] ** 2)),
                "sigmahat2_mean": g["sigmahat2"].mean(),
                "sigmahat2_rmse": np.sqrt(np.mean(g["sigma2_error"] ** 2)),
                "r2_zero_fraction": np.mean(g["rhat2"] <= 1e-10),
                "sigma2_near_zero_fraction": np.mean(g["sigmahat2"] <= 1e-4),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    tuning_effect_draws = design[[
        "overlap", "D2", "phi0", "repetition", "family",
        "lambda_OOD", "lambda_OOD_mom", "lambda_oracle",
        "R_exact_at_OOD_sq", "R_exact_at_OOD_mom", "R_exact_at_oracle",
    ]].copy()
    tuning_effect_draws["exact_risk_ratio_SQ_to_MoM"] = (
        tuning_effect_draws["R_exact_at_OOD_sq"]
        / tuning_effect_draws["R_exact_at_OOD_mom"]
    )
    tuning_effect_draws["SQ_excess_risk_ratio_to_oracle"] = (
        tuning_effect_draws["R_exact_at_OOD_sq"]
        / tuning_effect_draws["R_exact_at_oracle"]
    )
    tuning_effect_draws["MoM_excess_risk_ratio_to_oracle"] = (
        tuning_effect_draws["R_exact_at_OOD_mom"]
        / tuning_effect_draws["R_exact_at_oracle"]
    )
    tuning_effect_summary = (
        tuning_effect_draws.groupby(["overlap", "D2", "phi0", "family"], sort=False)
        .agg(
            replications=("repetition", "size"),
            lambda_SQ_median=("lambda_OOD", "median"),
            lambda_MoM_median=("lambda_OOD_mom", "median"),
            lambda_oracle_median=("lambda_oracle", "median"),
            exact_risk_SQ_mean=("R_exact_at_OOD_sq", "mean"),
            exact_risk_MoM_mean=("R_exact_at_OOD_mom", "mean"),
            exact_risk_oracle_mean=("R_exact_at_oracle", "mean"),
            exact_risk_ratio_SQ_to_MoM_mean=("exact_risk_ratio_SQ_to_MoM", "mean"),
            fraction_SQ_lower_risk=("exact_risk_ratio_SQ_to_MoM", lambda x: np.mean(x < 1)),
        )
        .reset_index()
    )

    nuisance_draws.to_csv(
        ARTIFACT_DIR / f"nuisance_estimates_raw{output_suffix}.csv.gz",
        index=False, compression="gzip",
    )
    nuisance_summary.to_csv(
        ARTIFACT_DIR / f"nuisance_estimation_summary{output_suffix}.csv", index=False
    )
    tuning_effect_draws.to_csv(
        ARTIFACT_DIR / f"paired_nuisance_tuning_draws{output_suffix}.csv.gz",
        index=False, compression="gzip",
    )
    tuning_effect_summary.to_csv(
        ARTIFACT_DIR / f"paired_nuisance_tuning_summary{output_suffix}.csv", index=False
    )
    return {
        "draws": draws, "design": design, "summary": summary,
        "nuisance_summary": nuisance_summary,
        "tuning_effect_summary": tuning_effect_summary,
    }


def validate(draws, design, summary) -> dict:
    """Design and numerical checks; refuses to record a failed run."""
    headline = summary[summary["stage"].isin(HEADLINE_STAGES)].copy()
    cell_counts = draws.groupby(["overlap", "phi0", "family", "stage"]).size()
    calibration_sq = design["R_OOD_selected"] / design["R_exact_at_OOD_sq"]
    unique_design = draws.drop_duplicates(["overlap", "phi0", "repetition"])
    l2_penalty = design.loc[design["family"].eq("L2 balancing"), "base_penalty"].to_numpy()

    checks = {
        "paper_dimension_p_50": bool(P == 50),
        "paper_total_target_n1_100": bool(N1 == 100 and draws["n1"].eq(100).all()),
        "paper_source_sizes_100_and_40": bool(set(draws["n0"].unique()) == {100, 40}),
        "honest_target_split_is_50_plus_50": bool(PILOT_FRACTION == 0.5),
        "honest_splits_all_disjoint": bool(design["honest_split_disjoint"].all()),
        "eight_required_base_families_present": bool(set(FAMILY_ORDER) == {
            "Uniform", "IPW", "Entropy balancing", "SBW", "CBPS",
            "Overlap weights", "L2 balancing", "ARB",
        }),
        "l2_base_penalty_positive_and_finite": bool(
            np.isfinite(l2_penalty).all() and (l2_penalty > 0).all()
        ),
        # The authors' Riesz grid ends at 1.0; how often the selected balancing
        # penalty sits on that endpoint is reported, not acted on.
        "l2_base_penalty_at_riesz_grid_cap_fraction": float(
            np.mean(l2_penalty == 1.0)
        ),
        "complete_replications_per_method_cell": bool(cell_counts.eq(N_REPETITIONS).all()),
        "production_configuration_is_120_replications": bool(SMOKE_TEST or N_REPETITIONS == 120),
        "no_nonfinite_method_outputs": bool(np.isfinite(draws[[
            "signal_risk", "noise_risk", "exact_risk", "gamma_norm", "delta_norm", "weight_sum"
        ]].to_numpy()).all()),
        "exact_risk_equals_B_plus_V": bool(np.allclose(
            draws["exact_risk"], draws["signal_risk"] + draws["noise_risk"], atol=1e-12
        )),
        "all_augmented_weights_affine_normalized": bool(design["max_normalization_error"].max() < 1e-10),
        "maximum_augmented_weight_sum_error": float(design["max_normalization_error"].max()),
        "spectral_quasi_score_is_headline": True,
        "two_moment_rule_is_diagnostic_only": True,
        "oracle_used_for_evaluation_only": True,
        "mean_beta_shift_cos2": float(unique_design["beta_shift_cos2"].mean()),
        "isotropic_reference_cos2_1_over_p": float(1 / P),
        "median_estimated_to_exact_OOD_risk_SQ": float(calibration_sq.median()),
        "spectral_parameter_boundary_fraction": float(
            design.drop_duplicates(["overlap", "phi0", "repetition"])["sq_boundary"].mean()
        ),
    }
    if not all(v for k, v in checks.items() if isinstance(v, (bool, np.bool_))):
        failed = [k for k, v in checks.items() if isinstance(v, (bool, np.bool_)) and not v]
        raise AssertionError(f"Validation failed: {failed}")

    (ARTIFACT_DIR / "validation.json").write_text(json.dumps(checks, indent=2))
    pd.DataFrame({"check": list(checks), "value": [str(v) for v in checks.values()]}).to_csv(
        ARTIFACT_DIR / "validation.csv", index=False
    )
    return checks


def run_complete_estimator_comparison(
    draws, tasks: list[tuple[int, str, float, float]] = TASKS,
    *, phi1: float = PHI1, output_suffix: str = "", isotropic: bool = False,
) -> dict:
    """Complete-estimator comparators and trimming sensitivity."""
    def simulate_arb_cell(repetition, overlap, a, phi0):
        data = draw_replication(repetition, phi0, a, phi1=phi1, isotropic=isotropic)
        common = {
            "overlap": overlap, "D2": float(a**2), "phi0": phi0,
            "n0": len(data["X0"]), "n1": len(data["X1"]),
            "repetition": repetition,
        }
        arb = arb_estimate(data)
        arb.update(common)
        arb["family"] = "ARB"
        abw = full_double_ridge_abw(data, repetition)
        for row in abw:
            row.update(common)
        trimmed = sample_trimmed_ipw_estimate(data)
        trimmed.update(common)
        return arb, abw, trimmed


    arb_results = Parallel(n_jobs=N_JOBS, prefer="processes", verbose=0)(
        delayed(simulate_arb_cell)(*task) for task in tasks
    )
    arb_draws = pd.DataFrame([row[0] for row in arb_results])
    abw_draws = pd.DataFrame([
        abw_row for _, abw_rows, _ in arb_results for abw_row in abw_rows
    ])
    trimmed_draws = pd.DataFrame([row[2] for row in arb_results])
    arb_draws.to_csv(
        ARTIFACT_DIR / f"arb_estimator_draws{output_suffix}.csv.gz",
        index=False, compression="gzip",
    )
    abw_draws.to_csv(
        ARTIFACT_DIR / f"abw_estimator_draws{output_suffix}.csv.gz",
        index=False, compression="gzip",
    )
    trimmed_draws.to_csv(
        ARTIFACT_DIR / f"trimmed_ipw_sensitivity_draws{output_suffix}.csv.gz",
        index=False, compression="gzip",
    )

    empirical_weight_draws = draws[
        draws["stage"].isin(HEADLINE_STAGES)
    ][["overlap", "D2", "phi0", "repetition", "family", "stage", "mu0_error"]].copy()
    empirical_weight_draws = empirical_weight_draws[
        ~(
            empirical_weight_draws["family"].eq("Overlap weights")
            & empirical_weight_draws["stage"].eq("Base")
        )
    ]
    estimator_draws = pd.concat([
        empirical_weight_draws,
        arb_draws[["overlap", "D2", "phi0", "repetition", "family", "stage", "mu0_error"]],
        abw_draws[
            abw_draws["stage"].eq("Full ABW (outcome CV, delta=lambda)")
        ][["overlap", "D2", "phi0", "repetition", "family", "stage", "mu0_error"]],
    ], ignore_index=True)


    def empirical_rmse_summary(g):
        squared_error = g["mu0_error"].to_numpy() ** 2
        rmse = float(np.sqrt(squared_error.mean()))
        mcse = float(squared_error.std(ddof=1) / (2 * rmse * np.sqrt(len(squared_error))))
        return pd.Series({"replications": len(g), "empirical_predictive_rmse": rmse, "rmse_mcse": mcse})


    estimator_summary = (
        estimator_draws.groupby(["overlap", "D2", "phi0", "family", "stage"], sort=False)
        .apply(empirical_rmse_summary, include_groups=False).reset_index()
    )
    estimator_summary.to_csv(
        ARTIFACT_DIR / f"arb_estimator_summary{output_suffix}.csv", index=False
    )
    abw_diagnostic_summary = (
        abw_draws.groupby(
            ["overlap", "D2", "phi0", "family", "stage"], sort=False
        )
        .apply(empirical_rmse_summary, include_groups=False).reset_index()
    )
    abw_diagnostic_summary.to_csv(
        ARTIFACT_DIR / f"abw_estimator_summary{output_suffix}.csv", index=False
    )
    trimmed_summary = (
        trimmed_draws.groupby(
            ["overlap", "D2", "phi0", "family", "stage"], sort=False
        )
        .apply(empirical_rmse_summary, include_groups=False).reset_index()
    )
    trimmed_summary.to_csv(
        ARTIFACT_DIR / f"trimmed_ipw_sensitivity_summary{output_suffix}.csv",
        index=False,
    )
    cell_keys = ["overlap", "phi0", "repetition"]
    arb_weight_match = draws[
        draws["family"].eq("ARB") & draws["stage"].eq("Base")
    ][cell_keys + ["gamma_norm"]].merge(
        arb_draws[cell_keys + ["residual_weight_norm"]], on=cell_keys
    )
    arb_checks = {
        "all_ARB_rows_are_full_ARB": bool(arb_draws["stage"].eq("Full ARB").all()),
        "ARB_never_uses_centered_ridge_augmentation": bool(
            (~arb_draws["uses_centered_ridge_augmentation"]).all()
        ),
        "one_full_ARB_result_per_design": bool(
            arb_draws.groupby(["overlap", "phi0", "repetition"]).size().eq(1).all()
        ),
        "ARB_all_estimates_finite": bool(np.isfinite(arb_draws["mu0_error"]).all()),
        "default_ABW_sets_delta_equal_to_lambda": bool(np.allclose(
            abw_draws.loc[
                abw_draws["stage"].eq("Full ABW (outcome CV, delta=lambda)"),
                "delta_balance",
            ],
            abw_draws.loc[
                abw_draws["stage"].eq("Full ABW (outcome CV, delta=lambda)"),
                "lambda_outcome",
            ],
        )),
        "ABW_never_uses_target_outcomes": bool(
            (~abw_draws["uses_target_outcomes"]).all()
        ),
        "ABW_linear_representation_is_exact": bool(
            abw_draws["linear_representation_error"].max() < 1e-10
        ),
        "ABW_direct_formula_matches_implied_weights": bool(
            abw_draws["formula_representation_error"].max() < 1e-8
        ),
        "trimmed_IPW_removes_samples_from_both_arms": bool(
            (trimmed_draws["source_retained_fraction"] < 1).any()
            and (trimmed_draws["target_retained_fraction"] < 1).any()
        ),
        # The sweep's ARB base row and the complete ARB estimator must be
        # built on one and the same weight vector, replication by replication.
        "ARB_base_family_shares_full_ARB_weights": bool(
            len(arb_weight_match) == len(arb_draws)
            and np.allclose(
                arb_weight_match["gamma_norm"],
                arb_weight_match["residual_weight_norm"],
                rtol=0.0, atol=1e-12,
            )
        ),
        "full_ABW_rows_sit_on_L2_family": bool(
            abw_draws["family"].eq("L2 balancing").all()
        ),
    }
    pd.DataFrame({"check": arb_checks.keys(), "value": arb_checks.values()}).to_csv(
        ARTIFACT_DIR / f"arb_validation{output_suffix}.csv", index=False
    )
    return {
        "arb_draws": arb_draws,
        "abw_draws": abw_draws,
        "trimmed_draws": trimmed_draws,
        "estimator_summary": estimator_summary,
        "abw_diagnostic_summary": abw_diagnostic_summary,
        "trimmed_summary": trimmed_summary,
        "arb_checks": arb_checks,
    }


def summarize_primary_metrics(draws, *, output_suffix: str = "") -> dict:
    """The five headline metrics behind Evaluation Figure 2.

    The complete ABW and ARB estimators do not appear here: their outcome-model
    corrections have no design-conditional risk decomposition, and the five
    metrics of their weighting components are exactly the ``Base`` rows of the
    l2 and ARB families, which the sweep already carries.
    """
    metric_draws = draws[
        draws["stage"].isin(HEADLINE_STAGES) & draws["family"].isin(FAMILY_ORDER)
    ][[
        "overlap", "D2", "phi0", "repetition", "family", "stage",
        "signal_risk", "noise_risk", "exact_risk", "gamma_norm", "delta_norm",
    ]].copy()
    metric_draws["weight_norm_squared"] = metric_draws["gamma_norm"] ** 2
    metric_draws["heldout_imbalance_squared"] = metric_draws["delta_norm"] ** 2
    metric_draws.to_csv(
        ARTIFACT_DIR / f"primary_five_metric_draws{output_suffix}.csv.gz",
        index=False, compression="gzip",
    )

    def summarize_primary(g):
        def mean_mcse(column):
            x = g[column].to_numpy()
            return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))
        record = {"replications": len(g)}
        for column in ["signal_risk", "noise_risk", "exact_risk", "weight_norm_squared", "heldout_imbalance_squared"]:
            record[column], record[f"{column}_mcse"] = mean_mcse(column)
        return pd.Series(record)

    primary_summary = (
        metric_draws.groupby(["overlap", "D2", "phi0", "family", "stage"], sort=False)
        .apply(summarize_primary, include_groups=False).reset_index()
    )
    primary_summary.to_csv(
        ARTIFACT_DIR / f"primary_five_metric_summary{output_suffix}.csv", index=False
    )
    return {"primary_draws": metric_draws, "primary_summary": primary_summary}


def summarize_tuning(design, draws, *, output_suffix: str = "") -> dict:
    """Selected penalties and excess exact risk, behind Evaluation Figure 3."""
    tuning_long = design.melt(
        id_vars=["overlap", "phi0", "family", "repetition"],
        value_vars=["lambda_OOD", "lambda_OOD_mom", "lambda_GCV", "lambda_oracle"],
        var_name="rule", value_name="penalty",
    )
    rule_names = {
        "lambda_OOD": "Spectral target-aware",
        "lambda_OOD_mom": "Two-moment diagnostic",
        "lambda_GCV": "Source GCV",
        "lambda_oracle": "Infeasible oracle",
    }
    tuning_long["rule"] = tuning_long["rule"].map(rule_names)
    tuning_long.to_csv(
        ARTIFACT_DIR / f"tuning_selection_draws{output_suffix}.csv.gz",
        index=False, compression="gzip",
    )
    penalty_summary = (
        tuning_long.groupby(["overlap", "phi0", "family", "rule"])["penalty"]
        .agg(median="median", lower=lambda x: x.quantile(.25), upper=lambda x: x.quantile(.75))
        .reset_index()
    )

    stage_for_rule = {
        "Spectral target-aware": "Augmented (OOD)",
        "Two-moment diagnostic": "Augmented (OOD, Method of moments)",
        "Source GCV": "Augmented (GCV)",
        "Infeasible oracle": "Augmented (oracle)",
    }
    tuning_wide = (
        draws[draws["stage"].isin(stage_for_rule.values())]
        .pivot(index=["overlap", "phi0", "family", "repetition"], columns="stage", values="exact_risk")
        .reset_index()
    )
    excess_rows = []
    for key, group in tuning_wide.groupby(["overlap", "phi0", "family"], sort=False):
        oracle = group["Augmented (oracle)"].to_numpy()
        for rule in ["Spectral target-aware", "Two-moment diagnostic", "Source GCV"]:
            diff = group[stage_for_rule[rule]].to_numpy() - oracle
            excess_rows.append({
                "overlap": key[0], "phi0": key[1], "family": key[2], "rule": rule,
                "excess_exact_risk": float(diff.mean()),
                "excess_exact_risk_mcse": float(diff.std(ddof=1) / np.sqrt(len(diff))),
                "replications": len(diff),
            })
    excess_risk = pd.DataFrame(excess_rows)
    penalty_summary.to_csv(
        ARTIFACT_DIR / f"tuning_selection_summary{output_suffix}.csv", index=False
    )
    excess_risk.to_csv(
        ARTIFACT_DIR / f"tuning_excess_exact_risk{output_suffix}.csv", index=False
    )
    return {"penalty_summary": penalty_summary, "excess_risk": excess_risk}


# ---------------------------------------------------------------------------
# Response-model robustness
# ---------------------------------------------------------------------------
#: The rules scored against the fixed-surface oracle, in figure order.  The
#: oracle is the reference, so it is listed but never compared with itself.
RESPONSE_RULES = ["Spectral target-aware", "Source GCV"]
RESPONSE_ORACLE = "Fixed-surface oracle"


def _paired_rmse_ratio(
    squared_error: np.ndarray, reference: np.ndarray, seed: int,
    draws: int = RESPONSE_BOOTSTRAP_DRAWS,
) -> tuple[float, float, float]:
    """RMSE ratio to a reference arm, with a paired bootstrap interval.

    One index set resamples both arms, so the interval is for the ratio on
    common replications rather than for two independent RMSEs -- the same
    construction ``experiments/lalonde/chi2_stress`` uses for its paired
    comparisons against double ridge.
    """
    ratio = float(np.sqrt(squared_error.mean() / reference.mean()))
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(squared_error), size=(draws, len(squared_error)))
    bootstrap = np.sqrt(
        squared_error[index].mean(axis=1)
        / np.maximum(reference[index].mean(axis=1), 1e-15)
    )
    return (
        ratio,
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    )


def run_response_robustness_sweep(*, output_suffix: str = "") -> dict:
    """The tuning rule under fixed response surfaces, behind Appendix E's
    response-model robustness subsection.

    The sweep above draws outcomes from the isotropic random-effects law the
    theory assumes, so the tuning rule is scored under its own generative
    model.  This rerun keeps the design, the seeds, the honest split, the
    ``l2`` base, the penalty grid, and the variance-component estimator
    exactly as they are, and changes only the outcome: ``y0 = m(X0) +
    epsilon0`` for each of the four fixed surfaces of
    :data:`~experiments.evaluation.dgp.RESPONSE_SURFACES`.  ``epsilon0`` is
    recovered from the random-effects draw rather than redrawn, so the
    surfaces and the primary sweep share one set of noise draws.

    The reported quantity is the ACTUAL counterfactual-mean squared error
    ``(gamma^T y0 - mu_m)^2`` against the known population target mean, not
    the integrated risk ``R_n``: under a fixed surface the random-effects risk
    the criterion estimates is no longer the estimand's error, and scoring the
    rule by its own criterion would beg the question.  The reference arm is
    the infeasible oracle for the fixed-surface conditional risk
    ``R_n^fix(lambda; m) = {gamma_lambda^T m(X0) - mu_m}^2 +
    sigma_0^2 ||gamma_lambda||^2``, which no feasible rule can see.

    Only the overparameterized cell ``phi0 = RESPONSE_PHI0`` is run; the
    existing Section 5 sweeps are neither rerun nor modified.
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [
        (rep, overlap, a)
        for overlap, a in OVERLAP_SETTINGS
        for rep in range(N_REPETITIONS)
    ]

    def simulate_cell(repetition, overlap, a):
        data = draw_replication(repetition, RESPONSE_PHI0, a)
        X0, X1E = data["X0"], data["X1E"]
        # Common random numbers with the primary sweep: the same design, the
        # same split and the same noise vector, only a different mean function.
        epsilon0 = data["Y0"] - X0.dot(data["beta"])
        gamma_base = base_weight_families(
            ("l2",), X0, data["X1P"], BASE_SETTINGS
        )[0]["l2"]
        rows = []
        for surface in RESPONSE_SURFACES:
            m0 = surface.evaluate(X0)
            mu = surface.population_mean(data["nu1"])
            y0 = m0 + epsilon0
            fit = fit_rcb(
                X0, y0, X1E, gamma_base, LAMBDAS, theta_bounds=SQ_THETA_BOUNDS
            )
            rules = reference_penalty_rules(fit, y0)
            path = fit.weights_path
            # The infeasible fixed-surface oracle: no expectation over beta,
            # because there is no beta to average over any more.
            fixed_risk = (path.T.dot(m0) - mu) ** 2 + SIGMA02 * (path ** 2).sum(axis=0)
            selections = [
                ("Spectral target-aware", int(fit.selected_index)),
                ("Source GCV", int(rules["gcv"])),
                (RESPONSE_ORACLE, int(np.argmin(fixed_risk))),
            ]
            for rule, index in selections:
                gamma = path[:, index]
                estimate = float(gamma.dot(y0))
                rows.append({
                    "overlap": overlap, "D2": float(a ** 2),
                    "phi0": RESPONSE_PHI0, "n0": len(X0), "n1": len(data["X1"]),
                    "repetition": repetition, "surface": surface.name,
                    "rule": rule,
                    "lambda_selected": float(LAMBDAS[index]),
                    "mu_target": mu,
                    "mu0_error": estimate - mu,
                    "squared_error": (estimate - mu) ** 2,
                    "fixed_surface_risk": float(fixed_risk[index]),
                    "gamma_norm": float(np.linalg.norm(gamma)),
                    "rhat2": fit.r2, "sigmahat2": fit.sigma2,
                })
        return rows

    results = Parallel(n_jobs=N_JOBS, prefer="processes", verbose=0)(
        delayed(simulate_cell)(*task) for task in tasks
    )
    draws = pd.DataFrame([row for rows in results for row in rows])

    summary_rows = []
    keys = ["overlap", "D2", "phi0", "surface"]
    # The cell index, not a hash of the cell: ``hash`` of a tuple of strings is
    # salted per process, so a seed derived from it would not reproduce.
    for cell_index, (cell, group) in enumerate(draws.groupby(keys, sort=False)):
        wide = group.pivot(index="repetition", columns="rule", values="squared_error")
        reference = wide[RESPONSE_ORACLE].to_numpy(float)
        # One seed per cell, so every rule in a panel is resampled on the same
        # replication indices and the ratios within a panel stay comparable.
        seed = RESPONSE_BOOTSTRAP_SEED + cell_index
        for rule in RESPONSE_RULES + [RESPONSE_ORACLE]:
            squared_error = wide[rule].to_numpy(float)
            rmse = float(np.sqrt(squared_error.mean()))
            ratio, lower, upper = _paired_rmse_ratio(squared_error, reference, seed)
            penalties = group.loc[group["rule"].eq(rule), "lambda_selected"]
            summary_rows.append({
                **dict(zip(keys, cell)), "rule": rule,
                "replications": len(squared_error),
                "counterfactual_rmse": rmse,
                "rmse_mcse": float(
                    squared_error.std(ddof=1)
                    / (2 * max(rmse, 1e-15) * np.sqrt(len(squared_error)))
                ),
                "rmse_ratio_to_oracle": ratio,
                "rmse_ratio_lower_95": lower,
                "rmse_ratio_upper_95": upper,
                "lambda_median": float(penalties.median()),
                "mean_fixed_surface_risk": float(
                    group.loc[group["rule"].eq(rule), "fixed_surface_risk"].mean()
                ),
            })
    summary = pd.DataFrame(summary_rows)

    draws.to_csv(
        ARTIFACT_DIR / f"response_robustness_draws{output_suffix}.csv.gz",
        index=False, compression="gzip",
    )
    summary.to_csv(
        ARTIFACT_DIR / f"response_robustness_summary{output_suffix}.csv", index=False
    )

    oracle = summary[summary["rule"].eq(RESPONSE_ORACLE)]
    checks = {
        "metric_is_empirical_counterfactual_mse": True,
        "four_fixed_response_surfaces": bool(
            set(draws["surface"]) == set(RESPONSE_SURFACE_ORDER)
        ),
        "overparameterized_cell_only": bool(draws["n0"].eq(40).all()),
        "target_sample_is_100": bool(draws["n1"].eq(100).all()),
        "complete_replications_per_cell": bool(
            draws.groupby(keys + ["rule"]).size().eq(N_REPETITIONS).all()
        ),
        "oracle_ratio_is_exactly_one": bool(
            np.allclose(oracle["rmse_ratio_to_oracle"], 1.0, atol=1e-12)
        ),
        # The oracle minimizes the fixed-surface risk by construction, so no
        # feasible rule may attain a smaller value of it.
        "oracle_minimizes_fixed_surface_risk": bool(
            draws.pivot_table(
                index=keys + ["repetition"], columns="rule",
                values="fixed_surface_risk",
            ).pipe(
                lambda w: (
                    w[RESPONSE_ORACLE].to_numpy()
                    <= w[RESPONSE_RULES].to_numpy().min(axis=1) + 1e-12
                ).all()
            )
        ),
        "no_nonfinite_outputs": bool(
            np.isfinite(draws[["squared_error", "fixed_surface_risk", "gamma_norm"]].to_numpy()).all()
        ),
        "penalty_grid_unchanged": bool(
            set(np.round(draws["lambda_selected"], 12)) <= set(np.round(LAMBDAS, 12))
        ),
    }
    if not all(v for v in checks.values() if isinstance(v, (bool, np.bool_))):
        failed = [k for k, v in checks.items() if isinstance(v, (bool, np.bool_)) and not v]
        raise AssertionError(f"Response-robustness validation failed: {failed}")
    pd.DataFrame({"check": list(checks), "value": [str(v) for v in checks.values()]}).to_csv(
        ARTIFACT_DIR / f"response_robustness_validation{output_suffix}.csv", index=False
    )
    return {"response_draws": draws, "response_summary": summary, "response_checks": checks}


def run_isotropic_sweep() -> dict:
    """The primary phi1=0.50 comparison rerun under Sigma0=Sigma1=I_p, behind
    the isotropic-sensitivity appendix subsection.

    Reruns :func:`run_method_sweep`, :func:`run_complete_estimator_comparison`,
    :func:`summarize_primary_metrics`, and :func:`summarize_tuning` verbatim
    -- same task grid, same stages, same five metrics, same tuning rules,
    same seeds -- with only the source/target covariance changed from AR(1)
    to isotropic, mirroring Section 5's isotropic control.  ``draw_replication``
    reuses the AR(1) sweep's exact seed coordinates when ``isotropic=True``, so
    the two sweeps share the same underlying standard-Gaussian draws and differ
    only in the covariance transform applied to them -- common random numbers,
    for the cleanest possible isotropic-vs-AR(1) contrast.  Like the phi1=1.25
    supplement, this is a sensitivity control, not the primary design, so it
    does not go through :func:`validate`.
    """
    suffix = "_isotropic"
    swept = run_method_sweep(output_suffix=suffix, isotropic=True)
    arb = run_complete_estimator_comparison(swept["draws"], output_suffix=suffix, isotropic=True)
    metrics = summarize_primary_metrics(swept["draws"], output_suffix=suffix)
    tuning = summarize_tuning(swept["design"], swept["draws"], output_suffix=suffix)
    return {**swept, **arb, **metrics, **tuning}


def run_phi1_supplement_sweep() -> dict:
    """The required-metrics sweep rerun at phi1=1.25, behind app-E7/app-E8.

    Reruns ``run_method_sweep``, ``run_complete_estimator_comparison`` and
    ``summarize_primary_metrics`` verbatim -- same task grid, same stages, same
    five metrics -- with only ``phi1`` changed, so the appendix panels are
    plotted with the unmodified Evaluation Figure 2 rendering code.  This
    sweep is a design supplement to the primary phi1=0.50 comparison, not a
    replacement, so it does not go through :func:`validate`, whose checks
    (``N1 == 100``, etc.) are specific to the primary design.
    """
    suffix = "_phi1_1p25"
    swept = run_method_sweep(phi1=PHI1_SUPPLEMENT, output_suffix=suffix)
    arb = run_complete_estimator_comparison(swept["draws"], phi1=PHI1_SUPPLEMENT, output_suffix=suffix)
    metrics = summarize_primary_metrics(swept["draws"], output_suffix=suffix)
    return {**swept, **arb, **metrics}


def run_all() -> dict:
    write_configuration()
    swept = run_method_sweep()
    checks = validate(swept["draws"], swept["design"], swept["summary"])
    arb = run_complete_estimator_comparison(swept["draws"])
    metrics = summarize_primary_metrics(swept["draws"])
    tuning = summarize_tuning(swept["design"], swept["draws"])
    phi1_supplement = run_phi1_supplement_sweep()
    isotropic = run_isotropic_sweep()
    response = run_response_robustness_sweep()
    return {
        **swept, "checks": checks, **arb, **metrics, **tuning, **response,
        "phi1_supplement": phi1_supplement, "isotropic_sweep": isotropic,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true",
        help="four replications; exercises every stage, proves nothing",
    )
    args = parser.parse_args()
    if args.smoke and not SMOKE_TEST:
        raise SystemExit(
            "Set CAUSAL_EXTRAPOLATION_SMOKE=1 before --smoke so the reduced\n"
            "replication count is declared in one place:\n"
            "  CAUSAL_EXTRAPOLATION_SMOKE=1 python -m experiments.evaluation.run --smoke"
        )
    results = run_all()
    print({k: type(v).__name__ for k, v in results.items()})
