#!/usr/bin/env python3
"""Complete, audited validation suite for the LaLonde real-data section."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import experiment as E
from . import methods as M
from rcb import paths


OUT = paths.results("lalonde", create=False)
DATA = M.DATA_DIR
CORE_METHODS = [
    "Pure weighting: uniform",
    "Pure weighting: l2",
    "Double ridge: outcome CV",
    "Double ridge: imbalance CV",
    "Double ridge: Riesz CV",
    "Ours: uniform base, full target",
    "Ours: l2 base, pilot/evaluation",
]
EXTENDED_METHODS = [
    "Pure weighting: entropy",
    "Ours: entropy base, pilot/evaluation",
    "ARB",
]
PRIMARY_METHODS = CORE_METHODS + EXTENDED_METHODS


def load_train(path: Path) -> dict:
    df = pd.read_csv(path)
    features = [c for c in df if c.startswith("x")]
    treat = df.treat.to_numpy(bool)
    return {
        "X0": df.loc[~treat, features].to_numpy(float),
        "y0": df.loc[~treat, "y"].to_numpy(float),
        "Xt": df.loc[treat, features].to_numpy(float),
        "yt": df.loc[treat, "y"].to_numpy(float),
        "features": features,
    }


def load_external_target(path: Path) -> dict:
    df = pd.read_csv(path)
    features = [c for c in df if c.startswith("x")]
    return {
        "Xt": df[features].to_numpy(float),
        "yt": df.y.to_numpy(float),
        "features": features,
    }


def _weight_diagnostics(X0: np.ndarray, Xt: np.ndarray, w: np.ndarray) -> dict:
    delta = Xt.mean(axis=0) - X0.T @ w
    d = E.diagnostics(X0, Xt, w)
    return {
        **d,
        "weight_norm_squared": float(w @ w),
        "residual_imbalance_squared": float(delta @ delta),
    }


def pilot_evaluation_adjustment(
    X0: np.ndarray,
    y0: np.ndarray,
    Xt: np.ndarray,
    seed: int,
    base: str = "l2",
    kappa: float = 1.0,
) -> dict:
    """Algorithm 2: one independent, approximately equal target split.

    The pilot observations construct the fixed-kappa ridge base.  The disjoint
    evaluation observations estimate and minimize the target-risk path.  This
    function returns the estimator from that one split; it does not swap folds
    or average estimators across partitions.
    """
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(Xt))
    pilot_n = len(Xt) // 2
    pilot = permutation[:pilot_n]
    evaluation = permutation[pilot_n:]
    if np.intersect1d(pilot, evaluation).size:
        raise AssertionError("Pilot/evaluation target leakage")
    if len(pilot) + len(evaluation) != len(Xt):
        raise AssertionError("Pilot/evaluation split does not exhaust target")

    Xc = X0 - X0.mean(axis=0)
    if base == "l2":
        shift_pilot = Xt[pilot].mean(axis=0) - X0.mean(axis=0)
        base_weights = M.l2_balancing_weights(Xc, shift_pilot, kappa)
        base_diagnostics = {"base": "l2", "kappa": kappa, "success": True}
    elif base == "entropy":
        base_weights, entropy_diag = M.regularized_entropy_base(
            X0, Xt[pilot]
        )
        base_diagnostics = {"base": "entropy", **entropy_diag}
        if not entropy_diag["success"]:
            raise RuntimeError("Regularized entropy pilot-base optimization failed")
    else:
        raise ValueError(f"Unknown pilot base: {base}")
    if abs(base_weights.sum() - 1.0) > 1e-8:
        raise AssertionError("Pilot base is not normalized")
    prepared = E.prepare_adjustment_paths(
        X0, Xt[evaluation], {"adaptive": base_weights}
    )
    r2, sigma2 = E.estimate_risk_nuisance(X0, y0, prepared["geometry"])
    result = E.select_prepared_adjustment(
        prepared["paths"]["adaptive"], y0, r2, sigma2
    )
    return {
        "weights": result["weights"],
        "muhat0": result["muhat0"],
        "selected_lambda": result["selected_lambda"],
        "selected_no_adjustment": result["selected_no_adjustment_endpoint"],
        "selected_lower_boundary": result["selected_index"] == 0,
        "selected_risk": float(result["risk"][result["selected_index"]]),
        "pilot_n": len(pilot),
        "evaluation_n": len(evaluation),
        "pilot_evaluation_overlap": int(
            np.intersect1d(pilot, evaluation).size
        ),
        "pilot_indices": pilot,
        "evaluation_indices": evaluation,
        "base_diagnostics": base_diagnostics,
    }


def fit_primary(
    X0: np.ndarray,
    y0: np.ndarray,
    Xt: np.ndarray,
    seed: int,
) -> list[dict]:
    """Fit primary methods using target covariates only."""
    n0 = len(X0)
    Xc = X0 - X0.mean(axis=0)
    shift = Xt.mean(axis=0) - X0.mean(axis=0)
    lam, _ = E.outcome_cv_fixed_folds_raw_grid(Xc, y0)
    tuners = M.design_only_balance_penalty_cv(X0, shift)
    uniform = np.full(n0, 1.0 / n0)
    geometry = E.prepare_adjustment_paths(
        X0, Xt, {"uniform": uniform}
    )
    r2, sigma2 = E.estimate_risk_nuisance(X0, y0, geometry["geometry"])
    ours_uniform = E.select_prepared_adjustment(
        geometry["paths"]["uniform"], y0, r2, sigma2
    )
    adaptive_l2 = pilot_evaluation_adjustment(
        X0, y0, Xt, seed=seed, base="l2", kappa=1.0
    )
    w_l2 = M.l2_balancing_weights(Xc, shift, delta=1.0)
    fitted = [
        ("Pure weighting: uniform", uniform, np.nan, np.nan, False, False),
        ("Pure weighting: l2", w_l2, np.nan, np.nan, False, False),
    ]
    for name, delta, upper in [
        ("Double ridge: outcome CV", lam, np.isclose(lam, 300.0 / n0)),
        (
            "Double ridge: imbalance CV",
            tuners["delta_cv_imbalance"],
            np.isclose(tuners["delta_cv_imbalance"], 20.0),
        ),
        (
            "Double ridge: Riesz CV",
            tuners["delta_cv_riesz"],
            np.isclose(tuners["delta_cv_riesz"], 1.0),
        ),
    ]:
        result = M.double_ridge_estimate(Xc, y0, shift, delta, lam)
        fitted.append((name, result["augmented_weights"], delta, np.nan,
                       False, bool(upper)))
    fitted.extend([
        (
            "Ours: uniform base, full target",
            ours_uniform["weights"],
            ours_uniform["selected_lambda"],
            float(ours_uniform["risk"][ours_uniform["selected_index"]]),
            ours_uniform["selected_no_adjustment_endpoint"],
            False,
        ),
        (
            "Ours: l2 base, pilot/evaluation",
            adaptive_l2["weights"], adaptive_l2["selected_lambda"],
            adaptive_l2["selected_risk"],
            adaptive_l2["selected_no_adjustment"],
            False,
        ),
    ])
    rows = []
    for name, weights, selected, risk, no_adjustment, upper in fitted:
        weights = np.asarray(weights, float).copy()
        affine_correction = float((weights.sum() - 1.0) / len(weights))
        # Project onto the affine-normalization hyperplane.  This is
        # numerically immaterial for the regularized estimators and stabilizes
        # the rank-deficient OLS pseudoinverse in bootstrap samples.
        weights -= affine_correction
        diag = _weight_diagnostics(X0, Xt, weights)
        rows.append({
            "method": name,
            "muhat0": float(weights @ y0),
            "selected_lambda": selected,
            "estimated_target_risk": risk,
            "selected_no_adjustment": float(no_adjustment),
            "selected_lower_boundary": float(
                np.isfinite(selected) and np.isclose(selected, E.PENALTY_GRID[0])
            ),
            "selected_upper_boundary": float(upper),
            "pilot_n": (
                adaptive_l2["pilot_n"]
                if name == "Ours: l2 base, pilot/evaluation"
                else np.nan
            ),
            "evaluation_n": (
                adaptive_l2["evaluation_n"]
                if name == "Ours: l2 base, pilot/evaluation"
                else np.nan
            ),
            "pilot_evaluation_overlap": (
                adaptive_l2["pilot_evaluation_overlap"]
                if name == "Ours: l2 base, pilot/evaluation"
                else np.nan
            ),
            "affine_projection_correction": abs(affine_correction),
            **diag,
        })
    if max(r["weight_sum_error"] for r in rows) > 1e-8:
        raise AssertionError("Affine weight normalization failed")
    return rows


def fit_extended(
    X0: np.ndarray,
    y0: np.ndarray,
    Xt: np.ndarray,
    seed: int,
    include_arb: bool = True,
) -> list[dict]:
    """Fit the entropy pair and the full ARB comparator.

    "Pure weighting" means that the base weights alone are applied to source
    outcomes.  In 171 dimensions exact (unregularized) entropy balance is not
    feasible on this sample, so the entropy baseline uses the pre-specified
    regularized entropy program in ``methods``.  Our entropy-base
    adjustment uses the same program on the pilot subset and Algorithm 2 on the
    disjoint evaluation subset.
    """
    entropy_weights, entropy_diag = M.regularized_entropy_base(X0, Xt)
    if not entropy_diag["success"]:
        raise RuntimeError("Regularized entropy weighting optimization failed")
    entropy_adjusted = pilot_evaluation_adjustment(
        X0, y0, Xt, seed=seed, base="entropy", kappa=1.0
    )

    specifications = [
        {
            "method": "Pure weighting: entropy",
            "weights": entropy_weights,
            "muhat0": float(entropy_weights @ y0),
            "selected_lambda": np.nan,
            "estimated_target_risk": np.nan,
            "selected_no_adjustment": 0.0,
            "selected_lower_boundary": 0.0,
            "selected_upper_boundary": 0.0,
            "pilot_n": np.nan,
            "evaluation_n": np.nan,
            "pilot_evaluation_overlap": np.nan,
            "diagnostic_weight_role": "final weighting estimator",
            "estimate_equals_weighted_source_outcome": True,
            "entropy_optimizer_success": True,
            "elastic_net_alpha_1se": np.nan,
        },
        {
            "method": "Ours: entropy base, pilot/evaluation",
            "weights": entropy_adjusted["weights"],
            "muhat0": entropy_adjusted["muhat0"],
            "selected_lambda": entropy_adjusted["selected_lambda"],
            "estimated_target_risk": entropy_adjusted["selected_risk"],
            "selected_no_adjustment": float(
                entropy_adjusted["selected_no_adjustment"]
            ),
            "selected_lower_boundary": float(
                entropy_adjusted["selected_lower_boundary"]
            ),
            "selected_upper_boundary": 0.0,
            "pilot_n": entropy_adjusted["pilot_n"],
            "evaluation_n": entropy_adjusted["evaluation_n"],
            "pilot_evaluation_overlap": entropy_adjusted[
                "pilot_evaluation_overlap"
            ],
            "diagnostic_weight_role": "final adjusted weights",
            "estimate_equals_weighted_source_outcome": True,
            "entropy_optimizer_success": True,
            "elastic_net_alpha_1se": np.nan,
        },
    ]
    if include_arb:
        arb = M.arb_estimate(X0, Xt, y0, seed=seed)
        specifications.append({
            "method": "ARB",
            "weights": arb["weights"],
            "muhat0": arb["muhat0"],
            "selected_lambda": arb["elastic_net_alpha_1se"],
            "estimated_target_risk": np.nan,
            "selected_no_adjustment": 0.0,
            "selected_lower_boundary": 0.0,
            "selected_upper_boundary": 0.0,
            "pilot_n": np.nan,
            "evaluation_n": np.nan,
            "pilot_evaluation_overlap": np.nan,
            "diagnostic_weight_role": "ARB residual-correction weights",
            "estimate_equals_weighted_source_outcome": False,
            "entropy_optimizer_success": np.nan,
            "elastic_net_alpha_1se": arb["elastic_net_alpha_1se"],
        })
    rows = []
    for spec in specifications:
        weights = np.asarray(spec.pop("weights"), float).copy()
        affine_correction = float((weights.sum() - 1.0) / len(weights))
        weights -= affine_correction
        rows.append({
            **spec,
            "affine_projection_correction": abs(affine_correction),
            **_weight_diagnostics(X0, Xt, weights),
        })
    if max(r["weight_sum_error"] for r in rows) > 1e-8:
        raise AssertionError("Extended-method weight normalization failed")
    return rows


def _summarize_bootstrap(draws: pd.DataFrame, score: bool) -> pd.DataFrame:
    rows = []
    for method, g in draws.groupby("method", sort=False):
        finite = g.loc[np.isfinite(g.selected_lambda), "selected_lambda"]
        row = {
            "method": method,
            "reps": len(g),
            "muhat0_mean": g.muhat0.mean(),
            "muhat0_sd": g.muhat0.std(ddof=1),
            "muhat0_q025": g.muhat0.quantile(.025),
            "muhat0_q975": g.muhat0.quantile(.975),
            "median_finite_lambda": finite.median() if len(finite) else np.nan,
            "lambda_iqr": (
                finite.quantile(.75) - finite.quantile(.25)
                if len(finite) else np.nan
            ),
            "no_adjustment_rate": g.selected_no_adjustment.mean(),
            "lower_boundary_rate": g.selected_lower_boundary.mean(),
            "upper_boundary_rate": g.selected_upper_boundary.mean(),
            "ess_mean": g.ess.mean(),
            "weight_norm_squared_mean": g.weight_norm_squared.mean(),
            "imbalance_squared_mean": g.residual_imbalance_squared.mean(),
            "negative_weight_fraction_mean": g.negative_weight_fraction.mean(),
            "max_weight_sum_error": g.weight_sum_error.max(),
            "max_pilot_evaluation_overlap": (
                g.pilot_evaluation_overlap.max()
                if g.pilot_evaluation_overlap.notna().any() else np.nan
            ),
            "failure_rate": 0.0,
        }
        if "att" in g:
            row.update({
                "att_mean": g.att.mean(), "att_sd": g.att.std(ddof=1),
                "att_q025": g.att.quantile(.025),
                "att_q975": g.att.quantile(.975),
            })
        if score:
            sq = g.squared_error.to_numpy()
            rmse = float(np.sqrt(sq.mean()))
            paired = g.paired_mse_difference.to_numpy()
            row.update({
                "bias": g.error.mean(),
                "rmse": rmse,
                "rmse_mcse": sq.std(ddof=1) / np.sqrt(len(g)) / (2 * rmse),
                "mae": g.error.abs().mean(),
                "paired_mse_difference": paired.mean(),
                "paired_mse_difference_mcse": paired.std(ddof=1) / np.sqrt(len(g)),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _core_replication(
    train: dict,
    target: dict,
    source_index: np.ndarray,
    target_index: np.ndarray,
    replication: int,
    seed: int,
    label: str,
    score: bool,
) -> dict:
    """One bootstrap replication of the core method set.

    A pure function of its two index rows, which is what makes the sweep
    parallelizable without changing a single number: the resamples are drawn up
    front from one generator, so no replication depends on any other.
    """
    X0 = train["X0"][source_index]
    y0 = train["y0"][source_index]
    Xt = target["Xt"][target_index]
    try:
        fitted = fit_primary(X0, y0, Xt, seed)
        truth = float(target["yt"][target_index].mean()) if score else np.nan
        ybar = float(target["yt"][target_index].mean()) if not score else np.nan
        rows = []
        for result in fitted:
            row = {"replication": replication, "experiment": label, **result}
            if score:
                error = result["muhat0"] - truth
                row.update({
                    "truth_mean": truth, "error": error,
                    "squared_error": error**2,
                })
            else:
                row.update({"ybar_t": ybar, "att": ybar - result["muhat0"]})
            rows.append(row)
        return {"rows": rows, "failure": None}
    except Exception as exc:  # retained in audit, never silently dropped
        return {"rows": [], "failure": {"replication": replication,
                                        "error": repr(exc)}}


def run_bootstrap(
    train: dict,
    target: dict,
    reps: int,
    seed: int,
    label: str,
    score: bool,
    jobs: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Paired two-sample bootstrap with target outcomes isolated until score."""
    rng = np.random.default_rng(seed)
    source_indices = rng.integers(0, len(train["y0"]), (reps, len(train["y0"])))
    target_indices = rng.integers(0, len(target["yt"]), (reps, len(train["Xt"])))
    start = time.time()
    results = Parallel(n_jobs=jobs, prefer="processes", verbose=10)(
        delayed(_core_replication)(
            train, target, source_indices[b], target_indices[b], b,
            seed + 100000 + b, label, score,
        )
        for b in range(reps)
    )
    # Collected in replication order regardless of completion order.
    rows = [row for result in results for row in result["rows"]]
    failures = [result["failure"] for result in results
                if result["failure"] is not None]
    print(f"{label}: {reps}/{reps} draws; failures={len(failures)}", flush=True)
    draws = pd.DataFrame(rows)
    if failures:
        raise RuntimeError(f"{label} had numerical failures: {failures[:3]}")
    if score:
        ref = draws.loc[
            draws.method == "Double ridge: outcome CV",
            ["replication", "squared_error"],
        ].rename(columns={"squared_error": "reference_squared_error"})
        draws = draws.merge(ref, on="replication")
        draws["paired_mse_difference"] = (
            draws.squared_error - draws.reference_squared_error
        )
    summary = _summarize_bootstrap(draws, score=score)
    index_path = OUT / f"{label}_bootstrap_indices.npz"
    np.savez_compressed(index_path, source=source_indices, target=target_indices)
    meta = {
        "experiment": label, "reps": reps, "seed": seed,
        "source_n": len(train["y0"]), "target_pool_n": len(target["yt"]),
        "target_draw_n": len(train["Xt"]), "failures": failures,
        "elapsed_seconds": time.time() - start,
        "indices_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "target_outcomes_passed_to_fit": False,
    }
    return summary, draws, meta


def _extended_replication(
    train: dict,
    target: dict,
    source_index: np.ndarray,
    target_index: np.ndarray,
    replication: int,
    seed: int,
    label: str,
    score: bool,
    include_arb: bool,
) -> dict:
    """One independently refitted entropy/ARB replication."""
    X0 = train["X0"][source_index]
    y0 = train["y0"][source_index]
    Xt = target["Xt"][target_index]
    try:
        fitted = fit_extended(X0, y0, Xt, seed=seed, include_arb=include_arb)
        truth = float(target["yt"][target_index].mean())
        rows = []
        for result in fitted:
            row = {"replication": replication, "experiment": label, **result}
            if score:
                error = result["muhat0"] - truth
                row.update({
                    "truth_mean": truth,
                    "error": error,
                    "squared_error": error**2,
                })
            else:
                row.update({"ybar_t": truth, "att": truth-result["muhat0"]})
            rows.append(row)
        return {"rows": rows, "error": None}
    except Exception as exc:
        return {
            "rows": [],
            "error": {"replication": replication, "error": repr(exc)},
        }


def extend_saved_bootstrap(
    train: dict,
    target: dict,
    reps: int,
    seed: int,
    label: str,
    score: bool,
    jobs: int,
    include_arb: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Add entropy and ARB using the first saved paired bootstrap draws."""
    with np.load(OUT / f"{label}_bootstrap_indices.npz") as z:
        source_indices, target_indices = z["source"], z["target"]
    if reps > len(source_indices):
        raise ValueError(
            f"Requested {reps} extended reps but only {len(source_indices)} "
            f"saved paired draws exist for {label}"
        )
    requested_methods = [
        "Pure weighting: entropy",
        "Ours: entropy base, pilot/evaluation",
    ] + (["ARB"] if include_arb else [])
    draw_path = OUT / f"{label}_draws.csv.gz"
    existing = pd.read_csv(draw_path)
    existing_counts = existing.groupby("method").size()
    if all(existing_counts.get(method, 0) >= reps for method in requested_methods):
        combined = existing.loc[
            existing.method.isin(CORE_METHODS + requested_methods)
        ].copy()
        summary = _summarize_bootstrap(combined, score=score)
        return summary, combined, {
            "experiment": label,
            "requested_reps": reps,
            "completed_reps_per_extended_method": {
                method: int(existing_counts[method]) for method in requested_methods
            },
            "jobs": jobs,
            "failures": [],
            "paired_indices_reused_from_core_experiment": True,
            "target_outcomes_passed_to_fit": False,
            "resumed_from_complete_saved_draws": True,
            "arb_included": include_arb,
        }
    results = Parallel(n_jobs=jobs, prefer="processes", verbose=10)(
        delayed(_extended_replication)(
            train, target, source_indices[b], target_indices[b], b,
            seed + 500000 + b, label, score, include_arb,
        )
        for b in range(reps)
    )
    failures = [r["error"] for r in results if r["error"] is not None]
    extended_rows = [row for result in results for row in result["rows"]]
    extended = pd.DataFrame(extended_rows)
    core = existing
    core = core.loc[core.method.isin(CORE_METHODS)].copy()
    if score and len(extended):
        reference = core.loc[
            core.method == "Double ridge: outcome CV",
            ["replication", "squared_error"],
        ].rename(columns={"squared_error": "reference_squared_error"})
        extended = extended.merge(reference, on="replication", how="left")
        if extended.reference_squared_error.isna().any():
            raise AssertionError("Missing paired double-ridge reference")
        extended["paired_mse_difference"] = (
            extended.squared_error - extended.reference_squared_error
        )
    combined = pd.concat([core, extended], ignore_index=True, sort=False)
    combined.to_csv(draw_path, index=False, compression="gzip")
    summary = _summarize_bootstrap(combined, score=score)
    for method in requested_methods:
        summary.loc[summary.method == method, "failure_rate"] = (
            len(failures) / reps
        )
    meta = {
        "experiment": label,
        "requested_reps": reps,
        "completed_reps_per_extended_method": (
            extended.groupby("method").size().to_dict()
            if len(extended) else {}
        ),
        "jobs": jobs,
        "failures": failures,
        "paired_indices_reused_from_core_experiment": True,
        "target_outcomes_passed_to_fit": False,
        "arb_included": include_arb,
    }
    return summary, combined, meta


def full_point_table(
    train: dict, seed: int, label: str, include_extended: bool = True
) -> pd.DataFrame:
    rows = fit_primary(train["X0"], train["y0"], train["Xt"], seed)
    if include_extended:
        rows.extend(fit_extended(
            train["X0"], train["y0"], train["Xt"], seed
        ))
    for row in rows:
        row["att"] = float(train["yt"].mean() - row["muhat0"])
        row["representation"] = label
    return pd.DataFrame(rows)


def single_split_stability(
    train: dict, repeats: int = 50
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sensitivity across separate Algorithm-2 pilot/evaluation splits.

    Every row is a complete estimator from one split.  The rows are summarized
    only to quantify partition sensitivity; weights or estimates are never
    averaged into a new estimator.
    """
    X0, y0, Xt = train["X0"], train["y0"], train["Xt"]
    rows = []
    for base in ("l2", "entropy"):
        for b in range(repeats):
            seed = E.SEED + 30000 + b
            result = pilot_evaluation_adjustment(
                X0, y0, Xt, seed=seed, base=base, kappa=1.0
            )
            rows.append({
                "base": base,
                "partition": b,
                "seed": seed,
                "pilot_n": result["pilot_n"],
                "evaluation_n": result["evaluation_n"],
                "pilot_evaluation_overlap": result["pilot_evaluation_overlap"],
                "muhat0": result["muhat0"],
                "att": float(train["yt"].mean() - result["muhat0"]),
                "selected_lambda": result["selected_lambda"],
                "selected_no_adjustment": result["selected_no_adjustment"],
                "selected_lower_boundary": result["selected_lower_boundary"],
            })
    draws = pd.DataFrame(rows)
    summary_rows = []
    for base, g in draws.groupby("base", sort=False):
        finite = g.loc[np.isfinite(g.selected_lambda), "selected_lambda"]
        summary_rows.append({
            "base": base,
            "partitions": len(g),
            "pilot_n": int(g.pilot_n.iloc[0]),
            "evaluation_n": int(g.evaluation_n.iloc[0]),
            "att_mean": g.att.mean(),
            "att_sd": g.att.std(ddof=1),
            "att_min": g.att.min(),
            "att_max": g.att.max(),
            "median_finite_lambda": finite.median() if len(finite) else np.nan,
            "lambda_q25": finite.quantile(.25) if len(finite) else np.nan,
            "lambda_q75": finite.quantile(.75) if len(finite) else np.nan,
            "no_adjustment_rate": g.selected_no_adjustment.mean(),
            "lower_boundary_rate": g.selected_lower_boundary.mean(),
            "maximum_pilot_evaluation_overlap": (
                g.pilot_evaluation_overlap.max()
            ),
        })
    summary = pd.DataFrame(summary_rows)
    return summary, draws


def run_partition_artifacts() -> None:
    train = load_train(DATA / "lalonde_common_support_p171.csv")
    r4s, r4d = single_split_stability(train, repeats=50)
    save_table("r4_single_split_stability_summary.csv", r4s)
    save_table("r4_single_split_stability_draws.csv", r4d)


#: R8 reads a draws file that nothing in this repository writes.
#:
#: ``lalonde_cv_fold_stability_draws.csv`` sits in ``results/`` and is read
#: here, but no script produces it -- not this one, and not the pre-refactor
#: ``lalonde171_complete_validation.py`` either, which read it from
#: ``results/support/`` exactly the same way.  It came from a script that was
#: never committed.  R8 therefore cannot be regenerated from a clean tree; it
#: survives only because the file is committed beside the code.
#:
#: Nothing in ``main.tex`` cites R8, so this costs the manuscript nothing, but
#: it is a real hole in reproducibility and is recorded rather than patched
#: over: reconstructing the file would mean inventing a procedure and claiming
#: it was the one that produced the committed numbers.
CV_FOLD_DRAWS_HAVE_NO_WRITER = True


def cv_boundary_summary() -> pd.DataFrame | None:
    path = OUT / "cv_fold_stability_draws.csv"
    if not path.exists():
        print(
            f"R8 skipped: {path.name} is absent and nothing in this repository"
            "\nwrites it -- see CV_FOLD_DRAWS_HAVE_NO_WRITER in validation.py.",
            flush=True,
        )
        return None
    d = pd.read_csv(path)
    rows = []
    for method, g in d.groupby("method", sort=False):
        rows.append({
            "method": method, "reps": len(g),
            "outcome_cv_top_grid_rate": np.mean(
                np.isclose(g.outcome_cv_lambda, 300.0/727)
            ),
            "outcome_cv_bottom_grid_rate": np.mean(
                np.isclose(g.outcome_cv_lambda, 1e-10/727)
            ),
            "outcome_cv_unique_values": g.outcome_cv_lambda.nunique(),
            "downstream_no_adjustment_rate": np.mean(
                np.isinf(g.selected_lambda)
            ),
            "downstream_lower_grid_rate": np.mean(
                np.isclose(g.selected_lambda, E.PENALTY_GRID[0])
            ),
            "att_mean": g.att.mean(), "att_sd": g.att.std(ddof=1),
        })
    return pd.DataFrame(rows)


def save_table(name: str, obj: pd.DataFrame) -> None:
    obj.to_csv(OUT / name, index=False)


def run_feature_observational(reps: int) -> dict:
    """Complete the p=11/14/171 observational uncertainty comparison."""
    summaries = []
    metadata = {}
    p171_path = OUT / "r2_observational_summary.csv"
    if p171_path.exists():
        p171 = pd.read_csv(p171_path)
        p171.insert(0, "representation", "p171")
        summaries.append(p171)
    for p in (11, 14):
        tr = load_train(DATA / f"lalonde_common_support_p{p}.csv")
        summary, draws, meta = run_bootstrap(
            tr, tr, reps, E.SEED + 21000,
            f"r6_observational_p{p}", False,
        )
        summary.insert(0, "representation", f"p{p}")
        summaries.append(summary)
        save_table(f"r6_observational_p{p}_summary.csv", summary)
        draws.to_csv(
            OUT / f"r6_observational_p{p}_draws.csv.gz",
            index=False, compression="gzip",
        )
        metadata[f"r6_observational_p{p}"] = meta
    save_table(
        "r6_observational_summary.csv",
        pd.concat(summaries, ignore_index=True),
    )
    return metadata


def run_feature_hidden(reps: int) -> dict:
    """Use exactly the R3 indices for every feature representation."""
    summaries = []
    metadata = {}
    p171_path = OUT / "r3_hidden_outcome_summary.csv"
    if p171_path.exists():
        summaries.append(
            pd.read_csv(p171_path).assign(representation="p171")
        )
    for p in (11, 14):
        tr = load_train(DATA / f"lalonde_common_support_p{p}.csv")
        target = load_external_target(
            DATA / f"lalonde_nsw_control_target_p{p}.csv"
        )
        summary, draws, meta = run_bootstrap(
            tr, target, reps, E.SEED + 22000,
            f"r6_hidden_p{p}", True,
        )
        summary.insert(0, "representation", f"p{p}")
        summaries.append(summary)
        save_table(f"r6_hidden_p{p}_summary.csv", summary)
        draws.to_csv(
            OUT / f"r6_hidden_p{p}_draws.csv.gz",
            index=False, compression="gzip",
        )
        metadata[f"r6_hidden_p{p}"] = meta
    save_table(
        "r6_hidden_outcome_summary.csv",
        pd.concat(summaries, ignore_index=True),
    )
    return metadata


def run_feature_points() -> None:
    """Use the same pre-specified target split for p=11, 14, and 171."""
    tables = []
    r1_path = OUT / "r1_point_estimates.csv"
    if r1_path.exists():
        tables.append(pd.read_csv(r1_path).assign(representation="p171"))
    else:
        tr = load_train(DATA / "lalonde_common_support_p171.csv")
        tables.append(full_point_table(tr, E.SEED + 20000, "p171"))
    for p in (11, 14):
        tr = load_train(DATA / f"lalonde_common_support_p{p}.csv")
        tables.append(full_point_table(tr, E.SEED + 20000, f"p{p}"))
    save_table("r6_point_estimates.csv", pd.concat(tables, ignore_index=True))


def run_extended_suite(reps: int, jobs: int) -> dict:
    """Add the entropy pair and ARB to saved paired experiments."""
    metadata = {}
    main_train = load_train(DATA / "lalonde_common_support_p171.csv")
    external171 = load_external_target(
        DATA / "lalonde_nsw_control_target_p171.csv"
    )
    experiments = [
        (
            "r2_observational", main_train, main_train, False,
            E.SEED + 21000, "r2_observational_summary.csv", True,
        ),
        (
            "r3_hidden_outcome", main_train, external171, True,
            E.SEED + 22000, "r3_hidden_outcome_summary.csv", True,
        ),
    ]
    for outcome, offset in (("re74", 23000), ("re75", 24000)):
        tr = load_train(DATA / f"lalonde_placebo_{outcome}.csv")
        experiments.append((
            f"r5_placebo_{outcome}", tr, tr, True, E.SEED + offset,
            f"r5_placebo_{outcome}_summary.csv", False,
        ))
    for p in (11, 14):
        tr = load_train(DATA / f"lalonde_common_support_p{p}.csv")
        ext = load_external_target(DATA / f"lalonde_nsw_control_target_p{p}.csv")
        experiments.extend([
            (
                f"r6_hidden_p{p}", tr, ext, True, E.SEED + 22000,
                f"r6_hidden_p{p}_summary.csv", False,
            ),
            (
                f"r6_observational_p{p}", tr, tr, False,
                E.SEED + 21000, f"r6_observational_p{p}_summary.csv", False,
            ),
        ])
    meta_path = OUT / "extended_validation.json"
    if meta_path.exists():
        metadata.update(json.loads(meta_path.read_text()))
    for label, train, target, score, seed, summary_name, include_arb in experiments:
        print(f"Extended methods: {label}", flush=True)
        summary, _, meta = extend_saved_bootstrap(
            train, target, reps, seed, label, score, jobs, include_arb
        )
        save_table(summary_name, summary)
        metadata[label] = meta
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    r5 = []
    for outcome in ("re74", "re75"):
        d = pd.read_csv(OUT / f"r5_placebo_{outcome}_summary.csv")
        d.insert(0, "placebo", outcome)
        r5.append(d)
    save_table("r5_placebo_summary.csv", pd.concat(r5, ignore_index=True))

    r6h = [pd.read_csv(OUT / "r3_hidden_outcome_summary.csv").assign(
        representation="p171"
    )]
    r6o = [pd.read_csv(OUT / "r2_observational_summary.csv").assign(
        representation="p171"
    )]
    for p in (11, 14):
        h = pd.read_csv(OUT / f"r6_hidden_p{p}_summary.csv")
        if "representation" not in h:
            h.insert(0, "representation", f"p{p}")
        o = pd.read_csv(OUT / f"r6_observational_p{p}_summary.csv")
        if "representation" not in o:
            o.insert(0, "representation", f"p{p}")
        r6h.append(h)
        r6o.append(o)
    save_table("r6_hidden_outcome_summary.csv", pd.concat(r6h, ignore_index=True))
    save_table("r6_observational_summary.csv", pd.concat(r6o, ignore_index=True))
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main(reps: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    validation = {"requested_reps": reps, "target_outcome_leakage": False}
    main_train = load_train(DATA / "lalonde_common_support_p171.csv")
    point = full_point_table(main_train, E.SEED + 20000, "p171")
    save_table("r1_point_estimates.csv", point)

    r2s, r2d, r2m = run_bootstrap(
        main_train, main_train, reps, E.SEED + 21000, "r2_observational", False
    )
    save_table("r2_observational_summary.csv", r2s)
    r2d.to_csv(OUT / "r2_observational_draws.csv.gz", index=False, compression="gzip")
    validation["r2"] = r2m

    external171 = load_external_target(DATA / "lalonde_nsw_control_target_p171.csv")
    r3s, r3d, r3m = run_bootstrap(
        main_train, external171, reps, E.SEED + 22000,
        "r3_hidden_outcome", True,
    )
    save_table("r3_hidden_outcome_summary.csv", r3s)
    r3d.to_csv(OUT / "r3_hidden_outcome_draws.csv.gz", index=False, compression="gzip")
    validation["r3"] = r3m

    run_partition_artifacts()

    placebo_summaries = []
    for outcome in ("re74", "re75"):
        tr = load_train(DATA / f"lalonde_placebo_{outcome}.csv")
        summary, draws, meta = run_bootstrap(
            tr, tr, reps, E.SEED + (23000 if outcome == "re74" else 24000),
            f"r5_placebo_{outcome}", True,
        )
        summary.insert(0, "placebo", outcome)
        placebo_summaries.append(summary)
        save_table(f"r5_placebo_{outcome}_summary.csv", summary.drop(columns="placebo"))
        draws.to_csv(OUT / f"r5_placebo_{outcome}_draws.csv.gz",
                     index=False, compression="gzip")
        validation[f"r5_{outcome}"] = meta
        validation[f"r5_{outcome}_p"] = len(tr["features"])
        validation[f"r5_{outcome}_n0"] = len(tr["y0"])
        validation[f"r5_{outcome}_n1"] = len(tr["yt"])
    save_table("r5_placebo_summary.csv", pd.concat(placebo_summaries, ignore_index=True))

    run_feature_points()
    validation.update(run_feature_hidden(reps))
    validation.update(run_feature_observational(reps))

    r8 = cv_boundary_summary()
    if r8 is not None:
        save_table("r8_cv_boundary_summary.csv", r8)
    validation.update({
        "primary_methods": PRIMARY_METHODS,
        "r4_single_split_partitions": 50,
        "r4_estimates_averaged_across_splits": False,
        "max_weight_sum_error": float(max(
            r2d.weight_sum_error.max(), r3d.weight_sum_error.max()
        )),
    })
    (OUT / "validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    print("Complete validation suite written to", OUT, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument(
        "--feature-observational-only", action="store_true",
        help="Run only the p=11/14 observational feature-map bootstrap.",
    )
    parser.add_argument(
        "--partition-only", action="store_true",
        help="Run only Algorithm-2 single-split sensitivity diagnostics.",
    )
    parser.add_argument(
        "--feature-hidden-only", action="store_true",
        help="Run only the p=11/14 hidden-outcome feature-map bootstrap.",
    )
    parser.add_argument(
        "--feature-points-only", action="store_true",
        help="Run only feature-map point estimates with a common partition.",
    )
    parser.add_argument(
        "--extended-only", action="store_true",
        help="Add entropy and ARB to saved paired core experiments.",
    )
    parser.add_argument("--extended-reps", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.feature_points_only:
        run_feature_points()
    elif args.extended_only:
        run_extended_suite(args.extended_reps, args.jobs)
    elif args.feature_hidden_only:
        metadata = run_feature_hidden(args.reps)
        (OUT / "r6_hidden_validation.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    elif args.partition_only:
        run_partition_artifacts()
    elif args.feature_observational_only:
        metadata = run_feature_observational(args.reps)
        (OUT / "r6_observational_validation.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    else:
        main(args.reps)
