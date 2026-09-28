#!/usr/bin/env python3
"""R9: classifier-chi-square overlap stress test for the LaLonde notebook."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import validation as V
from . import experiment as E
from . import overlap as D


OUT = V.OUT
SEED = 20260818
SPLIT_SEED = 20260819
DEFAULT_REPS = 120
#: Moderate and strong controlled shifts: 4x and 16x the baseline
#: classifier-induced chi-square.  "Weak" is avoided to prevent confusion with
#: weak overlap.  Seeds depend on the regime index, not the label.
REGIME_ORDER = ["Moderate shift", "Strong shift"]
CHI2_MULTIPLES = {"Moderate shift": 4.0, "Strong shift": 16.0}


def calibrate_shift_regimes(data: dict, density: dict) -> pd.DataFrame:
    """Calibrate 4x and 16x classifier-induced chi-square, using X only."""
    original = D.design_row(0.0, density)
    original_chi2 = original["classifier_chi2"]
    ess_floor = 0.20 * len(data["X0"])
    grid = np.linspace(-3.0, 0.0, 3001)
    curve = pd.DataFrame([D.design_row(alpha, density) for alpha in grid])
    admissible = curve[curve.source_sampling_ess >= ess_floor].copy()
    rows = []
    for regime in REGIME_ORDER:
        requested = CHI2_MULTIPLES[regime]
        target = requested * original_chi2
        index = (admissible.classifier_chi2 - target).abs().idxmin()
        selected = admissible.loc[index]
        rows.append(
            {
                "regime": regime,
                **selected.to_dict(),
                "original_classifier_chi2": original_chi2,
                "requested_chi2_multiple": requested,
                "achieved_chi2_multiple": (
                    selected.classifier_chi2 / original_chi2
                ),
                "source_sampling_ess_fraction": (
                    selected.source_sampling_ess / len(data["X0"])
                ),
            }
        )
    regimes = pd.DataFrame(rows)
    if not (
        regimes.iloc[0].classifier_chi2 < regimes.iloc[1].classifier_chi2
    ):
        raise AssertionError("Moderate/strong classifier-chi-square ordering failed")
    return regimes


def _one_replication(
    data: dict,
    density: dict,
    signal: dict,
    regime: dict,
    replication: int,
) -> dict:
    """Fit the complete ten-method set on one paired plasmode draw."""
    try:
        probability = D.source_distribution(float(regime["alpha"]), density)
        rng = np.random.default_rng(SEED + replication)
        source_index = rng.choice(
            len(data["X0"]),
            size=len(data["X0"]),
            replace=True,
            p=probability,
        )
        X0 = data["X0"][source_index]
        y0 = signal["m0_source"][source_index] + rng.normal(
            0.0, signal["residual_sigma"], len(source_index)
        )
        Xt = data["Xt"]
        split_seed = SPLIT_SEED + replication
        fitted = V.fit_primary(X0, y0, Xt, seed=split_seed)
        fitted.extend(V.fit_extended(
            X0, y0, Xt, seed=split_seed, include_arb=True
        ))
        rows = []
        for result in fitted:
            error = result["muhat0"] - signal["truth_mu0"]
            rows.append(
                {
                    "regime": regime["regime"],
                    "alpha": float(regime["alpha"]),
                    "classifier_chi2": float(regime["classifier_chi2"]),
                    "achieved_chi2_multiple": float(
                        regime["achieved_chi2_multiple"]
                    ),
                    "source_sampling_ess": float(
                        regime["source_sampling_ess"]
                    ),
                    "source_unique_fraction": float(
                        len(np.unique(source_index)) / len(source_index)
                    ),
                    "replication": int(replication),
                    "truth_mean": float(signal["truth_mu0"]),
                    "target_split_seed": int(split_seed),
                    **result,
                    "error": float(error),
                    "squared_error": float(error**2),
                }
            )
        return {"rows": rows, "error": None}
    except Exception as exc:
        return {
            "rows": [],
            "error": {
                "regime": regime["regime"],
                "replication": int(replication),
                "error": repr(exc),
            },
        }


def _rmse_ratio_interval(
    squared_error: np.ndarray,
    reference_squared_error: np.ndarray,
    seed: int,
    draws: int = 5000,
) -> tuple[float, float, float]:
    ratio = float(np.sqrt(
        squared_error.mean() / reference_squared_error.mean()
    ))
    rng = np.random.default_rng(seed)
    index = rng.integers(
        0, len(squared_error), size=(draws, len(squared_error))
    )
    bootstrap = np.sqrt(
        squared_error[index].mean(axis=1)
        / np.maximum(reference_squared_error[index].mean(axis=1), 1e-15)
    )
    return (
        ratio,
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    )


def summarize(draws: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """RMSE summaries and paired comparisons against imbalance-CV double ridge."""
    reference_name = "Double ridge: imbalance CV"
    summary_rows = []
    comparison_rows = []
    for regime_index, regime in enumerate(REGIME_ORDER):
        panel = draws[draws.regime == regime]
        reference = (
            panel[panel.method == reference_name]
            .sort_values("replication")
            .squared_error.to_numpy()
        )
        for method_index, (method, g) in enumerate(
            panel.groupby("method", sort=False)
        ):
            g = g.sort_values("replication")
            squared_error = g.squared_error.to_numpy()
            rmse = float(np.sqrt(squared_error.mean()))
            difference = squared_error - reference
            ratio, lower, upper = _rmse_ratio_interval(
                squared_error,
                reference,
                SEED + 1000 * regime_index + method_index,
            )
            summary_rows.append(
                {
                    "regime": regime,
                    "method": method,
                    "reps": len(g),
                    "bias": float(g.error.mean()),
                    "sd": float(g.muhat0.std(ddof=1)),
                    "rmse": rmse,
                    "rmse_mcse": float(
                        squared_error.std(ddof=1)
                        / np.sqrt(len(g)) / (2.0 * rmse)
                    ),
                    "mae": float(g.error.abs().mean()),
                    "no_adjustment_rate": float(
                        g.selected_no_adjustment.mean()
                    ),
                    "median_finite_lambda": float(
                        g.loc[
                            np.isfinite(g.selected_lambda), "selected_lambda"
                        ].median()
                    ),
                    "mean_ess": float(g.ess.mean()),
                    "mean_imbalance_squared": float(
                        g.residual_imbalance_squared.mean()
                    ),
                }
            )
            comparison_rows.append(
                {
                    "regime": regime,
                    "method": method,
                    "reference_method": reference_name,
                    "rmse_ratio": ratio,
                    "rmse_ratio_lower_95": lower,
                    "rmse_ratio_upper_95": upper,
                    "paired_mse_difference": float(difference.mean()),
                    "paired_mse_difference_mcse": float(
                        difference.std(ddof=1) / np.sqrt(len(difference))
                    ),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(comparison_rows)


def run(reps: int = DEFAULT_REPS, jobs: int = 4) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    data = E.load_lalonde171()
    density = D.fit_cross_fitted_density_ratio(data)
    regimes = calibrate_shift_regimes(data, density)
    signal = D.fit_plasmode_signal(data)
    tasks = [
        (regime._asdict(), replication)
        for regime in regimes.itertuples(index=False)
        for replication in range(reps)
    ]
    start = time.time()
    results = Parallel(n_jobs=jobs, prefer="processes", verbose=10)(
        delayed(_one_replication)(
            data, density, signal, regime, replication
        )
        for regime, replication in tasks
    )
    failures = [result["error"] for result in results if result["error"]]
    rows = [row for result in results for row in result["rows"]]
    if failures:
        (OUT / "r9_chi2_overlap_failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
        raise RuntimeError(
            f"R9 had {len(failures)} complete-replication failures; "
            f"first failures: {failures[:3]}"
        )
    draws = pd.DataFrame(rows)
    if set(draws.method.unique()) != set(V.PRIMARY_METHODS):
        raise AssertionError("R9 method set differs from the formal ten methods")
    expected_rows = reps * len(REGIME_ORDER) * len(V.PRIMARY_METHODS)
    if len(draws) != expected_rows:
        raise AssertionError(
            f"R9 expected {expected_rows} rows but found {len(draws)}"
        )
    summary, comparisons = summarize(draws)
    design = (
        draws.drop_duplicates(["regime", "replication"])
        .groupby("regime", sort=False)
        .agg(
            replications=("replication", "size"),
            classifier_chi2=("classifier_chi2", "first"),
            achieved_chi2_multiple=("achieved_chi2_multiple", "first"),
            source_sampling_ess=("source_sampling_ess", "first"),
            median_source_unique_fraction=("source_unique_fraction", "median"),
        )
        .reset_index()
    )
    validation = {
        "reps_per_regime": int(reps),
        "regimes": REGIME_ORDER,
        "method_count": int(draws.method.nunique()),
        "method_set_matches_primary_ten": bool(
            set(draws.method.unique()) == set(V.PRIMARY_METHODS)
        ),
        "records_per_regime_method": (
            draws.groupby(["regime", "method"]).size().to_dict()
        ),
        "overlap_calibration_uses_outcomes": False,
        "chi2_is_classifier_induced_not_population_truth": True,
        "target_sample_fixed_across_regimes": True,
        "truth_fixed_across_regimes": bool(draws.truth_mean.nunique() == 1),
        "one_single_split_per_adaptive_estimate": True,
        "estimates_averaged_across_splits": False,
        "split_seed_schedule": (
            f"{SPLIT_SEED} + replication; paired across regimes"
        ),
        "minimum_source_sampling_ess": float(
            regimes.source_sampling_ess.min()
        ),
        "maximum_pilot_evaluation_overlap": int(
            draws.pilot_evaluation_overlap.fillna(0).max()
        ),
        "all_outputs_finite": bool(
            np.isfinite(
                draws[["muhat0", "error", "squared_error"]]
            ).all().all()
        ),
        "numerical_failures": len(failures),
        "density_ratio_propensity_clip": density["propensity_clip"],
        "density_ratio_log_clip": density["log_ratio_clip"],
        "fixed_truth_mu0": float(signal["truth_mu0"]),
        "elapsed_seconds": time.time() - start,
        "jobs": int(jobs),
    }
    # JSON cannot encode tuple keys.
    validation["records_per_regime_method"] = {
        f"{regime} | {method}": int(count)
        for (regime, method), count in (
            draws.groupby(["regime", "method"]).size().items()
        )
    }
    checks = [
        "method_set_matches_primary_ten",
        "overlap_calibration_uses_outcomes",
        "target_sample_fixed_across_regimes",
        "truth_fixed_across_regimes",
        "one_single_split_per_adaptive_estimate",
        "all_outputs_finite",
    ]
    if not (
        validation["method_set_matches_primary_ten"]
        and not validation["overlap_calibration_uses_outcomes"]
        and validation["target_sample_fixed_across_regimes"]
        and validation["truth_fixed_across_regimes"]
        and validation["one_single_split_per_adaptive_estimate"]
        and validation["all_outputs_finite"]
    ):
        raise AssertionError(f"R9 validation failed: {checks}")

    regimes.to_csv(OUT / "r9_chi2_overlap_regimes.csv", index=False)
    design.to_csv(OUT / "r9_chi2_overlap_design.csv", index=False)
    draws.to_csv(
        OUT / "r9_chi2_overlap_draws.csv.gz",
        index=False,
        compression="gzip",
    )
    summary.to_csv(OUT / "r9_chi2_overlap_summary.csv", index=False)
    comparisons.to_csv(OUT / "r9_chi2_overlap_comparisons.csv", index=False)
    (OUT / "r9_chi2_overlap_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    return {
        "regimes": regimes,
        "design": design,
        "summary": summary,
        "comparisons": comparisons,
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    result = run(args.reps, args.jobs)
    print(result["regimes"].to_string(index=False))
    print(result["summary"].to_string(index=False))
    print(result["comparisons"].to_string(index=False))
    print(json.dumps(result["validation"], indent=2))


if __name__ == "__main__":
    main()
