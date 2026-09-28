#!/usr/bin/env python3
"""Create the concise paper-facing tables and the stability audits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import experiment as E
from rcb import paths
from .validation import CORE_METHODS, OUT


FIG = paths.figures("lalonde", create=False)
DOUBLE = "#D95F4C"
DOUBLE_2 = "#8C6BB1"
DOUBLE_3 = "#D49A20"
OURS = "#2B8C82"
OURS_2 = "#315F8D"
NEUTRAL = "#6B7280"


METHOD_LABELS = {
    "Pure weighting: uniform": "Weighting: uniform",
    "Pure weighting: entropy": "Weighting: entropy",
    "Pure weighting: l2": r"Weighting: $\ell_2$",
    "Double ridge: outcome CV": "Double ridge: outcome CV",
    "Double ridge: imbalance CV": "Double ridge: imbalance CV",
    "Double ridge: Riesz CV": "Double ridge: Riesz CV",
    "Ours: uniform base, full target": "Ours: uniform base",
    "Ours: entropy base, pilot/evaluation": "Ours: entropy base",
    "Ours: l2 base, pilot/evaluation": r"Ours: $\ell_2$ base",
    "ARB": "ARB",
    "Double ridge: CV outcome": "Double ridge: outcome CV",
    "Double ridge: CV imbalance": "Double ridge: imbalance CV",
    "Double ridge: CV Riesz": "Double ridge: Riesz CV",
    "Ours: tuned l2 base": r"Ours: tuned $\ell_2$ base",
    "Ours: uniform base": "Ours: uniform base",
}


def half_stability(draws: pd.DataFrame, score: bool) -> pd.DataFrame:
    cutoff = int(draws.replication.max() + 1) // 2
    d = draws.copy()
    d["half"] = np.where(d.replication < cutoff, "1--250", "251--500")
    rows = []
    for (half, method), g in d.groupby(["half", "method"], sort=False):
        row = {
            "half": half,
            "method": method,
            "reps": len(g),
            "no_adjustment_rate": g.selected_no_adjustment.mean(),
            "lower_boundary_rate": g.selected_lower_boundary.mean(),
            "upper_boundary_rate": g.selected_upper_boundary.mean(),
        }
        if score:
            sq = g.squared_error.to_numpy()
            row.update({
                "bias": g.error.mean(),
                "rmse": np.sqrt(sq.mean()),
                "paired_mse_difference": g.paired_mse_difference.mean(),
            })
        else:
            row.update({
                "att_mean": g.att.mean(),
                "att_sd": g.att.std(ddof=1),
                "att_q025": g.att.quantile(.025),
                "att_q975": g.att.quantile(.975),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def write_paper_tables() -> None:
    r1 = pd.read_csv(OUT / "r1_point_estimates.csv")
    r2 = pd.read_csv(OUT / "r2_observational_summary.csv")
    r3 = pd.read_csv(OUT / "r3_hidden_outcome_summary.csv")
    r4 = pd.read_csv(OUT / "r4_single_split_stability_summary.csv")
    r5 = pd.read_csv(OUT / "r5_placebo_summary.csv")
    r6h = pd.read_csv(OUT / "r6_hidden_outcome_summary.csv")
    r6o = pd.read_csv(OUT / "r6_observational_summary.csv")
    # R8's input has no writer in this repository; see
    # validation.CV_FOLD_DRAWS_HAVE_NO_WRITER.  Absent on a clean tree.
    r8_path = OUT / "r8_cv_boundary_summary.csv"
    r8 = pd.read_csv(r8_path) if r8_path.exists() else None

    r1[[
        "method", "muhat0", "att", "selected_lambda",
        "estimated_target_risk", "weight_norm_squared", "ess",
        "residual_imbalance_squared", "max_abs_smd", "mean_abs_smd",
        "negative_weight_fraction", "diagnostic_weight_role",
        "estimate_equals_weighted_source_outcome",
    ]].to_csv(OUT / "paper_table_r1.csv", index=False)
    r2[[
        "method", "reps", "att_mean", "att_sd", "att_q025", "att_q975",
        "median_finite_lambda", "lambda_iqr", "no_adjustment_rate",
        "lower_boundary_rate", "upper_boundary_rate",
    ]].to_csv(OUT / "paper_table_r2.csv", index=False)
    r3[[
        "method", "reps", "bias", "rmse", "rmse_mcse", "mae",
        "paired_mse_difference", "paired_mse_difference_mcse",
        "no_adjustment_rate", "lower_boundary_rate", "failure_rate",
    ]].to_csv(OUT / "paper_table_r3.csv", index=False)
    r4.to_csv(OUT / "paper_table_r4_single_split_stability.csv", index=False)
    r5[[
        "placebo", "method", "reps", "bias", "rmse", "rmse_mcse",
        "paired_mse_difference", "paired_mse_difference_mcse",
    ]].to_csv(OUT / "paper_table_r5.csv", index=False)
    r6h[[
        "representation", "method", "reps", "bias", "rmse", "rmse_mcse",
        "paired_mse_difference", "paired_mse_difference_mcse",
    ]].to_csv(OUT / "paper_table_r6_hidden.csv", index=False)
    r6o[[
        "representation", "method", "reps", "att_mean", "att_sd",
        "att_q025", "att_q975",
    ]].to_csv(OUT / "paper_table_r6_observational.csv", index=False)
    if r8 is not None:
        r8.to_csv(OUT / "paper_table_r8.csv", index=False)
    r9_summary = OUT / "r9_chi2_overlap_summary.csv"
    if r9_summary.exists():
        pd.read_csv(r9_summary).to_csv(
            OUT / "paper_table_r9_summary.csv", index=False
        )
        pd.read_csv(OUT / "r9_chi2_overlap_comparisons.csv").to_csv(
            OUT / "paper_table_r9_comparisons.csv", index=False
        )
        pd.read_csv(OUT / "r9_chi2_overlap_design.csv").to_csv(
            OUT / "paper_table_r9_design.csv", index=False
        )


def validation_audit() -> dict:
    base = json.loads((OUT / "validation.json").read_text())
    extended_path = OUT / "extended_validation.json"
    extended = json.loads(extended_path.read_text()) if extended_path.exists() else {}
    r2d = pd.read_csv(OUT / "r2_observational_draws.csv.gz")
    r3d = pd.read_csv(OUT / "r3_hidden_outcome_draws.csv.gz")
    r2_half = half_stability(r2d, score=False)
    r3_half = half_stability(r3d, score=True)
    r2_half.to_csv(OUT / "r2_half_stability.csv", index=False)
    r3_half.to_csv(OUT / "r3_half_stability.csv", index=False)
    with np.load(OUT / "r2_observational_bootstrap_indices.npz") as z:
        r2_source, r2_target = z["source"], z["target"]
    with np.load(OUT / "r3_hidden_outcome_bootstrap_indices.npz") as z:
        r3_source, r3_target = z["source"], z["target"]
    paired_feature_indices = True
    for p in (11, 14):
        with np.load(OUT / f"r6_observational_p{p}_bootstrap_indices.npz") as z:
            paired_feature_indices &= np.array_equal(z["source"], r2_source)
            paired_feature_indices &= np.array_equal(z["target"], r2_target)
        with np.load(OUT / f"r6_hidden_p{p}_bootstrap_indices.npz") as z:
            paired_feature_indices &= np.array_equal(z["source"], r3_source)
            paired_feature_indices &= np.array_equal(z["target"], r3_target)
    audit = {
        "r2_records_per_method": r2d.groupby("method").size().to_dict(),
        "r3_records_per_method": r3d.groupby("method").size().to_dict(),
        "r2_core_methods_have_500": bool(
            (r2d.loc[r2d.method.isin(CORE_METHODS)]
             .groupby("method").size() == 500).all()
        ),
        "r3_core_methods_have_500": bool(
            (r3d.loc[r3d.method.isin(CORE_METHODS)]
             .groupby("method").size() == 500).all()
        ),
        "max_affine_weight_sum_error": float(max(
            r2d.weight_sum_error.max(), r3d.weight_sum_error.max()
        )),
        "affine_error_below_1e-8": bool(max(
            r2d.weight_sum_error.max(), r3d.weight_sum_error.max()
        ) < 1e-8),
        "target_outcomes_passed_to_fit": bool(
            base["r2"]["target_outcomes_passed_to_fit"]
            or base["r3"]["target_outcomes_passed_to_fit"]
        ),
        "numerical_failures": len(base["r2"]["failures"])
        + len(base["r3"]["failures"])
        + sum(len(meta["failures"]) for meta in extended.values()),
        "maximum_pilot_evaluation_overlap": int(max(
            r2d.pilot_evaluation_overlap.fillna(0).max(),
            r3d.pilot_evaluation_overlap.fillna(0).max(),
        )),
        "primary_method_set_matches_validation_specification": bool(
            set(r2d.method.unique()) == set(base["primary_methods"])
            and set(r3d.method.unique()) == set(base["primary_methods"])
        ),
        "r6_indices_identical_across_representations": bool(
            paired_feature_indices
        ),
        "estimates_averaged_across_splits": bool(
            base.get("r4_estimates_averaged_across_splits", True)
        ),
        "extended_method_requested_reps": (
            extended.get("r3_hidden_outcome", {}).get("requested_reps")
        ),
        "extended_methods_use_saved_paired_indices": bool(
            extended and all(
                meta["paired_indices_reused_from_core_experiment"]
                for meta in extended.values()
            )
        ),
    }
    r7_path = OUT / "r7_validation.json"
    if r7_path.exists():
        audit["r7"] = json.loads(r7_path.read_text())
    r9_path = OUT / "r9_chi2_overlap_validation.json"
    if r9_path.exists():
        audit["r9"] = json.loads(r9_path.read_text())
    (OUT / "paper_validation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return audit


def write_design_facts() -> None:
    """Record the sample sizes the notebook reports.

    Reading them off the raw data is the experiment's job; the notebook only
    displays what was recorded, so it never touches ``data/``.
    """
    d = E.load_lalonde171()
    external = pd.read_csv(
        paths.data("lalonde") / "lalonde_nsw_control_target_p171.csv"
    )
    pd.DataFrame({
        "Sample": ["PSID source controls", "NSW treated target",
                   "NSW-control pseudo-target"],
        "n": [len(d["y0"]), len(d["yt"]), len(external)],
        "Outcome use": ["fit source outcome model", "ATT after fitting",
                        "score after fitting"],
    }).to_csv(OUT / "design_samples.csv", index=False)
    pd.DataFrame({
        "Representation": ["short", "intermediate", "LaLonde-171"],
        "p": [11, 14, 171],
        "Fixed common-support sample size": [912, 912, 912],
    }).to_csv(OUT / "design_representations.csv", index=False)


def main() -> None:
    """Write the paper tables and the consolidated audit.

    The figures that used to be drawn here are in ``3_lalonde.ipynb``: this module
    computes, the notebook plots.
    """
    write_design_facts()
    write_paper_tables()
    validation_audit()
    print("Paper-facing artifacts written to", OUT)


if __name__ == "__main__":
    main()
