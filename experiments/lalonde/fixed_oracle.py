#!/usr/bin/env python3
"""Independent fixed-path oracle calibration for the LaLonde stress test."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import experiment as E
from .validation import OUT


TUNED_FAMILY = "tuned-l2 / double-ridge shared path"
UNIFORM_FAMILY = "uniform path"


def calibrated_penalties(paths: pd.DataFrame) -> dict:
    selected = paths.loc[
        paths.groupby(
            ["surface", "noise_multiplier", "path_family"], sort=False
        )["empirical_path_mse"].idxmin()
    ]
    return {
        (row.surface, row.noise_multiplier, row.path_family): row.penalty
        for row in selected.itertuples()
    }


def main(calibration_reps: int, evaluation_reps: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = E.load_lalonde171()
    calibration_seed = E.SEED + 11701
    evaluation_seed = E.SEED + 21701

    calibration_summary, calibration_paths, _ = E.run_robustness_checks(
        d,
        reps_per_cell=calibration_reps,
        simulation_seed=calibration_seed,
    )
    fixed = calibrated_penalties(calibration_paths)
    fixed_rows = [
        {
            "surface": surface,
            "noise_multiplier": noise,
            "path_family": family,
            "best_fixed_path_penalty": penalty,
        }
        for (surface, noise, family), penalty in fixed.items()
    ]

    summary, paths, draws = E.run_robustness_checks(
        d,
        reps_per_cell=evaluation_reps,
        simulation_seed=evaluation_seed,
        fixed_penalties=fixed,
    )
    calibration_summary.to_csv(
        OUT / "r7_oracle_calibration_method_summary.csv", index=False
    )
    calibration_paths.to_csv(
        OUT / "r7_oracle_calibration_paths.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(fixed_rows).to_csv(
        OUT / "r7_independent_fixed_penalties.csv", index=False
    )
    summary.to_csv(OUT / "r7_stress_test_summary.csv", index=False)
    paths.to_csv(
        OUT / "r7_stress_test_paths.csv.gz", index=False, compression="gzip"
    )
    draws.to_csv(
        OUT / "r7_stress_test_draws.csv.gz", index=False, compression="gzip"
    )
    validation = {
        "oracle_calibration_reps_per_cell": calibration_reps,
        "evaluation_reps_per_cell": evaluation_reps,
        "oracle_calibration_seed": calibration_seed,
        "evaluation_seed": evaluation_seed,
        "independent_batches": calibration_seed != evaluation_seed,
        "path_families": [TUNED_FAMILY, UNIFORM_FAMILY],
        "fixed_penalties_locked_before_evaluation": True,
        "draw_wise_oracle_retained_only_as_lower_bound": True,
        "cells": 9,
        "methods": int(summary.method.nunique()),
        "finite_summary": bool(np.isfinite(summary.rmse).all()),
    }
    (OUT / "r7_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    print("Independent fixed-path oracle validation written to", OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-reps", type=int, default=1000)
    parser.add_argument("--evaluation-reps", type=int, default=200)
    args = parser.parse_args()
    main(args.calibration_reps, args.evaluation_reps)
