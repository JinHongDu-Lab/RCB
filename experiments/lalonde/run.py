"""The full LaLonde reproduction, as one command.

The R1--R9 sequence was previously a list of seven commands in the README, each
with its own replication counts, to be pasted in the right order.  That is a
protocol, not a script, and a protocol that lives in prose drifts from the code
it describes.

Run everything::

    python -m experiments.lalonde.run

or a fast pass that exercises every stage at reduced replication counts::

    python -m experiments.lalonde.run --smoke

The three R scripts that build the analysis matrices from the ECTA18515
replication data are not run here: they need R, they only need running when the
upstream data changes, and their outputs are committed.  ``--list`` prints them
alongside the Python stages so the full chain stays visible in one place.
"""

from __future__ import annotations

import argparse
import os
import time

from . import chi2_stress, fixed_oracle, tables, validation

#: Worker count for the two parallel stages, overridable so ``run_experiments.sh
#: --jobs N`` reaches Sections 6 and 7 through the same variable.  It cannot
#: move a number: ``run_bootstrap`` draws every bootstrap index up front from
#: one RNG and seeds replication ``b`` at ``seed + 100000 + b``, and
#: ``chi2_stress`` seeds at ``SPLIT_SEED + replication``, so a replication's
#: input does not depend on which worker picks it up or in what order.
JOBS = int(os.getenv("CAUSAL_EXTRAPOLATION_JOBS", "4"))

R_STAGES = (
    "Rscript build_lalonde171_analysis_data.R",
    "Rscript build_lalonde171_nsw_control_validation_data.R",
    "Rscript build_lalonde_placebo_data.R",
)

#: name -> (callable, production kwargs, smoke kwargs)
STAGES = {
    "core": (
        lambda **kw: validation.main(**kw),
        {"reps": 500},
        {"reps": 8},
    ),
    "extended": (
        lambda **kw: validation.run_extended_suite(**kw),
        {"reps": 200, "jobs": JOBS},
        {"reps": 4, "jobs": 1},
    ),
    "oracle": (
        lambda **kw: fixed_oracle.main(**kw),
        {"calibration_reps": 1000, "evaluation_reps": 200},
        {"calibration_reps": 8, "evaluation_reps": 4},
    ),
    "overlap": (
        lambda **kw: chi2_stress.run(**kw),
        {"reps": 120, "jobs": JOBS},
        {"reps": 4, "jobs": 1},
    ),
    "tables": (
        lambda **kw: tables.main(**kw),
        {},
        {},
    ),
}


def main(stages: list[str], smoke: bool) -> None:
    for name in stages:
        run, production, reduced = STAGES[name]
        options = reduced if smoke else production
        print(f"\n=== {name} {options or ''} ===", flush=True)
        started = time.time()
        run(**options)
        print(f"=== {name} finished in {time.time() - started:.1f}s ===",
              flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages", nargs="+", choices=sorted(STAGES), default=list(STAGES),
        help="subset of stages to run, in the given order",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="reduced replication counts; exercises every stage, proves nothing",
    )
    parser.add_argument(
        "--list", action="store_true", help="print the full chain and exit",
    )
    args = parser.parse_args()
    if args.list:
        print("Data construction (R, run only when the upstream data changes):")
        for stage in R_STAGES:
            print(f"  {stage}")
        print("Analysis (this script):")
        for name in STAGES:
            print(f"  {name}")
    else:
        main(args.stages, args.smoke)
