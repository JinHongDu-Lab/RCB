"""Reproduce both K562 analyses and record their input and run provenance."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import tempfile
import time

from rcb import paths
from rcb.io import save_json

STAGES = ("select", "prepare", "run_overlap", "run_full")
TABLES = (
    "perturbation_ranking.csv",
    "selected_perturbation_outcome_pairs.csv",
    "full_k562_outcome_panel.csv",
    "covariate_genes.csv",
    "att_overlap_tilt_repeated_raw.csv",
    "att_overlap_tilt_pair_summary.csv",
    "att_overlap_tilt_overall_summary.csv",
    "att_overlap_tilt_diagnostics.csv",
    "att_overlap_tilt_penalty_summary.csv",
    "full_k562_rcb_analysis.csv",
    "full_k562_gene_specificity.csv",
    "full_k562_perturbation_summary.csv",
    "full_k562_adjustment_path.csv",
    "full_k562_covariate_shift.csv",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state():
    def git(*args):
        result = subprocess.run(
            ["git", *args], cwd=paths.CODE, capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {"revision": git("rev-parse", "HEAD"), "status": git("status", "--short")}


def reproduce(config, *, smoke=False):
    # Import only when running, so help/list do not require the optional data stack.
    from .analysis import Analysis

    analysis = Analysis(config)
    output = analysis.output
    output.mkdir()
    record = {
        "status": "running",
        "mode": "smoke" if smoke else "production",
        "configuration": asdict(config),
        "git": git_state(),
        "python": platform.python_version(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "joblib",
                "anndata",
                "h5py",
                "crispyx",
            )
        },
        "inputs": [],
        "stage_seconds": {},
    }
    started = time.monotonic()
    save_json(record, output, "run_manifest.json")
    try:
        for path in analysis.inputs:
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing Replogle input: {path}. See data/replogle/README.md."
                )
            item = {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            record["inputs"].append(item)
        record["input_schema"] = analysis.validate_inputs()
        for name in STAGES:
            print(f"Replogle: {name}", flush=True)
            stage_start = time.monotonic()
            getattr(analysis, name)()
            record["stage_seconds"][name] = time.monotonic() - stage_start
            save_json(record, output, "run_manifest.json")
        expected_overlap = config.N_REPEATS * len(analysis.pair_table) * 2
        expected_full = len(analysis.selected_p) * len(analysis.full_panel_genes)
        if (
            len(analysis.att_table) != expected_overlap
            or len(analysis.full_rcb_table) != expected_full
        ):
            raise RuntimeError("Incomplete Replogle result rows")
        if (
            len(analysis.pair_table) != config.N_PERTURBATIONS * config.N_OUTCOMES_PER_P
            or len(analysis.full_panel_genes) != config.FULL_PANEL_SIZE
        ):
            raise RuntimeError(
                "Insufficient selected pairs/panel genes for requested configuration"
            )
        import numpy as np

        overlap_columns = [
            c
            for c in analysis.att_table
            if c.startswith(("effect_", "risk_", "ess_", "imbalance_"))
        ]
        full_columns = [
            "rcb_effect",
            "rcb_counterfactual_mean",
            "raw_effect",
            "rcb_estimated_risk",
            "rcb_effect_pi_low",
            "rcb_effect_pi_high",
            "rcb_ess",
            "rcb_imbalance",
        ]
        if (
            not np.isfinite(analysis.att_table[overlap_columns].to_numpy(float)).all()
            or not np.isfinite(
                analysis.full_rcb_table[full_columns].to_numpy(float)
            ).all()
        ):
            raise RuntimeError(
                "Non-finite estimates, risks, or diagnostics in Replogle output"
            )
        if (analysis.full_rcb_table["rcb_estimated_risk"] < 0).any():
            raise RuntimeError("Negative estimated prediction risk")
        record["outputs"] = [
            {
                "name": name,
                "bytes": (output / name).stat().st_size,
                "sha256": sha256(output / name),
            }
            for name in TABLES
        ]
        record["rows"] = {"overlap": expected_overlap, "full": expected_full}
        record["status"] = "complete"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        save_json(record, output, "run_manifest.json")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="Reduced run in an isolated output tree"
    )
    parser.add_argument(
        "--list", action="store_true", help="List stages without reading data"
    )
    parser.add_argument(
        "--jobs", type=int, default=int(os.getenv("CAUSAL_EXTRAPOLATION_JOBS", "4"))
    )
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(STAGES))
        return
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    smoke = args.smoke or os.getenv("CAUSAL_EXTRAPOLATION_SMOKE", "0") == "1"
    if smoke and not os.getenv("RCB_OUTPUT_ROOT"):
        os.environ["RCB_OUTPUT_ROOT"] = tempfile.mkdtemp(prefix="rcb-replogle-smoke-")
    # Match the original script's single-threaded numerical kernels before importing numpy.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    from threadpoolctl import threadpool_limits
    from .analysis import Config

    config = Config(N_JOBS=args.jobs)
    if smoke:
        config = replace(
            config,
            N_PERTURBATIONS=1,
            N_OUTCOMES_PER_P=1,
            FULL_PANEL_SIZE=2,
            N_NON_DE=32,
            N_REPEATS=2,
        )
    print(
        f"Replogle output: {paths.results('replogle', create=False).root}", flush=True
    )
    with threadpool_limits(limits=1):
        reproduce(config, smoke=smoke)


if __name__ == "__main__":
    main()
