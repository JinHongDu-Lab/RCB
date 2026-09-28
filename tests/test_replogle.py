"""Exercise Replogle with tiny H5AD inputs, without cloud data or published results."""

import json
from dataclasses import replace
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ad = pytest.importorskip("anndata")

from rcb import paths
from experiments.replogle.analysis import Analysis, Config, INPUT_NAMES
from experiments.replogle.run import TABLES, reproduce


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "inputs"
    raw = root / "data/replogle/raw"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(17)
    genes = pd.Index([f"gene_{j:02d}" for j in range(64)])
    labels = ["control"] * 32 + ["perturbation_a"] * 16
    pb = ad.AnnData(
        rng.normal(5, 0.3, (48, 64)),
        obs=pd.DataFrame(
            {"perturbation": labels}, index=[f"row_{i}" for i in range(48)]
        ),
        var=pd.DataFrame(index=genes),
    )
    de = ad.AnnData(
        np.zeros((1, 64)),
        obs=pd.DataFrame(index=["perturbation_a"]),
        var=pd.DataFrame(index=genes),
    )
    de.layers["z_score"] = np.linspace(8, 0, 64)[None, :]
    de.layers["pvalue_adj"] = np.linspace(0.001, 0.99, 64)[None, :]
    de.layers["logfoldchanges"] = np.linspace(1, 0, 64)[None, :]
    pb.write_h5ad(raw / INPUT_NAMES[0])
    de.write_h5ad(raw / INPUT_NAMES[1])
    return root


def small_config(jobs=1):
    return Config(
        PERTURBATIONS=("perturbation_a",),
        N_PERTURBATIONS=1,
        N_OUTCOMES_PER_P=1,
        FULL_PANEL_SIZE=2,
        N_NON_DE=8,
        N_REPEATS=2,
        N_JOBS=jobs,
    )


def test_import_has_no_io(tmp_path):
    env = dict(os.environ, RCB_OUTPUT_ROOT=str(tmp_path))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import experiments.replogle.analysis; import experiments.replogle.run",
        ],
        cwd=paths.CODE,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not result.stdout
    assert not list(tmp_path.iterdir())


def test_missing_input_fails_with_instructions(tmp_path):
    with paths.use_output_root(tmp_path):
        with pytest.raises(FileNotFoundError, match="data/replogle/README.md"):
            Analysis(small_config()).validate_inputs()


def test_invalid_schema_fails_before_computation(inputs):
    path = inputs / "data/replogle/raw" / INPUT_NAMES[1]
    de = ad.read_h5ad(path)
    del de.layers["pvalue_adj"]
    de.write_h5ad(path)
    with paths.use_output_root(inputs):
        with pytest.raises(ValueError, match="Missing DE layers"):
            Analysis(small_config()).validate_inputs()


def test_pipeline_complete_and_worker_independent(inputs, tmp_path, monkeypatch):
    from threadpoolctl import threadpool_limits

    # Real input tree stays fixed while each output run gets a separate scratch tree.
    monkeypatch.setattr(paths, "CODE", inputs)
    outputs = []
    with threadpool_limits(limits=1):
        for index, jobs in enumerate((1, 1, 2)):
            destination = tmp_path / f"run{index}"
            monkeypatch.setenv("RCB_OUTPUT_ROOT", str(destination))
            record = reproduce(small_config(jobs), smoke=True)
            assert record["status"] == "complete"
            assert record["rows"] == {"overlap": 4, "full": 2}
            outputs.append(
                {
                    name: (destination / "results/replogle" / name).read_bytes()
                    for name in TABLES
                }
            )
    assert outputs[0] == outputs[1] == outputs[2]


def test_failure_manifest_is_not_complete(inputs, monkeypatch):
    def fail(self):
        raise RuntimeError("injected estimator failure")

    monkeypatch.setattr(Analysis, "run_overlap", fail)
    with paths.use_output_root(inputs):
        with pytest.raises(RuntimeError, match="injected estimator failure"):
            reproduce(small_config(), smoke=True)
        record = json.loads(
            (paths.results("replogle") / "run_manifest.json").read_text()
        )
        assert record["status"] == "failed"
        assert "outputs" not in record


def test_design_cache_preserves_every_record(inputs, monkeypatch):
    from threadpoolctl import threadpool_limits

    with paths.use_output_root(inputs), threadpool_limits(limits=1):
        analysis = Analysis(replace(small_config(), N_OUTCOMES_PER_P=2))
        analysis.output.mkdir()
        analysis.select()
        analysis.prepare()
        original_prepare = analysis.prepare_design
        calls = []

        def counted(*args):
            calls.append(1)
            return original_prepare(*args)

        monkeypatch.setattr(analysis, "prepare_design", counted)
        cached = analysis.run_att_repeat(0)
        assert len(calls) == 2  # one per regime, shared by the two outcomes
        original_fit = analysis.fit_all_methods

        def uncached(X0, y0, X1, pilot, evaluation, prepared=None):
            return original_fit(X0, y0, X1, pilot, evaluation)

        monkeypatch.setattr(analysis, "fit_all_methods", uncached)
        reference = analysis.run_att_repeat(0)
        pd.testing.assert_frame_equal(
            pd.DataFrame(cached), pd.DataFrame(reference), check_exact=True
        )
