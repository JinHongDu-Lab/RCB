# risk-calibrated-causal-extrapolation

Code, results and figures for *Risk-Calibrated Balancing for High-Dimensional Causal Extrapolation*.

`experiments/` runs the studies and writes tables to `results/`; the notebooks read `results/`, draw the figures, and write them to `figures/`.

```
.
  rcb/                    the shared library: the estimator, the risk formulas,
                          the compared methods, the shared utilities
  experiments/
    simulation/           design-independent risk study
    evaluation/           adaptive-balancing comparison
    lalonde/              LaLonde application, plus the R data builders
    replogle/             K562 Perturb-seq application and its input preprocessing
  1_simulation.ipynb      reads results/, draws figures/
  2_evaluation.ipynb
  3_lalonde.ipynb
  4_replogle.ipynb
  data/<study>/           study inputs
  results/<study>/        tables written by the studies (not in the repository)
  figures/                the paper's figures, in PDF
  tests/
```

| study | what it is |
|---|---|
| `simulation` | conditional random-effects experiments; risk theory |
| `evaluation` | eight competing weight families, augmented and compared |
| `lalonde` | the LaLonde real-data application |
| `replogle` | K562 Perturb-seq overlap experiment and full-panel RCB |

## Running the estimator on your own base weights

The proposed estimator has one entry point, `rcb.estimator.fit_rcb`: give it the source design and outcomes, the *evaluation* fold of the target sample, any normalized base weight vector built without the outcomes, and the penalty grid, and it returns the selected weights, estimate and penalty together with the whole path.
For example, RCB on top of the ABW balancing component on one replication of the evaluation study's design:

```python
from experiments.evaluation import dgp            # the study's design and settings
from rcb.benchmarks.bases import base_weight_families
from rcb.estimator import fit_rcb

data = dgp.draw_replication(0, 1.25, 1.0)          # X0, Y0, pilot fold X1P, evaluation fold X1E
weights, records = base_weight_families(("l2",), data["X0"], data["X1P"], dgp.BASE_SETTINGS)
fit = fit_rcb(data["X0"], data["Y0"], data["X1E"], weights["l2"], dgp.LAMBDAS,
              theta_bounds=dgp.SQ_THETA_BOUNDS)
fit.muhat0, fit.selected_lambda, fit.weights        # the estimate, its penalty, the weights
```

Replace `"l2"` with `"entropy"`, `"sbw"`, `"cbps"`, `"arb"` or any other name in `rcb.benchmarks.bases.FAMILIES`, or pass your own normalized weight vector.
`rcb/README.md` describes the package and where each object in the paper is implemented.

## Setting up

```
conda env create -f env.yml
conda activate rcb
```

`env.yml` pins minor versions; `requirements.txt` pins the exact versions the paper's results were produced with (Python 3.10.0, numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2).

## Running the studies

```
./run_experiments.sh                  # every study, production settings
./run_experiments.sh simulation       # one study: simulation | evaluation | lalonde | replogle
./run_experiments.sh --smoke          # every study at reduced size, to check the pipeline runs
./run_experiments.sh --notebooks      # execute the four notebooks and redraw the figures
./run_experiments.sh --tests          # the test suite
./run_experiments.sh --jobs 8         # cap parallelism
```

Each study can also be run directly, for example `python -m experiments.lalonde.run`; `--smoke` runs it at reduced size in a temporary output directory and `--list` prints its stages.

`results/` is not part of the repository: run a study before executing its notebook.
The paper's figures are committed in `figures/`.

## Data

- LaLonde: the NSW, PSID and CPS files and the derived analysis matrices are in `data/lalonde/`; see [data/lalonde/README.md](data/lalonde/README.md).
- Replogle: the two K562 inputs are built from the public scPerturb release of Replogle et al. (2022); [data/replogle/README.md](data/replogle/README.md) gives the download link and the preprocessing command.

## Run times

Production run times on an 18-core workstation; expect considerably longer on a machine with fewer cores.

| study | what | time |
|---|---|---|
| `simulation` | deterministic equivalents, aspect ratio, risk-tuning calibration and diagnostics, sequential | 12 min |
| `evaluation` | eight base families, ABW/ARB comparators, trimming sensitivity and supplements | 7 min |
| `lalonde` | bootstrap, extended suite, fixed-oracle calibration, chi-square overlap and tables | 20 min |
| `replogle` | 100 repetitions of the overlap comparison and the full-panel analysis, 16 workers | 58 min |
| notebooks | the four notebooks | 1 min |
| **total** | | **about 1.5 h** |

Building the Replogle inputs adds a 1.55 GB download and a few minutes of preprocessing.

## Tests

```
python -m pytest tests -q
```

`tests/test_equivalence.py` checks the package against recorded reference outputs, and `tests/test_layout.py` checks the repository layout.
The equivalence test expects Apple Accelerate BLAS on osx-arm64; see the note at the end of `env.yml`.

## Citation

If you use this code, please cite the accompanying paper, *Risk-Calibrated Balancing for High-Dimensional Causal Extrapolation*.

## License

MIT; see [LICENSE](LICENSE).
