# LaLonde application

The formal analysis uses 185 NSW treated
observations, 727 common-support PSID controls, and 171 pre-specified
covariates.

## Directory map

```text
experiments/lalonde/
├── run.py                 the whole sequence, one command
├── methods.py             data construction and the study's numerical settings
├── experiment.py          exploratory pipeline and cached design geometry
├── validation.py          the audited validation driver
├── overlap.py             classifier-induced chi-square design
├── chi2_stress.py         the chi-square overlap stress test
├── fixed_oracle.py        independent fixed-oracle calibration
├── tables.py              paper tables and the consolidated audit
└── build_lalonde*.R       build the analysis matrices from the replication data

data/lalonde/             upstream inputs and derived matrices
results/lalonde/          summary tables and validation records
figures/lalonde_*         figures
3_lalonde.ipynb           reads results and plots
```

## Formal method set

The notebook compares ten estimators: three double-ridge CV variants; our
uniform-, regularized-entropy-, and l2-base estimators; pure uniform, entropy,
and l2 weighting; and ARB. Adaptive bases use one disjoint pilot/evaluation
target split, with no swapped-fold averaging.

## Reproduction

Build the analysis matrices (only when the upstream data changes), then run the
analysis:

```bash
Rscript experiments/lalonde/build_lalonde171_analysis_data.R
Rscript experiments/lalonde/build_lalonde171_nsw_control_validation_data.R
Rscript experiments/lalonde/build_lalonde_placebo_data.R

python -m experiments.lalonde.run              # the full sequence
python -m experiments.lalonde.run --smoke      # reduced, to check the chain
python -m experiments.lalonde.run --list       # print every stage

jupyter nbconvert --to notebook --execute --inplace 3_lalonde.ipynb
```

The consolidated audit is `results/lalonde/paper_validation_audit.json`.
