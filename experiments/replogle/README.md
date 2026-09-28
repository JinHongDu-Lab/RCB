# Replogle K562 application

Run from the repository root after building the inputs described in `data/replogle/README.md`.

```sh
python -m experiments.replogle.run --list      # print the stages
python -m experiments.replogle.run --jobs 4    # the full analysis
jupyter nbconvert --to notebook --execute --inplace 4_replogle.ipynb
```

`analysis.py` implements the study.
Its stages select perturbation/outcome pairs and a fixed full-data panel, prepare covariates, run repeated overlap comparisons, and run full-panel uniform-base RCB.
Tables and a run manifest are written to `results/replogle/`.
The notebook combines good-overlap RMSE, weak-overlap RMSE, RCB effects, and Wilcoxon effects into `figures/replogle_01_k562_overview.pdf`.

The production configuration uses 5 perturbations (SLC39A9, GAB2, ORC1, SF3B2 and NCBP2), 25 outcomes per perturbation, a 25-gene full panel, up to 500 non-DE covariates, 100 repetitions, seed 20260829, and a 121-point penalty grid plus the infinite endpoint.
The worker count defaults to 4 and can also be set with `CAUSAL_EXTRAPOLATION_JOBS`.
Each run records its configuration, input hashes, code revision, package versions, stage timings and output hashes in the run manifest.

RCB shifted prediction intervals are predictive diagnostics conditional on observed perturbed means, not ATT confidence intervals.
Raw-difference comparators are empirical references, not causal ground truth.
In the composite figure, RCB differences of mean log1p expression are divided by ln(2), while Wilcoxon values are log2 fold changes; their common color scale does not make them identical estimands.
