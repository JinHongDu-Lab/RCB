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
The notebook combines good-overlap RMSE, weak-overlap RMSE, RCB effects, and unadjusted effects with Wilcoxon significance flags into `figures/replogle_01_k562_overview.pdf`.
The full-data stage also writes the uniform-weight covariate shift (`full_k562_covariate_shift.csv`) and the whole penalty path of every perturbation-outcome fit (`full_k562_adjustment_path.csv`), and the overlap stage records each weight vector's imbalance against the pilot fold and against all perturbed profiles; the notebook draws the adjustment diagnostics into `figures/replogle_02_k562_adjustment_diagnostics.pdf`.

The production configuration uses 5 perturbations (SLC39A9, GAB2, ORC1, SF3B2 and NCBP2), 25 outcomes per perturbation, a 25-gene full panel, up to 500 non-DE covariates, 100 repetitions, seed 20260829, and a 121-point penalty grid plus the infinite endpoint.
The worker count defaults to 4 and can also be set with `CAUSAL_EXTRAPOLATION_JOBS`.
Each run records its configuration, input hashes, code revision, package versions, stage timings and output hashes in the run manifest.

RCB shifted prediction intervals are predictive diagnostics conditional on observed perturbed means, not ATT confidence intervals.
Raw-difference comparators are empirical references, not causal ground truth.
In the composite figure, both heatmaps show differences of mean log1p expression divided by ln(2): the RCB panel against the covariate-adjusted counterfactual mean, and the unadjusted panel against the raw control mean, so they differ only by the adjustment.
