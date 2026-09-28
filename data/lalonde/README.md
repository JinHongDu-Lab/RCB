# LaLonde inputs

The `.dta` files and `Farrell2015.csv` retain their upstream names.
The `lalonde*.csv` files are the derived analysis matrices the study reads.
To rebuild derived matrices from the upstream files, from the repository root:

```sh
Rscript experiments/lalonde/build_lalonde171_analysis_data.R
Rscript experiments/lalonde/build_lalonde171_nsw_control_validation_data.R
Rscript experiments/lalonde/build_lalonde_placebo_data.R
```

The builders require R with `foreign` and `nnet`.
