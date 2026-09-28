# `rcb` — risk-calibrated balancing

The implementation shared by the simulation, evaluation, LaLonde and Replogle studies.

Studies and notebooks run from the repository root, so `import rcb` needs no installation step.

## Layout

| Module | Contents |
|---|---|
| `estimator` | **Start here.** `fit_rcb` runs the proposed estimator end to end on any normalized base weight vector; `reference_penalty_rules` gives the GCV, two-moment, fixed and oracle comparison rules; the top-level README shows it on the ABW balancing component |
| `balancing` | The proposed method's pieces: design geometry, base weights, the ridge-augmented path, risk-based penalty selection, the honest pilot/evaluation split |
| `dgp` | The shared Gaussian AR(1) source covariance (eigensystem, matrix square root) and eigenbasis-relative shift directions the simulation and evaluation studies both build their covariates from |
| `risk` | The deterministic equivalent, the exact conditional risk, the feasible plug-in risk, and the variance components feeding it |
| `nuisance` | Spectral quasi-score estimation of `(r², σ₀²)` from the centered source design and outcomes |
| `benchmarks.bases` | The registry of base weighting rules (`uniform`, `ipw`, `trimmed_ipw`, `overlap`, `entropy`, `sbw`, `cbps`, `l2`, `arb`) and the `BaseSettings` each study drives it with |
| `benchmarks.weights` | The weight-family implementations the registry calls: propensity families, entropy, SBW, CBPS, l2 (ridge) balancing |
| `benchmarks.double_ridge` | The Bruns-Smith augmented l2 estimator and its three cross-validation rules |
| `benchmarks.arb` | Approximate residual balancing |
| `benchmarks.gcv` | Source-distribution GCV, the shift-blind baseline |
| `diagnostics` | Weight normalization, standardization, balance summaries |
| `montecarlo` | Monte Carlo summaries, paired resampling, RMSE-ratio intervals |
| `config`, `io`, `paths`, `style` | Seeding and grids; artifact writing; output locations; figure style |

## Where each manuscript object lives

| Manuscript object | One function |
|---|---|
| Ridge-augmented weights `gamma_lambda = gamma + X0c M_lambda Delta / n0` | `balancing.ridge_augmented_path` |
| The whole estimator: path, plug-in risk, penalty selection, estimate | `estimator.fit_rcb` |
| Feasible plug-in risk `r2 * qhat(lambda) + sigma2 * ||gamma_lambda||^2` | `risk.feasible_risk_geometry` + `risk.feasible_risk_path` |
| Variance components `(r2, sigma0^2)`: two-moment pilot, spectral quasi-score refinement | `risk.estimate_variance_components` (`nuisance.two_moment_estimates`, `nuisance.spectral_quasi_score_estimates`) |
| Exact conditional risk `B_n + V_n` for a fixed design | `risk.exact_risk_components` |
| Deterministic equivalent, general spectrum and isotropic closed form | `risk.deterministic_equivalent_diagonal`, `risk.deterministic_equivalent_isotropic` |
| Design-independent base weights of the deterministic-equivalent theorem; pilot-fold ridge base of the tuning theorem | `balancing.design_independent_base_weights`, `balancing.pilot_ridge_base_weights` |
| Source GCV, two-moment, fixed and oracle comparison rules | `estimator.reference_penalty_rules` |
| Base weighting rules (uniform, IPW, trimmed IPW, overlap, entropy, SBW, CBPS, l2, ARB) | `benchmarks.bases.base_weight_families` with a `BaseSettings` |
| Double-ridge ABW: estimate, outcome-CV rules, design-only balancing-penalty rules | `benchmarks.double_ridge.double_ridge_estimate`, `outcome_cv_*`, `design_only_balance_penalty_cv` |
| Approximate residual balancing: weights, elastic net, complete estimator | `benchmarks.arb.arb_residual_weights`, `arb_elastic_net_fit`, `arb_estimate` |
| Honest pilot/evaluation split | the caller's: base from the pilot fold, `fit_rcb` sees the evaluation fold only |

## Three risk quantities, never conflated

The manuscript distinguishes three things, and so does this code:

- **theoretical** — the finite-dimensional deterministic equivalent (`risk.deterministic_equivalent_diagonal`);
- **exact** — the exact conditional risk for a fixed design, given the true components (`risk.exact_risk_components`);
- **risk estimate** — the feasible plug-in risk under repeated outcomes (`risk.feasible_risk_geometry` with `risk.feasible_risk_path`).

Only the third may select a penalty.
The first is what the theory predicts and the second is what evaluation compares against; tuning on either would make the comparison circular.

## Tests

```
python -m pytest tests/test_equivalence.py -q
```

`tests/baseline/` holds reference implementations and 243 outputs recorded from them; the tests replay the same cases through `rcb` and check the results against those outputs.
