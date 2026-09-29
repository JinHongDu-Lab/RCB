"""Within-K562 overlap stress experiment and full-panel RCB analysis.

Migrated from 4_replogle.py; computation is explicit and imports perform no I/O.
Uniform bases use the full target; adaptive bases use pilot/evaluation folds.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from rcb import paths, risk
from rcb.estimator import fit_rcb, prepare_geometry
from rcb.benchmarks import double_ridge, weights as rcb_weights


@dataclass(frozen=True)
class Config:
    # The supplied analysis selected these five among many tied at 48 rows.
    # Pin their order rather than relying on pandas' unstable count tie ordering.
    PERTURBATIONS: tuple = ("SLC39A9", "GAB2", "ORC1", "SF3B2", "NCBP2")
    N_PERTURBATIONS: int = 5
    N_OUTCOMES_PER_P: int = 25
    FULL_PANEL_SIZE: int = 25
    MIN_ROWS_PER_PERTURBATION: int = 8
    PADJ_THRESHOLD: float = 0.05
    EXCLUDE_SELF_GENE: bool = True
    N_NON_DE: int = 500
    N_REPEATS: int = 100
    N_JOBS: int = 4
    RANDOM_SEED: int = 20260829
    LAMBDAS: tuple = tuple(np.logspace(-3, 3, 121))
    RCB_BASE_ALPHA: float = 1.0
    ATT_TARGET_SUBSET: int = 24
    ATT_WEAK_DIVERGENCE_MULTIPLE: float = 16.0
    ATT_TILT_ALPHAS: tuple = tuple(np.linspace(0, -8, 801))
    THETA_BOUNDS: tuple = ((1e-4, 25.0), (1e-4, 10.0))
    NUISANCE_GRID_SIZE: int = 81

    def __post_init__(self):
        for name in (
            "N_PERTURBATIONS",
            "N_OUTCOMES_PER_P",
            "FULL_PANEL_SIZE",
            "N_NON_DE",
            "N_REPEATS",
            "N_JOBS",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if (
            len(set(self.PERTURBATIONS)) != len(self.PERTURBATIONS)
            or len(self.PERTURBATIONS) < self.N_PERTURBATIONS
        ):
            raise ValueError(
                "Provide enough distinct, explicitly selected perturbations"
            )


INPUT_NAMES = (
    "Replogle-E-k562_pseudobulk.h5ad",
    "Replogle-E-k562_wilcoxon_batch_corrected.h5ad",
)
CONTROL_NAMES, CONTROL = {
    "control",
    "ctrl",
    "non-targeting",
    "nontargeting",
    "non_targeting",
}, "control"

PREDICTION_Z = 1.959963984540054

RCB_BASES, DR_TUNINGS = ("uniform", "l2", "entropy"), ("outcome", "imbalance", "riesz")
METHODS = {
    "pure_uniform": "Uniform weighting",
    "pure_l2": "l2 weighting",
    "pure_entropy": "Regularized entropy weighting",
    "double_ridge_outcome": "Double ridge (Outcome CV)",
    "double_ridge_imbalance": "Double ridge (Imbalance CV)",
    "double_ridge_riesz": "Double ridge (Riesz CV)",
    "rcb_uniform": "RCB (uniform base)",
    "rcb_l2": "RCB (l2 base)",
    "rcb_entropy": "RCB (entropy base)",
    "rcb_on_abw_outcome": "RCB on ABW (Outcome-CV base)",
    "rcb_on_abw_imbalance": "RCB on ABW (Imbalance-CV base)",
    "rcb_on_abw_riesz": "RCB on ABW (Riesz-CV base)",
}
REFERENCE_METHOD = "double_ridge_imbalance"


def summarize_overlap_penalties(att_table: pd.DataFrame) -> pd.DataFrame:
    """Selected-penalty spread for each RCB base, by overlap regime.

    The overlap experiment resamples the 24 target observations and the
    pilot/evaluation partition in every replication, so the spread of the
    selected penalty is a finite-sample split diagnostic rather than an
    estimate of anything.  Two spreads are reported because they answer
    different questions: the pooled quartiles describe the selected penalties
    over replications *and* perturbation--outcome pairs, which mixes split
    variation with genuine heterogeneity between pairs, while
    ``within_pair_median_log10_iqr`` is the median over pairs of the
    across-replication interquartile width at a fixed pair, which isolates
    the split variation.

    The exact no-augmentation endpoint is ``lambda = infinity`` and has no
    logarithm, so it is excluded from the quartiles and reported separately as
    ``endpoint_fraction``.  A pair contributes to the within-pair figure only
    when at least ten of its replications selected a finite penalty; under
    good overlap the endpoint is common enough that many pairs do not.
    """
    rows = []
    for base in RCB_BASES:
        column = f"penalty_rcb_{base}"
        for regime, group in att_table.groupby("regime", sort=False):
            penalties = group[column].to_numpy(float)
            finite = np.isfinite(penalties)
            logged = np.log10(penalties[finite])
            # A small run can select the endpoint in every replication.
            if logged.size:
                median, (q1, q3) = np.median(logged), np.percentile(logged, [25, 75])
            else:
                median = q1 = q3 = float("nan")
            widths = []
            for _, pair in group.groupby(["perturbation_p", "outcome_h"], sort=False):
                values = pair[column].to_numpy(float)
                pair_finite = np.isfinite(values)
                if pair_finite.sum() >= 10:
                    pq1, pq3 = np.percentile(np.log10(values[pair_finite]), [25, 75])
                    widths.append(pq3 - pq1)
            rows.append({
                "method": METHODS[f"rcb_{base}"],
                "base": base,
                "regime": regime,
                "selections": len(penalties),
                "pairs": group.groupby(["perturbation_p", "outcome_h"]).ngroups,
                "replications": group["repeat"].nunique(),
                "endpoint_fraction": float((~finite).mean()),
                "median_log10_penalty": float(median),
                "log10_penalty_q1": float(q1),
                "log10_penalty_q3": float(q3),
                "within_pair_median_log10_iqr": (
                    float(np.median(widths)) if widths else float("nan")
                ),
                "within_pair_count": len(widths),
            })
    return pd.DataFrame(rows)


class Analysis:
    def __init__(self, config: Config):
        self.config = config
        self.output = paths.results("replogle", create=False)
        self.inputs = tuple(
            paths.data("replogle") / "raw" / name for name in INPUT_NAMES
        )

    @contextmanager
    def open_inputs(self):
        datasets = []
        try:
            for path in self.inputs:
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing Replogle input: {path}. See data/replogle/README.md."
                    )
                datasets.append(ad.read_h5ad(path, backed="r"))
            yield tuple(datasets)
        finally:
            for dataset in datasets:
                if dataset.isbacked:
                    dataset.file.close()

    def validate_inputs(self):
        with self.open_inputs() as (pb, de):
            if "perturbation" not in pb.obs:
                raise ValueError("Pseudobulk obs must contain perturbation labels")
            if not pb.var_names.is_unique or not de.var_names.is_unique:
                raise ValueError("Gene identifiers must be unique")
            if not pb.var_names.equals(de.var_names):
                raise ValueError("Expected matching K562 gene identifiers and order")
            missing = {"pvalue_adj", "z_score", "logfoldchanges"} - set(de.layers)
            if missing:
                raise ValueError(f"Missing DE layers: {sorted(missing)}")
            labels = self.get_perturbation_labels(pb)
            if np.count_nonzero(labels == CONTROL) < 5:
                raise ValueError("At least five rows labelled control are required")
            de_labels = self.get_perturbation_labels(de)
            if len(set(de_labels)) != len(de_labels):
                raise ValueError("DE perturbation identifiers must be unique")
            counts = pd.Series(labels).value_counts()
            eligible = [
                p
                for p, n in counts.items()
                if p.lower() not in CONTROL_NAMES
                and n >= self.config.MIN_ROWS_PER_PERTURBATION
                and p in set(de_labels)
            ]
            if len(eligible) < self.config.N_PERTURBATIONS:
                raise ValueError(
                    "Too few eligible perturbations for the requested configuration"
                )
            if not set(self.config.PERTURBATIONS[: self.config.N_PERTURBATIONS]) <= set(
                eligible
            ):
                raise ValueError(
                    "A configured perturbation is missing or has too few observations"
                )
            return {
                "pseudobulk_shape": list(pb.shape),
                "de_shape": list(de.shape),
                "eligible_perturbations": len(eligible),
            }

    def to_dense(self, x):
        return np.asarray(x.toarray() if sparse.issparse(x) else x, dtype=float)

    def get_perturbation_labels(self, adata):
        return (
            adata.obs["perturbation"].astype(str).to_numpy()
            if "perturbation" in adata.obs
            else adata.obs_names.astype(str).to_numpy()
        )

    def read_expression(self, adata, rows, genes):
        # Backed dense HDF5 supports only one fancy-index axis at a time.
        # Read the selected rows, then reorder/subset genes in memory.
        columns = adata.var_names.get_indexer(genes)
        if np.any(columns < 0):
            raise ValueError("Selected genes are missing from the expression matrix")
        return self.to_dense(adata[np.asarray(rows, int), :].X)[:, columns]

    def read_de_row(self, de, row):
        return tuple(
            (
                self.to_dense(de.layers[layer][row, :]).reshape(-1)
                for layer in ("pvalue_adj", "z_score", "logfoldchanges")
            )
        )

    def extract_de_tables(self, de):
        return tuple(
            (
                self.to_dense(de.layers[layer])
                for layer in ("z_score", "pvalue_adj", "logfoldchanges")
            )
        )

    def sample_positions(self, n, size, rng):
        return (
            np.arange(n)
            if size is None or n <= size
            else np.sort(rng.choice(n, size, replace=False))
        )

    def split_target(self, X, seed):
        idx = np.random.default_rng(seed).permutation(len(X))
        cut = len(X) // 2
        return (X[idx[:cut]], X[idx[cut:]])

    def make_base_weights(self, X0, X1_pilot, base):
        X0, X1_pilot = (np.asarray(X0, float), np.asarray(X1_pilot, float))
        if base == "uniform":
            return np.full(len(X0), 1.0 / len(X0))
        if base == "l2":
            return rcb_weights.l2_balancing_weights(
                X0 - X0.mean(0),
                X1_pilot.mean(0) - X0.mean(0),
                self.config.RCB_BASE_ALPHA,
            )
        if base == "entropy":
            ridge = rcb_weights.entropy_stabilization_penalty(
                X0.shape[1], len(X1_pilot)
            )
            return rcb_weights.entropy_balancing_weights(
                X0,
                X1_pilot.mean(0),
                ridge=ridge,
                maxiter=250,
                ftol=1e-11,
                require_success=True,
            )[0]
        raise ValueError(base)

    def fit_double_ridge_variants(self, X0, y0, Xt, balance_tuners=None):
        X0, y0, Xt = (
            np.asarray(X0, float),
            np.asarray(y0, float).reshape(-1),
            np.asarray(Xt, float),
        )
        X0c, shift = (X0 - X0.mean(0), Xt.mean(0) - X0.mean(0))
        lam_cv = float(double_ridge.outcome_cv_fixed_folds_raw_grid(X0c, y0)[0])
        tuners = (
            double_ridge.design_only_balance_penalty_cv(X0, shift)
            if balance_tuners is None
            else balance_tuners
        )
        params = {
            "outcome": (lam_cv, lam_cv),
            "imbalance": (float(tuners["delta_cv_imbalance"]), lam_cv),
            "riesz": (float(tuners["delta_cv_riesz"]), lam_cv),
        }
        fits = {}
        for name, (delta, lam) in params.items():
            fit = double_ridge.double_ridge_estimate(X0c, y0, shift, delta, lam)
            gamma, base_gamma = (
                np.asarray(fit["augmented_weights"], float),
                np.asarray(fit["base_weights"], float),
            )
            if not np.all(np.isfinite(gamma)) or not np.all(np.isfinite(base_gamma)):
                raise FloatingPointError(f"Non-finite Double-Ridge weights for {name}.")
            fits[name] = {
                "mu": float(fit["muhat0"]),
                "gamma": gamma,
                "base_gamma": base_gamma,
                "penalty": float(delta),
                "outcome_penalty": float(lam),
                "weight_sum": float(gamma.sum()),
                "base_weight_sum": float(base_gamma.sum()),
                "linear_representation_error": float(
                    fit["linear_representation_error"]
                ),
                "outcome_penalty_at_lower_bound": bool(
                    np.isclose(lam * len(y0), double_ridge.ABW_AUTHORS_OUTCOME_GRID[0])
                ),
            }
        return fits

    def estimate_nuisance(self, X0, y0, geometry):
        r2, sigma2, _ = risk.estimate_variance_components(
            X0,
            y0,
            geometry["eigenvalues"],
            geometry["eigenvectors"],
            theta_bounds=self.config.THETA_BOUNDS,
            grid_size=self.config.NUISANCE_GRID_SIZE,
            standardize_outcomes=False,
        )
        return (
            float(np.asarray(r2).reshape(-1)[0]),
            float(np.asarray(sigma2).reshape(-1)[0]),
        )

    def fit_rcb_on_base(
        self, X0, y0, Xt_eval, gamma, geometry, r2, sigma2, *, return_path=False
    ):
        gamma = np.asarray(gamma, float).reshape(-1)
        total = float(gamma.sum())
        if (
            not np.all(np.isfinite(gamma))
            or not np.isfinite(total)
            or abs(total) < 1e-12
        ):
            raise FloatingPointError(
                "RCB base weights are non-finite or have near-zero total weight."
            )
        if not np.isclose(total, 1.0, atol=1e-08):
            gamma = gamma / total
        fit = fit_rcb(
            np.asarray(X0, float),
            np.asarray(y0, float).reshape(-1),
            np.asarray(Xt_eval, float),
            gamma,
            self.config.LAMBDAS,
            theta_bounds=self.config.THETA_BOUNDS,
            grid_size=self.config.NUISANCE_GRID_SIZE,
            standardize_outcomes=False,
            include_infinite_endpoint=True,
            geometry=geometry,
            variance_components=(r2, sigma2),
        )
        out = {
            "mu": float(fit.muhat0),
            "gamma": np.asarray(fit.weights),
            "penalty": float(fit.selected_lambda),
            "risk": float(fit.risk_path[fit.selected_index]),
            "no_adjustment": bool(fit.selected_no_adjustment_endpoint),
        }
        if return_path:
            # Whole penalty path, for the full-data adjustment diagnostics; the
            # last point is the no-augmentation endpoint.
            out["path"] = {
                "penalty": np.asarray(fit.penalty_grid, float),
                "counterfactual_mean": np.asarray(fit.estimate_path, float),
                "risk": np.asarray(fit.risk_path, float),
                "qhat": np.asarray(fit.qhat_path, float),
                "ess": 1.0 / np.asarray(fit.norm_squared_path, float),
            }
        return out

    def prepare_design(self, X0, X1_pilot):
        """Cache outcome-independent work for genes sharing this exact design."""
        return {
            "geometry": prepare_geometry(X0),
            "bases": {
                base: self.make_base_weights(X0, X1_pilot, base) for base in RCB_BASES
            },
            "balance_tuners": double_ridge.design_only_balance_penalty_cv(
                X0, X1_pilot.mean(0) - X0.mean(0)
            ),
        }

    def fit_all_methods(self, X0, y0, X1, X1_pilot, X1_eval, prepared=None):
        X0, y0, X1 = (
            np.asarray(X0, float),
            np.asarray(y0, float).reshape(-1),
            np.asarray(X1, float),
        )
        if prepared is None:
            prepared = self.prepare_design(X0, X1_pilot)
        geometry = prepared["geometry"]
        r2, sigma2 = self.estimate_nuisance(X0, y0, geometry)
        bases = prepared["bases"]
        dr_base = self.fit_double_ridge_variants(
            X0, y0, X1_pilot, prepared["balance_tuners"]
        )
        fits = {
            f"pure_{base}": {
                "mu": float(bases[base] @ y0),
                "gamma": bases[base],
                "penalty": np.nan,
            }
            for base in RCB_BASES
        }
        fits.update({f"double_ridge_{t}": dr_base[t] for t in DR_TUNINGS})
        fits.update(
            {
                f"rcb_{base}": self.fit_rcb_on_base(
                    X0,
                    y0,
                    X1 if base == "uniform" else X1_eval,
                    bases[base],
                    geometry,
                    r2,
                    sigma2,
                )
                for base in RCB_BASES
            }
        )
        fits.update(
            {
                f"rcb_on_abw_{t}": {
                    **self.fit_rcb_on_base(
                        X0, y0, X1_eval, dr_base[t]["base_gamma"], geometry, r2, sigma2
                    ),
                    "base_penalty": dr_base[t]["penalty"],
                    "base_outcome_penalty": dr_base[t]["outcome_penalty"],
                    "base_weight_sum": dr_base[t]["base_weight_sum"],
                    "base_linear_representation_error": dr_base[t][
                        "linear_representation_error"
                    ],
                    "base_outcome_penalty_at_lower_bound": dr_base[t][
                        "outcome_penalty_at_lower_bound"
                    ],
                }
                for t in DR_TUNINGS
            }
        )
        return fits

    @staticmethod
    def balance_diagnostics(fits, X0, X1_pilot, X_population):
        """Imbalance of each weight vector against the pilot-fold mean (the
        in-sample balancing target of the base weights) and against the mean of
        all perturbed profiles (a population reference), for diagnosing
        whether balancing on a small pilot fold carries over to the target."""
        out = {}
        for key, fit in fits.items():
            balanced = X0.T @ np.asarray(fit["gamma"], float)
            out[f"pilot_imbalance_{key}"] = float(
                np.linalg.norm(X1_pilot.mean(0) - balanced)
            )
            out[f"population_imbalance_{key}"] = float(
                np.linalg.norm(X_population.mean(0) - balanced)
            )
        return out

    def fit_metadata(self, fits, X0, Xt, prefix=""):
        out = {}
        for key, fit in fits.items():
            gamma = np.asarray(fit["gamma"], float)
            out[f"{prefix}ess_{key}"] = float(1 / np.sum(gamma**2))
            out[f"{prefix}imbalance_{key}"] = float(
                np.linalg.norm(Xt.mean(0) - X0.T @ gamma)
            )
            if "penalty" in fit:
                out[f"{prefix}penalty_{key}"] = float(fit["penalty"])
            if "outcome_penalty" in fit:
                out[f"{prefix}outcome_penalty_{key}"] = float(fit["outcome_penalty"])
            if "weight_sum" in fit:
                out[f"{prefix}weight_sum_{key}"] = float(fit["weight_sum"])
            if "linear_representation_error" in fit:
                out[f"{prefix}linear_representation_error_{key}"] = float(
                    fit["linear_representation_error"]
                )
            if "outcome_penalty_at_lower_bound" in fit:
                out[f"{prefix}outcome_penalty_at_lower_bound_{key}"] = bool(
                    fit["outcome_penalty_at_lower_bound"]
                )
            if "base_penalty" in fit:
                out[f"{prefix}base_penalty_{key}"] = float(fit["base_penalty"])
            if "base_outcome_penalty" in fit:
                out[f"{prefix}base_outcome_penalty_{key}"] = float(
                    fit["base_outcome_penalty"]
                )
            if "base_weight_sum" in fit:
                out[f"{prefix}base_weight_sum_{key}"] = float(fit["base_weight_sum"])
            if "base_linear_representation_error" in fit:
                out[f"{prefix}base_linear_representation_error_{key}"] = float(
                    fit["base_linear_representation_error"]
                )
            if "base_outcome_penalty_at_lower_bound" in fit:
                out[f"{prefix}base_outcome_penalty_at_lower_bound_{key}"] = bool(
                    fit["base_outcome_penalty_at_lower_bound"]
                )
            if "risk" in fit:
                out[f"{prefix}risk_{key}"] = float(fit["risk"])
            if "no_adjustment" in fit:
                out[f"{prefix}no_adjustment_{key}"] = bool(fit["no_adjustment"])
        return out

    def summarize_results(self, df, group_cols, references):
        rows = []
        for keys, g in df.groupby(group_cols, sort=False) if group_cols else [((), df)]:
            keys = keys if isinstance(keys, tuple) else (keys,)
            base = dict(zip(group_cols, keys))
            for ref_name, ref_col in references.items():
                ref = g[ref_col].to_numpy(float)
                baseline_err = g[f"effect_{REFERENCE_METHOD}"].to_numpy(float) - ref
                baseline_rmse = float(np.sqrt(np.mean(baseline_err**2)))
                for key, label in METHODS.items():
                    err = g[f"effect_{key}"].to_numpy(float) - ref
                    rmse = float(np.sqrt(np.mean(err**2)))
                    rows.append(
                        {
                            **base,
                            "reference_type": ref_name,
                            "method_key": key,
                            "method": label,
                            "n": len(g),
                            "bias": float(err.mean()),
                            "mae": float(np.abs(err).mean()),
                            "rmse": rmse,
                            "rmse_ratio_vs_dr_imbalance": (
                                rmse / baseline_rmse if baseline_rmse else np.nan
                            ),
                        }
                    )
        return pd.DataFrame(rows)

    def overall_summary(self, summary, group_cols):
        return (
            summary.groupby(
                group_cols + ["reference_type", "method_key", "method"], sort=False
            )
            .agg(
                mean_pair_rmse=("rmse", "mean"),
                mean_rmse_ratio_vs_dr_imbalance=("rmse_ratio_vs_dr_imbalance", "mean"),
            )
            .reset_index()
        )

    def classifier_source_tilt(self, X0, X1_pilot, seed):
        X, z = (
            np.vstack([X0, X1_pilot]),
            np.r_[np.zeros(len(X0), int), np.ones(len(X1_pilot), int)],
        )
        cv = StratifiedKFold(
            n_splits=max(2, min(5, np.bincount(z).min())),
            shuffle=True,
            random_state=seed,
        )
        e = np.clip(
            cross_val_predict(
                LogisticRegression(
                    C=1.0, solver="liblinear", max_iter=2000, random_state=seed
                ),
                X,
                z,
                cv=cv,
                method="predict_proba",
            )[: len(X0), 1],
            0.0001,
            1 - 0.0001,
        )
        r = np.clip(len(X0) / len(X1_pilot) * e / (1 - e), 0.001, 1000.0)
        p, log_r = (r / r.sum(), np.log(r))
        q = lambda a: (lambda w: w / w.sum())(np.exp(a * log_r - np.max(a * log_r)))
        chi2 = lambda qq: float(np.sum(p**2 / np.clip(qq, 1e-12, None)) - 1)
        q_good, baseline = (
            np.full(len(X0), 1 / len(X0)),
            chi2(np.full(len(X0), 1 / len(X0))),
        )
        discrepancies = np.array([chi2(q(a)) for a in self.config.ATT_TILT_ALPHAS])
        k = int(
            np.argmin(
                np.abs(
                    discrepancies - self.config.ATT_WEAK_DIVERGENCE_MULTIPLE * baseline
                )
            )
        )
        return {
            "q_good": q_good,
            "q_weak": q(self.config.ATT_TILT_ALPHAS[k]),
            "baseline": baseline,
            "weak": float(discrepancies[k]),
            "alpha": float(self.config.ATT_TILT_ALPHAS[k]),
        }

    def att_source_sets(self, X1_pilot, seed):
        tilt, n = (
            self.classifier_source_tilt(self.k562_x[CONTROL], X1_pilot, seed),
            len(self.k562_x[CONTROL]),
        )
        return (
            {
                "good": np.arange(n),
                "weak": np.random.default_rng(seed + 7919).choice(
                    n, n, replace=True, p=tilt["q_weak"]
                ),
            },
            tilt,
        )

    def overlap_diagnostics(self, X0, X1_eval):
        D, k = (
            np.linalg.norm(X1_eval[:, None, :] - X0[None, :, :], axis=2),
            min(3, len(X0)),
        )
        return (
            float(np.linalg.norm(X1_eval.mean(0) - X0.mean(0))),
            float(np.partition(D, k - 1, axis=1)[:, :k].mean()),
        )

    def run_att_repeat(self, repeat):
        rows = []
        # The original seed formula depends on repetition and perturbation,
        # not outcome: all its genes share the same target/source draws.
        designs = {}
        for p, h in self.pair_table[["perturbation_p", "outcome_h"]].itertuples(
            index=False, name=None
        ):
            seed = self.config.RANDOM_SEED + 1000 * repeat + self.selected_p.index(p)
            rng, j = (np.random.default_rng(seed), self.outcome_col[h])
            target_pos = self.sample_positions(
                len(self.k562_x[p]), self.config.ATT_TARGET_SUBSET, rng
            )
            X1, y1 = (self.k562_x[p][target_pos], self.k562_y[p][target_pos, j])
            X1_pilot, X1_eval = self.split_target(X1, seed)
            reference = float(y1.mean() - self.k562_y[CONTROL][:, j].mean())
            full_reference = float(
                self.k562_y[p][:, j].mean() - self.k562_y[CONTROL][:, j].mean()
            )
            source_sets, tilt = self.att_source_sets(X1_pilot, seed)
            for regime, source_pos in source_sets.items():
                try:
                    X0, y0 = (
                        self.k562_x[CONTROL][source_pos],
                        self.k562_y[CONTROL][source_pos, j],
                    )
                    key = (p, regime)
                    if key not in designs:
                        designs[key] = self.prepare_design(X0, X1_pilot)
                    fits = self.fit_all_methods(
                        X0, y0, X1, X1_pilot, X1_eval, designs[key]
                    )
                    shift, knn = self.overlap_diagnostics(X0, X1_eval)
                    q_regime = tilt["q_good"] if regime == "good" else tilt["q_weak"]
                    discrepancy = tilt["baseline"] if regime == "good" else tilt["weak"]
                    row = {
                        "repeat": repeat + 1,
                        "seed": seed,
                        "perturbation_p": p,
                        "outcome_h": h,
                        "regime": regime,
                        "target_fixed_full_control_reference": reference,
                        "full_data_reference": full_reference,
                        "classifier_discrepancy_pilot": discrepancy,
                        "classifier_discrepancy_multiple": (
                            discrepancy / tilt["baseline"]
                            if tilt["baseline"] > 0
                            else np.nan
                        ),
                        "tilt_alpha": 0.0 if regime == "good" else tilt["alpha"],
                        "source_sampling_ess": float(1 / np.sum(q_regime**2)),
                        "n_source": len(source_pos),
                        "n_unique_source": len(np.unique(source_pos)),
                        "mean_shift_eval": shift,
                        "mean_knn_distance_eval": knn,
                    }
                    row.update(
                        {
                            f"effect_{key}": float(y1.mean() - fit["mu"])
                            for (key, fit) in fits.items()
                        }
                    )
                    row.update(self.fit_metadata(fits, X0, X1_eval))
                    row.update(
                        self.balance_diagnostics(fits, X0, X1_pilot, self.k562_x[p])
                    )
                    rows.append(row)
                except Exception as exc:
                    raise RuntimeError(
                        f"ATT failed: repeat={repeat + 1}, ({p}, {h}), regime={regime}: {type(exc).__name__}: {exc}"
                    ) from exc
        return rows

    def select(self):
        with self.open_inputs() as (k562_pb_sel, k562_de_sel):
            k562_z, k562_padj, k562_lfc = self.extract_de_tables(k562_de_sel)
            k562_counts = k562_pb_sel.obs["perturbation"].astype(str).value_counts()
            k562_de_perts, k562_genes = (
                self.get_perturbation_labels(k562_de_sel),
                np.asarray(k562_de_sel.var_names.astype(str)),
            )
            k562_pert_row = {p: i for (i, p) in enumerate(k562_de_perts)}
            eligible = [
                p
                for (p, n) in k562_counts.items()
                if p.lower() not in CONTROL_NAMES
                and n >= self.config.MIN_ROWS_PER_PERTURBATION
                and (p in k562_pert_row)
            ]
            perturbation_summary = (
                pd.DataFrame(
                    {
                        "perturbation": eligible,
                        "n_k562": [int(k562_counts[p]) for p in eligible],
                    }
                )
                .sort_values("n_k562", ascending=False)
                .reset_index(drop=True)
            )
            selected_perts = list(
                self.config.PERTURBATIONS[: self.config.N_PERTURBATIONS]
            )
            rows = []
            for p in selected_perts:
                r = k562_pert_row[p]
                candidates = (
                    pd.DataFrame(
                        {
                            "perturbation_p": p,
                            "outcome_h": k562_genes,
                            "k562_z": k562_z[r],
                            "k562_padj": k562_padj[r],
                            "k562_logFC": k562_lfc[r],
                        }
                    )
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna(subset=["k562_z", "k562_padj", "k562_logFC"])
                )
                if self.config.EXCLUDE_SELF_GENE:
                    candidates = candidates[candidates["outcome_h"] != p]
                candidates["k562_significant"] = (
                    candidates["k562_padj"] < self.config.PADJ_THRESHOLD
                )
                candidates["k562_abs_z"], candidates["k562_abs_logFC"] = (
                    candidates["k562_z"].abs(),
                    candidates["k562_logFC"].abs(),
                )
                rows.extend(
                    candidates.sort_values(
                        [
                            "k562_significant",
                            "k562_abs_z",
                            "k562_abs_logFC",
                            "k562_padj",
                        ],
                        ascending=[False, False, False, True],
                    )
                    .head(self.config.N_OUTCOMES_PER_P)
                    .to_dict("records")
                )
            self.pair_table = (
                pd.DataFrame(rows)
                .drop_duplicates(["perturbation_p", "outcome_h"])
                .reset_index(drop=True)
            )
            sel_rows = np.array([k562_pert_row[p] for p in selected_perts])
            self.panel = pd.DataFrame(
                {
                    "outcome_h": k562_genes,
                    "max_abs_z": np.nanmax(np.abs(k562_z[sel_rows]), axis=0),
                    "min_padj": np.nanmin(
                        np.where(
                            np.isfinite(k562_padj[sel_rows]), k562_padj[sel_rows], 1.0
                        ),
                        axis=0,
                    ),
                    "n_de_significant": np.sum(
                        k562_padj[sel_rows] < self.config.PADJ_THRESHOLD, axis=0
                    ),
                }
            )
            self.panel = (
                self.panel[
                    np.isfinite(self.panel["max_abs_z"])
                    & ~self.panel["outcome_h"].isin(selected_perts)
                ]
                .sort_values(["max_abs_z", "min_padj"], ascending=[False, True])
                .head(self.config.FULL_PANEL_SIZE)
                .reset_index(drop=True)
            )
            self.full_panel_genes = self.panel["outcome_h"].tolist()
            perturbation_summary.to_csv(
                self.output / "perturbation_ranking.csv", index=False
            )
            self.pair_table.to_csv(
                self.output / "selected_perturbation_outcome_pairs.csv", index=False
            )
            self.panel.to_csv(self.output / "full_k562_outcome_panel.csv", index=False)
            print(f"\nSelected perturbations ({len(selected_perts)}):", selected_perts)
            print(
                f"Part I pairs: {len(self.pair_table)} = {len(selected_perts)} perturbations × up to {self.config.N_OUTCOMES_PER_P} outcomes"
            )
            print(
                "\nPairs per perturbation:\n",
                self.pair_table.groupby("perturbation_p").size().to_string(),
            )
            print("\nPart II fixed outcome panel:\n", self.panel.to_string(index=False))

    def prepare(self):
        with self.open_inputs() as (k562_pb, k562_de):
            self.selected_p = (
                self.pair_table["perturbation_p"].drop_duplicates().tolist()
            )
            synthetic_outcome_genes = (
                self.pair_table["outcome_h"].drop_duplicates().tolist()
            )
            outcome_genes = sorted(
                set(synthetic_outcome_genes) | set(self.full_panel_genes)
            )
            k562_labels, k562_de_labels = map(
                self.get_perturbation_labels, (k562_pb, k562_de)
            )
            k562_groups = {
                CONTROL: np.flatnonzero(k562_labels == CONTROL),
                **{p: np.flatnonzero(k562_labels == p) for p in self.selected_p},
            }
            k562_de_lookup = {p: i for (i, p) in enumerate(k562_de_labels)}
            k562_de_genes = k562_de.var_names.astype(str).to_numpy()
            self.de_cache = {
                p: dict(
                    zip(
                        ("p_adj", "z", "lfc"),
                        self.read_de_row(k562_de, k562_de_lookup[p]),
                    )
                )
                for p in self.selected_p
            }
            common_features = np.array(
                sorted(set(k562_pb.var_names.astype(str)) & set(k562_de_genes)),
                dtype=object,
            )
            self.de_gene_col = {g: i for (i, g) in enumerate(k562_de_genes)}
            excluded = set(self.selected_p) | set(outcome_genes)
            for p in self.selected_p:
                d = self.de_cache[p]
                order = np.lexsort(
                    (
                        -np.nan_to_num(np.abs(d["lfc"]), nan=0.0),
                        -np.nan_to_num(np.abs(d["z"]), nan=0.0),
                        np.nan_to_num(d["p_adj"], nan=1.0),
                    )
                )
                excluded.update(k562_de_genes[order[:20]])
            idx = np.array([self.de_gene_col[g] for g in common_features], int)
            p_mat = np.vstack([self.de_cache[p]["p_adj"][idx] for p in self.selected_p])
            z_mat = np.vstack([self.de_cache[p]["z"][idx] for p in self.selected_p])
            lfc_mat = np.vstack([self.de_cache[p]["lfc"][idx] for p in self.selected_p])
            mean_p = np.nanmean(
                np.nan_to_num(p_mat, nan=1.0, posinf=1.0, neginf=1.0), axis=0
            )
            mean_abs_z, mean_abs_lfc = (
                np.nanmean(np.abs(z_mat), axis=0),
                np.nanmean(np.abs(lfc_mat), axis=0),
            )
            eligible = np.array([g not in excluded for g in common_features])
            order = np.lexsort((common_features, mean_abs_lfc, mean_abs_z, -mean_p))
            self.feature_genes = common_features[
                order[eligible[order]][: self.config.N_NON_DE]
            ]
            X_control = self.read_expression(
                k562_pb, k562_groups[CONTROL], self.feature_genes
            )
            feature_mean, feature_std = (X_control.mean(0), X_control.std(0))
            keep = (
                np.isfinite(feature_mean)
                & np.isfinite(feature_std)
                & (feature_std > 1e-08)
            )
            self.feature_genes, feature_mean, feature_std = (
                self.feature_genes[keep],
                feature_mean[keep],
                feature_std[keep],
            )

            def standardize_x(adata, rows):
                return (
                    self.read_expression(adata, rows, self.feature_genes) - feature_mean
                ) / feature_std

            self.k562_x = {
                g: standardize_x(k562_pb, rows) for (g, rows) in k562_groups.items()
            }
            self.k562_y = {
                g: self.read_expression(k562_pb, rows, outcome_genes)
                for (g, rows) in k562_groups.items()
            }
            self.outcome_col = {h: j for (j, h) in enumerate(outcome_genes)}
            pd.DataFrame({"feature_gene": self.feature_genes}).to_csv(
                self.output / "covariate_genes.csv", index=False
            )
            print(
                f"\nCovariates: {len(self.feature_genes)} standardized K562 gene-expression features used directly (no PCA)."
            )
            for p in self.selected_p:
                print(
                    f"{p}: K562 control={len(k562_groups[CONTROL])}, perturbed={len(k562_groups[p])}"
                )
        if len(self.feature_genes) == 0:
            raise ValueError("No finite, nonconstant covariates remain after selection")
        if any(
            not np.isfinite(x).all()
            for x in (*self.k562_x.values(), *self.k562_y.values())
        ):
            raise ValueError("Non-finite expression values in selected analysis data")

    def run_overlap(self):
        with parallel_config(backend="loky", inner_max_num_threads=1):
            att_rows = [
                row
                for batch in Parallel(n_jobs=self.config.N_JOBS, verbose=10)(
                    (
                        delayed(self.run_att_repeat)(repeat)
                        for repeat in range(self.config.N_REPEATS)
                    )
                )
                for row in batch
            ]
        self.att_table = pd.DataFrame(att_rows)
        att_summary = self.summarize_results(
            self.att_table,
            ["perturbation_p", "outcome_h", "regime"],
            {
                "Target-fixed / full-control": "target_fixed_full_control_reference",
                "All-data raw difference": "full_data_reference",
            },
        )
        att_overall = self.overall_summary(att_summary, ["regime"])
        att_diagnostics = (
            self.att_table.groupby("regime", sort=False)
            .agg(
                classifier_discrepancy_pilot=("classifier_discrepancy_pilot", "mean"),
                classifier_discrepancy_multiple=(
                    "classifier_discrepancy_multiple",
                    "mean",
                ),
                tilt_alpha=("tilt_alpha", "mean"),
                source_sampling_ess=("source_sampling_ess", "mean"),
                n_unique_source=("n_unique_source", "mean"),
                mean_shift_eval=("mean_shift_eval", "mean"),
                mean_knn_distance_eval=("mean_knn_distance_eval", "mean"),
            )
            .reset_index()
        )
        print(
            "\nATT overlap diagnostics:\n",
            att_diagnostics.round(4).to_string(index=False),
        )
        print(
            "\nATT overall comparison:\n", att_overall.round(4).to_string(index=False)
        )
        self.att_table.to_csv(
            self.output / "att_overlap_tilt_repeated_raw.csv", index=False
        )
        att_summary.to_csv(
            self.output / "att_overlap_tilt_pair_summary.csv", index=False
        )
        att_overall.to_csv(
            self.output / "att_overlap_tilt_overall_summary.csv", index=False
        )
        att_diagnostics.to_csv(
            self.output / "att_overlap_tilt_diagnostics.csv", index=False
        )
        summarize_overlap_penalties(self.att_table).to_csv(
            self.output / "att_overlap_tilt_penalty_summary.csv", index=False
        )

    def run_full(self):
        X0 = self.k562_x[CONTROL]
        full_geometry = prepare_geometry(X0)
        uniform_gamma = np.full(len(X0), 1.0 / len(X0))
        full_nuisance = {
            h: self.estimate_nuisance(
                X0, self.k562_y[CONTROL][:, self.outcome_col[h]], full_geometry
            )
            for h in self.full_panel_genes
        }
        full_rows, path_rows, shift_rows = [], [], []
        for p in self.selected_p:
            Xt = self.k562_x[p]
            # Uniform-weight covariate shift of the standardized features.
            shift = Xt.mean(0) - X0.mean(0)
            shift_rows += [
                {"perturbation_p": p, "feature": g, "mean_shift": float(d)}
                for g, d in zip(self.feature_genes, shift)
            ]
            for h in self.full_panel_genes:
                j, jd = (self.outcome_col[h], self.de_gene_col[h])
                y0, y1 = (self.k562_y[CONTROL][:, j], self.k562_y[p][:, j])
                try:
                    fit = self.fit_rcb_on_base(
                        X0,
                        y0,
                        Xt,
                        uniform_gamma,
                        full_geometry,
                        *full_nuisance[h],
                        return_path=True,
                    )
                    path = fit["path"]
                    # Adjustment = RCB contrast minus unadjusted contrast
                    # = raw control mean minus RCB counterfactual mean.
                    path_rows += [
                        {
                            "perturbation_p": p,
                            "outcome_h": h,
                            "penalty": lam,
                            "rcb_adjustment": float(y0.mean() - mu),
                            "estimated_risk": risk_value,
                            "qhat": qhat,
                            "ess": ess,
                            "selected": bool(k == int(np.argmin(path["risk"]))),
                        }
                        for k, (lam, mu, risk_value, qhat, ess) in enumerate(
                            zip(
                                path["penalty"],
                                path["counterfactual_mean"],
                                path["risk"],
                                path["qhat"],
                                path["ess"],
                            )
                        )
                    ]
                    pred_sd = np.sqrt(max(fit["risk"], 0.0))
                    mu_lo, mu_hi = (
                        fit["mu"] - PREDICTION_Z * pred_sd,
                        fit["mu"] + PREDICTION_Z * pred_sd,
                    )
                    effect, effect_lo, effect_hi = (
                        float(y1.mean() - fit["mu"]),
                        float(y1.mean() - mu_hi),
                        float(y1.mean() - mu_lo),
                    )
                    gamma = np.asarray(fit["gamma"], float)
                    full_rows.append(
                        {
                            "perturbation_p": p,
                            "outcome_h": h,
                            "n_control": len(X0),
                            "n_perturbed": len(Xt),
                            "observed_perturbed_mean": float(y1.mean()),
                            "rcb_counterfactual_mean": fit["mu"],
                            "raw_control_mean": float(y0.mean()),
                            "raw_effect": float(y1.mean() - y0.mean()),
                            "rcb_effect": effect,
                            "rcb_counterfactual_pi_low": mu_lo,
                            "rcb_counterfactual_pi_high": mu_hi,
                            "rcb_effect_pi_low": effect_lo,
                            "rcb_effect_pi_high": effect_hi,
                            "rcb_interval_excludes_zero": bool(
                                effect_lo > 0 or effect_hi < 0
                            ),
                            "rcb_selected_lambda": fit["penalty"],
                            "rcb_estimated_risk": fit["risk"],
                            "rcb_no_adjustment": fit["no_adjustment"],
                            "rcb_ess": float(1 / np.sum(gamma**2)),
                            "rcb_r2": full_nuisance[h][0],
                            "rcb_sigma2": full_nuisance[h][1],
                            "rcb_qhat_endpoint": float(path["qhat"][-1]),
                            "rcb_imbalance": float(
                                np.linalg.norm(Xt.mean(0) - X0.T @ gamma)
                            ),
                            "de_z": float(self.de_cache[p]["z"][jd]),
                            "de_padj": float(self.de_cache[p]["p_adj"][jd]),
                            "de_logFC": float(self.de_cache[p]["lfc"][jd]),
                            "de_significant": bool(
                                self.de_cache[p]["p_adj"][jd]
                                < self.config.PADJ_THRESHOLD
                            ),
                        }
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Full K562 RCB failed: ({p}, {h}): {type(exc).__name__}: {exc}"
                    ) from exc
        self.full_rcb_table = pd.DataFrame(full_rows)
        pd.DataFrame(path_rows).to_csv(
            self.output / "full_k562_adjustment_path.csv", index=False
        )
        pd.DataFrame(shift_rows).to_csv(
            self.output / "full_k562_covariate_shift.csv", index=False
        )
        self.full_rcb_table.to_csv(
            self.output / "full_k562_rcb_analysis.csv", index=False
        )
        self.full_rcb_table[
            [
                "perturbation_p",
                "outcome_h",
                "rcb_effect",
                "rcb_effect_pi_low",
                "rcb_effect_pi_high",
                "rcb_interval_excludes_zero",
                "rcb_selected_lambda",
                "rcb_estimated_risk",
                "de_padj",
                "de_logFC",
            ]
        ].round(4)
        nonzero_perts = (
            self.full_rcb_table[self.full_rcb_table["rcb_interval_excludes_zero"]]
            .groupby("outcome_h")["perturbation_p"]
            .agg(lambda x: ", ".join(x))
        )
        de_perts = (
            self.full_rcb_table[self.full_rcb_table["de_significant"]]
            .groupby("outcome_h")["perturbation_p"]
            .agg(lambda x: ", ".join(x))
        )
        gene_summary = (
            self.full_rcb_table.groupby("outcome_h", sort=False)
            .agg(
                n_rcb_nonzero=("rcb_interval_excludes_zero", "sum"),
                n_de_significant=("de_significant", "sum"),
                max_abs_rcb_effect=("rcb_effect", lambda x: float(np.max(np.abs(x)))),
                mean_rcb_risk=("rcb_estimated_risk", "mean"),
            )
            .reset_index()
        )
        gene_summary["rcb_nonzero_perturbations"] = (
            gene_summary["outcome_h"].map(nonzero_perts).fillna("")
        )
        gene_summary["de_significant_perturbations"] = (
            gene_summary["outcome_h"].map(de_perts).fillna("")
        )
        gene_summary["response_pattern"] = np.select(
            [
                gene_summary["n_rcb_nonzero"] == 0,
                gene_summary["n_rcb_nonzero"] == len(self.selected_p),
            ],
            ["none", "all"],
            default="perturbation-specific",
        )
        gene_summary["de_response_pattern"] = np.select(
            [
                gene_summary["n_de_significant"] == 0,
                gene_summary["n_de_significant"] == len(self.selected_p),
            ],
            ["none", "all"],
            default="perturbation-specific",
        )
        gene_summary = gene_summary.sort_values(
            ["n_rcb_nonzero", "max_abs_rcb_effect"], ascending=[False, False]
        ).reset_index(drop=True)
        perturbation_summary_full = (
            self.full_rcb_table.groupby("perturbation_p", sort=False)
            .agg(
                n_panel_genes=("outcome_h", "size"),
                n_rcb_nonzero=("rcb_interval_excludes_zero", "sum"),
                n_de_significant=("de_significant", "sum"),
                median_selected_lambda=("rcb_selected_lambda", "median"),
                median_risk=("rcb_estimated_risk", "median"),
            )
            .reset_index()
        )
        specific_genes = gene_summary[
            gene_summary["response_pattern"] == "perturbation-specific"
        ]
        gene_summary.to_csv(self.output / "full_k562_gene_specificity.csv", index=False)
        perturbation_summary_full.to_csv(
            self.output / "full_k562_perturbation_summary.csv", index=False
        )
        print(
            "\nPerturbation summary:\n",
            perturbation_summary_full.round(4).to_string(index=False),
        )
        print(
            "\nPerturbation-specific genes:\n",
            specific_genes.round(4).to_string(index=False),
        )
