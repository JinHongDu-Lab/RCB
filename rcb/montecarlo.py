"""Monte Carlo summaries, resampling, and their uncertainty.

Two distinctions here are easy to lose and matter for how the numbers read.

**Standard error of a mean versus conditional sampling standard deviation.**
:func:`mean_with_two_se` divides by the square root of the replication count and
reports the precision of the Monte Carlo average.  :func:`single_design_summary`
deliberately does not divide: it reports the spread of a single plug-in estimate
around its own conditional mean on a fixed design, which is a property of the
estimator rather than of how long the simulation ran.  Both appear in the
manuscript and they answer different questions.

**Design-level clustering.**  When many outcome draws share one covariate
design, treating the draws as independent understates uncertainty.  Summaries
of such sweeps must aggregate per design first; the section drivers do this
with their own column names, which is why no generic version lives here.

Section-specific reporting -- anything keyed to particular column names or
replication counts -- stays with its section.  This module holds only what is
genuinely design-independent.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "single_design_summary",
    "mean_with_two_se",
]


def mean_with_two_se(
    frame: pd.DataFrame, value: str, groups: list[str]
) -> pd.DataFrame:
    """Group mean with its standard error and a two-standard-error band."""
    return (
        frame.groupby(groups, as_index=False)[value]
        .agg(mean="mean", sd="std", n="count")
        .assign(se=lambda x: x["sd"] / np.sqrt(x["n"]), two_se=lambda x: 2.0 * x["se"])
    )


def single_design_summary(
    frame: pd.DataFrame,
    groups: list[str],
    columns: list[str],
    *,
    draw_column: str = "draw",
    conditional_prefix: str = "estimate_",
) -> pd.DataFrame:
    """Group means, plus a conditional sampling SD for the estimate columns.

    Columns whose name starts with ``conditional_prefix`` also get a
    ``_conditional_se``: the standard deviation across outcome draws on one
    fixed design, *not* divided by the square root of the draw count.  That is
    intentional -- it is the sampling spread of a single estimate, not the
    precision of the simulation average.
    """
    rows = []
    for keys, g in frame.groupby(groups, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row["outcome_repetitions"] = g[draw_column].nunique()
        for col in columns:
            values = g[col].to_numpy(float)
            row[col] = values.mean()
            if col.startswith(conditional_prefix):
                row[f"{col}_conditional_se"] = values.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


