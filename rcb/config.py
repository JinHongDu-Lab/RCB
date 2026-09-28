"""Seeding and grid construction shared across the experiments.

The per-experiment configuration itself stays with its study: the simulation's
dimension grid, the evaluation's overlap settings and LaLonde's replication counts
are statements about those studies, not about the method, and pulling their
values into shared code would make the package look like it had opinions it does
not have.

What *is* shared is how randomness is derived for a parallel sweep:

:func:`replication_rng`
    A generator derived from the replication's coordinates -- repetition index,
    aspect ratio, shift magnitude -- through a ``SeedSequence``.  Each cell's
    stream depends only on where it sits in the design, not on when it happened
    to run, which the evaluation study needs because it evaluates its cells
    under ``joblib``.  The simulation study instead consumes one sequential
    stream from its own seed, in its own module.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "replication_rng",
]

def replication_rng(base_seed: int, *coordinates: int) -> np.random.Generator:
    """Generator addressed by a replication's position in the design.

    The stream depends only on ``coordinates``, so a cell draws the same data
    whether the sweep runs in one process or many, and in any order.  Pass
    integers; scale continuous design parameters to integers at the call site so
    the scaling is visible there.
    """
    return np.random.default_rng(
        np.random.SeedSequence([int(base_seed), *(int(c) for c in coordinates)])
    )
