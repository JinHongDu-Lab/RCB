"""Methods the proposed estimator is compared against.

Four groups, kept in separate modules because they answer different questions:

``weights``
    Competing balancing-weight families -- uniform, propensity-based, entropy,
    stable balancing, CBPS, l2, and the Bruns-Smith ridge base.  Any of these
    can also serve as the base the proposed method augments, which is the point
    of the comparison: the augmentation is base-agnostic.

``double_ridge``
    The Bruns-Smith augmented l2 estimator and the three cross-validation rules
    it is tuned by.  The same weight path as the proposed method, selected
    differently.

``arb``
    Approximate residual balancing, a two-stage weights-plus-outcome-model
    comparator that does not share the path at all.

``gcv``
    Source-distribution generalized cross-validation: the baseline that ignores
    covariate shift, included to show what ignoring it costs.
"""

__all__ = ["arb", "double_ridge", "gcv", "weights"]
