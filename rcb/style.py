"""One palette and one typographic hierarchy for every figure in the paper.

The three sections had drifted apart: the same semantic series appeared in
different hex values from one figure to the next, and the font hierarchy was
declared five separate times.  The palette below is the Okabe-Ito colorblind-safe
set Section 5 established, and the method colors are Section 7's assignment,
which is the one the manuscript's real-data figures already use.

The method-color mapping is a promise to the reader: a series keeps its color
across every panel of every figure, so the eye can carry a method from one plot
to the next without re-reading the legend.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = [
    "COLORS",
    "COMPONENT_COLORS",
    "COMPONENT_DISPLAY_LABELS",
    "METHOD_COLORS",
    "configure_style",
    "save_figure",
]

#: Okabe-Ito colorblind-safe base palette.
COLORS = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "gold": "#E69F00",
    "gray": "#6B7280",
    "black": "#111827",
}

COMPONENT_COLORS = {
    "Signal": COLORS["blue"],
    "Residual noise": COLORS["vermillion"],
    "Total": COLORS["green"],
}

# The stored data keeps the internal names; figures use the manuscript's terms.
# "Integrated signal" is the integrated squared conditional bias B_n and
# "Residual-noise variance" is the conditional residual-noise variance V_n
# (S5-6: these replace the earlier "Bias"/"Variance" labels).
COMPONENT_DISPLAY_LABELS = {
    "Signal": "Integrated signal",
    "Residual noise": "Residual-noise variance",
    "Total": "Total risk",
}

#: Method colors, fixed across panels.  Double ridge in warm tones, the proposed
#: estimator in cool tones, references and bases in neutral gray.
METHOD_COLORS = {
    "double_outcome": "#D55E4A",
    "double_imbalance": "#7568A9",
    "double_riesz": "#D39B24",
    "ours_tuned": "#23877F",
    "ours_uniform": "#315F8C",
    "reference": "#7B8087",
    "base": "#A4A8AD",
    "benchmark": "#2F3136",
    "entropy": "#D39B24",
    "arb": "#B86B1D",
    "grid": "#E4E7EA",
    "spine": "#AEB4BA",
    "error": "#41454A",
}


def configure_style() -> None:
    """Use one font family and one size hierarchy for every figure."""
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12.5,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.4,
            "lines.markersize": 6.5,
            "savefig.dpi": 240,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.22,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig, stem: str, figure_dir, *, aliases=()) -> None:
    """Write one figure as PNG and PDF, enforcing the typography rules.

    Any suptitle is removed: the manuscript's captions carry that text, and a
    title baked into the image would duplicate it. ``aliases`` writes extra
    copies under manuscript-facing filenames, which is how the paper's figure
    names stay stable while the generated stems stay descriptive.
    """
    if fig._suptitle is not None:
        fig._suptitle.remove()
        fig._suptitle = None
    for ax in fig.axes:
        ax.title.set_size(14)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        ax.tick_params(axis="both", which="both", labelsize=12)
        if ax.legend_ is not None:
            for text_item in ax.legend_.get_texts():
                text_item.set_fontsize(12.5)
    for legend in fig.legends:
        for text_item in legend.get_texts():
            text_item.set_fontsize(12.5)
    fig.savefig(figure_dir / f"{stem}.png")
    fig.savefig(figure_dir / f"{stem}.pdf")
    for alias in aliases:
        fig.savefig(figure_dir / alias)
    plt.close(fig)
