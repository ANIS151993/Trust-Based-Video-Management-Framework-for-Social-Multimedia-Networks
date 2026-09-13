"""Shared matplotlib/seaborn styling using the validated palette (config.py)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    CLASS_COLORS,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
)

CATEGORICAL_ORDER = list(CLASS_COLORS.values())


def apply_style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRIDLINE,
            "axes.labelcolor": INK_SECONDARY,
            "text.color": INK_PRIMARY,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "grid.color": GRIDLINE,
            "axes.grid": True,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 160,
        }
    )
    sns.set_style("white")
    sns.set_palette(CATEGORICAL_ORDER)
