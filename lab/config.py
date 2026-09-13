"""Shared configuration and the validated color palette for the lab.

The five categorical hues below (blue, orange, aqua, yellow, magenta) are the
first five slots of a colorblind-validated categorical order (fixed order,
never cycled) -- see docs/assets/css/theme.css for the same values used on
the website, so charts read consistently across the paper, the lab output,
and the site.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "generated"
RESULTS_DIR = ROOT / "results"

CLASSES = ["ak47", "gun", "knife", "sickle", "sword"]

# Categorical palette, fixed order (slot 1..5)
CATEGORICAL = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
}
CLASS_COLORS = dict(zip(CLASSES, CATEGORICAL.values()))

STATUS = {
    "good": "#0ca30c",      # auto-publish
    "warning": "#fab219",   # expedited review
    "critical": "#d03b3b",  # auto-reject
}

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]
DIVERGING = {"cold": "#2a78d6", "warm": "#e34948", "mid": "#f0efec"}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

# Trust scoring weights, matching the formula in the paper (Section III-B)
TRUST_WEIGHTS = {"w1": 0.30, "w2": 0.25, "w3": 0.20, "w4": 0.15, "w5": 0.10}
TRUST_THRESHOLDS = {"high": 75, "medium": 50}

RANDOM_SEED = 42
