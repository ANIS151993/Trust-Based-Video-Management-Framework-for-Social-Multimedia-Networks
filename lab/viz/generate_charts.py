"""Generates all chart figures for the paper, README, and website from the
lab's own output (lab/results/*.json, lab/results/trust_simulation.csv).

Produces: a distribution plot, a pie chart, a violin plot, a correlation
heatmap, a pair plot, and a joint plot (as requested), plus training curves,
confusion-matrix heatmaps, and a VGG16-vs-ResNet50 comparison bar chart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CATEGORICAL, CLASSES, CLASS_COLORS, DIVERGING, RESULTS_DIR, STATUS  # noqa: E402
from viz.style import apply_style  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def chart_distplot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["T"], bins=40, kde=True, color=list(CLASS_COLORS.values())[0], ax=ax)
    ax.axvline(75, color=STATUS["good"], linestyle="--", linewidth=1.5, label="High-trust threshold (75)")
    ax.axvline(50, color=STATUS["warning"], linestyle="--", linewidth=1.5, label="Medium-trust threshold (50)")
    ax.set_title("Distribution of Simulated User Trust Scores")
    ax.set_xlabel("Trust score (Tᵤ)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, "distplot_trust_scores.png")


def chart_pie(df: pd.DataFrame):
    counts = df["routing"].value_counts()
    labels_map = {
        "auto_publish": "Auto-Publish (High Trust)",
        "expedited_review": "Expedited Review (Medium Trust)",
        "auto_reject": "Auto-Reject (Low Trust)",
    }
    colors_map = {"auto_publish": STATUS["good"], "expedited_review": STATUS["warning"], "auto_reject": STATUS["critical"]}
    order = [k for k in labels_map if k in counts.index]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.pie(
        [counts[k] for k in order],
        labels=[labels_map[k] for k in order],
        colors=[colors_map[k] for k in order],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"linewidth": 2, "edgecolor": "#fcfcfb"},
        textprops={"fontsize": 9},
    )
    ax.set_title("Content Routing Decisions (Simulated Uploads)")
    _save(fig, "pie_content_routing.png")


def chart_violin(df: pd.DataFrame):
    factors = ["H", "Q", "F", "C", "V"]
    names = {"H": "Historical\nReliability", "Q": "Content\nQuality", "F": "Community\nFeedback", "C": "Consistency", "V": "Violation\nPenalty"}
    long = df[factors].melt(var_name="factor", value_name="score")
    long["factor"] = long["factor"].map(names)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = list(CLASS_COLORS.values())
    sns.violinplot(data=long, x="factor", y="score", hue="factor", palette=palette, ax=ax, legend=False, cut=0)
    ax.set_title("Distribution of Trust-Score Components")
    ax.set_xlabel("")
    ax.set_ylabel("Score (0–100)")
    _save(fig, "violin_trust_factors.png")


def chart_heatmap_correlation(df: pd.DataFrame):
    factors = ["H", "Q", "F", "C", "V", "T"]
    corr = df[factors].corr()
    cmap = sns.diverging_palette(220, 10, s=70, l=50, sep=10, as_cmap=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap=cmap, center=0, vmin=-1, vmax=1,
        square=True, linewidths=1, linecolor="#fcfcfb", cbar_kws={"shrink": 0.8}, ax=ax,
    )
    ax.set_title("Correlation Among Trust-Score Factors")
    _save(fig, "heatmap_trust_correlation.png")


def chart_pairplot(df: pd.DataFrame):
    factors = ["H", "Q", "F", "C", "T"]
    sample = df[factors].sample(min(800, len(df)), random_state=42)
    g = sns.pairplot(
        sample, diag_kind="kde", plot_kws={"alpha": 0.35, "s": 12, "color": list(CLASS_COLORS.values())[0]},
        diag_kws={"color": list(CLASS_COLORS.values())[1]},
    )
    g.fig.suptitle("Pairwise Relationships Among Trust Factors", y=1.02)
    g.fig.set_dpi(140)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "pairplot_trust_factors.png"
    g.savefig(path, bbox_inches="tight")
    plt.close(g.fig)
    print(f"saved {path}")


def chart_jointplot(df: pd.DataFrame):
    sample = df.sample(min(1500, len(df)), random_state=42)
    g = sns.jointplot(
        data=sample, x="Q", y="T", kind="reg",
        color=list(CLASS_COLORS.values())[0],
        scatter_kws={"alpha": 0.3, "s": 14},
        line_kws={"color": list(CLASS_COLORS.values())[1]},
        height=5.5,
    )
    g.set_axis_labels("Content Quality (Q)", "Trust Score (Tᵤ)")
    g.fig.suptitle("Content Quality vs. Trust Score", y=1.02)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "jointplot_quality_vs_trust.png"
    g.savefig(path, bbox_inches="tight")
    plt.close(g.fig)
    print(f"saved {path}")


def chart_training_curves(results: dict):
    if not results:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, res in results.items():
        color = CATEGORICAL["blue"] if name == "vgg16" else CATEGORICAL["orange"]
        hist = res["history"]
        epochs = [h["epoch"] for h in hist]
        axes[0].plot(epochs, [h["train_acc"] for h in hist], color=color, label=f"{name} train")
        axes[0].plot(epochs, [h["val_acc"] for h in hist], color=color, linestyle="--", label=f"{name} val")
        axes[1].plot(epochs, [h["train_loss"] for h in hist], color=color, label=f"{name} train")
        axes[1].plot(epochs, [h["val_loss"] for h in hist], color=color, linestyle="--", label=f"{name} val")
    axes[0].set_title("Accuracy per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_title("Loss per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, "training_curves.png")


def chart_confusion_matrices(results: dict):
    for name, res in results.items():
        cm = np.array(res["test"]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4.3))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES,
            cbar=False, linewidths=1, linecolor="#fcfcfb", ax=ax,
        )
        ax.set_title(f"{name.upper()} Confusion Matrix (Test Set)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        _save(fig, f"confusion_matrix_{name}.png")


def chart_comparison_bar(results: dict):
    if not results:
        return
    names = list(results.keys())
    accs = [results[n]["test"]["accuracy"] * 100 for n in names]
    colors = [CATEGORICAL["blue"] if n == "vgg16" else CATEGORICAL["orange"] for n in names]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(names, accs, color=colors, width=0.5)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 1, f"{a:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("VGG16 vs. ResNet50 — Validation Testbed")
    _save(fig, "comparison_bar_test_accuracy.png")


def main():
    apply_style()
    csv_path = RESULTS_DIR / "trust_simulation.csv"
    if not csv_path.exists():
        raise SystemExit("Run trust/trust_simulation.py first.")
    df = pd.read_csv(csv_path)

    chart_distplot(df)
    chart_pie(df)
    chart_violin(df)
    chart_heatmap_correlation(df)
    chart_pairplot(df)
    chart_jointplot(df)

    results = {}
    for name in ["vgg16", "resnet50"]:
        p = RESULTS_DIR / f"{name}_results.json"
        if p.exists():
            results[name] = json.loads(p.read_text())
    if results:
        chart_training_curves(results)
        chart_confusion_matrices(results)
        chart_comparison_bar(results)
    else:
        print("No model results found yet -- skipping training/confusion/comparison charts.")


if __name__ == "__main__":
    main()
