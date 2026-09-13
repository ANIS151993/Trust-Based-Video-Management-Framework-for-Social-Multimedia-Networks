"""Dynamic trust-scoring simulation.

Implements the multi-factor trust formula from the paper (Section III-B):

    T_u = w1*H + w2*Q + w3*F + w4*C - w5*V

and simulates a population of synthetic uploads/users to study the resulting
trust-score distribution and content-routing outcomes (auto-publish /
expedited review / auto-reject). This is an independent statistical
validation testbed -- it demonstrates the routing mechanism and produces
real, reproducible numbers from this simulation; it does not reproduce the
original paper's reported field-deployment figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RANDOM_SEED, RESULTS_DIR, TRUST_THRESHOLDS, TRUST_WEIGHTS  # noqa: E402


def _beta_scaled(rng, a, b, n):
    return rng.beta(a, b, n) * 100.0


def simulate(n_uploads: int = 10000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Historical reliability: an established platform population skews
    # strongly reliable (most users have a clean, long-standing upload history).
    H = _beta_scaled(rng, 12, 2, n_uploads)
    # Content quality: correlated with H but with its own noise + a small
    # fraction of uploads flagged by the weapon classifier (lower quality score).
    flagged = rng.random(n_uploads) < 0.05
    Q_base = 0.6 * H + 0.4 * _beta_scaled(rng, 10, 2, n_uploads)
    Q = np.where(flagged, Q_base * rng.uniform(0.1, 0.4, n_uploads), Q_base)
    # Community feedback: noisier, weakly correlated with quality.
    F = np.clip(0.5 * Q + 0.5 * _beta_scaled(rng, 9, 3, n_uploads), 0, 100)
    # Behavioral consistency.
    C = _beta_scaled(rng, 10, 2, n_uploads)
    # Violation penalty: most users have ~0, a minority have real penalties.
    V = np.where(rng.random(n_uploads) < 0.06, _beta_scaled(rng, 2, 5, n_uploads), rng.uniform(0, 5, n_uploads))

    w = TRUST_WEIGHTS
    T = w["w1"] * H + w["w2"] * Q + w["w3"] * F + w["w4"] * C - w["w5"] * V
    T = np.clip(T, 0, 100)

    routing = np.where(
        T >= TRUST_THRESHOLDS["high"],
        "auto_publish",
        np.where(T >= TRUST_THRESHOLDS["medium"], "expedited_review", "auto_reject"),
    )

    df = pd.DataFrame(
        {
            "H": H,
            "Q": Q,
            "F": F,
            "C": C,
            "V": V,
            "T": T,
            "flagged_by_classifier": flagged,
            "routing": routing,
        }
    )
    return df


def summarize(df: pd.DataFrame) -> dict:
    counts = df["routing"].value_counts(normalize=True) * 100
    baseline_review_rate = 100.0  # naive baseline: every upload gets full manual review
    actual_review_rate = counts.get("expedited_review", 0.0)
    workload_reduction = 100.0 * (1 - actual_review_rate / baseline_review_rate)
    return {
        "n_uploads": len(df),
        "routing_distribution_pct": counts.round(2).to_dict(),
        "mean_trust_score": float(df["T"].mean()),
        "median_trust_score": float(df["T"].median()),
        "std_trust_score": float(df["T"].std()),
        "flagged_rate_pct": float(df["flagged_by_classifier"].mean() * 100),
        "manual_review_workload_reduction_pct": round(workload_reduction, 2),
        "trust_factor_correlations": df[["H", "Q", "F", "C", "V", "T"]].corr().round(3).to_dict(),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run the trust-scoring simulation.")
    parser.add_argument("--n-uploads", type=int, default=10000)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = simulate(args.n_uploads)
    df.to_csv(RESULTS_DIR / "trust_simulation.csv", index=False)
    summary = summarize(df)
    with open(RESULTS_DIR / "trust_simulation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
