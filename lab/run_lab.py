"""One-command orchestrator for the Trust-Based Video Management Framework lab.

    python run_lab.py                       # quick run (small dataset, CPU-friendly)
    python run_lab.py --full                # closer to the paper's stated scale (slow on CPU)

Produces lab/results/summary.json aggregating every metric used in the
paper's Validation Testbed section and on the project website.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from config import RESULTS_DIR
from data.synthetic_dataset import generate_dataset
from models.train import train_model
from trust.trust_simulation import simulate, summarize
from viz.generate_charts import main as generate_all_charts


def main():
    parser = argparse.ArgumentParser(description="Run the full trust-based framework validation lab.")
    parser.add_argument("--per-class", type=int, default=120, help="images per class for the synthetic testbed")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-uploads", type=int, default=10000, help="uploads simulated for the trust engine")
    parser.add_argument("--full", action="store_true", help="paper-scale run: 224px, more images, more epochs (slow on CPU)")
    parser.add_argument("--skip-training", action="store_true", help="reuse existing model results, only redo trust sim + charts")
    args = parser.parse_args()

    if args.full:
        args.per_class = max(args.per_class, 1000)
        args.img_size = 224
        args.epochs = max(args.epochs, 10)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if not args.skip_training:
        print(f"== Generating synthetic dataset ({args.per_class}/class) ==")
        generate_dataset(per_class=args.per_class, size=max(args.img_size, 160))

        model_results = {}
        for name in ["vgg16", "resnet50"]:
            print(f"== Training {name} ==")
            res = train_model(name, img_size=args.img_size, epochs=args.epochs, batch_size=args.batch_size)
            model_results[name] = res
            (RESULTS_DIR / f"{name}_results.json").write_text(json.dumps(res, indent=2))
    else:
        model_results = {}
        for name in ["vgg16", "resnet50"]:
            p = RESULTS_DIR / f"{name}_results.json"
            if p.exists():
                model_results[name] = json.loads(p.read_text())

    print(f"== Running trust-scoring simulation ({args.n_uploads} uploads) ==")
    df = simulate(args.n_uploads)
    df.to_csv(RESULTS_DIR / "trust_simulation.csv", index=False)
    trust_summary = summarize(df)
    (RESULTS_DIR / "trust_simulation_summary.json").write_text(json.dumps(trust_summary, indent=2))

    print("== Generating charts ==")
    generate_all_charts()

    summary = {
        "wall_clock_seconds": time.time() - t0,
        "classification": {
            name: {
                "test_accuracy": res["test"]["accuracy"],
                "test_precision_macro": res["test"]["precision_macro"],
                "test_recall_macro": res["test"]["recall_macro"],
                "test_f1_macro": res["test"]["f1_macro"],
                "epochs": res["epochs"],
                "img_size": res["img_size"],
                "n_train": res["n_train"],
                "n_val": res["n_val"],
                "n_test": res["n_test"],
            }
            for name, res in model_results.items()
        },
        "trust_simulation": trust_summary,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nDone in {summary['wall_clock_seconds']:.1f}s. See lab/results/ for all outputs.")


if __name__ == "__main__":
    main()
