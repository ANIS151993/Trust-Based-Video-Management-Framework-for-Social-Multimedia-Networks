# 🧪 Trust-Based Framework — Validation Lab

A runnable, reproducible testbed that implements the ML and trust-scoring
pipeline described in the paper, so anyone can validate the methodology
without needing a sensitive or license-restricted weapon-image dataset.

> **What this is / is not.** This lab trains VGG16 and ResNet50 (transfer
> learning, ImageNet weights, frozen base, the same custom head and
> hyperparameters described in the paper) against a **synthetic
> silhouette-proxy dataset** (`data/synthetic_dataset.py`) — procedurally
> generated line-art shapes standing in for the five weapon categories. It
> also runs the dynamic trust-scoring simulation independently, over
> synthetic uploads. It is a methodology-validation testbed for the pipeline
> itself (data loading, transfer-learning setup, training loop, evaluation,
> trust routing), **not** a reproduction of the original paper's reported
> field results, which were obtained on a real, non-public image corpus.

## Quick start

```bash
cd lab
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
python run_lab.py                                       # ~120 imgs/class, 128px, 8 epochs — a few minutes on CPU
```

For a closer-to-paper-scale run (slower, ideally with a GPU):

```bash
python run_lab.py --full   # 224px input, 1000+ imgs/class, 10 epochs
```

Everything lands in `lab/results/`:

```
results/
├── vgg16_results.json / resnet50_results.json   # per-epoch history + test metrics
├── trust_simulation.csv                          # raw simulated uploads
├── trust_simulation_summary.json                 # routing distribution, workload stats
├── summary.json                                  # everything, aggregated
└── figures/
    ├── distplot_trust_scores.png
    ├── pie_content_routing.png
    ├── violin_trust_factors.png
    ├── heatmap_trust_correlation.png
    ├── pairplot_trust_factors.png
    ├── jointplot_quality_vs_trust.png
    ├── training_curves.png
    ├── confusion_matrix_vgg16.png / confusion_matrix_resnet50.png
    └── comparison_bar_test_accuracy.png
```

## Layout

| File | Purpose |
|---|---|
| `config.py` | Shared constants: class labels, trust weights/thresholds, color palette |
| `data/synthetic_dataset.py` | Procedural 5-class silhouette-proxy image generator |
| `models/train.py` | VGG16 / ResNet50 transfer learning, training loop, evaluation |
| `trust/trust_simulation.py` | Dynamic trust-score formula + upload-routing simulation |
| `viz/generate_charts.py` | All chart figures (dist/pie/violin/heatmap/pairplot/jointplot + training curves) |
| `run_lab.py` | Orchestrates the full pipeline end-to-end |

## Run it in your browser (no local setup)

Open `Trust_Based_Lab.ipynb` in Google Colab or Jupyter and run all cells —
useful for visitors who just want to see the pipeline work without cloning
and installing locally.

## Extending to a real dataset

Point `models/train.py`'s `stratified_split()` at your own `data_dir`
containing `<class_name>/*.png` (or `.jpg`) subfolders matching
`config.CLASSES`, and the rest of the pipeline (training, evaluation, trust
simulation, charts) works unchanged.
