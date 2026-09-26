# ML & Data Science Portfolio

A series of production-quality Jupyter notebook modules covering the core
machine learning and data science workflow — exploratory analysis,
supervised classification, and unsupervised learning — built to a
professional engineering bar rather than tutorial-quality "it runs once"
code.

Every module is validated end-to-end in a sandboxed environment before
delivery, put through a formal principal-engineer-style code review, and
shipped with a README documenting purpose, data provenance, design
decisions, and the defects found and fixed along the way. Bugs caught
during validation aren't swept under the rug — they're treated as evidence
of engineering rigor and documented as part of the deliverable.

## Repository Structure

```
.
├── README.md                              # This file
├── 01_exploratory_data_analysis/
│   ├── README.md
│   ├── data/
│   │   ├── sf_salaries.csv
│   │   └── ecommerce_purchases.csv
│   ├── pandas_data_wrangling.ipynb
│   └── interactive_visualizations.ipynb
├── 02_supervised_predictive_modeling/
│   ├── README.md
│   ├── data/
│   │   └── USA_Housing.csv
│   └── classifier_comparison.ipynb
└── 03_unsupervised_learning/
    ├── README.md
    ├── data/
    │   └── gene_expression_rna_seq.csv
    └── dimensionality_reduction_clustering.ipynb
```

Each module folder is self-contained: notebook(s), dataset (real or a
schema-accurate synthetic stand-in), and its own README with full technical
detail. This top-level README is the map; module READMEs are the territory.

## Modules

### 01 — Exploratory Data Analysis
Vectorized pandas pipelines over two datasets (`sf_salaries`,
`ecommerce_purchases`): method-chained transforms via `.pipe()`/`.assign()`,
memory-footprint profiling with dtype downcasting, group-conditional
imputation, and interactive Seaborn/Plotly diagnostics.

Refactored under a full code review: `pathlib` migration, a pandas 3.x
`StringDtype` compatibility fix, nullable-integer (`Int8`–`Int64`) handling
for NaN-containing numeric columns, a dual-constraint categorical
cardinality check, a NaN-safe `.query()` filter, and a `_downsample()` guard
capping interactive plots at 10,000 rows.

### 02 — Supervised Predictive Modeling
Three-classifier comparison — Logistic Regression, SVM (RBF kernel), and
Random Forest — on `USA_Housing.csv`, with a binary target derived via
median thresholding. Covers divergent preprocessing requirements per model
family, decision-boundary visualization, and ROC/PR curve evaluation.

A `StandardScaler` fit/transform type mismatch was found during validation,
fixed, and re-validated clean.

### 03 — Unsupervised Learning & Dimensionality Reduction
A PCA → UMAP → K-Means → DBSCAN pipeline on a gene-expression RNA-Seq
dataset (TCGA PAN-CAN schema) in the `p ≫ n` regime — more features than
samples. Includes silhouette-optimized K-Means, a percentile-sweep heuristic
for DBSCAN's `eps`, and a blind comparison against ground-truth labels via
ARI/NMI, with labels deliberately held out until the final evaluation step.

`umap-learn` was unavailable in the build sandbox (no network access); a
stub stands in, with the real-data/real-library swap-in explicitly
documented in the module README rather than silently glossed over.

## Engineering Principles

- **Validation is non-negotiable.** Every notebook is executed end-to-end
  in a sandbox before delivery; warnings are promoted to errors during
  re-validation passes to surface issues that would otherwise pass silently.
- **Defects are documentation, not embarrassments.** Bugs caught during
  validation are fixed and written up in the relevant README as part of the
  engineering trail.
- **Transparency over workarounds.** When a sandbox constraint blocks real
  data or a real library, that's disclosed plainly, with concrete
  instructions for swapping in the real thing.
- **Design decisions are justified, not just implemented.** Notebooks
  explain *why* a choice was made (e.g., clustering in PCA space rather than
  UMAP space, holding out labels until the final section) in Markdown cells
  alongside the code.
- **Notebooks are built programmatically**, via Python scripts that emit
  valid `nbformat` v4 JSON, rather than authored by hand cell-by-cell.

## Tech Stack

Python · Jupyter (`nbformat` v4) · pandas (3.x-aware) · NumPy · scikit-learn
· Seaborn · Matplotlib · Plotly · UMAP (`umap-learn`)

## Status & Roadmap

Three modules complete and packaged as zip deliverables. Further modules are
planned as the series continues.

GitHub integration is a known open item: no GitHub MCP connector is
available in this environment, so pushing these modules to a repository
currently requires a manual download-and-commit step, or Claude Code
(desktop), which has direct git access.
