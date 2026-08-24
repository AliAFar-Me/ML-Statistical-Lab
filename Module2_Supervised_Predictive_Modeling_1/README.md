# Module 2: Supervised Predictive Modeling

**Comparing Logistic Regression, SVM (RBF), and Random Forest to demonstrate the bias-variance tradeoff, algorithmic complexity, and appropriate evaluation-metric selection.**

---

## 1. Purpose

This module is a hands-on companion to a standard ML theory unit: *why does model choice matter beyond raw accuracy?* Three classifiers with fundamentally different inductive biases are trained on the same task and compared along three axes:

1. **Decision boundary geometry** — a direct visual read of high bias (linear) vs. high variance (blocky/jagged).
2. **Preprocessing requirements** — why distance-based models need scaling and tree-based models don't, treated as a first-class design decision rather than boilerplate.
3. **Threshold-independent evaluation** — ROC and Precision-Recall curves instead of a single accuracy number, since accuracy can hide how a model fails.

---

## 2. Dataset & Provenance

**Source dataset:** `USA_Housing.csv`, the standard Kaggle housing-price regression dataset. Columns:

| Column | Type | Notes |
|---|---|---|
| `Avg. Area Income` | float | |
| `Avg. Area House Age` | float | |
| `Avg. Area Number of Rooms` | float | |
| `Avg. Area Number of Bedrooms` | float | |
| `Area Population` | float | |
| `Price` | float | Original regression target |
| `Address` | string | Free-text, high-cardinality — dropped |

**⚠️ Data stand-in notice:** The dataset originally shipped for this module was not available in the build/validation sandbox (no network access to Kaggle). `data/USA_Housing.csv` in this package is a **synthetic stand-in**, generated to match the real dataset's schema, column names, and approximate distributional shape (income, house age, room counts, population drawn from realistic ranges; `Price` constructed as a linear combination of the features plus Gaussian noise, so the classification task is learnable but not trivial).

This stand-in was used **only to validate that the notebook executes correctly end-to-end** — every cell runs without errors and without warnings, and produces sane, interpretable output (see Section 5). It was **not** used to draw any substantive conclusion about which model "wins" — see the caveat in Section 5.

**To use the real dataset:** download `USA_Housing.csv` from Kaggle and replace the file at `data/USA_Housing.csv`, keeping the same filename and column headers. No code changes are required — the notebook's `pd.read_csv("USA_Housing.csv")` call and all downstream column references (`Price`, `Address`, `Avg. Area Income`, `Area Population`) match the standard schema exactly.

---

## 3. Feature Engineering: Regression → Classification

The dataset was originally built for regression. It is converted to a binary classification task by thresholding `Price` at its own median:

```
Is_Premium = 1 if Price > median(Price) else 0
```

**Why the median specifically:** it guarantees an almost exactly balanced 50/50 class split by construction, which removes class imbalance as a confound. Any performance differences observed between the three models can then be attributed to genuine model behavior rather than to one model handling skewed priors better than another. `Price` and `Address` are dropped post-engineering — `Price` because it is now direct label leakage, `Address` because it's unstructured text outside this module's scope.

---

## 4. Pipeline Design Decisions

### 4.1 Preprocessing divergence
Two parallel preprocessing paths are built from the same train/test split:

- **Scaled path** (`StandardScaler`) → feeds Logistic Regression and SVM (RBF), both of which are distance- or gradient-sensitive and require features on a common scale to avoid large-magnitude features dominating the loss surface or the RBF kernel's distance calculation.
- **Raw path** (unscaled) → feeds Random Forest, whose split-based decisions are invariant to monotonic feature transformations, so scaling is mathematically inert for it.

The scaler is `fit` **only on the training set** and applied via `transform` to the test set, avoiding any leakage of test-set statistics into preprocessing.

### 4.2 Decision boundary visualization is a separate, lightweight fit
Because the real feature space is 5-dimensional, Section 4 of the notebook trains **separate, 2-feature-only versions** of all three models (using `Avg. Area Income` and `Area Population`) purely to render a 2D contour plot. These 2D models are explicitly *not* the same fitted objects used for the Section 5 metric evaluation — the notebook comments this distinction directly in the code, since silently conflating the two would misrepresent what the visualization is actually showing.

### 4.3 Evaluation uses probability scores, not hard labels
Both ROC and PR curves are built from `predict_proba` output (all three models support it; SVM was initialized with `probability=True`), so the comparison is threshold-independent rather than tied to the default 0.5 cutoff.

---

## 5. Validation Workflow & Findings

Following the standard **build → validate → review → refactor → re-validate** cycle used across this portfolio:

1. **Built** the notebook programmatically via `nbformat` v4 JSON for reproducibility.
2. **Validated** by generating the synthetic stand-in dataset and executing every code cell end-to-end in a live sandbox — not just static code review.
3. **Found a real defect during review:** the decision-boundary function fit `StandardScaler` on a pandas DataFrame (which internally records feature names) but later transformed a raw numpy meshgrid array with no names attached, triggering an `sklearn UserWarning` on every run. Numerically harmless, but not acceptable output for a portfolio deliverable.
4. **Refactored:** the scaler in `plot_decision_boundaries()` now fits on `.values` throughout, keeping the function consistently numpy-typed end-to-end.
5. **Re-validated** with warnings promoted to errors (`python -W error::UserWarning`) — confirmed a fully clean run, zero errors, zero warnings.

**Illustrative results on the synthetic stand-in data** (5,000 rows, near-linear price construction):

| Model | Test Accuracy | Train Accuracy | Train–Test Gap | ROC AUC |
|---|---|---|---|---|
| Logistic Regression | 0.898 | 0.907 | 0.009 | 0.969 |
| SVM (RBF Kernel) | 0.896 | 0.914 | 0.018 | 0.967 |
| Random Forest | 0.878 | 1.000 | 0.122 | 0.954 |

The train–test gap ordering (RF ≫ SVM > LR) is the expected numerical signature of the bias-variance tradeoff and matches the qualitative decision-boundary shapes (linear vs. curved vs. blocky).

**Caveat:** because the synthetic price-generation function is close to linear in the input features, Logistic Regression and SVM edge out Random Forest on this particular stand-in dataset. This is an artifact of the synthetic data generator, not a property of the notebook's logic — the code makes no assumption about which model should win, and results on the real Kaggle dataset may differ. The notebook's job is to teach the comparison methodology, not to pre-determine its outcome.

---

## 6. File Structure

```
Module2_Supervised_Predictive_Modeling/
├── README.md                                   <- this file
├── Module2_Supervised_Predictive_Modeling.ipynb <- main deliverable
└── data/
    └── USA_Housing.csv                          <- synthetic stand-in (see Section 2 for real-data swap-in)
```

## 7. Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## 8. How to Run

1. (Optional but recommended) Replace `data/USA_Housing.csv` with the real Kaggle dataset, keeping the filename and headers identical.
2. Open `Module2_Supervised_Predictive_Modeling.ipynb` in Jupyter.
3. Run all cells top to bottom — no manual configuration required.
