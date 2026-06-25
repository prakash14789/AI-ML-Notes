# 31. Lecture notes - Dim-Red & Anomaly Detect - Dr. Surya Prakash - 29 Dec 2025

## [In-class resource](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/720e1f1b-afef-4276-beff-fe79be704baa/gmbnBov2oePpEqJu.zip)

# Dimensionality Reduction & Anomaly Detection: Lecture Notes

**Prerequisites:** Standardization/normalization, matrix intuition (variance, covariance), NumPy/Pandas, and comfort with train/validation/test splits.

**What you'll be able to do:**

- Decide when to reduce dimensions and pick PCA vs manifold methods
- Run PCA end-to-end and justify component counts with explained variance
- Build anomaly detectors (Isolation Forest, LOF, autoencoders) and pick thresholds with the right metrics
- Debug common issues: scaling mistakes, brittle embeddings, and bad contamination settings

---

## 1) Why This Matters

High-dimensional tables are noisy, slow to train on, and hard to visualize. Dimensionality reduction compresses features while keeping structure, speeding models and clarifying plots. Anomaly detection solves the "needle in a haystack" problem—fraudulent payments, failing machines, or malicious logins that occur in fractions of a percent.

**Core definitions**

- **Dimensionality reduction:** Transforming features into a lower-dimensional representation that preserves most information.
- **Anomaly detection:** Identifying observations that deviate strongly from the normal data distribution.

[Dimensionality Reduction Pipeline]

---

## 2) Data Prep for Dim-Red and Anomalies

- **Impute first:** PCA/manifold methods assume complete numeric data. Median imputation is robust for skewed features.
- **Scale consistently:** Use `StandardScaler` for PCA, LOF, and Isolation Forest; large-range features otherwise hijack components and distances.
- **Remove leakage:** Drop features that contain the target or post-event info; they distort components and anomaly scores.
- **Stratify splits:** Even in semi-supervised settings, keep a holdout set to sanity-check visuals and anomaly ranks.

## 3) PCA in Practice

**What PCA Does (Simple Explanation):**

Imagine you have 50 columns describing each transaction. Many columns are correlated—high income often means high spending. PCA finds the "directions" in your data that capture the most variation, then keeps only the top few directions.

Think of it like taking a photo of a 3D object. You choose the angle (direction) that shows the most information. PCA automatically finds the best angles for your high-dimensional data.

**Key Terms:**

- **Principal Component (PC):** A new axis that's a weighted combination of original features
- **Explained Variance:** How much of the data's variation this component captures
- **Cumulative Variance:** Total variance captured by keeping the first k components

**Workflow**

1. Standardize features (crucial—otherwise large-scale features dominate).
2. Compute covariance matrix or use SVD.
3. Sort eigenvectors by eigenvalues (largest = most variance).
4. Project data onto the top components.
5. Check cumulative explained variance—stop when it hits ~90-95%.

**Mini case (transactions with 10 numeric features)**

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_s = StandardScaler().fit_transform(X_train)
pca = PCA(n_components=0.9, random_state=42)
X_p = pca.fit_transform(X_s)
print(pca.explained_variance_ratio_.cumsum()[:8])

```

If cumulative variance crosses 0.9 at component 7, keep 7 components; the rest likely capture noise.

**Where it helps:** Wide, correlated numeric columns; decorrelating inputs for linear or distance-based models.

**Cautions:** Mixed features hurt interpretability, and linear components miss curved manifolds.

---

## 4) Evaluating Rare Events

1. **Metrics:** Precision, recall, F1, PR AUC. ROC AUC can look good even when precision is poor.
2. **Thresholds:** Use precision-recall curves, expected alert volume, or cost ratios.
3. **Contamination:** Start with a plausible anomaly rate (e.g., 0.002 for 0.2% fraud); tune on a labeled slice if possible.
4. **Inspect top alerts:** Manually review highest scores; adjust scaling/features if results look random.
5. **Stability:** Re-run with different seeds/bootstraps; big swings mean brittle pipelines.

**Metric helper**

```python
from sklearn.metrics import average_precision_score
scores = model.decision_function(X_val)
ap = average_precision_score(y_val, scores)

```

---

## 5) Worked Example: Credit Card Fraud

**Dataset:** Kaggle Credit Card Fraud Detection ([https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)), ~0.17% fraud, numeric features.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr); X_val_s = scaler.transform(X_val)

pca = PCA(n_components=0.9, random_state=42)
X_tr_p = pca.fit_transform(X_tr_s); X_val_p = pca.transform(X_val_s)

iso = IsolationForest(n_estimators=300, max_samples=256, contamination=0.002, random_state=42)
iso.fit(X_tr_p)
val_scores = -iso.decision_function(X_val_p)  # higher = more anomalous
ap = average_precision_score(y_val, val_scores)

```

Interpretation: Plot precision-recall, pick a threshold that fits alert budgets, and adjust contamination or scaling if precision is low.

---

## 6) Common Pitfalls

- Feeding unscaled data into PCA or distance-based methods.
- Treating t-SNE/UMAP embeddings as ready-made features without validation.
- Judging by accuracy on imbalanced data; use PR metrics instead.
- Overpowered autoencoders that reconstruct anomalies well—use bottlenecks and early stopping.
- Guessing contamination far from reality, leading to noisy alerts.
- Skipping manual review of top anomalies before shipping alerts.

---

## 7) Quick Reference

- **PCA:** Keep components until cumulative explained variance ~90–95%.
- **Isolation Forest:** High-dimensional numeric data with unclear clusters.
- **LOF:** Local density variations and cluster structure.
- **Autoencoder:** Plenty of normal samples; need nonlinear reconstruction scoring.
- **Metrics:** Precision, recall, PR AUC; avoid plain accuracy.

---

## 8) Practice Task (15–20 minutes)

1. Standardize the credit card dataset, run PCA to retain ~90% variance, and plot explained variance.
2. Create a 2D UMAP projection colored by fraud label, then fit Isolation Forest on PCA components; tune `contamination` so 0.2–0.5% of points are flagged and report average precision plus the top 10 scored transactions.

**Hint:** Start with `n_estimators=200`, `max_samples=256`. If embeddings look noisy, adjust `n_neighbors` or scale before UMAP.

---

## 9) Self-Check

1. Why does PCA usually precede t-SNE or UMAP?
2. What business input should drive your anomaly score threshold?

            .markdown-preview table, 
            .markdown-preview th, 
            .markdown-preview td {
              background-color: white !important;
              color: black !important;
            }
            .markdown-preview pre, 
            .markdown-preview code {
              background-color: inherit !important;
              color: inherit !important;
              box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }