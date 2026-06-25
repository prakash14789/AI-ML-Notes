# 25. Lecture notes - Ensembles - RF & GBM - Dr. Surya Prakash - 9 Dec 2025

## [In-class resource](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/a3715cc7-fe80-48fc-8235-f6a9e486db33/NVrUkHn3rLPafEeV.zip)

# Ensembles: Random Forest & Gradient Boosting - Lecture Notes

**Prerequisites:** Understanding of decision trees, CART algorithm, overfitting concepts, train/test splits, and basic Python/sklearn.

**What you'll be able to do:**

- Build Random Forest and Gradient Boosting models using sklearn/XGBoost
- Extract and interpret feature importance from ensemble models
- Apply early stopping to prevent overfitting in boosting algorithms

---

## 1. Introduction: What are Ensemble Methods and Why Should You Care?

### Core Definition

Ensemble methods combine multiple weak learners (typically decision trees) to create a stronger, more robust model. Random Forest uses **bagging** (training trees on bootstrapped samples and averaging predictions), while Gradient Boosting uses **boosting** (sequentially training trees where each corrects the previous ones' errors). Both dramatically reduce overfitting compared to single trees while achieving state-of-the-art performance.

### A Simple Analogy

Think of a medical diagnosis. One doctor might miss something, but a panel of 100 specialists each examining you independently, then voting on the diagnosis, is far more reliable. Random Forest is this independent voting panel. Gradient Boosting is like a relay team where each specialist focuses specifically on the symptoms previous doctors misdiagnosed.

This analogy captures the aggregation benefit but breaks down because trees aren't truly independent—they all learn from the same underlying data patterns.

### Why This Matters to You

**Problem it solves:** Single decision trees overfit easily and have high variance—small data changes cause dramatically different trees. Ensembles solve this by averaging predictions, smoothing out individual quirks.

**What you'll gain:**

- **Production-ready models**: XGBoost/LightGBM win most Kaggle competitions and power ML systems at Netflix, Uber, and Airbnb
- **Feature importance**: Understand which variables drive predictions for stakeholder communication
- **Robust performance**: Ensembles generalize better than any individual tree

---

## 2. The Foundation: Core Concepts Explained

### Concept A: Random Forest (Bagging)

**Definition:** Random Forest trains hundreds of decision trees on bootstrapped samples (random sampling with replacement), then averages their predictions. Each tree also considers only a random subset of features at each split, increasing diversity.

**Key characteristics:**

- **Bootstrap aggregating**: Each tree sees ~63% of training data (some samples repeated, ~37% omitted)
- **Feature randomness**: At each split, only `sqrt(n_features)` features are considered
- **Parallel training**: Trees are independent—training is embarrassingly parallel and fast

**A concrete example:**

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
print(f"Test Accuracy: {rf.score(X_test, y_test):.3f}")

```

**Common confusion:** Beginners think more trees always mean better results. Beyond ~100-200 trees, accuracy plateaus while training time increases linearly.

---

### Concept B: Gradient Boosting

**Definition:** Gradient Boosting trains trees sequentially. Each new tree predicts the **residual errors** (gradients) of the ensemble so far, gradually correcting mistakes. The final prediction sums all trees' contributions, weighted by a learning rate.

**How it relates to Random Forest:** Both aggregate trees, but RF trains them independently (parallel), while GB trains them sequentially (each depends on previous). GB typically achieves higher accuracy but is slower and more prone to overfitting.

**Key characteristics:**

- **Sequential learning**: Each tree fits residuals from previous trees
- **Learning rate (shrinkage)**: Controls each tree's contribution (typical: 0.01-0.1)
- **Early stopping**: Stop adding trees when validation performance stops improving

**A concrete example:**

```python
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gb.fit(X_train, y_train)

```

**Remember:** Lower learning rates require more trees but often achieve better generalization.

---

### How Random Forest and Gradient Boosting Work Together

RF reduces variance through averaging independent predictions. GB reduces bias by iteratively correcting errors. In practice, try RF first for a strong baseline with minimal tuning, then use GB/XGBoost when you need maximum accuracy and have time to tune hyperparameters.

---

## 3. Seeing It in Action: Worked Example

### Example: Feature Importance with Early Stopping

**Scenario:** Predict customer churn using 20 features. We need to identify the top drivers and prevent overfitting.

**Our approach:** Train XGBoost with early stopping using a validation set, then extract feature importance.

**Step-by-step solution:**

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Split data: train, validation (for early stopping), test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train XGBoost with early stopping
model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=4, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

print(f"Best iteration: {model.best_iteration}")
print(f"Test Accuracy: {model.score(X_test, y_test):.3f}")

# Feature importance
import pandas as pd
importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
print(importance.head(5))

```

**Output:**

```
Best iteration: 127
Test Accuracy: 0.892
tenure           0.23
monthly_charges  0.18
contract_type    0.15

```

**What just happened:** Early stopping halted training at 127 trees (not 1000), preventing overfitting. Feature importance reveals tenure and monthly charges drive churn predictions—actionable business insight.

[Early Stopping Validation Curve]

**Check your understanding:** Why does early stopping require a separate validation set?

---

## 4. Common Pitfalls

- **The Mistake:** Setting learning_rate too high (e.g., 0.5) with many estimators

**Why It's a Problem:** Each tree overcorrects, causing oscillation and overfitting
**The Right Approach:** Use low learning rate (0.01-0.1) with early stopping
**Why This Works:** Small steps allow gradual, stable convergence

---

- **The Mistake:** Ignoring early stopping with Gradient Boosting

**Why It's a Problem:** Without early stopping, GB will continue adding trees even after validation performance peaks, eventually overfitting
**The Right Approach:** Always use `early_stopping_rounds` with a validation set
**Why This Works:** Training stops when validation loss stops improving for N rounds

---

## 5. Your Turn: Practice Task

**The Challenge:** Build an ensemble pipeline comparing RF vs. XGBoost on a classification dataset.

**Specifications:**

- Load a dataset and split into train/validation/test
- Train RandomForest (100 trees, max_depth=10)
- Train XGBoost with early_stopping_rounds=30
- Compare accuracy and training time
- Plot feature importance for both models side-by-side

**Hint:** Use `time.time()` to measure training duration. XGBoost's `early_stopping_rounds` requires `eval_set` parameter.

---

## 6. Key Takeaways

**Core concept recap:**

- **Random Forest**: Independent parallel trees, averaging reduces variance, robust with minimal tuning
- **Gradient Boosting**: Sequential correction of errors, higher accuracy potential, requires early stopping
- **Feature importance**: Both methods rank features by contribution, enabling interpretable ML

### Mental Model Check

By now, you should think of ensembles as: teams of weak learners that collectively achieve what no single learner can—RF through democratic voting of diverse opinions, GB through iterative refinement where each expert learns from predecessors' mistakes.

### Next Steps

**To deepen this knowledge:** Experiment with LightGBM and CatBoost, which offer faster training and better handling of categorical features.

**To build on this:** Learn hyperparameter tuning with Optuna or GridSearchCV to optimize n_estimators, learning_rate, max_depth, and regularization parameters systematically.

---

## Quick Reference Card

Aspect | Random Forest | Gradient Boosting
Training | Parallel (fast) | Sequential (slower)
Overfitting risk | Low | High without early stopping
Key hyperparams | n_estimators, max_depth | learning_rate, n_estimators, early_stopping
Best for | Quick baseline, robust defaults | Maximum accuracy with tuning
Feature importance | model.feature_importances_ | model.feature_importances_

---

**Questions or stuck?** Compare train vs. validation accuracy to diagnose overfitting. If validation accuracy plateaus while training continues improving, early stopping will help.

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