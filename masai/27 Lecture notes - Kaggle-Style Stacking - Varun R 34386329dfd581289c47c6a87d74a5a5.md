# 27. Lecture notes - Kaggle-Style Stacking - Varun Raste - 12 Dec 2025

[In-class resource](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/5f11aaf7-3082-4852-a530-05a885d12731/xIvU3jLdbvkw3VhQ.zip)

# Kaggle-Style Stacking: Blueprints, Blending & Leaderboard Tricks

**Prerequisites:** Understanding of cross-validation, ensemble methods (Random Forest, Gradient Boosting), basic sklearn pipelines, and train/test splitting concepts.

**Time to complete:** 35-45 minutes

**What you'll be able to do:**

- Build multi-layer stacking ensembles using out-of-fold predictions
- Implement blending scripts that combine diverse model outputs
- Apply competition-proven techniques to maximize leaderboard scores

---

## 1. Introduction: What is Stacking and Why Should You Care?

### Core Definition

Stacking (stacked generalization) is a meta-learning technique where predictions from multiple base models become input features for a final "meta-learner" model. Unlike simple averaging, the meta-learner learns optimal weights and non-linear combinations of base predictions. Kaggle grandmasters use stacking to squeeze the last 0.1-0.5% accuracy that separates winners from top-10 finishers.

### A Simple Analogy

Imagine a panel of specialist doctors: a cardiologist, neurologist, and radiologist each examine a patient and give their diagnosis confidence. Instead of simple majority voting, a senior physician (meta-learner) reviews all three opinions, knowing from experience that the cardiologist is more reliable for certain symptoms. This analogy captures weighted combination but breaks down because stacking also learns non-linear interactions between predictions.

### Why This Matters to You

**Problem it solves:** Individual models have blind spots—XGBoost might miss patterns that neural networks capture, and vice versa. Stacking systematically exploits model diversity to reduce overall error.

**What you'll gain:**

- **Competition-winning ensembles**: Nearly every Kaggle winner uses stacking
- **Robust predictions**: Combining diverse models reduces variance and bias simultaneously
- **Systematic framework**: Repeatable blueprints replace ad-hoc model averaging

---

## 2. The Foundation: Core Concepts Explained

### Concept A: Stacker Blueprints

**Definition:** A stacker blueprint defines the architecture of your ensemble: which base models to use (Level-0), how to generate out-of-fold (OOF) predictions, and which meta-learner combines them (Level-1). The blueprint ensures no data leakage by using K-fold cross-validation to generate training data for the meta-learner.

**Key characteristics:**

- **Level-0 models**: Diverse base learners (XGBoost, LightGBM, CatBoost, Neural Nets, Linear models)
- **Out-of-fold predictions**: Each training sample gets a prediction from a model that never saw it during training
- **Level-1 meta-learner**: Typically a simple model (Logistic Regression, Ridge) to avoid overfitting

**A concrete example:**

```python
from sklearn.model_selection import cross_val_predict
# Generate OOF predictions for meta-learner training
oof_xgb = cross_val_predict(xgb_model, X_train, y_train, cv=5, method='predict_proba')[:, 1]
oof_lgb = cross_val_predict(lgb_model, X_train, y_train, cv=5, method='predict_proba')[:, 1]
meta_features = np.column_stack([oof_xgb, oof_lgb])

```

**Common confusion:** Beginners train base models on full training data, then use those same predictions as meta-features—this causes severe data leakage and inflated CV scores that crash on the leaderboard.

---

### Concept B: Blend Scripts

**Definition:** Blending is a simplified stacking variant where a holdout set (not K-fold) generates meta-features. You split training data into train/blend sets, train base models on the train portion, predict on the blend set, and use those predictions to train the meta-learner.

**How it relates to Stacker Blueprints:** Blending is faster (no K-fold overhead) but wastes data since the blend set isn't used for base model training. Stacking via OOF uses all data for both levels.

**Key characteristics:**

- **Single holdout split**: Typically 70% train, 30% blend
- **Faster iteration**: No K-fold loop means quicker experimentation
- **Data inefficiency**: 30% of training data is "sacrificed" for meta-feature generation

**A concrete example:**

```python
X_train_base, X_blend, y_train_base, y_blend = train_test_split(X, y, test_size=0.3)
model1.fit(X_train_base, y_train_base)
blend_pred1 = model1.predict_proba(X_blend)[:, 1]
# Use blend_pred1 as feature for meta-learner

```

**Remember:** Use blending for rapid prototyping, then switch to full OOF stacking for final submissions.

---

### How Stacker Blueprints and Blend Scripts Work Together

Experienced competitors start with blending to quickly test which model combinations show promise. Once a promising stack is identified, they convert to full OOF stacking for the final submission, gaining back the 30% data that blending sacrificed. Think of blending as the sketch and OOF stacking as the final painting.

---

## 3. Seeing It in Action: Worked Example

### Example: Two-Level Stacking Pipeline

**Scenario:** Build a stacking ensemble for a binary classification competition using XGBoost and LightGBM as base models with Logistic Regression as meta-learner.

**Our approach:** Generate OOF predictions using 5-fold CV, stack them as features, train a simple meta-learner to avoid overfitting.

**Step-by-step solution:**

```python
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Level-0: Base models
xgb = XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
lgb = LGBMClassifier(n_estimators=100, max_depth=4, random_state=42)

# Generate OOF predictions (no leakage)
oof_xgb = cross_val_predict(xgb, X_train, y_train, cv=5, method='predict_proba')[:, 1]
oof_lgb = cross_val_predict(lgb, X_train, y_train, cv=5, method='predict_proba')[:, 1]

# Stack as meta-features
meta_train = np.column_stack([oof_xgb, oof_lgb])

# Level-1: Meta-learner
meta_model = LogisticRegression()
meta_model.fit(meta_train, y_train)

# For test predictions: retrain base models on full training data
xgb.fit(X_train, y_train)
lgb.fit(X_train, y_train)
meta_test = np.column_stack([xgb.predict_proba(X_test)[:, 1],
                              lgb.predict_proba(X_test)[:, 1]])
final_pred = meta_model.predict_proba(meta_test)[:, 1]

```

**What just happened:** Each training sample's OOF prediction came from a model fold that never saw that sample—preventing leakage. The meta-learner learns that perhaps XGBoost is more reliable for certain prediction ranges.

**Check your understanding:** Why do we retrain base models on full training data before predicting on the test set?

---

## 4. Leaderboard Tricks: Competition-Proven Techniques

### Trick 1: Diversity Beats Individual Strength

**The insight:** Combining a 0.85 AUC XGBoost with a 0.83 AUC Neural Net often beats stacking two 0.85 XGBoosts. Diversity in error patterns matters more than individual accuracy.

**Implementation:** Include models with fundamentally different inductive biases—tree-based (XGBoost), linear (Ridge), neural (MLP), and distance-based (KNN).

### Trick 2: Target Encoding as Pseudo-Predictions

**The insight:** Target-encoded categorical features behave like weak model predictions and can be stacked alongside actual model outputs.

### Trick 3: Seed Averaging

**The insight:** Train the same model architecture with 5-10 different random seeds, then average predictions. This reduces variance without adding model complexity.

```python
preds = np.mean([train_model(seed=s).predict(X_test) for s in range(10)], axis=0)

```

---

## 5. Common Pitfalls

- **The Mistake:** Using in-fold predictions instead of out-of-fold for meta-features

**Why It's a Problem:** Base model predictions on data it trained on are overconfident, causing the meta-learner to overfit
**The Right Approach:** Always use `cross_val_predict` or manual OOF loops
**Why This Works:** Each sample's meta-feature comes from a model that never saw it

---

- **The Mistake:** Complex meta-learners (deep neural networks, GBMs)

**Why It's a Problem:** Meta-features are already highly predictive; complex meta-learners memorize noise
**The Right Approach:** Use simple models: Logistic Regression, Ridge, or small ElasticNet
**Why This Works:** Simple models learn robust combination weights without overfitting

---

## 6. Key Takeaways

**Core concept recap:**

- **Stacker blueprints**: OOF predictions from diverse Level-0 models feed a simple Level-1 meta-learner
- **Blend scripts**: Faster prototyping using holdout splits instead of K-fold
- **Leaderboard tricks**: Diversity trumps individual accuracy; seed averaging reduces variance cheaply

### Mental Model Check

By now, you should think of stacking as: a systematic way to let models vote with learned weights, where the meta-learner acts as an optimal weighted ensemble that exploits each base model's strengths while compensating for their weaknesses.

### Next Steps

**To deepen this knowledge:** Implement a 3-level stack with Level-2 combining Level-1 predictions from different meta-learners.

**To build on this:** Explore automated stacking libraries like `mlxtend.StackingClassifier` or `vecstack` for production-ready implementations.

---

## Quick Reference Card

Aspect | Stacking (OOF) | Blending
Data usage | 100% for base models | 70% for base models
Meta-feature generation | K-fold cross_val_predict | Single holdout split
Speed | Slower (K iterations) | Faster (1 iteration)
Best for | Final submissions | Rapid prototyping
Leakage risk | Low if done correctly | Low

[Stacking Architecture]

---

**Questions or stuck?** If your stacked model performs worse than individual base models, check for data leakage in OOF generation or try a simpler meta-learner.

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