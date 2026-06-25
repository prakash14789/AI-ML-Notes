# 22. Lecture notes - Decision Trees & Overfitting - Dr. Surya Prakash - 1 Dec 2025

## [In-class notes](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/3a947af3-b837-40c2-bfbc-c0439868a6ac/yTVkUONxvMu3oDmo.zip)

# Decision Trees & Overfitting: Lecture Notes

**Prerequisites:** Understanding of Python programming, basic supervised learning concepts, familiarity with classification tasks, and knowledge of train/test data splits.

**What you'll be able to do:**

- Implement decision trees using CART algorithm with Gini and entropy splitting criteria
- Diagnose and prevent overfitting through pruning and cross-validation
- Visualize and interpret decision boundaries to understand model behavior

---

## 1. Introduction: What are Decision Trees?

### Core Definition

A decision tree is a supervised learning algorithm that makes predictions by learning a hierarchy of if-else decision rules from training data. It recursively splits the feature space into rectangular regions, assigning each region a predicted class (classification) or value (regression). The CART (Classification and Regression Trees) algorithm is the most widely used method for building these trees, creating binary splits at each node by optimizing impurity measures like Gini index or entropy.

### A Simple Analogy

Think of a decision tree like a game of "20 Questions" where you try to guess what someone is thinking by asking yes/no questions. Each question narrows down the possibilities until you arrive at the answer. A decision tree asks questions about feature values ("Is age < 30?", "Is income > $50K?") and follows branches based on answers until reaching a leaf node with the final prediction.

This analogy works for understanding the hierarchical questioning structure, but breaks down when considering that decision trees can ask hundreds of questions and may memorize specific training examples rather than learn general patterns—this is the overfitting problem we'll address.

### Why This Matters

**Problem solved:** Many ML models are "black boxes"—they make accurate predictions but cannot explain *why*. Decision trees provide completely transparent, human-readable decision paths.

**Key benefits:**

- **Interpretability**: Trace exactly why each prediction was made
- **Feature importance**: Automatically rank which variables matter most
- **Versatility**: Handle both numerical and categorical features without encoding

**Real-world context:** Decision trees power Random Forests and Gradient Boosting (XGBoost, LightGBM)—algorithms that win Kaggle competitions and drive production ML systems at companies like Uber and Netflix.

---

## 2. Core Concepts

### CART Algorithm and Recursive Partitioning

CART builds trees top-down by **greedily** selecting the feature and threshold that best splits data at each node. "Greedy" means it makes locally optimal choices without considering future splits.

**Key characteristics:**

- **Binary splits**: Creates exactly two child nodes per split
- **Recursive process**: Same procedure applies to each child node
- **Stopping criteria**: Growth stops when nodes become pure, reach minimum sample size, or hit maximum depth

**Common confusion:** Beginners often think CART looks ahead to plan multiple splits. In reality, CART is myopic—it only optimizes the current split.

### Impurity Measures: Gini Index vs. Entropy

Impurity measures quantify how "mixed" the classes are in a node. Pure nodes (all samples same class) have impurity = 0.

**Gini Index:** Gini=1−∑i=1Cpi2Gini = 1 - \sum_{i=1}^{C} p_i^2Gini=1−∑i=1Cpi2

**Entropy:** Entropy=−∑i=1Cpilog⁡2(pi)Entropy = -\sum_{i=1}^{C} p_i \log_2(p_i)Entropy=−∑i=1Cpilog2(pi)

**Information Gain:** Gain=Iparent−∑nchildnparent×IchildGain = I_{parent} - \sum \frac{n_{child}}{n_{parent}} \times I_{child}Gain=Iparent−∑nparentnchild×Ichild

**Practical note:** Gini and entropy produce nearly identical trees. Gini is slightly faster (no logarithm), while entropy has information-theoretic foundations.

---

## 3. Worked Example: Overfitting Diagnosis

**Scenario:** Compare an unrestricted tree with a pruned tree on a classification dataset.

```python
from sklearn.tree import DecisionTreeClassifier

# Overfit tree: no restrictions
tree_overfit = DecisionTreeClassifier(
    max_depth=None,
    min_samples_leaf=1
)
tree_overfit.fit(X_train, y_train)

# Pruned tree: with constraints
tree_pruned = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5
)
tree_pruned.fit(X_train, y_train)

# Compare performance
print(f"Overfit - Train: {tree_overfit.score(X_train, y_train):.2f}")
print(f"Overfit - Test:  {tree_overfit.score(X_test, y_test):.2f}")
print(f"Pruned  - Train: {tree_pruned.score(X_train, y_train):.2f}")
print(f"Pruned  - Test:  {tree_pruned.score(X_test, y_test):.2f}")

```

**Typical Output:**

```
Overfit - Train: 1.00
Overfit - Test:  0.72
Pruned  - Train: 0.88
Pruned  - Test:  0.85

```

**Key Insight:** The overfit tree achieves 100% training accuracy by memorizing data, but only 72% test accuracy. The pruned tree accepts some training error (88%) but generalizes better (85% test). This demonstrates the **bias-variance tradeoff**.

---

## 4. Common Pitfalls

### Pitfall 1: Growing Trees Until Pure

**The Mistake:** Growing trees until all leaves are pure (100% training accuracy).

**Why It's a Problem:** The tree memorizes noise and outliers. Each leaf might represent a single training sample's quirks rather than general patterns.

**The Right Approach:** Use pre-pruning parameters:

```python
# Recommended pruning parameters
DecisionTreeClassifier(
    max_depth=5,        # Limit tree depth
    min_samples_leaf=10 # Require samples per leaf
)

```

**Why This Works:** Limiting depth forces the tree to learn only strong, generalizable patterns.

### Pitfall 2: Evaluating Only on Training Data

**The Mistake:** Judging model quality solely by training accuracy.

**Why It's a Problem:** Training accuracy is optimistically biased—the model has already seen this data.

**The Right Approach:** Use cross-validation:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree, X, y, cv=5)
print(f"CV Score: {scores.mean():.3f} (+/- {scores.std():.3f})")

```

**Why This Works:** CV provides robust performance estimates across multiple data partitions.

---

## 5. Quick Reference

Concept | Definition | Key Code
Gini Index | Impurity: 0=pure, 0.5=max | 1 - sum(p_i^2)
Pre-pruning | Stop growth early | max_depth=5
Cross-Validation | Robust evaluation | cv=5

**Overfitting Check:**

1. Train accuracy >> Test accuracy? → Overfitting
2. Fix: Reduce `max_depth`, increase `min_samples_leaf`

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