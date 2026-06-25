# 19. Lecture notes - Logistic Regression & Metrics - Dr. Surya Prakash - 24 Nov 2025

## [Click here for In-class Notes](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/5bf89ffc-b094-455f-a858-428b3b0e0a7b/i0Rl2tmdXjrOhKLo.zip)

# Logistic Regression & Metrics: Lecture Notes

**Prerequisites:** Basic Python programming, understanding of NumPy arrays and pandas DataFrames, familiarity with basic probability (what percentages and probabilities mean), and conceptual understanding of classification tasks.

**What you'll be able to do:**

- Implement logistic regression from scratch and understand the sigmoid function
- Calculate and interpret confusion matrix, precision, recall, and F1 score by hand and using scikit-learn
- Generate and analyze ROC curves to choose optimal classification thresholds
- Compare multiple models using AUC scores and select the best one for specific business contexts

---

## 1. Introduction: What is Logistic Regression and Why Should You Care?

### Core Definition

**Logistic regression** is a supervised machine learning algorithm for binary classification that predicts the probability of an instance belonging to a particular class (0 or 1) by applying the sigmoid function to a linear combination of input features. Despite its name containing "regression," it's a classification algorithm because it outputs probabilities that are then converted to discrete class labels using a decision threshold (typically 0.5).

**Classification metrics** (confusion matrix, precision, recall, F1, ROC, AUC) are quantitative measures that evaluate how well a classification model performs. Unlike simple accuracy (which can be misleading when classes are imbalanced), these metrics provide nuanced insights into different types of errors and tradeoffs, enabling data scientists to choose models that align with real-world business costs and requirements.

### A Simple Analogy

Think of logistic regression as a college admissions officer who doesn't just make binary accept/reject decisions, but assigns each applicant a probability score (0-100%) of being a successful student based on GPA, test scores, and extracurriculars. The officer combines these factors with learned weights (maybe GPA counts 50%, test scores 30%, extracurriculars 20%) and converts the weighted sum into a smooth probability using a formula that ensures the output stays between 0% and 100%. If the probability exceeds a threshold (say 70%), the applicant is admitted.

The confusion matrix is then like the year-end performance review: of all admitted students (predicted "yes"), how many actually succeeded (true positives) versus struggled (false positives)? And of all students who would have succeeded (actual "yes"), how many did we admit (true positives) versus wrongly reject (false negatives)?

**This analogy works for understanding the probability-based decision-making and evaluation framework, but breaks down when considering that logistic regression learns weights automatically from thousands of historical examples, whereas admissions officers use more subjective judgment.**

### Why This Matters to You

**Problem it solves:** Many critical real-world decisions are binary—approve or deny a loan, flag or allow a transaction, diagnose disease present or absent. Logistic regression provides a probabilistic framework for these decisions, and evaluation metrics ensure we understand the consequences of our model's mistakes before deploying it in production.

**What you'll gain:**

- **Foundational classification skills**: Logistic regression is the stepping stone to understanding more complex classifiers like neural networks, random forests, and gradient boosting—all of which use similar evaluation metrics.
- **Model debugging abilities**: When a model fails in production, metrics like precision and recall help you diagnose whether it's too conservative (low recall) or too aggressive (low precision), guiding your fixes.
- **Business communication skills**: Explaining to stakeholders why your 82% accurate model is better than a competitor's 90% accurate model requires understanding precision, recall, and how they relate to business costs.

**Real-world context:** PayPal's fraud detection uses logistic regression as a baseline, optimizing for high recall to catch fraud. Healthcare diagnostic systems from companies like PathAI use ROC curves to set thresholds that maximize sensitivity (recall) while maintaining acceptable specificity. LinkedIn's spam detection balances precision and recall using F1 optimization.

---

## 2. The Foundation: Core Concepts Explained

**Note:** We'll build these concepts incrementally, starting with the core algorithm.

### Concept A: Binary Logistic Regression

**Definition:** Binary logistic regression models the probability P(y=1|X) that the output y equals 1 given input features X, using the sigmoid function σ(z) = 1/(1 + e^(-z)) applied to a linear combination z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ, where w values are learned weights.

**Key characteristics:**

- **S-shaped (sigmoid) output**: Maps any real number to a probability between 0 and 1
- **Probabilistic predictions**: Outputs interpreted as P(class=1), not just hard 0/1 labels
- **Linear decision boundary**: The boundary where P(y=1)=0.5 forms a straight line (in 2D) or hyperplane (in higher dimensions)
- **Log-loss optimization**: Training minimizes binary cross-entropy loss, not squared error

**A concrete example:**

Predicting whether a customer will churn (1) or stay (0) based on:

- x₁ = monthly spending ($)
- x₂ = support tickets filed

With learned weights w₀=-2, w₁=0.05, w₂=0.8:

```python
# Customer A: $100 spending, 1 ticket
z = -2 + 0.05*100 + 0.8*1 = 3.8
P(churn) = 1/(1 + e^(-3.8)) = 0.978 ≈ 98% → Predict: Churn (1)

# Customer B: $200 spending, 0 tickets
z = -2 + 0.05*200 + 0.8*0 = 8
P(churn) = 1/(1 + e^(-8)) = 0.9997 ≈ 99.97% → Predict: Churn (1)

# Customer C: $50 spending, 2 tickets
z = -2 + 0.05*50 + 0.8*2 = 0.1
P(churn) = 1/(1 + e^(-0.1)) = 0.525 ≈ 52.5% → Predict: Churn (1) [barely!]

```

**Common confusion:** Beginners think logistic regression outputs 0 or 1 directly. In reality, it outputs a probability (continuous value 0-1), which we then convert to a class label by comparing to a threshold (default 0.5). Changing this threshold is crucial for tuning precision vs recall.

---

### Concept B: Confusion Matrix

**Definition:** A confusion matrix is a 2×2 table for binary classification showing the counts of true positives (TP), false positives (FP), false negatives (FN), and true negatives (TN), where "positive" refers to class 1 and "negative" to class 0.

**How it relates to Logistic Regression:** After logistic regression produces probability predictions, we convert them to binary predictions (0 or 1) and compare against actual labels to populate the confusion matrix. This matrix is the foundation for computing all other classification metrics.

**Structure:**

```
                    Predicted
                    0       1
Actual    0        TN      FP
          1        FN      TP

```

- **True Positives (TP)**: Correctly predicted positive class
- **False Positives (FP)**: Incorrectly predicted positive (Type I error)
- **False Negatives (FN)**: Incorrectly predicted negative (Type II error)
- **True Negatives (TN)**: Correctly predicted negative class

**A concrete example:**

Cancer screening test on 100 patients (10 actually have cancer):

```
Predicted:    No Cancer    Cancer
Actual No:       85           5     (FP = 5, healthy marked as sick)
Actual Yes:       3           7     (FN = 3, cancer missed; TP = 7, cancer caught)

TP=7, FP=5, FN=3, TN=85

```

**Remember:** The confusion matrix shows the complete picture of your model's behavior—every possible outcome is accounted for. Accuracy alone would be (85+7)/100 = 92%, which sounds great but hides that we missed 3 out of 10 cancer cases!

---

### Concept C: Precision, Recall, and F1

**Precision** = TP / (TP + FP) = "Of all positive predictions, what fraction are correct?"

**Recall** = TP / (TP + FN) = "Of all actual positives, what fraction did we find?"

**F1 Score** = 2 × (Precision × Recall) / (Precision + Recall) = harmonic mean of precision and recall

**How they relate:** Precision focuses on the quality of positive predictions (avoiding false alarms), while recall focuses on coverage of actual positives (avoiding misses). F1 balances both—it's low unless both precision and recall are reasonably high.

**Key differences:**

Metric | Question Answered | When to Prioritize
Precision | "How trustworthy are my positive predictions?" | When false positives are expensive (spam detection, irrelevant recommendations)
Recall | "How many actual positives am I catching?" | When false negatives are dangerous (disease screening, fraud detection)
F1 | "What's a balanced score?" | When you need both reasonably high, or classes are imbalanced

**Concrete calculation** (using cancer example above):

```
Precision = 7/(7+5) = 7/12 ≈ 0.583 (58.3%)
→ Of patients we diagnosed with cancer, 58.3% actually have it

Recall = 7/(7+3) = 7/10 = 0.7 (70%)
→ We caught 70% of all actual cancer cases

F1 = 2 × (0.583 × 0.7) / (0.583 + 0.7) = 0.636
→ Balanced score reflecting both metrics

```

**Check your understanding:** Why would a cancer screening test prioritize high recall even if it means lower precision?

---

### Concept D: ROC Curve and AUC

**ROC (Receiver Operating Characteristic) Curve**: A plot with False Positive Rate (FPR = FP/(FP+TN)) on the x-axis and True Positive Rate (TPR = Recall) on the y-axis, generated by varying the classification threshold from 0 to 1.

**AUC (Area Under the Curve)**: A single number between 0 and 1 representing the area under the ROC curve. Intuitively, it's the probability that the model ranks a random positive example higher than a random negative example.

**Why ROC curves matter:** They show the tradeoff between catching positives (TPR/recall) and falsely alarming on negatives (FPR) across all possible thresholds, letting you choose the optimal threshold for your business context.

**AUC interpretation:**

- AUC = 0.5: Random guessing (diagonal line)
- AUC = 0.7-0.8: Acceptable model
- AUC = 0.8-0.9: Excellent model
- AUC = 0.9-1.0: Outstanding model (or potential overfitting/data leakage)

**Key insight:** A model with higher AUC is better at ranking—it assigns higher probabilities to actual positives than to actual negatives, regardless of where you set the threshold.

---

## 3. Seeing It in Action: Worked Examples

**Tip:** Study these examples carefully to understand the calculations and tradeoffs.

### Example 1: Computing Metrics from Scratch

**Scenario:** You build a loan default predictor. Testing on 200 loans, you get:

- 150 loans did not default (actual negative)
- 50 loans defaulted (actual positive)
- Your model predicted 60 loans would default
- Of those 60 predictions, 35 actually defaulted (TP=35)
- Of the 50 actual defaults, you caught 35 (so FN=15)

**Our approach:** Build the confusion matrix, then compute all metrics.

**Step-by-step solution:**

```python
# Step 1: Fill in confusion matrix
TP = 35  # Correctly predicted defaults
FN = 15  # Missed defaults (50 actual - 35 caught)
FP = 25  # False alarms (60 predicted - 35 correct)
TN = 125 # Correctly predicted non-defaults (200 - 35 - 15 - 25)

# Verify: TP + FP + FN + TN = 35+25+15+125 = 200 ✓

# Step 2: Compute metrics
precision = TP / (TP + FP) = 35/60 = 0.583
recall = TP / (TP + FN) = 35/50 = 0.7
f1 = 2 * (0.583 * 0.7) / (0.583 + 0.7) = 0.636
accuracy = (TP + TN) / total = 160/200 = 0.8

```

**Output:**

```
Precision: 58.3% (of flagged loans, 58.3% actually defaulted)
Recall: 70% (caught 70% of all defaults)
F1: 63.6%
Accuracy: 80% (but misleading—would be 75% even if we never predicted default!)

```

**What just happened:** Accuracy looks good (80%) but is inflated by the class imbalance (150 vs 50). Precision shows we have many false alarms (25 out of 60 predictions), while recall shows we're missing 30% of defaults. The F1 score of 63.6% reflects both weaknesses.

**Check your understanding:** How would metrics change if we lowered the threshold to catch more defaults?

---

### Example 2: Threshold Impact on Precision vs Recall

**Scenario:** Same loan model, but now we lower the threshold from 0.5 to 0.3, causing more loans to be flagged as high-risk.

**What's different:** Lower threshold → more positive predictions → higher recall (catch more actual defaults) but lower precision (more false alarms).

**Solution:**

```python
# New predictions with threshold=0.3
TP = 45  # Caught 90% of defaults now
FN = 5   # Missed only 5 defaults
FP = 50  # But now 50 false alarms (up from 25)
TN = 100 # Fewer true negatives

# New metrics
precision = 45/95 = 0.474 (47.4%, down from 58.3%)
recall = 45/50 = 0.9 (90%, up from 70%)
f1 = 2 * (0.474 * 0.9) / (0.474 + 0.9) = 0.621

```

**Key lesson:** Lowering the threshold increased recall (caught more defaults) at the cost of precision (more false alarms). This tradeoff is visualized by the ROC curve. For loan defaults, banks might accept lower precision to avoid missing defaults (high recall priority).

---

### Example 3: Comparing Models with AUC

**Background:** A hospital is evaluating three models for predicting readmission risk:

**The challenge:** Model A has accuracy=85%, Model B has accuracy=82%, Model C has accuracy=84%. Which is best?

**The approach:** Calculate AUC for each model by testing on 1000 patients and examining ranking quality.

**Results:**

```
Model A: Accuracy=85%, AUC=0.73, Precision=0.65, Recall=0.60
Model B: Accuracy=82%, AUC=0.89, Precision=0.78, Recall=0.85
Model C: Accuracy=84%, AUC=0.81, Precision=0.72, Recall=0.70

```

**Why this approach:** Model B has the highest AUC (0.89), meaning it's best at ranking—it assigns higher risk scores to patients who actually get readmitted. It also has the best precision-recall balance for this critical healthcare application.

**The outcome:** The hospital chose Model B despite lower accuracy, because AUC and recall matter more than raw accuracy when the cost of missing a readmission (FN) is high—patients might not get preventive care they need.

**Caution:** Don't blindly trust accuracy, especially with imbalanced classes. A model that predicts "no readmission" for everyone might have 90% accuracy if only 10% of patients are readmitted, but it's useless.

---

## 4. Common Pitfalls: What Can Go Wrong and How to Avoid It

**Note:** Understanding these mistakes will deepen your intuition about classification metrics.

1. 
**The Mistake:** Using accuracy as the primary metric with imbalanced classes

**Why It's a Problem:** If 95% of transactions are legitimate, a model that always predicts "legitimate" achieves 95% accuracy while being completely useless for fraud detection
**The Right Approach:** Use precision, recall, and F1 for imbalanced datasets. For fraud detection specifically, prioritize recall (catch all fraud) and use precision to control false alarm rate
**Why This Works:** These metrics focus on performance on the minority class (fraud), not the majority

2. 
**The Mistake:** Ignoring the business cost of different errors

**Why It's a Problem:** Optimizing F1 score treats FP and FN as equally costly, but in cancer screening, missing a case (FN) might cost a life while a false alarm (FP) costs an extra test
**The Right Approach:** Define a custom metric that weights FP and FN by their actual business costs, or use precision-recall tradeoff visualization to set thresholds based on acceptable error rates
**Why This Works:** Aligns model optimization with real-world consequences rather than arbitrary mathematical balance

3. 
**The Mistake:** Comparing AUC scores without considering calibration

**Why It's a Problem:** A model with AUC=0.85 that outputs well-calibrated probabilities (when it says 70% risk, 70% of those cases are positive) is more useful than AUC=0.87 with poorly calibrated probabilities
**The Right Approach:** Use calibration curves to check if predicted probabilities match actual frequencies, and apply calibration methods (Platt scaling, isotonic regression) if needed
**Why This Works:** In applications like risk scoring or medical diagnosis, stakeholders rely on probability values being meaningful, not just relative rankings

**If you're stuck:** Revisit Section 2 on the confusion matrix. Every metric derives from TP, FP, FN, TN—if you understand those four numbers, you can reconstruct any metric.

---

## 5. Your Turn: Practice & Self-Assessment

### Practice Task (Estimated: 20 minutes)

**The Challenge:** Build and evaluate a simple logistic regression classifier for the classic Titanic survival prediction dataset.

**Specifications:**

- Load the Titanic dataset (available via seaborn or scikit-learn)
- Train a logistic regression model using features: Age, Fare, Sex, Pclass
- Generate predictions on a test set and create a confusion matrix
- Calculate precision, recall, F1, and accuracy manually (then verify with sklearn.metrics)
- Plot an ROC curve and compute AUC
- Experiment with thresholds (0.3, 0.5, 0.7) and observe how metrics change

**Hint:** Think about how you'd preprocess categorical features (Sex, Pclass) and handle missing values in Age before feeding them to the model. For the ROC curve, use `predict_proba()` to get probability scores, not just binary predictions from `predict()`.

**Extension (optional):** Implement a simple cost-benefit analysis: assume saving a life (TP) is worth 1M,unnecessarylifeboatallocation(FP)costs1M, unnecessary lifeboat allocation (FP) costs 1M,unnecessarylifeboatallocation(FP)costs10K, missing a death (FN) costs $1M, and correctly identifying survivors who don't need intervention (TN) costs nothing. Which threshold maximizes expected value?

---

### Check Your Understanding

Answer these questions to verify you've grasped the key concepts:

1. 
**Explanation question:** Explain in your own words why the F1 score uses the harmonic mean of precision and recall rather than the arithmetic mean.

2. 
**Application question:** You're building a content moderation system. Your model has precision=0.6 and recall=0.95 for detecting harmful content. A stakeholder says "precision is too low, increase it to 0.9." What tradeoff would you explain to them?

3. 
**Error analysis:** You have a confusion matrix: TP=80, FP=20, FN=10, TN=90. Calculate precision, recall, and explain whether you'd deploy this model for (a) spam filtering or (b) cancer screening.

4. 
**Transfer question:** Two models both have AUC=0.85, but Model A has precision=0.9 and recall=0.6, while Model B has precision=0.7 and recall=0.85. Which would you choose for fraud detection and why?

**Answers & Explanations:**

1. 
The harmonic mean penalizes extreme imbalances more than arithmetic mean. If precision=1.0 and recall=0.1, arithmetic mean=0.55 (looks decent) but harmonic mean (F1)=0.18 (correctly reflects poor performance). This prevents gaming the metric by optimizing only one component.

2. 
Explain that increasing precision to 0.9 will require raising the threshold, which will lower recall—possibly to 0.7 or below. This means more harmful content will slip through (higher FN). Ask whether preventing 5% more false positives (legitimate content wrongly flagged) is worth allowing potentially 25% more harmful content to remain visible (converting current recall 0.95 → 0.7).

3. 
Precision=80/100=0.8, Recall=80/90=0.889. For spam filtering (deploy if precision ≥0.7): YES—acceptable precision means most flagged emails are actually spam. For cancer screening (need recall ≥0.95): NO—missing 11.1% of cancer cases is unacceptable; need higher recall even if precision suffers.

4. 
Model B for fraud detection. Fraud detection prioritizes catching fraud cases (recall=0.85) over avoiding false alarms. Missing fraud (low recall) costs far more than investigating false positives. Model A's higher precision (0.9) would mean fewer false fraud alerts, but the lower recall (0.6) means 40% of fraud goes undetected—unacceptable.

---

### Self-Assessment Checklist

You've mastered this topic if you can:

- Explain how the sigmoid function converts linear combinations into probabilities
- Build a confusion matrix from predictions and calculate all four values (TP, FP, FN, TN)
- Compute precision, recall, and F1 by hand given a confusion matrix
- Explain when to prioritize precision vs recall based on business context
- Interpret an ROC curve and understand how threshold changes affect it
- Compare models using AUC and explain what it measures

**If you checked fewer than 5 boxes:** Review Section 2 (Core Concepts) and work through Example 1 step-by-step again. Focus on understanding how TP, FP, FN, TN relate to each metric.

---

## 6. Consolidation: Key Takeaways & Next Steps

### The Essential Ideas

**Core concept recap:**

- **Logistic Regression**: Uses sigmoid function to map linear combinations to probabilities between 0 and 1, enabling probabilistic binary classification
- **Confusion Matrix**: The 2×2 foundation showing TP, FP, FN, TN—every metric derives from these four numbers
- **Precision vs Recall**: Precision asks "Of my positive predictions, how many are right?" while Recall asks "Of actual positives, how many did I catch?"—different business contexts require different priorities
- **F1 Score**: Harmonic mean of precision and recall, useful when you need both to be reasonably high or when classes are imbalanced
- **ROC/AUC**: ROC visualizes TPR vs FPR across all thresholds; AUC summarizes this into one score measuring ranking quality

### Mental Model Check

By now, you should think of logistic regression as: A probabilistic decision-maker that learns from data to assign probability scores, with evaluation metrics that reveal different facets of performance beyond simple accuracy—enabling you to tune models for real-world costs and constraints.

### What You Can Now Do

You can build binary classifiers, evaluate them comprehensively using multiple metrics, understand why different applications optimize for different metrics (medical diagnosis vs spam filtering), and communicate model tradeoffs to non-technical stakeholders using precision, recall, and ROC curves.

### Next Steps

**To deepen this knowledge:**

- Practice with imbalanced datasets (fraud, rare diseases) to see why precision/recall matter more than accuracy
- Experiment with threshold tuning and observe the precision-recall tradeoff empirically
- Try calibrating probabilities and checking calibration curves

**To build on this:**

- Multiclass classification (extending confusion matrix to 3+ classes, micro/macro averaging)
- Advanced evaluation: precision-recall curves, cost-sensitive learning, calibration
- Regularization in logistic regression (L1/L2) to prevent overfitting

**Additional resources:**

- Scikit-learn's metrics module documentation for implementation details
- "An Introduction to Statistical Learning" Chapter 4 for mathematical depth on logistic regression

---

**Questions or stuck?** Refer back to the confusion matrix—it's the foundation of everything. If you can manually compute TP, FP, FN, TN for a small example, you can derive all other metrics.

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