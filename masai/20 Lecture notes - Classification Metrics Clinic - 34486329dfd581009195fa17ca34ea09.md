# 20. Lecture notes - Classification Metrics Clinic - Varun Raste - 26 Nov 2025

## [In-class Notes](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/1829e34a-fe99-4792-a6b6-86d73c979121/nS3eNyASwDoNCCbk.zip)

# Classification Metrics Clinic: Threshold Tuning, Imbalanced Data, and Cost Curves

**Prerequisites:** Understanding of confusion matrix, precision, recall, F1 score, and ROC curves. Experience with binary classification using scikit-learn.

**What you'll be able to do:**

- Tune decision thresholds to optimize for specific business metrics
- Apply techniques like SMOTE and class weighting to handle imbalanced datasets
- Build cost-sensitive classifiers that align with business objectives

---

## 1. Introduction: What Are Advanced Classification Techniques and Why Should You Care?

### Core Definition

Advanced classification techniques encompass threshold tuning (adjusting probability cutoffs for predictions), imbalanced data handling (techniques to train effective models when classes are unevenly distributed), and cost-sensitive learning (optimizing for real-world business costs rather than generic metrics). These methods bridge the gap between academic models that optimize accuracy and production systems that must minimize actual business harm or maximize profit.

### A Simple Analogy

Think of a metal detector at airport security with an adjustable sensitivity dial. Threshold tuning is like adjusting that dial—turn it up and you catch more threats but also stop more innocent people (high recall, low precision). Turn it down and fewer false alarms occur but some threats slip through (high precision, low recall). Handling imbalanced data is like training security staff when 99.99% of passengers are innocent—if you only train on typical passengers, staff won't recognize the rare threat. Cost-sensitive learning means recognizing that missing a weapon (false negative) costs infinitely more than delaying an innocent passenger (false positive), so you calibrate the system accordingly.

**This analogy works for understanding the tradeoffs, but breaks down when considering the mathematical complexity of synthetic sample generation and weighted loss functions.**

### Why This Matters to You

**Problem it solves:** Default classification models optimize for accuracy and use a fixed 0.5 threshold, which fails catastrophically on imbalanced data and ignores that different errors have different real-world costs. A fraud model with 99% accuracy sounds great until you realize it achieves this by predicting "not fraud" for everything when fraud is only 1% of cases.

**What you'll gain:**

- **Deploy production-ready models:** You'll know how to tune models for business KPIs, not just academic metrics, making your work immediately valuable to employers.
- **Handle real-world data:** Since most real datasets are imbalanced (fraud, disease, churn), you'll build models that actually work rather than achieving 95% accuracy by predicting the majority class.
- **Communicate with stakeholders:** By framing model performance in terms of business costs and revenue impact, you'll speak the language of product managers and executives.

**Real-world context:** Every production classification system at major tech companies uses these techniques—Netflix's recommendation confidence thresholds, Google's spam filters, financial fraud detection, and medical diagnostic AI all tune thresholds and handle imbalance.

---

## 2. The Foundation: Core Concepts Explained

### Concept A: Threshold Tuning

**Definition:** Threshold tuning involves changing the probability cutoff (default 0.5) that determines positive class predictions. By adjusting this threshold, you can shift the precision-recall tradeoff to match business priorities without retraining the model.

**Key characteristics:**

- Models output probabilities between 0 and 1; threshold determines when these become positive predictions
- Lowering threshold increases recall (catch more positives) but decreases precision (more false alarms)
- Raising threshold increases precision (fewer false alarms) but decreases recall (miss more positives)

**A concrete example:** Your fraud model outputs probability=0.45 for a transaction. With threshold=0.5, it's classified as legitimate. With threshold=0.3, it's flagged as fraud.

**Common confusion:** Many beginners think you must retrain the model to change precision/recall. Actually, the model's probability outputs stay the same—only your interpretation threshold changes.

---

### Concept B: Class Imbalance Techniques

**Definition:** Class imbalance techniques modify training data or the learning process to handle datasets where one class vastly outnumbers another, preventing models from simply learning to predict the majority class.

**How it relates to Threshold Tuning:** While threshold tuning helps at prediction time, imbalance techniques fix the problem during training so the model learns to recognize minority class patterns effectively.

**Key characteristics:**

- **Resampling:** Oversample minority class (duplicate or synthesize examples) or undersample majority class
- **SMOTE:** Creates synthetic minority examples by interpolating between neighboring minority samples
- **Class weights:** Penalize minority class misclassifications more heavily during training
- **Hybrid approaches:** Combine multiple techniques for maximum effectiveness

**A concrete example:** With 9,900 "not fraud" and 100 "fraud" cases, applying class_weight='balanced' in scikit-learn makes each fraud case count 99× more in the loss function.

**Remember:** This addresses the same problem as threshold tuning but from a different angle—improving what the model learns rather than how you interpret its outputs.

---

### Concept C: Cost-Sensitive Learning

**Definition:** Cost-sensitive learning explicitly incorporates the real-world business cost of different types of errors into model training and evaluation, optimizing for minimum total cost rather than maximum accuracy.

**How it relates to Threshold Tuning and Imbalance:** Cost-sensitive learning provides the framework for deciding optimal thresholds and whether imbalance techniques are worth the complexity—it quantifies "worth" in dollars or impact.

**Key characteristics:**

- Uses a cost matrix specifying C(FP), C(FN), C(TP), C(TN) in business terms
- Expected cost = (FP × Cost_FP) + (FN × Cost_FN) + (TP × Cost_TP) + (TN × Cost_TN)
- Optimal threshold minimizes expected cost, not maximize F1 or accuracy
- Cost curves visualize expected cost across different operating points

**A concrete example:** Medical screening where false negative (missed cancer) costs 500Kinliabilityandtreatment,whilefalsepositive(unnecessarybiopsy)costs500K in liability and treatment, while false positive (unnecessary biopsy) costs 500Kinliabilityandtreatment,whilefalsep

**Remember:** F1 score treats precision and recall equally, but in reality, different errors have vastly different impacts.

---

### How These Concepts Work Together

Threshold tuning provides the tool for adjusting model behavior post-training. Imbalance techniques ensure the model learns meaningful patterns from rare classes during training. Cost-sensitive learning provides the objective function that guides both—determining which threshold to use and whether the added complexity of SMOTE or class weights delivers sufficient cost reduction. Think of imbalance handling as building a better foundation, threshold tuning as the final adjustments, and cost analysis as the blueprint guiding everything.

---

## 3. Seeing It in Action: Worked Examples

### Example 1: Threshold Tuning for Fraud Detection

**Scenario:** You've trained a fraud detection model on 100,000 transactions (0.5% fraud rate = 500 fraud cases). At threshold=0.5, you achieve precision=0.75, recall=0.60. You need to decide the optimal threshold.

**Our approach:** Calculate business impact at different thresholds by mapping confusion matrix outcomes to costs, then plot a cost curve to visualize the optimal operating point.

**Step-by-step solution:**

```python
# Step 1: Define business costs
cost_fn = 3500  # Missing fraud costs $3,500 on average
cost_fp = 25    # False alarm costs $25 in support/lost goodwill

# Step 2: Calculate expected cost at current threshold (0.5)
total_fraud = 500
tp = 0.60 * 500  # 300 caught
fn = 0.40 * 500  # 200 missed
# Precision = TP/(TP+FP) = 0.75, so TP+FP = 300/0.75 = 400
fp = 400 - 300   # 100 false alarms

expected_cost_50 = (fn * cost_fn) + (fp * cost_fp)
# = (200 * 3500) + (100 * 25) = $700,000 + $2,500 = $702,500

# Step 3: Test threshold=0.35 (from ROC curve analysis)
# This gives recall=0.85, precision=0.65
tp_new = 0.85 * 500  # 425 caught
fn_new = 0.15 * 500  # 75 missed
fp_new = (425 / 0.65) - 425  # 229 false alarms

expected_cost_35 = (fn_new * cost_fn) + (fp_new * cost_fp)
# = (75 * 3500) + (229 * 25) = $262,500 + $5,725 = $268,225
# Savings: $434,275 daily just by lowering threshold!

```

**What's happening here:** By lowering the threshold from 0.5 to 0.35, we catch 125 more fraud cases (saving 437,500)atthecostof129morefalsepositives(costing437,500) at the cost of 129 more false positives (costing 437,500)atthecostof129morefalsepositives(costing3,225). The net benefit is massive because fraud costs far exceed false alarm costs.

**Check your understanding:** Why doesn't threshold=0.0 (flag everything as fraud) give the best result?

*Answer: While this catches all fraud (FN=0), it creates 99,500 false positives costing 2.5M,farexceedingthe2.5M, far exceeding the 2.5M,farexceedingthe1.75M saved from catching all 500 frauds.*

---

### Example 2: Handling Imbalance with SMOTE

**Scenario:** You're building a customer churn predictor with 5,000 customers: 4,750 stayed (95%) and 250 churned (5%). Your baseline model achieves 95% accuracy by predicting "no churn" for everyone, learning nothing useful.

**What's different:** Instead of accepting poor minority class performance, we'll apply SMOTE to balance the training set.

**Solution:**

```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Original imbalanced data: X (features), y (labels: 0=stayed, 1=churned)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Apply SMOTE to training data only
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Before: 3800 stayed, 200 churned in training
# After: 3800 stayed, 3800 churned (SMOTE created 3600 synthetic churn examples)

# Train model on balanced data
model = RandomForestClassifier(random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# Evaluate on original imbalanced test set
# Recall on churn class improves from 0.05 to 0.72
# Precision on churn drops slightly from 1.0 to 0.45
# But now you're actually catching churners instead of missing them all!

```

**Key lesson:** SMOTE creates synthetic minority examples by finding k-nearest neighbors in the minority class and generating new samples along the lines connecting them, teaching the model what minority class samples "look like" without just duplicating existing ones.

**Caution:** Only apply SMOTE to training data, never to test data—you want to evaluate on the real distribution to see how your model performs in production.

---

### Example 3: Cost-Sensitive Learning in Medical Diagnosis

**Background:** A hospital is deploying an AI system to screen chest X-rays for early-stage lung cancer. The model will flag high-risk cases for radiologist review.

**The challenge:** Class imbalance (only 2% cancer prevalence) and asymmetric costs—missing cancer (false negative) leads to delayed treatment costing lives and legal liability (~500,000),whilefalsepositivemeansanunnecessaryCTscanandbiopsy( 500,000), while false positive means an unnecessary CT scan and biopsy (~500,000),whilefalsepositivemeansanunnecessaryCTscanandbiopsy(

**The approach:**

**Step 1: Define the cost matrix**

- C(TN) = 0 (correct negative screen, no action needed)
- C(TP) = 0 (correct positive screen, life saved)
- C(FP) = $2,000 (unnecessary follow-up)
- C(FN) = $500,000 (missed cancer case)

**Step 2: Calculate expected cost across thresholds**
Test thresholds from 0.1 to 0.9 and calculate: Expected_Cost = (FP × 2000) + (FN × 500000)

**Step 3: Select optimal threshold**
At threshold=0.15: Recall=0.95, Precision=0.12, Expected_Cost = 110Kper1000patientsAtthreshold=0.50:Recall=0.75,Precision=0.35,ExpectedCost=110K per 1000 patients
At threshold=0.50: Recall=0.75, Precision=0.35, Expected_Cost = 110Kper1000patientsAtthreshold=0.50:Recall=0.75,Precis

**Why this approach:** Despite lower precision at threshold=0.15 (more false alarms), the expected cost is much lower because catching 95% of cancers versus 75% saves many more lives and avoids massive liability costs that dwarf the added screening expense.

**The outcome:** The hospital deployed with threshold=0.15, achieving 95% sensitivity (recall) in production. While radiologists review more false positives, they catch nearly all early-stage cancers, improving patient outcomes and reducing liability.

**What this shows:** The "best" model configuration has nothing to do with maximizing accuracy or F1—it's about minimizing real-world harm quantified through a cost function aligned with your specific use case.

---

## 4. Common Pitfalls: What Can Go Wrong and How to Avoid It

**The Mistake:** Applying SMOTE to the entire dataset before train-test split, including test data.

- **Why It's a Problem:** Synthetic samples in the test set are created based on training data, leading to data leakage. Your model's test performance will be artificially inflated because it's partially evaluated on data derived from training examples.
- **The Right Approach:** Always split first, then apply SMOTE only to training data: `X_train, X_test = split(); X_train_balanced = SMOTE(X_train)`
- **Why This Works:** Test data remains independent and representative of real-world distribution, giving honest performance estimates.

---

**The Mistake:** Using class_weight='balanced' and SMOTE simultaneously without understanding their interaction.

- **Why It's a Problem:** Both techniques address imbalance—SMOTE rebalances data via oversampling, while class_weight rebalances via loss function weighting. Combining them can lead to over-correction, where the model becomes too aggressive at predicting the minority class, creating excessive false positives.
- **The Right Approach:** Try each technique separately first: (1) baseline, (2) class_weight only, (3) SMOTE only, (4) both combined. Compare cost curves to see which delivers the best business outcome for your specific problem.
- **Why This Works:** Empirical comparison reveals which approach (or combination) works best for your data's specific characteristics and cost structure.

---

**The Mistake:** Optimizing for F1 score when business costs are highly asymmetric.

- **Why It's a Problem:** F1 score is the harmonic mean of precision and recall, implicitly treating false positives and false negatives as equally costly. When missing a fraud case costs 3,500butafalsealarmcosts3,500 but a false alarm costs 3,500butafalsealarmcosts25 (140× difference), F1 optimization leads to suboptimal thresholds.
- **The Right Approach:** Define your cost matrix explicitly, calculate expected cost at multiple thresholds, and select the threshold that minimizes total expected cost.
- **Why This Works:** You're optimizing the metric that actually matters to your business, not an arbitrary statistical measure that assumes equal error costs.

---

**If you're stuck:** Revisit Section 2 on cost-sensitive learning and ensure you clearly understand how to map confusion matrix cells to business costs.

---

## 5. Your Turn: Practice & Self-Assessment

### Practice Task (Estimated: 20 minutes)

**The Challenge:** You're building a customer churn prediction model for a subscription service. Historical data: 10,000 customers, 8,500 retained (85%), 1,500 churned (15%).

**Business costs:**

- Retaining a customer who would have churned (true positive): +$200 lifetime value
- Offering retention incentive to loyal customer (false positive): -$50 wasted incentive
- Missing a churner (false negative): -$0 (they leave anyway, no additional cost)
- Correctly identifying loyal customer (true negative): $0

**Specifications:**

- Build a cost matrix for this scenario
- Calculate expected profit at threshold=0.5 where precision=0.60, recall=0.70
- Determine whether lowering the threshold to 0.3 (precision=0.45, recall=0.85) improves business outcomes
- Explain your recommendation

**Hint:** This is different from typical classification because true positives have value (retained customers) rather than cost. Calculate expected profit = (TP × 200)−(FP×200) - (FP × 200)−(FP×50) - (FN × $0), and compare across thresholds. Remember that total churners = 1,500 regardless of threshold.

**Extension (optional):** Research the precision-recall curve and explain why it's often preferred over ROC curves for imbalanced datasets.

---

### Check Your Understanding

Answer these questions to verify you've grasped the key concepts:

1. 
**Explanation question:** Explain in your own words why SMOTE is generally preferred over simple random oversampling (duplicating minority class samples). What problem does it solve?

2. 
**Application question:** You're deploying a spam filter for a corporate email system. Would you prioritize high precision or high recall, and what threshold adjustment would support that priority?

3. 
**Error analysis:** A colleague applies SMOTE to their dataset, then splits into train/test, achieving 92% accuracy. What's wrong with this approach, and what would happen to performance in production?

4. 
**Transfer question:** How would you apply cost-sensitive learning to a quality control system in manufacturing where defective products cost 500torecallbutstoppingproductionforinspectioncosts500 to recall but stopping production for inspection costs 500torecallbutstoppingproductionforinspectioncosts2,000 per hour?

**Answers & Explanations:**

1. 
SMOTE generates synthetic samples by interpolating between existing minority samples in feature space, creating variations that help the model learn the minority class boundary. Simple oversampling just duplicates existing samples, which doesn't provide new information and can lead to overfitting on the exact duplicated examples. SMOTE gives the model more diverse minority examples to learn from.

2. 
Corporate spam filters should prioritize **high precision** (low false positive rate) because flagging legitimate business emails as spam causes serious problems—missed meetings, lost deals, communication breakdowns. You'd **raise the threshold** above 0.5 (perhaps to 0.7-0.8) so only emails with very high spam probability get filtered, minimizing false positives even if some spam slips through (lower recall is acceptable).

3. 
**Critical error:** Applying SMOTE before splitting causes data leakage because synthetic test samples are generated using information from training data. In production, the model will perform worse than the 92% test accuracy suggests because real-world test data won't have this artificial relationship to training data. **Correct approach:** Split first, then apply SMOTE only to training data.

4. 
Build a cost matrix: C(FP)=2,000(unnecessaryinspectionstopsproduction),C(FN)=2,000 (unnecessary inspection stops production), C(FN)=2,000(unnecessaryinspectionstopsproduction),C(FN)=500 (defect reaches customer). With defect rate and inspection time, calculate expected cost across different quality thresholds. If defects are rare (1%), you might set a higher threshold to reduce false alarms (unnecessary stoppages), as the cost of stopping production frequently exceeds occasional recalls. Use a cost curve to find the threshold that minimizes total expected cost.

---

### Self-Assessment Checklist

You've mastered this topic if you can:

- Adjust classification thresholds to optimize for business metrics rather than generic accuracy
- Apply SMOTE correctly (training data only, after splitting) to handle imbalanced datasets
- Build a cost matrix from business requirements and use it to select optimal thresholds
- Explain why maximizing F1 score can lead to poor business outcomes when costs are asymmetric
- Distinguish when to use class_weight versus SMOTE versus threshold tuning
- Calculate expected cost from a confusion matrix and cost matrix

**If you checked fewer than 5 boxes:** Review Section 3 (worked examples) carefully, especially the cost calculations in the fraud detection and medical diagnosis examples. Practice calculating expected cost manually for different confusion matrices.

---

## 6. Consolidation: Key Takeaways & Next Steps

### The Essential Ideas

**Core concept recap:**

- **Threshold tuning** lets you shift precision-recall tradeoffs post-training to match business priorities without retraining the model
- **Imbalance techniques** (SMOTE, class_weight) fix training issues when minority classes are underrepresented in data
- **Cost-sensitive learning** aligns model optimization with real business impact by explicitly incorporating error costs into decision-making

### Mental Model Check

By now, you should think of classification optimization as: Building a model that outputs good probabilities (training), ensuring minority classes are learned effectively (imbalance handling), then tuning the decision threshold to minimize real-world cost rather than maximize abstract metrics.

### What You Can Now Do

You can now deploy classification models that perform well on imbalanced real-world datasets and make threshold decisions that maximize business value. You can communicate model performance to non-technical stakeholders in terms of expected costs and revenue impact.

### Next Steps

**To deepen this knowledge:**

- Implement a full threshold tuning pipeline on an imbalanced dataset (credit card fraud, customer churn)
- Create cost curves for different business scenarios and compare optimal thresholds

**To build on this:**

- Explore calibration techniques (Platt scaling, isotonic regression) to ensure probability outputs are reliable for threshold tuning
- Learn about precision-recall curves as an alternative to ROC curves for imbalanced problems

---

**Questions or stuck?** Revisit the worked examples in Section 3, particularly the fraud detection cost calculation, to solidify your understanding of how confusion matrix outcomes map to business impact.

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