# 23. Lecture notes - Tree Visualisation - Varun Raste - 3 Dec 2025

## [In-class resource](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/e78c661d-52e1-4571-9288-46205734f2d7/AuYUN0pydZvjYVbu.zip)

# Tree Visualisation: Lecture Notes

---

## **1. The Big Picture — Why This Matters**

Decision Trees are widely used in real-world systems that need **quick, explainable decisions** — like loan approvals, fraud checks, medical predictions, and customer churn analysis.

But the model’s logic can get complex fast. Visualization makes this complexity understandable.

### **Where visualization is essential:**

- When a bank must explain *why* a loan was rejected
- When a hospital needs clarity on *why* a patient is marked high-risk
- When analysts want to see whether a model’s logic matches business rules

### **Roles that rely on this skill**

- **Data Scientists** → Debug trees, check split logic
- **ML Engineers** → Identify overfitting, optimize depth
- **Business Analysts** → Validate decisions & ensure fairness

### **Think of a tree like:**

A **giant flowchart** with hundreds of yes/no decisions.

Visualization tools = **Google Maps for this flowchart** — helping you explore, zoom in, and understand how the model thinks.

### **One limitation**

Tree visuals show the *decision logic*, but not deeper mathematical nuance. They simplify the model for understanding, not for statistical evaluation.

---

## **2. Roadmap for This Topic**

We’ll explore three major ideas:

### **1. sklearn’s `plot_tree`**

- Quick, built-in tree diagram
- Good for fast inspection
- Shows nodes, splits, and predicted class

### **2. dtreeviz**

- More detailed, colorful, and intuitive visualizations
- Shows feature distributions, decision boundaries
- Great for stakeholder presentations

### **3. Feature Importance Visualisation**

- Helps identify which features the tree relies on
- Useful for model interpretation & feature engineering

**Flow of the topic:**

Basic → Detailed → Insightful

(`plot_tree` → dtreeviz → feature importance)

---

## **3. Key Terms to Know**

### **plot_tree**

A simple visualization tool from scikit-learn that draws the tree structure — nodes, thresholds, predictions.

**Use it when:**

You want a quick “is this tree behaving logically?” check.

---

### **dtreeviz**

An advanced visualization library with:

- Color-coded branches
- Statistical summaries
- Decision boundaries

**Think of it as:**

From plain flowchart → to infographic-level clarity.

---

### **Feature Importance**

A score showing how much each feature contributed to splitting the data.

**Example:**

Credit scoring may show:

- credit_score → highly important
- region → low importance

This tells you what the tree values most.

---

### **Node Purity**

How “pure” or consistent the data in a node is.

Pure node → mostly one class

Impure node → mixed classes

---

### **Decision Boundary**

The point or rule where the tree splits data.

Example:

“If income > 45k → approve loan”

It's the line that separates predictions.

---

### **Key Insight**

All these tools help us make tree models **transparent**, **trustworthy**, and **explainable**.

---

## **4. Concepts in Action — Simple Scenario**

### **Scenario**

You built a model to predict which telecom customers will churn.

Accuracy is high — 85% — but business leaders ask:

> “Why is the model marking these customers as high risk?”
> 

### **What visualization helps us uncover:**

- Which features matter
- How decisions are being made
- Whether the splits align with business logic
- If the model is fair and unbiased

### **Typical insights from a tree visualization**

- “Contract type” may be the most influential
- Customers with high monthly charges may churn faster
- Short-tenure customers split early in the tree

Visualization takes a **black box** and turns it into a **story you can explain**.

### **Common misconception**

Feature importance only shows *usefulness*, not *cause*.

High importance ≠ causation.

---

## **5. How This Topic Fits In**

### **Builds on:**

- Basic understanding of decision trees
- How nodes split data
- Basics of training a model in scikit-learn

### **Enables you to:**

- Spot overfitting visually
- Explain predictions in plain language
- Identify important vs. unimportant features
- Increase model fairness & transparency
- Meet explainable AI requirements

### **This topic leads naturally into:**

- SHAP for instance-level explanations
- Partial dependence plots
- Tree pruning techniques
- Ensemble model interpretation (Random Forests / XGBoost)

---

## **6. Reflective Questions (Warm-Up)**

These aren’t factual questions—they build intuition.

1. How does seeing the tree structure change your confidence in the model’s decisions?
2. If you presented a fraud detection model to a compliance team, which visualization would help build more trust? Why?
3. When might feature importance be misleading? What extra checks would you do?
4. Think of a recent personal decision — could you draw a simple decision tree describing how you chose?

Encourage students to think visually — not mathematically.

---

## **7. Quick Self-Check**

You’re ready for class if you can:

- Explain **why** tree visualization matters
- Recognize the terms: `plot_tree`, `dtreeviz`, feature importance
- Describe one real-world system that uses decision trees
- Identify one question you want answered in the lecture

If 2 or fewer feel unclear → re-read the big-picture part.

---

# **Final Takeaway**

Tree visualization isn’t just a technical skill — it is a storytelling tool.

It shows *how* a model makes decisions, *why* it chooses certain paths, and *what* matters most.

It bridges the gap between data science and real-world understanding.

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