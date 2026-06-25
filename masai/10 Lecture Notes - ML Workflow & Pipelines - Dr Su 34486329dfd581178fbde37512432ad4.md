# 10. Lecture Notes - ML Workflow & Pipelines - Dr. Surya Prakash - 4 Nov 2025

## [Click here for In-class Notes](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/dcfd6363-66d7-47cc-9b54-d298a8db5eda/jpGmIocSQ1l1xdrp.zip)

# ML Workflow & Pipelines: Building Reproducible Machine Learning Systems

**Prerequisites:** Understanding of Python programming (functions, classes, basic data structures), familiarity with pandas DataFrames and numpy arrays, knowledge of what supervised learning is (features, labels, training), and experience using at least one scikit-learn model (like LinearRegression or LogisticRegression).

**What you'll be able to do:**

- Explain the complete ML workflow from raw data to deployed model
- Build reproducible preprocessing and training pipelines using scikit-learn
- Apply systematic feature engineering and preprocessing techniques
- Track experiments and manage ML project versions using DVC

---

## 1. Introduction: What is an ML Workflow and Why Should You Care?

### Core Definition

An ML workflow is the complete sequence of steps required to transform raw data into a deployed machine learning model, including data preprocessing, feature engineering, model training, evaluation, and deployment. Unlike writing isolated scripts, a proper workflow is **reproducible** (anyone can recreate your results), **maintainable** (easy to update when data changes), and **automated** (minimal manual intervention between steps). A pipeline is the code implementation that automates this workflow, ensuring each step executes in the correct order with proper data transformations.

### A Simple Analogy

Think of an ML workflow like a car assembly line. Raw materials (your data) enter at one end, go through multiple stations (preprocessing, feature engineering, training), with each station performing a specific transformation, and a finished car (trained model) comes out at the end. Just like a factory needs quality control and documentation at each stage, your ML workflow needs validation and tracking. This analogy works for understanding the sequential nature and importance of each stage, but breaks down when considering iteration—unlike a factory, you'll often need to go back and adjust earlier steps based on later results.

### Why This Matters to You

**Problem it solves:** Without a structured workflow, ML projects become chaotic—you forget which preprocessing you applied, can't reproduce your best model, waste hours debugging inconsistent transformations between training and deployment, and have no idea which experiments actually worked. A proper workflow prevents these nightmares.

**What you'll gain:**

- **Reproducibility:** Run the same workflow six months later and get identical results, making your work verifiable and debuggable
- **Efficiency:** Automate repetitive tasks so you spend time on insights rather than copy-pasting preprocessing code
- **Confidence:** Deploy models knowing they'll transform production data exactly as they did training data, eliminating the dreaded "it worked on my laptop" problem

**Real-world context:** Companies like Netflix, Uber, and Airbnb use structured ML pipelines to retrain thousands of models regularly—without pipelines, managing this scale would be impossible.

---

## 2. The Foundation: Core Concepts Explained

**Note:** Each concept below represents a distinct phase in the workflow. Understanding them individually before seeing how they connect is crucial.

### Concept A: Data Preprocessing

**Definition:** Data preprocessing is the set of transformations applied to raw data to make it suitable for machine learning algorithms. This includes handling missing values, scaling numerical features to similar ranges, encoding categorical variables as numbers, and removing outliers or duplicates. Preprocessing is not data cleaning—cleaning happens before preprocessing and involves fixing errors in the data itself.

**Key characteristics:**

- **Must be learned from training data only:** Calculate scaling parameters (mean, standard deviation) only from training data to avoid data leakage
- **Must be applied identically to all data:** Training, validation, test, and production data must undergo the exact same transformations
- **Order matters:** Some transformations must happen before others (e.g., handle missing values before scaling)

**A concrete example:**

```python
# Raw feature values: ages [25, 30, 45, 100]
# After scaling: [0.0, 0.067, 0.267, 1.0]  # Normalized to 0-1 range

```

**Common confusion:** Beginners often fit preprocessing transformers on the entire dataset (training + test), which causes data leakage—the model indirectly sees test data through the scaling parameters. The correct approach is to fit only on training data, then apply to test data.

---

### Concept B: Feature Engineering

**Definition:** Feature engineering is the process of creating new features or transforming existing ones to better represent the underlying patterns in your data for machine learning models. Unlike preprocessing (which prepares data), feature engineering creates information—it's where domain knowledge meets data. Examples include creating date features (day of week, month) from timestamps, polynomial features from numerical values, or text length from text columns.

**How it relates to Preprocessing:** Preprocessing standardizes existing features; feature engineering creates new features. Preprocessing typically happens after feature engineering because your new features also need standardization. Think of feature engineering as expanding your toolkit, while preprocessing is sharpening each tool.

**Key characteristics:**

- **Domain-specific:** Good features come from understanding your problem (e.g., for house prices, price-per-square-foot might be more informative than price and area separately)
- **Can be automated or manual:** Some features require human insight, while tools like polynomial features can be automated
- **Impacts model performance significantly:** Often more impactful than model choice—a simple model with great features beats a complex model with poor features

**A concrete example:**

```python
# Original: date = "2024-10-15"
# Engineered features: day_of_week = 2 (Tuesday), month = 10, is_weekend = False

```

**Remember:** This is similar to creating new columns in a pandas DataFrame based on existing ones, but differs in that you must ensure these transformations can be reproduced exactly on new data.

---

### Concept C: Train/Eval/Deploy Stages

**Definition:** The train/eval/deploy stages represent the lifecycle of an ML model. Training is where the model learns patterns from data. Evaluation is where you measure performance on unseen data to estimate real-world accuracy. Deployment is where the model starts making predictions on live data in production. Each stage has different goals and requirements.

**Key characteristics:**

- **Training:** Focus is on learning from data; uses training set only; you can experiment freely
- **Evaluation:** Focus is on honest performance measurement; uses validation/test sets; should mimic production conditions
- **Deployment:** Focus is on reliability, speed, and monitoring; uses real-world data; requires infrastructure and maintenance

**A concrete example:**

```python
# Training: model.fit(X_train, y_train)
# Evaluation: score = model.score(X_test, y_test)  # Never used in training!
# Deployment: prediction = model.predict(new_customer_data)

```

**Common confusion:** Beginners often evaluate on training data or don't separate validation from test sets, resulting in overly optimistic performance estimates. Always keep test data completely separate until final evaluation.

---

### Concept D: sklearn Pipeline

**Definition:** A scikit-learn Pipeline is a Python object that chains multiple data transformations and a final estimator (model) into a single object, executing them sequentially when you call fit or predict. Pipelines ensure transformations are applied consistently and prevent data leakage by guaranteeing that fit operations only see training data.

**How it relates to Preprocessing:** A Pipeline automates the workflow where preprocessing steps come first, followed by the model. Instead of manually running each preprocessing step and keeping track of fitted transformers, the Pipeline manages everything.

**Key characteristics:**

- **Sequential execution:** Steps execute in order; each step's output becomes the next step's input
- **Single fit/predict interface:** Call pipeline.fit() once to fit all transformers and the model; call pipeline.predict() to transform and predict in one go
- **Prevents leakage automatically:** When you call pipeline.fit(X_train), only X_train is seen by all transformers

**A concrete example:**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),      # Step 1: Scale features
    ('classifier', LogisticRegression())  # Step 2: Train model
])
pipeline.fit(X_train, y_train)  # Fits scaler AND trains model
predictions = pipeline.predict(X_test)  # Scales X_test then predicts

```

**Remember:** This is similar to function composition in mathematics (f(g(x))), but differs in that each step can be stateful—it learns parameters during fit and applies them during transform or predict.

---

### Concept E: DVC (Data Version Control)

**Definition:** DVC (Data Version Control) is a tool that brings Git-like version control to machine learning projects, specifically for tracking datasets, models, experiments, and pipelines. While Git tracks code changes, DVC tracks large files (data, models) and experiments (which parameters produced which metrics) without storing the files directly in Git. It stores file metadata in Git and actual files in remote storage (cloud or local).

**How it relates to ML Workflow:** DVC tracks every component of your workflow—the data versions, preprocessing steps, feature engineering code, model files, and evaluation metrics—making your entire workflow reproducible and allowing you to compare experiments systematically.

**Key characteristics:**

- **Lightweight in Git:** Stores only small metadata files (.dvc files) in Git, not the actual data
- **Pipeline automation:** Can define dependencies between workflow stages and re-run only what changed
- **Experiment tracking:** Records which hyperparameters and data versions produced which metrics, creating a searchable experiment history

**A concrete example:**

```bash
# Track a large dataset without putting it in Git
dvc add data/training_data.csv
git add data/training_data.csv.dvc  # Track only the metadata file
git commit -m "Add training data"

```

**Remember:** This is similar to Git for code versioning, but differs in that it's designed for large binary files and adds experiment tracking capabilities specifically for ML workflows.

---

### How These Concepts Work Together

Think of preprocessing and feature engineering as the data preparation factory, the Pipeline as the assembly line that ensures everything happens in the right order, train/eval/deploy as the quality control and delivery stages, and DVC as the factory's documentation system that records what happened when. In practice: you define preprocessing and feature engineering steps, wrap them in a Pipeline to ensure reproducibility, use train/eval/deploy stages to build and validate your model, and use DVC to track everything so you can reproduce or improve it later.

---

## 3. Seeing It in Action: Worked Examples

**Tip:** Study these examples carefully before attempting the practice task. Understanding *why* each step is taken is more important than memorizing the steps.

### Example 1: Basic Preprocessing Pipeline (Simple, Minimal Complexity)

**Scenario:** You have a dataset with customer ages and incomes to predict if they'll buy a product. Ages range from 18-80, incomes from 20K−20K-20K−200K. You need to standardize these features before training a logistic regression model.

**Our approach:** We'll use StandardScaler to standardize both features (mean=0, std=1) and wrap it in a Pipeline with LogisticRegression. This approach makes sense because many algorithms (like logistic regression) perform better when features are on similar scales, and a Pipeline ensures we apply the same scaling to new data.

**Step-by-step solution:**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Sample data
X = np.array([[25, 50000], [45, 80000], [35, 60000], [50, 120000]])  # age, income
y = np.array([0, 1, 0, 1])  # bought product (0=no, 1=yes)

# Step 1: Split data BEFORE creating pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Why: Must separate test data to evaluate honestly later

# Step 2: Create pipeline with preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),          # Standardize features
    ('classifier', LogisticRegression())   # Train classifier
])
# Why: Pipeline ensures scaler is fit only on training data

# Step 3: Fit pipeline on training data only
pipeline.fit(X_train, y_train)
# What happens: StandardScaler learns mean and std from X_train, transforms X_train, 
# then LogisticRegression trains on the transformed data

# Step 4: Evaluate on test data
accuracy = pipeline.score(X_test, y_test)
# What happens: Pipeline automatically scales X_test using the same mean/std from training,
# then predicts and calculates accuracy
print(f"Test accuracy: {accuracy}")

```

**Output:**

```
Test accuracy: 1.0

```

**What just happened:** The Pipeline fit the scaler on training data (learning the mean and standard deviation), transformed the training features, then trained the logistic regression on the transformed data. When scoring on test data, it automatically applied the same scaling (using the training data's parameters) before prediction. This is crucial—test data is scaled using training statistics, not its own statistics.

**Check your understanding:** Why did we split the data BEFORE creating the pipeline, not after?

---

### Example 2: Adding Feature Engineering (One New Element)

**Scenario:** Same customer dataset, but now we suspect that the ratio of income-to-age might be predictive (younger people with high income might behave differently). We need to engineer this feature before preprocessing.

**What's different:** We're creating a new feature (income/age) before scaling. This requires a custom transformer that creates features, followed by the same scaling and modeling pipeline.

**Solution:**

```python
from sklearn.base import BaseEstimator, TransformerMixin

# Step 1: Create custom transformer for feature engineering
class IncomeAgeRatioTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # No parameters to learn
    
    def transform(self, X):
        # X has shape (n_samples, 2): [age, income]
        income_age_ratio = X[:, 1] / X[:, 0]  # income divided by age
        # Add as new column
        return np.column_stack([X, income_age_ratio])

# Step 2: Build pipeline with feature engineering first
pipeline = Pipeline([
    ('feature_eng', IncomeAgeRatioTransformer()),  # Create new feature
    ('scaler', StandardScaler()),                   # Scale all features
    ('classifier', LogisticRegression())            # Train model
])
# Why this order: Create features first, then scale (including new features), then model

# Step 3: Fit and evaluate
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print(f"Test accuracy with engineered feature: {accuracy}")

# The pipeline now automatically:
# 1. Creates income/age ratio for any new data
# 2. Scales all three features (age, income, ratio)
# 3. Predicts using the trained model

```

**Output:**

```
Test accuracy with engineered feature: 1.0

```

**Key lesson:** Feature engineering happens BEFORE preprocessing in the pipeline. The custom transformer follows sklearn's interface (fit and transform methods), making it compatible with Pipeline. This pattern lets you add domain-specific feature engineering while maintaining the benefits of automated workflows.

---

### Example 3: Complete Workflow with DVC Tracking (Realistic Use Case)

**Background:** A small e-commerce company wants to predict customer churn. They have customer data (age, purchase history, website visits) that changes monthly. They need to track which model versions work best, ensure reproducibility when retraining monthly, and deploy the best model.

**The challenge:** They tried manual workflows but kept losing track of which data version was used with which model, couldn't reproduce old results, and wasted time debugging inconsistent preprocessing between training and production.

**The approach:** Set up a complete pipeline with preprocessing and feature engineering, use train_test_split for evaluation, track everything with DVC (data, models, metrics), and define a DVC pipeline to automate retraining when data changes.

**Implementation:**

```python
# File: train_model.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
import json

# Step 1: Load data (tracked by DVC)
data = pd.read_csv('data/customer_data.csv')
X = data[['age', 'total_purchases', 'website_visits']]
y = data['churned']

# Step 2: Feature engineering
X['purchase_frequency'] = X['total_purchases'] / X['website_visits']
X['purchase_frequency'].fillna(0, inplace=True)

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create and train pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = pipeline.predict(X_test)
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred)
}

# Step 6: Save model and metrics (tracked by DVC)
joblib.dump(pipeline, 'models/churn_model.pkl')
with open('metrics/metrics.json', 'w') as f:
    json.dump(metrics, f)

print(f"Model trained! Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}")

```

**DVC Setup:**

```bash
# Initialize DVC in your project
dvc init

# Track the data file (large, changes monthly)
dvc add data/customer_data.csv
git add data/customer_data.csv.dvc .gitignore
git commit -m "Track customer data"

# Define DVC pipeline (automates workflow)
# This goes in dvc.yaml file:
# stages:
#   train:
#     cmd: python train_model.py
#     deps:
#       - data/customer_data.csv
#       - train_model.py
#     outs:
#       - models/churn_model.pkl
#     metrics:
#       - metrics/metrics.json

# Run the pipeline
dvc repro

# Track experiment results
dvc exp show  # Shows metrics for all experiments

```

**Why this approach:**

- **sklearn Pipeline** ensures preprocessing is consistent between training and production—when you load the saved model, it includes the fitted scaler
- **train_test_split** provides honest evaluation by keeping test data separate
- **DVC tracking** ensures they know exactly which data version produced which model and metrics
- **DVC pipeline** automatically re-runs training when data changes, saving manual effort

**The outcome:** After two months, they have 8 tracked experiments in DVC. When performance drops in production, they can compare current data to previous versions, identify drift, and quickly retrain with updated data. The pipeline structure means new data scientists can understand and reproduce any past experiment within minutes.

**Caution:** A common mistake is tracking only the final model with DVC but not the data or metrics. Without tracking data versions, you can't reproduce the model because you don't know what it was trained on. Always track inputs (data), outputs (models), and results (metrics) together.

---

## 4. Common Pitfalls: What Can Go Wrong and How to Avoid It

**Note:** These aren't just mistakes to avoid—they're learning opportunities to deepen your understanding.

### Pitfall 1: Data Leakage Through Preprocessing

**The Mistake:**

```python
# WRONG: Fitting scaler on ALL data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses entire dataset!
X_train, X_test = train_test_split(X_scaled, ...)

```

**Why It's a Problem:** The scaler learned statistics (mean, standard deviation) from the entire dataset, including test data. When you evaluate, the model has indirectly "seen" information from the test set through these statistics. Your test accuracy will be artificially high, giving you false confidence. In production, your model will perform worse because real data won't have been included in these scaling calculations.

**The Right Approach:**

```python
# CORRECT: Fit only on training data
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Learn from training only
X_test_scaled = scaler.transform(X_test)         # Apply to test

```

**Why This Works:** The scaler learns only from training data, then applies those learned parameters to test data. This mimics production: your model will see new data that wasn't used for training. Using a Pipeline automates this pattern—`pipeline.fit(X_train, y_train)` fits transformers only on X_train, and `pipeline.predict(X_test)` applies transformations without refitting.

---

### Pitfall 2: Inconsistent Preprocessing Between Training and Production

**The Mistake:**

```python
# Training code
X_train = df[['age', 'income']].fillna(0)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
model.fit(X_train_scaled, y_train)

# Production code (in a different file, written months later)
new_data = pd.DataFrame({'age': [35], 'income': [75000]})
# Forgot to fill missing values!
new_data_scaled = scaler.transform(new_data)  # Will fail if NaN present
prediction = model.predict(new_data_scaled)

```

**Why It's a Problem:** Training code filled missing values with 0, but production code forgot this step. If production data has missing values, you'll get errors or wrong predictions. Even if there are no missing values, subtle differences (like filling with 0 vs median) will cause distribution shifts and poor predictions. Debugging this is time-consuming because training and deployment often use different codebases.

**The Right Approach:**

```python
# Use Pipeline to bundle ALL preprocessing with the model
from sklearn.impute import SimpleImputer

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Training
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'model.pkl')  # Save entire pipeline

# Production (can be different file/codebase)
pipeline = joblib.load('model.pkl')
new_data = pd.DataFrame({'age': [35], 'income': [75000]})
prediction = pipeline.predict(new_data)  # Automatically applies imputer and scaler

```

**Why This Works:** The Pipeline bundles all preprocessing steps with the model. When you save the pipeline, you save fitted transformers (imputer and scaler with their learned parameters) along with the trained model. In production, loading the pipeline gives you everything needed for consistent preprocessing—no manual steps to remember or forget.

---

### Pitfall 3: Not Tracking Data Versions or Experiments

**The Mistake:** Manually managing experiments in spreadsheets or file naming schemes:

```
models/
  logistic_regression_v1.pkl
  logistic_regression_v2_better.pkl
  random_forest_v1_old.pkl
  random_forest_v2_final.pkl
  random_forest_v3_final_FINAL.pkl

```

You have no idea which data was used, what preprocessing was applied, or which hyperparameters were used for each model. Six months later, you can't reproduce your best result.

**Why It's a Problem:** Without systematic tracking, you lose institutional knowledge. When someone asks "why is this model underperforming?" you can't answer because you don't know what data it was trained on or how it differs from previous versions. You waste time re-running experiments you've already done. Regulatory requirements (like GDPR or financial regulations) may require model reproducibility, which you can't provide.

**The Right Approach:**

```bash
# Initialize DVC for experiment tracking
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"

# Track data versions
dvc add data/customer_data.csv
git add data/customer_data.csv.dvc
git commit -m "Add data v1"

# Define reproducible pipeline in dvc.yaml
# When you train, DVC automatically tracks:
# - Data version (hash of customer_data.csv)
# - Code version (train_model.py in git)
# - Model outputs (saved .pkl file)
# - Metrics (accuracy, f1_score)

# Run experiment
dvc repro

# All information is now tracked:
dvc exp show  # Compare all experiments with metrics
git log       # See which code/data versions were used when

```

**Why This Works:** DVC connects data versions (tracked via content hashing), code versions (via git commits), and results (models and metrics). You can answer questions like "What accuracy did we get with the October dataset using RandomForest with 100 trees?" immediately. You can checkout any past state and reproduce it exactly. DVC's pipeline automation also means new team members can understand dependencies (train.py depends on customer_data.csv) without reading documentation.

---

### Pitfall 4: Misunderstanding Feature Engineering Timing

**The Mistake:** Creating features after splitting or inside the model training loop:

```python
# WRONG: Creating features after split
X_train, X_test = train_test_split(X, ...)

# Create feature for training data
X_train['new_feature'] = X_train['col1'] / X_train['col2']

# Oops! Test data doesn't have this feature
model.fit(X_train, y_train)  
model.predict(X_test)  # ERROR: missing 'new_feature' column

```

**Why It's a Problem:** Features created only on training data won't exist on test/production data, causing errors. Even if you remember to add them to test data, you might apply different logic (e.g., different missing value handling during feature creation), leading to distribution shifts and poor performance.

**The Right Approach:**

```python
# Create features BEFORE splitting
X['new_feature'] = X['col1'] / X['col2']
X['new_feature'].fillna(0, inplace=True)  # Handle division by zero

# Now split
X_train, X_test = train_test_split(X, ...)

# OR use a custom transformer in a Pipeline:
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy['new_feature'] = X_copy['col1'] / X_copy['col2']
        X_copy['new_feature'].fillna(0, inplace=True)
        return X_copy

pipeline = Pipeline([
    ('feature_eng', FeatureEngineer()),  # Applied to both train and test
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

```

**Why This Works:** Creating features before splitting ensures both training and test data have the same features. Using a Pipeline transformer is even better—it guarantees the exact same feature engineering logic is applied to training, test, and production data. The transformer's `transform` method encapsulates the logic, preventing human error.

---

### Pitfall 5: Forgetting to Save/Load Preprocessing Steps in Production

**The Mistake:**

```python
# Training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
joblib.dump(model, 'model.pkl')  # Only saved the model!

# Production
model = joblib.load('model.pkl')
new_data_scaled = StandardScaler().fit_transform(new_data)  # WRONG!
# This creates a NEW scaler with different parameters than training
prediction = model.predict(new_data_scaled)

```

**Why It's a Problem:** You saved only the model, not the scaler. In production, you created a new scaler and fit it on production data (which might be just one row!). This scaler will have completely different mean/std values than the training scaler. Your model expects features scaled with training statistics but receives features scaled with production statistics, causing wildly incorrect predictions.

**The Right Approach:**

```python
# Training: Save EVERYTHING as a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'complete_pipeline.pkl')  # Saves scaler AND model

# Production: Load the complete pipeline
pipeline = joblib.load('complete_pipeline.pkl')
new_data = pd.DataFrame({'age': [35], 'income': [75000]})
prediction = pipeline.predict(new_data)  # Uses the SAME scaler from training

```

**Why This Works:** Saving the entire Pipeline saves all fitted transformers (with their learned parameters) along with the trained model. In production, calling `pipeline.predict()` automatically applies the exact same transformations that were applied during training, using the same parameters. This is the core value proposition of scikit-learn Pipelines.

---

**If you're stuck:** If preprocessing isn't working, revisit **Section 2: Concept A** on preprocessing and the data leakage pitfall. If feature engineering is confusing, review **Section 2: Concept B** and Example 2. If DVC commands aren't clear, check Example 3 which shows a complete DVC workflow. If Pipelines are mysterious, study Example 1's step-by-step code—it shows exactly what happens at each stage.

---

## 5. Your Turn: Practice & Self-Assessment

### Practice Task (Estimated: 20-25 minutes)

**The Challenge:** You work for a bank that wants to predict whether customers will default on loans. You have a dataset with customer age, income, credit_score, and loan_amount. Build a complete ML pipeline that:

1. Engineers a new feature: debt-to-income ratio (loan_amount / income)
2. Handles any missing values by filling with the median
3. Scales all numerical features
4. Trains a RandomForestClassifier
5. Evaluates on held-out test data
6. Saves the complete pipeline so it can be used in production

**Specifications:**

- Use an 80-20 train-test split with random_state=42 for reproducibility
- Your pipeline must include (in order): feature engineering, imputation, scaling, and the classifier
- Print the test accuracy and F1 score
- The saved pipeline file should be named `loan_default_pipeline.pkl`

**Hint:** Think about the order carefully—you need to engineer features first (because you'll create debt-to-income ratio), then handle missing values (because division might create NaN), then scale (because features might be on different scales), then classify. Build this incrementally: first create the custom transformer for feature engineering (following the pattern from Example 2), then add SimpleImputer, then StandardScaler, then RandomForestClassifier. Test after each addition.

**Extension (optional):** Set up DVC tracking for this project—track the data file, define a DVC pipeline in dvc.yaml, and run dvc repro to execute the workflow. Use dvc params to make the train-test split ratio and random forest n_estimators adjustable without changing code.

---

### Check Your Understanding

Answer these questions to verify you've grasped the key concepts:

1. 
**Explanation question:** Explain in your own words why we fit preprocessing transformers only on training data, not on the entire dataset. What would happen if we included test data when fitting a StandardScaler?

2. 
**Application question:** You built a model pipeline and saved it. Six months later in production, you notice predictions are terrible. The pipeline has imputation, scaling, and a logistic regression model. What are three possible causes, and how would you investigate each one?

3. 
**Error analysis:** Look at this code:
`scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled, y)
model = LogisticRegression()
model.fit(X_train, y_train)`

What's wrong with this approach? How would you fix it using a Pipeline?

4. 
**Transfer question:** You're building a recommendation system that needs to process user features (age, location, purchase history) and item features (price, category, popularity) before predicting user-item match scores. How would you structure a pipeline to handle this? Consider that users and items might have different preprocessing needs.

**Answers & Explanations:**

1. 
**Data leakage prevention:** Fitting transformers on training data only ensures the model never sees test data, even indirectly through statistics. If we included test data when fitting StandardScaler, it would calculate mean and standard deviation using both training and test data. During evaluation, the test data would be scaled using parameters that include its own statistics—meaning the model has indirectly "seen" test data. This makes test accuracy artificially high and unrepresentative of production performance, where the model sees truly new data that wasn't used for any calculations. The correct approach mimics production: learn parameters from training data, then apply to new data.

2. 
**Production debugging:** Three possible causes and investigations:

**Data drift:** Production data distribution has changed (e.g., different age ranges, income levels). Investigate by comparing production data statistics to training data statistics—check means, standard deviations, and value ranges for each feature.
**Missing pipeline step:** You might have saved only the model without the complete pipeline, so production code is applying different preprocessing. Investigate by checking what you saved—load the file and verify it's a Pipeline containing all preprocessing steps, not just the model.
**Feature engineering inconsistency:** If any features were created manually (not in the pipeline), production code might be calculating them differently or omitting them. Investigate by logging the exact features being passed to the model in production and comparing to training features—check column names, order, and value ranges.

3. 
**Error and fix:** The error is data leakage—the scaler was fit on all data before splitting, so test data influenced the scaling parameters. This gives optimistic test performance that won't generalize. Fix using a Pipeline:
`# Split first, THEN create pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Pipeline fits scaler only on X_train, applies to X_test during predict
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)`

The Pipeline ensures transformers fit only on training data and apply to test data without refitting.

4. 
**Dual feature pipeline:** You need separate preprocessing for user and item features, then combine them before the model:
`from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define which columns are user vs item features
user_features = ['age', 'purchase_history']  # Might need scaling
item_features = ['price', 'popularity']      # Different scale, might need different preprocessing
categorical = ['location', 'category']       # Need encoding

# Separate preprocessing for each feature type
preprocessor = ColumnTransformer([
    ('user_scaler', StandardScaler(), user_features),
    ('item_scaler', StandardScaler(), item_features),
    ('encoder', OneHotEncoder(), categorical)
]`

---

### Self-Assessment Checklist

You've mastered this topic if you can:

- Explain the complete ML workflow from raw data to deployment to someone else in simple terms without looking at notes
- Recognize when preprocessing should happen before vs after splitting data in new situations
- Build a functioning sklearn Pipeline with custom transformers, preprocessing, and a model from scratch
- Identify and correct the three most common mistakes (data leakage, inconsistent preprocessing, not saving preprocessing with the model)
- Distinguish between preprocessing (standardizing existing features) and feature engineering (creating new features), and know which comes first in a pipeline
- Set up basic DVC tracking for a project (tracking data with dvc add, creating a simple dvc.yaml pipeline)

**If you checked fewer than 5 boxes:** Review Section 3's worked examples, focusing on Example 1 (basic pipeline mechanics) and Example 2 (feature engineering order). Then re-read Section 4 Pitfalls 1-2, which explain data leakage and consistency. Try the practice task again, building the pipeline incrementally and testing after each addition. If DVC is unclear, re-read Concept E in Section 2 and Example 3, then follow the DVC getting started guide at docs.dvc.org.

---

## 6. Consolidation: Key Takeaways & Next Steps

### The Essential Ideas

**Core concept recap:**

- **ML workflows are systematic processes:** Every ML project follows data → preprocess → engineer features → train → evaluate → deploy, and structuring this systematically prevents errors and enables reproducibility
- **Pipelines prevent leakage and ensure consistency:** sklearn Pipelines guarantee preprocessing is applied identically to training, test, and production data by bundling transformers with models
- **Version everything or reproduce nothing:** DVC tracks data versions, code, models, and metrics together, making any past experiment reproducible and enabling systematic comparison

### Mental Model Check

By now, you should think of an ML workflow as: A factory assembly line where raw data enters, passes through automated preprocessing and feature engineering stations (managed by a Pipeline), gets trained into a model with proper validation, and is tracked at every stage (via DVC) so you can reproduce or improve the process at any time.

### What You Can Now Do

You can now build production-ready ML systems rather than one-off scripts. You understand how to prevent subtle bugs like data leakage, ensure your model behaves identically in training and production, and maintain a history of experiments. These skills are fundamental to being a trusted ML practitioner—someone whose work is reproducible, debuggable, and maintainable.

### Next Steps

**To deepen this knowledge:**

- Build a small end-to-end project using a dataset from Kaggle, applying the complete workflow: custom feature engineering, Pipeline with multiple preprocessing steps, proper evaluation, and DVC tracking
- Experiment with ColumnTransformer to handle mixed data types (numerical and categorical features needing different preprocessing)
- Learn about cross-validation for more robust evaluation (combining with Pipeline)

**To build on this:**

- Study hyperparameter tuning with GridSearchCV and RandomizedSearchCV (which work seamlessly with Pipelines)
- Learn about model deployment options (REST APIs with Flask/FastAPI, model serving platforms)
- Explore MLflow or Weights & Biases as alternatives to DVC for experiment tracking
- Investigate feature stores for managing and sharing features across teams

**Additional resources:**

- scikit-learn Pipeline documentation: scikit-learn.org/stable/modules/compose.html (comprehensive guide with advanced patterns)
- DVC Get Started tutorial: dvc.org/doc/start (interactive tutorial, takes 30 minutes)

---

## Quick Reference Card

Component | Purpose | Key Rule
Preprocessing | Standardize features (scaling, encoding, imputation) | Fit only on training data to prevent leakage
Feature Engineering | Create new informative features | Happens before preprocessing; use transformers for automation
Pipeline | Chain transformations + model into one object | Ensures consistent preprocessing; always save/load the complete pipeline
Train/Eval Split | Separate data for honest performance measurement | Split before any transformations; never use test data during training
DVC | Version control for data, models, and experiments | Track data withdvc add, define pipelines in dvc.yaml, compare experiments withdvc exp show

**Common Pipeline Pattern:**

```python
Pipeline([
    ('feature_engineering', CustomTransformer()),
    ('imputation', SimpleImputer()),
    ('scaling', StandardScaler()),
    ('model', RandomForestClassifier())
])

```

**DVC Quick Commands:**

- `dvc add data.csv` - Track large file
- `dvc repro` - Run pipeline
- `dvc exp show` - Compare experiments
- `dvc push/pull` - Sync data with remote storage

---

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