# 11. Lecture Notes: Building Reproducible Pipelines - Varun Raste - 5 Nov 2025

# Building Reproducible Pipelines: Lecture Notes

**Prerequisites:** Understanding of ML workflows and sklearn Pipelines (preprocessing, feature engineering, train/test splits), basic Git operations (commit, push, pull), familiarity with the command line, and a conceptual understanding of what CI/CD (Continuous Integration/Continuous Deployment) means.

**What you'll be able to do:**

- Track ML experiments systematically using MLflow with parameters, metrics, and models
- Set up remote storage for datasets and models using DVC to enable team collaboration
- Organize ML project configurations using config files for maintainable, version-controlled parameters
- Implement CI hooks to automatically validate code quality and run pipeline tests before commits

---

## 1. Introduction: What Are Reproducible Pipelines and Why Should You Care?

### Core Definition

A reproducible pipeline is an ML workflow where every component—data versions, code, dependencies, configurations, and environment—is tracked and automated such that anyone (including your future self) can recreate the exact same results from the same inputs. Reproducibility extends beyond just versioning: it includes experiment tracking (which parameters produced which results), remote storage (so teams share the same data), configuration management (parameters separated from code), and automated validation (quality checks before code is committed). This is different from a working pipeline, which might produce results but can't be reliably re-run or understood by others.

### A Simple Analogy

Think of reproducible pipelines like a published recipe with photos of each step. A regular recipe says "add flour and eggs, bake"—it might work but you can't verify or debug it. A reproducible recipe specifies exact amounts (200g flour, tracked like parameters), includes a photo of each stage (tracked like MLflow logs), tells you where to buy ingredients (remote storage like DVC), provides a checklist before starting (CI hooks), and uses a standardized oven setting file (config files). Anyone following it gets identical results. This analogy works for understanding the multiple layers of reproducibility, but breaks down when considering collaboration—ML pipelines involve parallel work and merging changes, which is more complex than following a recipe.

### Why This Matters to You

**Problem it solves:** Without reproducible pipelines, ML teams face daily nightmares: "This model worked last week, what changed?", "I can't access the training data from three months ago", "Which hyperparameters did we use for the production model?", "Your code breaks my setup", and "We deployed a model but no one knows what data it was trained on." These problems destroy productivity and create compliance risks.

**What you'll gain:**

- **Experiment tracking:** Record every experiment automatically with MLflow—parameters, metrics, models, and artifacts—searchable and comparable, so you never lose a good result
- **Team collaboration:** Share data and models through DVC remotes without cluttering Git, enabling teammates to reproduce your work instantly
- **Maintainability:** Separate code from configuration so you can adjust hyperparameters without code changes, and automate checks to prevent broken code from entering the repository

**Real-world context:** Companies like Airbnb use MLflow to track thousands of experiments across hundreds of models. Spotify uses DVC for sharing multi-terabyte datasets across global teams. Netflix implements CI pipelines that automatically test ML code before deployment, preventing production failures.

---

## 2. The Foundation: Core Concepts Explained

**Note:** Each concept addresses a different reproducibility challenge. Understanding them independently before seeing how they connect is crucial.

### Concept A: MLflow for Experiment Tracking

**Definition:** MLflow is an open-source platform that tracks ML experiments by automatically logging parameters (hyperparameters and configurations used), metrics (performance measurements like accuracy or loss), artifacts (files like plots or trained models), and metadata (timestamp, user, code version) for every experiment run. Unlike manual spreadsheet tracking, MLflow provides a UI for comparing runs, querying results programmatically, and managing the model lifecycle from experimentation through production deployment.

**Key characteristics:**

- **Automatic logging:** With minimal code (`mlflow.autolog()` for many frameworks), MLflow captures parameters and metrics without manual tracking
- **Centralized storage:** All experiment data goes into one place (local directory or remote server), accessible through a web UI and programmatic API
- **Model registry:** Beyond tracking, MLflow can manage model versions (staging, production, archived) with approval workflows

**A concrete example:**

```python
import mlflow

mlflow.start_run()
mlflow.log_param("learning_rate", 0.01)
mlflow.log_metric("accuracy", 0.92)
mlflow.sklearn.log_model(model, "model")
mlflow.end_run()

# Later: Search for the best run
best_run = mlflow.search_runs().sort_values("metrics.accuracy", ascending=False).iloc[0]

```

**Common confusion:** Beginners think MLflow is only for logging metrics, but it's a complete lifecycle tool—it tracks experiments (what you tried), manages models (which to deploy), and can even deploy models. The tracking component is just one part. For this lesson, we focus on tracking and model management, not MLflow's deployment capabilities.

---

### Concept B: DVC Remotes for Data Storage

**Definition:** DVC remotes are external storage locations (cloud storage like S3, Google Cloud Storage, Azure Blob, or even local network drives) where DVC stores actual data files and models, while keeping only lightweight metadata (.dvc files) in Git. When you `dvc push`, large files upload to the remote; when teammates `dvc pull`, they download from the remote. This solves Git's limitation with large files and enables data versioning across teams without repository bloat.

**How it relates to DVC basics:** Basic DVC tracks files locally with `dvc add`, creating .dvc metadata files. DVC remotes extend this by adding synchronization—you can share tracked files with teams by pushing to and pulling from shared storage, similar to how Git push/pull shares code.

**Key characteristics:**

- **Storage flexibility:** Supports multiple backends (S3, GCS, Azure, SSH, local paths), configured once and used transparently
- **Automatic file management:** DVC handles uploading/downloading efficiently, only transferring changed files (like Git for large files)
- **Access control:** Storage backends handle permissions—use cloud provider's IAM policies to control who can access data

**A concrete example:**

```bash
# Configure a remote (one-time setup)
dvc remote add -d myremote s3://my-company-ml-data/project1

# Push your tracked data to the remote
dvc push

# Teammate on different machine
dvc pull  # Downloads data from s3://my-company-ml-data/project1

```

**Remember:** This is similar to Git's remote repositories (like GitHub), but differs in that DVC remotes store large binary files efficiently, whereas Git remotes are optimized for text-based code files.

---

### Concept C: Configuration Files for Parameter Management

**Definition:** Configuration files (often YAML, JSON, or Python files) store all adjustable parameters and settings for your ML pipeline—hyperparameters, file paths, model choices, preprocessing options—separate from your code. Instead of hardcoding values like `n_estimators=100` in your Python script, you read from a config file. This separation enables changing parameters without modifying code, makes parameters visible and version-controlled, and allows different configs for different environments (development vs production).

**How it relates to pipelines:** In your ML code, you read configuration values instead of using literals. This makes pipelines configurable—the same code can run different experiments or in different environments just by swapping config files. DVC can track these config files as dependencies, re-running pipelines when configs change.

**Key characteristics:**

- **Declarative:** Configs describe what parameters to use (declarative) rather than how to use them (imperative code)
- **Version-controlled:** Config files live in Git, so you know exactly which parameters produced which results
- **Environment-specific:** Can have configs for dev, staging, production without duplicating code

**A concrete example:**

```yaml
# config.yaml
model:
  type: "RandomForest"
  n_estimators: 100
  max_depth: 10
  
preprocessing:
  fill_missing: "median"
  scale_method: "standard"
  
data:
  train_path: "data/train.csv"
  test_split: 0.2

```

```python
# train.py reads from config instead of hardcoding
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

n_estimators = config['model']['n_estimators']  # 100
model = RandomForestClassifier(n_estimators=n_estimators)

```

**Remember:** This is similar to using variables instead of magic numbers in programming, but differs in that configs are external files that can change without touching code, enabling non-programmers to adjust parameters.

---

### Concept D: CI Hooks for Automated Validation

**Definition:** CI (Continuous Integration) hooks are automated scripts that run before or after Git operations (like commits or pushes), performing checks such as code formatting, linting, running tests, or even executing ML pipeline stages. Pre-commit hooks run locally before commits succeed, catching issues early. CI pipelines (like GitHub Actions) run on servers after pushing, performing more extensive checks like training models or running full test suites. Hooks ensure code quality and reproducibility by preventing broken or poorly formatted code from entering the repository.

**How it relates to ML workflows:** In ML projects, hooks can validate that data files are tracked with DVC (not committed directly to Git), check that config files have required fields, run unit tests for preprocessing functions, or even execute fast versions of your pipeline to ensure they don't error before committing changes.

**Key characteristics:**

- **Automatic enforcement:** Hooks run without manual intervention, ensuring consistency across team members
- **Fast feedback:** Pre-commit hooks fail immediately if issues exist, preventing bad commits rather than discovering problems later
- **Configurable:** Can enable/disable specific checks per project, balancing strictness with convenience

**A concrete example:**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.0.0
    hooks:
      - id: black  # Auto-formats Python code
        
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8  # Checks code style issues
        
  - repo: local
    hooks:
      - id: check-dvc-files
        name: Check large files use DVC
        entry: python scripts/check_dvc.py
        language: system

```

When you run `git commit`, these hooks execute automatically. If any fail, the commit is rejected.

**Common confusion:** Hooks are not the same as running manual scripts before commits. Hooks are enforced—you can't commit without passing them (unless deliberately bypassed with flags). This automatic enforcement ensures team-wide consistency, whereas manual scripts depend on discipline.

---

### How These Concepts Work Together

Think of MLflow as your lab notebook recording every experiment, DVC remotes as your shared storage closet where everyone accesses the same materials, config files as your recipe cards that tell you what settings to use, and CI hooks as quality inspectors at the door who check your work before it enters the official lab records (Git). In practice: you modify a config file (changing hyperparameters), CI hooks validate it on commit, you run experiments that MLflow tracks automatically, you push new model files to DVC remotes for sharing, and everything is version-controlled and reproducible.

---

## 3. Seeing It in Action: Worked Examples

**Tip:** Study these examples carefully before attempting the practice task. Understanding *why* each component is configured and *how* they interact is more important than memorizing commands.

### Example 1: Basic MLflow Experiment Tracking (Simple, Minimal Complexity)

**Scenario:** You're training a RandomForest classifier for customer churn prediction. You want to track which hyperparameters (n_estimators, max_depth) produce the best accuracy, and save the best model for later use.

**Our approach:** We'll use MLflow to log parameters, metrics, and the model itself. This approach makes sense because manually tracking experiments in spreadsheets is error-prone and doesn't save models—MLflow automates everything and provides searchability.

**Step-by-step solution:**

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# Sample data
data = pd.read_csv('customer_data.csv')
X = data.drop('churned', axis=1)
y = data['churned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: Set experiment name (organizes related runs)
mlflow.set_experiment("churn_prediction")
# Why: Groups all churn-related experiments together in the UI

# Step 2: Start a run (each run is one experiment)
with mlflow.start_run(run_name="random_forest_v1"):
    # Step 3: Log hyperparameters
    n_estimators = 100
    max_depth = 10
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    # Why: You'll want to know which settings produced these results
    
    # Step 4: Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Step 5: Evaluate and log metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    # Why: Metrics are what you optimize for—need them for comparison
    
    # Step 6: Save the model
    mlflow.sklearn.log_model(model, "model")
    # Why: The trained model is saved as an artifact, loadable later
    
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Accuracy: {accuracy:.3f}, F1: {f1:.3f}")

# Step 7: View in UI
# Run: mlflow ui
# Visit http://localhost:5000 to see all runs with searchable parameters and metrics

```

**Output:**

```
Run ID: 7f3c8e9a1b2c3d4e5f6g7h8i9j0k
Accuracy: 0.876, F1: 0.843

```

**What just happened:** MLflow created a "run" (a single experiment) inside the "churn_prediction" experiment. It saved parameters (n_estimators=100, max_depth=10), metrics (accuracy=0.876, f1=0.843), and the trained model. All this information is stored in the `mlruns/` directory and viewable in the web UI. The run_id uniquely identifies this experiment—you can load this exact model later using just the ID.

**Check your understanding:** Why did we call mlflow.log_param before training the model, not after?

---

### Example 2: Comparing Multiple Runs with MLflow (Adding Complexity)

**Scenario:** You want to try different hyperparameter combinations (n_estimators: 50, 100, 200 and max_depth: 5, 10, 15) to find the best model. Manually tracking this gets messy quickly.

**What's different:** We're running multiple experiments programmatically and using MLflow's search API to find the best result automatically, rather than manually comparing in the UI.

**Solution:**

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load data
data = pd.read_csv('customer_data.csv')
X = data.drop('churned', axis=1)
y = data['churned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("churn_prediction_grid_search")

# Grid of hyperparameters to try
n_estimators_options = [50, 100, 200]
max_depth_options = [5, 10, 15]

# Run experiments for each combination
for n_est in n_estimators_options:
    for depth in max_depth_options:
        with mlflow.start_run(run_name=f"rf_n{n_est}_d{depth}"):
            # Log parameters
            mlflow.log_param("n_estimators", n_est)
            mlflow.log_param("max_depth", depth)
            
            # Train
            model = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            accuracy = accuracy_score(y_test, model.predict(X_test))
            mlflow.log_metric("accuracy", accuracy)
            
            # Save model
            mlflow.sklearn.log_model(model, "model")
            
            print(f"n_estimators={n_est}, max_depth={depth}, accuracy={accuracy:.3f}")

# After all runs, find the best one programmatically
experiment = mlflow.get_experiment_by_name("churn_prediction_grid_search")
runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Sort by accuracy (descending) and get best run
best_run = runs_df.sort_values("metrics.accuracy", ascending=False).iloc[0]

print("\n=== Best Model ===")
print(f"Run ID: {best_run['run_id']}")
print(f"Parameters: n_estimators={best_run['params.n_estimators']}, max_depth={best_run['params.max_depth']}")
print(f"Accuracy: {best_run['metrics.accuracy']:.3f}")

# Load the best model
best_model_uri = f"runs:/{best_run['run_id']}/model"
best_model = mlflow.sklearn.load_model(best_model_uri)

```

**Output:**

```
n_estimators=50, max_depth=5, accuracy=0.845
n_estimators=50, max_depth=10, accuracy=0.868
...
n_estimators=200, max_depth=15, accuracy=0.891

=== Best Model ===
Run ID: 3d4e5f6g7h8i9j0k1l2m
Parameters: n_estimators=200, max_depth=15
Accuracy: 0.891

```

**Key lesson:** MLflow's search API (`mlflow.search_runs()`) lets you query experiments programmatically, not just through the UI. This is powerful for automated workflows—you can select the best model based on metrics, load it, and deploy it, all in code. The runs_df is a pandas DataFrame with columns like `metrics.accuracy` and `params.n_estimators`, making analysis easy.

---

### Example 3: Complete Reproducible Pipeline with DVC Remotes, Configs, and CI (Real-World Application)

**Background:** A data science team at an e-commerce company builds models to predict product recommendations. They have a 5GB training dataset that multiple team members need, they experiment with different hyperparameters frequently, and they've had issues with broken code being committed that crashes the training pipeline.

**The challenge:** Without a proper setup, team members couldn't access each other's data (too large for Git), experiments were tracked in scattered spreadsheets, changing hyperparameters required code edits (risking bugs), and broken code often made it to the main branch, wasting team time.

**The approach:** Set up a complete reproducible pipeline infrastructure:

1. Use DVC remotes (S3 bucket) for sharing the 5GB dataset
2. Use config files (YAML) to manage hyperparameters separately from code
3. Use MLflow for tracking all experiments
4. Use pre-commit hooks to validate code quality and ensure DVC tracking before commits

**Implementation:**

**Step 1: DVC Remote Setup (One-time team setup)**

```bash
# Initialize DVC
dvc init

# Add S3 bucket as remote storage
dvc remote add -d myremote s3://company-ml-data/recommendation-project
dvc remote modify myremote region us-west-2

# Configure AWS credentials (if needed)
dvc remote modify myremote access_key_id YOUR_ACCESS_KEY
dvc remote modify myremote secret_access_key YOUR_SECRET_KEY

# Track the large dataset
dvc add data/training_data.csv
git add data/training_data.csv.dvc data/.gitignore
git commit -m "Track training data with DVC"

# Push data to S3
dvc push

# Now teammates can pull the data
# On another machine:
# git pull  (gets the .dvc file)
# dvc pull  (downloads actual data from S3)

```

**Step 2: Configuration File Structure**

```yaml
# config/train_config.yaml
data:
  train_path: "data/training_data.csv"
  test_split: 0.2
  random_state: 42

model:
  type: "RandomForest"
  n_estimators: 100
  max_depth: 15
  min_samples_split: 5

preprocessing:
  scaler: "standard"
  handle_missing: "median"

mlflow:
  experiment_name: "product_recommendations"
  tracking_uri: "http://mlflow-server.company.com"  # Or "file:./mlruns" for local

# Different config for production
# config/prod_config.yaml
data:
  train_path: "s3://company-ml-data/prod/training_data.csv"
  test_split: 0.2
  random_state: 42

model:
  type: "RandomForest"
  n_estimators: 200  # More estimators for production
  max_depth: 20
  min_samples_split: 2

```

**Step 3: Training Script Using Config**

```python
# train.py
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import sys

# Load config (can switch configs easily)
config_path = sys.argv[1] if len(sys.argv) > 1 else 'config/train_config.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)

# Set up MLflow
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment(config['mlflow']['experiment_name'])

# Load data
data = pd.read_csv(config['data']['train_path'])
X = data.drop('purchased', axis=1)
y = data['purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=config['data']['test_split'],
    random_state=config['data']['random_state']
)

# Start MLflow run
with mlflow.start_run():
    # Log all config as parameters
    mlflow.log_params({
        "n_estimators": config['model']['n_estimators'],
        "max_depth": config['model']['max_depth'],
        "min_samples_split": config['model']['min_samples_split'],
        "test_split": config['data']['test_split'],
        "scaler": config['preprocessing']['scaler']
    })
    
    # Log the config file itself as artifact
    mlflow.log_artifact(config_path)
    
    # Build pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=config['model']['n_estimators'],
            max_depth=config['model']['max_depth'],
            min_samples_split=config['model']['min_samples_split'],
            random_state=config['data']['random_state']
        ))
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }
    
    # Log metrics
    mlflow.log_metrics(metrics)
    
    # Save model
    mlflow.sklearn.log_model(pipeline, "model")
    
    print(f"Training complete! Metrics: {metrics}")

# Usage:
# python train.py config/train_config.yaml  (dev environment)
# python train.py config/prod_config.yaml   (production environment)

```

**Step 4: DVC Pipeline for Automation**

```yaml
# dvc.yaml - defines the ML pipeline
stages:
  train:
    cmd: python train.py config/train_config.yaml
    deps:
      - data/training_data.csv
      - train.py
      - config/train_config.yaml
    params:
      - train_config.yaml:
          - model.n_estimators
          - model.max_depth
    outs:
      - models/model.pkl
    metrics:
      - metrics/metrics.json:
          cache: false

# Run the pipeline
# dvc repro  (runs only if dependencies changed)

```

**Step 5: CI Hooks for Quality Control**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.0.0
    hooks:
      - id: black
        language_version: python3.9
        
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=88', '--extend-ignore=E203']
        
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']  # Prevent large files in Git
      - id: trailing-whitespace
      
  - repo: local
    hooks:
      - id: check-dvc-tracking
        name: Ensure large files are tracked with DVC
        entry: python scripts/check_dvc_tracking.py
        language: system
        pass_filenames: false
        
      - id: validate-config
        name: Validate config files
        entry: python scripts/validate_config.py
        language: system
        files: ^config/.*\.yaml$

# Install hooks (one-time per developer)
# pip install pre-commit
# pre-commit install

```

```python
# scripts/check_dvc_tracking.py
import os
import sys

# Check if there are .csv or .pkl files staged without .dvc files
result = os.popen('git diff --cached --name-only').read()
staged_files = result.strip().split('\n')

large_files = [f for f in staged_files if f.endswith(('.csv', '.pkl', '.h5'))]
dvc_tracked = [f for f in staged_files if f.endswith('.dvc')]

if large_files and not dvc_tracked:
    print("ERROR: Large files detected without DVC tracking:")
    for f in large_files:
        print(f"  - {f}")
    print("\nUse 'dvc add <file>' to track large files.")
    sys.exit(1)

print("✓ No large files without DVC tracking")

```

**Why this approach:**

- **DVC remotes:** Team members pull data from S3, not passing around USB drives or Slack uploads. Everyone has access to the exact same data version.
- **Config files:** Changing hyperparameters is now a config edit, not a code change. Non-technical team members can adjust parameters. Different environments (dev/prod) use different configs but the same code.
- **MLflow:** Every experiment is logged automatically with the config file as an artifact, so you always know which config produced which results.
- **CI hooks:** Developers can't commit code that fails linting, can't add large files to Git without DVC tracking, and config files are validated before commits.

**The outcome:** After implementation, the team's velocity increased dramatically. New team members onboard in hours (dvc pull gets data, pre-commit hooks install quality checks). Experiments are searchable (MLflow UI). Hyperparameter tuning happens through config changes, not code edits. Broken code is caught before reaching the main branch. When a model underperforms in production, they can compare its config to previous successful experiments in MLflow and identify what changed.

**Caution:** A common mistake is setting up only some pieces (e.g., MLflow but not DVC remotes, or config files but not CI hooks). Each component addresses a different reproducibility problem. Without DVC remotes, teammates can't share data. Without config files, parameter changes risk code bugs. Without CI hooks, quality degrades over time. The components work together—implement all of them for full reproducibility.

---

## 4. Common Pitfalls: What Can Go Wrong and How to Avoid It

**Note:** These aren't just mistakes to avoid—they're learning opportunities to deepen your understanding of why each component matters.

### Pitfall 1: Not Logging Config Parameters to MLflow

**The Mistake:**

```python
# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Train using config values
model = RandomForestClassifier(n_estimators=config['model']['n_estimators'])

# Start MLflow run but don't log the config
with mlflow.start_run():
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    # Forgot to log n_estimators!

```

**Why It's a Problem:** Later when reviewing experiments in MLflow, you see accuracy scores but don't know which hyperparameters produced them. You can't reproduce the experiment because the parameters aren't recorded. You have the model but not the context of how it was trained. If you have 50 runs, you can't sort by n_estimators to see the effect of this parameter.

**The Right Approach:**

```python
with open('config.yaml') as f:
    config = yaml.safe_load(f)

with mlflow.start_run():
    # Log all relevant config parameters
    mlflow.log_params({
        "n_estimators": config['model']['n_estimators'],
        "max_depth": config['model']['max_depth'],
        "learning_rate": config['model'].get('learning_rate', 'N/A'),
    })
    
    # Log the entire config file as an artifact
    mlflow.log_artifact('config.yaml')
    
    # Train and log model
    model = RandomForestClassifier(n_estimators=config['model']['n_estimators'])
    model.fit(X_train, y_train)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

```

**Why This Works:** You log individual parameters so they're searchable in MLflow's UI and API (you can filter or sort by n_estimators). You also log the entire config file as an artifact, preserving the complete configuration for perfect reproduction. If someone wants to know "what settings produced 0.92 accuracy?", they can see all parameters immediately in the MLflow UI.

---

### Pitfall 2: Forgetting to Push DVC Data to Remote

**The Mistake:**

```bash
# Developer A
dvc add data/new_training_set.csv
git add data/new_training_set.csv.dvc
git commit -m "Add new training data"
git push  # Pushes only the .dvc file, not the actual data!

# Developer B (on different machine)
git pull  # Gets the .dvc file
dvc pull  # ERROR: Data not found in remote!

```

**Why It's a Problem:** Git only tracks the .dvc metadata file, not the actual data. When you forget `dvc push`, the .dvc file points to data that only exists on your machine. Teammates can't reproduce your work because the data isn't accessible. This breaks the entire reproducibility chain—they have your code and your model, but not the data it was trained on.

**The Right Approach:**

```bash
# Developer A
dvc add data/new_training_set.csv
git add data/new_training_set.csv.dvc
git commit -m "Add new training data"

# CRITICAL: Push to DVC remote
dvc push  # Uploads data to S3/GCS/etc
git push  # Uploads .dvc file to GitHub

# Developer B
git pull   # Gets .dvc file
dvc pull   # Successfully downloads data from remote

```

**Why This Works:** The `dvc push` command uploads actual data files to the remote storage. Now when teammates `dvc pull`, they download from the shared remote. Think of it as a two-step push: `dvc push` for data, `git push` for metadata. Both are required for sharing.

**Pro tip:** Add this to your workflow checklist or use git hooks to remind you:

```bash
# .git/hooks/pre-push (make this file executable)
#!/bin/sh
dvc status --remote
if [ $? -ne 0 ]; then
    echo "WARNING: You may have unpushed DVC data. Run 'dvc push' before pushing to Git."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

```

---

### Pitfall 3: Hardcoding Values Despite Having Config Files

**The Mistake:**

```python
# You have a config file with n_estimators: 100
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# But then hardcode values in multiple places
model = RandomForestClassifier(n_estimators=100, max_depth=10)  # Hardcoded!
scaler = StandardScaler()  # Should this come from config?

# Later, you change config to n_estimators: 200
# But the code still uses 100 because it's hardcoded
# Your MLflow logs show n_estimators=200 (from logging the config)
# But the actual model used 100!

```

**Why It's a Problem:** The config file becomes documentation fiction—it claims to control parameters but doesn't actually control them. When you change the config, nothing changes in execution. Your MLflow experiments become misleading because logged parameters don't match what was actually used. Debugging becomes impossible because you trust the logs but they're wrong.

**The Right Approach:**

```python
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Read ALL values from config
model = RandomForestClassifier(
    n_estimators=config['model']['n_estimators'],
    max_depth=config['model']['max_depth'],
    min_samples_split=config['model']['min_samples_split'],
    random_state=config['data']['random_state']
)

# Preprocessing also comes from config
scaler_type = config['preprocessing']['scaler']
if scaler_type == 'standard':
    scaler = StandardScaler()
elif scaler_type == 'minmax':
    scaler = MinMaxScaler()
else:
    raise ValueError(f"Unknown scaler type: {scaler_type}")

# Now config is the single source of truth

```

**Why This Works:** Every adjustable value comes from the config file. There's one source of truth. When you change config['model']['n_estimators'], the actual training uses the new value. The config file becomes a contract—what it says is what happens. This enables experimentation by editing config files without touching code, and it makes MLflow logs accurate.

---

### Pitfall 4: CI Hooks That Are Too Strict or Not Strict Enough

**The Mistake (Too Strict):**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: full-pipeline-test
        name: Run complete ML pipeline
        entry: python train.py  # Runs 30-minute training on every commit!
        language: system

```

Developers wait 30 minutes per commit. They start bypassing hooks with `--no-verify`. The team abandons pre-commit hooks entirely.

**The Mistake (Too Lenient):**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.0.0
    hooks:
      - id: black
# That's it! No other checks.

```

Black formats code, but doesn't catch logic errors, missing documentation, or large files committed to Git. Bugs and bad practices make it to the repository regularly.

**Why It's a Problem:** Hooks that are too strict waste developer time and get bypassed. Hooks that are too lenient don't catch real issues. The balance is crucial—hooks should catch important issues quickly without blocking legitimate work.

**The Right Approach:**

```yaml
# .pre-commit-config.yaml - Balanced approach
repos:
  # Fast formatting (< 1 second)
  - repo: https://github.com/psf/black
    rev: 23.0.0
    hooks:
      - id: black

  # Fast linting (< 2 seconds)
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=88']

  # Fast checks (< 1 second each)
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: check-yaml          # Validates YAML syntax
      - id: check-added-large-files
        args: ['--maxkb=1000']  # Prevents large files
      - id: end-of-file-fixer   # Ensures files end with newline

  # Fast custom checks (< 3 seconds)
  - repo: local
    hooks:
      - id: validate-config
        name: Validate config structure
        entry: python scripts/validate_config.py
        language: system
        files: ^config/.*\.yaml$
        
      - id: check-dvc-tracking
        name: Ensure large data files use DVC
        entry: python scripts/check_dvc_tracking.py
        language: system

  # Unit tests for critical functions (< 10 seconds)
      - id: fast-tests
        name: Run fast unit tests
        entry: pytest tests/unit/ -v --tb=short
        language: system
        pass_filenames: false

# Slow tests run in CI pipeline (GitHub Actions), not pre-commit
# .github/workflows/ci.yml
# on: push
# jobs:
#   full-pipeline-test:
#     runs-on: ubuntu-latest
#     steps:
#       - uses: actions/checkout@v2
#       - name: Run full training pipeline
#         run: python train.py --fast-mode  # Subset of data for speed

```

**Why This Works:** Pre-commit hooks are fast (< 15 seconds total), catching formatting issues, syntax errors, and policy violations (large files, missing DVC tracking) before commit. Slow checks (full pipeline tests) run on the server in CI pipelines after pushing, where they don't block individual developers. Developers keep hooks enabled because they're fast and helpful, not burdensome.

**Decision framework for hook speed:**

- **Pre-commit:** Fast checks (< 20 seconds total) that catch common mistakes
- **CI pipeline:** Slow checks (minutes) that validate complete functionality
- **Rule of thumb:** If a check takes > 30 seconds, move it to CI pipeline, not pre-commit

---

### Pitfall 5: Not Tracking MLflow Server Location in Config

**The Mistake:**

```python
# Hardcoded MLflow tracking URI
import mlflow
mlflow.set_tracking_uri("http://mlflow.company.com")

# Different developer has local setup
# mlflow.set_tracking_uri("file:./mlruns")  # Commented out

# Experiments go to different locations!

```

**Why It's a Problem:** Some developers log to a shared MLflow server, others log locally. Team members can't see each other's experiments. When you onboard a new team member, they don't know where MLflow is configured. Production code might point to a development MLflow instance, mixing environments.

**The Right Approach:**

```yaml
# config/dev_config.yaml
mlflow:
  tracking_uri: "file:./mlruns"  # Local for development
  experiment_name: "dev_experiments"

# config/prod_config.yaml
mlflow:
  tracking_uri: "http://mlflow.company.com"  # Shared server
  experiment_name: "production_models"

```

```python
# train.py
import os
import yaml
import mlflow

# Load config based on environment variable or command line arg
env = os.getenv('ENV', 'dev')
config_path = f'config/{env}_config.yaml'

with open(config_path) as f:
    config = yaml.safe_load(f)

# Set MLflow from config
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment(config['mlflow']['experiment_name'])

# Now all experiments go to the right place based on environment

```

**Why This Works:** MLflow configuration lives in config files like everything else. Different environments (dev, staging, prod) have different configs, but the code is identical. New team members see the tracking URI in the config and know where experiments are stored. You can switch environments with a single environment variable: `ENV=prod python train.py`.

---

**If you're stuck:** If MLflow isn't logging what you expect, revisit **Section 2: Concept A** and Example 1—check that you're calling log_param and log_metric inside the run context. If DVC remote issues arise, see **Pitfall 2** and verify you ran `dvc push`. If configs aren't being used, check **Pitfall 3** for hardcoded values. If CI hooks are problematic, review **Pitfall 4** for balancing speed and strictness.

---

## 5. Your Turn: Practice & Self-Assessment

### Practice Task (Estimated: 25-30 minutes)

**The Challenge:** You're building a sentiment analysis model for product reviews. Set up a complete reproducible pipeline infrastructure with:

1. **MLflow tracking:** Log hyperparameters (model type, learning rate), metrics (accuracy, F1), and the trained model
2. **Config file:** Create a YAML config storing all adjustable parameters (data path, test split ratio, model hyperparameters, MLflow experiment name)
3. **DVC setup:** Track your training data file using DVC (simulate remote by using a local directory as remote storage)
4. **Pre-commit hook:** Create one custom pre-commit hook that validates your config file has required keys (data.train_path, model.type, mlflow.experiment_name)

**Specifications:**

- Use any simple classifier (LogisticRegression or RandomForestClassifier)
- Your config must control at least 3 model hyperparameters
- Training script must read all values from config (no hardcoded parameters)
- The pre-commit hook must fail if any required config key is missing
- MLflow experiment name must come from config
- Document your setup in a README.md explaining how a teammate would reproduce your workflow

**Hint:** Build incrementally:

1. Start with a simple training script with hardcoded values
2. Add MLflow logging (wrap training in mlflow.start_run())
3. Create config.yaml and refactor script to read from it
4. Initialize DVC, add your data file, set up a local directory as remote
5. Create .pre-commit-config.yaml with a local hook
6. Write validate_config.py that checks for required keys
7. Test everything: change a config value, commit (should run validation), train (should use new config), check MLflow UI (should show new experiment)

**Extension (optional):**

- Add a DVC pipeline (dvc.yaml) that automatically re-runs training when config or data changes
- Set up GitHub Actions workflow that runs training on every push
- Implement MLflow model registry to promote the best model to "production"

---

### Check Your Understanding

Answer these questions to verify you've grasped the key concepts:

1. 
**Explanation question:** Explain why we need both DVC remotes AND Git remotes in an ML project. Why can't we just use Git for everything?

2. 
**Application question:** Your team's MLflow shows 100 experiments with various hyperparameters, but you need to find all experiments where n_estimators > 100 and accuracy > 0.85. How would you do this programmatically (not through the UI)?

3. 
**Error analysis:** A teammate complains: "I pulled your latest code from Git and ran `dvc pull`, but I get an error 'cache not found'. What did I do wrong?" What are two possible causes, and how would you fix each?

4. 
**Transfer question:** You're working on a computer vision project where models are 500MB and datasets are 50GB. You experiment with different model architectures (ResNet, VGG, EfficientNet) and dataset augmentation strategies. How would you structure your configs, DVC tracking, and MLflow logging to handle this efficiently? Consider that teammates have varying internet bandwidth.

**Answers & Explanations:**

1. 
**Different storage needs:** Git is designed for text-based source code with rich version control (diffs, merges, branches). It struggles with large binary files (datasets, models) because it stores complete copies of every version, bloating repository size. DVC remotes solve this by storing large files efficiently in external storage (S3, GCS) while keeping only lightweight pointers in Git. You need Git for code versioning (branching, pull requests, history) and DVC for data versioning (efficient storage of large binaries). Think of it as division of labor: Git handles what it's good at (code), DVC handles what Git can't (large data).

2. 
**Programmatic search:** Use MLflow's search_runs with filter strings:
`import mlflow

# Filter syntax: SQL-like expressions on params and metrics
filter_string = "params.n_estimators > '100' AND metrics.accuracy > 0.85"

# Search across all experiments (or specify experiment_ids)
filtered_runs = mlflow.search_runs(
    filter_string=filter_string,
    order_by=["metrics.accuracy DESC"]
)

print(f"Found {len(filtered_runs)} matching runs")
print(filtered_runs[['params.n_estimators', 'metrics.accuracy', 'run_id']])`

This returns a pandas DataFrame with only runs matching the criteria, sortable and filterable further. Note that params are stored as strings, so numeric comparisons require converting.

3. 
**Two possible causes:**

**Cause 1: Teammate forgot to push data to remote.** You ran `dvc add` and committed the .dvc file, but didn't run `dvc push`. Fix: You run `dvc push` to upload data, teammate runs `dvc pull` again.
**Cause 2: Teammate doesn't have remote configured.** They don't have credentials or the remote URL configured locally. Fix: Share the DVC remote configuration. Teammate runs `dvc remote add -d myremote s3://bucket-name` with proper credentials, then `dvc pull` succeeds.
Both issues stem from DVC's two-tier system: Git tracks pointers (.dvc files), remote storage holds actual data. Both must be configured correctly.

4. 
**Large-scale project structure:**
`# config/experiment_config.yaml
data:
  train_path: "data/imagenet_subset_50gb.tar"  # Tracked with DVC
  augmentation: "heavy"  # light, medium, heavy
  
model:
  architecture: "ResNet50"  # ResNet18, ResNet50, VGG16, EfficientNetB0
  pretrained: true
  fine_tune_layers: 10
  
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001

storage:
  dvc_remote: "s3://company-cv-data"
  cache_dir: "/mnt/fast-storage/dvc-cache"  # Local SSD for speed`

**DVC strategy:**

Track the 50GB dataset once with `dvc add`, push to S3
Track trained models (500MB each) with `dvc add models/resnet50_v1.h5`
Use 

---

### Self-Assessment Checklist

You've mastered this topic if you can:

- Set up MLflow tracking for a new project and log parameters, metrics, and models without referring to documentation
- Configure a DVC remote (local or cloud), push data to it, and explain to a teammate how to pull
- Create a YAML config file for an ML project and refactor hardcoded parameters to read from it
- Write a simple pre-commit hook that validates project structure or runs fast tests
- Explain to a non-technical stakeholder why reproducibility matters and how these tools provide it
- Debug common issues: "DVC cache not found", "MLflow can't find experiment", "config changes don't affect training"

**If you checked fewer than 5 boxes:** Start by reviewing **Example 1** (MLflow basics) and **Example 3** (complete integration). Then re-read **Pitfall 2** (DVC remote push/pull) and **Pitfall 3** (config usage). Try the practice task step-by-step, testing each component (MLflow, config, DVC, hooks) individually before combining them. If MLflow is confusing, practice logging a few simple experiments without the complexity of configs. If DVC is unclear, practice `dvc add`, `dvc push`, `dvc pull` with a small test file before tackling large datasets.

---

## 6. Consolidation: Key Takeaways & Next Steps

### The Essential Ideas

**Core concept recap:**

- **MLflow tracks experiments systematically:** Automatically logs parameters, metrics, and models, providing searchable history and eliminating manual tracking errors
- **DVC remotes enable team data sharing:** External storage for large files keeps Git fast while ensuring everyone accesses the same data versions
- **Config files centralize parameter management:** Separating settings from code enables experimentation without code changes, environment-specific configs, and clear documentation
- **CI hooks enforce quality automatically:** Pre-commit validation catches errors early; CI pipelines run comprehensive tests without blocking developers

### Mental Model Check

By now, you should think of reproducible pipelines as: A well-organized lab where every experiment is automatically logged in a searchable notebook (MLflow), all materials are stored in a central supply room accessible to everyone (DVC remotes), equipment settings are documented on standardized forms (config files), and quality checks happen before anything enters the official record (CI hooks). No experiment is lost, no result is unreproducible, and new team members can replicate any past work.

### What You Can Now Do

You can now build ML systems that are truly reproducible and collaborative rather than fragile and isolated. You understand how to prevent the "it works on my machine" problem, how to make experiments searchable and comparable, how to share data efficiently across teams, and how to maintain code quality automatically. These are the hallmarks of professional ML engineering—skills that distinguish production-ready systems from prototype scripts.

### Next Steps

**To deepen this knowledge:**

- Build a multi-month project using all four components, experiencing how they help as the project grows in complexity
- Explore MLflow's model registry for managing model lifecycles (staging → production transitions)
- Set up a shared MLflow tracking server for your team (on AWS, GCP, or locally)
- Implement more sophisticated DVC pipelines with multiple stages and dependencies

**To build on this:**

- Learn about advanced CI/CD patterns: automated model retraining, blue-green deployments, A/B testing frameworks
- Study model monitoring and drift detection (ensuring production models stay performant)
- Explore experiment orchestration tools like Airflow, Prefect, or Kubeflow for managing complex pipelines
- Investigate feature stores (Feast, Tecton) for managing and sharing features across teams

**Additional resources:**

- MLflow documentation: mlflow.org/docs/latest/index.html (comprehensive tutorials and API reference)
- DVC documentation: dvc.org/doc (step-by-step guides for remotes, pipelines, and integration)
- Pre-commit framework: pre-commit.com (examples of hooks for various languages and tools)

---

## Quick Reference Card

Tool | Purpose | Key Commands
MLflow | Track experiments (params, metrics, models) | mlflow.start_run(),mlflow.log_param(),mlflow.log_metric(),mlflow.sklearn.log_model(),mlflow ui
DVC Remote | Share large files across team | dvc remote add -d name url,dvc push,dvc pull
Config Files | Centralize parameters outside code | Use YAML/JSON, read withyaml.safe_load(), log to MLflow as artifact
CI Hooks | Automate quality checks | pre-commit install, define in.pre-commit-config.yaml,pre-commit run --all-files

**Typical Workflow:**

```bash
# 1. Setup (one-time)
mlflow ui &  # Start MLflow UI in background
dvc remote add -d myremote s3://bucket/path
pre-commit install

# 2. Development cycle
# Edit config.yaml (change hyperparameters)
git add config.yaml
git commit -m "Adjust hyperparameters"  # Hooks run automatically

# 3. Training
python train.py  # Reads config, logs to MLflow, saves model

# 4. Share results
dvc add models/trained_model.pkl
git add models/trained_model.pkl.dvc
git commit -m "Add trained model"
dvc push  # Upload model to remote
git push  # Share .dvc file

# 5. Teammate reproduces
git pull
dvc pull  # Downloads model from remote
mlflow ui  # Views your experiments

```

**Config File Template:**

```yaml
data:
  train_path: "data/train.csv"
  test_split: 0.2
  
model:
  type: "RandomForest"
  n_estimators: 100
  max_depth: 15
  
mlflow:
  experiment_name: "my_experiment"
  tracking_uri: "file:./mlruns"

```

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