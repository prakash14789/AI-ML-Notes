# 18. Lecture Notes - Experiment Tracking - Varun Raste - 20 Nov 2025

## [In-class notes](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/dc240af3-adbc-4ad4-8f4e-1f9de3bbdfef/uqqrUzAvBfNZTdyc.zip)

# Experiment Tracking with Weights & Biases

**What you'll be able to do:**

- Set up Weights & Biases (W&B) to automatically log experiments, hyperparameters, and metrics during model training
- Create comparative dashboards to visualize and analyze multiple experiments side-by-side
- Generate professional reports that document findings, model performance, and insights for stakeholders

---

## 1. Introduction: What Is Experiment Tracking and Why Should You Care?

### Core Definition

**Experiment tracking** is the systematic recording of all information related to machine learning experiments: hyperparameters, metrics (accuracy, loss), training curves, model artifacts, dataset versions, code versions, and environment details. Unlike manually copying results into spreadsheets, automated tracking tools like Weights & Biases capture this information in real-time, making experiments reproducible and comparable. This creates a searchable history of what you've tried, what worked, and why.

### A Simple Analogy

Think of experiment tracking like a lab notebook in chemistry: Scientists meticulously record every experiment—exact measurements, procedures, observations, and results—so they can reproduce successful experiments and avoid repeating failures. Without this notebook, they'd waste months re-discovering what didn't work. Machine learning experiment tracking serves the same purpose but automates the recording process, capturing thousands of data points per experiment that would be impossible to log manually.

**Limitation:** This analogy works for understanding the documentation aspect, but breaks down because ML experiment tracking is interactive and collaborative—multiple team members can view live experiments, filter results, and build on each other's work in real-time.

### Why This Matters to You

**Problem it solves:** Without experiment tracking, teams waste 30-50% of their time on "research debt"—forgetting what they tried last month, re-running failed experiments, unable to reproduce published results, and spending hours creating comparison charts in spreadsheets. When you finally achieve good results, you can't remember which exact hyperparameters produced them. When models fail in production, you can't trace back to the training conditions that created them.

**What you'll gain:**

- **Zero manual logging:** Replace hours of copying metrics into spreadsheets with automatic logging—just add 3 lines of code to your training script
- **Instant comparisons:** Visualize 100+ experiments simultaneously to identify patterns (e.g., "all successful runs used learning_rate < 0.01")
- **Team collaboration:** Share live experiment dashboards with colleagues who can filter, analyze, and build on your work without asking for data
- **Reproducibility guarantee:** Every experiment is tagged with code version, dataset version, and exact environment, making any result reproducible months later

**Real-world context:** DeepMind uses experiment tracking to manage thousands of daily reinforcement learning experiments. Tesla's Autopilot team tracks millions of model training runs to ensure production models are traceable. Kaggle winners use W&B to systematically explore hyperparameter spaces, often crediting it for finding the final 1-2% improvement that secured victory.

---

## 2. The Foundation: Core Concepts Explained

### Concept A: The Experiment as a Unit

**Definition:** An experiment (or "run" in W&B terminology) is a single execution of your training script with specific hyperparameters, data, and code. Each experiment has a unique ID and logs: (1) **config** (hyperparameters, dataset version), (2) **metrics** (loss, accuracy over time), (3) **system metrics** (GPU usage, memory), and (4) **artifacts** (saved models, visualizations). The experiment is the atomic unit of comparison—you compare runs to find what works.

**Key characteristics:**

- **Immutable record:** Once logged, experiment data can't be changed (prevents tampering)
- **Automatic capture:** Logs continuously during training without manual intervention
- **Searchable metadata:** Tag experiments with notes, dataset names, or model architecture for later filtering

**A concrete example:**

```python
import wandb

# Initialize experiment
wandb.init(project="fraud-detection", name="experiment-001")

# Log hyperparameters
wandb.config.learning_rate = 0.001
wandb.config.batch_size = 32
wandb.config.architecture = "ResNet50"

# Training loop
for epoch in range(100):
    train_loss = train_one_epoch()
    val_accuracy = validate()
    
    # Log metrics automatically
    wandb.log({"train_loss": train_loss, "val_accuracy": val_accuracy})

wandb.finish()

```

After running this script, W&B captures everything: the exact hyperparameters, how loss decreased over 100 epochs, peak GPU memory usage, and the trained model—all without manual spreadsheet entry.

**Common confusion:** Beginners think each epoch is an experiment. Actually, one experiment encompasses the entire training run (all epochs). Each `wandb.log()` call adds a data point to that single experiment's timeline.

---

### Concept B: Comparative Dashboards

**Definition:** A comparative dashboard is an interactive visualization that displays multiple experiments simultaneously, enabling pattern recognition across runs. Unlike static charts, dashboards let you filter experiments (e.g., "show only runs with learning_rate < 0.01"), compare metrics side-by-side, and identify which hyperparameters correlate with success. Dashboards update in real-time as experiments run.

**How it relates to Individual Experiments:** While a single experiment shows "what happened," comparative dashboards answer "what works." You might have 50 experiments with varying learning rates and batch sizes—the dashboard reveals that learning_rate=0.001 consistently outperforms 0.01, regardless of batch size.

**Key characteristics:**

- **Multi-run visualization:** Display 10-1000+ experiments on the same axes
- **Dynamic filtering:** Show/hide experiments based on criteria ("only show runs with val_accuracy > 0.85")
- **Correlation analysis:** Parallel coordinates plots reveal which hyperparameter combinations produce best results
- **Live updates:** Dashboards refresh as experiments progress, enabling mid-training decisions

**A concrete example:**

Imagine training 50 models with different learning rates and architectures. The dashboard shows:

- Line chart: All 50 validation accuracy curves overlaid
- Scatter plot: Final accuracy vs. learning rate (reveals optimal range)
- Parallel coordinates: Lines connecting hyperparameters to final score (shows architecture matters more than learning rate)

You instantly see that ResNet50 with learning_rate=0.001 dominates—information that would take hours to extract from spreadsheets.

**Remember:** This extends the single-experiment view. Just as one training curve shows progress, the dashboard shows patterns across all your work, turning isolated experiments into systematic knowledge.

---

### Concept C: Reports for Communication

**Definition:** A report is a shareable document that combines visualizations, tables, and narrative text to communicate experiment findings to stakeholders (managers, teammates, clients). Reports pull data directly from experiments—when you update experiments, report charts auto-update. This makes reports "living documents" rather than static PowerPoints that become outdated instantly.

**How it relates to Dashboards:** Dashboards are for exploration ("let me filter experiments to understand what's happening"). Reports are for presentation ("here's what I discovered and why it matters"). Reports select specific insights from dashboards and add context, conclusions, and recommendations.

**Key characteristics:**

- **Narrative structure:** Combine markdown text with embedded charts to tell a story
- **Selective presentation:** Show 5 key experiments out of 100 total, highlighting what stakeholders need to know
- **Auto-updating:** Charts pull live data; if you re-run an experiment, the report updates automatically
- **Shareable links:** Send URL to anyone (even non-technical stakeholders) without installing software

**A concrete example:**

After running 50 experiments, you create a report titled "Fraud Detection Model Selection - Q4 2024":

- Section 1: "Objective" (text explaining the goal)
- Section 2: "Experiments Overview" (table showing 5 best models)
- Section 3: "Learning Rate Impact" (embedded chart from dashboard)
- Section 4: "Recommendation" (text: "Deploy ResNet50 with lr=0.001")
- Section 5: "Next Steps" (text outlining production plan)

Your manager opens the link, sees clear visualizations with your recommendations, and approves deployment—no 30-slide PowerPoint needed.

---

### How These Concepts Work Together

In a complete workflow: (1) You run experiments, each logging hyperparameters and metrics automatically. (2) You use comparative dashboards to explore these experiments, filtering and visualizing to identify patterns. (3) Once you understand what works, you create a report that distills key findings into a narrative for stakeholders. The report stays connected to live data, so if you run follow-up experiments, you just update the report rather than rebuilding it from scratch.

---

## 3. Seeing It in Action: Setting Up Weights & Biases

### Example 1: Basic Experiment Logging (Classification)

**Scenario:** You're training a simple neural network for image classification and want to track training progress without manually recording metrics.

**Our approach:** Add W&B initialization and logging calls to your existing training code—just 5 lines total.

**Step-by-step solution:**

```python
import wandb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Step 1: Initialize W&B (logs into your account)
wandb.init(
    project="image-classifier",  # Project name (groups related experiments)
    name="random-forest-baseline",  # Experiment name
    tags=["baseline", "random-forest"]  # Tags for filtering
)

# Step 2: Log hyperparameters
wandb.config.update({
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "dataset": "synthetic-v1"
})

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(
    n_estimators=wandb.config.n_estimators,
    max_depth=wandb.config.max_depth,
    min_samples_split=wandb.config.min_samples_split
)
model.fit(X_train, y_train)

# Step 3: Log metrics
y_pred = model.predict(X_test)
wandb.log({
    "test_accuracy": accuracy_score(y_test, y_pred),
    "test_f1": f1_score(y_test, y_pred)
})

# Step 4: Log the model artifact
wandb.log_model(path="./model.pkl", name="rf_classifier")

# Step 5: Finish the run
wandb.finish()

print("✓ Experiment logged to W&B!")

```

**Output:**

```
wandb: Run summary:
wandb:   test_accuracy: 0.875
wandb:   test_f1: 0.862
wandb: View run at https://wandb.ai/your-username/image-classifier/runs/abc123

```

**What just happened:** W&B captured your hyperparameters (n_estimators=100, max_depth=10), final metrics (accuracy=0.875), the trained model file, and created a unique URL where you can view everything. No spreadsheets needed.

**Check your understanding:** Why do we log hyperparameters using `wandb.config` instead of just logging them as regular metrics?

Answer
Hyperparameters are configuration inputs that define the experiment, while metrics are outputs that measure performance. W&B treats them differently: config values appear in the "Overview" tab and can be used for filtering/grouping experiments, while metrics appear in charts. This separation lets you answer questions like "show me all experiments where learning_rate=0.001" efficiently.

---

### Example 2: Tracking Training Progress (Iterative Logging)

**Scenario:** Training a model over 50 epochs. You want to see training/validation curves update in real-time to detect overfitting early.

**What's different:** Instead of logging once at the end, we log metrics after each epoch, creating time-series charts.

**Solution:**

```python
import wandb
import numpy as np

wandb.init(project="timeseries-forecasting", name="lstm-50epochs")

wandb.config.learning_rate = 0.001
wandb.config.batch_size = 32
wandb.config.epochs = 50

# Simulated training loop
for epoch in range(wandb.config.epochs):
    # Simulated training
    train_loss = 1.0 / (epoch + 1) + np.random.rand() * 0.1
    val_loss = 1.2 / (epoch + 1) + np.random.rand() * 0.15
    train_acc = min(0.95, 0.5 + epoch * 0.01)
    val_acc = min(0.90, 0.45 + epoch * 0.009)
    
    # Log metrics for this epoch
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc
    })
    
    print(f"Epoch {epoch}: train_loss={train_loss:.3f}, val_acc={val_acc:.3f}")

wandb.finish()

```

**Key lesson:** Each `wandb.log()` call adds a data point to your experiment's timeline. W&B automatically creates line charts showing how train_loss and val_loss evolve over epochs. You can watch these charts update live in your browser while training runs—if validation loss starts increasing while training loss decreases, you know you're overfitting and can stop training early.

---

### Example 3: Hyperparameter Sweep Integration

**Scenario:** Running 50 experiments with different hyperparameters using Optuna or grid search. You want all experiments grouped in one project.

**The approach:** Initialize W&B inside your objective function so each trial becomes a separate run.

```python
import wandb
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

def objective(trial):
    # Initialize W&B for this trial
    run = wandb.init(
        project="hyperparameter-sweep",
        name=f"trial-{trial.number}",
        reinit=True  # Allows multiple inits in same script
    )
    
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10)
    }
    
    # Log config to W&B
    wandb.config.update(params)
    
    # Train and evaluate
    model = GradientBoostingClassifier(**params)
    score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
    
    # Log result
    wandb.log({"cv_accuracy": score})
    
    run.finish()
    return score

# Run Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print(f"Best accuracy: {study.best_value:.4f}")
print("All 20 experiments logged to W&B!")

```

**Why this approach:** Each Optuna trial becomes a separate W&B run in the same project. You can now use W&B's dashboard to visualize all 20 trials, identify which hyperparameter ranges work best, and compare results—something Optuna's built-in visualization doesn't handle as elegantly.

---

## 4. Creating Comparative Dashboards

### What you'll learn

- How to filter and group experiments to focus on specific comparisons
- How to create custom visualizations (scatter plots, parallel coordinates, tables)
- How to identify patterns across dozens of experiments

### Building Your First Comparison

**Scenario:** You've run 30 experiments with varying learning rates and batch sizes. You want to identify which combination produces the best validation accuracy.

**Step 1: Access the dashboard**

- Go to your W&B project page
- Click "Workspace" → "Create new view"

**Step 2: Add comparison charts**

```python
# Your experiments are already logged; now configure the dashboard:

# 1. Line Chart: Validation accuracy over time for all runs
#    - X-axis: Step/Epoch
#    - Y-axis: val_accuracy
#    - Group by: learning_rate (creates separate lines per LR)

# 2. Scatter Plot: Final accuracy vs. learning rate
#    - X-axis: config.learning_rate
#    - Y-axis: val_accuracy (max value)
#    - Color by: config.batch_size

# 3. Parallel Coordinates: Multi-dimensional view
#    - Columns: learning_rate, batch_size, max_depth, val_accuracy
#    - This shows which hyperparameter combinations lead to high accuracy

```

**Dashboard Configuration Example:**

In the W&B UI, you can create these views without code:

1. 
**Table View:** Lists all runs with their config and final metrics

Columns: Run name, learning_rate, batch_size, val_accuracy, train_time
Sort by: val_accuracy (descending)
Quick insight: "Top 5 runs all have learning_rate between 0.001-0.003"

2. 
**Custom Chart:** Learning rate vs. accuracy

Drag learning_rate to X-axis
Drag val_accuracy to Y-axis
Reveals optimal learning rate range visually

**Filtering experiments:**

```python
# In W&B UI, create filters:
# - Show only: config.learning_rate < 0.01
# - Show only: tags contains "production-candidate"
# - Show only: val_accuracy > 0.85

# This narrows 100 experiments to the 12 best ones instantly

```

**Interpretation:** After filtering, you see that learning_rate=0.002 with batch_size=64 consistently achieves val_accuracy > 0.90. This pattern would be invisible in spreadsheets but jumps out in the dashboard.

---

### Advanced: Parallel Coordinates

**What it shows:** Parallel coordinates display multiple dimensions simultaneously, revealing correlations.

**Example interpretation:**

```
learning_rate  |  batch_size  |  dropout  |  val_accuracy
0.001 ---------|----------32------------|----0.2----|------0.92
0.01 ----------|----------64------------|----0.3----|------0.78
0.001 ---------|----------64------------|----0.2----|------0.94

```

Lines that end at high val_accuracy share similar patterns in other columns—you can see "lr=0.001 + batch=64 + dropout=0.2 → good results" visually.

**Key Takeaways:**

- Dashboards turn 100+ experiments into actionable insights
- Filtering lets you focus on relevant subsets
- Multiple visualization types reveal different patterns
- Live updates mean you can make decisions mid-training

---

## 5. Creating Professional Reports

### What you'll learn

- How to combine charts, tables, and text into a cohesive narrative
- How to select specific experiments to highlight
- How to share findings with non-technical stakeholders

### Building a Report

**Scenario:** After running 50 experiments, you need to present findings to your manager who wants to know: "Which model should we deploy?"

**Step 1: Create a new report**

In W&B:

1. Go to your project
2. Click "Reports" → "Create report"
3. Title: "Fraud Detection Model Selection - Q4 2024"

**Step 2: Structure your report**

```markdown
# Fraud Detection Model Selection - Q4 2024

## Executive Summary
After testing 50 model configurations, we recommend deploying **ResNet50 with learning_rate=0.001**, 
which achieves 94.2% accuracy on validation data—exceeding our 90% target.

## Methodology
- **Dataset:** 100K transactions (2% fraud rate)
- **Models tested:** ResNet50, VGG16, MobileNetV2
- **Hyperparameters:** learning_rate (0.0001-0.1), batch_size (16-128)
- **Evaluation:** 5-fold cross-validation

## Key Findings

### Finding 1: Architecture matters more than learning rate
[Insert embedded parallel coordinates chart from dashboard]

The chart above shows ResNet50 consistently outperforms other architectures regardless of learning rate.

### Finding 2: Optimal learning rate = 0.001
[Insert embedded scatter plot: learning_rate vs val_accuracy]

Performance peaks at lr=0.001. Higher learning rates cause instability; lower rates train too slowly.

### Finding 3: Best model details
[Insert embedded table showing top 3 runs with hyperparameters and metrics]

| Model | Learning Rate | Batch Size | Val Accuracy | Train Time |
|-------|---------------|------------|--------------|------------|
| ResNet50 | 0.001 | 64 | 94.2% | 45 min |
| ResNet50 | 0.002 | 32 | 93.8% | 38 min |
| VGG16 | 0.001 | 64 | 91.5% | 52 min |

## Recommendation
Deploy ResNet50 (learning_rate=0.001, batch_size=64). Expected production accuracy: 93-94%.

## Next Steps
1. Run final training on full dataset
2. Deploy to staging environment
3. Monitor for one week before production rollout

```

**Step 3: Embed visualizations**

In the W&B report editor:

- Click "Add panel" → "Chart"
- Select existing dashboard chart to embed
- Chart auto-updates if you run more experiments

**Step 4: Share**

- Click "Share" → Copy link
- Anyone with link can view (no W&B account needed)
- Stakeholders see live data without asking you for updates

**Why this approach works:** Reports combine data (charts auto-pull from experiments) with narrative (your interpretation and recommendations). When you run follow-up experiments, you can add them to the report instantly—no rebuilding slide decks.

---

## 6. Common Pitfalls

**Mistake 1: Forgetting to call `wandb.finish()`**

```python
# WRONG: Run never officially completes
wandb.init(project="my-project")
# ... training code ...
# Script ends without wandb.finish()

```

**Why it's a problem:** The run appears "crashed" in W&B even though it completed successfully. Final metrics might not be saved.

**The right approach:**

```python
wandb.init(project="my-project")
try:
    # ... training code ...
    wandb.log({"final_metric": value})
finally:
    wandb.finish()  # Always called, even if training crashes

```

---

**Mistake 2: Logging too frequently in tight loops**

```python
# WRONG: Logging every batch (thousands of times)
for batch in dataloader:
    loss = train_batch(batch)
    wandb.log({"batch_loss": loss})  # Creates huge overhead

```

**Why it's a problem:** Logging every batch creates thousands of data points, slowing training and making charts unreadable.

**The right approach:**

```python
# Log every N batches or once per epoch
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        loss = train_batch(batch)
        
        if batch_idx % 100 == 0:  # Log every 100 batches
            wandb.log({"batch_loss": loss})

```

---

**Mistake 3: Not using consistent naming conventions**

```python
# WRONG: Inconsistent naming across experiments
wandb.init(project="my-proj", name="experiment_1")
wandb.init(project="my-proj", name="testing")
wandb.init(project="my-proj", name="final_model_v2")

```

**Why it's a problem:** Impossible to sort or filter experiments meaningfully.

**The right approach:**

```python
# Use consistent naming: <model>-<variant>-<date>
wandb.init(project="my-proj", name="resnet50-lr001-20241106")
wandb.init(project="my-proj", name="resnet50-lr002-20241106")
wandb.init(project="my-proj", name="vgg16-lr001-20241107")

# And use tags for filtering
wandb.init(project="my-proj", tags=["resnet50", "production-candidate"])

```

---

## 7. Your Turn: Practice Task

**Challenge:** Track a complete hyperparameter sweep and create a report.

**Specifications:**

1. 
**Run 20 experiments:**

Use `GradientBoostingClassifier`
Vary `n_estimators` (50-200), `learning_rate` (0.01-0.3), `max_depth` (3-10)
Log each experiment to W&B project "practice-sweep"

2. 
**Create a comparative dashboard:**

Scatter plot: learning_rate vs. accuracy
Table: Top 5 runs sorted by accuracy
Parallel coordinates: All 3 hyperparameters + accuracy

3. 
**Generate a report:**

Title: "Hyperparameter Search Results"
Include: Executive summary, embedded scatter plot, top 3 models table, recommendation

**Hint:** Initialize W&B inside your training loop. Use `wandb.config.update()` for hyperparameters and `wandb.log()` for metrics. Access the W&B web interface to create dashboards and reports—no additional code needed for visualization.

---

## 8. Key Takeaways

- **Automated logging saves hours:** Add 5 lines to your training script; W&B captures everything automatically
- **Dashboards reveal patterns:** Visualize 100+ experiments simultaneously to identify what works
- **Reports communicate findings:** Combine charts and narrative for stakeholders; reports auto-update with new experiments
- **Reproducibility guaranteed:** Every experiment tagged with code version, hyperparameters, and environment
- **Log smart, not excessively:** Log per epoch, not per batch; use consistent naming conventions

### Mental Model Check

By now, you should think of experiment tracking as: **An automated lab notebook that captures every detail of your ML experiments without manual effort, enables instant comparison of dozens of approaches through interactive dashboards, and lets you communicate findings through living reports that stay current as your work progresses.**

---

## Quick Reference

**Basic Setup:**

```python
import wandb

# Start experiment
wandb.init(project="project-name", name="exp-name", tags=["tag1"])

# Log config
wandb.config.learning_rate = 0.001

# Log metrics
wandb.log({"accuracy": 0.95, "loss": 0.12})

# Finish
wandb.finish()

```

**Best Practices:**

- Log once per epoch, not per batch
- Use descriptive names: "resnet50-lr001-20241106"
- Tag experiments for filtering: ["baseline", "production"]
- Always call `wandb.finish()` (use try/finally)

**Dashboard Tips:**

- Filter: Show runs where config.lr < 0.01
- Group: Color lines by architecture
- Sort table: By val_accuracy descending

**Report Structure:**

1. Executive summary
2. Methodology
3. Key findings (with embedded charts)
4. Recommendation
5. Next steps

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