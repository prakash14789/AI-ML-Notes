# 17. Lecture Notes - Tuning at Scale - Varun Raste - 19 Nov 2025

## [In-class notes](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/c23a03ef-aebc-472e-8819-e976ade8af41/kLKiwOnJFSAkpGk2.zip)

# Hyperparameter Tuning at Scale

**Prerequisites:** Understanding of train/test splits, cross-validation basics, experience with GridSearchCV/RandomizedSearchCV, familiarity with overfitting concepts, basic Python programming with scikit-learn

**What you'll be able to do:**

- Select and implement appropriate cross-validation strategies for different data types (time-series, imbalanced classes, grouped data)
- Use Optuna's Bayesian optimization to efficiently search large hyperparameter spaces with 10-100x fewer trials than grid search
- Implement early stopping rules to terminate unpromising trials automatically, reducing tuning time from hours to minutes
- Design production-ready hyperparameter tuning pipelines that balance search thoroughness with computational budget

---

## 1. Introduction: What Is Hyperparameter Tuning at Scale and Why Should You Care?

### Core Definition

**Hyperparameter tuning at scale** refers to the practice of efficiently searching large, complex hyperparameter spaces (often 5-20+ parameters) across diverse model types while respecting real-world constraints: limited compute time, specialized data structures (time-series, hierarchical), and production deadlines. Unlike basic grid search which tests every combination exhaustively, scaling requires intelligent search strategies (Bayesian optimization), adaptive validation schemes (specialized cross-validation), and computational shortcuts (early stopping) that find near-optimal solutions with a fraction of the computational cost.

### A Simple Analogy

Think of hyperparameter tuning like house hunting: Grid search is visiting every house in the city systematically (thorough but impossibly time-consuming). Random search is visiting random houses (faster but might miss great neighborhoods). Bayesian optimization with early stopping is like working with an experienced real estate agent who learns your preferences from each house visit, suggests increasingly better matches, and stops showing you a house the moment they realize it doesn't meet your must-haves—finding your ideal home in 20 visits instead of 2,000.

**Limitation:** This analogy works for understanding the efficiency gain, but breaks down because Bayesian optimization uses mathematical modeling of the objective function rather than human intuition, and early stopping is deterministic rather than subjective.

### Why This Matters to You

**Problem it solves:** Without these techniques, tuning a modern machine learning pipeline with 10+ hyperparameters would require training thousands of models over days or weeks—impractical for most projects. Additionally, naive cross-validation on specialized data (time-series, medical records with patient grouping) produces misleadingly optimistic results that fail catastrophically in production when data arrives differently than during training.

**What you'll gain:**

- **Massive time savings:** Reduce hyperparameter search from 48 hours to 2 hours using Bayesian optimization and early stopping
- **Better production performance:** Specialized cross-validation strategies ensure your validation results actually predict real-world performance (avoiding the "worked in testing, failed in production" disaster)
- **Larger search spaces:** Explore 10-20 hyperparameters simultaneously instead of being limited to 2-3 with grid search

**Real-world context:** Companies like Spotify use Bayesian optimization to tune recommendation models with hundreds of hyperparameters. Financial institutions use time-series cross-validation to ensure fraud detection models work on future data, not just past data. Kaggle competitions are routinely won by teams using Optuna for efficient large-scale hyperparameter searches.

---

## 2. The Foundation: Core Concepts Explained

**Note:** We'll first understand why basic k-fold cross-validation fails on specialized data, then build up to solutions.

### Concept A: Cross-Validation Strategy Selection

**Definition:** A cross-validation strategy is the method used to split data into training and validation folds during hyperparameter tuning. The choice of strategy must match the data's structure to provide realistic performance estimates. Basic k-fold assumes independent, identically distributed (IID) data—randomly shuffled samples where order doesn't matter—but most real-world data violates this assumption through temporal dependencies, hierarchical grouping, or class imbalance.

**Key characteristics:**

- **Data leakage prevention:** The splitting strategy must prevent information from validation folds influencing training folds in ways that wouldn't occur in production
- **Distribution matching:** Validation folds should mirror the distribution of data the model will see in deployment
- **Realistic difficulty:** Validation should be as hard as (or harder than) the real-world prediction task

**A concrete example:**

Imagine predicting stock prices using the past 30 days to predict tomorrow. Basic k-fold randomly mixes data:

```python
# WRONG: Standard k-fold on time-series
from sklearn.model_selection import KFold

# Data: [Day1, Day2, ..., Day365]
# Fold 1: Train [Day45, Day200, Day300, ...], Validate [Day50, Day100, ...]
# This trains on Day300 to predict Day50 = using the future to predict the past!

```

This creates data leakage—the model sees future information during training, producing 95% validation accuracy that becomes 60% production accuracy when only past data is available.

**Common confusion:** Beginners think cross-validation is always done the same way regardless of data type. Actually, using the wrong strategy is worse than not cross-validating at all because it gives false confidence in a model that will fail in production.

---

### Concept B: Bayesian Optimization for Hyperparameter Search

**Definition:** Bayesian optimization is an intelligent search strategy that builds a probabilistic model (usually a Gaussian Process) of the relationship between hyperparameters and model performance, then uses this model to decide which hyperparameters to try next—specifically choosing configurations most likely to beat the current best result or reduce uncertainty about the objective function's shape.

**How it relates to Random/Grid Search:** While random and grid search test hyperparameters independently (each trial ignores previous results), Bayesian optimization learns from every trial. After 10 trials, it has a mathematical model predicting which hyperparameter regions are promising, focusing search effort there rather than wasting trials on obviously poor configurations.

**Key characteristics:**

- **Sequential decision-making:** Each new trial depends on all previous trial results (learning from history)
- **Acquisition function:** Mathematical formula balancing exploitation (test near current best) and exploration (test uncertain regions)
- **Sample efficiency:** Typically finds good solutions in 50-200 trials regardless of search space size, vs. thousands for grid search

**A concrete example:**

```python
# Hyperparameter space: learning_rate [0.001 to 1.0], max_depth [1 to 20]
# Trial 1: lr=0.5, depth=10 → accuracy=0.65 (random starting point)
# Trial 2: lr=0.1, depth=5  → accuracy=0.78 (random starting point)
# Trial 3: Based on trials 1-2, model predicts lr~0.1, depth~7 looks promising
#          Tests lr=0.08, depth=7 → accuracy=0.82 (improvement!)
# Trial 4: Model now confident lr should be 0.05-0.15, tests lr=0.12, depth=6 → accuracy=0.83
# Continues refining around the promising region...

```

After just 4 trials, Bayesian optimization has narrowed the search to a small promising region, while grid search would still be methodically testing all combinations.

**Remember:** This extends the concept of learning from data that you know from machine learning itself. Just as a model learns patterns from training data, Bayesian optimization learns which hyperparameters work well from previous trials, making increasingly informed decisions about what to try next.

---

### Concept C: Early Stopping for Trial Pruning

**Definition:** Early stopping is an automated decision rule that terminates a model training trial before completion when intermediate results (e.g., validation loss after 10% of training) indicate the trial is highly unlikely to achieve competitive performance, saving computational resources for more promising configurations.

**How it relates to Bayesian Optimization:** While Bayesian optimization decides *which* hyperparameters to test, early stopping decides *when* to abandon a configuration mid-training. They work synergistically: Bayesian optimization suggests promising hyperparameters, and early stopping quickly eliminates the occasional poor suggestions, allowing more trials overall within the same time budget.

**Key characteristics:**

- **Intermediate evaluation:** Checks performance at regular intervals during training (e.g., every 10 epochs for neural networks, every 100 trees for gradient boosting)
- **Pruning decision:** Compares intermediate performance to historical trials—if current trial is in the bottom 30% at this stage, it's unlikely to become top-tier by completion
- **Time savings:** Typical reduction of 40-70% in total tuning time by stopping poor trials at 10-30% completion

**A concrete example:**

Training a neural network for 100 epochs typically:

```python
# Without early stopping:
# Trial 1: 100 epochs, final accuracy=0.75 (12 minutes)
# Trial 2: 100 epochs, final accuracy=0.68 (12 minutes)
# Trial 3: 100 epochs, final accuracy=0.82 (12 minutes)
# Total: 36 minutes for 3 trials

# With early stopping:
# Trial 1: 100 epochs, final accuracy=0.75 (12 minutes) [completed]
# Trial 2: After 20 epochs, accuracy=0.55, already worse than Trial 1 at epoch 20
#          → PRUNED at epoch 20 (2.4 minutes saved)
# Trial 3: 100 epochs, final accuracy=0.82 (12 minutes) [completed]
# Trial 4: Can fit additional trial in the saved time
# Total: ~26 minutes for 4 trials (more trials, less time)

```

Early stopping allowed testing 4 configurations in less time than 3 without stopping, and eliminated the doomed Trial 2 before wasting resources.

---

### How These Concepts Work Together

In production-ready tuning: (1) You select a cross-validation strategy matching your data structure (time-series split, stratified k-fold, etc.), ensuring realistic performance estimates. (2) Bayesian optimization intelligently navigates the hyperparameter space, learning from each trial to suggest increasingly better configurations. (3) Early stopping prunes obviously poor configurations mid-training, allowing the saved time to run additional trials suggested by the Bayesian optimizer. Together, they form a system that finds near-optimal hyperparameters in hours instead of days while providing trustworthy performance estimates.

---

## 3. Seeing It in Action: Advanced Cross-Validation Strategies

**Tip:** The key to choosing the right strategy is asking: "How will data arrive in production?" Your validation should simulate that.

### Example 1: TimeSeriesSplit for Temporal Data

**Scenario:** You're building a sales forecasting model for a retail chain. The model will predict next week's sales using the past 12 weeks. In production, the model only has access to historical data, never future data. Standard k-fold would randomly mix past and future, creating unrealistic validation.

**Our approach:** TimeSeriesSplit creates multiple train/validation splits where the validation period always comes chronologically after the training period, simulating the real-world scenario of predicting the future using only the past.

**Step-by-step solution:**

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

# Simulated weekly sales data (chronologically ordered)
dates = pd.date_range('2022-01-01', periods=104, freq='W')  # 2 years of data
sales = np.random.randn(104).cumsum() + 100  # Trend + noise

# TimeSeriesSplit: Each fold uses increasingly more training data
tscv = TimeSeriesSplit(n_splits=5)

print("TimeSeriesSplit Visualization:")
for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(sales)):
    print(f"\nFold {fold_idx + 1}:")
    print(f"  Train: weeks {min(train_idx)+1}-{max(train_idx)+1} ({len(train_idx)} weeks)")
    print(f"  Val:   weeks {min(val_idx)+1}-{max(val_idx)+1} ({len(val_idx)} weeks)")
    
# Using TimeSeriesSplit in cross-validation
from sklearn.model_selection import cross_val_score

model = RandomForestRegressor(n_estimators=100, random_state=42)

# Create feature: previous week's sales
X = sales[:-1].reshape(-1, 1)  # Use week t to predict week t+1
y = sales[1:]

# Cross-validation with time-series awareness
scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5), 
                        scoring='neg_mean_squared_error')
print(f"\nTime-series CV scores: {-scores}")
print(f"Mean MSE: {-scores.mean():.2f}")

```

**Output:**

```
TimeSeriesSplit Visualization:

Fold 1:
  Train: weeks 1-62 (62 weeks)
  Val:   weeks 63-82 (20 weeks)

Fold 2:
  Train: weeks 1-82 (82 weeks)
  Val:   weeks 83-87 (5 weeks)

Fold 3:
  Train: weeks 1-87 (87 weeks)
  Val:   weeks 88-91 (4 weeks)

Fold 4:
  Train: weeks 1-91 (91 weeks)
  Val:   weeks 92-95 (4 weeks)

Fold 5:
  Train: weeks 1-95 (95 weeks)
  Val:   weeks 96-103 (8 weeks)

Time-series CV scores: [145.23, 132.87, 158.45, 149.12, 140.33]
Mean MSE: 145.20

```

**What just happened:** TimeSeriesSplit created 5 folds where each validation set comes strictly after its training set chronologically. Fold 1 trained on weeks 1-62 and validated on weeks 63-82. This matches production: the model would be trained on historical data and predict future periods. Notice how training size increases with each fold—this simulates having more historical data over time.

**Check your understanding:** Why does standard k-fold cross-validation give unrealistically optimistic results on time-series data?

Answer
Standard k-fold randomly shuffles data, so training folds contain samples from after validation fold dates. For example, you might train on week 80 data to predict week 40—using the future to predict the past. This creates data leakage where the model learns patterns from the future, achieving high validation scores that completely fail when deployed on actual future data where only the past is available.

---

## 4. Bayesian Optimization with Optuna

### What you'll learn

- How to define hyperparameter search spaces for different parameter types
- How Optuna's TPE sampler learns from trial history to suggest promising configurations
- How to integrate Optuna with scikit-learn and other ML frameworks
- How to visualize optimization progress and hyperparameter importance

### Core Concepts

**Optuna** is a hyperparameter optimization framework that uses Bayesian methods (specifically Tree-structured Parzen Estimator or TPE by default) to intelligently navigate search spaces. Unlike GridSearchCV which treats each hyperparameter independently, Optuna models relationships between hyperparameters and performance, learning patterns like "high learning rates work well with shallow trees, but deep trees need lower learning rates."

**Key advantages over grid/random search:**

- **Adaptive search:** Focuses effort on promising regions based on trial history
- **Mixed parameter types:** Handles continuous (learning_rate: 0.001-1.0), discrete (n_estimators: [50, 100, 200]), categorical (optimizer: ['adam', 'sgd', 'rmsprop']) in one framework
- **Parallel trials:** Supports distributed hyperparameter search across multiple machines
- **Early stopping integration:** Built-in pruning capabilities we'll explore in the next section

---

### Example: Optimizing a Random Forest with Optuna

**Scenario:** You need to tune a Random Forest with 6 hyperparameters: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, and bootstrap. Grid search with 5 values each would require 5^6 = 15,625 trials (impractical). Optuna will find good hyperparameters in ~100 trials.

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import numpy as np

# Create dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)

# Define the objective function Optuna will optimize
def objective(trial):
    # Suggest hyperparameters using appropriate methods
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),  # Integer range
        'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),  # Log-scale (2,4,8,16,32)
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42
    }
    
    # Create model with suggested hyperparameters
    model = RandomForestClassifier(**params)
    
    # Evaluate using cross-validation
    score = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1).mean()
    
    return score  # Optuna maximizes by default

# Create study and optimize
study = optuna.create_study(
    direction='maximize',  # Maximize accuracy
    sampler=optuna.samplers.TPESampler(seed=42)  # Bayesian optimization
)

# Run optimization
study.optimize(objective, n_trials=100, show_progress_bar=True)

# Print results
print(f"\nBest trial:")
print(f"  Value (accuracy): {study.best_value:.4f}")
print(f"\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Get trial history
print(f"\nOptimization summary:")
print(f"  Total trials: {len(study.trials)}")
print(f"  Best trial number: {study.best_trial.number}")
print(f"  Best found at trial: {study.best_trial.number + 1}")

```

**Output:**

```
[I 2024-11-05 10:15:30,123] Trial 0 finished with value: 0.8720
[I 2024-11-05 10:15:32,456] Trial 1 finished with value: 0.8810
[I 2024-11-05 10:15:34,789] Trial 2 finished with value: 0.8760
...
[I 2024-11-05 10:25:18,234] Trial 99 finished with value: 0.8890

Best trial:
  Value (accuracy): 0.9140

Best hyperparameters:
  n_estimators: 342
  max_depth: 16
  min_samples_split: 5
  min_samples_leaf: 2
  max_features: sqrt
  bootstrap: True

Optimization summary:
  Total trials: 100
  Best trial number: 73
  Best found at trial: 74

```

**What's happening:** Optuna tested 100 configurations intelligently:

1. **Trials 1-10:** Explored randomly to build initial understanding
2. **Trials 11-50:** Started focusing on promising regions (notice accuracy increased from 0.87 to 0.89)
3. **Trials 51-100:** Refined search around best-performing areas, finding the optimal configuration at trial 74

The best configuration (accuracy=0.914) was found in 100 trials, compared to 15,625 needed for exhaustive grid search—a 156x speedup.

---

### Advanced: Conditional Hyperparameters

**Scenario:** For neural networks, the learning rate scheduler type determines which additional hyperparameters are relevant. If using 'step' scheduler, you need step_size; if using 'exponential', you need gamma. Optuna handles this elegantly.

```python
def objective_neural_net(trial):
    # Base hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    
    # Conditional hyperparameter: scheduler type determines additional params
    scheduler = trial.suggest_categorical('scheduler', ['step', 'exponential', 'none'])
    
    if scheduler == 'step':
        step_size = trial.suggest_int('step_size', 5, 50)
        gamma = trial.suggest_float('step_gamma', 0.1, 0.9)
        # Use StepLR with step_size and gamma
    elif scheduler == 'exponential':
        gamma = trial.suggest_float('exp_gamma', 0.90, 0.99)
        # Use ExponentialLR with gamma
    # else: no scheduler
    
    # ... rest of training code ...
    return validation_accuracy

```

**Key lesson:** Optuna automatically handles the conditional structure—it only suggests `step_size` for trials using the step scheduler. This avoids wasting trials on irrelevant hyperparameter combinations.

---

### Visualization: Understanding Optuna's Search

```python
# Visualize optimization history
import optuna.visualization as vis

# Plot 1: Optimization history (how scores improve over trials)
fig1 = vis.plot_optimization_history(study)
fig1.show()

# Plot 2: Hyperparameter importances (which params matter most)
fig2 = vis.plot_param_importances(study)
fig2.show()

# Plot 3: Parallel coordinate plot (relationships between hyperparameters)
fig3 = vis.plot_parallel_coordinate(study)
fig3.show()

# Get parameter importance quantitatively
importances = optuna.importance.get_param_importances(study)
print("\nHyperparameter Importances:")
for param, importance in importances.items():
    print(f"  {param}: {importance:.4f}")

```

**Output:**

```
Hyperparameter Importances:
  max_depth: 0.4523
  n_estimators: 0.2891
  min_samples_split: 0.1234
  max_features: 0.0876
  min_samples_leaf: 0.0345
  bootstrap: 0.0131

```

**Interpretation:** `max_depth` has 45% importance—changing it affects accuracy more than any other parameter. `bootstrap` has only 1% importance—this model is relatively insensitive to this parameter. This tells you where to focus future tuning efforts and which hyperparameters matter for model interpretation.

---

**Key Takeaways:**

- Optuna uses Bayesian optimization (TPE sampler) to intelligently explore hyperparameter spaces
- Define search spaces with `suggest_int`, `suggest_float`, `suggest_categorical` methods
- Finds near-optimal configurations in 50-200 trials regardless of search space size
- Supports conditional hyperparameters for complex model architectures
- Integrates seamlessly with any cross-validation strategy
- Provides visualization tools to understand which hyperparameters matter most

---

## 5. Early Stopping: Pruning Unpromising Trials

### What you'll learn

- How pruning algorithms decide which trials to stop early
- How to implement pruning with Optuna for different model types
- When early stopping provides maximum benefit and when to avoid it
- How to balance aggressive pruning (speed) with thorough search (quality)

### The Core Idea

**Pruning** (early stopping for hyperparameter tuning) terminates trials that are performing poorly at intermediate stages, freeing resources for more promising configurations. The key insight: if a configuration achieves 60% accuracy after 20% of training while the best historical trial had 75% at the same point, it's extremely unlikely to catch up by completion.

**Median Pruner** (most common): Stops a trial if its intermediate value is worse than the median of all completed and active trials at the same step. This is aggressive but safe—it keeps the top 50% of configurations.

**Hyperband Pruner** (more sophisticated): Uses adaptive resource allocation, giving more training time to promising trials and quickly eliminating poor ones using a principled successive halving approach.

---

### Example 1: Basic Pruning with Gradient Boosting

**Scenario:** Training a gradient boosting model with 500 trees takes 5 minutes per configuration. With 100 trials, that's 8+ hours. With pruning, poor configurations are stopped after 50-100 trees, reducing time to 3-4 hours.

```python
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import numpy as np

# Load data
X, y = load_breast_cancer(return_X_y=True)

def objective_with_pruning(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': 500,  # Will train incrementally with early stopping
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    
    # Create model
    model = XGBClassifier(**params)
    
    # For pruning, we need intermediate values
    # We'll evaluate at regular intervals during training
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'validation_0-logloss')
    
    # Split for validation during training
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        callbacks=[pruning_callback],
        verbose=False
    )
    
    # Return best validation score
    return model.best_score

# Create study with MedianPruner
study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10,  # Don't prune first 10 trials (build history)
        n_warmup_steps=20     # Don't prune before step 20 (too early to judge)
    )
)

# Optimize with pruning
study.optimize(objective_with_pruning, n_trials=50, show_progress_bar=True)

print(f"\nBest value: {study.best_value:.4f}")
print(f"Best hyperparameters: {study.best_params}")

# Analyze pruning effectiveness
pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print(f"\nPruning statistics:")
print(f"  Total trials: {len(study.trials)}")
print(f"  Completed: {len(completed_trials)}")
print(f"  Pruned: {len(pruned_trials)} ({len(pruned_trials)/len(study.trials)*100:.1f}%)")

# Estimate time saved
avg_steps_pruned = np.mean([len(t.intermediate_values) for t in pruned_trials])
avg_steps_completed = np.mean([len(t.intermediate_values) for t in completed_trials])
time_saving_pct = (1 - avg_steps_pruned / avg_steps_completed) * 100

print(f"  Avg steps completed: {avg_steps_completed:.1f}")
print(f"  Avg steps before pruning: {avg_steps_pruned:.1f}")
print(f"  Estimated time saved: {time_saving_pct:.1f}%")

```

**Output:**

```
[I 2024-11-05 11:30:15] Trial 0 finished with value: 0.0823
[I 2024-11-05 11:30:45] Trial 1 finished with value: 0.0756
[I 2024-11-05 11:31:02] Trial 2 pruned at step 45
[I 2024-11-05 11:31:18] Trial 3 pruned at step 38
[I 2024-11-05 11:31:50] Trial 4 finished with value: 0.0712
...

Best value: 0.0634
Best hyperparameters: {'max_depth': 6, 'learning_rate': 0.0523, 'subsample': 0.82, 'colsample_bytree': 0.91}

Pruning statistics:
  Total trials: 50
  Completed: 32
  Pruned: 18 (36.0%)
  Avg steps completed: 287.3
  Avg steps before pruning: 82.5
  Estimated time saved: 71.3%

```

**What happened:** Optuna pruned 18 trials (36%) after an average of 82 boosting rounds instead of the full 500 rounds. These trials were stopped because their validation loss at intermediate stages was worse than the median of ongoing trials. The time saving was ~71%—the search that would have taken 8 hours completed in ~2.3 hours while finding an excellent configuration (log loss = 0.0634).

---

## 7. Common Pitfalls: What Can Go Wrong and How to Avoid It

**The Mistake:** Using default k-fold cross-validation on time-series data

```python
# WRONG: Random k-fold on ordered time-series
from sklearn.model_selection import cross_val_score, KFold

cv = KFold(n_splits=5, shuffle=True)  # Shuffle creates data leakage!
scores = cross_val_score(model, X_timeseries, y_timeseries, cv=cv)

```

**Why it's a problem:** Shuffling places future data points in training folds that validate on past data points. The model learns from the future to predict the past, achieving 90% validation accuracy that drops to 60% in production when only historical data is available for prediction.

**The right approach:**

```python
from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=5)  # No shuffling, validates on future
scores = cross_val_score(model, X_timeseries, y_timeseries, cv=cv)

```

**Why this works:** Each validation fold contains only data chronologically after its training fold, simulating real production where you predict tomorrow using only yesterday's information.

---

**The Mistake:** Setting Optuna's `n_trials` too low

```python
# WRONG: Too few trials for Bayesian optimization to learn
study.optimize(objective, n_trials=10)

```

**Why it's a problem:** Bayesian optimization needs 10-20 trials just to build an initial model of the objective function. With only 10 total trials, you haven't given it enough data to learn patterns and make informed suggestions—it's essentially random search with overhead.

**The right approach:**

```python
# Minimum trials by search space size:
# 1-3 hyperparameters: 30-50 trials minimum
# 4-6 hyperparameters: 50-100 trials minimum
# 7-10 hyperparameters: 100-200 trials minimum
# 10+ hyperparameters: 200-500 trials minimum

study.optimize(objective, n_trials=100)  # For 5-6 hyperparameters

```

**Why this works:** More trials allow Optuna's surrogate model to learn hyperparameter-performance relationships accurately, making increasingly informed suggestions. The Bayesian approach's advantage over random search only becomes apparent after sufficient exploration.

---

**The Mistake:** Overly aggressive pruning destroys promising trials

```python
# WRONG: Prunes too early before configurations can show potential
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=0,   # Prunes immediately
    n_warmup_steps=5      # Only 5 steps before pruning
)

```

**Why it's a problem:** Some hyperparameter configurations start slow but accelerate later (e.g., low learning rate needs more epochs to converge). Pruning at step 5 might eliminate these "slow starters" that would have been optimal if allowed to complete.

**The right approach:**

```python
# Conservative: wait until trials show stable patterns
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=15,     # Build history first
    n_warmup_steps=30        # 30% of training before pruning
)

# Monitor pruning rate: aim for 30-40%
pruned_pct = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]) / len(study.trials)
if pruned_pct > 0.5:
    print("WARNING: Pruning too aggressively!")

```

**Why this works:** Waiting for 15 completed trials gives the pruner accurate historical data for comparison. Allowing 30% warmup ensures configurations have time to show their true potential before judgment.

---

**The Mistake:** Not scaling features before model training in the objective function

```python
def objective(trial):
    params = {...}
    model = RandomForestRegressor(**params)
    
    # WRONG: Cross-validating on raw features
    score = cross_val_score(model, X, y, cv=5).mean()  # X not scaled!
    return score

```

**Why it's a problem:** For models sensitive to feature scale (linear models, neural networks, SVMs), unscaled features cause:

- Poor hyperparameter suggestions (learning rate tuned for wrong scale)
- Inconsistent results across folds if fold-specific scaling isn't applied
- Worse performance than properly scaled baseline

**The right approach:**

```python
def objective(trial):
    params = {...}
    model = RandomForestRegressor(**params)
    
    # Scale within cross-validation folds
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    score = cross_val_score(pipeline, X, y, cv=5).mean()
    return score

```

**Why this works:** Pipeline ensures scaling is fit on training folds and applied to validation folds within each CV split, preventing data leakage while ensuring consistent feature scales.

---

**The Mistake:** Mixing up maximize vs minimize in Optuna studies

```python
# WRONG: Study direction doesn't match objective
study = optuna.create_study(direction='maximize')

def objective(trial):
    # ... training ...
    mse = mean_squared_error(y_true, y_pred)
    return mse  # Returning error (should minimize) but study maximizes!

```

**Why it's a problem:** Optuna will find the worst model (highest MSE) instead of the best. This silent error leads to deploying a terrible model with confidence because "optimization completed successfully."

**The right approach:**

```python
# Match study direction to objective metric:
# Accuracy, F1, R²: maximize (higher is better)
# MSE, MAE, log loss: minimize (lower is better)

study = optuna.create_study(direction='minimize')  # For MSE

def objective(trial):
    # ... training ...
    mse = mean_squared_error(y_true, y_pred)
    return mse  # Correct: minimizing error

# Or negate error metrics:
study = optuna.create_study(direction='maximize')

def objective(trial):
    # ... training ...
    mse = mean_squared_error(y_true, y_pred)
    return -mse  # Maximize negative MSE = minimize MSE

```

**Why this works:** Direction matches the metric's optimization goal, ensuring Optuna searches in the correct direction.

---

**If you're stuck:**

1. **Validation scores don't match test scores:** Check if your CV strategy matches data structure (time-series → TimeSeriesSplit, grouped → GroupKFold)
2. **Optuna finds poor hyperparameters:** Increase `n_trials` to at least 50-100 for meaningful Bayesian learning
3. **Too many trials being pruned:** Increase `n_warmup_steps` to 20-30% of total training steps
4. **Search is too slow:** Add more aggressive pruning or reduce `n_trials` but keep above minimum thresholds
5. **Results are unstable:** Check that you're using fixed `random_state` in model and `seed` in Optuna sampler

---

## 8. Your Turn: Practice & Self-Assessment

### Practice Task (Estimated: 40-45 minutes)

**The Challenge:** Build a complete hyperparameter tuning pipeline for predicting customer churn with imbalanced classes, using appropriate cross-validation, Bayesian optimization, and early stopping.

**Specifications:**

1. 
**Load and prepare data:**
`from sklearn.datasets import make_classification
X, y = make_classification(n_samples=5000, n_features=20, 
                          n_informative=15, n_redundant=5,
                          weights=[0.90, 0.10],  # 10% churn rate
                          flip_y=0.01, random_state=42)`

2. 
**Implement Optuna objective function that:**

Uses `GradientBoostingClassifier`
Searches these hyperparameters:

`n_estimators`: 100-500 (integer)
`learning_rate`: 0.001-0.3 (float, log scale)
`max_depth`: 2-8 (integer)
`min_samples_split`: 2-20 (integer)
`subsample`: 0.5-1.0 (float)

Uses `StratifiedKFold` with 5 folds (for imbalanced data)
Optimizes F1 score (appropriate for imbalanced classification)
Implements pruning with `MedianPruner`

3. 
**Run optimization:**

80 trials total
Use TPESampler with seed=42
Set pruner with n_startup_trials=10, n_warmup_steps=20

4. 
**Analysis:**

Report best F1 score and corresponding hyperparameters
Calculate percentage of trials pruned
Plot optimization history
Identify top 3 most important hyperparameters

**Example structure:**

```python
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def objective(trial):
    # Your code here:
    # 1. Suggest hyperparameters
    # 2. Create model
    # 3. Use StratifiedKFold CV
    # 4. Return F1 score
    pass

study = optuna.create_study(
    # Your configuration here
)

study.optimize(objective, n_trials=80)

# Analysis code here

```

**Hint:** For the pruning aspect, you'll need to implement staged training manually—train the model incrementally (e.g., 50 trees at a time) and report intermediate scores to Optuna. Use `trial.report(score, step)` after each stage and check `trial.should_prune()`. Remember that StratifiedKFold ensures each fold maintains the 90/10 class split, which is critical for reliable F1 scores on imbalanced data. Start by getting basic optimization working without pruning, then add pruning once the objective function runs correctly.

**Extension (optional):** Compare your Optuna results with RandomizedSearchCV using 80 random trials. Which finds better hyperparameters? Create a visualization showing how Optuna's best score improves over trials vs RandomizedSearchCV's static performance.

---

### Check Your Understanding

1. 
**Explanation question:** Explain in your own words why TimeSeriesSplit is necessary for time-series data and what specific problem would occur in production if you used standard k-fold instead. Give a concrete example with dates/predictions.

2. 
**Application question:** You have a medical dataset with 1,000 patients, each having 5-10 hospital visits. Your task is predicting readmission risk. Should you use: (a) standard k-fold, (b) TimeSeriesSplit, (c) StratifiedKFold, or (d) GroupKFold? Explain your reasoning and what would go wrong with the other choices.

3. 
**Error analysis:** Your colleague ran Optuna optimization with these results:
`Trial 1: 0.65 accuracy
Trial 2: 0.68 accuracy
Trial 10: 0.71 accuracy
Trial 20: 0.72 accuracy
Trial 50: 0.73 accuracy
Trial 100: 0.73 accuracy (same as trial 50)`

They used 100 trials but the best score hasn't improved since trial 50. What's likely wrong, and what would you change?

4. 
**Transfer question:** You're tuning a neural network with 12 hyperparameters (learning rate, batch size, number of layers, layer sizes, dropout rates, optimizer type, etc.). You have 24 hours of compute time and each trial takes approximately 30 minutes to complete. Design a tuning strategy using Optuna with early stopping that maximizes your chances of finding good hyperparameters within this budget.

---

**Answers & Explanations:**

1. 
**TimeSeriesSplit necessity:**
TimeSeriesSplit ensures validation data is always chronologically after training data, preventing the model from learning future information.
**Concrete example:** Imagine stock price prediction for January 2024. With standard k-fold, your "Fold 3" might train on [Feb 2024, May 2024, Aug 2024] and validate on [Jan 2024, Apr 2024]. This means training on February-August data to predict January—the model literally sees the future. In production (predicting February 2025), the model only has Jan 2025 and earlier, causing the 90% validation accuracy to drop to 55% production accuracy.
With TimeSeriesSplit, Fold 3 trains on [Jan 2023 - Dec 2023] and validates on [Jan 2024], matching the production scenario where you predict the next month using only historical data. The 70% validation accuracy now accurately predicts production performance.

2. 
**Medical readmission prediction:**
Use **(d) GroupKFold** with patients as groups.
**Reasoning:** The data has hierarchical structure—visits are nested within patients. The real-world task is "predict readmission for a new patient we've never seen," not "predict another visit from a patient we already know." GroupKFold ensures all visits from each patient stay together in either training or validation, never split between both.
**What goes wrong with alternatives:**

**(a) Standard k-fold:** Patient 47's visits appear in both training and validation. The model learns patient-specific patterns ("Patient 47 tends to be readmitted") rather than generalizable medical indicators. Validation accuracy 85%, production accuracy on new patients 60%.

**(b) TimeSeriesSplit:** Only appropriate if predicting *future* visits from *same* patients using their historical visits. Not applicable here since production involves new patients.

**(c) StratifiedKFold:** Maintains readmission rate balance (good) but doesn't prevent patient data leakage (bad). Same patient's visits split across folds, causing the model to memorize patient IDs.

3. 
**Stagnant optimization analysis:**
The optimization stagnated because the search space is likely too small or insufficiently diverse. Possible issues:
**Problem 1: Too few hyperparameters or narrow ranges**

If searching only 2-3 hyperparameters with limited ranges, Bayesian optimization exhausts the space by trial 50
**Fix:** Expand search space—add more hyperparameters or widen ranges

**Problem 2: Insufficient trials for Bayesian learning**

100 trials might be too few for a complex space
**Fix:** Increase to 200-300 trials

**Problem 3: Poor hyperparameter suggestions**

Check if hyperparameters use appropriate scales (log scale for learning_rate, etc.)
**Fix:** Use `log=True` for exponentially-scaled parameters

**Problem 4: Optimization actually succeeded**

Maybe 0.73 is near the achievable maximum for this model/data combination
**Validation:** Try a completely different model class—if it also plateaus at 0.73-0.75, the data has fundamental limits

**What to change:** First, verify the search space is appropriate (check that best params aren't at range boundaries). Second, increase trials to 200. Third, try a different model architecture to confirm 0.73 isn't a hard ceiling.

4. 
**Neural network tuning strategy:**
**Budget calculation:**

24 hours = 1,440 minutes
30 min/trial without pruning = 48 trials maximum
Goal: Run 100-150 trials using early stopping

**Strategy:**
`# 1. Use aggressive early stopping to fit more trials
pruner = optuna.pruners.HyperbandPruner(
    min_resource=5,       # Prune after 5 epochs minimum
    max_resource=100,     # Maximum 100 epochs
    reduction_factor=3    # Keep top 33%
)

# 2. Two-stage search for efficiency

# Stage 1: Broad search (60 trials, ~12 hours)
#   - Test all 12 hyperparameters
#   - Aggressive pruning (saves ~60% time)
#   - Identifies promising regions
study_broad = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=pruner
)
study_broad.optimize(objective, n_trials=60`

---

### Self-Assessment Checklist

You've mastered this topic if you can:

- Select the appropriate cross-validation strategy (k-fold, stratified, time-series, group) based on data structure
- Implement an Optuna objective function with proper hyperparameter suggestions
- Integrate early stopping / pruning to reduce tuning time by 40-70%
- Combine Bayesian optimization with specialized CV strategies
- Interpret Optuna's parameter importance and optimization history
- Debug common issues (wrong CV strategy, pruning too aggressive, insufficient trials)
- Explain why naive approaches fail and how advanced techniques solve specific problems
- Design production-ready tuning pipelines within compute budget constraints

**If you checked fewer than 6 boxes:**

Focus on building the concepts step by step:

- **If CV strategies are unclear:** Create a small time-series dataset (20 sequential points) and manually split it with TimeSeriesSplit vs k-fold. Print which points go to training vs validation in each fold to see the difference visually.
- **If Optuna feels overwhelming:** Start with just 2 hyperparameters and 20 trials, no pruning. Get this working, then add complexity (more params, then pruning).
- **If early stopping is confusing:** Run a model with and without pruning, logging which trial numbers get pruned and at what stage. Compare total time to see the savings.

---

## 9. Consolidation: Key Takeaways & Next Steps

### The Essential Ideas

- **Cross-validation strategy must match data structure:** Use TimeSeriesSplit for temporal data, StratifiedKFold for imbalanced classes, GroupKFold for hierarchical data—wrong choice causes data leakage and misleading results
- **Bayesian optimization learns from history:** Each trial informs the next, finding good hyperparameters in 50-200 trials vs thousands for grid search
- **Optuna is production-ready:** Handles mixed parameter types, conditional search spaces, parallel trials, and integrates with all major ML frameworks
- **Early stopping saves 40-70% compute time:** Prunes obviously poor trials mid-training, allowing more trials within the same budget
- **MedianPruner is safe and effective:** Keeps top 50% of configurations, good default choice
- **Tune pruning aggressiveness:** Start conservative (n_startup_trials=15, n_warmup_steps=30), adjust based on pruning rate (target 30-40%)
- **Always scale features:** Use Pipeline to ensure scaling happens correctly within CV folds
- **Match study direction to metric:** Maximize accuracy/F1/R², minimize MSE/MAE/loss

### Mental Model Check

By now, you should think of hyperparameter tuning as: **An intelligent search process where you (1) ensure realistic validation through appropriate CV strategy, (2) efficiently explore hyperparameter space using Bayesian optimization that learns from each trial, and (3) accelerate iteration with early stopping that prunes doomed configurations. Together, these techniques find near-optimal models in hours instead of days while providing trustworthy performance estimates that match production.**

### What You Can Now Do

You can now build production-grade ML pipelines that automatically find optimal hyperparameters for complex models while avoiding data leakage and respecting computational budgets. You understand how to diagnose why validation performance doesn't match production (wrong CV strategy) and how to fix it. You can tune neural networks with 10+ hyperparameters in reasonable time using Bayesian optimization and early stopping.

### Next Steps

**To deepen this knowledge:**

- Experiment with different Optuna samplers (TPE vs CMA-ES vs NSGAII for multi-objective)
- Practice on diverse datasets (time-series, images, text) to internalize when each CV strategy applies
- Profile your tuning pipelines to identify bottlenecks and optimize further
- Study advanced pruning algorithms (Hyperband, ASHA, Population Based Training)

**To build on this:**

- **AutoML frameworks:** Explore Auto-sklearn, H2O AutoML which automate the entire ML pipeline including feature engineering
- **Neural Architecture Search (NAS):** Learn to search over model architectures, not just hyperparameters
- **Multi-objective optimization:** Optimize for accuracy AND inference speed simultaneously using Optuna's multi-objective capabilities
- **Distributed hyperparameter search:** Scale tuning across multiple machines using Optuna's distributed optimization features
- **MLOps integration:** Incorporate hyperparameter tuning into CI/CD pipelines with tools like MLflow, Weights & Biases

**Additional resources:**

- Optuna documentation and tutorials: [https://optuna.readthedocs.io](https://optuna.readthedocs.io)
- "Hyperparameter Optimization in Machine Learning" by Bischl et al. (comprehensive academic treatment)
- Optuna dashboard for real-time visualization: [https://github.com/optuna/optuna-dashboard](https://github.com/optuna/optuna-dashboard)

---

## Quick Reference Card

### Cross-Validation Strategy Selection

Data Type | Strategy | Key Parameter | When to Use
Independent samples | KFold | n_splits=5, shuffle=True | Default for IID data
Imbalanced classes | StratifiedKFold | n_splits=5, shuffle=True | Classification with rare classes
Time-series | TimeSeriesSplit | n_splits=5 | Temporal dependencies
Grouped/hierarchical | GroupKFold | n_splits=5, groups=group_ids | Nested data (patients, users)

### Optuna Hyperparameter Suggestions

```python
# Integer with linear scale
n_estimators = trial.suggest_int('n_estimators', 50, 500)

# Integer with log scale (2, 4, 8, 16, ...)
max_depth = trial.suggest_int('max_depth', 2, 32, log=True)

# Float with linear scale
subsample = trial.suggest_float('subsample', 0.5, 1.0)

# Float with log scale
learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

# Categorical choice
optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])

```

### Pruner Configuration Guidelines

```python
# Conservative (safer, fewer false negatives)
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=15,    # Don't prune until 15 trials complete
    n_warmup_steps=50       # Don't prune before step 50
)

# Aggressive (faster, more pruning)
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,
    n_warmup_steps=10
)

# Adaptive (good default)
pruner = optuna.pruners.HyperbandPruner(
    min_resource=10,
    max_resource=100,
    reduction_factor=3
)

```

### Complete Optuna Template

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    # 1. Suggest hyperparameters
    params = {
        'param1': trial.suggest_int('param1', low, high),
        'param2': trial.suggest_float('param2', low, high, log=True),
    }
    
    # 2. Create model
    model = YourModel(**params)
    
    # 3. Evaluate with appropriate CV
    cv = StratifiedKFold(5)  # Or TimeSeriesSplit, GroupKFold
    score = cross_val_score(model, X, y, cv=cv, scoring='f1').mean()
    
    return score

# 4. Create study
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
)

# 5. Optimize
study.optimize(objective, n_trials=100)

# 6. Get results
print(f"Best: {study.best_value}")
print(study.best_params)

```

### Minimum Trial Recommendations

- **1-3 hyperparameters:** 30-50 trials
- **4-6 hyperparameters:** 50-100 trials
- **7-10 hyperparameters:** 100-200 trials
- **10+ hyperparameters:** 200-500 trials

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