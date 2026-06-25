# 67. AutoML for LLMs - Suman - 10 Apr 2026

# Lecture Notes: AutoML for LLMs

### PPT File: [Click Here](https://drive.google.com/file/d/11MN5pQoOXESr9HdM9JA8hbRnaYvJz-SI/view?usp=sharing)

### Colab File: [Click Here](https://drive.google.com/file/d/1YDzrTBE1PUHChEM6fZPn-q71ei8mJE_e/view?usp=sharing)

---

## Learning Objectives

By the end of this session, you will:

1. Understand Bayesian optimization for hyperparameter tuning
2. Implement early stopping to save training time
3. Design effective search spaces
4. Use Optuna for automated hyperparameter tuning
5. Deploy AutoML in production LLM fine-tuning

---

## Part 1: Why AutoML for LLMs? (10 minutes)

### The Hyperparameter Tuning Problem

**LLM Fine-tuning has many hyperparameters:**

```python
hyperparameters = {
    'learning_rate': ???,      # How fast to learn
    'batch_size': ???,         # How many samples per update
    'epochs': ???,             # How long to train
    'warmup_steps': ???,       # Learning rate warmup
    'weight_decay': ???,       # Regularization
    'max_grad_norm': ???,      # Gradient clipping
    'dropout': ???,            # Dropout rate
    'adam_epsilon': ???,       # Optimizer epsilon
    'num_train_epochs': ???,   # Training duration
}

# How do you find the best combination?

```

### Traditional Approaches and Their Problems

**1. Manual Tuning**

```
Problem: Slow, requires expertise, not reproducible
Time: Days to weeks
Cost: High (expert time + GPU time)
Result: Suboptimal (limited exploration)

```

**2. Grid Search**

```python
# Try all combinations
learning_rates = [1e-5, 3e-5, 5e-5]
batch_sizes = [8, 16, 32]
epochs = [3, 5, 7]

total_experiments = 3 × 3 × 3 = 27
time_per_experiment = 2 hours
total_time = 54 hours

Problem: Combinatorial explosion
- 4 parameters with 3 values each = 81 experiments
- 5 parameters with 4 values each = 1,024 experiments

```

**3. Random Search**

```python
# Random sampling
for i in range(50):
    lr = random.uniform(1e-6, 1e-4)
    bs = random.choice([8, 16, 32])
    epochs = random.randint(3, 7)
    
    score = train(lr, bs, epochs)

Problem: No learning from past results
- Experiment 1: lr=1e-5 → good
- Experiment 2: lr=8e-5 → bad
- Experiment 3: lr=2e-5 → ??? (doesn't use info from 1&2)

```

### The AutoML Solution

**Key Insight:** Learn from past experiments to guide future ones

```
Traditional: Try everything equally
AutoML: Learn what works, focus there

Traditional: Complete every experiment
AutoML: Stop bad experiments early

Traditional: Fixed grid of values
AutoML: Explore continuous space

```

---

## Part 2: Bayesian Optimization (30 minutes)

### Core Concept

**Bayesian Optimization = Smart Sequential Search**

```
1. Build a probabilistic model of hyperparameter → performance
2. Use model to decide which hyperparameters to try next
3. Balance: Explore (try new areas) vs Exploit (improve known good areas)
4. Update model with new results
5. Repeat

```

### The Mathematics (Simplified)

**Gaussian Process (GP): The Surrogate Model**

```python
# Conceptually:
# GP learns: f(hyperparameters) → performance

# Given past results:
past_experiments = [
    ({'lr': 1e-5, 'bs': 16}, 0.73),  # (params, f1_score)
    ({'lr': 3e-5, 'bs': 16}, 0.82),
    ({'lr': 5e-5, 'bs': 16}, 0.65),
]

# GP predicts for new params:
new_params = {'lr': 2e-5, 'bs': 16}

# Prediction:
mean = 0.78      # Expected performance
uncertainty = 0.05  # Confidence interval

# Interpretation:
# "I think lr=2e-5 will give F1=0.78±0.05"
# "I'm fairly confident because it's between tested values"

```

**Visualization:**

```
F1 Score
   |
0.9|                    ╱‾‾‾╲
0.8|              ╱‾‾‾‾╱     ╲
0.7|        ╱‾‾‾‾╱            ╲___
0.6|  ╱‾‾‾‾╱                      
   |_________________________________
      1e-5   2e-5  3e-5  4e-5  5e-5   Learning Rate
      ↑             ↑          ↑
    tested       predict    tested
    
GP learns this curve from just 3 points!
Then predicts what's likely at untested points.

```

### Acquisition Functions

**How to choose next experiment?**

**1. Expected Improvement (EI)**

```python
def expected_improvement(params):
    """How much better than current best?"""
    current_best = 0.82
    predicted_mean, predicted_std = gp.predict(params)
    
    # How likely to beat current best?
    improvement = max(0, predicted_mean - current_best)
    
    # Weight by uncertainty
    # High uncertainty → high EI (exploration)
    # Low uncertainty + high mean → high EI (exploitation)
    
    return improvement * predicted_std

# Try params with highest EI
next_params = maximize(expected_improvement)

```

**2. Upper Confidence Bound (UCB)**

```python
def upper_confidence_bound(params, kappa=2.0):
    """Optimistic estimate"""
    predicted_mean, predicted_std = gp.predict(params)
    
    # Optimistic: mean + uncertainty
    # kappa controls exploration (higher = more exploration)
    ucb = predicted_mean + kappa * predicted_std
    
    return ucb

# Try params with highest UCB
next_params = maximize(upper_confidence_bound)

```

**3. Probability of Improvement (PI)**

```python
def probability_of_improvement(params):
    """Probability of beating current best"""
    current_best = 0.82
    predicted_mean, predicted_std = gp.predict(params)
    
    # Calculate probability
    z = (predicted_mean - current_best) / predicted_std
    pi = norm.cdf(z)  # Cumulative distribution function
    
    return pi

# Try params with highest PI
next_params = maximize(probability_of_improvement)

```

**Comparison:**

```
Acquisition Function | Exploration | Exploitation | Use Case
---------------------|-------------|--------------|----------
EI (Expected Imp.)   | Balanced    | Balanced     | General purpose
UCB                  | High        | Medium       | Large search space
PI                   | Low         | High         | Exploit known good region

```

### Bayesian Optimization Algorithm

```python
def bayesian_optimization(
    objective_function,
    search_space,
    n_initial=5,
    n_iterations=20
):
    """Complete Bayesian optimization loop"""
    
    results = []
    
    # Step 1: Random initialization
    print("Phase 1: Random Exploration")
    for i in range(n_initial):
        params = sample_random(search_space)
        score = objective_function(params)
        results.append((params, score))
        print(f"  Trial {i}: {params} → {score:.3f}")
    
    # Step 2: Bayesian optimization
    print("\nPhase 2: Bayesian Optimization")
    for i in range(n_iterations):
        # Build GP model from past results
        gp_model = fit_gaussian_process(results)
        
        # Find next best parameters using acquisition function
        next_params = optimize_acquisition(gp_model, search_space)
        
        # Evaluate
        score = objective_function(next_params)
        results.append((next_params, score))
        
        # Track best so far
        best_score = max(r[1] for r in results)
        print(f"  Trial {n_initial+i}: {next_params} → {score:.3f} (best: {best_score:.3f})")
    
    # Return best
    best_params, best_score = max(results, key=lambda x: x[1])
    return best_params, best_score

# Example usage
def train_model(params):
    """Objective function: train and evaluate"""
    model = build_model(params)
    model.train()
    f1 = model.evaluate()
    return f1

search_space = {
    'learning_rate': (1e-6, 1e-4, 'log'),
    'batch_size': [8, 16, 32, 64],
    'dropout': (0.1, 0.5, 'uniform')
}

best_params, best_score = bayesian_optimization(
    train_model,
    search_space,
    n_initial=5,
    n_iterations=20
)

```

### Practical Implementation with Optuna

**Optuna: Production-Ready Bayesian Optimization**

```python
import optuna
from transformers import AutoModelForSequenceClassification, Trainer

def objective(trial):
    """Optuna objective function"""
    
    # Define search space
    params = {
        'learning_rate': trial.suggest_loguniform('lr', 1e-6, 1e-4),
        'per_device_train_batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
        'num_train_epochs': trial.suggest_int('epochs', 3, 7),
        'warmup_ratio': trial.suggest_uniform('warmup', 0.0, 0.1),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-1),
    }
    
    # Train model
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    
    training_args = TrainingArguments(
        output_dir='./results',
        **params,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    metrics = trainer.evaluate()
    f1_score = metrics['eval_f1']
    
    return f1_score

# Create study
study = optuna.create_study(
    direction='maximize',  # Maximize F1 score
    sampler=optuna.samplers.TPESampler(),  # Bayesian optimization
)

# Optimize
study.optimize(objective, n_trials=25)

# Results
print(f"Best F1: {study.best_value:.3f}")
print(f"Best params: {study.best_params}")

```

---

## Part 3: Early Stopping (20 minutes)

### Why Early Stopping?

**Problem: Overfitting and Wasted Time**

```python
# Without early stopping
for epoch in range(10):
    train_loss = train_epoch()
    val_loss = validate()
    
    print(f"Epoch {epoch}: Train={train_loss:.3f}, Val={val_loss:.3f}")

# Output:
# Epoch 0: Train=0.500, Val=0.450
# Epoch 1: Train=0.400, Val=0.380
# Epoch 2: Train=0.320, Val=0.330  ← Best validation
# Epoch 3: Train=0.260, Val=0.350  ← Overfitting starts
# Epoch 4: Train=0.210, Val=0.380
# Epoch 5: Train=0.180, Val=0.410  ← Wasted 3 epochs!
# ...
# Epoch 9: Train=0.050, Val=0.550  ← Severely overfitted

# Problems:
# 1. Wasted 7 epochs of training time
# 2. Final model is worse than epoch 2
# 3. Need to remember to use epoch 2 checkpoint

```

### Early Stopping Mechanics

**Basic Algorithm:**

```python
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        """
        patience: How many epochs to wait for improvement
        min_delta: Minimum change to count as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model = None
        
    def __call__(self, val_loss, model):
        """Check if should stop training"""
        
        # Is this better than best so far?
        if val_loss < self.best_loss - self.min_delta:
            # Yes! New best
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = model.state_dict().copy()
            return False  # Don't stop
        else:
            # No improvement
            self.counter += 1
            
            if self.counter >= self.patience:
                # Waited long enough, stop
                print(f"Early stopping triggered after {self.counter} epochs without improvement")
                return True  # Stop training
            
            return False  # Keep going

# Usage
early_stopping = EarlyStopping(patience=3)

for epoch in range(100):  # Max epochs
    train_loss = train_epoch()
    val_loss = validate()
    
    if early_stopping(val_loss, model):
        print(f"Stopped at epoch {epoch}")
        break

# Load best model
model.load_state_dict(early_stopping.best_model)

```

### Advanced Early Stopping Strategies

**1. Relative Improvement**

```python
class RelativeEarlyStopping:
    def __init__(self, patience=3, min_relative_delta=0.01):
        """Stop if improvement < 1% of current loss"""
        self.patience = patience
        self.min_relative_delta = min_relative_delta
        self.best_loss = float('inf')
        self.counter = 0
        
    def __call__(self, val_loss, model):
        relative_improvement = (self.best_loss - val_loss) / self.best_loss
        
        if relative_improvement > self.min_relative_delta:
            # Significant relative improvement
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

```

**2. Adaptive Patience**

```python
class AdaptiveEarlyStopping:
    def __init__(self, initial_patience=3, max_patience=10):
        """Increase patience if showing slow improvement"""
        self.patience = initial_patience
        self.max_patience = max_patience
        self.best_loss = float('inf')
        self.counter = 0
        self.improvement_history = []
        
    def __call__(self, val_loss, model):
        improvement = self.best_loss - val_loss
        self.improvement_history.append(improvement)
        
        if improvement > 0:
            # Getting better
            self.best_loss = val_loss
            self.counter = 0
            
            # If consistent small improvements, increase patience
            if len(self.improvement_history) >= 3:
                avg_improvement = sum(self.improvement_history[-3:]) / 3
                if avg_improvement > 0 and avg_improvement < 0.01:
                    self.patience = min(self.patience + 1, self.max_patience)
            
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

```

**3. Metric-Specific Early Stopping**

```python
class MultiMetricEarlyStopping:
    """Stop only if ALL metrics stop improving"""
    
    def __init__(self, metrics=['loss', 'f1', 'accuracy'], patience=3):
        self.patience = patience
        self.best_values = {m: float('-inf') if m != 'loss' else float('inf') 
                           for m in metrics}
        self.counters = {m: 0 for m in metrics}
        
    def __call__(self, metrics_dict, model):
        any_improved = False
        
        for metric, value in metrics_dict.items():
            is_loss = (metric == 'loss')
            improved = (value < self.best_values[metric]) if is_loss else (value > self.best_values[metric])
            
            if improved:
                self.best_values[metric] = value
                self.counters[metric] = 0
                any_improved = True
            else:
                self.counters[metric] += 1
        
        # Stop only if ALL metrics haven't improved
        all_stagnant = all(c >= self.patience for c in self.counters.values())
        
        return all_stagnant

# Usage
early_stopping = MultiMetricEarlyStopping(['loss', 'f1'], patience=3)

for epoch in range(100):
    train()
    metrics = evaluate()  # {'loss': 0.35, 'f1': 0.82}
    
    if early_stopping(metrics, model):
        break

```

### Integration with Optuna (Pruning)

**Pruning = Early stopping for hyperparameter trials**

```python
import optuna

def objective(trial):
    params = {
        'learning_rate': trial.suggest_loguniform('lr', 1e-6, 1e-4),
        'batch_size': trial.suggest_categorical('bs', [8, 16, 32]),
    }
    
    model = create_model(params)
    
    for epoch in range(10):
        train_loss = train_epoch(model)
        val_loss = validate(model)
        
        # Report intermediate value to Optuna
        trial.report(val_loss, epoch)
        
        # Optuna decides: Should we stop this trial early?
        # If val_loss is worse than other trials at this epoch, prune it
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return final_score

# Use median pruner
study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,    # Don't prune first 5 trials
        n_warmup_steps=2,      # Don't prune before epoch 2
    )
)

study.optimize(objective, n_trials=50)

# Many trials will be pruned early, saving time!

```

**Pruning Example:**

```
Trial 1: lr=1e-5
  Epoch 0: val_loss=0.45
  Epoch 1: val_loss=0.38
  Epoch 2: val_loss=0.35
  ... continues

Trial 2: lr=1e-3 (too high!)
  Epoch 0: val_loss=0.65
  Epoch 1: val_loss=0.72  ← Worse than trial 1
  [PRUNED] Saved 8 epochs!

Trial 3: lr=3e-5
  Epoch 0: val_loss=0.42
  Epoch 1: val_loss=0.36  ← Better than trial 1
  ... continues

```

---

## Part 4: Search Space Design (20 minutes)

### The Importance of Scale

**Different hyperparameters need different scales**

**1. Learning Rate: LOG SCALE**

```python
# Wrong: Linear scale
learning_rates = [0.00001, 0.00003, 0.00005]

# Problem:
# - Difference between 1e-5 and 3e-5 is 2e-5 (200%)
# - Difference between 3e-5 and 5e-5 is 2e-5 (67%)
# - Same absolute difference, very different relative effect!

# Right: Log scale
learning_rate = trial.suggest_loguniform('lr', 1e-6, 1e-4)
# Samples: 2.3e-6, 1.7e-5, 4.8e-5, 8.2e-5
# Equal spacing in log space = proportional spacing in linear space

# Why it matters:
# Learning rate effect is multiplicative, not additive
# 2x learning rate ≈ 2x faster learning
# So we should search in multiplicative (log) space

```

**Visualization:**

```
Linear Scale:
|-------|-------|-------|-------|
1e-5   2e-5   3e-5   4e-5   5e-5
  ↓       ↓       ↓       ↓
small  small  medium large  (uneven effects)

Log Scale:
|------------|------------|------------|
1e-6        1e-5        1e-4
   ↓           ↓           ↓
 small      medium      large  (even effects)

```

**2. Batch Size: POWERS OF 2**

```python
# Wrong: Arbitrary values
batch_sizes = [10, 20, 30, 40]

# Right: Powers of 2
batch_size = trial.suggest_categorical('bs', [8, 16, 32, 64, 128])

# Why?
# - GPU memory organized in powers of 2
# - Better hardware utilization
# - More efficient computation
# - Industry standard

# Example:
# batch_size=32 → 100% GPU utilization
# batch_size=30 → 94% GPU utilization (waste)

```

**3. Dropout: LINEAR SCALE**

```python
# Right: Linear scale
dropout = trial.suggest_uniform('dropout', 0.1, 0.5)

# Why?
# - Dropout is a probability (0 to 1)
# - Effects are roughly linear
# - 0.2 vs 0.3 has similar impact as 0.3 vs 0.4
# - Small range (0.1 to 0.5), so linear is fine

```

### Search Space Types

**1. Continuous Parameters**

```python
# Uniform (linear)
dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
# Samples evenly: 0.15, 0.27, 0.33, 0.44

# Log-uniform (logarithmic)
learning_rate = trial.suggest_loguniform('lr', 1e-6, 1e-4)
# Samples evenly in log space: 2.3e-6, 1.7e-5, 4.8e-5

# Discrete uniform
num_layers = trial.suggest_int('layers', 2, 6)
# Samples: 2, 3, 4, 5, 6

```

**2. Categorical Parameters**

```python
# Discrete choices
optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'adamw'])
batch_size = trial.suggest_categorical('bs', [8, 16, 32, 64])
activation = trial.suggest_categorical('activation', ['relu', 'gelu', 'tanh'])

```

**3. Conditional Parameters**

```python
def objective(trial):
    # Main parameter
    use_warmup = trial.suggest_categorical('use_warmup', [True, False])
    
    # Conditional parameter (only used if use_warmup=True)
    if use_warmup:
        warmup_ratio = trial.suggest_uniform('warmup_ratio', 0.0, 0.1)
    else:
        warmup_ratio = 0.0
    
    # Another conditional
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    
    if optimizer == 'adam':
        beta1 = trial.suggest_uniform('beta1', 0.8, 0.95)
        beta2 = trial.suggest_uniform('beta2', 0.95, 0.999)
    else:
        # SGD doesn't have betas
        beta1, beta2 = None, None
    
    # Use parameters
    params = {
        'optimizer': optimizer,
        'warmup_ratio': warmup_ratio,
        'beta1': beta1,
        'beta2': beta2,
    }
    
    return train(params)

```

### Complete Search Space Example

```python
def define_search_space(trial):
    """Production-ready LLM fine-tuning search space"""
    
    search_space = {
        # Learning rate: LOG SCALE (most important!)
        'learning_rate': trial.suggest_loguniform('lr', 1e-6, 1e-4),
        
        # Batch size: POWERS OF 2
        'per_device_train_batch_size': trial.suggest_categorical(
            'batch_size', [8, 16, 32, 64]
        ),
        
        # Epochs: INTEGER
        'num_train_epochs': trial.suggest_int('epochs', 3, 7),
        
        # Warmup: LINEAR (small range)
        'warmup_ratio': trial.suggest_uniform('warmup', 0.0, 0.1),
        
        # Weight decay: LOG SCALE (regularization)
        'weight_decay': trial.suggest_loguniform('wd', 1e-5, 1e-1),
        
        # Gradient clipping: LINEAR
        'max_grad_norm': trial.suggest_uniform('grad_clip', 0.5, 2.0),
        
        # Dropout: LINEAR
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.5),
        
        # Optimizer: CATEGORICAL
        'optimizer': trial.suggest_categorical('opt', ['adam', 'adamw']),
        
        # Learning rate schedule: CATEGORICAL
        'lr_scheduler_type': trial.suggest_categorical(
            'scheduler', ['linear', 'cosine', 'polynomial']
        ),
    }
    
    # Conditional: Adam-specific parameters
    if search_space['optimizer'] in ['adam', 'adamw']:
        search_space['adam_beta1'] = trial.suggest_uniform('beta1', 0.85, 0.95)
        search_space['adam_beta2'] = trial.suggest_uniform('beta2', 0.95, 0.999)
        search_space['adam_epsilon'] = trial.suggest_loguniform('eps', 1e-9, 1e-6)
    
    return search_space

```

### Domain Knowledge Integration

**Use what we know to narrow the search space**

```python
# Generic search space (wasteful)
generic_space = {
    'learning_rate': (1e-8, 1.0),  # Way too wide!
    'batch_size': [1, 2, 4, 8, 16, 32, 64, 128, 256],  # Too many options
    'dropout': (0.0, 0.9),  # Includes extreme values
}

# Informed search space (better)
informed_space = {
    # LLM fine-tuning typically uses 1e-6 to 1e-4
    'learning_rate': (1e-6, 1e-4),
    
    # Modern GPUs work well with 8-64
    'batch_size': [8, 16, 32, 64],
    
    # Dropout >0.5 usually too aggressive for transformers
    'dropout': (0.1, 0.5),
    
    # Transformers typically train 3-7 epochs
    'epochs': (3, 7),
    
    # Warmup helps transformer training
    'warmup_ratio': (0.0, 0.1),  # 0-10% of training
}

# Result:
# - Generic: ~1000 possible combinations to explore
# - Informed: ~50 relevant combinations to explore
# - 20x more efficient!

```

---

## Part 5: Complete Production Example (10 minutes)

### End-to-End AutoML Pipeline

```python
import optuna
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_dataset

class LLMAutoMLOptimizer:
    def __init__(self, model_name, dataset):
        self.model_name = model_name
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def objective(self, trial):
        """Optuna objective function"""
        
        # 1. Define search space
        params = {
            'learning_rate': trial.suggest_loguniform('lr', 1e-6, 1e-4),
            'per_device_train_batch_size': trial.suggest_categorical(
                'batch_size', [8, 16, 32]
            ),
            'num_train_epochs': trial.suggest_int('epochs', 3, 7),
            'warmup_ratio': trial.suggest_uniform('warmup', 0.0, 0.1),
            'weight_decay': trial.suggest_loguniform('wd', 1e-5, 1e-2),
        }
        
        # 2. Create model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )
        
        # 3. Training arguments with early stopping
        training_args = TrainingArguments(
            output_dir=f'./results/trial_{trial.number}',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            **params
        )
        
        # 4. Create trainer with early stopping
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['validation'],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # 5. Train
        trainer.train()
        
        # 6. Evaluate
        metrics = trainer.evaluate()
        
        # 7. Return metric to optimize
        return metrics['eval_f1']
    
    def optimize(self, n_trials=25):
        """Run optimization"""
        
        # Create study with pruning
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=1
            )
        )
        
        # Optimize
        study.optimize(self.objective, n_trials=n_trials)
        
        # Results
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial': study.best_trial,
            'study': study
        }

# Usage
optimizer = LLMAutoMLOptimizer(
    model_name='bert-base-uncased',
    dataset=load_dataset('glue', 'sst2')
)

results = optimizer.optimize(n_trials=25)

print(f"Best F1: {results['best_value']:.3f}")
print(f"Best params: {results['best_params']}")

# Visualize
import optuna.visualization as vis

# Optimization history
vis.plot_optimization_history(results['study'])

# Parameter importances
vis.plot_param_importances(results['study'])

# Hyperparameter relationships
vis.plot_parallel_coordinate(results['study'])

```

---

## Summary & Best Practices

### Key Takeaways

**1. Bayesian Optimization**

- Learns from past experiments
- Balances exploration vs exploitation
- 5-10x more efficient than grid search
- Use Optuna/Hyperopt for production

**2. Early Stopping**

- Save 30-50% of training time
- Prevent overfitting
- Always use with validation set
- Combine with checkpointing

**3. Search Space Design**

- Log scale for: learning rate, weight decay
- Powers of 2 for: batch size
- Linear scale for: dropout, warmup ratio
- Use domain knowledge to narrow ranges

**4. Production Recommendations**

- Start with 5-10 random trials
- Run 20-50 Bayesian optimization trials
- Use early stopping (patience=2-3)
- Enable pruning for bad trials
- Track all experiments (MLflow, Weights & Biases)

### Common Pitfalls

**❌ Don't:**

- Use linear scale for learning rate
- Search too wide ranges
- Ignore early stopping
- Run too few trials (<15)
- Trust single run (use multiple seeds)

**✓ Do:**

- Use log scale for multiplicative parameters
- Narrow search space with domain knowledge
- Always use early stopping
- Run 20-50 trials minimum
- Validate on held-out test set

---

## Further Resources

1. Optuna Documentation: [https://optuna.org](https://optuna.org)
2. "Algorithms for Hyper-Parameter Optimization" (Bergstra et al.)
3. "Taking the Human Out of the Loop: A Review of Bayesian Optimization"
4. Ray Tune: [https://docs.ray.io/en/latest/tune/](https://docs.ray.io/en/latest/tune/)

---

**End of Lecture**

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