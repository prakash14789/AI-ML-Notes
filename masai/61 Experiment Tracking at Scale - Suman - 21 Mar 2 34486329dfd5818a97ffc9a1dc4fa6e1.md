# 61. Experiment Tracking at Scale - Suman - 21 Mar 2026

# Experiment Tracking at Scale - Lecture Notes

## PPT File: [Click Here](https://docs.google.com/presentation/d/1hR1bGEJg6KqZJKEjF1UhVHF79mMZZBQp/edit?usp=sharing&ouid=109670361591839907311&rtpof=true&sd=true)

## Colab File: [Click Here](https://drive.google.com/file/d/1nFh7FG8_kcVFOb7Kc_Uoz0p7F0fhiL1-/view?usp=sharing)

**Program:** Vishlesan i-Hub IIT Patna x Masai School — AIM (AI & Machine Learning)

**Duration:** 2 hours (120 minutes)

**Prerequisites:** PyTorch/TensorFlow, ML basics, Python programming

**Difficulty:** Intermediate to Advanced

---

## Session Overview

This session covers production-grade experiment tracking using Weights & Biases (W&B). You'll learn to track thousands of experiments, run intelligent hyperparameter sweeps, version datasets, and create collaborative dashboards for team coordination.

---

## Learning Objectives

By the end of this session, you will be able to:

1. **Integrate W&B tracking** into training pipelines with minimal code
2. **Run hyperparameter sweeps** using Bayesian optimization
3. **Version datasets** with artifacts for reproducibility
4. **Create custom dashboards** for experiment visualization
5. **Compare experiments** systematically across metrics
6. **Implement team workflows** with shared projects
7. **Deploy production tracking** for MLOps pipelines

---

## Part 1: W&B Fundamentals (15 minutes)

### Why Experiment Tracking Matters

**The problem scale:**

```
Typical ML project:
- Hyperparameters: 10-20 variables
- Values per variable: 5-10 options
- Search space: 10^10 - 10^20 combinations

Experiments run: 100-10,000
Metrics tracked: 10-100 per experiment
Total data points: 1,000 - 1,000,000

```

**Manual tracking fails at:**

- ~10 experiments (spreadsheet gets unwieldy)
- ~50 experiments (copy-paste errors accumulate)
- ~100 experiments (impossible to find best config)

### W&B Architecture

**Components:**

1. **wandb.init()** - Initialize tracking for a run
2. **wandb.config** - Log hyperparameters
3. **wandb.log()** - Log metrics during training
4. **wandb.Artifact** - Version datasets/models
5. **wandb.sweep** - Run hyperparameter searches

**Data flow:**

```
Training Script → W&B SDK → W&B Cloud → Dashboard
                              ↓
                         Storage
                    (experiments, artifacts)

```

### Basic Integration

**Minimal example:**

```python
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 1. Initialize W&B
wandb.init(
    project="image-classification",
    name="baseline-resnet18",
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "architecture": "resnet18",
        "dataset": "cifar10"
    }
)

# 2. Access config
config = wandb.config

# 3. Training loop
model = create_model(config.architecture)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

for epoch in range(config.epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 4. Log metrics
        if batch_idx % 10 == 0:
            wandb.log({
                "loss": loss.item(),
                "epoch": epoch,
                "batch": batch_idx
            })
    
    # Validation
    val_acc, val_loss = validate(model, val_loader)
    wandb.log({
        "val_accuracy": val_acc,
        "val_loss": val_loss,
        "epoch": epoch
    })

# 5. Finish run
wandb.finish()

```

**What gets tracked automatically:**

- System info (GPU, CPU, RAM)
- Git commit hash (if in git repo)
- Command line arguments
- Console output (stdout/stderr)
- Execution time

### Advanced Logging

**Logging different types of data:**

```python
import wandb
import matplotlib.pyplot as plt
import numpy as np

# Initialize
run = wandb.init(project="advanced-tracking")

# Log scalars
wandb.log({"accuracy": 0.95, "loss": 0.23})

# Log images
image = torch.randn(3, 224, 224)
wandb.log({"example_image": wandb.Image(image)})

# Log plots
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
wandb.log({"training_curve": wandb.plot.line_series(
    xs=[[1, 2, 3]],
    ys=[[4, 5, 6]],
    keys=["accuracy"],
    title="Training Accuracy"
)})

# Log histograms
weights = model.fc.weight.data.cpu().numpy()
wandb.log({"weight_histogram": wandb.Histogram(weights)})

# Log tables
table = wandb.Table(
    columns=["epoch", "loss", "accuracy"],
    data=[[1, 0.5, 0.8], [2, 0.3, 0.85], [3, 0.2, 0.9]]
)
wandb.log({"results_table": table})

# Log custom charts
wandb.log({"pr_curve": wandb.plot.pr_curve(
    y_true, y_scores, labels=["cat", "dog"]
)})

# Log confusion matrix
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
    probs=predictions,
    y_true=targets,
    class_names=["cat", "dog", "bird"]
)})

```

### Watch Models

**Automatic gradient and parameter tracking:**

```python
import wandb

wandb.init(project="model-monitoring")

model = create_model()

# Watch model - logs gradients and parameters
wandb.watch(
    model,
    log="all",        # Log gradients and parameters
    log_freq=100,     # Log every 100 batches
    log_graph=True    # Log computational graph
)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        loss = train_step(model, batch)
        # Gradients automatically logged!

```

**What `wandb.watch()` tracks:**

- Parameter histograms (layer weights over time)
- Gradient histograms (gradient flow)
- Gradient norms (detect vanishing/exploding gradients)
- Model topology (computational graph)

---

## Part 2: Hyperparameter Sweeps (30 minutes)

### Sweep Configuration

**Defining a sweep:**

```python
sweep_config = {
    'method': 'bayes',  # or 'grid', 'random'
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-1
        },
        'batch_size': {
            'values': [16, 32, 64, 128]
        },
        'optimizer': {
            'values': ['adam', 'adamw', 'sgd']
        },
        'weight_decay': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
        },
        'epochs': {
            'value': 10  # Fixed value
        }
    }
}

# Create sweep
sweep_id = wandb.sweep(sweep_config, project="my-project")

```

### Search Methods

**1. Grid Search:**

```python
sweep_config = {
    'method': 'grid',
    'parameters': {
        'learning_rate': {'values': [0.001, 0.01, 0.1]},
        'batch_size': {'values': [16, 32, 64]}
    }
}
# Tries all 3 × 3 = 9 combinations

```

**2. Random Search:**

```python
sweep_config = {
    'method': 'random',
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-1
        },
        'batch_size': {'values': [16, 32, 64, 128]}
    }
}
# Tries random combinations until stopped

```

**3. Bayesian Optimization:**

```python
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-1
        }
    }
}
# Intelligently explores space using Gaussian processes

```

### Parameter Distributions

**Available distributions:**

```python
# Uniform (linear scale)
'param1': {
    'distribution': 'uniform',
    'min': 0,
    'max': 1
}

# Log-uniform (logarithmic scale, good for learning rates)
'learning_rate': {
    'distribution': 'log_uniform_values',
    'min': 1e-5,
    'max': 1e-1
}

# Normal distribution
'param2': {
    'distribution': 'normal',
    'mu': 0,
    'sigma': 1
}

# Categorical (discrete choices)
'optimizer': {
    'values': ['adam', 'sgd', 'rmsprop']
}

# Integer range
'num_layers': {
    'distribution': 'int_uniform',
    'min': 2,
    'max': 10
}

# Q-uniform (quantized)
'dropout': {
    'distribution': 'q_uniform',
    'min': 0.1,
    'max': 0.9,
    'q': 0.1  # Steps of 0.1
}

```

### Training Function for Sweeps

**Sweep-compatible training:**

```python
import wandb

def train():
    # Initialize run with sweep
    run = wandb.init()
    
    # Get hyperparameters from sweep
    config = wandb.config
    
    # Build model with sweep params
    model = create_model(
        lr=config.learning_rate,
        batch_size=config.batch_size,
        optimizer=config.optimizer
    )
    
    # Training loop
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader)
        val_acc = validate(model, val_loader)
        
        # Log metrics (used by sweep for optimization)
        wandb.log({
            'train_loss': train_loss,
            'val_accuracy': val_acc,
            'epoch': epoch
        })

# Run sweep
sweep_id = wandb.sweep(sweep_config, project="my-project")
wandb.agent(sweep_id, function=train, count=50)  # Run 50 experiments

```

### Running Parallel Agents

**Multiple GPUs, same sweep:**

```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python train.py --sweep_id <sweep_id>

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python train.py --sweep_id <sweep_id>

# Terminal 3 (GPU 2)
CUDA_VISIBLE_DEVICES=2 python train.py --sweep_id <sweep_id>

# All agents pull from same sweep queue
# Bayesian optimization shares learnings across all agents

```

**Python script for agents:**

```python
import wandb
import argparse

def train():
    run = wandb.init()
    # Training code...
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', type=str, required=True)
    args = parser.parse_args()
    
    # This agent will run indefinitely, pulling configs from sweep
    wandb.agent(args.sweep_id, function=train)

```

### Early Stopping for Sweeps

**Hyperband scheduling:**

```python
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'early_terminate': {
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 27
    },
    'parameters': {
        'learning_rate': {'min': 1e-5, 'max': 1e-1}
    }
}

```

**How Hyperband works:**

1. Starts many configs with small budgets
2. Progressively eliminates poor performers
3. Allocates more resources to promising configs
4. Finds good configs faster than trying all equally

**Example:**

```
Round 1: Run 81 configs for 1 epoch each
         Keep top 27 (eliminate bottom 54)

Round 2: Run 27 configs for 3 epochs each
         Keep top 9 (eliminate bottom 18)

Round 3: Run 9 configs for 9 epochs each
         Keep top 3 (eliminate bottom 6)

Round 4: Run 3 configs for 27 epochs each
         Select best

Total epochs: 81×1 + 27×3 + 9×9 + 3×27 = 81 + 81 + 81 + 81 = 324 epochs
vs. Grid search: 81 configs × 27 epochs = 2,187 epochs
Savings: 6.7× faster!

```

---

## Part 3: Dataset Versioning with Artifacts (25 minutes)

### Creating Artifacts

**Basic artifact creation:**

```python
import wandb

run = wandb.init(project="my-project")

# Create artifact
artifact = wandb.Artifact(
    name='cifar10-dataset',
    type='dataset',
    description='CIFAR-10 training data with augmentations',
    metadata={
        'num_samples': 50000,
        'num_classes': 10,
        'augmentations': ['flip', 'crop', 'normalize']
    }
)

# Add files
artifact.add_file('data/train.csv')
artifact.add_dir('data/images/')

# Log artifact
run.log_artifact(artifact)

```

### Versioning Strategy

**Automatic versioning:**

```python
# First version
artifact_v1 = wandb.Artifact('my-dataset', type='dataset')
artifact_v1.add_file('data_v1.csv')
run.log_artifact(artifact_v1)  # Creates v0

# Update dataset
artifact_v2 = wandb.Artifact('my-dataset', type='dataset')
artifact_v2.add_file('data_v2.csv')  # Different content
run.log_artifact(artifact_v2)  # Creates v1 (auto-increments)

# Same content = no new version
artifact_v3 = wandb.Artifact('my-dataset', type='dataset')
artifact_v3.add_file('data_v2.csv')  # Same content as v1
run.log_artifact(artifact_v3)  # Still v1 (deduplication)

```

**Version naming:**

- v0, v1, v2, ... (automatic incrementing)
- Aliases: "latest", "best", "production"

### Using Artifacts in Training

**Download and use artifacts:**

```python
import wandb

run = wandb.init(project="my-project")

# Download specific version
artifact = run.use_artifact('my-dataset:v2')
artifact_dir = artifact.download()

# Load data from artifact
import pandas as pd
df = pd.read_csv(f'{artifact_dir}/data.csv')

# Train with this specific version
# All logged with experiment
train_model(df)

```

**Automatic lineage tracking:**

```
Experiment #123 used:
  - Dataset: my-dataset:v2
  - Model: resnet18:v1
  
Experiment #124 used:
  - Dataset: my-dataset:v3  ← Different version!
  - Model: resnet18:v1
  
Can always reproduce #123 by loading exact artifacts

```

### Dataset Preprocessing Pipeline

**Track preprocessing as artifacts:**

```python
import wandb
import pandas as pd

def preprocess_data(raw_data_path, output_path):
    run = wandb.init(project="data-pipeline", job_type="preprocessing")
    
    # Use raw data artifact
    raw_artifact = run.use_artifact('raw-data:latest')
    raw_dir = raw_artifact.download()
    
    # Preprocess
    df = pd.read_csv(f'{raw_dir}/data.csv')
    df_clean = clean_data(df)
    df_augmented = augment_data(df_clean)
    
    # Save processed data
    df_augmented.to_csv(output_path, index=False)
    
    # Create processed artifact
    processed_artifact = wandb.Artifact(
        name='processed-data',
        type='dataset',
        metadata={
            'preprocessing_steps': ['clean', 'augment'],
            'num_samples': len(df_augmented)
        }
    )
    processed_artifact.add_file(output_path)
    
    # Link to raw data (lineage)
    processed_artifact.add_reference(raw_artifact)
    
    run.log_artifact(processed_artifact)
    run.finish()

```

**Lineage visualization:**

```
raw-data:v0 → [preprocess] → processed-data:v0 → [train] → model:v0
                                                           → results

Can trace back: "Model v0 came from processed-data v0 from raw-data v0"

```

### Model Artifacts

**Versioning trained models:**

```python
import wandb
import torch

run = wandb.init(project="my-project")

# Train model
model = train_model()

# Save model
torch.save(model.state_dict(), 'model.pth')

# Create model artifact
model_artifact = wandb.Artifact(
    name='my-model',
    type='model',
    metadata={
        'architecture': 'resnet18',
        'accuracy': 0.95,
        'framework': 'pytorch'
    }
)
model_artifact.add_file('model.pth')

# Link to dataset used
dataset_artifact = run.use_artifact('my-dataset:v2')
model_artifact.add_reference(dataset_artifact)

run.log_artifact(model_artifact)

```

### Artifact References

**Linking artifacts without duplication:**

```python
# Large dataset stored once
dataset_artifact = wandb.Artifact('large-dataset', type='dataset')
dataset_artifact.add_dir('s3://my-bucket/data/')  # External reference

# Multiple experiments reference same data
for i in range(10):
    run = wandb.init(project="experiments")
    data = run.use_artifact('large-dataset:latest')
    # Data not duplicated, just referenced
    train_model(data)

```

---

## Part 4: Dashboards and Visualization (20 minutes)

### Built-in Visualizations

**Automatic charts:**

- Line plots (metrics over time)
- Parallel coordinates (compare configs)
- Scatter plots (hyperparameter vs metric)
- Confusion matrices
- PR curves, ROC curves
- Distribution plots

**Comparing runs:**

```python
# W&B automatically creates comparison views
# Access via UI: Compare multiple runs
# - Side-by-side config comparison
# - Overlaid metric plots
# - Performance tables

```

### Custom Charts

**Panel types:**

```python
import wandb

# Line plot
wandb.log({
    "custom_plot": wandb.plot.line(
        table=wandb.Table(data=data, columns=["x", "y"]),
        x="x",
        y="y",
        title="Custom Line Plot"
    )
})

# Scatter plot
wandb.log({
    "scatter": wandb.plot.scatter(
        table=wandb.Table(data=data, columns=["x", "y", "label"]),
        x="x",
        y="y",
        title="Hyperparameter Correlation"
    )
})

# Bar chart
wandb.log({
    "bar_chart": wandb.plot.bar(
        table=wandb.Table(data=data, columns=["category", "value"]),
        label="category",
        value="value",
        title="Performance by Category"
    )
})

# Histogram
wandb.log({
    "histogram": wandb.plot.histogram(
        table=wandb.Table(data=data, columns=["values"]),
        value="values",
        title="Value Distribution"
    )
})

```

### Reports

**Creating shareable reports:**

```python
# Via Python API
import wandb

api = wandb.Api()
report = wandb.apis.reports.Report(
    project="my-project",
    title="Q1 Model Performance",
    description="Summary of experiments in Q1"
)

# Add panels
report.blocks = [
    wandb.apis.reports.PanelGrid(
        panels=[
            wandb.apis.reports.LinePlot(
                title="Training Loss",
                x="Step",
                y=["train_loss", "val_loss"]
            ),
            wandb.apis.reports.ScatterPlot(
                title="Hyperparameter Impact",
                x="learning_rate",
                y="val_accuracy"
            )
        ]
    ),
    wandb.apis.reports.MarkdownBlock(
        text="## Key Findings\n\n- Best LR: 0.001\n- Optimal batch: 32"
    )
]

report.save()

```

**Or create via UI:**

1. Select runs to include
2. Add panels (plots, tables, markdown)
3. Arrange layout
4. Share URL (public or team-only)

### Workspace Organization

**Project structure:**

```
company/
├── research/
│   ├── nlp-experiments/
│   ├── cv-experiments/
│   └── rl-experiments/
├── production/
│   ├── model-monitoring/
│   └── data-pipelines/
└── team-experiments/
    ├── alice-exploration/
    └── bob-exploration/

```

**Tags and grouping:**

```python
wandb.init(
    project="experiments",
    name="resnet-sweep-1",
    tags=["resnet", "sweep", "baseline"],
    group="architecture-comparison",
    job_type="train"
)

```

**Filtering in UI:**

```
Filter runs by:
- Tags: "baseline" AND "resnet"
- Config: learning_rate > 0.001
- Metrics: val_accuracy > 0.9
- State: finished, running, crashed
- Created: last 7 days

```

---

## Part 5: Advanced Features (20 minutes)

### W&B Python API

**Programmatic access to runs:**

```python
import wandb

api = wandb.Api()

# Get all runs from project
runs = api.runs("username/project")

# Filter runs
best_runs = [r for r in runs if r.summary.get("val_accuracy", 0) > 0.9]

# Get specific run
run = api.run("username/project/run_id")

# Access data
config = run.config
summary = run.summary  # Final metrics
history = run.history()  # All logged data as DataFrame

# Download files
for file in run.files():
    file.download(replace=True)

# Get best run
best_run = max(runs, key=lambda r: r.summary.get("val_accuracy", 0))
print(f"Best accuracy: {best_run.summary['val_accuracy']}")
print(f"Config: {best_run.config}")

```

### Analysis Example

**Compare hyperparameters systematically:**

```python
import wandb
import pandas as pd
import matplotlib.pyplot as plt

api = wandb.Api()
runs = api.runs("my-project")

# Extract data
data = []
for run in runs:
    if run.state == "finished":
        data.append({
            'learning_rate': run.config.get('learning_rate'),
            'batch_size': run.config.get('batch_size'),
            'accuracy': run.summary.get('val_accuracy'),
            'training_time': run.summary.get('training_time')
        })

df = pd.DataFrame(data)

# Analyze
print("Best learning rate:")
print(df.groupby('learning_rate')['accuracy'].mean().sort_values())

print("\nBest batch size:")
print(df.groupby('batch_size')['accuracy'].mean().sort_values())

# Visualize
plt.scatter(df['learning_rate'], df['accuracy'])
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Accuracy')
plt.title('Hyperparameter Impact')
plt.show()

```

### Alerts and Notifications

**Set up alerts:**

```python
# Via Python
import wandb

run = wandb.init(project="my-project")

# Alert if metric exceeds threshold
if val_accuracy > 0.95:
    wandb.alert(
        title="High Accuracy Achieved",
        text=f"Model achieved {val_accuracy:.2%} accuracy",
        level=wandb.AlertLevel.INFO
    )

# Alert on failure
if val_loss > 10.0:
    wandb.alert(
        title="Training Diverged",
        text="Loss exceeded 10.0, training likely failed",
        level=wandb.AlertLevel.WARN
    )

```

**Configure via UI:**

- Email notifications
- Slack integration
- Webhook triggers

### Multi-GPU Tracking

**Track distributed training:**

```python
import torch.distributed as dist
import wandb

def main(rank, world_size):
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Only rank 0 logs to W&B
    if rank == 0:
        wandb.init(project="distributed-training")
    
    # Training loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, rank)
        
        # Aggregate metrics across GPUs
        train_loss_tensor = torch.tensor(train_loss).cuda()
        dist.all_reduce(train_loss_tensor)
        avg_loss = train_loss_tensor.item() / world_size
        
        # Log from rank 0
        if rank == 0:
            wandb.log({"train_loss": avg_loss, "epoch": epoch})

if __name__ == "__main__":
    world_size = 4  # 4 GPUs
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)

```

### Integration with Other Tools

**MLflow compatibility:**

```python
import mlflow
import wandb

# Log to both MLflow and W&B
wandb.init(project="my-project")
mlflow.start_run()

# Training
wandb.log({"loss": loss})
mlflow.log_metric("loss", loss)

```

**Hydra configuration:**

```python
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Convert Hydra config to dict
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Initialize W&B with Hydra config
    wandb.init(
        project="my-project",
        config=config_dict
    )
    
    # Training...

```

---

## Part 6: Production MLOps (15 minutes)

### Model Registry

**Track production models:**

```python
import wandb

# Training
run = wandb.init(project="production")
model = train_model()

# Save as artifact
model_artifact = wandb.Artifact(
    'product-classifier',
    type='model',
    metadata={'accuracy': 0.95}
)
model_artifact.add_file('model.pth')
run.log_artifact(model_artifact, aliases=['latest', 'staging'])

# Promote to production
artifact = run.use_artifact('product-classifier:staging')
artifact.aliases.append('production')
artifact.save()

```

**Deployment workflow:**

```
1. Train model → Log as artifact with alias "staging"
2. Validation passes → Add alias "production"
3. Deploy service loads artifact "production"
4. New model → Replace "production" alias
5. Rollback → Revert "production" to previous version

```

### Monitoring Production Models

**Track model performance over time:**

```python
import wandb

# Production inference service
def predict(data):
    # Load production model
    run = wandb.init(project="production-monitoring", job_type="inference")
    model_artifact = run.use_artifact('product-classifier:production')
    model = load_model(model_artifact)
    
    # Predict
    prediction = model(data)
    
    # Log prediction distribution
    wandb.log({
        "prediction_confidence": prediction.max(),
        "prediction_class": prediction.argmax()
    })
    
    return prediction

# Batch monitoring
def monitor_batch(predictions, labels):
    run = wandb.init(project="production-monitoring", job_type="monitoring")
    
    accuracy = (predictions == labels).mean()
    
    wandb.log({
        "batch_accuracy": accuracy,
        "num_samples": len(predictions),
        "timestamp": datetime.now()
    })
    
    # Alert if accuracy drops
    if accuracy < 0.85:
        wandb.alert(
            title="Model Performance Degraded",
            text=f"Accuracy dropped to {accuracy:.2%}",
            level=wandb.AlertLevel.WARN
        )

```

### Data Drift Detection

**Monitor input distribution shifts:**

```python
import wandb
import numpy as np

def log_data_statistics(data, labels, split="train"):
    run = wandb.init(project="data-monitoring")
    
    # Log distributions
    wandb.log({
        f"{split}/feature_mean": data.mean(axis=0).tolist(),
        f"{split}/feature_std": data.std(axis=0).tolist(),
        f"{split}/label_distribution": wandb.Histogram(labels)
    })
    
    # Log sample images
    wandb.log({
        f"{split}/samples": [wandb.Image(img) for img in data[:10]]
    })

# Compare distributions over time
# W&B dashboard shows drift

```

### CI/CD Integration

**Automated testing with W&B:**

```python
# .github/workflows/train.yml
name: Train Model

on: [push]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Train model
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: python train.py
      - name: Check performance
        run: python check_performance.py

```

**Performance check script:**

```python
import wandb
import sys

api = wandb.Api()
runs = api.runs("my-project")
latest_run = runs[0]

# Check if performance meets threshold
accuracy = latest_run.summary.get("val_accuracy", 0)
if accuracy < 0.90:
    print(f"Performance below threshold: {accuracy:.2%}")
    sys.exit(1)  # Fail CI/CD

print(f"Performance acceptable: {accuracy:.2%}")

```

---

## Part 7: Best Practices (10 minutes)

### Naming Conventions

**Good practices:**

```python
# Descriptive project names
project="image-classification-production"  # Good
project="test"  # Bad

# Informative run names
name=f"resnet18-lr{lr}-bs{batch_size}-{timestamp}"  # Good
name="run1"  # Bad

# Clear artifact names
artifact_name="cifar10-train-augmented-v2"  # Good
artifact_name="data"  # Bad

# Useful tags
tags=["baseline", "resnet", "augmented", "high-priority"]  # Good
tags=["exp"]  # Bad

```

### Configuration Management

**Store everything in config:**

```python
config = {
    # Model
    'architecture': 'resnet18',
    'num_layers': 18,
    'pretrained': True,
    
    # Training
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'optimizer': 'adamw',
    'weight_decay': 0.01,
    'scheduler': 'cosine',
    
    # Data
    'dataset': 'cifar10',
    'augmentation': ['flip', 'crop', 'normalize'],
    'train_split': 0.8,
    'val_split': 0.2,
    
    # System
    'num_workers': 4,
    'pin_memory': True,
    'mixed_precision': True,
    
    # Random seeds
    'seed': 42
}

wandb.init(project="my-project", config=config)

```

### Metric Logging Strategy

**What to log:**

```python
# Every step (can be expensive, log sparingly)
wandb.log({"train_loss": loss}, step=global_step)

# Every epoch
wandb.log({
    "epoch": epoch,
    "train_loss": avg_train_loss,
    "train_acc": train_accuracy,
    "val_loss": val_loss,
    "val_acc": val_accuracy,
    "learning_rate": current_lr
})

# End of training
wandb.summary.update({
    "best_val_acc": best_accuracy,
    "best_epoch": best_epoch,
    "total_training_time": total_time,
    "final_model_size": model_size_mb
})

```

### Cost Optimization

**Reduce logging overhead:**

```python
# Log less frequently
if step % 100 == 0:  # Every 100 steps, not every step
    wandb.log({"loss": loss})

# Don't log large objects frequently
# Bad: wandb.log({"activations": huge_tensor}) every step
# Good: wandb.log({"activations": wandb.Histogram(huge_tensor)}) every epoch

# Use offline mode for slow networks
wandb.init(mode="offline")
# Later: wandb sync <run_dir>

```

---

## Key Takeaways

### Core Concepts

1. **Automatic tracking** beats manual tracking at >10 experiments
2. **Bayesian sweeps** find optimal configs 10-100× faster than grid search
3. **Dataset versioning** is essential for reproducibility
4. **Dashboards** enable team coordination at scale
5. **Artifacts** provide full lineage tracking

### When to Use What

Use Case | Solution
Track single experiment | wandb.init() + wandb.log()
Find best hyperparameters | wandb.sweep() with Bayesian
Version datasets | wandb.Artifact()
Compare experiments | W&B Dashboard
Team coordination | Shared projects + reports
Production monitoring | Model registry + alerts

### Common Pitfalls

1. **Not logging enough config** → Can't reproduce
2. **Logging too frequently** → Expensive, slow
3. **No dataset versioning** → Different results on "same" data
4. **Poor naming** → Can't find experiments later
5. **No sweep early stopping** → Waste compute on bad configs

---

**End of Lecture Notes**

**Vishlesan i-Hub IIT Patna × Masai School**

*From chaos to clarity: Professional experiment tracking for production ML*

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