# 65. Fine-Tuning LLMs & AutoML - Suman - 7 Apr 2026

# Lecture Notes: Fine-Tuning LLMs & AutoML

## PPT File: [Click Here](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/f85498aa-5f6d-49e2-bb68-f93734501453/udicMj1nZI1X0SNI.pdf)

## Session Overview

This session covers parameter-efficient fine-tuning techniques for Large Language Models (LLMs), focusing on LoRA, QLoRA, and automated hyperparameter optimization with Optuna. You'll learn to fine-tune billion-parameter models on consumer hardware while maintaining quality.

**Duration:** 3 hours

**Prerequisites:** Deep learning basics, transformers architecture, PyTorch fundamentals

## Learning Objectives

By the end of this session, you will be able to:

1. **Understand the memory challenges** of full fine-tuning
2. **Implement LoRA** for parameter-efficient training
3. **Apply QLoRA** for memory-constrained scenarios
4. **Use Optuna** for hyperparameter optimization
5. **Fine-tune models** for FAQ and domain-specific tasks
6. **Deploy LoRA adapters** in production

---

## Part 1: The Fine-Tuning Challenge

### Why Fine-Tuning?

Pre-trained LLMs (GPT, LLaMA, BLOOM) are trained on general text. Fine-tuning adapts them to:

- **Domain-specific tasks** (medical, legal, coding)
- **Instruction following** (chatbots, assistants)
- **Custom knowledge bases** (company documentation, FAQs)
- **Style adaptation** (formal, casual, technical)

### Traditional Fine-Tuning Memory Requirements

**For a 7B parameter model:**

```
Memory Components:

1. Model Weights (FP32):
   7B parameters × 4 bytes = 28GB

2. Optimizer States (Adam):
   - First moment: 28GB
   - Second moment: 28GB
   - Total: 56GB

3. Gradients:
   7B parameters × 4 bytes = 28GB

4. Activation Memory (forward pass):
   Varies by batch size
   Typical: 8-12GB

Total Memory: ~112-120GB

```

**Problem:** Most researchers don't have A100 80GB GPUs!

**Comparison:**

GPU | Memory | Can Full Fine-Tune 7B?
RTX 3090 | 24GB | ❌ No
RTX 4090 | 24GB | ❌ No
A100 (40GB) | 40GB | ❌ No
A100 (80GB) | 80GB | ✅ Yes (barely)
H100 | 80GB | ✅ Yes

### Alternative: Parameter-Efficient Fine-Tuning (PEFT)

**Core Idea:** Don't update all parameters—only train a small subset.

**PEFT Methods:**

1. **Adapter Layers** - Insert small trainable layers between frozen layers
2. **Prefix Tuning** - Optimize continuous task-specific vectors
3. **LoRA** - Low-rank decomposition of weight updates
4. **QLoRA** - LoRA + quantization

---

## Part 2: LoRA (Low-Rank Adaptation)

### Mathematical Foundation

**Weight Update Decomposition:**

In traditional fine-tuning, we update weights:

Wnew=Wpretrained+ΔWW_{new} = W_{pretrained} + \Delta WWnew=Wpretrained+ΔW

where ΔW\Delta WΔW is a full-rank matrix.

**LoRA Hypothesis:** Weight updates during fine-tuning have low "intrinsic rank"

Instead of learning full ΔW∈Rd×d\Delta W \in \mathbb{R}^{d \times d}ΔW∈Rd×d, learn:

ΔW=BA\Delta W = BAΔW=BA

where:

- B∈Rd×rB \in \mathbb{R}^{d \times r}B∈Rd×r (down-projection)
- A∈Rr×dA \in \mathbb{R}^{r \times d}A∈Rr×d (up-projection)
- r≪dr \ll dr≪d (rank is much smaller than dimension)

**Parameter Count:**

```
Full rank: d × d parameters
Low rank: d × r + r × d = 2dr parameters

Example (d=4096, r=16):
Full: 4096 × 4096 = 16,777,216
LoRA: 2 × 4096 × 16 = 131,072

Reduction: 128× fewer parameters!

```

### LoRA Architecture

**Transformer Layer with LoRA:**

```
Standard Transformer:
x → Linear(W) → output

LoRA Transformer:
x → Linear(W_frozen) + LoRA(BA) → output
         ↑                ↑
      Frozen          Trainable

Forward Pass:
h = W_frozen × x + B × (A × x) × α/r

Where:
- W_frozen: Pre-trained weights (frozen)
- A, B: LoRA matrices (trainable)
- α: Scaling factor (typically = r)
- r: Rank

```

**Which Layers to Apply LoRA?**

```
Transformer Block:
┌─────────────────────────┐
│ Multi-Head Attention    │
│   Q, K, V, O ← LoRA    │ ✓ Most important
├─────────────────────────┤
│ Feed-Forward Network    │
│   W1, W2 ← LoRA       │ ✓ Optional
└─────────────────────────┘

Common configurations:
1. Q, V only (minimal)
2. Q, K, V, O (standard)
3. All linear layers (maximum)

```

### LoRA Hyperparameters

**Key Parameters:**

**1. Rank (r)**

```python
# Low rank (faster, less expressive)
r = 4   # 4 dimensions

# Medium rank (balanced)
r = 16  # Default choice

# High rank (slower, more expressive)
r = 64  # For complex tasks

```

**Trade-offs:**

Rank | Parameters | Training Speed | Quality | Use Case
4 | Smallest | Fastest | Good | Simple tasks
8-16 | Small | Fast | Very good | Most tasks
32-64 | Medium | Moderate | Excellent | Complex tasks
128+ | Large | Slow | Marginal gains | Rarely needed

**2. Alpha (α)**

Scaling factor that controls update magnitude:

```python
lora_alpha = 16  # Typically 1× to 2× the rank

# Effective learning rate scaling:
scaling = alpha / r

Example:
r = 8, alpha = 16 → scaling = 2.0
r = 16, alpha = 16 → scaling = 1.0
r = 32, alpha = 16 → scaling = 0.5

```

**Rule of thumb:** Start with `alpha = r`

**3. Dropout**

```python
lora_dropout = 0.05  # Light regularization
lora_dropout = 0.1   # Standard
lora_dropout = 0.2   # Heavy (for small datasets)

```

### LoRA Implementation (PyTorch)

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Scaling
        self.scaling = alpha / rank
        
        # Initialize A with kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    
    def forward(self, x):
        # x shape: (batch, seq_len, in_features)
        # LoRA forward: x @ A @ B
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return lora_output

class LinearWithLoRA(nn.Module):
    def __init__(self, linear_layer, rank=16, alpha=16):
        super().__init__()
        
        # Freeze original weights
        self.linear = linear_layer
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # Add LoRA
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        self.lora = LoRALayer(in_features, out_features, rank, alpha)
    
    def forward(self, x):
        # Frozen linear + LoRA adaptation
        return self.linear(x) + self.lora(x)

```

### LoRA with HuggingFace PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,                        # Rank
    lora_alpha=16,               # Scaling
    target_modules=[             # Which layers
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.05,           # Dropout
    bias="none",                 # Don't train biases
    task_type="CAUSAL_LM"        # Task type
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%

```

---

## Part 3: QLoRA (Quantized LoRA)

### Quantization Basics

**Precision Formats:**

Format | Bits | Range | Precision
FP32 | 32 | ±3.4e38 | ~7 digits
FP16 | 16 | ±65,504 | ~3 digits
INT8 | 8 | -128 to 127 | Integer
INT4 | 4 | -8 to 7 | Integer
NF4 | 4 | Normalized | Special

**Memory Impact:**

```
7B model in different formats:

FP32: 7B × 4 bytes = 28GB
FP16: 7B × 2 bytes = 14GB
INT8: 7B × 1 byte = 7GB
INT4: 7B × 0.5 bytes = 3.5GB

Savings: 8× from FP32 to INT4!

```

### QLoRA Innovations

**1. NormalFloat4 (NF4)**

Standard INT4 is uniform: -8, -7, ..., 6, 7

NF4 is **information-theoretically optimal** for normally distributed weights:

```python
# NF4 quantization levels
NF4_LEVELS = [
    -1.0, -0.6961928009986877, -0.5250730514526367,
    -0.39491748809814453, -0.28444138169288635,
    -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725,
    0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0
]

# These levels are optimized for weights following N(0, σ²)

```

**Why NF4 works:**

Neural network weights after training follow approximately normal distribution. NF4 allocates more quantization levels near zero (where most weights cluster) and fewer at extremes.

**2. Double Quantization**

Quantize the quantization constants themselves!

```
Standard quantization:
- Values: Quantized to 4-bit
- Scale factors: Stored in FP32
- Memory: N×0.5 + N/64×4 bytes

Double quantization:
- Values: Quantized to 4-bit
- Scale factors: Quantized to 8-bit
- Memory: N×0.5 + N/64×1 bytes

Savings: ~0.4 bytes per parameter
For 7B model: 2.8GB → 2.4GB

```

**3. Paged Optimizers**

Handle memory spikes during training:

```
Problem: CUDA out-of-memory during gradient updates

Solution: Page optimizer states to CPU RAM
- Keep on GPU: Active gradients
- Move to CPU: Inactive optimizer states
- Swap as needed

```

### QLoRA Configuration

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Use 4-bit
    bnb_4bit_quant_type="nf4",           # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in BF16
    bnb_4bit_use_double_quant=True       # Double quantization
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Add LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

```

### Memory Comparison

```
7B Model Fine-Tuning Memory:

Full Fine-Tuning (FP32):
- Model: 28GB
- Optimizer: 56GB
- Gradients: 28GB
- Activations: 12GB
Total: ~124GB

LoRA (FP16):
- Model: 14GB (frozen, FP16)
- LoRA params: 0.3GB (trainable)
- Optimizer: 0.6GB (only LoRA)
- Gradients: 0.3GB (only LoRA)
- Activations: 8GB
Total: ~23GB

QLoRA (4-bit):
- Model: 3.5GB (frozen, NF4)
- LoRA params: 0.3GB (trainable, FP16)
- Optimizer: 0.6GB (only LoRA)
- Gradients: 0.3GB (only LoRA)
- Activations: 4GB (quantized)
Total: ~9GB

Fits on RTX 3090! (24GB)

```

---

## Part 4: Hyperparameter Optimization with Optuna

### The Hyperparameter Search Problem

**Hyperparameters in LLM Fine-Tuning:**

```python
hyperparameters = {
    # Model architecture
    'lora_rank': [4, 8, 16, 32, 64],
    'lora_alpha': [8, 16, 32, 64],
    'lora_dropout': [0.0, 0.05, 0.1],
    
    # Training
    'learning_rate': [1e-5, 3e-5, 5e-5, 1e-4],
    'batch_size': [4, 8, 16],
    'num_epochs': [3, 5, 10],
    'warmup_ratio': [0.03, 0.06, 0.1],
    
    # Regularization
    'weight_decay': [0.0, 0.01, 0.1],
    'max_grad_norm': [0.3, 1.0, 2.0]
}

# Total combinations: 5×4×3×4×3×3×3×3×3 = 29,160

```

**Search Strategies:**

Method | Trials Needed | Intelligence | Parallelizable
Grid Search | All (29,160) | None | ✓ Yes
Random Search | 100-500 | Low | ✓ Yes
Bayesian (Optuna) | 20-50 | High | Partial

### Optuna Architecture

**Core Concepts:**

**1. Study** - Optimization experiment

```python
study = optuna.create_study(
    direction="maximize",  # or "minimize"
    study_name="llm-finetuning"
)

```

**2. Trial** - Single experiment

```python
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    return train_and_eval(lr)

```

**3. Sampler** - Search algorithm

- **TPE** (Tree-structured Parzen Estimator) - Default
- **CmaEs** - Covariance Matrix Adaptation
- **Random** - Baseline

**4. Pruner** - Early stopping

```python
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,
    n_warmup_steps=100
)

```

### Tree-Structured Parzen Estimator (TPE)

**How TPE Works:**

```
1. Separate trials into "good" and "bad" based on threshold
   Good: Top 25% by performance
   Bad: Bottom 75%

2. Model hyperparameter distributions:
   P(x | good) - Distribution of params that worked well
   P(x | bad) - Distribution of params that worked poorly

3. Select next hyperparameters by maximizing:
   EI(x) = P(x | good) / P(x | bad)
   
   High EI → Likely to be good

4. Train model, add result, repeat

```

**Example:**

```
After 10 trials:

Learning rate distribution:
Good trials: Clustered around 3e-5
Bad trials: Spread across 1e-5 to 1e-3

TPE suggests: lr ≈ 3e-5 (high EI)

After 20 trials:
Good trials: lr ∈ [2e-5, 5e-5], rank ∈ [16, 32]
TPE suggests: lr=3.5e-5, rank=24

```

### Optuna for LLM Fine-Tuning

```python
import optuna
from transformers import TrainingArguments, Trainer

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    lora_r = trial.suggest_int("lora_rank", 4, 64, step=4)
    lora_alpha = trial.suggest_int("lora_alpha", 8, 64, step=8)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.1)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA to model
    model = get_peft_model(base_model, lora_config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./trial_{trial.number}",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        num_train_epochs=3,
        warmup_ratio=warmup_ratio,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    trainer.train()
    
    # Evaluate
    metrics = trainer.evaluate()
    eval_loss = metrics["eval_loss"]
    
    # Report intermediate values for pruning
    trial.report(eval_loss, step=0)
    
    # Check if trial should be pruned
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    return eval_loss

# Create study
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
)

# Optimize
study.optimize(objective, n_trials=30, timeout=86400)  # 24 hours

# Results
print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")

```

### Optuna Visualization

```python
import optuna.visualization as vis

# Plot optimization history
fig = vis.plot_optimization_history(study)
fig.show()

# Plot parameter importances
fig = vis.plot_param_importances(study)
fig.show()

# Plot parallel coordinate
fig = vis.plot_parallel_coordinate(study)
fig.show()

# Plot slice (per parameter)
fig = vis.plot_slice(study)
fig.show()

```

---

## Part 5: FAQ Model Fine-Tuning

### Dataset Preparation

**FAQ Dataset Structure:**

```json
[
  {
    "question": "How do I reset my password?",
    "answer": "Click 'Forgot Password' on the login page...",
    "category": "account"
  },
  {
    "question": "What payment methods do you accept?",
    "answer": "We accept credit cards, PayPal, and wire transfer...",
    "category": "billing"
  }
]

```

**Convert to Instruction Format:**

```python
def format_faq_instruction(example):
    """Convert FAQ to instruction-following format."""
    instruction = f"""### Question:
{example['question']}

### Answer:"""
    
    response = example['answer']
    
    # Combine for training
    text = f"{instruction}\n{response}"
    
    return {"text": text}

# Apply to dataset
dataset = dataset.map(format_faq_instruction)

```

**Tokenization:**

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # Tokenize
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    
    # For causal LM, labels = input_ids
    outputs["labels"] = outputs["input_ids"].copy()
    
    return outputs

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)

```

### Complete Fine-Tuning Pipeline

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch

# 1. Load model with QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

# 2. Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 3. Load and prepare dataset
dataset = load_dataset("json", data_files="faq_data.json")
dataset = dataset["train"].train_test_split(test_size=0.1)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized = dataset.map(tokenize, batched=True)

# 4. Training arguments
training_args = TrainingArguments(
    output_dir="./faq-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine"
)

# 5. Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# 6. Train
trainer.train()

# 7. Save adapter
model.save_pretrained("./faq-lora-adapter")

```

### Inference with Fine-Tuned Model

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "./faq-lora-adapter"
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Generate answer
def answer_faq(question):
    prompt = f"""### Question:
{question}

### Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer (after prompt)
    answer = response.split("### Answer:")[-1].strip()
    
    return answer

# Test
question = "How do I reset my password?"
answer = answer_faq(question)
print(answer)

```

---

## Part 6: Advanced Techniques

### Multi-Adapter Serving

```python
from peft import PeftModel

# Load base model once
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Dictionary of adapters
adapters = {
    "legal": "./adapters/legal-lora",
    "medical": "./adapters/medical-lora",
    "customer_support": "./adapters/support-lora"
}

# Switch adapters dynamically
def generate_with_adapter(prompt, adapter_name):
    # Load adapter
    model = PeftModel.from_pretrained(base_model, adapters[adapter_name])
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=128)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Use different adapters
legal_response = generate_with_adapter("What is contract law?", "legal")
medical_response = generate_with_adapter("What causes diabetes?", "medical")

```

### Merging LoRA Weights

```python
from peft import PeftModel

# Load base + adapter
model = PeftModel.from_pretrained(base_model, "./lora-adapter")

# Merge adapter into base weights
merged_model = model.merge_and_unload()

# Save merged model (can use without PEFT)
merged_model.save_pretrained("./merged-model")
tokenizer.save_pretrained("./merged-model")

# Now can load as standard model
merged = AutoModelForCausalLM.from_pretrained("./merged-model")

```

### Gradient Checkpointing

Save memory by recomputing activations:

```python
model.gradient_checkpointing_enable()

# Trade: 20% slower training for 40% less memory

```

---

## Part 7: Best Practices

### Choosing LoRA Rank

```
Task Complexity → Rank

Simple (sentiment analysis): r=4-8
Medium (FAQ, summarization): r=16-32
Complex (code generation): r=32-64
Very complex (multi-task): r=64-128

Rule: Start with r=16, increase if underfitting

```

### Learning Rate Guidelines

```python
# LoRA typically needs higher LR than full fine-tuning
lr_guidelines = {
    "full_finetuning": 1e-5,
    "lora": 2e-4,
    "qlora": 3e-4  # Can be even higher
}

# Scheduler
"cosine"  # Smooth decay (recommended)
"linear"  # Linear decay
"constant_with_warmup"  # Flat after warmup

```

### Data Requirements

```
Minimum samples per task:
- Simple classification: 100-500
- FAQ/QA: 500-2000
- General instruction: 2000-10000
- Domain adaptation: 10000+

Quality > Quantity

```

### Evaluation Metrics

```python
# For FAQ/QA
metrics = {
    "perplexity": lower is better,
    "exact_match": answer == ground_truth,
    "f1_score": token overlap,
    "bleu": n-gram overlap,
    "human_eval": quality rating
}

```

---

## Part 8: Common Pitfalls

### Overfitting

**Symptoms:**

- Train loss ↓↓, Eval loss ↑
- Perfect training accuracy, poor test

**Solutions:**

```python
- Increase dropout: lora_dropout=0.1
- Add weight decay: weight_decay=0.01
- Reduce epochs
- Use more data
- Decrease rank

```

### Catastrophic Forgetting

**Problem:** Model forgets general knowledge after fine-tuning

**Solutions:**

```python
- Use lower learning rate
- Shorter training (3-5 epochs max)
- Mix general data with task data
- Use LoRA instead of full fine-tuning

```

### CUDA Out of Memory

**Solutions:**

```python
1. Reduce batch size
2. Enable gradient checkpointing
3. Use QLoRA instead of LoRA
4. Reduce sequence length
5. Use gradient accumulation

```

---

## Key Takeaways

**LoRA Principles:**

1. Low-rank decomposition: ΔW = BA
2. Freeze base model, train adapters
3. 100-1000× parameter reduction
4. Minimal quality loss (<1%)

**QLoRA Advantages:**

1. NF4 quantization (4-bit)
2. Double quantization
3. Paged optimizers
4. 7B models on 24GB GPUs

**Optuna Benefits:**

1. Bayesian optimization (TPE)
2. 50-100× fewer trials than grid search
3. Early pruning
4. Parallel execution

**FAQ Fine-Tuning:**

1. Instruction format critical
2. 500-2000 examples sufficient
3. r=16, lr=2e-4 typical
4. 3-5 epochs optimal

**Production Deployment:**

1. Multi-adapter serving
2. Sub-second adapter switching
3. Merge for deployment
4. Quantize for inference

---

## Practical Exercise

**Build an FAQ chatbot:**

1. Collect 1000 FAQ pairs from your domain
2. Format as instructions
3. Fine-tune LLaMA-2-7B with QLoRA
4. Optimize hyperparameters with Optuna
5. Deploy adapter
6. Evaluate quality

**Expected results:**

- Training: 4-6 hours on single A100
- Memory: <12GB
- Quality: 90%+ on domain FAQs
- Cost: <$50 total

---

## Further Reading

- **LoRA Paper:** "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- **QLoRA Paper:** "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- **Optuna:** [https://optuna.org](https://optuna.org)
- **HuggingFace PEFT:** [https://github.com/huggingface/peft](https://github.com/huggingface/peft)

---

**End of Lecture Notes**

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