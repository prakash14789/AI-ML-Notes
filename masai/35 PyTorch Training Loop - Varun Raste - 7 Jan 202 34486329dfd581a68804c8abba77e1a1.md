# 35. PyTorch Training Loop - Varun Raste - 7 Jan 2026

# PyTorch Training Loop: Comprehensive Lecture Notes

In Class Notes:
[Neural_Networks_with_Pytorch-main](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/1425a678-3c9c-40c1-bd33-0200a2b98027/nPnKCqGojRJDzyye.zip)

**Prerequisites:** Basic Python, understanding of neural network architecture, familiarity with gradient descent concept, NumPy experience.

**Time to complete:** 35-45 minutes

**What you'll be able to do:**

- Implement a complete training loop with proper gradient handling
- Choose between SGD and Adam optimizers based on your use case
- Configure learning rate schedulers for optimal convergence
- Save and load checkpoints for reproducible training

---

## 1. Introduction: What is the Training Loop and Why Should You Care?

### Core Definition

The training loop is the iterative process that teaches a neural network to make accurate predictions. It consists of four repeated steps: forward pass (make predictions), loss computation (measure errors), backward pass (calculate gradients), and weight update (improve the model). This loop runs for thousands to millions of iterations until the model converges.

### A Simple Analogy

Think of training like learning to shoot basketball free throws. Each attempt (forward pass) either scores or misses (loss). Your brain analyzes what went wrong—angle, force, release point (backward pass). Then you adjust your technique for the next shot (optimizer step). Thousands of shots later, you're accurate.

This analogy breaks down because neural networks don't have intuition—they can only follow gradient signals mathematically.

### Why This Matters to You

**Problem it solves:** Raw neural networks start with random weights and make random predictions. The training loop systematically improves these weights until the model is useful.

**What you'll gain:**

- Full control over how your model learns (not just calling `.fit()`)
- Ability to debug training issues like exploding gradients or plateaus
- Power to customize training with advanced techniques (gradient clipping, mixed precision)

**Real-world context:** This is exactly what happens when OpenAI trains GPT models—billions of training loop iterations on massive GPU clusters.

---

## 2. The Foundation: Core Concepts Explained

### Concept A: Tensor Operations

**Definition:** Tensors are PyTorch's fundamental data structure—multi-dimensional arrays that can execute on GPUs. Operations on tensors (addition, multiplication, matrix operations) form the computational backbone of neural networks.

**Key characteristics:**

- **GPU acceleration:** Tensors can move between CPU and GPU with `.cuda()` or `.to(device)`
- **Automatic differentiation:** Operations are tracked for gradient computation
- **Broadcasting:** Tensors of different shapes can operate together following NumPy rules

**A concrete example:**

```python
x = torch.randn(64, 784).cuda()  # 64 images, flattened
w = torch.randn(784, 256, requires_grad=True).cuda()
output = x @ w  # Matrix multiplication on GPU

```

**Common confusion:** Beginners often forget to move both model and data to the same device, causing device mismatch errors.

---

### Concept B: The Optimizer (SGD vs Adam)

**Definition:** Optimizers update model weights based on computed gradients. They determine the direction and magnitude of weight changes during training.

**How it relates to Tensors:** Optimizers receive gradient tensors from backpropagation and use them to modify weight tensors.

**Key characteristics:**

- **SGD (Stochastic Gradient Descent):** Simple—updates weights proportional to gradients
- **Adam:** Adaptive—maintains per-parameter learning rates based on gradient history
- **Momentum:** Both can use momentum to smooth updates and escape local minima

**A concrete example:**

```python
# SGD - simple, predictable
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam - adaptive, often converges faster
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001)

```

**Remember:** Adam often works well out-of-the-box, but SGD with proper tuning can achieve better final performance on some tasks.

---

### How Tensors and Optimizers Work Together

The forward pass creates a computational graph of tensor operations. Backpropagation traverses this graph to compute gradients stored in `.grad` attributes. The optimizer reads these gradients and updates the corresponding weight tensors.

---

## 3. Seeing It in Action: Worked Examples

**Tip:** Study these examples carefully before attempting the practice task. Understanding *why* each step is taken is more important than memorizing the steps.

### Example 1: Basic Training Loop Structure

**Scenario:** Train a simple MLP on synthetic data to understand the core loop mechanics.

**Our approach:** We'll implement the canonical forward-backward-update cycle with explicit steps.

**Step-by-step solution:**

```python
import torch
import torch.nn as nn

# Step 1: Setup - model, loss, optimizer
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 2: Training loop
for epoch in range(100):
    # Generate synthetic batch
    X = torch.randn(32, 10)
    y = torch.randn(32, 1)

    # Forward pass: compute predictions
    predictions = model(X)

    # Compute loss
    loss = criterion(predictions, y)

    # Backward pass: compute gradients
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Compute new gradients

    # Update weights
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

```

**Output:**

```
Epoch 0, Loss: 1.2847
Epoch 20, Loss: 0.9823
Epoch 40, Loss: 0.8156
Epoch 60, Loss: 0.7234
Epoch 80, Loss: 0.6891

```

**What just happened:** Each iteration, the model made predictions, measured error, computed how to improve, and adjusted weights. Over 100 iterations, loss decreased as the model learned patterns.

**Check your understanding:** Why do we call `zero_grad()` before `backward()`? What would happen if we didn't?

---

### Example 2: Adding Learning Rate Schedulers

**Scenario:** Training often benefits from reducing learning rate as we approach convergence. Let's add a scheduler.

**What's different:** We introduce `StepLR` which reduces learning rate by a factor every N epochs.

**Solution:**

```python
import torch.optim.lr_scheduler as lr_scheduler

# Setup with scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    # ... training code ...
    loss.backward()
    optimizer.step()

    # Step the scheduler after each epoch
    scheduler.step()

    if epoch % 30 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}, LR: {current_lr:.6f}")

```

**Output:**

```
Epoch 0, LR: 0.010000
Epoch 30, LR: 0.001000
Epoch 60, LR: 0.000100
Epoch 90, LR: 0.000010

```

**Key lesson:** Schedulers automatically adjust learning rate, allowing aggressive early training and fine-grained late training.

---

### Example 3: Implementing Checkpoints

**Background:** Training large models takes hours or days. Crashes, preemption, or wanting to resume later requires saving state.

**The challenge:** Save not just model weights, but optimizer state and epoch number for perfect resumption.

**The approach:** Use `torch.save()` with a dictionary containing all necessary state.

```python
# Saving a checkpoint
def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

# Loading a checkpoint
def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    return start_epoch

# Usage in training loop
for epoch in range(start_epoch, 100):
    # ... training code ...

    # Save every 10 epochs
    if epoch % 10 == 0:
        save_checkpoint(model, optimizer, epoch, loss, f'checkpoint_epoch_{epoch}.pt')

```

**Why this approach:** Saving optimizer state preserves momentum buffers (for SGD) or adaptive learning rates (for Adam). Without this, resuming training would behave differently than continuous training.

**Caution:** Always verify checkpoints can be loaded before running long training jobs. A corrupted checkpoint after 10 hours of training is painful.

---

## 4. Common Pitfalls: What Can Go Wrong and How to Avoid It

- **The Mistake:** Forgetting to call `optimizer.zero_grad()`

**Why It's a Problem:** Gradients accumulate across batches, causing incorrect updates and unstable training
**The Right Approach:** Always call `zero_grad()` before `backward()` in each iteration
**Why This Works:** Each batch should contribute only its own gradients to the update

---

- **The Mistake:** Mismatched devices (CPU vs GPU)

**Why It's a Problem:** Operations between tensors on different devices fail with cryptic errors
**The Right Approach:** Move model and all data to the same device consistently

`device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
inputs = inputs.to(device)`

**Why This Works:** All tensor operations require operands on the same device

---

- **The Mistake:** Not saving optimizer state in checkpoints

**Why It's a Problem:** Adam stores running averages of gradients; losing these means different training dynamics when resuming
**The Right Approach:** Always save both `model.state_dict()` and `optimizer.state_dict()`
**Why This Works:** Complete state restoration ensures training continues exactly as if never interrupted

**If you're stuck:** Revisit the Example 1 code structure—it shows the correct order of operations.

---

## 5. Your Turn: Practice & Self-Assessment

### Practice Task (Estimated: 20 minutes)

**The Challenge:** Implement a complete training pipeline for MNIST digit classification with all components covered.

**Specifications:**

- Build a 2-layer MLP (784 -> 256 -> 10)
- Use Adam optimizer with initial lr=0.001
- Add CosineAnnealingLR scheduler
- Save checkpoint after best validation loss
- Train for 10 epochs, print train/val loss each epoch

**Hint:** Structure your code in three parts: (1) setup model/optimizer/scheduler, (2) training loop with forward-backward-update, (3) validation loop without gradients using `torch.no_grad()`. For checkpointing, track `best_val_loss` and save only when current loss improves.

**Extension (optional):** Add gradient clipping with `torch.nn.utils.clip_grad_norm_()` to handle potential gradient explosions.

---

### Check Your Understanding

1. 
**Explanation question:** In your own words, explain why Adam often converges faster than vanilla SGD on the same problem.

2. 
**Application question:** Your training loss decreases smoothly but validation loss starts increasing after epoch 20. Would adjusting the learning rate scheduler help? What other approaches might work?

3. 
**Error analysis:**

```python
for batch in dataloader:
    predictions = model(batch)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()

```

What's wrong with this code, and how would you fix it?

1. **Transfer question:** How would you modify your training loop to implement gradient accumulation for effective batch size of 128 when your GPU only fits batch size 32?

**Answers & Explanations:**

1. 
Adam maintains adaptive per-parameter learning rates based on gradient history. Parameters with consistently small gradients get larger effective learning rates, while those with large gradients are dampened. This allows faster convergence without manual learning rate tuning per layer.

2. 
Scheduler adjustment alone won't fix overfitting—this is a generalization problem. Better approaches: add dropout, use weight decay, implement early stopping, or augment training data. The scheduler helps convergence, not generalization.

3. 
Missing `optimizer.zero_grad()` before `backward()`. Gradients accumulate, causing increasingly wrong updates. Fix: add `optimizer.zero_grad()` as the first line inside the loop.

4. 
Accumulate gradients over 4 batches before stepping: call `backward()` each iteration but only call `optimizer.step()` and `zero_grad()` every 4th iteration. Divide loss by 4 to normalize.

---

### Self-Assessment Checklist

You've mastered this topic if you can:

- Write a training loop from scratch without referring to examples
- Explain when to use SGD vs Adam and configure each appropriately
- Implement learning rate scheduling with at least two different schedulers
- Save and load checkpoints that perfectly resume training
- Debug common training loop bugs (device mismatch, gradient accumulation)
- Extend the basic loop with techniques like gradient clipping or accumulation

**If you checked fewer than 5 boxes:** Review the worked examples and trace through each line's purpose.

---

## 6. Consolidation: Key Takeaways & Next Steps

### The Essential Ideas

- **Core pattern:** Every training loop follows forward → loss → zero_grad → backward → step
- **Optimizer choice:** Adam for quick experiments and prototyping; SGD with momentum for best final performance when tuned
- **Checkpointing:** Save model AND optimizer state to ensure perfect training resumption

### Mental Model Check

By now, you should think of the training loop as: A systematic process that repeatedly measures prediction errors and nudges weights in the direction that reduces those errors, with optimizer and scheduler controlling how aggressively those nudges happen.

### What You Can Now Do

You can implement production-quality training pipelines with proper gradient handling, adaptive learning rates, and robust checkpointing. This foundation enables you to train any PyTorch model and debug issues when training doesn't go as expected.

### Next Steps

**To deepen this knowledge:** Implement training loops for different architectures (CNNs, Transformers) and observe how the same patterns apply.

**To build on this:** Explore distributed training with `DistributedDataParallel` and mixed precision with `torch.cuda.amp`.

**Additional resources:** PyTorch's official training loop tutorial and the fastai library's training abstractions for comparison.

---

## Quick Reference Card

Component | Purpose | Key Method
Forward pass | Compute predictions | outputs = model(inputs)
Loss | Measure prediction error | loss = criterion(outputs, targets)
Zero gradients | Clear old gradients | optimizer.zero_grad()
Backward pass | Compute gradients | loss.backward()
Update weights | Apply gradients | optimizer.step()
Schedule LR | Adjust learning rate | scheduler.step()
Save checkpoint | Persist training state | torch.save({...}, path)

---

**Questions or stuck?** The PyTorch forums and documentation are excellent resources. Start with the official tutorials at pytorch.org/tutorials.

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