# 36. Notes: Profiling & Debugging - Suman - 9 Jan 2026

# Profiling & Debugging Deep Learning Models: Lecture Notes

**Prerequisites:** PyTorch training loop implementation, understanding of backpropagation, basic GPU computing concepts.

**What you'll be able to do:**

- Profile GPU utilization and identify bottlenecks in training pipelines
- Visualize weight histograms to diagnose training health
- Detect and fix gradient anomalies including NaN and exploding gradients
- Apply systematic debugging strategies to training failures

---

## 1. Introduction: What is Model Profiling and Why Should You Care?

### Core Definition

Profiling and debugging in deep learning encompasses techniques to measure computational performance (GPU utilization, memory usage, operation timing) and diagnose training correctness issues (gradient flow, weight evolution, numerical stability). These skills transform model training from a black-box process into an observable, debuggable system.

### A Simple Analogy

Think of profiling like a car's diagnostic system. When your engine light comes on, the OBD-II port tells mechanics exactly which sensor failed. Similarly, profiling tools reveal whether your GPU is starved for data, your gradients are exploding, or your weights have collapsed to zero.

This analogy breaks down because neural networks have far more interacting components than a car engine, requiring multiple complementary profiling approaches.

### Why This Matters to You

**Problem it solves:** Training failures are often silent—loss might decrease slowly while the model secretly fails to learn. Without profiling, you waste days discovering that gradients vanished in layer 2, or that data loading consumed 80% of training time.

**What you'll gain:**

- Ability to diagnose why training is slow, unstable, or failing to converge
- Skills to optimize GPU utilization from typical 30% to 80%+
- Confidence to debug novel architectures and experimental code

**Real-world context:** Google's TensorBoard, PyTorch Profiler, and NVIDIA Nsight are used daily by ML teams to ensure their multi-million dollar training runs don't waste compute.

---

## 2. The Foundation: Core Concepts Explained

### Concept A: GPU Utilization Monitoring

**Definition:** GPU utilization measures what fraction of time the GPU compute units are actively processing operations, as opposed to waiting for data, memory transfers, or CPU coordination. High utilization (80%+) indicates efficient training; low utilization signals bottlenecks.

**Key characteristics:**

- **Compute utilization:** Percentage of time GPU cores are executing kernels
- **Memory utilization:** Fraction of GPU memory currently allocated
- **Memory bandwidth:** How fast data moves between GPU memory and compute units

**A concrete example:**

```python
import torch
import nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

# Check GPU utilization during training
info = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f"GPU Compute: {info.gpu}%")
print(f"GPU Memory IO: {info.memory}%")

```

**Common confusion:** High memory usage doesn't mean high compute utilization. A large batch can fill GPU memory while the GPU still waits for data loading, showing 100% memory but 30% compute.

---

### Concept B: Weight Histograms

**Definition:** Weight histograms visualize the distribution of parameter values in each layer at specific training checkpoints. They reveal patterns like vanishing gradients (weights stuck near zero), exploding weights (extreme values), or dead neurons (all zeros or constants).

**How it relates to GPU Utilization:** Weight histograms diagnose correctness issues (is the model learning right?) while GPU utilization diagnoses efficiency issues (is training fast?).

**Key characteristics:**

- **Shape:** Healthy weights typically form a bell curve centered near zero
- **Evolution:** Weights should gradually shift as training progresses
- **Layer comparison:** All layers should show similar healthy patterns; outliers indicate problems

**A concrete example:**

```python
import matplotlib.pyplot as plt
import torch

def plot_weight_histogram(model, layer_name):
    for name, param in model.named_parameters():
        if layer_name in name and 'weight' in name:
            plt.hist(param.detach().cpu().numpy().flatten(), bins=50)
            plt.title(f'{name} - Epoch X')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.show()

```

**Remember:** A histogram showing all weights clustered at exactly zero often indicates dead ReLU neurons—a specific and fixable problem.

---

### How GPU Utilization and Weight Histograms Work Together

Efficient GPU utilization ensures training runs fast, while healthy weight histograms ensure it runs correctly. A model might train efficiently (high GPU utilization) while learning nothing useful (collapsed weight distributions). Conversely, correct training with poor utilization wastes compute resources. Monitor both.

---

## 3. Seeing It in Action: Worked Examples

**Tip:** Study these examples carefully before attempting the practice task. Understanding *why* each step is taken is more important than memorizing the steps.

### Example 1: Diagnosing Low GPU Utilization

**Scenario:** Training is slow despite having a powerful GPU. GPU-Z shows only 25% compute utilization.

**Our approach:** Use PyTorch Profiler to identify where time is spent across data loading, forward pass, backward pass, and optimizer step.

**Step-by-step solution:**

```python
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

# Step 1: Set up profiler with activities to trace
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=tensorboard_trace_handler('./profiler_logs'),
    record_shapes=True,
    with_stack=True
) as prof:
    # Step 2: Run training steps inside profiler context
    for step, (images, labels) in enumerate(train_loader):
        if step >= 5:  # Profile only a few steps
            break

        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step 3: Tell profiler each step is complete
        prof.step()

# Step 4: Print summary sorted by CUDA time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

```

**Output:**

```
Name                    CPU Time   CUDA Time   Calls
----------------------  ---------  ----------  -----
DataLoader              450ms      0ms         5
aten::conv2d            20ms       180ms       15
aten::batch_norm        5ms        30ms        15
aten::linear            2ms        15ms        5
Optimizer.step          3ms        10ms        5

```

**What just happened:** The profiler revealed that DataLoader takes 450ms while actual GPU computation only takes ~235ms. The GPU is idle 65% of the time waiting for data!

**Check your understanding:** How would you fix the data loading bottleneck revealed by this profile?

---

### Example 2: Detecting Gradient Anomalies

**Scenario:** Training loss suddenly becomes NaN at epoch 23. We need to find which operation produced the first invalid gradient.

**What's different:** We enable autograd anomaly detection to get stack traces pointing to the problematic forward operation.

**Solution:**

```python
import torch

# Enable anomaly detection BEFORE creating tensors/model
torch.autograd.set_detect_anomaly(True)

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())

try:
    for epoch in range(100):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()  # Will raise error with stack trace if NaN
            optimizer.step()

except RuntimeError as e:
    print(f"Caught anomaly: {e}")
    # Stack trace will point to forward operation that caused NaN

finally:
    # Disable when done debugging (performance overhead)
    torch.autograd.set_detect_anomaly(False)

```

**Output:**

```
RuntimeError: Function 'LogBackward0' returned nan values in its 1st output.
...
  File "model.py", line 45, in forward
    x = torch.log(self.softmax(x))  # <- Problematic line!

```

**Key lesson:** The error points to `torch.log(self.softmax(x))`—when softmax produces values very close to zero, log produces -inf, which becomes NaN during backprop. Fix: use `log_softmax` instead.

---

### Example 3: Weight Histogram Analysis with TensorBoard

**Background:** You suspect your deep network has vanishing gradients because early layers seem to not learn while later layers improve.

**The challenge:** Visualize weight distributions across all layers throughout training to confirm the diagnosis.

**The approach:** Log weight histograms to TensorBoard at regular intervals.

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/gradient_analysis')

def log_weight_histograms(model, epoch, writer):
    """Log histograms of all layer weights."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            writer.add_histogram(f'weights/{name}', param, epoch)

        if param.grad is not None:
            writer.add_histogram(f'gradients/{name}', param.grad, epoch)

# In training loop:
for epoch in range(100):
    train_one_epoch(model, train_loader, optimizer)

    # Log histograms every 5 epochs
    if epoch % 5 == 0:
        log_weight_histograms(model, epoch, writer)

writer.close()

# Run: tensorboard --logdir=runs

```

**Why this approach:** TensorBoard shows histogram evolution over time. Healthy training shows weight distributions gradually spreading and shifting. Vanishing gradients show early-layer weights frozen while late-layer weights change. Exploding gradients show distributions stretching to extreme values.

**Caution:** Logging every epoch creates large TensorBoard files and slows training. Log every 5-10 epochs for long training runs.

---

## 4. Common Pitfalls: What Can Go Wrong and How to Avoid It

- **The Mistake:** Profiling with anomaly detection enabled in production

**Why It's a Problem:** Anomaly detection adds 2-3x overhead, making training much slower
**The Right Approach:** Enable only when actively debugging, disable for normal training
**Why This Works:** You get accurate timing from profiling and detailed errors when debugging, without mixing the overhead

---

- **The Mistake:** Assuming high GPU memory usage means efficient training

**Why It's a Problem:** Memory can be full while GPU compute units are idle waiting for data
**The Right Approach:** Monitor both memory AND compute utilization; aim for 80%+ compute

`# Use nvidia-smi or nvitop to monitor both
# $ watch -n 1 nvidia-smi`

**Why This Works:** Compute utilization directly measures productive work, not just memory occupation

---

- **The Mistake:** Ignoring weight histogram warnings until training fails

**Why It's a Problem:** By the time loss shows problems, the model may be irrecoverably corrupted
**The Right Approach:** Set up early warning systems that check for extreme values

`def check_weight_health(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            raise ValueError(f"NaN weights in {name}")
        if param.abs().max() > 1000:
            print(f"Warning: Large weights in {name}")`

**Why This Works:** Early detection prevents wasted compute on a corrupted training run

**If you're stuck:** Start with the simplest profiling (nvidia-smi for GPU usage) and add complexity only as needed.

---

## 5. Your Turn: Practice & Self-Assessment

### Practice Task

**The Challenge:** Build a comprehensive training monitor that tracks GPU utilization, weight statistics, and gradient health.

**Specifications:**

- Create a TrainingMonitor class with methods for each metric type
- Log GPU compute/memory utilization every N batches
- Track min/max/mean of weights and gradients per layer
- Implement automatic alerts for: NaN values, extreme gradients (>100), dead layers (all zeros)

**Hint:** Use nvidia-smi for GPU stats (via subprocess or pynvml), param.data for weights, param.grad for gradients. Structure as callback functions that integrate with any training loop.

**Extension (optional):** Add integration with Weights & Biases (wandb) for cloud-based monitoring and alerting.

---

### Check Your Understanding

1. 
**Explanation question:** Why does using more DataLoader workers not always improve GPU utilization?

2. 
**Application question:** You observe that gradient magnitudes decrease by 100x between the last layer and the first layer. What does this indicate, and what would you try first to fix it?

3. 
**Error analysis:**

```python
for epoch in range(100):
    for batch in loader:
        loss = model(batch).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

```

What profiling issue might this code cause, and how would you fix it?

1. **Transfer question:** How would you adapt weight histogram monitoring for a Transformer model with attention layers?

**Answers & Explanations:**

1. 
DataLoader workers prepare batches in parallel, but if CPU-GPU transfer or GPU computation is the bottleneck, more workers just create more batches waiting in memory. Profile first to identify the actual bottleneck.

2. 
This indicates vanishing gradients. Try first: replace sigmoid/tanh with ReLU, add batch normalization, use skip connections, or reduce network depth. These architectural changes improve gradient flow.

3. 
`optimizer.zero_grad()` is called AFTER `optimizer.step()`, meaning gradients accumulate across batches (wrong). Also, profiling would show accurate timing if this pattern continues. Fix: call zero_grad() before backward().

4. 
Attention weights have different healthy ranges than feedforward weights. Monitor attention patterns (should be sparse/focused), Q/K/V weights separately, and track gradient flow through attention vs. FFN layers independently.

---

### Self-Assessment Checklist

You've mastered this topic if you can:

- Set up PyTorch Profiler and interpret its output
- Identify data loading vs. computation bottlenecks from profiling
- Read weight histograms and diagnose vanishing/exploding gradient patterns
- Enable and use autograd anomaly detection effectively
- Build monitoring systems that alert on training problems
- Apply fixes for common issues: dead ReLU, gradient explosion, data bottlenecks

**If you checked fewer than 5 boxes:** Revisit Examples 1-3 and try running them with your own models.

---

## 6. Consolidation: Key Takeaways & Next Steps

### The Essential Ideas

- **GPU utilization:** Low compute utilization often indicates data loading bottlenecks, not model problems
- **Weight histograms:** Frozen or extreme distributions reveal gradient flow problems before loss metrics do
- **Gradient anomalies:** Enable autograd anomaly detection when debugging NaN/inf issues

### Mental Model Check

By now, you should think of profiling and debugging as: A systematic diagnostic process where you observe training internals (timing, distributions, gradients) to distinguish between efficiency problems (slow but correct) and correctness problems (fast but wrong).

### What You Can Now Do

You can diagnose why training is slow or failing, optimize data pipelines for GPU efficiency, and catch gradient problems early. These skills apply to any PyTorch project and scale to distributed training across many GPUs.

### Next Steps

**To deepen this knowledge:** Profile a real project you're working on—you'll almost certainly find optimization opportunities.

**To build on this:** Learn about mixed precision profiling, distributed training monitoring, and production ML observability tools.

**Additional resources:** PyTorch Profiler documentation, NVIDIA Nsight Systems for deep GPU analysis, Weights & Biases for cloud monitoring.

---

## Quick Reference Card

Symptom | Diagnostic Tool | Likely Cause
Slow training, low GPU % | PyTorch Profiler | Data loading bottleneck
Loss becomes NaN | Anomaly detection | Numerical instability (log, div)
Early layers don't learn | Weight histograms | Vanishing gradients
Weights become huge | Gradient monitoring | Exploding gradients, high LR
Model outputs all same | Weight histograms | Dead neurons, collapsed layers

---

**Questions or stuck?** The PyTorch forums have extensive discussions on profiling. Start with `torch.profiler` documentation for comprehensive examples.

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