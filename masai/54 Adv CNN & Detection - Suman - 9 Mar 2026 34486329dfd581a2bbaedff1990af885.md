# 54. Adv CNN & Detection - Suman - 9 Mar 2026

# Advanced CNN & Object Detection: Complete Lecture Notes

## PPT File: [Click Here](https://drive.google.com/file/d/12gq-rpELl5VWtYwJ-RA7wdMeWRu5SF-X/view?usp=sharing)

**Program:** Vishlesan i-Hub IIT Patna x Masai School — AIM (AI & Machine Learning)

**Topic:** Batch Normalization, ResNet, Transfer Learning, Object Detection, Segmentation

**Duration:** 3 hours (180 minutes)

**Prerequisites:** Basic CNNs, backpropagation, gradient descent

---

## Session Overview

**Learning Objectives:**
By the end of this session, you will:

1. Understand and implement batch normalization for training stability
2. Master residual networks (ResNet) and skip connections
3. Apply transfer learning for efficient model training
4. Build object detection systems (YOLO, SSD)
5. Implement semantic segmentation for pixel-level predictions

**Session Structure:**

- Part 1: Batch Normalization (30 mins)
- Part 2: ResNet & Skip Connections (35 mins)
- Part 3: Transfer Learning (30 mins)
- Part 4: Object Detection (40 mins)
- Part 5: Semantic Segmentation (30 mins)
- Part 6: Hands-On Implementation (15 mins)

---

# PART 1: BATCH NORMALIZATION (30 minutes)

## 1.1 The Problem: Internal Covariate Shift

### What is Internal Covariate Shift?

**Definition:** The distribution of layer inputs changes during training as parameters of previous layers update.

**The Problem Illustrated:**

```
Training Iteration 1:
Input to Layer 3: mean=0, std=1 (normalized)
Layer 3 learns with these statistics

Training Iteration 100:
Input to Layer 3: mean=15, std=8 (shifted!)
Layer 3 struggles - learned weights expect mean=0

Training Iteration 500:
Input to Layer 3: mean=-5, std=3 (shifted again!)
Layer 3 must constantly readjust

Result: SLOW CONVERGENCE, INSTABILITY

```

### Real-World Analogy

**Standardized Test Scenario:**

```
Imagine you're a teacher grading exams:

WITHOUT normalization:
- Week 1: Students score 40-60 (mean=50)
  You set grading: 50+ = Pass
  
- Week 2: Students score 70-90 (mean=80)
  Your grading rule (50+ = Pass) is now too easy!
  
- Week 3: Students score 10-30 (mean=20)
  Your grading rule (50+ = Pass) is now impossible!

You must constantly readjust grading criteria!

WITH normalization:
- Every week: Normalize scores to mean=50, std=10
- Your grading rule stays consistent
- Students know what to expect

Result: STABLE, FASTER LEARNING ✓

```

### Mathematical Problem

```
Deep network: 10 layers

Layer 1 output: mean=0, std=1
After activation: mean=0.5, std=0.8

Layer 2 output: mean=1.2, std=2.1
After activation: mean=0.8, std=1.5

Layer 3 output: mean=3.5, std=5.2
After activation: mean=2.1, std=3.8

...

Layer 10 output: mean=157, std=423 (EXPLODED!)
Gradients become unstable
Learning breaks down ✗

```

---

## 1.2 Batch Normalization: The Solution

### Core Idea

**Normalize layer inputs to have mean=0, std=1 for each mini-batch**

```
For each mini-batch:
1. Calculate batch mean μ and variance σ²
2. Normalize: x_norm = (x - μ) / √(σ² + ε)
3. Scale and shift: y = γ × x_norm + β

Where:
- μ, σ²: Batch statistics (calculated)
- γ, β: Learned parameters (trained)
- ε: Small constant (10⁻⁵) to prevent division by zero

```

### Step-by-Step Algorithm

```python
# Given: Mini-batch of activations X = [x₁, x₂, ..., xₘ]
# Shape: (batch_size=m, features=n)

# STEP 1: Calculate batch statistics
μ = (1/m) × Σ xᵢ                    # Mean
σ² = (1/m) × Σ (xᵢ - μ)²           # Variance

# STEP 2: Normalize
x_norm = (x - μ) / √(σ² + ε)       # Mean=0, Std=1

# STEP 3: Scale and shift (learnable!)
y = γ × x_norm + β                  # Restore representational power

# γ and β are LEARNED during training
# Network can recover original distribution if needed!

```

### Detailed Example

```
Mini-batch of 4 samples after Conv layer:
Feature 1 values: [100, 120, 80, 100]

STEP 1: Calculate statistics
μ = (100 + 120 + 80 + 100) / 4 = 100
σ² = [(0)² + (20)² + (-20)² + (0)²] / 4 = 200
σ = √200 ≈ 14.14

STEP 2: Normalize
x₁_norm = (100 - 100) / 14.14 = 0
x₂_norm = (120 - 100) / 14.14 = 1.41
x₃_norm = (80 - 100) / 14.14 = -1.41
x₄_norm = (100 - 100) / 14.14 = 0

Normalized: [0, 1.41, -1.41, 0]
Mean ≈ 0, Std ≈ 1 ✓

STEP 3: Scale and shift (assume γ=2, β=5)
y₁ = 2 × 0 + 5 = 5
y₂ = 2 × 1.41 + 5 = 7.82
y₃ = 2 × (-1.41) + 5 = 2.18
y₄ = 2 × 0 + 5 = 5

Final output: [5, 7.82, 2.18, 5]
Output mean = 5 (= β)
Output std = 2 (= γ)

Network LEARNED that β=5, γ=2 are optimal!

```

### Why γ and β are Learnable

```
Question: Why not just normalize to mean=0, std=1?

Answer: Sometimes the network NEEDS different statistics!

Example: Sigmoid activation
sigmoid(x) = 1 / (1 + e⁻ˣ)

If x always has mean=0, std=1:
- sigmoid(0) = 0.5 (stuck in middle)
- Limited expressiveness!

With learned γ and β:
- Network can shift mean to -5 (sigmoid ≈ 0) or +5 (sigmoid ≈ 1)
- Full range of sigmoid used
- More expressive! ✓

Batch Norm gives network the CHOICE:
- Start with normalized inputs (stable training)
- Learn to shift/scale as needed (expressiveness)

```

---

## 1.3 Training vs Inference

### Training Mode

```python
# During training: Use BATCH statistics

for batch in train_loader:
    # Calculate batch mean and variance
    μ_batch = batch.mean(dim=0)
    σ²_batch = batch.var(dim=0)
    
    # Normalize
    x_norm = (batch - μ_batch) / sqrt(σ²_batch + ε)
    
    # Scale and shift
    output = γ × x_norm + β
    
    # Also: Track running statistics for inference
    running_mean = 0.9 × running_mean + 0.1 × μ_batch
    running_var = 0.9 × running_var + 0.1 × σ²_batch

```

### Inference Mode

```python
# During inference: Use RUNNING statistics (from training)

# Problem: Single image or small batch
# Cannot calculate meaningful batch statistics!

# Solution: Use running averages from training
x_norm = (x - running_mean) / sqrt(running_var + ε)
output = γ × x_norm + β

# running_mean and running_var computed during training
# Represent overall dataset statistics

```

### Why Different Modes?

```
TRAINING:
Batch size = 64
Mean from 64 samples: STABLE, MEANINGFUL ✓

INFERENCE:
Single image (batch size = 1)
Mean from 1 sample = that sample's value!
Variance from 1 sample = 0!
Normalization breaks! ✗

Solution:
Use statistics accumulated during training
Represents "typical" data distribution

```

---

## 1.4 Benefits of Batch Normalization

### 1. Faster Training (Higher Learning Rates)

```
WITHOUT Batch Norm:
- Small LR required (0.001)
- Epochs to converge: 200
- Training time: 10 hours

WITH Batch Norm:
- Can use higher LR (0.01 - 10× larger!)
- Epochs to converge: 50
- Training time: 2 hours

Speedup: 5× faster! ✓

```

### 2. Reduced Sensitivity to Initialization

```
WITHOUT Batch Norm:
Xavier init: 92% accuracy
He init: 91% accuracy
Random init: 45% accuracy (fails!)

Initialization is CRITICAL!

WITH Batch Norm:
Xavier init: 92% accuracy
He init: 92% accuracy  
Random init: 90% accuracy (still works!)

Initialization matters LESS ✓

```

### 3. Regularization Effect (Slight)

```
Batch Norm adds noise:
- Each sample normalized using batch statistics
- Batch statistics vary batch-to-batch
- Acts as mild regularization

Empirical observation:
- Can reduce dropout slightly
- Model generalizes better
- But NOT a replacement for dropout/L2!

```

### 4. Enables Deeper Networks

```
WITHOUT Batch Norm:
10 layers: 88% accuracy
20 layers: 85% accuracy (degradation!)
50 layers: 72% accuracy (fails!)

WITH Batch Norm:
10 layers: 88% accuracy
20 layers: 91% accuracy ✓
50 layers: 93% accuracy ✓

Can train MUCH deeper networks!

```

---

## 1.5 Placement in Network

### Where to Place Batch Norm?

**Two common placements:**

```python
# OPTION 1: After activation (original paper)
x = Conv2d(x)
x = ReLU(x)
x = BatchNorm2d(x)

# OPTION 2: Before activation (more common now)
x = Conv2d(x)
x = BatchNorm2d(x)
x = ReLU(x)

# Modern consensus: Before activation works better

```

### Complete Conv Block

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)      # Convolution
        x = self.bn(x)        # Batch Normalization
        x = self.relu(x)      # Activation
        return x

```

### When NOT to Use Batch Norm

```
DON'T use Batch Norm when:

1. Very small batch sizes (< 4)
   → Batch statistics unstable
   → Use Layer Norm or Group Norm instead

2. Recurrent networks (RNNs/LSTMs)
   → Time dimension makes it tricky
   → Use Layer Norm instead

3. GANs (discriminator)
   → Can cause mode collapse
   → Use Spectral Normalization instead

4. Style transfer
   → Don't want to normalize style!
   → Use Instance Normalization

DO use Batch Norm for:
✓ Classification CNNs
✓ Object detection
✓ Semantic segmentation
✓ Most supervised learning tasks

```

---

## 1.6 PyTorch Implementation

### Basic Usage

```python
import torch
import torch.nn as nn

# 1D Batch Norm (for fully connected layers)
bn1d = nn.BatchNorm1d(num_features=128)

# 2D Batch Norm (for CNNs)
bn2d = nn.BatchNorm2d(num_features=64)

# Example: CNN with Batch Norm
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv + BN + ReLU blocks
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        return x

# Training mode
model.train()
output = model(images)  # Uses batch statistics

# Inference mode
model.eval()
output = model(images)  # Uses running statistics

```

### Complete Training Example

```python
import torch.optim as optim

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    model.train()  # Enable batch norm training mode
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()  # Enable batch norm inference mode
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            # Evaluate...

```

---

# PART 2: RESIDUAL NETWORKS (ResNet) (35 minutes)

## 2.1 The Degradation Problem

### Deeper is NOT Always Better

**The Surprising Discovery (2015):**

```
Experiment: Plain CNNs on CIFAR-10

20-layer network:
- Training accuracy: 87.8%
- Test accuracy: 85.3%

56-layer network (deeper):
- Training accuracy: 72.1% (WORSE!) ✗
- Test accuracy: 69.4% (WORSE!) ✗

Wait... training accuracy is WORSE?
This is NOT overfitting!
This is DEGRADATION!

```

### What is the Degradation Problem?

```
Degradation ≠ Overfitting

OVERFITTING:
Train accuracy: 99% (high)
Test accuracy: 75% (low)
→ Model memorized training data

DEGRADATION:
Train accuracy: 72% (low)
Test accuracy: 69% (low)
→ Model cannot even fit training data!

Problem: Deep networks are HARDER TO OPTIMIZE
Not an overfitting problem - an optimization problem!

```

### Why Do Deep Networks Degrade?

**Hypothesis: Vanishing/Exploding Gradients**

```
56-layer network during backpropagation:

Layer 56: ∂L/∂w₅₆ = 0.5
Layer 50: ∂L/∂w₅₀ = 0.5 × 0.8⁶ = 0.13
Layer 40: ∂L/∂w₄₀ = 0.5 × 0.8¹⁶ = 0.014
Layer 30: ∂L/∂w₃₀ = 0.5 × 0.8²⁶ = 0.0007
Layer 20: ∂L/∂w₂₀ = 0.5 × 0.8³⁶ = 0.00003
Layer 10: ∂L/∂w₁₀ = 0.5 × 0.8⁴⁶ = 0.000001 (vanished!)
Layer 1:  ∂L/∂w₁ ≈ 0 (no learning!)

Early layers cannot learn!
Network stuck in poor local minimum!

```

### The Identity Mapping Insight

**Key Observation:**

```
Shallow network (20 layers): 87.8% accuracy

Deep network (56 layers): Should be AT LEAST 87.8%!

Why?
The 56-layer network could learn:
- Layers 1-20: Same as shallow network
- Layers 21-56: IDENTITY MAPPING (do nothing)

Result: Same performance as 20-layer network!

But in practice: 72.1% accuracy (worse!)

Conclusion: Deep networks struggle to learn even IDENTITY!

```

---

## 2.2 Residual Learning Solution

### The Skip Connection

**Instead of learning H(x), learn the residual F(x) = H(x) - x**

```
TRADITIONAL BLOCK:
Input x → [Conv-BN-ReLU-Conv-BN] → Output H(x)

Goal: Learn H(x) directly
Problem: Hard for deep networks!

RESIDUAL BLOCK:
Input x → [Conv-BN-ReLU-Conv-BN] → F(x)
      ↓                                ↓
      └────────────────────────────────⊕ → Output H(x) = F(x) + x
           Skip Connection

Goal: Learn F(x) = H(x) - x (the residual)
Benefit: Much easier to learn!

```

### Why Residual Learning is Easier

```
Scenario: Identity mapping is optimal

TRADITIONAL:
Learn: H(x) = x
Must learn: Weight matrix = Identity matrix [[1,0],[0,1]]
Difficult! Requires precise weight values

RESIDUAL:
Learn: F(x) = H(x) - x = x - x = 0
Must learn: Weight matrix ≈ 0 (all zeros)
Easy! Weights naturally decay toward zero

Residual learning has better optimization landscape! ✓

```

### Mathematical Formulation

```
Traditional block:
y = H(x)

Residual block:
y = F(x) + x

Where:
- x: Input
- F(x): Learned residual function (Conv-BN-ReLU-Conv-BN)
- y: Output
- +: Element-wise addition

If identity is optimal:
F(x) = 0 (easy to learn)
y = 0 + x = x (identity achieved!)

```

---

## 2.3 ResNet Architecture

### Basic Building Block

```python
class BasicBlock(nn.Module):
    """ResNet Basic Block (used in ResNet-18, ResNet-34)"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Main path (learns residual F(x))
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (identity)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Projection shortcut (match dimensions)
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        identity = x  # Save input for skip connection
        
        # Main path (residual F(x))
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        out += self.skip(identity)  # F(x) + x
        out = self.relu(out)
        
        return out

```

### Bottleneck Block

```python
class BottleneckBlock(nn.Module):
    """ResNet Bottleneck Block (used in ResNet-50, 101, 152)"""
    
    expansion = 4  # Output channels = in_channels × 4
    
    def __init__(self, in_channels, bottleneck_channels, stride=1):
        super().__init__()
        
        out_channels = bottleneck_channels * self.expansion
        
        # 1×1 conv: Reduce dimensions
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 
                               kernel_size=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        
        # 3×3 conv: Main computation
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        
        # 1×1 conv: Restore dimensions
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels,
                               kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        identity = x
        
        # Bottleneck path
        out = self.relu(self.bn1(self.conv1(x)))      # 1×1 reduce
        out = self.relu(self.bn2(self.conv2(out)))     # 3×3 conv
        out = self.bn3(self.conv3(out))                # 1×1 expand
        
        # Skip connection
        out += self.skip(identity)
        out = self.relu(out)
        
        return out

```

### Complete ResNet-18 Architecture

```python
class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # Initial conv layer (stem)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        # First block (may downsample)
        layers.append(BasicBlock(in_channels, out_channels, stride))
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Input: (N, 3, 224, 224)
        
        # Stem
        x = self.conv1(x)       # (N, 64, 112, 112)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # (N, 64, 56, 56)
        
        # Residual layers
        x = self.layer1(x)      # (N, 64, 56, 56)
        x = self.layer2(x)      # (N, 128, 28, 28)
        x = self.layer3(x)      # (N, 256, 14, 14)
        x = self.layer4(x)      # (N, 512, 7, 7)
        
        # Classification
        x = self.avgpool(x)     # (N, 512, 1, 1)
        x = torch.flatten(x, 1) # (N, 512)
        x = self.fc(x)          # (N, num_classes)
        
        return x

```

### ResNet Family

```
ResNet-18:
- Basic blocks: [2, 2, 2, 2]
- Parameters: 11.7M
- ImageNet Top-1: 69.8%

ResNet-34:
- Basic blocks: [3, 4, 6, 3]
- Parameters: 21.8M
- ImageNet Top-1: 73.3%

ResNet-50:
- Bottleneck blocks: [3, 4, 6, 3]
- Parameters: 25.6M
- ImageNet Top-1: 76.1%

ResNet-101:
- Bottleneck blocks: [3, 4, 23, 3]
- Parameters: 44.5M
- ImageNet Top-1: 77.4%

ResNet-152:
- Bottleneck blocks: [3, 8, 36, 3]
- Parameters: 60.2M
- ImageNet Top-1: 78.3%

All trained on ImageNet (1.28M images, 1000 classes)

```

---

## 2.4 Why ResNets Work: Gradient Flow

### Gradient Highway

```
BACKPROPAGATION IN RESNET:

Forward pass:
y = F(x) + x

Backward pass (chain rule):
∂L/∂x = ∂L/∂y × ∂y/∂x
      = ∂L/∂y × ∂(F(x) + x)/∂x
      = ∂L/∂y × (∂F(x)/∂x + ∂x/∂x)
      = ∂L/∂y × (∂F(x)/∂x + 1)
                              ↑
                    Always adds 1!

Key insight: Gradient flows through skip connection UNIMPEDED!

Even if ∂F(x)/∂x → 0 (vanishing):
∂L/∂x = ∂L/∂y × (0 + 1) = ∂L/∂y

Gradient preserved! ✓

```

### Numerical Example

```
ResNet with 50 layers:

Layer 50 → 49:
∂L/∂x₄₉ = ∂L/∂x₅₀ × (∂F₅₀/∂x₄₉ + 1)
        = 1.0 × (0.1 + 1)
        = 1.1

Layer 40 → 39:
∂L/∂x₃₉ = 1.1 × (0.1 + 1) = 1.21

Layer 30 → 29:
∂L/∂x₂₉ = 1.21 × (0.1 + 1) = 1.33

Layer 20 → 19:
∂L/∂x₁₉ = 1.33 × (0.1 + 1) = 1.46

Layer 10 → 9:
∂L/∂x₉ = 1.46 × (0.1 + 1) = 1.61

Layer 1:
∂L/∂x₁ = ... ≈ 2.0

Gradient stays STRONG throughout! ✓

Compare to plain network:
∂L/∂x₁ = 1.0 × 0.1⁵⁰ ≈ 10⁻⁵⁰ (vanished!)

```

### Ensemble Perspective

```
ResNet can be viewed as ENSEMBLE of many shallow paths!

Consider 3-block ResNet:
         Input
           │
       ┌───┼───┐
       │   │   │
     Block │ Block
       │   │   │
       └───┼───┘
       ┌───┼───┐
       │   │   │
     Block │ Block
       │   │   │
       └───┼───┘
           │
        Output

Total paths from input to output: 2³ = 8 paths!

Paths of different lengths:
Length 0: 1 path (all skip)
Length 1: 3 paths (one block)
Length 2: 3 paths (two blocks)
Length 3: 1 path (all blocks)

Average path length: Much shorter than 3!
Effective ensemble of shallow networks!

```

---

# PART 3: TRANSFER LEARNING (30 minutes)

## 3.1 The Core Concept

### What is Transfer Learning?

**Use knowledge learned on one task to improve learning on a different (but related) task**

```
ANALOGY: Learning to play piano

Scenario A (From Scratch):
- Never touched instrument before
- Learn music theory: notes, rhythm, scales
- Learn piano technique: finger position, posture
- Learn to read sheet music
- Practice basic pieces
- Practice advanced pieces
Time: 5-10 years

Scenario B (Transfer Learning):
- Already play guitar (similar task!)
- Already know: music theory ✓, rhythm ✓, reading music ✓
- Only learn: piano-specific technique
- Practice advanced pieces
Time: 6-12 months

Transfer learning: 10× faster! ✓

```

### Machine Learning Analogy

```
WITHOUT Transfer Learning (Train from scratch):
Task: Classify dog breeds (120 classes)
Dataset: 10,000 dog images

Start: Random weights
Learn:
- Low-level features: edges, corners, textures
- Mid-level features: eyes, ears, fur patterns
- High-level features: dog body parts
- Breed-specific features: unique characteristics

Training: 100 epochs × 30 min = 50 hours
Accuracy: 65% (limited data!)

WITH Transfer Learning (Pre-trained on ImageNet):
Task: Classify dog breeds (120 classes)
Dataset: 10,000 dog images

Start: Pre-trained ResNet-50 (ImageNet)
Already knows:
- Low-level features: edges, corners, textures ✓
- Mid-level features: eyes, ears, fur patterns ✓
- High-level features: animal body parts ✓

Only learn:
- Breed-specific features

Training: 10 epochs × 5 min = 50 minutes
Accuracy: 92% (much better!)

Speedup: 60× faster!
Accuracy boost: +27%!

```

---

## 3.2 Why Transfer Learning Works

### Feature Hierarchy in CNNs

```
LAYER 1 (Early layers):
Learns: Edges, colors, simple textures
Examples:
- Horizontal edges ─
- Vertical edges │
- Diagonal edges ╱╲
- Color blobs 🔴🟢🔵

UNIVERSAL across all images! ✓
Dogs, cats, cars, buildings all have edges!

LAYER 2-3 (Middle layers):
Learns: Shapes, patterns, textures
Examples:
- Circles ⭕
- Squares ⬜
- Triangles 🔺
- Fur texture
- Metal texture

SOMEWHAT GENERAL ✓
Many objects share these patterns

LAYER 4-5 (Deep layers):
Learns: Object parts
Examples:
- Eyes 👁
- Wheels 🛞
- Windows 🪟
- Ears 👂

TASK-SPECIFIC
Dogs have eyes and ears
Cars have wheels

FINAL LAYER:
Learns: Class predictions
Examples:
- Dog breed: [Golden Retriever, Poodle, ...]
- Car model: [Tesla, BMW, ...]

COMPLETELY TASK-SPECIFIC ✗
Must retrain for new task!

```

### The Transfer Learning Hypothesis

**Features learned on large, diverse datasets (like ImageNet) are broadly useful**

```
ImageNet:
- 1.28 million images
- 1000 classes
- Diverse: animals, vehicles, objects, scenes

Network trained on ImageNet learns:
✓ General edge detectors
✓ Common texture patterns
✓ Object part detectors
✓ Compositional features

These features transfer to NEW tasks:
- Medical imaging (X-rays, CT scans)
- Satellite imagery
- Product classification
- Face recognition
- etc.

```

---

## 3.3 Transfer Learning Strategies

### Strategy 1: Feature Extraction (Freeze Pre-trained Layers)

```python
# Load pre-trained ResNet-50
model = torchvision.models.resnet50(pretrained=True)

# FREEZE all layers (don't update weights)
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for your task
num_classes = 10  # Your dataset classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Only train the final layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Training is FAST (only updating final layer)
# Works well when: New dataset is SMALL and SIMILAR to ImageNet

```

**When to use:**

- Small dataset (< 10,000 images)
- Similar to ImageNet (natural images)
- Limited compute resources

**Pros:**

- Very fast training
- Low risk of overfitting
- Minimal compute required

**Cons:**

- May not achieve best accuracy
- Pre-trained features might not be optimal

---

### Strategy 2: Fine-Tuning (Unfreeze Some Layers)

```python
# Load pre-trained ResNet-50
model = torchvision.models.resnet50(pretrained=True)

# FREEZE early layers (layer1, layer2)
for name, param in model.named_parameters():
    if 'layer1' in name or 'layer2' in name:
        param.requires_grad = False

# UNFREEZE later layers (layer3, layer4, fc)
# These will be fine-tuned

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Different learning rates for different layers
optimizer = optim.Adam([
    {'params': model.layer3.parameters(), 'lr': 1e-5},  # Low LR
    {'params': model.layer4.parameters(), 'lr': 1e-4},  # Medium LR
    {'params': model.fc.parameters(), 'lr': 1e-3}        # High LR
])

# Fine-tune frozen layers gradually
for epoch in range(epochs):
    train(...)
    
    # Unfreeze layer2 after epoch 10
    if epoch == 10:
        for param in model.layer2.parameters():
            param.requires_grad = True

```

**When to use:**

- Medium dataset (10,000 - 100,000 images)
- Somewhat different from ImageNet
- Sufficient compute available

**Layer freezing strategy:**

```
EARLY LAYERS (layer1, layer2):
- Learn general features (edges, textures)
- Usually FREEZE (same for all tasks)

MIDDLE LAYERS (layer3):
- Learn mid-level features
- FINE-TUNE with low learning rate (1e-5)

LATE LAYERS (layer4):
- Learn high-level features
- FINE-TUNE with medium learning rate (1e-4)

FINAL LAYER (fc):
- Task-specific classification
- TRAIN from scratch with high learning rate (1e-3)

```

---

### Strategy 3: Full Fine-Tuning

```python
# Load pre-trained model
model = torchvision.models.resnet50(pretrained=True)

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Unfreeze ALL layers
for param in model.parameters():
    param.requires_grad = True

# Use SMALL learning rate (pre-trained weights are already good!)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Optional: Different LR for different layers
optimizer = optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-6},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])

```

**When to use:**

- Large dataset (> 100,000 images)
- Very different from ImageNet (e.g., medical images)
- Plenty of compute resources

---

### Strategy Selection Guide

```
┌─────────────────────────────────────────────────────┐
│          TRANSFER LEARNING DECISION TREE            │
└─────────────────────────────────────────────────────┘

                    Dataset Size?
                         │
         ┌───────────────┼───────────────┐
         │               │               │
       Small          Medium           Large
     (< 10K)       (10K-100K)        (> 100K)
         │               │               │
         │               │               │
    Similar to       Similar?        Similar?
    ImageNet?            │               │
         │         ┌─────┴─────┐   ┌─────┴─────┐
    ┌────┴────┐   │           │   │           │
   Yes       No  Yes         No  Yes          No
    │         │   │           │   │            │
    │         │   │           │   │            │
Feature   Fine- Fine-    Fine-  Full      Full Fine-
Extract   Tune  Tune     Tune   Fine-     Tune + More
(Freeze) Early  Early+   All    Tune      Data
         Layers Middle   Layers (Small     Augmentation
                Layers           LR)

RECOMMENDATIONS:

Small + Similar:
→ Feature Extraction (freeze all, train final layer)
→ Fast, prevents overfitting

Small + Different:
→ Fine-Tune early layers (low LR)
→ Balance between adaptation and stability

Medium + Similar:
→ Fine-Tune middle layers
→ Good accuracy-speed tradeoff

Medium + Different:
→ Fine-Tune all layers (low LR)
→ Full adaptation while leveraging pre-training

Large + Similar:
→ Full Fine-Tune (small LR)
→ Optimal performance

Large + Different:
→ Full Fine-Tune + More epochs
→ May even train from scratch if very different

```

---

## 3.4 Practical Example: Medical Image Classification

### Scenario

```
Task: Classify chest X-rays (3 classes: Normal, Pneumonia, COVID-19)
Dataset: 5,000 X-ray images (small!)
Challenge: Medical images DIFFERENT from ImageNet (natural photos)

```

### Approach

```python
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

# Step 1: Load pre-trained ResNet-50
model = models.resnet50(pretrained=True)

# Step 2: Modify first conv layer (X-rays are grayscale, not RGB)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)

# Step 3: Freeze early layers (general features still useful)
for name, param in model.named_parameters():
    if 'layer1' in name or 'layer2' in name:
        param.requires_grad = False

# Step 4: Replace final layer (3 classes instead of 1000)
model.fc = nn.Linear(model.fc.in_features, 3)

# Step 5: Different learning rates for different parts
optimizer = optim.Adam([
    # Fine-tune layer3, layer4 with LOW learning rate
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    # Train final layer with HIGHER learning rate
    {'params': model.fc.parameters(), 'lr': 1e-3}
])

# Step 6: Data augmentation (compensate for small dataset)
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale
])

# Step 7: Training loop
for epoch in range(50):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Gradually unfreeze layers
    if epoch == 20:
        for param in model.layer2.parameters():
            param.requires_grad = True

```

### Results

```
WITHOUT Transfer Learning (train from scratch):
- Training time: 20 hours
- Final accuracy: 67%
- Overfitting: High (small dataset)

WITH Transfer Learning:
- Training time: 2 hours
- Final accuracy: 89%
- Overfitting: Minimal (pre-trained features regularize)

Improvement:
- 10× faster training
- +22% accuracy boost
- Better generalization

```

---

## 3.5 Common Pitfalls and Solutions

### Pitfall 1: Using Too High Learning Rate

```
Problem:
Pre-trained weights are already GOOD!
High LR destroys learned features!

# BAD
optimizer = optim.SGD(model.parameters(), lr=0.1)
# Learned features destroyed in first epoch!

# GOOD
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# Gentle fine-tuning preserves features

```

### Pitfall 2: Not Freezing Batch Norm Layers

```python
# During fine-tuning, freeze batch norm statistics!

model.train()  # Sets model to training mode

# But freeze batch norm layers
for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.eval()  # Keep batch norm in eval mode
        # Uses pre-trained statistics instead of batch statistics

```

### Pitfall 3: Forgetting to Match Input Size

```python
# ImageNet models expect 224×224 input

# BAD: Your images are 128×128
# Model receives wrong size, features misaligned

# GOOD: Resize to 224×224
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

```

---

# PART 4: OBJECT DETECTION (40 minutes)

## 4.1 From Classification to Detection

### The Evolution

```
IMAGE CLASSIFICATION:
Input: Image
Output: Single label (cat, dog, car)
Question: "WHAT is in the image?"

OBJECT LOCALIZATION:
Input: Image
Output: Label + Bounding box (x, y, width, height)
Question: "WHAT and WHERE?"
Constraint: Single object

OBJECT DETECTION:
Input: Image
Output: Multiple labels + Multiple bounding boxes
Question: "WHAT, WHERE, and HOW MANY?"
Challenge: Variable number of objects!

INSTANCE SEGMENTATION:
Input: Image
Output: Multiple labels + Pixel-wise masks
Question: "WHAT, WHERE, and EXACT SHAPE?"
Most detailed!

```

### Detection Challenges

```
Challenge 1: Multiple Objects
- Image may contain 0, 1, 5, or 100 objects
- Output size varies!
- Cannot use fixed-size output layer

Challenge 2: Different Sizes
- Small objects (person far away): 10×15 pixels
- Large objects (person close): 200×300 pixels
- Must detect across scales!

Challenge 3: Different Aspect Ratios
- Person standing: tall and thin (1:3 ratio)
- Car: wide and short (3:1 ratio)
- Must handle various shapes!

Challenge 4: Real-Time Requirements
- Self-driving cars: Need 30+ FPS
- Accuracy AND speed both critical!

```

---

## 4.2 Two-Stage Detectors: R-CNN Family

### R-CNN (2014): Region-based CNN

```
PIPELINE:

Step 1: Selective Search (CPU)
→ Propose ~2000 region proposals
→ Time: 2-3 seconds per image

Step 2: For each proposal:
→ Warp to fixed size (227×227)
→ Feed through CNN (AlexNet)
→ Extract features
→ Time: 47 seconds per image (2000 forward passes!)

Step 3: SVM Classification
→ Classify each region
→ Time: 0.5 seconds

Total: ~50 seconds per image
Accuracy: Good ✓
Speed: Terrible ✗

```

### Fast R-CNN (2015)

```
IMPROVEMENT: Share computation!

Step 1: Single CNN forward pass
→ Process ENTIRE image once
→ Generate feature map

Step 2: ROI Pooling
→ For each proposal, extract features from feature map
→ No need for 2000 separate CNN passes!

Speed: 2 seconds per image (25× faster than R-CNN!)
Accuracy: Same ✓
Bottleneck: Selective Search still slow ✗

```

### Faster R-CNN (2016)

```
IMPROVEMENT: Learn region proposals!

Architecture:
Image → CNN backbone → Feature map
                         ↓
        ┌───────────────┴───────────────┐
        │                               │
   RPN (Region                    ROI Pooling +
   Proposal Network)              Classification
   Learns proposals!              Refines boxes
        │                               │
        └───────────────┬───────────────┘
                        ↓
                  Final Detections

Speed: 0.2 seconds per image (5 FPS)
Accuracy: Excellent ✓
Still not real-time for video ✗

```

---

## 4.3 One-Stage Detectors: YOLO & SSD

### YOLO: You Only Look Once

**Core Idea: Direct prediction in single pass**

```
GRID-BASED DETECTION:

Step 1: Divide image into S×S grid (e.g., 13×13)

┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
├─┼─┼─┼─┼─┼─🚗┼─┼─┼─┼─┼─┼─┤  ← Car center falls here
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
...

Step 2: Each grid cell predicts B bounding boxes
- Each box: (x, y, w, h, confidence)
- x, y: Box center (relative to cell)
- w, h: Box dimensions (relative to image)
- confidence: P(object) × IOU

Step 3: Each grid cell predicts C class probabilities
- One probability distribution over all classes

Step 4: Final predictions
- Total: S × S × B bounding boxes
- For S=13, B=2: 13×13×2 = 338 boxes!
- Non-Maximum Suppression removes duplicates

```

### YOLO Architecture

```python
class YOLOv3(nn.Module):
    """Simplified YOLOv3 architecture"""
    
    def __init__(self, num_classes=80):
        super().__init__()
        
        # Backbone: Darknet-53
        self.backbone = Darknet53()
        
        # Detection heads at 3 scales
        self.detect1 = self._make_detection_head(1024, num_classes)  # 13×13
        self.detect2 = self._make_detection_head(512, num_classes)   # 26×26
        self.detect3 = self._make_detection_head(256, num_classes)   # 52×52
        
        # 3 anchor boxes per scale (9 total)
        self.anchors = [
            # Small objects (52×52 grid)
            [(10,13), (16,30), (33,23)],
            # Medium objects (26×26 grid)
            [(30,61), (62,45), (59,119)],
            # Large objects (13×13 grid)
            [(116,90), (156,198), (373,326)]
        ]
    
    def _make_detection_head(self, in_channels, num_classes):
        # Each box predicts: 4 coords + 1 confidence + num_classes
        num_predictions = 3 * (5 + num_classes)  # 3 anchors per scale
        return nn.Conv2d(in_channels, num_predictions, kernel_size=1)
    
    def forward(self, x):
        # Backbone
        feat1, feat2, feat3 = self.backbone(x)
        
        # Detections at multiple scales
        det1 = self.detect1(feat1)  # (N, 255, 13, 13) for 80 classes
        det2 = self.detect2(feat2)  # (N, 255, 26, 26)
        det3 = self.detect3(feat3)  # (N, 255, 52, 52)
        
        return det1, det2, det3

```

### YOLO Prediction Format

```
Output tensor shape: (Batch, 255, 13, 13) for 80 classes

Breakdown of 255 channels:
3 anchors × (4 box coords + 1 confidence + 80 classes) = 255

For each grid cell (i, j):
For each anchor k (k=0,1,2):
  Predict:
  - x, y: Box center offset (0 to 1)
  - w, h: Box size multiplier (relative to anchor)
  - confidence: P(object) × IOU
  - class_probs: [P(class_0), P(class_1), ..., P(class_79)]

Total predictions per image:
13×13×3 + 26×26×3 + 52×52×3 = 507 + 2028 + 8112 = 10,647 boxes!

After NMS: ~10-50 final boxes per image

```

---

### SSD: Single Shot MultiBox Detector

**Key Difference from YOLO: Multi-scale feature maps**

```
SSD ARCHITECTURE:

Base Network (VGG-16 or ResNet):
Input (300×300) → Conv layers → Feature maps

Detection at multiple scales:
┌─────────────────────────────────────────┐
│ Conv4_3: 38×38 → Detect small objects   │
│ Conv7:   19×19 → Detect medium objects  │
│ Conv8_2: 10×10 → Detect large objects   │
│ Conv9_2: 5×5   → Detect very large      │
│ Conv10_2: 3×3  → Detect huge objects    │
│ Conv11_2: 1×1  → Detect entire image    │
└─────────────────────────────────────────┘

Total predictions:
38×38×4 + 19×19×6 + 10×10×6 + 5×5×6 + 3×3×4 + 1×1×4
= 5776 + 2166 + 600 + 150 + 36 + 4
= 8732 boxes per image!

Advantage: Better at detecting small objects ✓
Disadvantage: More anchors to process ✗

```

### YOLO vs SSD Comparison

```
┌──────────────┬────────────┬────────────┐
│   Feature    │    YOLO    │    SSD     │
├──────────────┼────────────┼────────────┤
│ Speed (FPS)  │  60-100    │  40-60     │
│ mAP (COCO)   │  ~55%      │  ~58%      │
│ Architecture │  Grid-based│ Multi-scale│
│ Small objects│  Weaker    │  Better    │
│ Speed/Acc    │  Speed ✓   │  Accuracy ✓│
└──────────────┴────────────┴────────────┘

YOLO:
✓ Faster inference (single grid)
✓ Simpler architecture
✗ Struggles with small objects
✗ Lower accuracy

SSD:
✓ Better accuracy (multi-scale)
✓ Good small object detection
✗ Slower than YOLO
✗ More complex architecture

Use YOLO when: Speed critical (real-time video)
Use SSD when: Accuracy critical (small objects matter)

```

---

## 4.4 Detection Metrics

### Intersection over Union (IoU)

```
IoU measures overlap between predicted and ground truth boxes

         Ground Truth Box
         ┌──────────────┐
         │              │
    ┌────┼────┐         │
    │    │    │         │
    │    │ ∩  │         │  ← Intersection
    │    └────┼─────────┘
    │         │
    └─────────┘
    Predicted Box

IoU = Area of Intersection / Area of Union

Calculation:
1. Intersection = Overlapping area
2. Union = Total area covered by both boxes
3. IoU = Intersection / Union (0 to 1)

Examples:
Perfect match: IoU = 1.0
50% overlap: IoU = 0.5
No overlap: IoU = 0.0

Threshold: IoU > 0.5 typically considered "correct"

```

### Precision and Recall

```
CONFUSION MATRIX FOR DETECTION:

True Positive (TP): Predicted box with IoU > 0.5
False Positive (FP): Predicted box with IoU < 0.5
False Negative (FN): Ground truth box not detected

Precision = TP / (TP + FP)
- Of all predictions, how many were correct?
- High precision = few false alarms

Recall = TP / (TP + FN)
- Of all ground truths, how many were detected?
- High recall = found most objects

Example:
Ground truth: 10 cars
Predictions: 8 cars detected correctly (IoU > 0.5)
             2 false detections (trash cans mistaken for cars)

TP = 8 (correct detections)
FP = 2 (wrong detections)
FN = 2 (missed cars)

Precision = 8 / (8 + 2) = 0.8 (80%)
Recall = 8 / (8 + 2) = 0.8 (80%)

```

### Mean Average Precision (mAP)

```
mAP: THE standard metric for object detection

Step 1: For each class, calculate Average Precision (AP)

For "car" class:
- Sort predictions by confidence score (highest first)
- Calculate precision and recall at each threshold
- Plot Precision-Recall curve
- AP = Area under PR curve

Step 2: Average across all classes

mAP = (AP_car + AP_person + AP_bicycle + ...) / num_classes

Common variants:
- mAP@0.5: IoU threshold = 0.5
- mAP@0.75: IoU threshold = 0.75 (stricter)
- mAP@0.5:0.95: Average over IoU 0.5 to 0.95 (COCO metric)

Example:
AP_car = 0.85
AP_person = 0.78
AP_bicycle = 0.72

mAP@0.5 = (0.85 + 0.78 + 0.72) / 3 = 0.783 (78.3%)

```

---

## 4.5 Non-Maximum Suppression (NMS)

### The Problem: Duplicate Detections

```
YOLO predicts 10,647 boxes!
Many boxes detect the SAME object!

Example: Single car detected by:
Box 1: Confidence 0.95, IoU 0.87
Box 2: Confidence 0.92, IoU 0.83
Box 3: Confidence 0.88, IoU 0.79
Box 4: Confidence 0.85, IoU 0.72

Problem: 4 boxes for 1 car!
Solution: Keep only the best box!

```

### NMS Algorithm

```python
def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    Args:
        boxes: List of bounding boxes [(x1,y1,x2,y2), ...]
        scores: Confidence scores for each box
        iou_threshold: IoU threshold for suppression
    
    Returns:
        keep: Indices of boxes to keep
    """
    # Step 1: Sort boxes by confidence score (descending)
    sorted_indices = sorted(range(len(scores)), 
                           key=lambda i: scores[i], reverse=True)
    
    keep = []
    
    while sorted_indices:
        # Step 2: Pick box with highest confidence
        current = sorted_indices[0]
        keep.append(current)
        sorted_indices.pop(0)
        
        # Step 3: Remove boxes with high IoU overlap
        to_remove = []
        for idx in sorted_indices:
            iou = calculate_iou(boxes[current], boxes[idx])
            if iou > iou_threshold:
                to_remove.append(idx)
        
        # Remove suppressed boxes
        for idx in to_remove:
            sorted_indices.remove(idx)
    
    return keep

# Example usage:
boxes = [
    [100, 100, 200, 200],  # Box 1: 0.95 confidence
    [105, 105, 205, 205],  # Box 2: 0.92 (overlaps with 1)
    [300, 300, 400, 400],  # Box 3: 0.88 (different object)
    [110, 110, 210, 210]   # Box 4: 0.85 (overlaps with 1)
]
scores = [0.95, 0.92, 0.88, 0.85]

keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
# keep = [0, 2]  → Keep box 1 and box 3 only

```

### NMS Visualization

```
BEFORE NMS:
   ┌─────┐
   │  1  │ (0.95)
   └─────┘
     ┌─────┐
     │  2  │ (0.92)  ← High overlap with 1
     └─────┘
       ┌─────┐
       │  4  │ (0.85)  ← High overlap with 1
       └─────┘

              ┌─────┐
              │  3  │ (0.88)  ← Different object
              └─────┘

NMS Process:
1. Pick box 1 (highest confidence: 0.95) → KEEP
2. Remove box 2 (IoU with 1 > 0.5) → REMOVE
3. Pick box 3 (next highest: 0.88, different location) → KEEP
4. Remove box 4 (IoU with 1 > 0.5) → REMOVE

AFTER NMS:
   ┌─────┐
   │  1  │ (0.95) ✓
   └─────┘

              ┌─────┐
              │  3  │ (0.88) ✓
              └─────┘

Result: 2 clean detections! ✓

```

---

# PART 5: SEMANTIC SEGMENTATION (30 minutes)

## 5.1 From Detection to Segmentation

### The Progression

```
CLASSIFICATION:
Input: Image
Output: Label
Granularity: Entire image
Example: "This is a cat"

OBJECT DETECTION:
Input: Image
Output: Boxes + Labels
Granularity: Object-level
Example: "Cat at (100, 150, 200, 300)"

SEMANTIC SEGMENTATION:
Input: Image
Output: Pixel-wise labels
Granularity: Pixel-level
Example: "Pixel (50, 75) = cat, Pixel (200, 100) = background"

INSTANCE SEGMENTATION:
Input: Image
Output: Pixel-wise labels + Instance IDs
Granularity: Pixel-level + instances
Example: "Pixel (50, 75) = cat #1, Pixel (250, 75) = cat #2"

```

### Semantic vs Instance Segmentation

```
SEMANTIC SEGMENTATION:
Classes: [road, car, person, sky, building]
Output: Each pixel assigned to ONE class

        Sky    Sky    Sky    Sky
      Building Building Building
        Road   Road   Road   Road
        Car    Car    Person  Person

Problem: Cannot distinguish MULTIPLE instances!
Both cars labeled as "car" (same color)

INSTANCE SEGMENTATION:
Output: Each pixel has class AND instance ID

        Sky    Sky    Sky    Sky
      Bldg_1  Bldg_1  Bldg_2  Bldg_2
        Road   Road   Road   Road
        Car_1  Car_1  Car_2  Person_1

Can distinguish: Car_1 ≠ Car_2 ✓

```

---

## 5.2 Fully Convolutional Networks (FCN)

### The Problem with Standard CNNs

```
CLASSIFICATION CNN:
Input: (3, 224, 224)
Conv layers → (512, 7, 7)
Flatten → (512×7×7 = 25088,)
FC → (4096,)
FC → (1000,)
Output: Single prediction vector

Problem: Spatial information LOST after flatten!
Cannot produce per-pixel predictions!

```

### FCN Solution: Replace FC with Conv

```
FULLY CONVOLUTIONAL NETWORK:
Input: (3, 224, 224)
Conv layers → (512, 7, 7)
Conv 1×1 (instead of FC) → (4096, 7, 7)  ← Spatial info preserved!
Conv 1×1 → (1000, 7, 7)
Conv 1×1 → (21, 7, 7)  ← 21 classes
Upsample → (21, 224, 224)  ← Match input size!
Output: Per-pixel class predictions ✓

Key insights:
1. No flatten operation
2. Fully convolutional (works on any input size!)
3. Upsampling restores spatial resolution

```

---

## 5.3 Encoder-Decoder Architecture

### U-Net: The Classic Architecture

```
U-NET STRUCTURE:

    Input (572×572×1)
         │
    ┌────▼────┐
    │ Conv    │ 64 channels
    │ Conv    │
    ├─────────┤
    │MaxPool  │ ↓ 2× downsampling
    └────┬────┘
    ┌────▼────┐
    │ Conv    │ 128 channels
    │ Conv    │
    ├─────────┤
    │MaxPool  │ ↓ 2× downsampling
    └────┬────┘
         │
    (Continue encoding...)
         │
    ┌────▼────┐
    │ Conv    │ 1024 channels (bottleneck)
    │ Conv    │
    └────┬────┘
         │
    (Begin decoding...)
         │
    ┌────▼────┐
    │ UpConv  │ ↑ 2× upsampling
    │ Concat  │ ← Skip connection from encoder!
    │ Conv    │ 512 channels
    │ Conv    │
    └────┬────┘
    ┌────▼────┐
    │ UpConv  │ ↑ 2× upsampling
    │ Concat  │ ← Skip connection
    │ Conv    │ 256 channels
    │ Conv    │
    └────┬────┘
         │
    (Continue decoding...)
         │
    ┌────▼────┐
    │ Conv 1×1│ Output layer
    └────┬────┘
         │
    Output (388×388×2) ← Per-pixel classification

```

### Why Skip Connections Matter

```
WITHOUT Skip Connections:
Encoder → Bottleneck → Decoder

Problem:
- Encoder downsamples: 572×572 → 28×28
- Bottleneck: 28×28 (low resolution!)
- Decoder upsamples: 28×28 → 388×388
- Upsampling from 28×28 loses fine details!

Result: Blurry boundaries ✗

WITH Skip Connections:
Encoder (high-res) → Concat → Decoder

Benefit:
- Skip connections provide high-resolution features
- Decoder combines:
  * What: Semantic info from bottleneck
  * Where: Spatial info from encoder
- Sharp boundaries! ✓

Example:
Encoder layer: 256×256×64 (detailed edges)
Decoder layer: 256×256×64 (semantic features)
Concatenate: 256×256×128 (best of both!)

```

---

## 5.4 Advanced Techniques

### Atrous/Dilated Convolution

```
STANDARD CONVOLUTION:
3×3 kernel, rate=1 (normal)

[1] [1] [1]
[1] [1] [1]
[1] [1] [1]

Receptive field: 3×3 = 9 pixels

DILATED CONVOLUTION:
3×3 kernel, rate=2 (dilated)

[1] . [1] . [1]
 .  .  .  .  .
[1] . [1] . [1]
 .  .  .  .  .
[1] . [1] . [1]

Receptive field: 5×5 = 25 pixels!

BENEFIT:
- Larger receptive field WITHOUT more parameters
- Captures more context for segmentation
- No resolution loss (no pooling needed)

Used in: DeepLab, PSPNet

```

### Atrous Spatial Pyramid Pooling (ASPP)

```
ASPP MODULE:

Input feature map (H×W×C)
    │
    ├──────────┬──────────┬──────────┬──────────┐
    │          │          │          │          │
1×1 Conv   3×3 Conv   3×3 Conv   3×3 Conv   Global
           rate=6     rate=12    rate=18    AvgPool
    │          │          │          │          │
    └──────────┴──────────┴──────────┴──────────┘
                          │
                    Concatenate
                          │
                      1×1 Conv
                          │
                    Output (H×W×C')

Captures context at multiple scales:
- 1×1: Pixel-level features
- Rate 6: Small context
- Rate 12: Medium context
- Rate 18: Large context
- Global pool: Entire image context

Result: Multi-scale context! ✓

```

---

## 5.5 Segmentation Metrics

### Pixel Accuracy

```
Simple metric: % of correctly classified pixels

Pixel Accuracy = Correct Pixels / Total Pixels

Example: 256×256 image
Total pixels: 65,536
Correct predictions: 60,000

Pixel Accuracy = 60,000 / 65,536 = 91.5%

PROBLEM: Biased toward majority class!

Image: 90% road, 10% cars
Predict everything as "road": 90% accuracy!
But completely missed cars! ✗

```

### Intersection over Union (IoU) per Class

```
For each class:

IoU_class = (True Positives) / (TP + FP + FN)

Example: "Car" class
True Positives: 1000 pixels (predicted car, actually car)
False Positives: 200 pixels (predicted car, actually road)
False Negatives: 300 pixels (predicted road, actually car)

IoU_car = 1000 / (1000 + 200 + 300) = 1000 / 1500 = 0.667

Mean IoU (mIoU): Average across all classes
mIoU = (IoU_road + IoU_car + IoU_person + ...) / num_classes

mIoU is the STANDARD metric for segmentation!

```

### Dice Coefficient

```
Dice = (2 × TP) / (2×TP + FP + FN)

Relation to IoU:
Dice = (2 × IoU) / (1 + IoU)

Used in: Medical image segmentation
Reason: More sensitive to small structures

Example: Tumor segmentation
TP = 50 pixels
FP = 10 pixels
FN = 10 pixels

IoU = 50 / (50 + 10 + 10) = 0.714
Dice = (2×50) / (100 + 10 + 10) = 0.833

Dice weights TP twice → better for small objects

```

---

## 5.6 PyTorch Implementation

### Simple U-Net

```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):
        super().__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)  # 1024 = 512+512 (concat)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, num_classes, 1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)        # (N, 64, H, W)
        enc2 = self.enc2(self.pool(enc1))  # (N, 128, H/2, W/2)
        enc3 = self.enc3(self.pool(enc2))  # (N, 256, H/4, W/4)
        enc4 = self.enc4(self.pool(enc3))  # (N, 512, H/8, W/8)
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))  # (N, 1024, H/16, W/16)
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)  # (N, 512, H/8, W/8)
        dec4 = torch.cat([dec4, enc4], dim=1)  # (N, 1024, H/8, W/8)
        dec4 = self.dec4(dec4)  # (N, 512, H/8, W/8)
        
        dec3 = self.upconv3(dec4)  # (N, 256, H/4, W/4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)  # (N, 256, H/4, W/4)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)  # (N, 128, H/2, W/2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)  # (N, 64, H, W)
        
        # Output
        out = self.out(dec1)  # (N, num_classes, H, W)
        
        return out

# Training
model = UNet(in_channels=3, num_classes=21)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for images, masks in train_loader:
    # images: (N, 3, H, W)
    # masks: (N, H, W) with class indices
    
    outputs = model(images)  # (N, 21, H, W)
    loss = criterion(outputs, masks)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Inference
model.eval()
with torch.no_grad():
    outputs = model(test_image)  # (1, 21, H, W)
    predictions = torch.argmax(outputs, dim=1)  # (1, H, W)

```

---

# PART 6: HANDS-ON IMPLEMENTATION (15 minutes)

## Complete Object Detection Example

```python
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load pre-trained Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load and preprocess image
image = Image.open('street.jpg')
image_tensor = F.to_tensor(image)

# Inference
with torch.no_grad():
    predictions = model([image_tensor])

# Extract predictions
boxes = predictions[0]['boxes'].cpu().numpy()
labels = predictions[0]['labels'].cpu().numpy()
scores = predictions[0]['scores'].cpu().numpy()

# Filter by confidence threshold
threshold = 0.5
keep = scores > threshold

boxes = boxes[keep]
labels = labels[keep]
scores = scores[keep]

# Visualize
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)

COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', ...]

for box, label, score in zip(boxes, labels, scores):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    
    # Draw bounding box
    rect = patches.Rectangle((x1, y1), width, height,
                             linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    
    # Add label
    class_name = COCO_CLASSES[label-1]
    ax.text(x1, y1-5, f'{class_name}: {score:.2f}',
           bbox=dict(facecolor='red', alpha=0.5),
           fontsize=10, color='white')

plt.axis('off')
plt.tight_layout()
plt.show()

```

---

## Summary: Complete Architecture Evolution

```
┌──────────────────────────────────────────────────────────┐
│         COMPUTER VISION EVOLUTION TIMELINE               │
└──────────────────────────────────────────────────────────┘

2012: AlexNet
- Deep CNN for classification
- 5 conv + 3 FC layers
- ImageNet winner (Top-5 error: 15.3%)

2014: VGG-16
- Deeper networks (16 layers)
- Small 3×3 filters throughout
- Top-5 error: 7.3%

2015: BATCH NORMALIZATION
- Internal covariate shift solution
- 10× faster training
- Enables very deep networks ✓

2015: ResNet
- Skip connections solve degradation
- 152 layers trainable!
- Top-5 error: 3.57% (superhuman!)

2014-2016: OBJECT DETECTION
- R-CNN → Fast R-CNN → Faster R-CNN
- YOLO: Real-time detection
- SSD: Multi-scale detection

2015-2017: SEMANTIC SEGMENTATION
- FCN: Fully convolutional
- U-Net: Encoder-decoder + skip connections
- DeepLab: Atrous convolution + ASPP

2017: INSTANCE SEGMENTATION
- Mask R-CNN: Faster R-CNN + segmentation
- Per-pixel instance masks

MODERN ERA (2020+):
- Vision Transformers
- DETR (detection without anchors)
- Segment Anything Model (SAM)

```

---

## Key Takeaways

1. 
**Batch Normalization:**

Normalizes layer inputs for stable training
Enables higher learning rates
Critical for deep networks

2. 
**ResNet:**

Skip connections prevent degradation
Enables 100+ layer networks
Gradient highway preserves information

3. 
**Transfer Learning:**

Leverage pre-trained models
10× faster training, +20% accuracy
Essential for limited data

4. 
**Object Detection:**

YOLO: Speed (60-100 FPS)
SSD: Accuracy (multi-scale)
Faster R-CNN: High accuracy

5. 
**Semantic Segmentation:**

U-Net: Medical imaging standard
DeepLab: State-of-the-art
Skip connections critical for details

---

**End of Lecture Notes**

**Next Steps:**

- Complete hands-on assignment
- Implement ResNet from scratch
- Train object detector on custom data
- Build segmentation model for medical images

**Resources:**

- Papers: ResNet, YOLO, U-Net
- Code: PyTorch model zoo
- Datasets: COCO, Pascal VOC, Cityscapes

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