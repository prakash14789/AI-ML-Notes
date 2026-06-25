# 52. Transfer Learning Primer - Suman - 5 Mar 2026

# Transfer Learning Primer: Lecture Notes

## In-Class Resources: [Click Here](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/a1d7a51c-9f1c-4cc6-93a3-b9f3c7c5595f/PXgqKgcGHefBAOiu.zip)

**Program:** Vishlesan i-Hub IIT Patna x Masai School — AIM (AI & Machine Learning)
**Session:** Transfer Learning Primer
**Prerequisites:** CNNs, Image Classification, Backpropagation, PyTorch Basics

---

## MENTAL MAP

```
Transfer Learning Primer Session

Part 1: CNN Fundamentals Review
├─ Convolution, Padding, Stride
├─ Activation Functions (ReLU)
└─ Pooling Operations

Part 2: Deep Network Challenges
├─ Vanishing Gradients Problem
├─ Exploding Gradients Problem
└─ ResNet Solution (Skip Connections)

Part 3: Transfer Learning Core
├─ What is Transfer Learning?
├─ Pre-trained Models (ImageNet)
├─ Freezing vs Fine-Tuning
└─ Knowledge Distillation

Part 4: Optimization Techniques
├─ Quantization (8-bit, 16-bit)
└─ Deployment Strategies

Part 5: Object Detection
├─ R-CNN (Accuracy-focused)
└─ YOLO (Speed-focused)

```

---

## THE BIG PICTURE

### The Problem This Session Solves

**Scenario:** A medical imaging startup needs to classify 5 types of skin lesions with:

- Only 5,000 labeled images (small dataset)
- 1 GPU machine
- 2-week deadline
- Minimum 90% accuracy requirement

**Approach 1: Training from Scratch (Fails)**

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    Flatten(),
    Dense(5, activation='softmax')
])
# Result: 65% accuracy after 2 weeks, project fails

```

**Approach 2: Transfer Learning (Succeeds)**

```python
base_model = ResNet50(weights='imagenet', include_top=False)
base_model.trainable = False  # Freeze pre-trained weights

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(5, activation='softmax')
])
# Result: 92% accuracy in 2 hours, project succeeds!

```

**Key Insight:** Transfer learning gives you 10× faster training and 27% better accuracy.

---

## PART 1: CNN FUNDAMENTALS REVIEW

### Topic 1.1: Convolution Operation

**What is Convolution?**

Convolution applies a filter (kernel) to an image to extract features.

**Example:**

```
Image (5×5):              Kernel (3×3):
[1 2 3 4 5]              [1 0 -1]
[6 7 8 9 0]              [1 0 -1]
[1 2 3 4 5]              [1 0 -1]
[6 7 8 9 0]
[1 2 3 4 5]

Convolution Output (3×3):
[value1 value2 value3]
[value4 value5 value6]
[value7 value8 value9]

```

**What Kernels Detect:**

- Horizontal edges
- Vertical edges
- Diagonal patterns
- Textures
- Colors

---

### Topic 1.2: Padding

**Problem:** Convolution shrinks the output size.

**Solution:** Add padding (usually zeros) around the image.

**Types:**

```python
# Valid padding (no padding)
Input: 5×5, Kernel: 3×3 → Output: 3×3

# Same padding (preserve size)
Input: 5×5 + padding, Kernel: 3×3 → Output: 5×5

```

**Why Padding Matters:**

- Preserves spatial dimensions
- Allows deeper networks without losing information
- Prevents edge pixels from being underutilized

---

### Topic 1.3: Stride

**Definition:** Number of pixels the kernel moves at each step.

**Examples:**

```
Stride = 1:
Kernel slides 1 pixel at a time (more overlap, larger output)

Stride = 2:
Kernel slides 2 pixels at a time (less overlap, smaller output)

```

**Effect on Output Size:**

```
Output_size = (Input_size - Kernel_size + 2×Padding) / Stride + 1

Example:
Input=28×28, Kernel=3×3, Padding=1, Stride=1
Output = (28 - 3 + 2×1) / 1 + 1 = 28×28

Input=28×28, Kernel=3×3, Padding=0, Stride=2
Output = (28 - 3 + 0) / 2 + 1 = 13×13

```

---

### Topic 1.4: ReLU Activation Function

**Formula:**

```
ReLU(x) = max(0, x)

```

**Why ReLU is Preferred:**

**Comparison with Sigmoid:**

```
Sigmoid(x) = 1 / (1 + e^(-x))
- Range: [0, 1]
- Derivative: sigmoid'(x) = sigmoid(x) × (1 - sigmoid(x))
- Max derivative: 0.25 (causes vanishing gradients)

ReLU(x) = max(0, x)
- Range: [0, ∞)
- Derivative: 1 if x > 0, else 0
- No saturation for positive values

```

**Example:**

```
Input:  [-2, -1, 0, 1, 2, 3]
ReLU:   [ 0,  0, 0, 1, 2, 3]

Keeps positive features, suppresses negative

```

---

### Topic 1.5: Pooling Operations

**Max Pooling:**

```
Input (4×4):
[1 3 | 2 4]
[5 6 | 7 8]
-----------
[2 1 | 3 4]
[9 2 | 1 5]

Max Pool (2×2, stride=2):
[6 | 8]
[9 | 5]

Takes maximum value in each window

```

**Average Pooling:**

```
Same input → Average Pool (2×2):
[3.75 | 5.25]
[3.5  | 3.25]

Takes average value in each window

```

**Why Pooling?**

- Reduces spatial dimensions
- Reduces computation
- Provides translation invariance
- Keeps important features

---

## PART 2: DEEP NETWORK CHALLENGES

### Topic 2.1: The Vanishing Gradient Problem

**What Happens:**

In deep networks, gradients become extremely small as they backpropagate through layers.

**Math Behind It:**

```
Chain Rule in Backpropagation:
∂Loss/∂W₁ = ∂Loss/∂a₅₀ × ∂a₅₀/∂a₄₉ × ... × ∂a₂/∂W₁

If each ∂aᵢ/∂aᵢ₋₁ ≈ 0.9:
Product = 0.9⁵⁰ ≈ 0.005 (vanishes!)

```

**Effect on Training:**

```python
# Early layer update
weight_update = learning_rate × gradient
              = 0.001 × 0.000001  # gradient vanished
              = 0.000000001       # barely changes!

# Early layers stop learning

```

**Visual Example:**

```
Loss over Epochs:

Loss
  │
  │  ● Layer 50 (learns fast)
  │    ●
  │      ● Layer 30 (learns slowly)
  │        ●●
  │          ●●●● Layers 1-10 (stuck)
  │_________________________→ Epochs

```

**Why Sigmoid Makes It Worse:**

```
Sigmoid derivative max = 0.25
50 layers: 0.25⁵⁰ ≈ 10⁻³⁰ (completely vanishes!)

```

---

### Topic 2.2: The Exploding Gradient Problem

**What Happens:**

Gradients grow exponentially, causing training instability.

**Math:**

```
If each ∂aᵢ/∂aᵢ₋₁ ≈ 1.1:
Product = 1.1⁵⁰ ≈ 117 (explodes!)

```

**Effect:**

```python
weight_update = 0.001 × 117 = 0.117 (huge update!)

Weights oscillate wildly:
Epoch 1: Weight = 0.5
Epoch 2: Weight = 5.2
Epoch 3: Weight = -8.9
Epoch 4: Weight = NaN (overflow!)

```

**Symptoms:**

- Loss becomes NaN
- Weights blow up to infinity
- Model predictions become all zeros or all ones

---

### Topic 2.3: ResNet Solution - Skip Connections

**Traditional Deep Network:**

```
x → [Layer1] → [Layer2] → ... → [Layer50] → H(x)

Must learn H(x) directly (difficult)

```

**ResNet Residual Block:**

```
        x
        │
    ┌───┴───┐
    │       │
    │   [Conv3×3]
    │       │
    │   [ReLU]
    │       │
    │   [Conv3×3]
    │       │
    └───┬───┘
        │
      [Add]  ← x + F(x)
        │
      [ReLU]
        │
     Output

Output = F(x) + x
Network learns F(x) = H(x) - x (residual)

```

**Why It Works:**

**1. Gradient Flow:**

```
Traditional backprop:
∂Loss/∂x = ∂Loss/∂H × ∂H/∂Layer50 × ... × ∂Layer2/∂x
           (many multiplications → vanishes)

ResNet backprop:
∂Loss/∂x = ∂Loss/∂(F(x) + x)
         = ∂Loss/∂F(x) + ∂Loss/∂x  (chain rule)
         ↑              ↑
       Learned      Identity (always 1!)

The "+1" prevents vanishing!

```

**2. Easier Optimization:**

```
If optimal mapping is identity:
Traditional: Must learn H(x) = x (50 layers doing nothing)
ResNet: Just set F(x) = 0, then H(x) = F(x) + x = 0 + x = x

```

**PyTorch Implementation:**

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        identity = x  # Save input
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity  # Add skip connection
        out = self.relu(out)
        
        return out

```

**ResNet Variants:**

- **ResNet18:** 18 layers, ~11M parameters
- **ResNet34:** 34 layers, ~21M parameters
- **ResNet50:** 50 layers, ~25M parameters
- **ResNet101:** 101 layers, ~44M parameters
- **ResNet152:** 152 layers, ~60M parameters

---

## PART 3: TRANSFER LEARNING CORE

### Topic 3.1: What is Transfer Learning?

**Definition:**

Transfer learning uses a model trained on one task (source) and adapts it to a different but related task (target).

**Analogy:**

```
Human Learning:
- Learn to drive car → Easier to drive truck
- Learn Spanish → Easier to learn Portuguese
- Learn violin → Easier to learn viola

Machine Learning:
- Trained on ImageNet → Adapt to medical images
- Trained on general images → Adapt to satellite imagery
- Trained on English → Adapt to French

```

**Why It Works:**

Early CNN layers learn **universal features**:

```
Layer 1: Edges, corners, basic shapes (universal)
Layer 2: Textures, patterns (mostly universal)
Layer 3: Object parts (semi-specific)
Layer 4: Complete objects (task-specific)

```

**ImageNet Pre-training:**

- 1.2 million images
- 1,000 categories
- Takes 2-4 weeks on 8 GPUs
- Costs $50,000+ in compute

**Your Benefit:**

- Download pre-trained weights in 5 seconds
- Free!
- Proven to work on millions of images

---

### Topic 3.2: Freezing Weights

**Concept:**

Keep pre-trained layer weights unchanged (no gradient updates).

**PyTorch Implementation:**

```python
from torchvision.models import resnet50

# Load pre-trained model
model = resnet50(weights='IMAGENET1K_V1')

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for your task (10 classes)
model.fc = nn.Linear(model.fc.in_features, 10)

# Only the final layer will train

```

**What Gets Frozen:**

```
ResNet50:
├─ conv1 (FROZEN) ← Detects edges
├─ layer1 (FROZEN) ← Detects textures
├─ layer2 (FROZEN) ← Detects patterns
├─ layer3 (FROZEN) ← Detects object parts
├─ layer4 (FROZEN) ← Complex features
└─ fc (TRAINABLE) ← Your classifier

Only fc trains on your data

```

**When to Freeze:**

- Small dataset (<10K images)
- Target task similar to ImageNet
- Limited compute budget
- Need fast training

**Training Time Comparison:**

```
From Scratch: 100 epochs × 2 hours = 200 hours
Frozen Base: 20 epochs × 5 minutes = 100 minutes

100× speedup!

```

---

### Topic 3.3: Fine-Tuning

**Concept:**

Unfreeze some later layers and train with low learning rate.

**Two-Stage Process:**

**Stage 1: Train Classifier**

```python
# Freeze base
for param in model.parameters():
    param.requires_grad = False

# Replace and train classifier
model.fc = nn.Linear(2048, 10)
optimizer = Adam(model.fc.parameters(), lr=1e-3)

# Train for 10 epochs

```

**Stage 2: Fine-Tune Last Block**

```python
# Unfreeze layer4 (last residual block)
for param in model.layer4.parameters():
    param.requires_grad = True

# Use MUCH lower learning rate
optimizer = Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])

# Train for 10 more epochs

```

**Learning Rate Strategy:**

```
Frozen base training: LR = 1e-3 (normal)
Fine-tuning: LR = 1e-5 to 1e-4 (10-100× lower!)

Why? Prevent catastrophic forgetting of pre-trained features

```

**When to Fine-Tune:**

- Medium-large dataset (>10K images)
- Target task different from ImageNet
- Need maximum accuracy
- Have compute budget

**Performance Comparison:**

```
Frozen only: 87% accuracy, 2 hours
+ Fine-tuning: 92% accuracy, 4 hours

Extra 5% accuracy for 2 more hours

```

---

### Topic 3.4: Knowledge Distillation

**Concept:**

Train a small "student" model to mimic a large "teacher" model.

**Process:**

```
Teacher Model (Large):
Input → ResNet152 → Soft Labels (probabilities)

Student Model (Small):
Input → ResNet18 → Try to match teacher's outputs

```

**Loss Function:**

```python
# Teacher predictions (soft targets)
teacher_logits = teacher_model(x)
teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

# Student predictions
student_logits = student_model(x)
student_probs = F.softmax(student_logits / temperature, dim=1)

# Distillation loss
distill_loss = KL_divergence(student_probs, teacher_probs)

# Final loss
total_loss = 0.7 × distill_loss + 0.3 × cross_entropy_loss

```

**Benefits:**

- Student model is 10× smaller
- 5× faster inference
- 95% of teacher's accuracy

**Use Cases:**

- Deploy on mobile devices
- Real-time applications
- Edge computing

---

## PART 4: OPTIMIZATION TECHNIQUES

### Topic 4.1: Quantization

**Concept:**

Reduce precision of weights and activations.

**Float Precision Levels:**

```
FP32 (32-bit float): 3.14159265 (normal)
FP16 (16-bit float): 3.141 (half precision)
INT8 (8-bit integer): 3 (quantized)

```

**Size Comparison:**

```
ResNet50 weights:
FP32: 25M params × 4 bytes = 100 MB
FP16: 25M params × 2 bytes = 50 MB (50% reduction)
INT8: 25M params × 1 byte = 25 MB (75% reduction)

```

**Performance:**

```
Model: ResNet50 on mobile CPU

FP32: 250 ms/image, 100 MB
FP16: 150 ms/image, 50 MB (1.6× faster)
INT8: 80 ms/image, 25 MB (3× faster)

Accuracy loss: 0.5-2%

```

**PyTorch Quantization:**

```python
import torch.quantization as quantization

# Post-training quantization
model_fp32 = resnet50(pretrained=True)
model_fp32.eval()

# Convert to INT8
model_int8 = quantization.quantize_dynamic(
    model_fp32,
    {nn.Linear, nn.Conv2d},
    dtype=torch.qint8
)

# Model is now 4× smaller and faster

```

**When to Quantize:**

- Deploying to mobile/edge devices
- Need real-time inference
- Limited memory/power budget
- Can tolerate 1-2% accuracy loss

---

## PART 5: OBJECT DETECTION

### Topic 5.1: R-CNN (Region-based CNN)

**How It Works:**

**Step 1: Generate Region Proposals**

```
Input Image → Selective Search → ~2000 candidate regions

Example regions:
[Box1: (x=10, y=20, w=50, h=60)]
[Box2: (x=100, y=150, w=80, h=90)]
...

```

**Step 2: Classify Each Region**

```
For each region:
1. Crop from image
2. Resize to 224×224
3. Pass through CNN
4. Classify object
5. Refine bounding box

```

**Architecture:**

```
Image → [Selective Search] → 2000 regions
            ↓
Each region → [CNN Feature Extraction]
            ↓
         [SVM Classifier] → Class + Box

```

**Advantages:**

- High accuracy
- Handles multiple objects
- Precise bounding boxes

**Disadvantages:**

- Very slow (2000 CNN passes per image!)
- Not real-time
- High computation

**Performance:**

```
Speed: 47 seconds per image (GPU)
mAP: 58.5% on PASCAL VOC

```

---

### Topic 5.2: YOLO (You Only Look Once)

**How It Works:**

**Step 1: Divide Image into Grid**

```
Input: 416×416 image
Grid: 13×13 cells

Each cell predicts:
- B bounding boxes (usually 3)
- Confidence scores
- Class probabilities (80 classes)

```

**Step 2: Single CNN Pass**

```
Image (416×416×3)
    ↓
[YOLO CNN] (single pass!)
    ↓
Output: 13×13×255
        ↑    ↑
      grid  (3 boxes × 85 params)

```

**Predictions Per Cell:**

```
For each bounding box:
- x, y (center coordinates)
- w, h (width, height)
- confidence (objectness score)
- 80 class probabilities

Total: 5 + 80 = 85 values per box

```

**Non-Maximum Suppression (NMS):**

```
Problem: Multiple boxes for same object

Solution:
1. Sort boxes by confidence
2. Keep box with highest confidence
3. Remove overlapping boxes (IoU > threshold)
4. Repeat until no overlaps

```

**Advantages:**

- Extremely fast (45 FPS)
- Real-time detection
- Single network pass
- Good for video

**Disadvantages:**

- Lower accuracy than R-CNN
- Struggles with small objects
- Misses closely spaced objects

**YOLO Versions:**

```
YOLOv1 (2015): 45 FPS, 63.4 mAP
YOLOv3 (2018): 30 FPS, 57.9 mAP
YOLOv5 (2020): 140 FPS, 66.9 mAP
YOLOv8 (2023): 80 FPS, 69.8 mAP

```

---

### Topic 5.3: R-CNN vs YOLO Comparison

Feature | R-CNN | YOLO
Speed | 0.02 FPS (slow) | 45+ FPS (fast)
Accuracy | High | Moderate
Architecture | 2000 CNN passes | Single CNN pass
Real-time | No | Yes
Small objects | Good | Poor
Use case | High-precision tasks | Real-time video

**When to Use:**

```
Use R-CNN when:
- Accuracy is critical
- Real-time not needed
- Offline processing
- Medical imaging, satellite analysis

Use YOLO when:
- Speed is critical
- Real-time needed
- Video processing
- Autonomous vehicles, surveillance

```

---

## PRACTICAL CODE EXAMPLES

### Example 1: Transfer Learning with Frozen Layers

```python
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

# Load pre-trained ResNet18
model = models.resnet18(weights='IMAGENET1K_V1')

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for CIFAR-10 (10 classes)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

# Only fc layer will train
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
# Output: ~5,000 parameters (only fc layer)

# Prepare data
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

```

---

### Example 2: Fine-Tuning Last Layers

```python
# Stage 1: Train classifier with frozen base (done above)

# Stage 2: Fine-tune layer4 (last residual block)
for param in model.layer4.parameters():
    param.requires_grad = True

# Use different learning rates for different layers
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])

# Continue training
for epoch in range(10):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Fine-tune Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

```

---

### Example 3: Simple YOLO Inference

```python
import cv2
from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # nano model (fastest)

# Load image
image = cv2.imread('image.jpg')

# Run inference
results = model(image)

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        # Get confidence and class
        conf = box.conf[0].cpu().numpy()
        cls = box.cls[0].cpu().numpy()
        
        # Draw box if confidence > threshold
        if conf > 0.5:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f'{model.names[int(cls)]}: {conf:.2f}'
            cv2.putText(image, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save result
cv2.imwrite('detected.jpg', image)

```

---

## REAL-WORLD APPLICATIONS

### 1. Agriculture - Satellite Imagery for Loan Eligibility

**Problem:** Banks need to verify crop health for agricultural loans.

**Solution:**

```
Satellite Image → ResNet50 (transfer learning)
                → Crop health classification
                → Loan decision

Pre-trained on ImageNet → Fine-tuned on satellite images
Accuracy: 94% crop health detection

```

---

### 2. OCR/ICR (Optical Character Recognition)

**Problem:** Extract text from documents, handwritten forms.

**Solution:**

```
Document Image → CNN for character segmentation
              → ResNet18 for character classification
              → Text output

Transfer learning from MNIST → Fine-tuned on custom fonts

```

---

### 3. Exam Proctoring

**Problem:** Detect cheating behavior in online exams.

**Solution:**

```
Webcam Feed → YOLO for face detection
           → Person count detection
           → Alert if multiple people

Real-time: 30 FPS
Accuracy: 98% face detection

```

---

### 4. Surveillance and Defense

**Problem:** Detect suspicious objects/people in security footage.

**Solution:**

```
Video Stream → YOLO for object detection
            → Track suspicious behavior
            → Alert security

Real-time tracking of 50+ objects simultaneously

```

---

## KEY TAKEAWAYS

### 5 Core Concepts to Remember

1. 
**ResNet solves vanishing gradients** with skip connections that allow gradients to flow directly through the network

2. 
**Transfer learning saves time and resources** by reusing pre-trained features from ImageNet, reducing training from weeks to hours

3. 
**Freezing vs Fine-tuning strategy:**

Freeze base + train classifier (fast, small data)
Unfreeze last layers + fine-tune (better accuracy, more data)

4. 
**Quantization enables deployment** by reducing model size 4× and inference time 3× with minimal accuracy loss

5. 
**R-CNN vs YOLO tradeoff:**

R-CNN: High accuracy, slow (0.02 FPS)
YOLO: Real-time speed, moderate accuracy (45+ FPS)

---

## COMMON MISTAKES & TIPS

Mistake | Why It's Wrong | Correct Approach
Using same LR for fine-tuning | Destroys pre-trained features | Use 10-100× lower LR
Not freezing during initial training | Wastes pre-trained knowledge | Always freeze first, then fine-tune
Fine-tuning with tiny dataset | Overfits badly | Only fine-tune if >10K samples
Forgetting to set model.eval() | BatchNorm/Dropout behave incorrectly | Always use model.eval() during inference
Using R-CNN for real-time | Too slow | Use YOLO for video/real-time

---

**Vishlesan i-Hub IIT Patna × Masai School**

**Next Session:** Advanced Object Detection - Faster R-CNN, SSD, and YOLOv8 architectures

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