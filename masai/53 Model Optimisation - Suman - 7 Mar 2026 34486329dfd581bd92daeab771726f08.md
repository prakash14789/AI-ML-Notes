# 53. Model Optimisation - Suman - 7 Mar 2026

# Model Optimization: Lecture Notes

## In-Class Resources: [Click Here](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/d2a48db3-27c8-466c-a16c-74c43291e461/mtEZUoXnNgJidbj3.zip)

**Program:** Vishlesan i-Hub IIT Patna x Masai School — AIM (AI & Machine Learning)
**Session:** Model Optimization - Quantization, Pruning, TensorRT, Edge Deployment
**Prerequisites:** Deep Learning, PyTorch/TensorFlow, CNN Architecture

---

## MENTAL MAP

```
Model Optimization Session

Part 1: The Deployment Problem
├─ Research vs Production Gap
├─ Hardware Constraints
└─ Business Case for Optimization

Part 2: Quantization
├─ Precision Levels (FP32, FP16, INT8, INT4)
├─ Post-Training Quantization
├─ Quantization-Aware Training
└─ Trade-offs and Best Practices

Part 3: Pruning
├─ Unstructured Pruning
├─ Structured Pruning
├─ Iterative Pruning Workflow
└─ Combining with Quantization

Part 4: TensorRT Optimization
├─ Layer Fusion
├─ Kernel Auto-Tuning
├─ Mixed Precision
└─ Performance Gains

Part 5: Edge Deployment
├─ Mobile Frameworks (TFLite, PyTorch Mobile)
├─ Hardware Accelerators
├─ Cross-Platform Deployment
└─ Production Considerations

```

---

## THE BIG PICTURE

### The Problem We're Solving

**Scenario:** You've trained a state-of-the-art image classifier:

```
Model: ResNet152
Accuracy: 82% on ImageNet
Parameters: 60 million
Model size: 240 MB
Inference time: 250ms on V100 GPU
Training cost: $5,000

```

**Now you need to deploy it on:**

- Mobile app (50 MB limit, <100ms latency)
- Edge device (2 GB RAM, ARM CPU)
- IoT sensor (512 MB RAM, <1W power)

**The Question:** How do you make this work?

**The Answer:** Model Optimization

---

## PART 1: THE DEPLOYMENT PROBLEM

### Topic 1.1: Research vs Production Gap

**Research Environment:**

```
Hardware: NVIDIA A100 GPU ($15,000)
Memory: 80 GB VRAM
Power: 400W unlimited
Latency: No hard constraints
Budget: Grant-funded

Training: Can take days/weeks
Inference: Batch processing acceptable

```

**Production Environment:**

```
Hardware: Smartphone ARM CPU ($300 device)
Memory: 4 GB shared RAM
Power: 5W battery budget
Latency: <50ms for real-time
Budget: Per-device cost matters

Training: One-time cost
Inference: Millions of times per day, must be fast

```

**The Gap:**

```
Model Size: 240 MB vs 50 MB limit → 5× too large
Speed: 250ms vs 50ms target → 5× too slow
Power: 400W vs 5W budget → 80× too much
Cost: $15K GPU vs $300 device → 50× more expensive

```

---

### Topic 1.2: Hardware Constraints

**1. Compute Constraints:**

```
Desktop GPU (RTX 4090):
- 16,384 CUDA cores
- 82.6 TFLOPS FP32
- Can run any model

Mobile CPU (Snapdragon 8 Gen 2):
- 8 cores (1×3.2GHz + 3×2.8GHz + 4×2.0GHz)
- ~100 GFLOPS
- 826× less compute than GPU!

```

**2. Memory Constraints:**

```
Server:
- 128 GB RAM
- 80 GB VRAM
- No limits

Mobile:
- 8 GB RAM total
- OS uses 3 GB
- Apps use 2 GB
- Available: 3 GB max
- Model must fit in <1 GB

```

**3. Power Constraints:**

```
Server: Unlimited power from outlet
Mobile: 15 Wh battery

Running FP32 ResNet152:
- Power draw: 5W continuous
- Battery life: 15 Wh / 5W = 3 hours total
- With screen, apps: 1.5 hours
- UNUSABLE

Running optimized INT8 model:
- Power draw: 0.5W
- Marginal impact on battery life
- ACCEPTABLE

```

---

### Topic 1.3: Business Case for Optimization

**Case Study: Computer Vision API**

**Without Optimization:**

```
Model: ResNet152 FP32
Throughput: 100 requests/second per GPU
GPU cost: $5,000 each
Target load: 10,000 requests/second

Required GPUs: 10,000 / 100 = 100 GPUs
Capital cost: 100 × $5,000 = $500,000
Operating cost: $50,000/month (cloud)
Annual cost: $600,000

```

**With Optimization (TensorRT INT8):**

```
Model: ResNet152 INT8 + TensorRT
Throughput: 800 requests/second per GPU
Same target: 10,000 requests/second

Required GPUs: 10,000 / 800 = 13 GPUs
Capital cost: 13 × $5,000 = $65,000
Operating cost: $6,500/month
Annual cost: $78,000

Savings: $522,000 per year (87% reduction!)

```

**ROI Calculation:**

```
Optimization effort: 2 engineer-weeks = $10,000
Annual savings: $522,000
ROI: 5,220%
Payback period: 1 week

```

---

## PART 2: QUANTIZATION

### Topic 2.1: Understanding Precision Levels

**Floating Point Representations:**

**FP32 (32-bit Float):**

```
Bits: 1 sign + 8 exponent + 23 mantissa = 32 bits
Range: ±3.4 × 10³⁸
Precision: ~7 decimal digits
Example: 3.14159265

Storage: 4 bytes per number

```

**FP16 (16-bit Float):**

```
Bits: 1 sign + 5 exponent + 10 mantissa = 16 bits
Range: ±65,504
Precision: ~3 decimal digits
Example: 3.14159

Storage: 2 bytes per number
Reduction: 2× smaller than FP32

```

**INT8 (8-bit Integer):**

```
Range: -128 to +127 (signed)
or 0 to 255 (unsigned)
Precision: Integer only
Example: 3

Storage: 1 byte per number
Reduction: 4× smaller than FP32

```

**INT4 (4-bit Integer):**

```
Range: -8 to +7 (signed)
or 0 to 15 (unsigned)
Precision: Integer only
Example: 3

Storage: 0.5 bytes per number
Reduction: 8× smaller than FP32

```

---

### Topic 2.2: Post-Training Quantization (PTQ)

**What is PTQ?**

Quantize a trained model without retraining.

**Process:**

**Step 1: Collect Statistics**

```python
# Run calibration dataset through model
calibration_data = get_calibration_data(1000 samples)

for data in calibration_data:
    activations = model(data)
    # Record min/max of each layer's activations

```

**Step 2: Calculate Quantization Parameters**

```
For each layer:
1. Find activation range: [min_val, max_val]
2. Map to INT8 range: [-128, 127]

scale = (max_val - min_val) / 255
zero_point = -min_val / scale

Example:
Activations range: [-5.2, 3.8]
scale = (3.8 - (-5.2)) / 255 = 9.0 / 255 = 0.0353
zero_point = -(-5.2) / 0.0353 = 147

```

**Step 3: Quantize Weights**

```python
# Quantize each weight
quantized_weight = round(weight / scale) + zero_point
quantized_weight = clip(quantized_weight, -128, 127)

# Example
weight_fp32 = 0.523
scale = 0.01
quantized = round(0.523 / 0.01) = 52 (INT8)

```

**Step 4: Quantized Inference**

```python
# During inference
input_int8 = quantize(input_fp32)
output_int8 = quantized_model(input_int8)
output_fp32 = dequantize(output_int8)

```

**PyTorch Implementation:**

```python
import torch.quantization as quantization

# Load trained FP32 model
model_fp32 = resnet50(pretrained=True)
model_fp32.eval()

# Dynamic quantization (easiest)
model_int8 = quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear, torch.nn.Conv2d},  # Layers to quantize
    dtype=torch.qint8
)

# Save quantized model
torch.save(model_int8.state_dict(), 'model_int8.pth')

# Size comparison
fp32_size = os.path.getsize('model_fp32.pth') / 1e6  # MB
int8_size = os.path.getsize('model_int8.pth') / 1e6
print(f'FP32: {fp32_size:.1f} MB')
print(f'INT8: {int8_size:.1f} MB')
print(f'Reduction: {fp32_size/int8_size:.1f}×')

```

**Expected Output:**

```
FP32: 97.8 MB
INT8: 24.4 MB
Reduction: 4.0×

```

---

### Topic 2.3: Quantization-Aware Training (QAT)

**What is QAT?**

Train the model with quantization in mind, allowing it to adapt to lower precision.

**Why QAT?**

```
Post-Training Quantization:
- Easy (no retraining)
- Fast (quantize in minutes)
- Accuracy loss: 1-3%

Quantization-Aware Training:
- Harder (requires retraining)
- Slow (retrain for 10-20 epochs)
- Accuracy loss: 0.5-1%

Trade-off: Extra training time for better accuracy

```

**How QAT Works:**

**Fake Quantization:**

```
During training:
1. Forward pass in FP32
2. Simulate quantization: round to INT8 values
3. Backward pass in FP32
4. Model learns to be robust to quantization

forward_fp32 = weight_fp32 × input_fp32
forward_quantized = quantize(weight_fp32) × quantize(input_fp32)
                  = (round to INT8) × (round to INT8)

Gradient flows through quantization operation

```

**PyTorch QAT:**

```python
import torch.quantization as quantization

# Prepare model for QAT
model.train()
model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
quantization.prepare_qat(model, inplace=True)

# Train with fake quantization
for epoch in range(10):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)  # Forward with fake quant
        loss = criterion(output, target)
        loss.backward()  # Gradients aware of quantization
        optimizer.step()

# Convert to true INT8 model
model.eval()
quantization.convert(model, inplace=True)

# Now model runs in INT8

```

**QAT vs PTQ Results:**

```
Model: ResNet50 on ImageNet

FP32 baseline: 76.1% Top-1 accuracy

Post-Training Quantization (INT8):
Accuracy: 75.3% (-0.8%)
Time: 10 minutes

Quantization-Aware Training (INT8):
Accuracy: 75.8% (-0.3%)
Time: 8 hours (10 epochs retraining)

QAT: Better accuracy, but 48× more time investment

```

---

### Topic 2.4: Mixed Precision Quantization

**Concept:**

Different layers tolerate quantization differently. Use optimal precision per layer.

**Layer Sensitivity Analysis:**

```python
def measure_layer_sensitivity(model, layer_name):
    # Quantize only this layer
    quantize_layer(model, layer_name)
    
    # Measure accuracy drop
    accuracy = evaluate(model, val_loader)
    accuracy_drop = baseline_accuracy - accuracy
    
    # Restore layer
    restore_layer(model, layer_name)
    
    return accuracy_drop

# Measure all layers
sensitivities = {}
for name, layer in model.named_modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        sensitivities[name] = measure_layer_sensitivity(model, name)

# Sort by sensitivity
sorted_layers = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)

print("Most sensitive layers (keep FP16):")
for name, drop in sorted_layers[:5]:
    print(f"{name}: {drop:.2f}% accuracy drop")

```

**Mixed Precision Strategy:**

```
ResNet50 Analysis:

Layer 1 (conv1): 0.1% drop → INT8 ✓
Layer 2-20: 0.2% drop each → INT8 ✓
Layer 21-40: 0.5% drop each → INT8 ✓
Layer 41-48: 1.0% drop each → FP16 (sensitive!)
Final FC: 2.0% drop → FP16 (very sensitive!)

Strategy:
- Layers 1-40: INT8 (98% of parameters)
- Layers 41-49: FP16 (2% of parameters)

Result:
- Model size: 26 MB (mostly INT8)
- Accuracy loss: 0.4% (FP16 on sensitive layers saves accuracy)
- Inference: 3.5× faster (most layers INT8)

```

---

## PART 3: PRUNING

### Topic 3.1: Unstructured Pruning

**Concept:**

Remove individual weights based on magnitude.

**Algorithm:**

```
1. Train model to convergence
2. Identify smallest weights: |weight| < threshold
3. Set those weights to zero
4. Fine-tune remaining weights
5. Repeat until target sparsity

```

**Example:**

```
Original weights matrix:
[0.52, -0.003, 0.89, 0.001]
[-0.67, 0.91, -0.005, 0.78]
[0.001, -0.82, 0.002, -0.71]

After magnitude pruning (threshold=0.01):
[0.52, 0, 0.89, 0]
[-0.67, 0.91, 0, 0.78]
[0, -0.82, 0, -0.71]

Pruned: 5 out of 12 weights (42% sparsity)

```

**PyTorch Implementation:**

```python
import torch.nn.utils.prune as prune

# Prune 30% of weights in conv1 layer
prune.l1_unstructured(
    model.conv1,
    name='weight',
    amount=0.3  # 30% pruning
)

# Check sparsity
sparsity = 100. * float(torch.sum(model.conv1.weight == 0)) / float(model.conv1.weight.nelement())
print(f'Sparsity: {sparsity:.1f}%')

# Make pruning permanent
prune.remove(model.conv1, 'weight')

```

**Benefits:**

```
Sparsity: 60%
Model size: 100 MB → 40 MB (with compression)
Inference: 1.5× faster (with sparse libraries)
Accuracy loss: 1-2%

```

**Limitations:**

```
- Irregular sparsity pattern
- Hard to accelerate on standard hardware
- Needs specialized sparse libraries (NVIDIA cuSPARSE, Intel MKL-DNN)
- Actual speedup often less than theoretical

```

---

### Topic 3.2: Structured Pruning

**Concept:**

Remove entire structures (channels, filters, layers) rather than individual weights.

**Why Structured?**

```
Unstructured (remove random weights):
- Irregular pattern
- Hard to accelerate
- Actual speedup: 1.5× (vs 2.5× theoretical)

Structured (remove entire channels):
- Regular pattern
- Easy to accelerate
- Actual speedup: 2× (matches theoretical)

```

**Channel Pruning:**

```
Original Conv Layer:
- Input: 128 channels
- Output: 256 channels
- Filters: 256 filters of size 3×3×128
- Parameters: 256 × 3 × 3 × 128 = 295,936

After Pruning (remove 50% output channels):
- Input: 128 channels
- Output: 128 channels (removed 128)
- Filters: 128 filters of size 3×3×128
- Parameters: 128 × 3 × 3 × 128 = 147,968

Reduction: 50% parameters, 50% FLOPs, 2× speedup

```

**Channel Importance Scoring:**

```python
def calculate_channel_importance(layer):
    # Method 1: L1 norm of filter weights
    importance = torch.sum(torch.abs(layer.weight), dim=(1,2,3))
    
    # Method 2: Average activation magnitude
    # importance = torch.mean(torch.abs(activations), dim=(0,2,3))
    
    return importance

# Prune least important channels
importance = calculate_channel_importance(model.conv1)
num_to_prune = int(0.3 * len(importance))  # Prune 30%
indices_to_prune = torch.argsort(importance)[:num_to_prune]

# Remove channels
model.conv1 = prune_channels(model.conv1, indices_to_prune)

```

**Structured Pruning Results:**

```
ResNet50:
Original: 25M parameters, 4.1B FLOPs

Prune 30% of channels:
Parameters: 17.5M (-30%)
FLOPs: 2.87B (-30%)
Inference: 1.43× faster (actual measured)
Accuracy: 75.2% (original 76.1%, -0.9% loss)

More predictable than unstructured pruning

```

---

### Topic 3.3: Iterative Magnitude Pruning (IMP)

**The Lottery Ticket Hypothesis:**

> A randomly-initialized neural network contains a subnetwork ("winning ticket") that, when trained in isolation, can match the accuracy of the original network.
> 

**Iterative Pruning Process:**

```
1. Train network to convergence → 76% accuracy
2. Prune 10% of smallest weights → 10% sparsity
3. Reset remaining weights to initial values
4. Retrain from scratch → 75.8% accuracy
5. Prune another 10% → 19% sparsity
6. Reset and retrain → 75.5% accuracy
7. Repeat until target sparsity
8. Final: 60% sparsity, 74.5% accuracy

One-shot pruning 60% → 71% accuracy (much worse!)
Iterative pruning 60% → 74.5% accuracy (much better!)

```

**PyTorch Implementation:**

```python
# Iterative pruning
initial_weights = copy.deepcopy(model.state_dict())  # Save initial weights
sparsity = 0
target_sparsity = 0.6

while sparsity < target_sparsity:
    # Train to convergence
    train(model, epochs=90)
    
    # Prune 10%
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=0.1)
    
    sparsity += 0.1
    
    # Reset unpruned weights to initial values
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                mask = (param != 0).float()
                param.copy_(initial_weights[name] * mask)
    
    print(f'Sparsity: {sparsity:.0%}, retraining...')

# Final model: 60% sparse, trained with iterative pruning

```

**Results Comparison:**

```
Method: One-shot pruning
Sparsity: 60%
Accuracy: 71.2%
Training time: 90 epochs

Method: Iterative pruning (IMP)
Sparsity: 60%
Accuracy: 74.5%
Training time: 6 rounds × 90 epochs = 540 epochs

IMP: 3.3% better accuracy, but 6× more training time

```

---

### Topic 3.4: Combining Quantization and Pruning

**Sequential Optimization:**

**Strategy 1: Prune then Quantize**

```
Step 1: Prune (60% sparsity)
- Size: 100 MB → 40 MB
- Accuracy: 76.1% → 75.1%

Step 2: Quantize pruned model (INT8)
- Size: 40 MB → 10 MB
- Accuracy: 75.1% → 74.3%

Final: 10× smaller, 74.3% accuracy (1.8% loss)

```

**Strategy 2: Quantize then Prune**

```
Step 1: Quantize (INT8)
- Size: 100 MB → 25 MB
- Accuracy: 76.1% → 75.3%

Step 2: Prune quantized model
- Size: 25 MB → 10 MB
- Accuracy: 75.3% → 73.8%

Final: 10× smaller, 73.8% accuracy (2.3% loss)

Strategy 1 (Prune→Quantize) is better!

```

**Joint Optimization (Best):**

```
Train with both pruning and quantization:
1. Start with QAT (quantization-aware training)
2. Gradually prune during QAT
3. Model adapts to both simultaneously

Final: 10× smaller, 75.0% accuracy (1.1% loss)
Best of both worlds!

```

---

## PART 4: TENSORRT OPTIMIZATION

### Topic 4.1: What TensorRT Does

**TensorRT Optimizations:**

**1. Layer Fusion:**

```
Before: Separate layers
Conv → BatchNorm → ReLU (3 kernel launches)

After TensorRT fusion:
Conv-BatchNorm-ReLU (1 fused kernel)

Savings:
- Memory bandwidth: 3 reads/writes → 1 read/write
- Kernel launches: 3 → 1
- Latency: 2.1ms → 0.9ms (2.3× faster)

```

**2. Vertical Fusion:**

```
Before:
Layer A → Intermediate tensor → Layer B

After:
Layer A-B fused (no intermediate tensor)

Benefit: Saves memory allocation and bandwidth

```

**3. Horizontal Fusion:**

```
Before:
    Input
   /     \
Conv1   Conv2  (2 separate kernels)
   \     /
    Concat

After:
    Input
      |
  Conv1+Conv2 (1 fused kernel writing to 2 outputs)
      |
    Output

Benefit: Process both convs in same kernel

```

---

### Topic 4.2: Kernel Auto-Tuning

**What is Kernel Tuning?**

TensorRT tests multiple implementations and chooses fastest:

```
Conv2d layer possibilities:
- cuDNN: 8 different algorithms
- Winograd transform: 3 variants
- FFT-based: 2 implementations
- Direct convolution: 4 tile sizes

Total: 17 different implementations!

TensorRT tests all 17:
Algorithm 1: 3.2ms
Algorithm 2: 2.8ms ← fastest!
Algorithm 3: 3.5ms
...

Selects Algorithm 2 (2.8ms)

```

**Hardware-Specific Tuning:**

```
Same model on different GPUs:

Tesla T4:
- Algorithm 7 fastest: 2.1ms

RTX 3090:
- Algorithm 3 fastest: 0.9ms

Jetson Xavier (embedded):
- Algorithm 12 fastest: 8.3ms

TensorRT picks optimal for each hardware

```

---

### Topic 4.3: Mixed Precision

**TensorRT FP16 Mode:**

```python
import tensorrt as trt

# Build TensorRT engine with FP16
builder = trt.Builder(TRT_LOGGER)
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16

engine = builder.build_engine(network, config)

# TensorRT automatically:
# 1. Uses FP16 where beneficial
# 2. Keeps FP32 for sensitive operations
# 3. Inserts conversions where needed

```

**Automatic Mixed Precision:**

```
TensorRT analyzes each layer:

Layer 1 (Conv): FP16 (2× faster, negligible accuracy loss)
Layer 2 (Conv): FP16
...
Layer 48 (Conv): FP16
Layer 49 (FC): FP32 (sensitive to precision)
Layer 50 (Softmax): FP32 (needs accuracy)

Result: 
- 96% of operations in FP16 (fast)
- 4% in FP32 (accuracy-critical)
- Best of both worlds

```

---

### Topic 4.4: TensorRT Performance

**Benchmark: ResNet50**

```
Hardware: NVIDIA T4 GPU
Batch size: 1
Input: 224×224×3

PyTorch eager mode:
Latency: 15.2ms
Throughput: 66 FPS

PyTorch JIT (TorchScript):
Latency: 10.8ms
Throughput: 93 FPS

TensorRT FP32:
Latency: 4.2ms (3.6× faster than PyTorch)
Throughput: 238 FPS

TensorRT FP16:
Latency: 2.1ms (7× faster!)
Throughput: 476 FPS

TensorRT INT8:
Latency: 1.3ms (11× faster!)
Throughput: 769 FPS

```

**Production Impact:**

```
API serving 10,000 requests/second:

PyTorch (66 FPS):
GPUs needed: 10,000 / 66 = 152 GPUs
Cost: 152 × $5,000 = $760,000

TensorRT INT8 (769 FPS):
GPUs needed: 10,000 / 769 = 13 GPUs
Cost: 13 × $5,000 = $65,000

Savings: $695,000 (91% reduction!)

```

---

### Topic 4.5: TensorRT Workflow

**Step-by-Step Conversion:**

```python
import torch
import torch.onnx
import tensorrt as trt
from torch2trt import torch2trt

# Step 1: Export PyTorch model to ONNX
model = resnet50(pretrained=True).cuda().eval()
dummy_input = torch.randn(1, 3, 224, 224).cuda()

torch.onnx.export(
    model,
    dummy_input,
    "resnet50.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)

# Step 2: Build TensorRT engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# Parse ONNX
parser = trt.OnnxParser(network, TRT_LOGGER)
with open("resnet50.onnx", 'rb') as model_file:
    parser.parse(model_file.read())

# Configure
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB
config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16
config.set_flag(trt.BuilderFlag.STRICT_TYPES)  # Or INT8 calibration

# Build engine
engine = builder.build_engine(network, config)

# Save engine
with open("resnet50.trt", "wb") as f:
    f.write(engine.serialize())

print("TensorRT engine built successfully!")

```

**Inference with TensorRT:**

```python
import pycuda.driver as cuda
import pycuda.autoinit

# Load engine
with open("resnet50.trt", "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

# Create context
context = engine.create_execution_context()

# Allocate memory
input_shape = (1, 3, 224, 224)
output_shape = (1, 1000)

input_mem = cuda.mem_alloc(np.prod(input_shape) * np.dtype(np.float32).itemsize)
output_mem = cuda.mem_alloc(np.prod(output_shape) * np.dtype(np.float32).itemsize)

# Inference
stream = cuda.Stream()
bindings = [int(input_mem), int(output_mem)]

# Copy input
cuda.memcpy_htod_async(input_mem, input_data, stream)

# Execute
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

# Copy output
cuda.memcpy_dtoh_async(output_data, output_mem, stream)
stream.synchronize()

print(f"Output: {output_data}")

```

---

## PART 5: EDGE DEPLOYMENT

### Topic 5.1: TensorFlow Lite

**Purpose:**

Lightweight framework for mobile and embedded deployment.

**Conversion Pipeline:**

```python
import tensorflow as tf

# Load Keras model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimization flags
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Weight quantization
converter.target_spec.supported_types = [tf.float16]  # FP16

# Convert
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Size comparison
keras_size = os.path.getsize('model.h5') / 1e6
tflite_size = len(tflite_model) / 1e6
print(f'Keras: {keras_size:.1f} MB')
print(f'TFLite: {tflite_size:.1f} MB')
print(f'Reduction: {keras_size/tflite_size:.1f}×')

```

**Output:**

```
Keras: 14.2 MB
TFLite: 3.5 MB
Reduction: 4.1×

```

**INT8 Quantization:**

```python
def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        yield [data]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

```

**Result:**

```
FP32 TFLite: 14.2 MB
INT8 TFLite: 3.6 MB (4× smaller)
Inference: 3× faster on mobile CPU
Accuracy loss: 1.2%

```

---

### Topic 5.2: PyTorch Mobile

**Mobile Optimization:**

```python
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

# Load model
model = torchvision.models.mobilenet_v2(pretrained=True)
model.eval()

# Trace model
dummy_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, dummy_input)

# Optimize for mobile
optimized_model = optimize_for_mobile(traced_model)

# Save
optimized_model._save_for_lite_interpreter("model_mobile.ptl")

print("Mobile model saved!")

```

**Mobile Optimizations Applied:**

```
1. Operator fusion (Conv+BN+ReLU)
2. Constant folding
3. Dead code elimination
4. Memory planning
5. Specialized mobile kernels

Result: 2-3× faster inference on mobile

```

---

### Topic 5.3: Hardware Accelerators

**1. Apple Neural Engine:**

```
Available on: iPhone 12+ (A14 chip onwards)
Performance: 11 TOPS
Power: 0.5W
Precision: INT8, FP16
Framework: Core ML

Optimization:
- Convert to Core ML format
- Quantize to INT8
- Runs on Neural Engine automatically

Result: 10× faster than CPU, 5× more power-efficient

```

**Core ML Conversion:**

```python
import coremltools as ct

# Convert PyTorch to Core ML
traced_model = torch.jit.trace(model, dummy_input)
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=(1, 3, 224, 224))]
)

# Quantize
mlmodel_int8 = ct.models.neural_network.quantization_utils.quantize_weights(mlmodel, nbits=8)

# Save
mlmodel_int8.save("model.mlmodel")

```

**2. Google Edge TPU:**

```
Hardware: Coral USB Accelerator / Dev Board
Performance: 4 TOPS
Power: 2W
Cost: $60
Precision: INT8 only

Use case: Smart cameras, robotics, IoT

```

**Edge TPU Compilation:**

```bash
# Requires TensorFlow Lite INT8 model
edgetpu_compiler model_quant.tflite

# Output: model_quant_edgetpu.tflite
# Optimized for Edge TPU

```

**Performance:**

```
MobileNetV2 on Raspberry Pi 4:
CPU only: 150ms per frame
Edge TPU: 12ms per frame

12× speedup!

```

---

### Topic 5.4: Cross-Platform Deployment

**ONNX Runtime:**

**Universal Format:**

```python
# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=11,
    do_constant_folding=True
)

# Run with ONNX Runtime
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: input_data})

```

**Platform Support:**

```
ONNX Runtime runs on:
- Windows (x86, ARM)
- macOS (x86, ARM/M1)
- Linux (x86, ARM, RISC-V)
- iOS
- Android
- WebAssembly (browsers!)

Single model file, runs everywhere

```

---

## PRACTICAL CODE EXAMPLES

### Example 1: Complete Quantization Workflow

```python
import torch
import torchvision
import torch.quantization as quantization
from torch.utils.data import DataLoader

# Load pre-trained model
model = torchvision.models.resnet50(pretrained=True)
model.eval()

# Prepare calibration dataset
val_dataset = torchvision.datasets.ImageNet(
    'path/to/imagenet',
    split='val',
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)
cal_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# Configure quantization
model.qconfig = quantization.get_default_qconfig('fbgemm')
quantization.prepare(model, inplace=True)

# Calibrate (collect statistics)
print("Calibrating...")
with torch.no_grad():
    for i, (images, _) in enumerate(cal_loader):
        model(images)
        if i >= 100:  # 100 batches for calibration
            break

# Convert to INT8
quantization.convert(model, inplace=True)
print("Quantization complete!")

# Evaluate
def evaluate(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

accuracy = evaluate(model, cal_loader)
print(f"INT8 Accuracy: {accuracy:.2f}%")

# Save quantized model
torch.save(model.state_dict(), 'resnet50_int8.pth')

# Size comparison
fp32_size = 97.8  # MB (known)
int8_size = os.path.getsize('resnet50_int8.pth') / 1e6
print(f"Size reduction: {fp32_size/int8_size:.1f}×")

```

---

### Example 2: Iterative Pruning

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy

def train(model, train_loader, epochs=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def iterative_pruning(model, train_loader, val_loader, target_sparsity=0.6):
    # Save initial weights
    initial_state = copy.deepcopy(model.state_dict())
    
    sparsity = 0.0
    prune_rate = 0.1  # 10% per iteration
    
    results = []
    
    while sparsity < target_sparsity:
        print(f"\n=== Iteration: {sparsity:.0%} sparsity ===")
        
        # Train
        train(model, train_loader, epochs=20)
        
        # Evaluate
        accuracy = evaluate(model, val_loader)
        results.append((sparsity, accuracy))
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Prune 10%
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=prune_rate)
        
        sparsity += prune_rate
        
        # Reset remaining weights to initial values (Lottery Ticket)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight_orig' in name:
                    # Get mask
                    mask_name = name.replace('_orig', '_mask')
                    mask = dict(model.named_buffers())[mask_name]
                    
                    # Reset to initial values
                    init_name = name.replace('_orig', '')
                    param.copy_(initial_state[init_name] * mask)
    
    return results

# Run iterative pruning
model = torchvision.models.resnet18(pretrained=True)
results = iterative_pruning(model, train_loader, val_loader, target_sparsity=0.6)

# Plot results
import matplotlib.pyplot as plt
sparsities, accuracies = zip(*results)
plt.plot(sparsities, accuracies, marker='o')
plt.xlabel('Sparsity')
plt.ylabel('Accuracy (%)')
plt.title('Iterative Pruning Results')
plt.show()

```

---

## KEY TAKEAWAYS

### 5 Core Principles

1. 
**Quantization reduces precision:** FP32 → INT8 gives 4× smaller models and 2-4× faster inference with 1-2% accuracy loss

2. 
**Pruning removes redundancy:** 60% of weights can often be removed with minimal accuracy impact, especially with iterative pruning

3. 
**TensorRT optimizes inference:** Layer fusion, kernel tuning, and mixed precision provide 5-10× speedup on NVIDIA GPUs

4. 
**Edge deployment requires optimization:** Mobile/embedded devices need <50 MB models and <50ms latency, impossible without optimization

5. 
**Combine techniques for best results:** Prune then quantize gives 10× compression and 4-8× speedup with <2% accuracy loss

---

## DECISION FRAMEWORK

### When to Use Each Technique

Scenario | Recommended Optimization | Why
Cloud API (GPU available) | TensorRT FP16 | 5× speedup, no accuracy loss, easy
Mobile app | Quantization INT8 + Pruning 40% | Small size (<50 MB), fast, -1.5% accuracy OK
IoT/Embedded | Aggressive INT4 + 70% pruning | Extreme size/power constraints
Real-time video (edge) | TensorRT INT8 + structured pruning | Speed critical, GPU available
High accuracy critical | Quantization FP16 only | Minimal accuracy loss (<0.5%)

---

**Vishlesan i-Hub IIT Patna × Masai School**

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