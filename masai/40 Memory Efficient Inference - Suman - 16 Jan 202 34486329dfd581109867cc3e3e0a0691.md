# 40. Memory Efficient Inference - Suman - 16 Jan 2026

# Memory-Efficient Inference: Comprehensive Lecture Notes

**Prerequisites:** PyTorch basics, HuggingFace model loading, understanding of data types (float32, float16, int8).

**Time to complete:** 35-45 minutes

**What you'll be able to do:**

- Load large models with 8-bit and 4-bit quantization
- Choose appropriate precision based on task requirements
- Implement GPU/CPU hybrid inference strategies
- Measure and optimize memory usage

---

## 1. Introduction: What is Memory-Efficient Inference and Why Should You Care?

### Core Definition

Memory-efficient inference refers to techniques that reduce the memory footprint of neural network models during inference (prediction), enabling larger models to run on limited hardware. The primary technique is quantization—representing weights in lower precision formats (8-bit or 4-bit integers instead of 32-bit floats). This trades a small amount of accuracy for dramatic memory savings.

### A Simple Analogy

Think of quantization like choosing between a high-resolution photo and a compressed JPEG. The original might be 50MB, while the JPEG is 2MB—you lose some detail, but for most purposes the image looks identical. Quantization compresses model weights the same way: significant size reduction with minimal visible quality loss.

This analogy breaks down because neural network accuracy degradation isn't always "invisible"—some tasks (especially math) are more sensitive to precision loss than others.

### Why This Matters to You

**Problem it solves:** A 7B parameter model needs about 28GB of VRAM in full precision, requiring expensive A100 or H100 GPUs. With 8-bit quantization, it fits on consumer GPUs costing a fraction of the price.

**What you'll gain:**

- Ability to run large models on accessible hardware
- Knowledge to optimize inference costs in production
- Skills to make informed precision/quality trade-offs

**Real-world context:** Projects like llama.cpp, GGML, and bitsandbytes have made local LLM inference a reality for millions of users on consumer hardware.

---

## 2. The Foundation: Core Concepts Explained

### Concept A: Memory Requirements Calculation

**Definition:** The memory a model needs depends on the number of parameters and the precision of each parameter. For inference without gradients, memory ≈ parameters × bytes_per_parameter.

**Key characteristics:**

- **FP32 (32-bit float):** 4 bytes per parameter — 7B model needs ~28GB
- **FP16 (16-bit float):** 2 bytes per parameter — 7B model needs ~14GB
- **INT8 (8-bit integer):** 1 byte per parameter — 7B model needs ~7GB
- **INT4 (4-bit integer):** 0.5 bytes per parameter — 7B model needs ~3.5GB

**A concrete example:**

```python
# Memory calculation
def calculate_memory(params_billions, precision_bits):
    bytes_per_param = precision_bits / 8
    memory_gb = params_billions * bytes_per_param
    return memory_gb

print(f"7B FP32: {calculate_memory(7, 32):.1f}GB")  # 28GB
print(f"7B FP16: {calculate_memory(7, 16):.1f}GB")  # 14GB
print(f"7B INT8: {calculate_memory(7, 8):.1f}GB")   # 7GB
print(f"7B INT4: {calculate_memory(7, 4):.1f}GB")   # 3.5GB

```

**Common confusion:** These are just weight storage requirements. Actual inference also needs memory for activations, KV cache, and framework overhead—typically 1.5-2x the weight memory.

---

### Concept B: Quantization with bitsandbytes

**Definition:** The bitsandbytes library provides efficient 8-bit and 4-bit matrix multiplication routines that maintain accuracy while dramatically reducing memory. It integrates seamlessly with HuggingFace Transformers.

**How it relates to Memory Calculation:** Bitsandbytes implements the INT8 and INT4 formats, handling the conversion from floating-point to integer and back during computation.

**Key characteristics:**

- **LLM.int8():** 8-bit quantization that handles outliers specially for better accuracy
- **4-bit NormalFloat (NF4):** Custom data type optimized for normally-distributed weights
- **Double quantization:** Quantizes the quantization constants for additional savings

**A concrete example:**

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 8-bit configuration
config_8bit = BitsAndBytesConfig(load_in_8bit=True)

# 4-bit configuration with NF4
config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    quantization_config=config_4bit,
    device_map="auto"
)

```

**Remember:** 8-bit is the safe default—nearly identical to full precision. 4-bit offers more savings but requires careful evaluation for your specific task.

---

### How Memory Calculation and Quantization Work Together

Understanding memory requirements tells you what precision you need. If your GPU has 8GB and you want to run a 7B model, you need at least 8-bit (7GB) or 4-bit (3.5GB). Bitsandbytes implements the quantization to make this possible.

---

## 3. Seeing It in Action: Worked Examples

**Tip:** Study these examples carefully before attempting the practice task. Understanding *why* each step is taken is more important than memorizing the steps.

### Example 1: Basic 8-bit Model Loading

**Scenario:** Load a 7B model on a GPU with 12GB VRAM.

**Our approach:** Use 8-bit quantization which requires about 7GB, leaving room for activations.

**Step-by-step solution:**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Check available GPU memory
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {gpu_memory:.1f}GB")

# Step 2: Load tokenizer (same regardless of precision)
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 3: Load model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

# Step 4: Check memory usage
memory_used = torch.cuda.memory_allocated() / 1e9
print(f"Model memory: {memory_used:.1f}GB")

# Step 5: Run inference
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```

**Output:**

```
GPU Memory: 12.0GB
Model memory: 7.2GB
The future of artificial intelligence is both promising and uncertain...

```

**What just happened:** The model loaded using only 7.2GB instead of the full 28GB, leaving 4.8GB for inference operations. Generation works normally.

**Check your understanding:** Why do we need to leave memory headroom beyond just the model weights?

---

### Example 2: 4-bit Quantization for Limited Hardware

**Scenario:** You have a laptop with only 6GB GPU VRAM but want to run a 7B model.

**What's different:** We use 4-bit quantization which cuts requirements in half again.

**Solution:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Step 1: Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 - optimized for weights
    bnb_4bit_compute_dtype=torch.float16, # Compute in fp16 for speed
    bnb_4bit_use_double_quant=True        # Quantize the quantization constants
)

# Step 2: Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Step 3: Check memory
memory_used = torch.cuda.memory_allocated() / 1e9
print(f"4-bit model memory: {memory_used:.1f}GB")  # About 4GB

# Step 4: Inference
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0]))

```

**Output:**

```
4-bit model memory: 4.1GB

```

**Key lesson:** 4-bit NF4 with double quantization achieves remarkable compression—4.1GB for a 7B model. Quality remains high for most text generation tasks.

---

### Example 3: CPU-GPU Hybrid Inference

**Background:** Sometimes you want to run a model larger than your GPU can fully hold.

**The challenge:** Run a 13B model with only 8GB GPU VRAM by offloading some layers to CPU.

**The approach:** Use device_map to automatically split the model across GPU and CPU.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Step 1: Create a memory map allowing CPU offload
max_memory = {
    0: "7GB",    # Use 7GB on GPU 0
    "cpu": "16GB" # Allow 16GB on CPU
}

# Step 2: Load model with automatic layer distribution
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    load_in_8bit=True,
    device_map="auto",
    max_memory=max_memory
)

# Step 3: Inspect where layers ended up
print("Layer distribution:")
for name, param in model.named_parameters():
    if 'weight' in name and 'layers.0' in name:
        print(f"  {name}: {param.device}")
        break

# Step 4: Inference (automatic device movement)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
inputs = tokenizer("Hello", return_tensors="pt")
# Note: Don't manually move to cuda - let device_map handle it

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))

```

**Why this approach:** Layers on GPU run fast; layers on CPU run slower. The model automatically moves activations between devices as the forward pass progresses. This is slower than pure GPU but makes running larger models possible.

**Caution:** CPU offload significantly increases inference time (2-10x slower depending on how much is offloaded). Use pure GPU when possible.

---

## 4. Common Pitfalls: What Can Go Wrong and How to Avoid It

- **The Mistake:** Using 4-bit quantization for math-heavy tasks

**Why It's a Problem:** Quantization accumulates small errors that compound in arithmetic operations
**The Right Approach:** Use 8-bit or full precision for tasks requiring precise calculations
**Why This Works:** 8-bit has much smaller quantization error; full precision has none

---

- **The Mistake:** Manually moving tensors to CUDA with device_map="auto"

**Why It's a Problem:** device_map handles placement; manual moves can break the layer distribution
**The Right Approach:** Let the model handle device placement when using device_map

`# WRONG when using device_map
inputs = tokenizer(text, return_tensors="pt").to("cuda")

# RIGHT - just pass inputs, model handles devices
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)  # Works correctly`

**Why This Works:** The model internally manages where each layer's inputs/outputs should be

---

- **The Mistake:** Expecting identical outputs between quantized and full-precision models

**Why It's a Problem:** Quantization is lossy—outputs will differ slightly
**The Right Approach:** Evaluate quantized models on your specific task; accept small quality trade-offs
**Why This Works:** Most tasks tolerate the tiny accuracy differences; some don't—testing reveals which

**If you're stuck:** Start with 8-bit—it's the safest balance of memory savings and quality preservation.

---

## 5. Your Turn: Practice & Self-Assessment

### Practice Task (Estimated: 20 minutes)

**The Challenge:** Build a script that benchmarks different quantization levels on the same model and task.

**Specifications:**

- Load a model in: full precision, 8-bit, and 4-bit
- Measure memory usage for each
- Generate text from the same prompt with each version
- Compare outputs for quality differences
- Measure inference speed for each

**Hint:** Use `torch.cuda.memory_allocated()` for memory, `time.time()` for speed. Keep prompts identical across versions. Calculate perplexity or manual quality rating for comparison.

**Extension (optional):** Test on different task types (creative writing vs. math) to see where quantization hurts most.

---

### Check Your Understanding

1. 
**Explanation question:** Why does 8-bit quantization maintain accuracy better than simple rounding to integers would suggest?

2. 
**Application question:** You have a 24GB GPU and want to serve a 70B model for production inference. What combination of techniques would you use?

3. 
**Error analysis:**

```python
model = AutoModelForCausalLM.from_pretrained(
    "llama-7b",
    load_in_4bit=True
)
inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
outputs = model(inputs)

```

What's wrong with this code?

1. **Transfer question:** How would you decide whether to use a smaller full-precision model versus a larger quantized model for a production application?

**Answers & Explanations:**

1. 
bitsandbytes LLM.int8() uses vector-wise quantization and handles outlier features (values far from normal distribution) in higher precision. This preserves the most important weight values while aggressively compressing the majority.

2. 
For 70B on 24GB: Use 4-bit quantization (~35GB → ~17GB), combine with device_map to split across GPU and CPU if needed. Consider GPTQ/AWQ for even more efficient quantization. Use paged attention (vLLM) to handle KV cache efficiently.

3. 
Missing `device_map="auto"`. When using load_in_4bit, you must specify device_map or the model won't load correctly. Also, after loading with device_map, you shouldn't manually move inputs to cuda—let the model handle device placement.

4. 
Compare: (a) accuracy on your specific task, (b) latency requirements, (c) cost per inference. A quantized larger model often beats a smaller full-precision model on quality, but has higher latency. For real-time apps, the smaller model may win; for batch processing, larger quantized may be better.

---

### Self-Assessment Checklist

You've mastered this topic if you can:

- Calculate memory requirements for different precisions
- Load models with 8-bit and 4-bit quantization using bitsandbytes
- Configure device_map for automatic multi-device distribution
- Measure and compare memory usage between configurations
- Identify tasks where quantization quality loss is acceptable
- Implement CPU offloading for models too large for GPU

**If you checked fewer than 5 boxes:** Practice loading the same model with different configurations and measuring the differences.

---

## 6. Consolidation: Key Takeaways & Next Steps

### The Essential Ideas

- **Memory equation:** memory_GB = parameters_billions × bytes_per_precision
- **8-bit is the safe default:** ~75% memory savings with minimal quality loss
- **4-bit for maximum savings:** Enables running 7B on 6GB GPU, but evaluate quality

### Mental Model Check

By now, you should think of precision/quantization as: A compression dial that trades accuracy for memory. Turn it down (fewer bits) to fit larger models; keep it high for tasks requiring precision. Most text generation works fine at 8-bit or even 4-bit.

### What You Can Now Do

You can run models that previously required expensive hardware on consumer GPUs. This enables local LLM deployment, cost-effective cloud inference, and experimentation with larger models.

### Next Steps

**To deepen this knowledge:** Explore GPTQ and AWQ for even more efficient quantization with careful calibration.

**To build on this:** Learn about vLLM and TensorRT-LLM for production inference optimization.

**Additional resources:** bitsandbytes documentation, HuggingFace Quantization guide, TheBloke's model repository for pre-quantized models.

---

## Quick Reference Card

Precision | Memory per 7B | Typical Quality | Best For
FP32 | 28GB | 100% | Training, gold reference
FP16 | 14GB | ~99.9% | Standard inference
INT8 | 7GB | ~99% | Memory-constrained GPU
INT4 (NF4) | 3.5GB | ~97% | Very limited hardware

**Loading Commands:**

```python
# 8-bit
model = AutoModel.from_pretrained(name, load_in_8bit=True, device_map="auto")

# 4-bit
model = AutoModel.from_pretrained(name, load_in_4bit=True, device_map="auto")

```

---

**Questions or stuck?** The bitsandbytes GitHub issues and HuggingFace forums are great resources for troubleshooting quantization issues.

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