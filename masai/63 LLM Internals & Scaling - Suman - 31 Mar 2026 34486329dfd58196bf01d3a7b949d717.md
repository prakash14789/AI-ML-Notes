# 63. LLM Internals & Scaling - Suman - 31 Mar 2026

# Lecture Notes: LLM Internals & Scaling

## PPT File: [Click Here](https://drive.google.com/file/d/1LRuMA7yBFjUZ9UNH-DE-pouPuqqRsm5p/view?usp=sharing)

## Session Overview

This session explores the internal architecture of large language models, focusing on the fundamental design choices that determine their capabilities and computational costs. We'll examine GPT-2's architecture in detail, understand attention mechanisms, derive scaling laws, and use interpretability techniques to see what these models actually learn.

## Learning Objectives

By the end of this session, you will be able to:

1. Explain the depth vs. width trade-offs in transformer architectures
2. Describe how multi-head attention enables parallel information processing
3. Apply scaling laws to predict model performance before training
4. Use interpretability probes to understand what different layers learn
5. Make informed architecture decisions for resource-constrained scenarios

---

## 1. Transformer Architecture Fundamentals

### 1.1 The Stack: Depth in Transformers

When we say GPT-2 has "48 layers," what does that actually mean?

**Layer Structure:**

```
Input Embeddings
    ↓
[Layer 1: Attention + Feed-Forward]
    ↓
[Layer 2: Attention + Feed-Forward]
    ↓
    ...
    ↓
[Layer 48: Attention + Feed-Forward]
    ↓
Output Predictions

```

Each layer performs two main operations:

1. **Multi-head self-attention**: Looks at relationships between all tokens
2. **Feed-forward network**: Processes each token independently

**Why Stack Layers?**

Think of layers as stages of understanding:

- **Shallow layers (1-12)**: Learn surface patterns—syntax, basic grammar, word associations
- **Middle layers (13-36)**: Capture semantic relationships, factual knowledge, entity recognition
- **Deep layers (37-48)**: Perform reasoning, inference, task-specific computations

**Mathematical Flow:**

```
h₀ = Embedding(tokens)
h₁ = Layer₁(h₀) = FFN(Attention(h₀))
h₂ = Layer₂(h₁) = FFN(Attention(h₁))
...
h₄₈ = Layer₄₈(h₄₇)
Output = Softmax(h₄₈ · W_output)

```

Each layer transforms the representation, adding more abstract understanding.

### 1.2 Width: The Hidden Dimension

GPT-2 comes in multiple sizes:

- **Small**: d_model = 768
- **Medium**: d_model = 1024
- **Large**: d_model = 1280
- **XL**: d_model = 1600

**What does d_model mean?**

Every token is represented as a vector of length d_model. This is the "width" of the network—how many features can represent each token at each layer.

**Width vs. Depth Trade-off:**

For a fixed parameter budget:

- **Wider, shallower**: Can represent more features simultaneously, but limited reasoning depth
- **Deeper, narrower**: More sequential reasoning steps, but fewer features per step

**Example:**

- Model A: 24 layers × 1536 width = ~37M parameters per layer
- Model B: 48 layers × 768 width = ~37M parameters per layer

Both have similar total parameters, but very different capabilities. Research shows **deeper models generally perform better** for complex reasoning tasks.

---

## 2. Multi-Head Attention: Looking from Multiple Angles

### 2.1 What Are Attention Heads?

Multi-head attention is like having multiple "experts" looking at the same text from different perspectives.

**GPT-2 Configuration:**

- Small: **12 heads**
- Medium: **16 heads**
- Large: **20 heads**
- XL: **25 heads**

**Why Multiple Heads?**

Each head can specialize in different patterns:

- **Head 1**: Might focus on syntactic dependencies (subject-verb agreement)
- **Head 2**: Might track coreference (pronouns → entities)
- **Head 3**: Might identify semantic relationships
- **Head 4**: Might handle positional patterns

**Mathematical Formulation:**

For input X with dimension d_model = 768 and 12 heads:

```
1. Split d_model across heads:
   d_head = d_model / num_heads = 768 / 12 = 64

2. Each head computes attention independently:
   head_i = Attention(X·W_Q^i, X·W_K^i, X·W_V^i)
   where W_Q^i, W_K^i, W_V^i project to d_head dimensions

3. Concatenate all heads:
   MultiHead(X) = Concat(head₁, head₂, ..., head₁₂) · W_O

4. Result has original dimension d_model = 768

```

**Key Insight:** All heads process in parallel, enabling the model to attend to multiple relationships simultaneously.

### 2.2 Attention Patterns in Practice

When we visualize attention in trained models, we discover heads specialize:

**Positional Heads**: Attend to adjacent tokens (capturing local context)

```
[The] [cat] [sat] [on] [the] [mat]
  ↓    ↓    ↓    ↓    ↓    ↓
Strong attention to immediate neighbors

```

**Syntactic Heads**: Connect words by grammatical role

```
[The] [cat] [that] [I] [saw] [ran]
       ↓                ↓      ↓
   Subject ←←←←←←←←←← Verb

```

**Semantic Heads**: Link related concepts

```
[The] [doctor] [prescribed] [medicine]
       ↓                        ↓
    Entity ←←←←←←←←←←←← Related concept

```

---

## 3. Scaling Laws: The Mathematics of Bigger Models

### 3.1 The Empirical Discovery

In 2020, researchers at OpenAI discovered that model performance follows predictable mathematical laws based on three factors:

1. **N**: Number of parameters
2. **D**: Size of dataset
3. **C**: Amount of compute (FLOPs)

**The Power Law:**

```
Loss ∝ (N/N₀)^(-α_N) · (D/D₀)^(-α_D) · (C/C₀)^(-α_C)

Where:
- α_N ≈ 0.076 (parameters exponent)
- α_D ≈ 0.095 (data exponent)  
- α_C ≈ 0.050 (compute exponent)

```

**What This Means:**

If you **double** the parameters (N), loss improves by approximately:

```
Improvement = 2^(-0.076) ≈ 0.95 (5% reduction in loss)

```

If you **10x** the parameters:

```
Improvement = 10^(-0.076) ≈ 0.83 (17% reduction in loss)

```

### 3.2 Optimal Allocation

For a fixed compute budget C, how should you allocate resources?

**Chinchilla Scaling Laws (2022 Update):**

Optimal ratio: **Parameters : Training Tokens ≈ 1 : 20**

Example:

- **1B parameter model** → Train on **20B tokens**
- **10B parameter model** → Train on **200B tokens**
- **100B parameter model** → Train on **2T tokens**

**Why This Matters:**

Many models were undertrained! GPT-3 (175B parameters) was trained on 300B tokens—far below the optimal 3.5T tokens.

**Chinchilla (70B parameters, 1.4T tokens)** outperformed GPT-3 (175B parameters, 300B tokens) because it followed the optimal ratio.

### 3.3 Cost Predictions

Using scaling laws, we can predict training costs:

**Compute Required:**

```
C ≈ 6 · N · D

Where:
N = number of parameters
D = number of training tokens
6 = approximate FLOPs per parameter per token

```

**Example: GPT-3**

```
N = 175B parameters
D = 300B tokens
C = 6 · 175B · 300B = 3.15 × 10²³ FLOPs

At $1 per 10¹⁵ FLOPs (typical GPU pricing):
Cost ≈ $315,000 × efficiency factor ≈ $5-10M actual cost

```

---

## 4. Interpretability: Understanding What Models Learn

### 4.1 Layer-wise Specialization

Using probing techniques, we can see what information each layer captures:

**Methodology:**
Train a simple classifier (probe) on layer activations to predict linguistic properties:

```python
# Pseudo-code
for layer in [1, 2, ..., 48]:
    activations = model.get_layer_activations(layer, text)
    probe = train_classifier(activations, target_labels)
    accuracy[layer] = probe.evaluate()

```

**Findings in GPT-2:**

**Layers 1-8 (Surface)**:

- Part-of-speech tagging: 95% accuracy
- Sentence boundary detection: 98% accuracy
- Simple word associations: 90% accuracy

**Layers 9-24 (Syntactic)**:

- Dependency parsing: 85% accuracy
- Coreference resolution: 78% accuracy
- Named entity recognition: 92% accuracy

**Layers 25-40 (Semantic)**:

- Sentiment analysis: 88% accuracy
- Factual recall: 82% accuracy
- Semantic role labeling: 80% accuracy

**Layers 41-48 (Task-specific)**:

- Question answering: Peak performance
- Text completion: Highest coherence
- Reasoning tasks: Best accuracy

### 4.2 Attention Head Specialization

Individual attention heads develop specific behaviors:

**Positional Heads** (Layers 1-10):

- Attend primarily to previous token
- Enable local context understanding
- Essential for grammar

**Syntactic Heads** (Layers 10-25):

- Connect subjects to verbs
- Track noun phrase boundaries
- Handle long-distance dependencies

**Semantic Heads** (Layers 25-40):

- Link entities to descriptions
- Connect questions to relevant context
- Track topical coherence

**Mixed Heads** (Layers 40-48):

- Combine multiple patterns
- Task-dependent behavior
- Hardest to interpret

### 4.3 Neuron-Level Interpretability

Some neurons in FFN layers develop interpretable functions:

**Example Findings:**

- **Neuron 47:892** (Layer 47): Activates for bridge/connection concepts
- **Neuron 23:145** (Layer 23): Tracks sentiment polarity
- **Neuron 35:678** (Layer 35): Responds to numeric patterns

**Technique:**

```python
# Find maximally activating examples
for text in large_corpus:
    activation = model.forward(text)[layer][neuron]
    if activation > threshold:
        save(text, activation)

# Analyze patterns in top activations

```

---

## 5. Practical Architecture Decisions

### 5.1 Resource-Constrained Scenarios

**Scenario**: You have budget for 1B parameters. How to configure?

**Option A: Shallow and Wide**

- 12 layers × 2048 width × 16 heads
- Fast inference
- Good for: Pattern matching, retrieval

**Option B: Deep and Narrow**

- 48 layers × 1024 width × 8 heads
- Better reasoning
- Good for: Complex tasks, multi-step reasoning

**Option C: Balanced**

- 24 layers × 1536 width × 12 heads
- Middle ground
- Good for: General purpose

**Research Consensus:** Slightly favor depth over width for most tasks.

### 5.2 Training vs. Inference Trade-offs

**Training Cost ∝ N · D** (parameters × tokens)
**Inference Cost ∝ N · L** (parameters × sequence length)

For deployment:

- **Depth** increases latency (sequential processing)
- **Width** increases memory (larger activations)
- **Heads** increase compute (parallel attention)

**Optimization Strategies:**

1. **Knowledge Distillation**: Train large (teacher) → compress to small (student)
2. **Quantization**: Reduce precision (FP32 → INT8)
3. **Pruning**: Remove less important connections

---

## 6. Key Takeaways

### Architecture Principles

1. **Depth enables sequential reasoning**: More layers = more computation steps
2. **Width enables representation capacity**: Larger d_model = more features
3. **Multiple heads enable specialization**: Different patterns learned in parallel

### Scaling Laws

1. **Performance is predictable**: Power laws govern loss vs. resources
2. **Optimal allocation exists**: ~20 tokens per parameter is ideal
3. **Compute-optimal ≠ Parameter-optimal**: Chinchilla beats GPT-3 with fewer parameters

### Interpretability

1. **Layers specialize hierarchically**: Surface → Syntax → Semantics → Reasoning
2. **Heads develop specific functions**: Positional, syntactic, semantic roles
3. **Individual neurons can be meaningful**: Some track interpretable concepts

### Practical Implications

1. **Bigger ≠ Always Better**: Must match architecture to task and budget
2. **Training regimen matters**: Number of tokens as important as parameters
3. **Interpretability is limited**: We understand patterns, not all mechanisms

---

## Real-World Application

When designing your own language model:

1. **Estimate budget**: $C compute available
2. **Apply scaling laws**: Determine optimal N (parameters) and D (tokens)
3. **Choose architecture**: Favor depth for reasoning, width for representation
4. **Configure heads**: ~8-16 heads is typical, more doesn't always help
5. **Plan training**: Ensure sufficient tokens per parameter (~20:1 ratio)

---

## Further Reading

**Foundational Papers:**

- Attention Is All You Need (Vaswani et al., 2017)
- Language Models are Few-Shot Learners (Brown et al., 2020)
- Scaling Laws for Neural Language Models (Kaplan et al., 2020)
- Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)

**Interpretability:**

- A Mathematical Framework for Transformer Circuits (Elhage et al., 2021)
- Transformer Feed-Forward Layers Are Key-Value Memories (Geva et al., 2021)

---

## Summary

Large language models are:

- **Architecturally**: Deep stacks of attention + FFN layers, with careful depth/width/head balance
- **Mathematically**: Governed by scaling laws that predict performance from compute
- **Behaviorally**: Develop hierarchical representations from surface to semantic to reasoning
- **Practically**: Require massive compute but follow predictable cost-performance curves

Understanding these internals enables informed decisions about model design, training regimens, and deployment strategies—critical skills as language models become central to modern AI systems.

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