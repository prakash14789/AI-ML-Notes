# 57. Text Pipeline Engineering 2 - Suman - 13 Mar 2026

# Text Pipeline Engineering  2- Advanced NLP Techniques

## In-Class Resources: [Click Here](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/5b142fde-9c1e-4be5-813f-09942a87e78b/kYgMktfFeHElcWVx.zip)

**Program:** Vishlesan i-Hub IIT Patna x Masai School — AIM (AI & Machine Learning)

**Prerequisites:** Basic understanding of neural networks, Python, PyTorch fundamentals

**Difficulty:** Advanced

---

## Session Overview

This session bridges theoretical NLP concepts with practical implementation using PyTorch Text. You'll understand the evolution from traditional RNNs to modern architectures (GRU/LSTM), master word embedding techniques (Word2Vec, GloVe), and implement efficient text preprocessing pipelines. The session also covers deployment strategies using Hugging Face and best practices for production NLP systems.

---

## Learning Objectives

By the end of this session, you will be able to:

1. **Differentiate** between GRU and LSTM architectures and their role in handling sequential dependencies
2. **Explain** Word2Vec (CBOW and Skip-Gram) and GloVe embedding generation mechanisms
3. **Implement** tokenization, vocabulary building, and padding using PyTorch Text library
4. **Optimize** batch processing using token bucketing and dynamic padding strategies
5. **Deploy** language models using Hugging Face platform with appropriate licensing considerations

---

## Part 1: Recurrent Neural Network Architectures

### The Sequential Data Challenge

**Problem with standard neural networks:**

```
Sentence: "The cat sat on the mat"

Standard feedforward network:
- Treats each word independently
- No memory of previous words
- Cannot capture "cat" relates to "sat"

Need: Architecture that maintains context across sequence

```

### RNN Evolution: From Vanilla to Gated Architectures

**Vanilla RNN problems:**

```
Vanishing gradients:
Long sequence → gradients shrink exponentially
Early words forgotten by end of sequence

Exploding gradients:
Gradients grow too large → unstable training
Network fails to converge

```

### LSTM: Long Short-Term Memory

**Architecture components:**

**Two separate memory states:**

```
1. Cell state (Ct): Long-term memory
   - Information highway through sequence
   - Selective updates via gates

2. Hidden state (ht): Short-term memory
   - Immediate context
   - What network outputs at each step

```

**Three gates controlling information flow:**

**1. Forget Gate:**

```
Purpose: Decide what to discard from cell state

Formula: ft = σ(Wf · [ht-1, xt] + bf)

Example:
Input: "The cat, which was very fluffy, sat on the mat"
At "sat": Forget gate might reduce emphasis on "fluffy" 
         (less relevant to action)

```

**2. Input Gate:**

```
Purpose: Decide what new information to add to cell state

Formula: 
it = σ(Wi · [ht-1, xt] + bi)
C̃t = tanh(WC · [ht-1, xt] + bC)

Example:
At "sat": Input gate adds information about the action
New cell state: Ct = ft ⊙ Ct-1 + it ⊙ C̃t

```

**3. Output Gate:**

```
Purpose: Decide what to output from cell state

Formula: 
ot = σ(Wo · [ht-1, xt] + bo)
ht = ot ⊙ tanh(Ct)

Example:
At "sat": Output gate produces hidden state for next step

```

**Visual flow:**

```
      ┌─────────────────────────────────┐
      │    Cell State (Long-term)       │
      │  Ct-1 ──→ [Gates] ──→ Ct        │
      └─────────────────────────────────┘
                    ↓
      ┌─────────────────────────────────┐
      │  Hidden State (Short-term)      │
      │  ht-1 ──→ [Combine] ──→ ht      │
      └─────────────────────────────────┘
                    ↓
                 Output

```

### GRU: Gated Recurrent Unit

**Key innovation: Simplification**

**Single hidden state (no separate cell state):**

```
LSTM: Cell state (Ct) + Hidden state (ht)
GRU:  Hidden state (ht) only

Advantage: Fewer parameters, faster training

```

**Two gates instead of three:**

**1. Reset Gate (rt):**

```
Purpose: Control how much past information to forget

Formula: rt = σ(Wr · [ht-1, xt])

When rt ≈ 0: Ignore past hidden state
When rt ≈ 1: Keep past information

Example:
"The weather was nice. It rained heavily."
At "It": Reset gate ≈ 0 (new topic, forget "nice")

```

**2. Update Gate (zt):**

```
Purpose: Control how much new information to add

Formula: zt = σ(Wz · [ht-1, xt])

Acts like combined forget + input gate in LSTM

Example:
At "rained": Update gate ≈ 1 (important new info)

```

**Hidden state update:**

```
Candidate hidden state:
h̃t = tanh(W · [rt ⊙ ht-1, xt])

Final hidden state:
ht = (1 - zt) ⊙ ht-1 + zt ⊙ h̃t
     ↑                    ↑
  Keep old          Add new

```

### GRU vs LSTM Comparison

Aspect | LSTM | GRU
States | 2 (cell + hidden) | 1 (hidden only)
Gates | 3 (forget, input, output) | 2 (reset, update)
Parameters | More (~4× weight matrices) | Fewer (~3× weight matrices)
Training Speed | Slower | Faster (30-40% speedup)
Memory Usage | Higher | Lower
Performance | Slightly better on very long sequences | Comparable on most tasks
Use Case | When accuracy is critical | When speed/efficiency matters

**Parameter calculation example:**

```
Input size: d = 100
Hidden size: h = 128

LSTM parameters:
4 gates × (d×h + h×h + h) = 4 × (100×128 + 128×128 + 128) 
                          = 4 × 29,312 = 117,248

GRU parameters:
3 gates × (d×h + h×h + h) = 3 × 29,312 = 87,936

GRU has 25% fewer parameters!

```

**When to use each:**

```
Use LSTM when:
✓ Very long sequences (100+ steps)
✓ Complex dependencies
✓ Maximum accuracy required
✓ Computational resources available

Use GRU when:
✓ Moderate sequence lengths (<100 steps)
✓ Training speed matters
✓ Limited computational resources
✓ Real-time applications

```

---

## Part 2: Word Embeddings - Capturing Semantic Meaning

### The Distributional Hypothesis

**Core principle:** "You shall know a word by the company it keeps" (J.R. Firth, 1957)

**Intuition:**

```
Sentence 1: "The cat sat on the mat"
Sentence 2: "The dog sat on the rug"

Words appearing in similar contexts likely have similar meanings:
- "cat" and "dog" (both animals, both can sit)
- "mat" and "rug" (both floor coverings)

```

### Word2Vec: Neural Word Embeddings

**Objective:** Learn dense vector representations that capture semantic relationships

**Two training architectures:**

### 1. Continuous Bag of Words (CBOW)

**Task:** Predict center word from context words

**Example:**

```
Sentence: "The quick brown fox jumps over the lazy dog"
Window size: 2 (2 words on each side)

Training sample:
Context: ["quick", "brown", "jumps", "over"]
Target:  "fox"

Model learns: What word likely appears with these neighbors?

```

**Architecture:**

```
Input Layer:
Context words: [w(t-2), w(t-1), w(t+1), w(t+2)]
↓
One-hot encode each word (vocab_size dimensional)

Hidden Layer:
Average/sum word vectors
Dimension: embedding_size (e.g., 300)

Output Layer:
Softmax over entire vocabulary
Predict: center word

Loss: Cross-entropy

```

**Forward pass example:**

```
Vocabulary: ["the", "quick", "brown", "fox", ...] (10,000 words)
Embedding size: 300

Context: ["quick", "brown", "jumps", "over"]

1. One-hot encode each:
   quick: [0, 1, 0, 0, ..., 0] (10,000 dimensions)
   brown: [0, 0, 1, 0, ..., 0]
   ...

2. Lookup embeddings (10,000 → 300):
   quick: [0.21, -0.15, 0.33, ..., 0.08]
   brown: [0.18, -0.12, 0.29, ..., 0.11]
   ...

3. Average context embeddings:
   context_vec = mean([quick_emb, brown_emb, jumps_emb, over_emb])
                = [0.19, -0.13, 0.31, ..., 0.09]

4. Softmax prediction:
   P(word | context) for all 10,000 words
   Highest probability: "fox" ✓

```

**Characteristics:**

```
✓ Fast training (simpler objective)
✓ Good for frequent words
✓ Smooths over individual context differences
✗ Less effective for rare words

```

### 2. Skip-Gram

**Task:** Predict context words from center word (inverse of CBOW)

**Example:**

```
Sentence: "The quick brown fox jumps over the lazy dog"
Window size: 2

Training sample:
Input:   "fox"
Targets: ["quick", "brown", "jumps", "over"]

Model learns: What words likely appear around "fox"?

```

**Architecture:**

```
Input: Center word (one-hot encoded)
Hidden: Embedding layer
Output: Multiple predictions (one per context position)

Loss: Sum of cross-entropies for all context positions

```

**Why Skip-Gram excels with rare words:**

```
CBOW example:
Word "vicuña" (rare South American animal) appears once:
Context: ["the", "rare", "lives", "in"]

One training sample for "vicuña"

Skip-Gram example:
Input: "vicuña"
Generates 4 training samples:
- "vicuña" → "the"
- "vicuña" → "rare"
- "vicuña" → "lives"
- "vicuña" → "in"

4× more training signal for rare word!

```

**CBOW vs Skip-Gram:**

Aspect | CBOW | Skip-Gram
Prediction | Context → Center | Center → Context
Training samples | 1 per window | window_size×2 per window
Speed | Faster | Slower (more samples)
Rare words | Weaker | Stronger
Common words | Better | Comparable
Use case | Large corpus, frequent words | Small corpus, rare words

**Famous vector arithmetic:**

```
king - man + woman ≈ queen

Mathematically:
vec("king") - vec("man") + vec("woman") ≈ vec("queen")

Why this works:
vec("king") - vec("man") ≈ vec("royalty") (gender-neutral)
vec("royalty") + vec("woman") ≈ vec("queen")

Other examples:
Paris - France + Germany ≈ Berlin
walked - walking + swimming ≈ swam

```

### GloVe: Global Vectors for Word Representation

**Key innovation:** Use global corpus statistics, not just local context windows

**Approach:**

**Step 1: Co-occurrence Matrix**

```
Count word pairs appearing together within window

Example corpus (window=1):
"I like NLP"
"I like ML"
"NLP is fun"

Co-occurrence matrix:
        I    like  NLP   ML   is   fun
I       0    2     0     0    0    0
like    2    0     1     1    0    0
NLP     0    1     0     0    1    1
ML      0    1     0     0    0    0
is      0    0     1     0    0    1
fun     0    0     1     0    1    0

Example: "like" and "NLP" co-occur 1 time

```

**Step 2: Logarithmic Bilinear Regression**

```
Objective: Learn vectors such that dot product approximates log co-occurrence

For word i and context j:
wi · wj + bi + bj = log(Xij)

Where:
wi, wj = word vectors (what we learn)
bi, bj = bias terms
Xij = co-occurrence count

```

**Loss function:**

```
J = Σ f(Xij) × (wi · wj + bi + bj - log(Xij))²

Weighting function f(Xij):
- Caps influence of very frequent pairs
- Reduces noise from rare pairs

f(Xij) = min(1, (Xij / xmax)^α)

```

**Why GloVe works:**

```
Traditional approaches:
- Local context (Word2Vec): Ignores global statistics
- Global counts (LSA): Doesn't capture complex patterns

GloVe:
✓ Uses global statistics (co-occurrence matrix)
✓ Captures complex relationships (via regression)
✓ Explicit model of word similarity (dot product)

```

**Word2Vec vs GloVe:**

Aspect | Word2Vec | GloVe
Method | Predictive (neural network) | Count-based + factorization
Training | Local context windows | Global co-occurrence matrix
Speed | Faster on large corpus | Faster on small corpus
Memory | Lower | Higher (stores matrix)
Performance | Comparable | Comparable
Interpretability | Lower | Higher (explicit statistics)

---

## Part 3: PyTorch Text Library - Practical Pipeline

### Setting Up the Text Processing Pipeline

**Problem:** Raw text → Model-ready tensors

**Steps:**

1. Tokenization (text → tokens)
2. Vocabulary building (tokens → indices)
3. Encoding (sentences → integer sequences)
4. Padding (variable length → fixed length)
5. Batching (individual samples → batches)

### 1. Tokenization with `get_tokenizer`

**Purpose:** Break text into tokens (words, subwords, or characters)

**Implementation:**

```python
from torchtext.data.utils import get_tokenizer

# Basic English tokenizer
tokenizer = get_tokenizer('basic_english')

sentence = "Hello, World! This is NLP."
tokens = tokenizer(sentence)
print(tokens)
# Output: ['hello', ',', 'world', '!', 'this', 'is', 'nlp', '.']

```

**Tokenizer options:**

```python
# Basic English (default)
tokenizer = get_tokenizer('basic_english')
# - Lowercases
# - Splits on whitespace and punctuation

# SpaCy (more sophisticated)
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
# - Linguistic rules
# - Better handling of contractions ("don't" → "do", "n't")

# Moses (machine translation standard)
tokenizer = get_tokenizer('moses')
# - Multilingual support
# - Handles punctuation better

```

**Token vs Word:**

```
Sentence: "Don't tokenize incorrectly!"

Basic tokens: ["don't", "tokenize", "incorrectly", "!"]
SpaCy tokens: ["do", "n't", "tokenize", "incorrectly", "!"]
Subword tokens: ["don", "'", "t", "token", "ize", "in", "correct", "ly", "!"]

Subword tokenization (BPE, WordPiece):
- Handles rare words better
- Smaller vocabulary
- Used in modern transformers (BERT, GPT)

```

### 2. Vocabulary Building with `build_vocab_from_iterator`

**Purpose:** Map tokens to unique integer indices

**Implementation:**

```python
from torchtext.vocab import build_vocab_from_iterator

# Sample data
sentences = [
    "I love NLP",
    "NLP is amazing",
    "I love ML too"
]

# Tokenize all sentences
def yield_tokens(sentences):
    for sentence in sentences:
        yield tokenizer(sentence)

# Build vocabulary
vocab = build_vocab_from_iterator(
    yield_tokens(sentences),
    specials=['<pad>', '<unk>'],  # Special tokens
    special_first=True             # Put specials at beginning
)

# Set default index for unknown tokens
vocab.set_default_index(vocab['<unk>'])

print(f"Vocabulary size: {len(vocab)}")
print(f"vocab['nlp']: {vocab['nlp']}")
print(f"vocab['<pad>']: {vocab['<pad>']}")
print(f"vocab['<unk>']: {vocab['<unk>']}")

```

**Output:**

```
Vocabulary size: 10
vocab['nlp']: 2
vocab['<pad>']: 0
vocab['<unk>']: 1

Full vocabulary:
<pad>     → 0  (padding token)
<unk>     → 1  (unknown token)
nlp       → 2
i         → 3
love      → 4
is        → 5
amazing   → 6
ml        → 7
too       → 8
,         → 9

```

**Special tokens:**

```
<pad> (index 0):
- Padding token for sequences of different lengths
- Typically ignored in loss calculation
- Always index 0 for convenience

<unk> (index 1):
- Unknown token for out-of-vocabulary words
- Handles words not seen during training
- Default index when token not found

<sos>, <eos> (optional):
- Start/end of sequence markers
- Used in seq2seq models (translation, summarization)

```

### 3. Encoding: Tokens to Indices

**Convert tokenized sentences to integer sequences:**

```python
sentence = "I love NLP"
tokens = tokenizer(sentence)  # ['i', 'love', 'nlp']
indices = vocab(tokens)       # [3, 4, 2]

# Handle unknown words
sentence2 = "I hate NLP"  # "hate" not in vocabulary
tokens2 = tokenizer(sentence2)  # ['i', 'hate', 'nlp']
indices2 = vocab(tokens2)       # [3, 1, 2]  (1 = <unk>)

```

### 4. Padding Strategies

**Problem:** Neural networks require fixed-length inputs, but sentences vary in length

**Naive Padding: Global Maximum**

```python
sentences = [
    "I love NLP",           # 3 tokens
    "NLP is amazing",       # 3 tokens  
    "This is a very long sentence about NLP"  # 8 tokens
]

# Encode all sentences
encoded = [vocab(tokenizer(s)) for s in sentences]
# [[3, 4, 2], [2, 5, 6], [10, 5, 11, 12, 13, 14, 15, 2]]

# Global max length
max_len = max(len(seq) for seq in encoded)  # 8

# Pad to max length
from torch.nn.utils.rnn import pad_sequence
import torch

padded = [seq + [vocab['<pad>']] * (max_len - len(seq)) for seq in encoded]
# [[3, 4, 2, 0, 0, 0, 0, 0],
#  [2, 5, 6, 0, 0, 0, 0, 0],
#  [10, 5, 11, 12, 13, 14, 15, 2]]

```

**Problem with naive padding:**

```
Total tokens: 3 + 3 + 8 = 14
After padding: 8 + 8 + 8 = 24
Wasted tokens: 24 - 14 = 10 (42% waste!)

Computational cost:
- Network processes all 24 tokens
- But only 14 contain real information
- 42% wasted computation!

```

**Batch-wise Padding: Local Maximum**

**Idea:** Pad each batch only to its maximum length, not global maximum

```python
# Batch 1: Short sentences
batch1 = [
    [3, 4, 2],        # "I love NLP" (3 tokens)
    [2, 5, 6],        # "NLP is amazing" (3 tokens)
]
max_len1 = 3
# Padded: [[3, 4, 2], [2, 5, 6]]
# No padding needed!

# Batch 2: Long sentence
batch2 = [
    [10, 5, 11, 12, 13, 14, 15, 2]  # 8 tokens
]
max_len2 = 8
# Padded: [[10, 5, 11, 12, 13, 14, 15, 2]]
# No padding needed!

Total tokens: 3 + 3 + 8 = 14 (same as input!)
Waste: 0%

```

**Implementation:**

```python
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
    """
    Custom collate function for DataLoader
    Pads sequences to max length in THIS batch only
    """
    # batch: list of (sequence, label) tuples
    sequences, labels = zip(*batch)
    
    # Convert to tensors
    sequences = [torch.tensor(seq) for seq in sequences]
    
    # Pad to max in THIS batch
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    labels = torch.tensor(labels)
    
    return padded, labels

# Use in DataLoader
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collate_batch  # Use custom collate
)

```

### 5. Token Bucketing for Optimal Batching

**Problem:** Even batch-wise padding wastes tokens if sequences vary widely within batch

**Example:**

```
Batch (random sampling):
Sequence 1: 5 tokens
Sequence 2: 50 tokens  ← Forces all to 50
Sequence 3: 8 tokens
Sequence 4: 45 tokens

Padded lengths: [50, 50, 50, 50] = 200 tokens
Actual tokens:  [5, 50, 8, 45] = 108 tokens
Waste: 92 tokens (46%!)

```

**Solution: Group similar-length sequences**

**Token bucketing algorithm:**

```python
# Step 1: Sort dataset by sequence length
sorted_dataset = sorted(dataset, key=lambda x: len(x[0]))

# Step 2: Create buckets of similar lengths
bucket_boundaries = [10, 20, 30, 40, 50]  # Max lengths per bucket

buckets = {
    '0-10': [],
    '10-20': [],
    '20-30': [],
    '30-40': [],
    '40-50': []
}

for seq, label in sorted_dataset:
    seq_len = len(seq)
    if seq_len <= 10:
        buckets['0-10'].append((seq, label))
    elif seq_len <= 20:
        buckets['10-20'].append((seq, label))
    # ... etc

# Step 3: Sample batches from buckets
# Each batch contains sequences of similar length

```

**Efficiency improvement:**

```
Random batching:
Batch 1: [5, 50, 8, 45] → pad to 50 → 200 tokens (46% waste)
Batch 2: [12, 48, 15, 42] → pad to 48 → 192 tokens (43% waste)
Average waste: ~43%

Bucketed batching:
Batch 1 (bucket 0-10): [5, 8, 7, 9] → pad to 9 → 36 tokens (19% waste)
Batch 2 (bucket 40-50): [45, 48, 42, 47] → pad to 48 → 192 tokens (6% waste)
Average waste: ~10%

Improvement: 43% → 10% waste (4.3× more efficient!)

```

**Implementation with PyTorch sampler:**

```python
from torch.utils.data import Sampler

class BucketSampler(Sampler):
    def __init__(self, data_source, batch_size, sort_key):
        self.data_source = data_source
        self.batch_size = batch_size
        self.sort_key = sort_key
        
    def __iter__(self):
        # Sort indices by sequence length
        indices = sorted(
            range(len(self.data_source)),
            key=lambda i: self.sort_key(self.data_source[i])
        )
        
        # Group into batches
        batches = [
            indices[i:i+self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        
        # Shuffle batches (not samples within batches)
        random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

# Usage
sampler = BucketSampler(
    dataset,
    batch_size=32,
    sort_key=lambda x: len(x[0])  # Sort by sequence length
)

dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=collate_batch
)

```

---

## Part 4: Hugging Face Platform and LLM Deployment

### Hugging Face Ecosystem

**Platform components:**

```
1. Model Hub:
   - 100,000+ pre-trained models
   - Transformers, diffusion, RL models
   - Community contributions

2. Datasets Hub:
   - 10,000+ datasets
   - Pre-processed, ready to use

3. Spaces:
   - Host ML demos (Gradio, Streamlit)
   - Share applications

4. Libraries:
   - transformers: NLP models
   - diffusers: Image generation
   - datasets: Data loading

```

### Using Hugging Face Models

**Basic workflow:**

**1. Authentication:**

```python
from huggingface_hub import login

# Get token from: https://huggingface.co/settings/tokens
login(token="hf_your_token_here")

```

**2. Model loading:**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text
inputs = tokenizer("Hello, I am", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))

```

**3. Model cards:**

```
Each model has documentation:
- Model description
- Training data
- Performance metrics
- Limitations
- License information
- Citation

Always check before using in production!

```

### Model Size Considerations

**Parameter scale:**

```
Small models (1B-3B parameters):
✓ Run on consumer GPUs (RTX 3090, 24GB)
✓ Fast inference
✓ Good for experimentation
✗ Lower capability

Medium models (7B-13B parameters):
✓ Balance of performance and efficiency
✓ Run on high-end GPUs (A100, 40GB)
~ Moderate inference speed

Large models (30B-70B parameters):
✓ State-of-the-art performance
✗ Require multi-GPU or cloud
✗ Expensive inference
✗ Can crash systems if improperly loaded

Recommendation: Start with 3B-7B for learning

```

### Deployment Strategies

**API-based (OpenAI, Anthropic, Cohere):**

```
Advantages:
✓ No infrastructure management
✓ Always latest models
✓ Pay-per-use pricing
✓ Scales automatically

Disadvantages:
✗ Data leaves your system (privacy concern)
✗ Ongoing costs
✗ Dependent on third-party uptime
✗ Rate limits

```

**Self-hosted (Local deployment):**

```
Advantages:
✓ Full data privacy
✓ No per-request costs
✓ Customization freedom
✓ No rate limits

Disadvantages:
✗ Significant infrastructure costs
✗ Maintenance burden
✗ Scaling challenges
✗ Requires ML engineering expertise

```

**Local hosting tools:**

**1. vLLM:**

```python
# Optimized inference server
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(
    ["Hello, my name is"],
    sampling_params
)

```

**2. Ollama:**

```bash
# Simple local deployment
ollama pull llama2:7b
ollama run llama2:7b "Explain quantum computing"

```

**3. Triton Inference Server:**

```
NVIDIA's production-grade server
- Multi-model serving
- Dynamic batching
- Model versioning
- Monitoring integration

```

### Licensing Considerations

**Common licenses:**

```
Apache 2.0:
✓ Commercial use allowed
✓ Modification allowed
✓ Distribution allowed
✓ No warranty
Example: Llama 2

MIT License:
✓ Very permissive
✓ Commercial use
✓ Minimal restrictions

GPL/AGPL:
⚠ Copyleft licenses
⚠ Derivative works must be open-source
⚠ Check carefully for commercial use

Custom Research Licenses:
⚠ Often non-commercial only
⚠ Read terms carefully
Example: Original Llama 1

```

**Enterprise checklist:**

```
Before deploying:
□ Verify license permits commercial use
□ Check data usage restrictions
□ Review model card for limitations
□ Test for bias in your domain
□ Implement monitoring
□ Plan for model updates

```

---

## Part 5: Best Practices and Practical Guidance

### Version Management

**TorchText versioning:**

```
Problem: Frequent breaking changes

Solution:
# Pin exact versions
pip install torchtext==0.15.2 torch==2.0.1

# Or use requirements.txt
torchtext==0.15.2
torch==2.0.1
transformers==4.30.0

Always test after upgrades!

```

### Efficient Pipeline Design

**Memory optimization:**

```python
# Bad: Load all data into memory
all_data = [process(item) for item in huge_dataset]  # OOM!

# Good: Use iterators
def data_iterator(dataset):
    for item in dataset:
        yield process(item)

# Better: Use DataLoader with num_workers
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Parallel data loading
    pin_memory=True  # Faster GPU transfer
)

```

### Padding best practices:**

```
1. Use batch-wise padding (not global)
2. Implement bucket sampling for variable-length data
3. Mask padded positions in loss calculation
4. Monitor padding waste percentage

# Mask padding in loss
loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # 0 = <pad>

```

### Production deployment:**

```
Checklist:
□ Model quantization (INT8, FP16) for faster inference
□ Batch requests when possible
□ Implement caching for common queries
□ Monitor latency and throughput
□ Set up logging and error handling
□ Plan for model updates
□ A/B test new models

```

---

## Key Takeaways

### Core Concepts

1. 
**RNN Architectures:**

LSTM: 3 gates, separate cell state, more parameters
GRU: 2 gates, single hidden state, faster training
Both solve vanishing gradients, choose based on task complexity

2. 
**Word Embeddings:**

Word2Vec CBOW: Context → center word (fast, good for frequent words)
Word2Vec Skip-Gram: Center → context (better for rare words)
GloVe: Global co-occurrence statistics (explicit modeling)
All capture semantic relationships through vector arithmetic

3. 
**PyTorch Text Pipeline:**

Tokenization: Text → tokens
Vocabulary: Tokens → indices (with , )
Padding: Batch-wise > global maximum
Bucketing: Group similar lengths (4× more efficient)

4. 
**Hugging Face & Deployment:**

Model Hub: 100K+ pre-trained models
Start with 3B-7B parameter models
API vs self-hosted: Privacy vs convenience trade-off
Always check licenses for commercial use

### Mathematical Insights

**GRU update:**

```
ht = (1 - zt) ⊙ ht-1 + zt ⊙ h̃t

```

**Word2Vec objective (Skip-Gram):**

```
maximize: Σ log P(context | center)

```

**Padding waste:**

```
Waste % = (total_padded - total_real) / total_padded × 100
Random batching: ~43% waste
Bucket batching: ~10% waste

```

### Decision Guide

**Choose LSTM when:**

- Very long sequences (100+ steps)
- Maximum accuracy required
- Computational resources available

**Choose GRU when:**

- Moderate sequences (<100 steps)
- Speed/efficiency matters
- Limited resources

**Choose CBOW when:**

- Large corpus
- Focus on frequent words
- Training speed matters

**Choose Skip-Gram when:**

- Small corpus
- Rare words important
- Have computational resources

**Choose API deployment when:**

- Rapid prototyping
- Variable workload
- No ML infrastructure

**Choose self-hosted when:**

- Data privacy critical
- High volume (cost-effective at scale)
- Need customization

---

## Looking Ahead

**Next session topics:**

- Transformer architecture deep dive
- Attention mechanisms
- Fine-tuning strategies
- Advanced prompt engineering
- Production monitoring and maintenance

---

**End of Lecture Notes**

**Vishlesan i-Hub IIT Patna × Masai School**

*From tokens to transformers, from embeddings to deployment*

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