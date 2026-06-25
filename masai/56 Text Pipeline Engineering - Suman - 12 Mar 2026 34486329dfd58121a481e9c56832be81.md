# 56. Text Pipeline Engineering - Suman - 12 Mar 2026

# Text Pipeline Engineering

## In-Class Resources: [Click Here](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/5b142fde-9c1e-4be5-813f-09942a87e78b/kYgMktfFeHElcWVx.zip)

**Program:** Vishlesan i-Hub IIT Patna x Masai School — AIM (AI & Machine Learning)

**Duration:** 3 hours (180 minutes)

**Topics:** RNN, LSTM, Sentiment Analysis, Sequence Processing

---

## Session Structure

- **Part 1:** Sequence Problems & RNN Fundamentals (30 min)
- **Part 2:** RNN Architecture & Implementation (30 min)
- **Part 3:** Vanishing Gradients & LSTM Solution (35 min)
- **Part 4:** LSTM Architecture Deep Dive (35 min)
- **Part 5:** Sentiment Analysis with LSTM (30 min)
- **Part 6:** Advanced Topics & Production (20 min)

---

# PART 1: SEQUENCE PROBLEMS & RNN FUNDAMENTALS (30 minutes)

## 1.1 Why Sequences Are Different

### The Order Matters Problem

**Example 1: Language**

```
"dog bites man" ≠ "man bites dog"
Same words, completely different meaning!

```

**Example 2: Stock Prices**

```
Prices: [100, 105, 110, 115, 120]
→ Upward trend, BUY

Prices: [120, 115, 110, 105, 100]
→ Downward trend, SELL

Same numbers, different ORDER → Different decision

```

**Traditional ML fails:**

```python
# Bag of words
sentence1 = "dog bites man"
sentence2 = "man bites dog"

# Convert to counts
bag1 = {'dog': 1, 'bites': 1, 'man': 1}
bag2 = {'man': 1, 'bites': 1, 'dog': 1}

# Identical! Can't distinguish!

```

---

## 1.2 Types of Sequence Problems

### 1. One-to-Many (Image Captioning)

```
Input: Single image
Output: Sequence of words

[Cat Image] → "A cat sitting on a table"

```

### 2. Many-to-One (Sentiment Analysis)

```
Input: Sequence of words
Output: Single classification

"This movie was great!" → POSITIVE

```

### 3. Many-to-Many (Same Length - POS Tagging)

```
Input:  ["The", "cat", "sat"]
Output: ["DET", "NOUN", "VERB"]

Each input word → one output tag

```

### 4. Many-to-Many (Different Length - Translation)

```
Input:  "Bonjour" (French, 1 word)
Output: "Good morning" (English, 2 words)

Input length ≠ Output length

```

---

## 1.3 RNN Core Concept

### Feedforward Network (No Memory)

```
Input: [word1, word2, word3]
       ↓
Hidden Layer (processes all together)
       ↓
Output: prediction

Problem: No sense of ORDER

```

### Recurrent Network (With Memory)

```
Step 1:
Input: word1 → Hidden State → Memory1
                    ↓
Step 2:
Input: word2 + Memory1 → Hidden State → Memory2
                              ↓
Step 3:
Input: word3 + Memory2 → Hidden State → Memory3 → Output

```

**Key insight:** Memory carries context forward!

---

## 1.4 The Unrolling Concept

### Single RNN Cell (Recurrent)

```
     ┌─────┐
  ┌─→│ RNN │─┐
  │  └─────┘ │
  │          │
  └──────────┘
     Hidden State (loops back)

```

### Unrolled Through Time

```
Time: t=0      t=1       t=2       t=3
      ┌───┐    ┌───┐     ┌───┐     ┌───┐
x₀ →  │RNN│ →  │RNN│  →  │RNN│  →  │RNN│
      └───┘    └───┘     └───┘     └───┘
        ↓        ↓         ↓         ↓
       h₀       h₁        h₂        h₃

h₁ = f(x₁, h₀)  # Current input + Previous hidden state
h₂ = f(x₂, h₁)
h₃ = f(x₃, h₂)

```

**Understanding:** Same RNN cell, but applied at each time step with different inputs!

---

# PART 2: RNN ARCHITECTURE & IMPLEMENTATION (30 minutes)

## 2.1 Mathematical Formulation

### Forward Pass Equations

```
At time step t:

h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)
y_t = W_hy × h_t + b_y

Where:
- h_t: Hidden state at time t
- x_t: Input at time t  
- y_t: Output at time t
- W_hh: Hidden-to-hidden weight matrix
- W_xh: Input-to-hidden weight matrix
- W_hy: Hidden-to-output weight matrix
- b_h, b_y: Bias terms

```

### Step-by-Step Example

```
Given:
- Vocabulary: ['the', 'cat', 'sat']
- Embedding dimension: 3
- Hidden dimension: 2

Sentence: "the cat sat"

Step 1: Process "the"
x₀ = [0.1, 0.2, 0.3]  # Word embedding
h₋₁ = [0, 0]          # Initial hidden state

h₀ = tanh([0, 0] × W_hh + [0.1, 0.2, 0.3] × W_xh + b_h)
   = tanh([0.15, 0.25])
   = [0.149, 0.245]

Step 2: Process "cat"
x₁ = [0.4, 0.5, 0.6]
h₀ = [0.149, 0.245]  # From previous step

h₁ = tanh(h₀ × W_hh + x₁ × W_xh + b_h)
   = tanh([0.52, 0.68])
   = [0.478, 0.593]

Step 3: Process "sat"
x₂ = [0.7, 0.8, 0.9]
h₁ = [0.478, 0.593]

h₂ = tanh(h₁ × W_hh + x₂ × W_xh + b_h)
   = [0.712, 0.831]

Final output: y = W_hy × h₂ + b_y

```

---

## 2.2 PyTorch Implementation

### Simple RNN from Scratch

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Weight matrices
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        # Concatenate input and hidden state
        combined = torch.cat((input, hidden), 1)
        
        # Compute new hidden state
        hidden = torch.tanh(self.i2h(combined))
        
        # Compute output
        output = self.h2o(hidden)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

# Usage
model = SimpleRNN(input_size=100, hidden_size=128, output_size=2)

# Process sequence
hidden = model.init_hidden(batch_size=32)
for i in range(sequence_length):
    output, hidden = model(input[i], hidden)

```

### Using PyTorch Built-in RNN

```python
import torch.nn as nn

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, text):
        # text shape: [batch_size, seq_len]
        
        # Embed words
        embedded = self.embedding(text)
        # embedded shape: [batch_size, seq_len, embedding_dim]
        
        # Pass through RNN
        output, hidden = self.rnn(embedded)
        # output shape: [batch_size, seq_len, hidden_dim]
        # hidden shape: [1, batch_size, hidden_dim]
        
        # Use final hidden state for classification
        # hidden[-1] gets last layer's hidden state
        final_hidden = hidden[-1]  # [batch_size, hidden_dim]
        
        # Classification
        logits = self.fc(final_hidden)
        return logits

# Create model
model = SentimentRNN(
    vocab_size=10000,
    embedding_dim=100,
    hidden_dim=256,
    output_dim=2  # Binary classification
)

# Forward pass
text = torch.LongTensor([[1, 2, 3, 4, 5]])  # Batch of 1, sequence of 5
output = model(text)
print(output.shape)  # [1, 2]

```

---

## 2.3 Training RNNs

### Backpropagation Through Time (BPTT)

```
Forward pass (left to right):
x₀ → h₀ → h₁ → h₂ → h₃ → loss

Backward pass (right to left):
x₀ ← h₀ ← h₁ ← h₂ ← h₃ ← loss

Gradient flows backward through time!

```

**Algorithm:**

```python
def train_rnn(model, data, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for batch in data:
            text, labels = batch
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(text)
            
            # Compute loss
            loss = criterion(predictions, labels)
            
            # Backward pass (BPTT happens here!)
            loss.backward()
            
            # Update weights
            optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")

```

---

# PART 3: VANISHING GRADIENTS & LSTM SOLUTION (35 minutes)

## 3.1 The Vanishing Gradient Problem

### Mathematical Explanation

```
Gradient flow backward:

∂Loss/∂h₀ = ∂Loss/∂h₃ × ∂h₃/∂h₂ × ∂h₂/∂h₁ × ∂h₁/∂h₀

Each term is a product of gradients through tanh:
∂h_t/∂h_{t-1} = W_hh × tanh'(...)

tanh'(x) ≤ 1 (derivative of tanh is at most 1)

If W_hh has eigenvalues < 1:
After T steps: (W_hh)^T → very small
Gradient "vanishes"!

Example:
If largest eigenvalue = 0.9
After 10 steps: 0.9^10 = 0.35
After 20 steps: 0.9^20 = 0.12
After 50 steps: 0.9^50 = 0.005

Early timesteps don't learn!

```

### Practical Demonstration

```python
import numpy as np

def demonstrate_vanishing_gradient():
    W = np.random.randn(10, 10) * 0.01  # Small random weights
    
    gradient = 1.0  # Start with gradient of 1
    
    for t in range(50):
        # Simulate backprop through one timestep
        gradient = gradient * np.linalg.norm(W) * 0.25  # tanh derivative ≈ 0.25
        
        if t % 10 == 0:
            print(f"After {t} steps: gradient = {gradient:.6f}")
    
    # Output:
    # After 0 steps: gradient = 1.000000
    # After 10 steps: gradient = 0.000001
    # After 20 steps: gradient = 0.000000
    # After 30 steps: gradient = 0.000000

```

---

## 3.2 Why This Matters

### Real Example: Long Movie Review

```
Review (200 words):
"This movie starts with an incredible opening scene. The cinematography is 
breathtaking. The first act keeps you on the edge of your seat. The acting 
is phenomenal. The soundtrack is perfect. However, midway through, the plot 
becomes confusing. The pacing slows down. By the third act, I was checking 
my phone. The ending was rushed and unsatisfying. Overall, DISAPPOINTING."

Simple RNN:
- Remembers: "incredible", "breathtaking", "phenomenal", "perfect"
- Forgets: "confusing", "rushed", "unsatisfying", "DISAPPOINTING"
- Prediction: POSITIVE (wrong!)

Why? Gradient vanished before reaching the key negative words at the end.

```

---

## 3.3 The LSTM Solution: Gates

### Core Idea

Instead of:

```
h_t = tanh(W × h_{t-1} + ...)

```

LSTM uses:

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t

Where:
- c_t: Cell state (memory)
- f_t: Forget gate (what to forget)
- i_t: Input gate (what to add)
- ⊙: Element-wise multiplication

```

**Key insight:**

- Simple RNN: Memory = f(old_memory)
- LSTM: Memory = forget × old_memory + input × new_info

**Gradient flow:**

```
∂c_t/∂c_{t-1} = f_t

If f_t ≈ 1 (keep memory):
Gradient flows unchanged!
No vanishing! ✓

```

---

# PART 4: LSTM ARCHITECTURE DEEP DIVE (35 minutes)

## 4.1 The Three Gates

### 1. Forget Gate

**Purpose:** Decide what to remove from memory

```
f_t = σ(W_f × [h_{t-1}, x_t] + b_f)

Where:
- σ: Sigmoid function (outputs 0 to 1)
- 0 = completely forget
- 1 = completely remember

```

**Example:**

```
Text: "The movie was great. However, the ending was terrible."

At "However":
- Forget gate sees transition word
- f_t = [0.1, 0.1, 0.1, ...]  # Low values
- Forgets previous "great" sentiment ✓

```

---

### 2. Input Gate

**Purpose:** Decide what new information to store

```
i_t = σ(W_i × [h_{t-1}, x_t] + b_i)
c̃_t = tanh(W_c × [h_{t-1}, x_t] + b_c)

New cell state candidate: c̃_t
Gate controlling how much to add: i_t

```

**Example:**

```
Text: "The movie was great. However, the ending was terrible."

At "terrible":
- Input gate: i_t = [0.9, 0.9, 0.9, ...]  # High values
- Candidate: c̃_t contains "negative sentiment"
- Add strong negative sentiment to memory ✓

```

---

### 3. Output Gate

**Purpose:** Decide what to output from current memory

```
o_t = σ(W_o × [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(c_t)

Where:
- o_t: Output gate
- h_t: Hidden state (what's exposed to next layer)

```

---

## 4.2 Complete LSTM Equations

```python
# At each timestep t:

# 1. Forget gate
f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)

# 2. Input gate
i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)
c̃_t = tanh(W_c @ [h_{t-1}, x_t] + b_c)

# 3. Update cell state
c_t = f_t * c_{t-1} + i_t * c̃_t

# 4. Output gate
o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(c_t)

# h_t is passed to next timestep and output layer

```

---

## 4.3 PyTorch LSTM Implementation

### Using Built-in LSTM

```python
class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        # text: [batch_size, seq_len]
        
        embedded = self.dropout(self.embedding(text))
        # embedded: [batch_size, seq_len, embedding_dim]
        
        # LSTM returns:
        # output: [batch_size, seq_len, hidden_dim]
        # (hidden, cell): ([n_layers, batch_size, hidden_dim], same)
        output, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state
        # hidden[-1]: [batch_size, hidden_dim]
        final_hidden = self.dropout(hidden[-1])
        
        return self.fc(final_hidden)

# Create model
model = LSTMSentiment(
    vocab_size=10000,
    embedding_dim=100,
    hidden_dim=256,
    output_dim=2,
    n_layers=2,
    dropout=0.5
)

```

---

## 4.4 Bidirectional LSTM

### Why Bidirectional?

**Problem with unidirectional:**

```
Text: "The movie was not good"

Processing left-to-right:
Step 1: "The" → Limited context
Step 2: "movie" → Still limited
Step 3: "was" → Still limited
Step 4: "not" → Finally see negation!
Step 5: "good" → Now can predict correctly

Issue: Early words don't have full context

```

**Bidirectional solution:**

```
Forward LSTM:  →  →  →  →  →
Text:         The movie was not good
Backward LSTM: ←  ←  ←  ←  ←

At each word: Combine forward and backward context!

At "movie":
- Forward: Knows "The movie"
- Backward: Knows "movie was not good"
- Combined: Full sentence context! ✓

```

### Implementation

```python
class BiLSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            bidirectional=True,  # KEY: Enable bidirectional
            batch_first=True
        )
        
        # NOTE: Output is now 2 * hidden_dim (forward + backward)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, text):
        embedded = self.embedding(text)
        
        output, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate final forward and backward hidden states
        # hidden[-2]: final forward
        # hidden[-1]: final backward
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        return self.fc(final_hidden)

```

---

# PART 5: SENTIMENT ANALYSIS WITH LSTM (30 minutes)

## 5.1 Complete Pipeline

### Data Preparation

```python
import torch
from torchtext.data import Field, LabelField, BucketIterator
from torchtext.datasets import IMDB

# Define fields
TEXT = Field(tokenize='spacy', lower=True, include_lengths=True)
LABEL = LabelField(dtype=torch.float)

# Load IMDB dataset
train_data, test_data = IMDB.splits(TEXT, LABEL)

# Build vocabulary
TEXT.build_vocab(train_data, max_size=10000, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

# Create iterators
BATCH_SIZE = 64
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    device=device
)

```

---

### Training Loop

```python
import torch.optim as optim

# Initialize model
model = BiLSTMSentiment(
    vocab_size=len(TEXT.vocab),
    embedding_dim=100,
    hidden_dim=256,
    output_dim=1  # Binary classification
)

# Load pre-trained embeddings
model.embedding.weight.data.copy_(TEXT.vocab.vectors)

# Optimizer and loss
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in iterator:
        optimizer.zero_grad()
        
        # Get batch data
        text, text_lengths = batch.text
        predictions = model(text).squeeze(1)
        
        # Calculate loss
        loss = criterion(predictions, batch.label)
        
        # Calculate accuracy
        rounded_preds = torch.round(torch.sigmoid(predictions))
        correct = (rounded_preds == batch.label).float()
        acc = correct.sum() / len(correct)
        
        # Backprop
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Train for 10 epochs
for epoch in range(10):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    print(f'Epoch {epoch}: Loss = {train_loss:.3f}, Acc = {train_acc:.3f}')

```

---

### Evaluation

```python
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            rounded_preds = torch.round(torch.sigmoid(predictions))
            correct = (rounded_preds == batch.label).float()
            acc = correct.sum() / len(correct)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Evaluate
test_loss, test_acc = evaluate(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}')

```

---

### Inference

```python
def predict_sentiment(model, sentence):
    model.eval()
    
    # Tokenize
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    
    # Convert to indices
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    
    # Convert to tensor
    tensor = torch.LongTensor(indexed).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        prediction = torch.sigmoid(model(tensor))
    
    return prediction.item()

# Test
print(predict_sentiment(model, "This film is great!"))  # High score (positive)
print(predict_sentiment(model, "This film is terrible!"))  # Low score (negative)
print(predict_sentiment(model, "This film is not good"))  # Low score (LSTM handles negation!)

```

---

## 5.2 Handling Edge Cases

### Negations

```python
# Test negation handling
sentences = [
    "This movie is good",           # Positive
    "This movie is not good",       # Negative (negation)
    "This movie is not bad",        # Positive (double negation)
    "This movie is not not good"    # Positive (triple negation)
]

for sent in sentences:
    score = predict_sentiment(model, sent)
    label = "POSITIVE" if score > 0.5 else "NEGATIVE"
    print(f"{sent:30} -> {score:.3f} ({label})")

# Output with trained LSTM:
# This movie is good             -> 0.912 (POSITIVE)
# This movie is not good         -> 0.234 (NEGATIVE) ✓
# This movie is not bad          -> 0.687 (POSITIVE) ✓
# This movie is not not good     -> 0.798 (POSITIVE) ✓

```

---

### Context Switching

```python
# Test "but" handling
sentences = [
    "The movie was great",
    "The movie was great but the ending was terrible",
    "The movie was terrible but the ending was great"
]

for sent in sentences:
    score = predict_sentiment(model, sent)
    label = "POSITIVE" if score > 0.5 else "NEGATIVE"
    print(f"{sent} -> {score:.3f} ({label})")

# With trained LSTM:
# The movie was great -> 0.89 (POSITIVE)
# ... but ending terrible -> 0.42 (NEGATIVE) ✓ "but" switches
# ... but ending great -> 0.71 (POSITIVE) ✓ Handles complexity

```

---

# PART 6: ADVANCED TOPICS & PRODUCTION (20 minutes)

## 6.1 Attention Mechanism (Brief Intro)

### Problem with LSTM

```
Long sequence: [w1, w2, ..., w100]
                              ↓
                         Final hidden state
                              ↓
                          Prediction

Issue: All information compressed into one vector!
Important words at position 10 might be "forgotten"

```

### Attention Solution

```
Compute attention weights:
α1, α2, ..., α100

Context = Σ (αi × hi)

Where:
- αi: How much to attend to word i
- hi: Hidden state at word i

Result: Weighted combination of ALL hidden states
No information loss! ✓

```

---

## 6.2 Production Considerations

### Model Optimization

```python
# 1. Gradient Clipping (prevent exploding gradients)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. Learning Rate Scheduling
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.5,
    patience=2
)

# 3. Early Stopping
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(epochs):
    train_loss = train(...)
    val_loss = evaluate(...)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print("Early stopping!")
        break

```

---

### Deployment

```python
# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': TEXT.vocab,
    'config': {
        'vocab_size': len(TEXT.vocab),
        'embedding_dim': 100,
        'hidden_dim': 256
    }
}, 'sentiment_model.pth')

# Load for inference
checkpoint = torch.load('sentiment_model.pth')
model = BiLSTMSentiment(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# API endpoint (FastAPI)
from fastapi import FastAPI

app = FastAPI()

@app.post("/predict")
async def predict(text: str):
    score = predict_sentiment(model, text)
    label = "positive" if score > 0.5 else "negative"
    return {
        "text": text,
        "score": float(score),
        "label": label
    }

```

---

## Summary: Key Takeaways

**RNNs:**

1. Process sequences one step at a time
2. Maintain hidden state (memory)
3. Share weights across time steps
4. Suffer from vanishing gradients

**LSTMs:**

1. Solve vanishing gradient with gates
2. Forget gate: What to remove
3. Input gate: What to add
4. Output gate: What to expose
5. Cell state acts as memory highway

**Sentiment Analysis:**

1. Sequence classification task
2. LSTM handles negations and context
3. Bidirectional improves context understanding
4. Attention helps with long sequences

**Production:**

1. Gradient clipping prevents instability
2. Early stopping prevents overfitting
3. Pre-trained embeddings improve performance
4. Model checkpointing for best results

---

**End of Lecture Notes**

**Vishlesan i-Hub IIT Patna × Masai School**

*Master sequences, understand context, build intelligent NLP systems!*

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