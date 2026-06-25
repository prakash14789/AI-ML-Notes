# 62. Transformers & BERT Tune - Suman - 26 Mar 2026

# Transformers & BERT Fine-Tuning - Lecture Notes

## PPT File: [Click Here](https://drive.google.com/file/d/1sf96EDKxaqaUwCpGtULW4ZPBDFNyUtrm/view?usp=sharing)

**Program:** Vishlesan i-Hub IIT Patna x Masai School — AIM (AI & Machine Learning)

**Prerequisites:** Deep learning basics, RNNs/LSTMs, PyTorch, attention mechanism basics

---

## Session Overview

This session covers the Transformer architecture and BERT, the foundation of modern NLP. You'll learn self-attention mechanisms, positional encoding, build a complete mini-transformer from scratch, and master practical BERT fine-tuning for downstream tasks.

---

## Learning Objectives

By the end of this session, you will be able to:

1. **Understand self-attention** mechanism and its advantages over recurrent approaches
2. **Implement positional encoding** using sine/cosine functions
3. **Build a complete mini-Transformer** from scratch with all components
4. **Explain BERT architecture** and pre-training objectives (MLM, NSP)
5. **Fine-tune BERT** for classification tasks with hands-on examples
6. **Optimize fine-tuning** with layer freezing and learning rate strategies
7. **Deploy BERT models** efficiently for production use

---

## Part 1: The Limitations of RNNs and the Need for Transformers (15 minutes)

[Content remains the same as original...]

---

## Part 2: Self-Attention Mechanism (30 minutes)

[Content remains the same as original through the concrete example and PyTorch implementation...]

---

## Part 3: Positional Encoding (20 minutes)

[Content remains the same as original...]

---

## Part 4: Building a Mini-Transformer from Scratch (30 minutes)

### Why Build a Mini-Transformer?

**Learning objective:** Understand transformers by building one from first principles.

**What we'll build:**

- Complete mini-transformer encoder
- Handles sequence classification
- ~500 lines of working code
- Demonstrates all core concepts in practice

**Architecture overview:**

```
Input tokens
    ↓
Token Embedding (vocab → d_model)
    ↓
Positional Encoding (add position info)
    ↓
Transformer Block 1 (attention + FFN)
    ↓
Transformer Block 2
    ↓
... (N blocks)
    ↓
[CLS] token extraction
    ↓
Classification head
    ↓
Output logits

```

### Step 1: Input Embeddings

```python
import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    """
    Convert token IDs to dense vectors
    
    Example:
        Token "cat" (ID: 524) → [0.12, -0.45, 0.89, ..., 0.34] (d_model dims)
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len] token IDs
        Returns:
            embeddings: [batch, seq_len, d_model]
        """
        # Scale embeddings by sqrt(d_model) as in original paper
        # This helps gradient flow in later layers
        return self.embedding(x) * math.sqrt(self.d_model)

# Example usage
vocab_size = 10000
d_model = 256

embedding_layer = InputEmbedding(vocab_size, d_model)

# Input: batch of 2 sequences, each 10 tokens long
token_ids = torch.randint(0, vocab_size, (2, 10))
embeddings = embedding_layer(token_ids)

print(f"Input shape: {token_ids.shape}")      # [2, 10]
print(f"Output shape: {embeddings.shape}")    # [2, 10, 256]

```

### Step 2: Positional Encoding

```python
class PositionalEncoding(nn.Module):
    """
    Add position information to embeddings
    
    Why needed: Self-attention is permutation-invariant
    Solution: Inject position using sine/cosine functions
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        # Register as buffer (not a trainable parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model] embeddings
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Visualization of positional encoding pattern
import matplotlib.pyplot as plt

pe_layer = PositionalEncoding(d_model=128, max_len=50)
pe_matrix = pe_layer.pe.squeeze(0).numpy()  # [50, 128]

plt.figure(figsize=(15, 5))
plt.pcolormesh(pe_matrix.T, cmap='RdBu')
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.colorbar()
plt.title('Positional Encoding Pattern\n(Notice smooth transitions)')
plt.show()

```

**Key insight from visualization:**

- Low dimensions (bottom): Change slowly across positions
- High dimensions (top): Change rapidly across positions
- This multi-scale pattern helps model learn both local and global relationships

### Step 3: Multi-Head Attention Block

```python
class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-head self-attention with residual connection
    
    Input: [batch, seq_len, d_model]
    Output: [batch, seq_len, d_model] (same shape!)
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        # Use PyTorch's built-in multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Important: input is [batch, seq, features]
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, seq_len] optional padding mask
        Returns:
            output: [batch, seq_len, d_model]
        """
        # Self-attention: query=key=value=x
        attended, attention_weights = self.attention(
            x, x, x, 
            key_padding_mask=mask
        )
        
        # Residual connection + layer norm
        x = self.norm(x + self.dropout(attended))
        
        return x

```

### Step 4: Feed-Forward Network Block

```python
class FeedForwardBlock(nn.Module):
    """
    Position-wise feed-forward network
    
    Architecture: Linear → ReLU → Dropout → Linear
    Typical expansion: d_model → 4*d_model → d_model
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),      # Expand
            nn.ReLU(),                      # Non-linearity
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)        # Project back
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
        """
        # Feed-forward + residual + layer norm
        ff_out = self.ffn(x)
        x = self.norm(x + self.dropout(ff_out))
        
        return x

```

### Step 5: Complete Mini-Transformer Block

```python
class MiniTransformerBlock(nn.Module):
    """
    Single transformer encoder block
    
    Components:
    1. Multi-head self-attention
    2. Add & Norm
    3. Feed-forward network
    4. Add & Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention_block = MultiHeadAttentionBlock(
            d_model, num_heads, dropout
        )
        self.ffn_block = FeedForwardBlock(d_model, d_ff, dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, seq_len] padding mask
        Returns:
            output: [batch, seq_len, d_model]
        """
        # Attention sub-block
        x = self.attention_block(x, mask)
        
        # Feed-forward sub-block
        x = self.ffn_block(x)
        
        return x

# Test the transformer block
block = MiniTransformerBlock(d_model=256, num_heads=8, d_ff=1024)

# Input: batch=2, seq_len=10, d_model=256
x = torch.randn(2, 10, 256)
output = block(x)

print(f"Input shape: {x.shape}")       # [2, 10, 256]
print(f"Output shape: {output.shape}") # [2, 10, 256] - same!

```

### Step 6: Complete Mini-Transformer Model

```python
class MiniTransformer(nn.Module):
    """
    Complete mini-transformer for sequence classification
    
    Architecture:
    Input tokens → Embeddings → Positional Encoding → 
    N Transformer Blocks → [CLS] pooling → Classifier → Logits
    """
    def __init__(
        self,
        vocab_size,           # Size of vocabulary
        num_classes,          # Number of output classes
        d_model=256,          # Model dimension
        num_heads=8,          # Number of attention heads
        num_layers=4,         # Number of transformer blocks
        d_ff=1024,            # Feed-forward dimension
        max_len=512,          # Maximum sequence length
        dropout=0.1           # Dropout probability
    ):
        super().__init__()
        
        # Input processing
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            MiniTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len] token IDs
            mask: [batch, seq_len] padding mask (1 = valid, 0 = padding)
        
        Returns:
            logits: [batch, num_classes]
        """
        # 1. Embed tokens
        x = self.embedding(x)          # [batch, seq_len, d_model]
        
        # 2. Add positional encoding
        x = self.pos_encoding(x)       # [batch, seq_len, d_model]
        
        # 3. Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)         # [batch, seq_len, d_model]
        
        # 4. Pool: use [CLS] token (first token) for classification
        cls_output = x[:, 0, :]        # [batch, d_model]
        
        # 5. Classify
        logits = self.classifier(cls_output)  # [batch, num_classes]
        
        return logits

# Initialize mini-transformer
model = MiniTransformer(
    vocab_size=10000,
    num_classes=2,      # Binary classification (positive/negative)
    d_model=256,
    num_heads=8,
    num_layers=4,
    d_ff=1024,
    dropout=0.1
)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
# ~5-6 million parameters

```

### Step 7: Training the Mini-Transformer

```python
import torch.optim as optim
from torch.utils.data import DataLoader

# Training configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Example training loop
def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        mask = batch['mask'].to(device) if 'mask' in batch else None
        
        # Forward pass
        logits = model(input_ids, mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

```

### Mini-Transformer vs BERT: Comparison

Aspect | Mini-Transformer | BERT-Base | BERT-Large
Architecture | Encoder-only | Encoder-only | Encoder-only
Layers | 4 | 12 | 24
Hidden size (d_model) | 256 | 768 | 1024
Attention heads | 8 | 12 | 16
FFN size (d_ff) | 1024 | 3072 | 4096
Total parameters | ~5M | 110M | 340M
Training | From scratch | Pre-trained | Pre-trained
Training data | Your dataset | 3.3B words | 3.3B words
Training time | Hours | Days (~$500k) | Weeks (~$1.5M)
Use case | Learning, prototyping | Production | Max accuracy

**When to use Mini-Transformer:**

- ✅ Learning transformer architecture
- ✅ Prototyping new ideas quickly
- ✅ Small datasets (< 10k examples)
- ✅ Domain-specific tasks with limited data
- ✅ Limited compute resources

**When to use BERT:**

- ✅ Production applications
- ✅ Need state-of-the-art accuracy
- ✅ Transfer learning from massive pre-training
- ✅ Have compute for fine-tuning
- ✅ Standard NLP tasks (classification, NER, QA)

### Key Insights from Building Mini-Transformer

**1. Modularity:**

- Each component is independent
- Can swap/upgrade individual parts
- Easy to experiment with variants

**2. Residual Connections:**

- Enable training deep networks (4+ layers)
- Gradient flows directly to earlier layers
- Model can learn identity mapping if needed

**3. Layer Normalization:**

- Stabilizes training
- Works with variable sequence lengths
- Applied after each sub-layer

**4. Positional Encoding:**

- Absolutely critical for performance
- Without it, model is permutation-invariant
- Can't distinguish "dog bites man" from "man bites dog"

**5. Computational Cost:**

- Self-attention: O(n²) where n = sequence length
- Feed-forward: O(n) but with large constant (d_ff = 4 × d_model)
- Trade-off: Parallelization vs memory

---

## Part 5: BERT Architecture and Pre-Training (20 minutes)

### BERT Overview

**BERT:** Bidirectional Encoder Representations from Transformers

**Key innovation:** Pre-train on massive unlabeled data, then fine-tune for specific tasks

**Architecture:**

- Stack of Transformer encoder blocks (like our mini-transformer!)
- No decoder (encoder-only architecture)
- Bidirectional context (sees both left and right simultaneously)

**Model sizes:**

```
BERT-Base:
- 12 transformer encoder layers
- 768 hidden dimensions (d_model)
- 12 attention heads
- 110M parameters

BERT-Large:
- 24 transformer encoder layers
- 1024 hidden dimensions
- 16 attention heads
- 340M parameters

```

[Rest of BERT pre-training content remains the same...]

---

## Part 6: Fine-Tuning BERT - Complete Hands-On Guide (30 minutes)

### Fine-Tuning Strategy Overview

**Concept:** Use pre-trained BERT as a feature extractor, add task-specific head

**Why fine-tuning works:**

1. BERT already understands language (from pre-training)
2. We just teach it the specific task (classification, NER, QA)
3. Requires minimal data (1k-10k examples vs 3.3B words for pre-training)
4. Fast (minutes-hours vs days-weeks)

**General approach:**

1. Load pre-trained BERT weights
2. Add task-specific output layer
3. Optionally freeze some layers
4. Train on your task-specific data
5. Evaluate and deploy

### Hands-On Example 1: Sentiment Classification

**Task:** Classify movie reviews as positive or negative

**Step 1: Install and Import**

```python
# Install transformers library
!pip install transformers torch

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

```

**Step 2: Load Pre-trained BERT**

```python
# Load tokenizer (converts text → token IDs)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load model with classification head
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,           # Binary classification
    output_attentions=False,
    output_hidden_states=False
)

print(f"Model has {model.num_parameters():,} parameters")
# Output: 109,483,778 parameters (110M)

```

**Step 3: Prepare Data**

```python
class SentimentDataset(Dataset):
    """Dataset for sentiment classification"""
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,      # Add [CLS] and [SEP]
            max_length=self.max_len,
            padding='max_length',          # Pad to max_length
            truncation=True,               # Truncate if too long
            return_attention_mask=True,    # Return attention mask
            return_tensors='pt'            # Return PyTorch tensors
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Example data
texts = [
    "This movie was absolutely fantastic! I loved every minute.",
    "Terrible waste of time. Boring and predictable.",
    "One of the best films I've seen this year!",
    "Disappointing. Expected much more."
]
labels = [1, 0, 1, 0]  # 1 = positive, 0 = negative

# Create dataset and dataloader
dataset = SentimentDataset(texts, labels, tokenizer)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Inspect a batch
batch = next(iter(train_loader))
print("Input IDs shape:", batch['input_ids'].shape)      # [2, 128]
print("Attention mask shape:", batch['attention_mask'].shape)  # [2, 128]
print("Labels shape:", batch['labels'].shape)            # [2]

```

**Step 4: Set Up Optimizer and Scheduler**

```python
# Optimizer: AdamW (Adam with weight decay)
optimizer = AdamW(
    model.parameters(),
    lr=2e-5,              # Small learning rate for fine-tuning
    eps=1e-8,             # Epsilon for numerical stability
    weight_decay=0.01     # L2 regularization
)

# Learning rate scheduler with warmup
num_epochs = 3
num_training_steps = len(train_loader) * num_epochs
num_warmup_steps = num_training_steps // 10  # 10% warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

```

**Step 5: Training Loop**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    
    for batch in data_loader:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(predictions == labels)
    
    return total_loss / len(data_loader), correct_predictions.double() / len(dataset)

# Training loop
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print("-" * 50)
    
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, scheduler, device
    )
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")

```

**Step 6: Inference**

```python
def predict(text, model, tokenizer, device):
    """Predict sentiment for a single text"""
    model.eval()
    
    # Tokenize
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        probabilities = torch.softmax(logits, dim=1).squeeze()
    
    label = "Positive" if prediction == 1 else "Negative"
    confidence = probabilities[prediction].item()
    
    return label, confidence

# Test predictions
test_texts = [
    "Amazing movie! Highly recommend.",
    "Worst film ever made.",
    "It was okay, nothing special."
]

for text in test_texts:
    label, confidence = predict(text, model, tokenizer, device)
    print(f"Text: {text}")
    print(f"Prediction: {label} (confidence: {confidence:.2%})\n")

```

### Hands-On Example 2: Layer Freezing Strategies

**Strategy 1: Freeze All BERT, Train Only Classifier**

```python
# Fastest training, works with small datasets (100-1000 examples)
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# Freeze all BERT parameters
for param in model.bert.parameters():
    param.requires_grad = False

# Only classifier head is trainable
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")  # ~1,500 (just classifier)

```

**Strategy 2: Freeze Bottom Layers, Fine-Tune Top Layers**

```python
# Good balance: freeze first 8 layers, fine-tune last 4
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# Freeze embeddings
for param in model.bert.embeddings.parameters():
    param.requires_grad = False

# Freeze first 8 encoder layers
for layer in model.bert.encoder.layer[:8]:
    for param in layer.parameters():
        param.requires_grad = False

# Top 4 layers + classifier are trainable
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")  # ~40M

```

**Strategy 3: Gradual Unfreezing (Best Performance)**

```python
def gradually_unfreeze(model, epoch, total_epochs=5):
    """
    Progressively unfreeze layers as training progresses
    
    Epoch 1: Only classifier
    Epoch 2: Classifier + last 2 layers
    Epoch 3: Classifier + last 4 layers
    Epoch 4: Classifier + last 6 layers
    Epoch 5: All layers (full fine-tuning)
    """
    num_layers = 12  # BERT-Base has 12 layers
    
    # Start with all frozen
    for param in model.bert.parameters():
        param.requires_grad = False
    
    # Gradually unfreeze from top
    layers_to_unfreeze = min(2 * epoch, num_layers)
    
    for layer in model.bert.encoder.layer[-layers_to_unfreeze:]:
        for param in layer.parameters():
            param.requires_grad = True
    
    # Classifier always trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Epoch {epoch}: {trainable:,} trainable parameters")

# Use in training loop
for epoch in range(1, num_epochs + 1):
    gradually_unfreeze(model, epoch, num_epochs)
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)

```

### Hyperparameter Guidelines for Fine-Tuning

Hyperparameter | Recommended Value | Rationale
Learning rate | 2e-5 to 5e-5 | Much lower than training from scratch (1e-3)
Batch size | 16 or 32 | Limited by GPU memory; use gradient accumulation if needed
Epochs | 2-4 | BERT fine-tunes quickly; more epochs risk overfitting
Max sequence length | 128 or 512 | 128 for most tasks; 512 for long documents (more memory)
Warmup steps | 10% of total steps | Stabilizes training in early phase
Weight decay | 0.01 | L2 regularization prevents overfitting
Gradient clipping | 1.0 | Prevents exploding gradients
Dropout | 0.1 | Already in BERT; don't change

### Common Fine-Tuning Pitfalls and Solutions

**Problem 1: Catastrophic Forgetting**

- **Symptom:** Model performs worse than random after fine-tuning
- **Cause:** Learning rate too high, destroying pre-trained knowledge
- **Solution:** Use 2e-5 learning rate (not 1e-3), freeze bottom layers

**Problem 2: Overfitting on Small Datasets**

- **Symptom:** Perfect training accuracy, poor validation accuracy
- **Solution:** Freeze more layers, use dropout, data augmentation, early stopping

**Problem 3: Out of Memory Errors**

- **Symptom:** CUDA out of memory during training
- **Solutions:**

Reduce batch size (16 → 8 → 4)
Reduce max_length (512 → 256 → 128)
Use gradient accumulation
Use mixed precision training (FP16)

**Problem 4: Slow Training**

- **Symptom:** Takes too long to train
- **Solutions:**

Use smaller BERT variant (DistilBERT)
Freeze more layers
Reduce max_length
Use gradient checkpointing

### Saving and Loading Fine-Tuned Models

```python
# Save fine-tuned model
model.save_pretrained('./fine_tuned_bert_sentiment')
tokenizer.save_pretrained('./fine_tuned_bert_sentiment')

# Load fine-tuned model
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained('./fine_tuned_bert_sentiment')
tokenizer = BertTokenizer.from_pretrained('./fine_tuned_bert_sentiment')

# Use for inference
text = "This product is amazing!"
encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
outputs = model(**encoding)
prediction = torch.argmax(outputs.logits, dim=1).item()

```

---

## Part 7: Practical Considerations and Optimization (10 minutes)

[Content remains mostly the same, with additions...]

### Production Deployment Checklist

**Before deploying BERT to production:**

1. 
**Model Optimization:**

✅ Export to ONNX for 2-3× faster inference
✅ Use DistilBERT (40% smaller, 60% faster, 97% accuracy)
✅ Quantize model (INT8) for mobile/edge deployment

2. 
**Inference Optimization:**

✅ Batch similar-length sequences together
✅ Use dynamic padding (not max_length)
✅ Cache tokenizer
✅ Use GPU if available

3. 
**Monitoring:**

✅ Track latency (p50, p95, p99)
✅ Monitor accuracy drift over time
✅ Log prediction confidence scores

4. 
**Error Handling:**

✅ Handle empty inputs gracefully
✅ Set maximum input length limits
✅ Provide fallback for low-confidence predictions

---

## Key Takeaways

### Core Concepts Mastered

1. **Self-Attention:** Allows parallel processing, O(n²) complexity but worth it
2. **Positional Encoding:** Critical for position-aware understanding
3. **Mini-Transformer:** Built complete encoder from scratch (~500 lines)
4. **BERT Architecture:** 12/24 layer transformer with MLM + NSP pre-training
5. **Fine-Tuning:** Transfer learning with 2-4 epochs, 2e-5 learning rate
6. **Layer Freezing:** Strategic freezing speeds training and prevents overfitting

### Hands-On Skills Acquired

- ✅ Built mini-transformer from first principles
- ✅ Loaded and fine-tuned pre-trained BERT
- ✅ Implemented layer freezing strategies
- ✅ Set up proper optimizers and schedulers
- ✅ Saved and deployed fine-tuned models
- ✅ Optimized for production use

### When to Use What

Task | Model Choice | Training Approach
Learning transformers | Mini-Transformer | Train from scratch
Production classification | BERT-Base | Fine-tune 2-4 epochs
Maximum accuracy | BERT-Large | Fine-tune all layers
Mobile/Edge deployment | DistilBERT | Fine-tune + quantize
Very small dataset (<1k) | BERT frozen | Train only classifier

---

**End of Updated Lecture Notes**

**Vishlesan i-Hub IIT Patna × Masai School**

*From self-attention fundamentals to production BERT deployment*

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