# 64. Interpreting Transformers - Suman - 4 Apr 2026

# Lecture Notes: Interpreting Transformers

## Session Overview

Transformers have revolutionized NLP, but understanding *how* they work remains challenging. This session covers three powerful interpretability techniques: attention roll-out (tracking information flow), probing classifiers (decoding layer representations), and causal tracing (identifying critical components).

## Learning Objectives

By the end of this session, you will be able to:

1. Compute and visualize attention roll-out to trace information flow
2. Train probing classifiers to decode layer activations
3. Perform causal interventions to identify critical model components
4. Debug transformer behavior using interpretability tools
5. Understand the hierarchical processing in transformer models

---

## 1. The Interpretability Challenge

### 1.1 Why Interpretability Matters

**The Black Box Problem:**

```python
# We can use transformers easily
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

input_text = "The trophy didn't fit in the suitcase because it was too big."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model(**inputs)

# But what happened inside?
# - 12 layers × 12 heads = 144 attention mechanisms
# - 110 million parameters
# - Hidden states of dimension 768
# How do we understand what it learned?

```

**Key Questions:**

1. What linguistic features does each layer encode?
2. How does information flow through the network?
3. Which components are responsible for specific behaviors?
4. Why does the model fail on certain inputs?

---

## 2. Attention Roll-Out

### 2.1 Understanding Attention Composition

**The Problem:**
Looking at attention in a single layer shows local patterns, but information flows through ALL layers. How do attention patterns compose?

**Attention Roll-Out Formula:**

```
A_rollout = A_L × A_{L-1} × ... × A_2 × A_1

Where:
- A_i is the attention matrix from layer i
- Matrix multiplication captures composition
- Result: end-to-end attention from input to output

```

### 2.2 Mathematical Foundation

```python
import numpy as np
import torch

def compute_attention_rollout(attention_matrices, head_fusion='mean'):
    """
    Compute attention roll-out across all layers.
    
    Args:
        attention_matrices: List of attention matrices [num_layers, num_heads, seq_len, seq_len]
        head_fusion: How to combine multiple heads ('mean', 'max', 'min')
    
    Returns:
        Rolled-out attention matrix [seq_len, seq_len]
    """
    num_layers = len(attention_matrices)
    
    # Step 1: Fuse attention heads in each layer
    fused_attention = []
    for layer_attention in attention_matrices:
        # layer_attention shape: [num_heads, seq_len, seq_len]
        if head_fusion == 'mean':
            fused = layer_attention.mean(dim=0)  # Average across heads
        elif head_fusion == 'max':
            fused = layer_attention.max(dim=0)[0]
        else:
            fused = layer_attention.min(dim=0)[0]
        
        fused_attention.append(fused)
    
    # Step 2: Add residual connections
    # In transformers: output = attention(input) + input
    # So: A_effective = 0.5 * A + 0.5 * I (identity)
    seq_len = fused_attention[0].shape[0]
    identity = torch.eye(seq_len)
    
    for i in range(num_layers):
        fused_attention[i] = 0.5 * fused_attention[i] + 0.5 * identity
    
    # Step 3: Roll out attention across layers
    rollout = identity.clone()
    
    for layer_idx in range(num_layers):
        # Matrix multiplication: how token i affects token j through all layers so far
        rollout = fused_attention[layer_idx] @ rollout
    
    return rollout

# Example usage
num_layers, num_heads, seq_len = 12, 12, 10

# Simulate attention patterns (in practice, extract from model)
attention_patterns = [
    torch.softmax(torch.randn(num_heads, seq_len, seq_len), dim=-1)
    for _ in range(num_layers)
]

rollout = compute_attention_rollout(attention_patterns)
print("Rolled-out attention shape:", rollout.shape)
print("\nAttention from token 0 to all tokens:")
print(rollout[0])

```

### 2.3 Visualization and Interpretation

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention_rollout(rollout, tokens):
    """
    Visualize rolled-out attention as a heatmap.
    
    Args:
        rollout: Attention matrix [seq_len, seq_len]
        tokens: List of token strings
    """
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(
        rollout.numpy(),
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='YlOrRd',
        annot=True,
        fmt='.2f',
        square=True,
        cbar_kws={'label': 'Attention Weight'}
    )
    
    plt.title('Attention Roll-Out: End-to-End Information Flow', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Attended To (Source)', fontsize=12)
    plt.ylabel('Attending From (Target)', fontsize=12)
    plt.tight_layout()
    plt.show()

# Example: Pronoun resolution
tokens = ['The', 'trophy', 'did', "n't", 'fit', 'because', 'it', 'was', 'big']
visualize_attention_rollout(rollout[:9, :9], tokens)

```

### 2.4 Practical Example: Coreference Resolution

```python
from transformers import BertModel, BertTokenizer

def analyze_pronoun_attention(sentence, pronoun_idx):
    """
    Analyze what a pronoun attends to across all layers.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    
    # Tokenize
    inputs = tokenizer(sentence, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Forward pass
    outputs = model(**inputs)
    attentions = outputs.attentions  # Tuple of [1, num_heads, seq_len, seq_len]
    
    # Extract attention matrices (remove batch dimension)
    attention_matrices = [attn[0] for attn in attentions]
    
    # Compute roll-out
    rollout = compute_attention_rollout(attention_matrices)
    
    # What does the pronoun attend to?
    pronoun_attention = rollout[pronoun_idx]
    
    # Find top attended tokens
    top_k = 5
    top_indices = torch.argsort(pronoun_attention, descending=True)[:top_k]
    
    print(f"Pronoun '{tokens[pronoun_idx]}' attends to:")
    for idx in top_indices:
        print(f"  {tokens[idx]}: {pronoun_attention[idx]:.3f}")
    
    return rollout, tokens

# Test
sentence = "The trophy didn't fit in the suitcase because it was too big."
rollout, tokens = analyze_pronoun_attention(sentence, pronoun_idx=13)  # 'it'

```

---

## 3. Probing Classifiers

### 3.1 What Are Probing Classifiers?

**Core Idea:**
If a layer's hidden states encode some linguistic property (e.g., part-of-speech), we should be able to train a simple classifier to extract it.

**Probing Classifier Architecture:**

```
Layer Activations → Linear Classifier → Prediction
     [768]         →    [768 → K]      →    [K]
                                             K classes

```

**Key Principle:**
Use simple classifiers (linear or small MLP) to avoid learning the property from scratch. We want to test if the information is *already there*, not if we can compute it.

### 3.2 Implementing Probing Classifiers

```python
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class LinearProbe(nn.Module):
    """
    Simple linear classifier for probing layer activations.
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

def extract_layer_activations(model, tokenizer, sentences, layer_idx, token_idx=-1):
    """
    Extract activations from a specific layer for a set of sentences.
    
    Args:
        model: Transformer model
        tokenizer: Corresponding tokenizer
        sentences: List of sentences
        layer_idx: Which layer to extract (0-11 for BERT-base)
        token_idx: Which token position (-1 for [CLS], or specific position)
    
    Returns:
        activations: Tensor of shape [num_sentences, hidden_dim]
    """
    model.eval()
    activations = []
    
    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
            outputs = model(**inputs, output_hidden_states=True)
            
            # outputs.hidden_states: tuple of [batch, seq_len, hidden_dim]
            # Index: [layer_idx][batch_idx, token_idx, :]
            layer_activation = outputs.hidden_states[layer_idx][0, token_idx, :]
            activations.append(layer_activation)
    
    return torch.stack(activations)

def train_probe(activations, labels, probe_type='linear'):
    """
    Train a probing classifier.
    
    Args:
        activations: [num_samples, hidden_dim]
        labels: [num_samples]
        probe_type: 'linear' or 'mlp'
    
    Returns:
        Trained classifier and accuracy
    """
    # Convert to numpy for sklearn
    X = activations.numpy()
    y = labels.numpy()
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if probe_type == 'linear':
        # Logistic regression (linear probe)
        clf = LogisticRegression(max_iter=1000, random_state=42)
    else:
        # Could use sklearn MLPClassifier for non-linear probe
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=42)
    
    # Train
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return clf, accuracy

# Example: Probing for part-of-speech
sentences = [
    "The cat sat on the mat.",
    "She runs every morning.",
    "They are happy.",
    # ... more sentences
]
pos_labels = torch.tensor([0, 1, 0, ...])  # 0=noun, 1=verb, etc.

model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Extract activations from layer 5
activations_layer5 = extract_layer_activations(model, tokenizer, sentences, layer_idx=5)

# Train probe
probe, accuracy = train_probe(activations_layer5, pos_labels)
print(f"Layer 5 POS accuracy: {accuracy:.1%}")

```

### 3.3 Probing Across All Layers

```python
def probe_all_layers(model, tokenizer, sentences, labels, task_name):
    """
    Probe all layers to see where information is encoded.
    """
    num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    
    results = []
    
    for layer_idx in range(num_layers):
        # Extract activations
        activations = extract_layer_activations(model, tokenizer, sentences, layer_idx)
        
        # Train probe
        _, accuracy = train_probe(activations, labels)
        
        results.append({
            'layer': layer_idx,
            'accuracy': accuracy
        })
        
        print(f"Layer {layer_idx:2d}: {accuracy:.1%}")
    
    # Visualize
    import pandas as pd
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['layer'], df['accuracy'], marker='o', linewidth=2, markersize=8)
    plt.axhline(0.5, color='red', linestyle='--', label='Random baseline')
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Probe Accuracy', fontsize=12)
    plt.title(f'Probing Accuracy for {task_name} Across Layers', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return df

# Run for different tasks
tasks = {
    'Part-of-Speech': pos_labels,
    'Named Entity': ner_labels,
    'Semantic Role': role_labels
}

for task_name, labels in tasks.items():
    probe_all_layers(model, tokenizer, sentences, labels, task_name)

```

### 3.4 Interpreting Probe Results

**Typical Findings:**

```python
print("""
Probing Results Interpretation:

Layer 0-2 (Early):
  - POS tagging: 75-85% accuracy
  - Syntax tree depth: 70-80%
  - Character-level features: 90%+
  
  → Learn SURFACE-level features

Layer 3-6 (Middle):
  - Named entities: 80-90%
  - Coreference: 70-85%
  - Dependency relations: 75-85%
  
  → Learn STRUCTURAL features

Layer 7-12 (Deep):
  - Semantic roles: 75-90%
  - Sentiment: 80-90%
  - World knowledge: 60-80%
  
  → Learn SEMANTIC features

Pattern: Clear hierarchy from syntax to semantics
""")

```

---

## 4. Causal Tracing

### 4.1 From Correlation to Causation

**The Problem:**
Probing shows what information is *present*, but not what's *used*. A layer might encode POS tags without actually using them for the task.

**Causal Question:**
"If I corrupt layer L, does the model's behavior change?"

**Intervention Logic:**

```
Normal:     Input → Layer 1 → ... → Layer L → ... → Output
Corrupted:  Input → Layer 1 → ... → NOISE   → ... → Output?

If output changes significantly → Layer L is causally important
If output unchanged → Layer L is not used (despite encoding info)

```

### 4.2 Implementing Causal Interventions

```python
def causal_intervention(model, tokenizer, sentence, corrupt_layer, noise_level=1.0):
    """
    Perform causal intervention by corrupting a specific layer.
    
    Args:
        model: Transformer model
        tokenizer: Tokenizer
        sentence: Input sentence
        corrupt_layer: Which layer to corrupt
        noise_level: How much noise to add
    
    Returns:
        normal_output, corrupted_output, causal_effect
    """
    inputs = tokenizer(sentence, return_tensors='pt')
    
    # Normal forward pass
    with torch.no_grad():
        normal_outputs = model(**inputs, output_hidden_states=True)
        normal_hidden = normal_outputs.last_hidden_state
    
    # Corrupted forward pass using hooks
    corrupted_hidden = None
    
    def corruption_hook(module, input, output):
        """Hook to corrupt layer activations."""
        nonlocal corrupted_hidden
        
        # Add Gaussian noise
        noise = torch.randn_like(output) * noise_level * output.std()
        corrupted = output + noise
        
        # Store for analysis
        corrupted_hidden = corrupted
        
        return corrupted
    
    # Register hook on the target layer
    layer = model.encoder.layer[corrupt_layer]  # For BERT
    handle = layer.register_forward_hook(corruption_hook)
    
    # Forward pass with corruption
    with torch.no_grad():
        corrupted_outputs = model(**inputs, output_hidden_states=True)
        corrupted_final = corrupted_outputs.last_hidden_state
    
    # Remove hook
    handle.remove()
    
    # Measure causal effect (L2 distance)
    causal_effect = torch.norm(normal_hidden - corrupted_final, p=2)
    
    return normal_hidden, corrupted_final, causal_effect.item()

# Test causal importance of each layer
def test_causal_importance(model, tokenizer, sentence):
    """
    Test causal importance of all layers.
    """
    num_layers = model.config.num_hidden_layers
    effects = []
    
    for layer_idx in range(num_layers):
        _, _, effect = causal_intervention(model, tokenizer, sentence, layer_idx)
        effects.append(effect)
        print(f"Layer {layer_idx:2d}: Causal effect = {effect:.3f}")
    
    # Visualize
    plt.figure(figsize=(12, 6))
    plt.bar(range(num_layers), effects, color='steelblue', alpha=0.7)
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Causal Effect (L2 Distance)', fontsize=12)
    plt.title('Causal Importance of Each Layer', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    return effects

sentence = "The trophy didn't fit in the suitcase because it was too big."
effects = test_causal_importance(model, tokenizer, sentence)

```

### 4.3 Activation Patching

**More Precise Intervention:**
Instead of adding noise, *replace* activations from a corrupted context.

```python
def activation_patching(model, tokenizer, clean_sentence, corrupted_sentence, 
                       patch_layer, patch_position):
    """
    Replace activation at a specific position with activation from corrupted run.
    
    This reveals what specific information is necessary.
    """
    inputs_clean = tokenizer(clean_sentence, return_tensors='pt')
    inputs_corrupt = tokenizer(corrupted_sentence, return_tensors='pt')
    
    # Get corrupted activation
    with torch.no_grad():
        corrupt_outputs = model(**inputs_corrupt, output_hidden_states=True)
        corrupt_activation = corrupt_outputs.hidden_states[patch_layer][0, patch_position, :]
    
    # Patch into clean run
    patched_output = None
    
    def patching_hook(module, input, output):
        nonlocal patched_output
        # Replace specific position
        output = output.clone()
        output[0, patch_position, :] = corrupt_activation
        patched_output = output
        return output
    
    layer = model.encoder.layer[patch_layer]
    handle = layer.register_forward_hook(patching_hook)
    
    with torch.no_grad():
        final_outputs = model(**inputs_clean, output_hidden_states=True)
        final_hidden = final_outputs.last_hidden_state
    
    handle.remove()
    
    return final_hidden

# Example: Does corrupting subject information affect verb prediction?
clean = "The cat sits on the mat."
corrupt = "The dog sits on the mat."  # Different subject

patched = activation_patching(model, tokenizer, clean, corrupt, 
                              patch_layer=3, patch_position=2)  # Position of "cat"

```

---

## 5. Integrated Analysis Workflow

### 5.1 Complete Interpretability Pipeline

```python
class TransformerInterpreter:
    """
    Complete toolkit for transformer interpretability.
    """
    def __init__(self, model_name='bert-base-uncased'):
        self.model = BertModel.from_pretrained(model_name, output_attentions=True, 
                                               output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model.eval()
    
    def analyze_sentence(self, sentence):
        """
        Complete analysis: attention, probing, and causal tracing.
        """
        print("="*60)
        print(f"ANALYZING: {sentence}")
        print("="*60)
        
        # 1. Attention Roll-Out
        print("\n1. ATTENTION ROLL-OUT")
        rollout, tokens = self.attention_rollout(sentence)
        
        # 2. Layer-wise Probing
        print("\n2. LAYER-WISE INFORMATION")
        info = self.probe_layers(sentence)
        
        # 3. Causal Importance
        print("\n3. CAUSAL IMPORTANCE")
        effects = self.causal_trace(sentence)
        
        return {
            'attention': rollout,
            'tokens': tokens,
            'layer_info': info,
            'causal_effects': effects
        }
    
    def attention_rollout(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors='pt')
        outputs = self.model(**inputs)
        
        attentions = [attn[0] for attn in outputs.attentions]
        rollout = compute_attention_rollout(attentions)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        return rollout, tokens
    
    def probe_layers(self, sentence):
        # Implement probing for multiple properties
        pass
    
    def causal_trace(self, sentence):
        # Implement causal interventions
        pass

# Usage
interpreter = TransformerInterpreter()
results = interpreter.analyze_sentence(
    "The trophy didn't fit in the suitcase because it was too big."
)

```

---

## 6. Key Takeaways

### Attention Roll-Out

✅ Shows end-to-end information flow through all layers
✅ Reveals which tokens influence final representations
✅ Helps understand coreference, long-range dependencies

### Probing Classifiers

✅ Decode what information each layer encodes
✅ Reveals hierarchical processing (syntax → semantics)
✅ Simple classifiers = information is explicitly represented

### Causal Tracing

✅ Identifies which components are actually *used*
✅ Distinguishes correlation from causation
✅ Reveals critical layers and positions

### Integration

- **Attention Roll-Out**: "How does information flow?"
- **Probing**: "What information is present?"
- **Causal Tracing**: "What information is used?"

Together, they provide a complete picture of transformer internals.

---

## 7. Further Reading

**Papers:**

- "Attention is Not Explanation" (Jain & Wallace, 2019)
- "What Does BERT Look At?" (Clark et al., 2019)
- "Transformer Feed-Forward Layers Are Key-Value Memories" (Geva et al., 2021)
- "Locating and Editing Factual Associations in GPT" (Meng et al., 2022)

**Tools:**

- BertViz: Attention visualization
- Captum: Interpretability for PyTorch models
- AllenNLP Interpret: Suite of interpretation tools

---

**End of Lecture Notes**

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
          

[19. CNN Fundamentals - Suman - 25 Feb](64%20Interpreting%20Transformers%20-%20Suman%20-%204%20Apr%202026/19%20CNN%20Fundamentals%20-%20Suman%20-%2025%20Feb%2034386329dfd581c48c26e396cbee9b58.md)