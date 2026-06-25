# 34. Neural Nets Fundamentals - Varun Raste - 6 Jan 2026

### In-Class Resoures: [23_Neural_Networks_Fundamentals](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/2be0c288-bc69-499c-a51d-e0a050d5df1a/KJRixFaIHNBwma0b.pdf)

# Neural Network Basics - Lecture Notes

*Prerequisites: Comfort with vector dot products, taking derivatives of simple functions (power rule + chain rule), and writing short Python scripts with NumPy or pandas.*

**Learning outcomes**

- Explain how weights and biases encode patterns from data and compute parameter counts for a given multilayer perceptron.
- Execute forward propagation and backpropagation on a two-layer network and interpret each intermediate tensor.
- Build, train, and evaluate a small MLP in TensorFlow and in PyTorch, including reading loss curves and gradient norms.

---

## 1. Build the Mental Model: Stacking Simple Functions

A neural network is a composition of simple functions. Each layer computes a weighted sum, adds a bias, applies an activation, and hands its output to the next layer. Stacking those transformations lets the model represent highly non-linear relationships while remaining differentiable so gradients can flow.

> **Tip:** Keep the mental picture of "inputs -> weighted sum -> activation -> output." You can debug many issues by stepping through that loop one layer at a time.
> 

**How the pieces fit**

1. Inputs form the initial vector (or batch matrix).
2. Each layer multiplies by its weight matrix and adds a bias vector.
3. Non-linear activations keep the representation expressive.
4. The final layer delivers logits or probabilities, depending on your task.

### Check your understanding

1. Why does a network need activation functions even if linear layers are powerful?
2. What happens to expressiveness if every layer used only linear operations?

Suggested answers

Without activations the entire network collapses to a single linear transformation, so it cannot model curved or piecewise relationships.
Stacking linear layers is equivalent to one giant matrix multiply; the model cannot bend decision boundaries and will underfit most real problems.

## 2. Parameters: Weights and Biases in Context

Weights shape how strongly each input feature contributes to a neuron's pre-activation value. Biases shift that activation threshold, allowing neurons to fire even when inputs are near zero. During training, gradient descent adjusts both to minimize loss.

Component | Shape (for layer with n inputs, m outputs) | Role
Weight matrixW | n x m | Rotates and scales input features
Bias vectorb | m | Shifts activations before non-linearity

> **Remember:** Bias gradients accumulate from every example in the batch. If they all push in the same direction, the bias quickly moves, even when weight updates are small.
> 

### Worked example: Counting parameters

Suppose you build a network with 3 inputs, a hidden layer of 4 neurons (ReLU), and an output layer of 2 neurons (softmax).

1. Input to hidden: `3 weights * 4 neurons = 12`, plus `4` biases.
2. Hidden to output: `4 weights * 2 neurons = 8`, plus `2` biases.
3. Total trainable parameters: `12 + 4 + 8 + 2 = 26`.

### Check your understanding

1. If you remove biases from the hidden layer, how many parameters remain?
2. Why does adding a new feature to the input layer change parameters throughout downstream layers?

Suggested answers

You keep `12` input-to-hidden weights, `8` hidden-to-output weights, and `2` output biases: `22` total.
Every neuron in the next layer gets an additional incoming weight, so the parameter matrix grows by the number of neurons in that layer.

## 3. Forward Propagation Step by Step

Forward propagation computes outputs by applying the network function to the current parameters.

### Manual walkthrough

For one sample `x`, the hidden layer pre-activation is `z1 = W1 x + b1`. After applying ReLU, `h1 = ReLU(z1)`. The output logits are `z2 = W2 h1 + b2`. You then pass `z2` through a softmax (for classification) or leave it as-is for regression.

### Code walkthrough (NumPy)

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def forward_pass(x, params):
    W1, b1, W2, b2 = params
    z1 = x @ W1 + b1            # shape: (batch_size, hidden)
    h1 = relu(z1)
    z2 = h1 @ W2 + b2           # shape: (batch_size, outputs)
    return z1, h1, z2

# Demo with a tiny batch
x_batch = np.array([[0.5, -1.0, 2.0], [1.5, 0.2, -0.3]])
params = (
    np.array([[0.2, -0.4], [0.7, 0.5], [-0.3, 0.8]]),  # W1
    np.array([0.1, -0.2]),                             # b1
    np.array([[0.6, -0.1], [-0.5, 0.9]]),              # W2
    np.array([0.0, 0.05])                              # b2
)
z1, h1, z2 = forward_pass(x_batch, params)
print(z1)
print(h1)
print(z2)

```

> **Caution:** Shape mismatches often come from transposing matrices incorrectly. Check that each matrix multiply uses `(batch, features)` @ `(features, neurons)` so dimensions align.
> 

### Check your understanding

1. What would happen to `h1` if every element of `z1` were negative when using ReLU?
2. How do you convert logits from `z2` into probabilities?

Suggested answers

ReLU would zero out the entire hidden layer, so gradients could disappear and the model would momentarily stop learning.
Apply `softmax = exp(z2) / sum(exp(z2))` along the output dimension to create probabilities.

## 4. Backpropagation: Letting Error Flow Back

Backpropagation uses the chain rule to compute gradients of the loss with respect to each parameter. It starts at the output, multiplies gradients through activations and weights, and accumulates partial derivatives layer by layer.

### Manual gradient example

Consider a single training example with mean squared error loss. The output layer uses identity activation.

1. Forward pass results: `z1`, `h1`, `z2`.
2. Output gradient: `dL/dz2 = 2 * (z2 - y_true)`.
3. Gradient w.r.t. `W2`: `h1.T @ dL/dz2`.
4. Gradient w.r.t. `b2`: equal to `dL/dz2` (sum over batch).
5. Propagate to hidden layer: `dL/dh1 = dL/dz2 @ W2.T`.
6. Apply activation derivative: `dL/dz1 = dL/dh1 * ReLU'(z1)` where `ReLU'(z1)` is `1` when `z1 > 0`, else `0`.
7. Gradients for `W1`: `x.T @ dL/dz1`.
8. Gradients for `b1`: sum of `dL/dz1` across the batch.

```python
def backward_pass(x, y_true, params):
    W1, b1, W2, b2 = params
    z1, h1, z2 = forward_pass(x, params)
    y_pred = z2  # identity activation for simplicity
    dz2 = 2.0 * (y_pred - y_true)
    dW2 = h1.T @ dz2
    db2 = dz2.sum(axis=0)
    dh1 = dz2 @ W2.T
    dz1 = dh1 * (z1 > 0).astype(float)
    dW1 = x.T @ dz1
    db1 = dz1.sum(axis=0)
    return (dW1, db1, dW2, db2)

```

> **Note:** This implementation omits averaging over the batch. Add `/ batch_size` before applying the optimizer so updates remain scale-invariant.
> 

### Check your understanding

1. Why do we zero out gradients through a ReLU when its input was negative?
2. Which matrices get transposed during backprop and why?

Suggested answers

ReLU's slope is zero for negative inputs, meaning changes to earlier weights do not affect the output when the neuron is inactive.
We transpose weight matrices to align dimensions when distributing gradients backward: e.g., `dL/dh1 = dL/dz2 @ W2.T`.

## 5. Training Loop Essentials

A typical mini-batch gradient descent loop:

1. Sample a batch of inputs and targets.
2. Run forward propagation to compute predictions.
3. Evaluate the loss function.
4. Execute backpropagation to compute gradients.
5. Update parameters with an optimizer (SGD, Adam).
6. Track metrics and repeat for many epochs.

```python
for epoch in range(num_epochs):
    for xb, yb in dataloader:
        z1, h1, logits = forward_pass(xb, params)
        loss = criterion(logits, yb)
        grads = backward_pass(xb, yb, params)
        params = optimizer_step(params, grads, lr)

```

> **Caution:** Always reset or zero optimizer state (like momentum buffers) only when reinitializing training. Forgetting to zero PyTorch gradients via `optimizer.zero_grad()` retains gradients from previous batches.
> 

### Check your understanding

How would you detect that your learning rate is too high using only the loss curve?

Suggested answer
If the loss spikes erratically or fails to decrease after several epochs, the learning rate is likely too high. You may also see divergence toward infinity.

## 6. Building a Small MLP in TensorFlow

We will classify two interleaving half-moons.

```python
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=2000, noise=0.25, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss="binary_crossentropy",
              metrics=["accuracy"])

history = model.fit(X_train, y_train, batch_size=32, epochs=100,
                    validation_data=(X_val, y_val), verbose=0)

tf.keras.utils.plot_model(model, show_shapes=True)

```

**What to observe**

- Training accuracy should surpass 95% quickly; validation should be similar if noise is moderate.
- If loss plateaus early, try lowering the learning rate or adding one more hidden unit.
- `history.history` stores loss curves you can plot to diagnose convergence.

### Check your understanding

What hyperparameter change would you try first if validation accuracy lags far behind training accuracy?

Suggested answer
Reduce model capacity (fewer neurons) or add regularization such as `tf.keras.layers.Dropout` to combat overfitting.

## 7. Building a Small MLP in PyTorch

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=2000, noise=0.25, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)

scaler = StandardScaler().fit(X_train)
X_train = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
X_val = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256)

class MoonsMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = MoonsMLP()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(100):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val).item()
            val_acc = ((val_preds > 0.5) == y_val).float().mean().item()
        print(f"Epoch {epoch+1}: val_loss={val_loss:.3f}, val_acc={val_acc:.3f}")

```

> **Tip:** Track gradient norms with `model.layer.weight.grad.norm()` to confirm training is healthy. Zero or exploding norms signal numerical issues.
> 

### Check your understanding

How would you switch the final layer to output logits instead of probabilities?

Suggested answer
Remove `nn.Sigmoid()` from `nn.Sequential`, use `nn.BCEWithLogitsLoss()`, and interpret outputs via `torch.sigmoid` only when reporting probabilities.

## 8. Common Pitfalls and Debugging Patterns

- **Vanishing gradients:** Deep networks with sigmoid or tanh can stall. Try ReLU or residual connections.
- **Exploding gradients:** Watch for large gradient norms; gradient clipping or smaller learning rates help.
- **Mismatched shapes:** Logits and targets must align (`(batch, classes)` vs `(batch,)`).
- **Data leakage:** Standardize using training split statistics only.
- **Initialization traps:** Start with Xavier/He initialization; avoid all-zero weights which break symmetry.

## Practice Task: Compare TensorFlow and PyTorch on Tabular Data

Goal: Classify whether a breast cancer tumor is malignant.

1. Load `sklearn.datasets.load_breast_cancer`. Split into 70% train, 15% validation, 15% test with stratification.
2. Standardize features with `StandardScaler` fit on the training set.
3. Build the same architecture in both TensorFlow and PyTorch: two hidden layers (32 -> 16 neurons) with ReLU, dropout 0.2 after each hidden layer, and a single sigmoid output.
4. Train for up to 100 epochs with early stopping on validation loss (patience 10).
5. Report final metrics: validation accuracy, test accuracy, ROC-AUC, and confusion matrix.
6. Plot training vs. validation loss for both frameworks on the same chart.

Deliverables: notebook or script, loss plots, metric table, and short commentary (3 bullets) comparing the frameworks' training experience.

### Solution outline

- Set seeds for reproducibility (`tf.random.set_seed`, `torch.manual_seed`, `np.random.seed`).
- Use `ModelCheckpoint` + `EarlyStopping` in Keras; in PyTorch, track the best state dict manually.
- For PyTorch, implement a `validate` function to reuse across epochs.
- Compute ROC-AUC using `sklearn.metrics.roc_auc_score`.
- Summarize findings: convergence speed, logging convenience, any metric gaps.

## Additional resources

- Stanford CS231n notes on backpropagation
- TensorFlow Playground: [https://playground.tensorflow.org](https://playground.tensorflow.org)
- PyTorch autograd mechanics: [https://pytorch.org/docs/stable/autograd.html](https://pytorch.org/docs/stable/autograd.html)

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