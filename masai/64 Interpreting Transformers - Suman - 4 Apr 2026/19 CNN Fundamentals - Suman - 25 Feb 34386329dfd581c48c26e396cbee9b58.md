# 19. CNN Fundamentals - Suman - 25 Feb

# Session: CNN Fundamentals – Lecture Notes

## PPT File: [Click Here](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/63f9022c-cbc0-40c3-8d73-054dbdd05fae/iJTwygmQRhvNocOH.pdf)

## Convolution, Kernels, Stride, Pooling, Cats vs Dogs CNN

**Program:** AIM101 – IITP-AIML-2506 | **Institution:** Vishlesan i-Hub IIT Patna × Masai School

**Session Type:** Instructor (Hands-on + Theory) ·

---

## Mental Map (Session Outline)

```
CNN Fundamentals
├─ A: Motivation & Why CNNs beat dense nets for images
├─ B: Mathematical definition of 2D convolution (single-channel)
├─ C: Multi-channel convolutions & filters
├─ D: Stride, Padding (VALID vs SAME), Output shape formula
├─ E: Pooling (max, average) & global pooling
├─ F: Building block: Conv -> BatchNorm -> ReLU -> Pool
├─ G: Cats vs Dogs CNN — architecture, training tips, pitfalls
└─ H: Practical debugging & common mistakes

```

---

# A. Why CNNs (Deep Intuition)

- Spatial locality: nearby pixels more correlated than distant ones.
- Parameter sharing: same kernel applied across image reduces parameters vs dense layers.
- Hierarchical features: early layers detect edges, later layers detect motifs and class-specific parts (ears, snout).

**Illustration:** Flattening a 128×128×3 image gives 49k inputs — connecting to 1k neurons => 49M parameters (inefficient).

---

# B. 2D Convolution (Single-channel) — Step-by-step

**Operation:** Place a filter (kernel) `F×F` on top of the input patch `F×F`, do element-wise multiply and sum → single scalar.

**Code (NumPy pseudocode):**

```python
def conv2d_single_channel(input, kernel, stride=1):
    H, W = input.shape
    F, _ = kernel.shape
    out_h = (H - F)//stride + 1
    out_w = (W - F)//stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = input[i*stride:i*stride+F, j*stride:j*stride+F]
            out[i,j] = np.sum(patch * kernel)
    return out

```

**Visualization:** sliding 3×3 kernel over 2D array → produces feature map.

---

# C. Multi-Channel Convolution & Filters

If input shape = `H×W×C_in` and we want `C_out` output channels, each filter is `F×F×C_in` and there are `C_out` such filters. The convolution per output channel is sum of channel-wise convolutions + bias → result `H_out × W_out × C_out`.

**Parameter count for one conv layer:** `C_out * (C_in * F * F) + C_out (biases)`

**Example:** Input `32×32×3`, Conv(64 filters, 3×3) → params = 64*(3*3*3)+64 = 64*27+64 = 1792

---

# D. Stride & Padding (Practical rules)

- **Stride (S):** step at which kernel moves. Larger stride reduces feature map size by factor approximately `S` and may lose fine-grained info.
- **Padding (P):** zero pads around input. Useful to control output size.

`VALID` ⇒ `P=0`
`SAME` ⇒ choose `P` so `Out = ceil(N/S)`

**Output formula recap:** `Out = floor((N + 2P - F) / S) + 1`

**Examples:**

- `N=28, F=3, S=1, P=1 (SAME)` ⇒ Out = 28
- `N=28, F=5, S=2, P=0` ⇒ Out = floor((28-5)/2)+1 = 12

---

# E. Pooling Layers

**MaxPooling(2×2, stride=2)** typical to downsample by 2.

**AveragePooling** keeps average; rarely used in middle layers.

**GlobalAveragePooling** collapses spatial dims and is useful before final dense layer for classification (reduces parameters).

**Trade-offs:** Pooling reduces spatial resolution and computation but can remove small but important details. Use carefully (or consider strided conv instead).

---

# F. Common Block Patterns

A common block used in CNNs:

```
Conv2D(Filters=k, kernel=3x3, stride=1, padding='same') -> BatchNorm -> ReLU -> MaxPool(2x2)

```

BatchNorm helps training stability and faster convergence. Dropout may be used in dense layers to reduce overfitting.

---

# G. Cats vs Dogs — Case Study (Design walk-through)

**Dataset:** Kaggle Dogs vs Cats (binary labels). Typical images range 128×128 to 256×256. We will assume `128×128×3` input for speed in examples.

**Recommended baseline architecture (compact):**

1. `Conv2D(32, 3x3, stride=1, padding='same')` → ReLU → `MaxPool(2x2)`  → Output shape `64×64×32`
2. `Conv2D(64, 3x3, stride=1, padding='same')` → ReLU → `MaxPool(2x2)`  → Output shape `32×32×64`
3. `Conv2D(128, 3x3, stride=1, padding='same')` → ReLU → `MaxPool(2x2)` → Output shape `16×16×128`
4. `Conv2D(256, 3x3)` → ReLU → `GlobalAvgPool()` → Output `256`
5. `Dense(128)` → ReLU → `Dropout(0.5)`
6. `Dense(1)` → Sigmoid (binary output)

**Why this design?**

- Progressive channel growth (32→64→128→256) increases representational power while pooling reduces spatial dimensions.
- GlobalAvgPool reduces parameters drastically before the dense layer, improves generalization.
- Use data augmentation (random flip, rotation, zoom), and early stopping during training to combat overfitting.

---

# H. Training & Practical Tips

- **Loss:** Binary Crossentropy (for single-unit Sigmoid).
- **Optimizer:** Adam with lr ≈ 1e-4 (tune).
- **Batch size:** 32 or 64, depending on GPU memory.
- **Augmentation:** Horizontal flip, random crops, color jitter.
- **Class imbalance:** use balanced batches or class weights if imbalance exists.
- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC (for production readiness).

---

# I. Debugging Checklist (Common Mistakes)

1. **Shape mismatch** when flattening before dense layers — check shapes after each layer.
2. **Wrong activation/loss pairing** — use sigmoid + binary crossentropy (or softmax + categorical crossentropy for multi-class).
3. **Vanishing gradients** for very deep networks — use BatchNorm and careful initialization.
4. **Overfitting fast** — add augmentation, dropout, or reduce model capacity.
5. **Inference mismatch** — ensure `training=False` for layers like BatchNorm/Dropout during evaluation.

---

# J. Example: Numerical Convolution (Worked example)

Input (single-channel, 5×5):

```
[[1, 2, 0, 1, 0],
 [0, 1, 3, 1, 2],
 [2, 1, 0, 0, 1],
 [1, 2, 1, 3, 0],
 [0, 1, 2, 1, 1]]

```

Kernel (3×3):

```
[[ 1, 0, -1],
 [ 1, 0, -1],
 [ 1, 0, -1]]

```

Compute top-left output (position 0,0): element-wise product + sum of the 3×3 patch → result value. Repeat for stride and the defined output map.

(Students: do this by hand for 2 positions to internalize the math.)

---

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