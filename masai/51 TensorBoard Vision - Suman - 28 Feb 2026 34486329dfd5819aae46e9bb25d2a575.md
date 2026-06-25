# 51. TensorBoard Vision - Suman - 28 Feb 2026

# TensorBoard Vision: Lecture Notes

## Colab File: [Click Here](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/027aaf0a-cca1-4a98-b836-8152856e705e/LeWTKaNGanREkVYS.zip)

**Program:** Vishlesan i-Hub IIT Patna × Masai School — AIM (AI & Machine Learning)

**Topic:** TensorBoard Vision - Visualizing Deep Learning Models

**Subtopics:** Filters, Activations, Embeddings, PR Curves

---

## Table of Contents

1. Introduction to TensorBoard
2. Filter Visualization
3. Activation Visualization
4. Embedding Visualization
5. Precision-Recall Curves
6. Practical Implementation
7. Best Practices
8. Common Pitfalls

---

## 1. Introduction to TensorBoard

### What is TensorBoard?

TensorBoard is TensorFlow's visualization toolkit that provides a suite of web-based tools for:

- Understanding model architecture
- Debugging neural networks
- Monitoring training progress
- Visualizing learned representations
- Analyzing model performance

### Why Visualization Matters

**The Black Box Problem:**

```
Input → [Neural Network ???] → Output

```

Deep learning models are often called "black boxes" because their internal workings are opaque. Visualization helps us:

- **Debug models** — Identify where learning fails
- **Build intuition** — Understand what models learn
- **Communicate insights** — Explain models to stakeholders
- **Improve performance** — Diagnose bottlenecks

### Setting Up TensorBoard

```python
# Installation
pip install tensorboard

# Basic setup
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

# Create log directory
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

# Create TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,  # Log histograms every epoch
    write_graph=True,   # Visualize model graph
    write_images=True,  # Log model weights as images
    update_freq='epoch',
    profile_batch='500,520'  # Profile batches 500-520
)

# Use callback during training
model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_val, y_val),
    callbacks=[tensorboard_callback]
)

```

**Launch TensorBoard:**

```bash
tensorboard --logdir logs/fit
# Open browser to http://localhost:6006

```

---

## 2. Filter Visualization

### What Are Filters?

**Convolutional Filters (Kernels)** are small matrices that scan across images to detect patterns:

```
Input Image (28×28×1)  →  Conv Layer (3×3 filter)  →  Feature Map

Example 3×3 Edge Detection Filter:
┌─────────┐
│ -1  -1  -1 │
│  0   0   0 │
│  1   1   1 │
└─────────┐

Detects horizontal edges

```

### Why Visualize Filters?

**Early layers** learn low-level features:

- Edges (horizontal, vertical, diagonal)
- Corners
- Color blobs
- Simple textures

**Deep layers** learn high-level features:

- Object parts (eyes, wheels, fur)
- Complex textures (bricks, scales)
- Abstract patterns

### Visualizing Learned Filters

**Method 1: Direct Visualization**

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_filters(model, layer_name, num_filters=64):
    """
    Visualize convolutional filters from a specific layer
    """
    # Get the layer
    layer = model.get_layer(name=layer_name)
    filters = layer.get_weights()[0]  # Shape: (height, width, in_channels, out_channels)
    
    # Normalize filters to [0, 1]
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    
    # Plot filters
    n_cols = 8
    n_rows = num_filters // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*1.5, n_rows*1.5))
    
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            # Get filter (take first input channel for visualization)
            filter_img = filters[:, :, 0, i]
            ax.imshow(filter_img, cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Filter {i}')
    
    plt.tight_layout()
    plt.savefig('filters_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

# Example usage
model = keras.models.load_model('cnn_model.h5')
visualize_filters(model, layer_name='conv2d_1', num_filters=32)

```

**Method 2: TensorBoard Integration**

```python
import io
import tensorflow as tf
from tensorflow import keras

def log_filters_to_tensorboard(model, layer_name, log_dir, step=0):
    """
    Log filter visualizations to TensorBoard
    """
    # Create file writer
    file_writer = tf.summary.create_file_writer(log_dir + '/filters')
    
    # Get filters
    layer = model.get_layer(name=layer_name)
    filters = layer.get_weights()[0]  # (H, W, in_channels, out_channels)
    
    # Normalize
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    
    # Take first input channel, arrange filters in grid
    filter_grid = filters[:, :, 0, :]  # (H, W, num_filters)
    filter_grid = tf.expand_dims(filter_grid, axis=0)  # (1, H, W, num_filters)
    
    # Log to TensorBoard
    with file_writer.as_default():
        tf.summary.image(
            f"{layer_name}_filters",
            filter_grid,
            step=step,
            max_outputs=64  # Show up to 64 filters
        )

# Usage after training
log_filters_to_tensorboard(model, 'conv2d_1', 'logs/filters', step=0)

```

### Interpreting Filter Visualizations

**What to Look For:**

1. 
**Random Noise (Bad Sign)**
`░▒▓█▓▒░
▓█░▒▓█▒
░▒▓█░▒▓

Problem: Model didn't learn meaningful patterns
Solutions: Train longer, check learning rate, add regularization`

2. 
**Edge Detectors (Good Sign)**
`Layer 1 Filters:
- Horizontal edges: ━━━━
- Vertical edges:   ┃
- Diagonal edges:   ╱ ╲

Indicates: Model learning low-level features correctly`

3. 
**Gabor-like Patterns (Excellent Sign)**
`Layer 2-3 Filters:
- Oriented edges at multiple scales
- Texture patterns
- Corner detectors

Indicates: Hierarchical feature learning`

4. 
**Dead Filters (Warning Sign)**
`All zeros: ░░░░░
          ░░░░░
          ░░░░░

Problem: Filter never activated, wasted capacity
Solutions: Lower learning rate, check initialization, reduce model size`

---

## 3. Activation Visualization

### What Are Activations?

**Activations** are the outputs of neurons after applying filters and activation functions:

```
Input Image
    ↓
[Conv Layer + ReLU]  ← Filter application
    ↓
Activation Map  ← What we visualize
    ↓
[Pooling]
    ↓
Next Layer

```

### Why Visualize Activations?

1. **Understand what the model "sees"**
2. **Debug dead neurons** (always zero)
3. **Identify which parts of input activate neurons**
4. **Verify hierarchical learning** (simple → complex features)

### Activation Visualization Techniques

### Technique 1: Feature Map Visualization

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def visualize_activations(model, layer_name, input_image):
    """
    Visualize activation maps for a specific layer given an input image
    
    Args:
        model: Keras model
        layer_name: Name of layer to visualize
        input_image: Input image (preprocessed)
    """
    # Create a model that outputs activations from specified layer
    activation_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    
    # Get activations
    activations = activation_model.predict(input_image[np.newaxis, ...])
    # Shape: (1, height, width, num_filters)
    
    # Remove batch dimension
    activations = activations[0]
    
    # Plot activations
    num_filters = activations.shape[-1]
    n_cols = 8
    n_rows = (num_filters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            # Get activation map
            activation_map = activations[:, :, i]
            
            # Plot
            ax.imshow(activation_map, cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Filter {i}', fontsize=8)
        else:
            ax.axis('off')
    
    plt.suptitle(f'Activations: {layer_name}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return activations

# Example usage
img = load_and_preprocess_image('cat.jpg')  # Your preprocessing
activations = visualize_activations(model, 'conv2d_2', img)

```

### Technique 2: Activation Maximization

**Goal:** Find input that maximally activates a specific filter

```python
def activation_maximization(model, layer_name, filter_index, steps=50, step_size=1.0):
    """
    Generate an input image that maximally activates a specific filter
    
    This helps understand what pattern a filter is looking for
    """
    # Create a model that outputs the activation of the target filter
    layer = model.get_layer(layer_name)
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=layer.output)
    
    # Initialize with random noise
    input_img = tf.Variable(tf.random.uniform((1, 224, 224, 3)))
    
    # Optimization loop
    for step in range(steps):
        with tf.GradientTape() as tape:
            # Forward pass
            activation = feature_extractor(input_img)
            
            # Loss: negative mean activation of target filter
            # (negative because we want to maximize)
            filter_activation = activation[:, :, :, filter_index]
            loss = -tf.reduce_mean(filter_activation)
        
        # Compute gradients
        gradients = tape.gradient(loss, input_img)
        
        # Normalize gradients
        gradients /= tf.math.reduce_std(gradients) + 1e-8
        
        # Update input image
        input_img.assign_add(gradients * step_size)
        
        # Clip to valid range
        input_img.assign(tf.clip_by_value(input_img, 0, 255))
    
    # Convert to displayable image
    img = input_img.numpy()[0]
    img = np.clip(img / 255.0, 0, 1)
    
    return img

# Visualize what different filters are looking for
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    img = activation_maximization(model, 'conv2d_3', filter_index=i)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'Filter {i}')
plt.tight_layout()
plt.show()

```

### Technique 3: Grad-CAM (Class Activation Mapping)

**Goal:** Highlight which parts of the image are important for a prediction

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def make_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap
    
    Args:
        model: Keras model
        img_array: Preprocessed input image
        last_conv_layer_name: Name of last convolutional layer
        pred_index: Class index to visualize (None = predicted class)
    
    Returns:
        heatmap: Grad-CAM heatmap (0-1 range)
    """
    # Create model that outputs last conv layer and predictions
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute gradient of predicted class with respect to conv layer output
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        # Get the score for the predicted class
        class_channel = predictions[:, pred_index]
    
    # Gradient of the class score with respect to conv output
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight each channel by the gradient importance
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads.numpy()
    
    # Weighted sum of channels
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image
    """
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose heatmap on image
    superimposed_img = heatmap * alpha + img * (1 - alpha)
    superimposed_img = np.uint8(superimposed_img)
    
    return superimposed_img

# Example usage
img = load_and_preprocess_image('dog.jpg')
img_array = np.expand_dims(img, axis=0)

# Get prediction
preds = model.predict(img_array)
pred_class = np.argmax(preds[0])

# Generate Grad-CAM
heatmap = make_gradcam_heatmap(model, img_array, 'conv5_block3_out', pred_index=pred_class)

# Display
original_img = load_image('dog.jpg')  # Load without preprocessing
gradcam_img = display_gradcam(original_img, heatmap)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(original_img)
ax1.set_title('Original Image')
ax1.axis('off')

ax2.imshow(gradcam_img)
ax2.set_title(f'Grad-CAM (Pred: Class {pred_class})')
ax2.axis('off')

plt.tight_layout()
plt.show()

```

### Interpreting Activation Visualizations

**Layer 1 (Early Layers):**

```
✓ Activation maps show edges, colors, simple patterns
✓ Most filters activate on different parts of image
✗ If all blank: Model not learning, check data preprocessing

```

**Layer 3-4 (Middle Layers):**

```
✓ Activation maps show textures, object parts
✓ Fewer but larger activated regions
✓ Some filters specialize (e.g., one for fur, one for eyes)

```

**Layer 5+ (Deep Layers):**

```
✓ Activation maps highlight entire objects or complex patterns
✓ Very few filters activate per image (sparse activation)
✓ Semantically meaningful responses (activates on "dog" concept)

```

**Warning Signs:**

```
⚠ Dead neurons: Always output 0 (all activations black)
⚠ Saturated neurons: Always max value (all activations white)
⚠ Identical activations: Multiple filters produce same output

```

---

## 4. Embedding Visualization

### What Are Embeddings?

**Embeddings** are dense, low-dimensional representations of high-dimensional data:

```
Input Space                  Embedding Space
(High-dimensional)          (Low-dimensional)

Image (224×224×3)     →     Vector (512-d)
    150,528 dims                 512 dims

Word "cat"            →     Vector (300-d)
   (One-hot: 50,000)            300 dims

```

**Properties of Good Embeddings:**

- **Similar items are close together** in embedding space
- **Dissimilar items are far apart**
- **Semantic relationships preserved** (e.g., king - man + woman ≈ queen)

### Why Visualize Embeddings?

1. **Verify model learns meaningful representations**
2. **Identify clusters** (natural groupings in data)
3. **Debug classification issues** (overlapping classes)
4. **Explore data relationships** interactively

### TensorBoard Projector

TensorBoard's Projector provides interactive 3D visualization of embeddings.

**Step 1: Extract Embeddings**

```python
import tensorflow as tf
import numpy as np
from tensorflow import keras

def extract_embeddings(model, data, embedding_layer_name):
    """
    Extract embeddings from a specific layer for all data samples
    
    Args:
        model: Keras model
        data: Input data (images, text, etc.)
        embedding_layer_name: Name of embedding layer
    
    Returns:
        embeddings: Numpy array of shape (num_samples, embedding_dim)
    """
    # Create embedding extraction model
    embedding_model = keras.Model(
        inputs=model.input,
        outputs=model.get_layer(embedding_layer_name).output
    )
    
    # Extract embeddings
    embeddings = embedding_model.predict(data, batch_size=32, verbose=1)
    
    # If output is 2D+ (e.g., conv layer), flatten
    if len(embeddings.shape) > 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    
    return embeddings

# Example: Extract embeddings from penultimate layer
X_test, y_test = load_test_data()  # Your test data
embeddings = extract_embeddings(model, X_test, embedding_layer_name='dense_1')

print(f"Embeddings shape: {embeddings.shape}")
# Output: Embeddings shape: (10000, 128)

```

**Step 2: Log to TensorBoard**

```python
import io
import os
import tensorflow as tf
from tensorboard.plugins import projector

def log_embeddings_to_tensorboard(embeddings, labels, images=None, log_dir='logs/embeddings'):
    """
    Log embeddings to TensorBoard Projector
    
    Args:
        embeddings: Numpy array (num_samples, embedding_dim)
        labels: List of labels (num_samples,)
        images: Optional, sprite image or list of images
        log_dir: Directory to save logs
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Save embeddings as TensorFlow checkpoint
    embedding_var = tf.Variable(embeddings, name='embeddings')
    checkpoint = tf.train.Checkpoint(embedding=embedding_var)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))
    
    # Save labels metadata
    with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
        f.write("Label\n")  # Header
        for label in labels:
            f.write(f"{label}\n")
    
    # Configure projector
    config = projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding_config.metadata_path = 'metadata.tsv'
    
    # Optional: Add sprite images
    if images is not None:
        # Create sprite image (grid of all images)
        sprite_path = os.path.join(log_dir, 'sprite.png')
        create_sprite_image(images, sprite_path)
        
        embedding_config.sprite.image_path = 'sprite.png'
        embedding_config.sprite.single_image_dim.extend([28, 28])  # Image size
    
    # Save projector config
    projector.visualize_embeddings(log_dir, config)
    
    print(f"Embeddings saved to {log_dir}")
    print(f"Run: tensorboard --logdir {log_dir}")

def create_sprite_image(images, save_path, sprite_size=(28, 28)):
    """
    Create a sprite image (grid of all images) for TensorBoard
    """
    import math
    from PIL import Image
    
    num_images = len(images)
    grid_size = int(math.ceil(math.sqrt(num_images)))
    
    # Create blank sprite canvas
    sprite_height = grid_size * sprite_size[0]
    sprite_width = grid_size * sprite_size[1]
    sprite = Image.new('L', (sprite_width, sprite_height))
    
    # Paste images into grid
    for i, img in enumerate(images):
        row = i // grid_size
        col = i % grid_size
        
        # Convert to PIL image if needed
        if not isinstance(img, Image.Image):
            img = Image.fromarray((img * 255).astype('uint8'))
        
        # Resize to sprite size
        img = img.resize(sprite_size)
        
        # Paste into sprite
        x = col * sprite_size[0]
        y = row * sprite_size[1]
        sprite.paste(img, (x, y))
    
    sprite.save(save_path)

# Usage example
embeddings = extract_embeddings(model, X_test, 'dense_1')
labels = y_test.argmax(axis=1) if len(y_test.shape) > 1 else y_test

log_embeddings_to_tensorboard(
    embeddings=embeddings,
    labels=labels,
    images=X_test[:1000],  # First 1000 images for sprite
    log_dir='logs/embeddings'
)

```

**Step 3: Explore in TensorBoard**

```bash
tensorboard --logdir logs/embeddings
# Navigate to: http://localhost:6006/#projector

```

**TensorBoard Projector Features:**

1. 
**3D Visualization**

Rotate, zoom, pan
PCA, t-SNE, UMAP projections

2. 
**Search & Select**

Search by label
Select points to inspect
Find nearest neighbors

3. 
**Coloring**

Color by label
Color by metadata

4. 
**Bookmarking**

Save interesting views
Export configurations

### Dimensionality Reduction for Visualization

**Why Reduce Dimensions?**

- Embeddings are high-dimensional (512-d, 1024-d)
- Humans can visualize 2D or 3D
- Need to project high-dimensional space → 2D/3D

**Common Techniques:**

### 1. PCA (Principal Component Analysis)

**Linear projection** that preserves variance

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_embeddings_pca(embeddings, labels, n_components=2):
    """
    Visualize embeddings using PCA
    """
    # Apply PCA
    pca = PCA(n_components=n_components)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            label=f'Class {label}',
            alpha=0.6,
            s=20
        )
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA Projection of Embeddings')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Usage
visualize_embeddings_pca(embeddings, labels)

```

**Pros:**

- Fast
- Deterministic (same result every time)
- Preserves global structure

**Cons:**

- Linear (may miss non-linear relationships)
- May not separate clusters well

### 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Non-linear projection** that preserves local structure

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings_tsne(embeddings, labels, perplexity=30, n_iter=1000):
    """
    Visualize embeddings using t-SNE
    """
    # Apply t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=42,
        verbose=1
    )
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            label=f'Class {label}',
            alpha=0.7,
            s=30,
            edgecolors='w',
            linewidths=0.5
        )
    
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f't-SNE Projection (perplexity={perplexity})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Usage
visualize_embeddings_tsne(embeddings, labels, perplexity=30)

```

**Pros:**

- Excellent at revealing clusters
- Preserves local neighborhoods
- Beautiful visualizations

**Cons:**

- Slow (O(n²) complexity)
- Non-deterministic (different runs give different results)
- Sensitive to hyperparameters (perplexity)
- Doesn't preserve global structure (distances between clusters meaningless)

### 3. UMAP (Uniform Manifold Approximation and Projection)

**Modern alternative** to t-SNE with better properties

```python
# Install: pip install umap-learn
import umap
import matplotlib.pyplot as plt

def visualize_embeddings_umap(embeddings, labels, n_neighbors=15, min_dist=0.1):
    """
    Visualize embeddings using UMAP
    """
    # Apply UMAP
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        verbose=True
    )
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            label=f'Class {label}',
            alpha=0.7,
            s=20
        )
    
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title(f'UMAP Projection (n_neighbors={n_neighbors}, min_dist={min_dist})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Usage
visualize_embeddings_umap(embeddings, labels, n_neighbors=15, min_dist=0.1)

```

**Pros:**

- Faster than t-SNE
- Better preservation of global structure
- More consistent results
- Scales better to large datasets

**Cons:**

- Still non-deterministic
- Sensitive to hyperparameters

### Interpreting Embedding Visualizations

**Good Embedding Space:**

```
✓ Clear clusters for each class
✓ Tight within-class groupings
✓ Large between-class separation
✓ Similar classes are nearby (e.g., cats and dogs closer than cats and cars)

```

**Poor Embedding Space:**

```
✗ Overlapping classes (classification will fail)
✗ Scattered points (no clear structure)
✗ Outliers (potential data quality issues)
✗ Unexpected groupings (model learned wrong features)

```

**Example Interpretations:**

```python
# Scenario 1: Perfect Separation
"""
Class A: ●●●●●●
Class B:          ◆◆◆◆◆◆
Class C:                    ▲▲▲▲▲▲

Interpretation: Model learned excellent discriminative features
Action: Model is ready for deployment
"""

# Scenario 2: Overlapping Classes
"""
Class A: ●●●●●●
Class B:    ●●◆◆◆◆
Class C:       ◆◆▲▲▲▲

Interpretation: Classes A & B have overlap, B & C have overlap
Action: Collect more distinguishing features, or accept confusion
"""

# Scenario 3: Subgroups within Class
"""
Class A: ●●●●●●        ●●●●●●
         (group 1)    (group 2)

Interpretation: Class A has two distinct subgroups
Action: Investigate - might be different subtypes worth separating
"""

# Scenario 4: Outliers
"""
Class A: ●●●●●●●●●●    ●
                       ↑
                     outlier

Interpretation: One sample very different from others
Action: Check if mislabeled, corrupted data, or legitimate rare case
"""

```

---

## 5. Precision-Recall Curves

### What Are Precision-Recall Curves?

**Precision-Recall (PR) curves** visualize the trade-off between precision and recall at different classification thresholds.

**Definitions:**

```
Precision = TP / (TP + FP)
"Of all positive predictions, how many were correct?"

Recall = TP / (TP + FN)
"Of all actual positives, how many did we catch?"

Threshold: Probability cutoff for classifying as positive
(Usually 0.5 by default, but can be tuned)

```

### Why Use PR Curves?

**Better than accuracy for imbalanced data:**

```python
# Example: Fraud detection (0.1% fraud rate)
Scenario 1: Always predict "Not Fraud"
Accuracy: 99.9% (excellent!)
Recall: 0% (terrible! Misses all fraud)

Scenario 2: Model with tuned threshold
Accuracy: 99.2% (slightly lower)
Recall: 85% (much better! Catches most fraud)

→ Accuracy misleading, PR curve reveals true performance

```

### Precision-Recall vs ROC Curve

Metric | When to Use
PR Curve | Imbalanced data, focus on positive class
ROC Curve | Balanced data, care about both classes equally

**Example:**

```
Fraud Detection (0.1% fraud):
✓ Use PR curve (focus on catching fraud)
✗ ROC curve inflated by large number of true negatives

Medical Diagnosis (50% disease rate):
✓ Use ROC curve (balanced classes)
✓ PR curve also works

```

### Creating PR Curves in TensorBoard

```python
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_recall_curve
import io
import matplotlib.pyplot as plt

def log_pr_curve_to_tensorboard(y_true, y_pred_proba, log_dir, step=0, class_name=''):
    """
    Log Precision-Recall curve to TensorBoard
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities for positive class
        log_dir: TensorBoard log directory
        step: Training step/epoch
        class_name: Name of the class (for multi-class)
    """
    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot PR curve
    ax.plot(recall, precision, linewidth=2, label='PR Curve')
    ax.fill_between(recall, 0, precision, alpha=0.2)
    
    # Add random classifier baseline
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(baseline, linestyle='--', color='red', 
               label=f'Random Classifier (AP={baseline:.3f})')
    
    # Calculate Average Precision (area under PR curve)
    from sklearn.metrics import average_precision_score
    ap_score = average_precision_score(y_true, y_pred_proba)
    
    # Formatting
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve{" - " + class_name if class_name else ""}\nAP = {ap_score:.3f}', 
                 fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    # Save figure to TensorBoard
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    # Convert to image tensor
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    
    # Log to TensorBoard
    file_writer = tf.summary.create_file_writer(log_dir)
    with file_writer.as_default():
        tf.summary.image(f'PR_Curve{"_" + class_name if class_name else ""}', 
                        image, step=step)
    
    return precision, recall, thresholds, ap_score

# Example usage after training
y_true_test = y_test  # Ground truth labels
y_pred_proba = model.predict(X_test)[:, 1]  # Predicted probabilities

log_pr_curve_to_tensorboard(
    y_true=y_true_test,
    y_pred_proba=y_pred_proba,
    log_dir='logs/pr_curves',
    step=0,
    class_name='Fraud'
)

```

### Multi-Class PR Curves

For multi-class classification, create one PR curve per class:

```python
def log_multiclass_pr_curves(y_true, y_pred_proba, class_names, log_dir, step=0):
    """
    Log PR curves for each class in multi-class classification
    
    Args:
        y_true: True labels (integers 0 to num_classes-1)
        y_pred_proba: Predicted probabilities (num_samples, num_classes)
        class_names: List of class names
        log_dir: TensorBoard log directory
        step: Training step/epoch
    """
    num_classes = len(class_names)
    
    # One-vs-rest PR curves
    for class_idx, class_name in enumerate(class_names):
        # Binary labels: current class vs all others
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_class = y_pred_proba[:, class_idx]
        
        # Log PR curve for this class
        _, _, _, ap = log_pr_curve_to_tensorboard(
            y_true=y_true_binary,
            y_pred_proba=y_pred_class,
            log_dir=log_dir,
            step=step,
            class_name=class_name
        )
        
        print(f"Class {class_name}: AP = {ap:.4f}")

# Example usage
y_true = np.argmax(y_test, axis=1)  # Convert one-hot to integers
y_pred_proba = model.predict(X_test)
class_names = ['Cat', 'Dog', 'Bird', 'Fish']

log_multiclass_pr_curves(
    y_true=y_true,
    y_pred_proba=y_pred_proba,
    class_names=class_names,
    log_dir='logs/pr_curves',
    step=0
)

```

### Interpreting PR Curves

**Perfect Classifier:**

```
Precision
1.0 |●●●●●●●●●●●
    |           ●
    |            ●
0.5 |             ●
    |              ●
0.0 |_______________●___ Recall
    0.0            1.0

→ High precision AND high recall at all thresholds
→ AP (Average Precision) = 1.0

```

**Good Classifier:**

```
Precision
1.0 |●●●●●●
    |      ●●●
    |         ●●●
0.5 |            ●●●
    |               ●●●
0.0 |__________________●●_ Recall
    0.0                1.0

→ Gradual trade-off between precision and recall
→ AP ≈ 0.85

```

**Poor Classifier:**

```
Precision
1.0 |
    |
    |  ●
0.5 |___●●●●●●●●●●●●●●●
    |
0.0 |___________________ Recall
    0.0              1.0

→ Precision drops quickly as recall increases
→ AP ≈ 0.3 (barely better than random)

```

**Random Classifier:**

```
Precision
1.0 |
    |
 P₀ |━━━━━━━━━━━━━━━━━  (P₀ = positive class frequency)
    |
0.0 |___________________ Recall
    0.0              1.0

→ Flat line at baseline
→ AP = P₀ (e.g., 0.1 for 10% positive class)

```

### Key Metrics from PR Curves

**1. Average Precision (AP)**

```python
from sklearn.metrics import average_precision_score

ap = average_precision_score(y_true, y_pred_proba)
print(f"Average Precision: {ap:.3f}")

# Interpretation:
# AP = 1.0: Perfect
# AP > 0.9: Excellent
# AP > 0.7: Good
# AP > 0.5: Fair
# AP < 0.5: Poor (worse than random if imbalanced)

```

**2. F1 Score at Optimal Threshold**

```python
from sklearn.metrics import f1_score

# Find threshold that maximizes F1
precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"F1 score at optimal threshold: {optimal_f1:.3f}")

# Use optimal threshold for predictions
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

```

**3. Precision at Fixed Recall**

```python
# Example: Find precision at 90% recall
target_recall = 0.90

# Find threshold that gives closest recall to target
recall_diff = np.abs(recall - target_recall)
idx = np.argmin(recall_diff)

precision_at_90_recall = precision[idx]
threshold_at_90_recall = thresholds[idx]

print(f"Precision at {target_recall*100}% recall: {precision_at_90_recall:.3f}")
print(f"Required threshold: {threshold_at_90_recall:.3f}")

```

### Logging PR Curves During Training

Create a custom callback to log PR curves every epoch:

```python
class PRCurveCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to log PR curves to TensorBoard during training
    """
    def __init__(self, X_val, y_val, log_dir, class_names=None):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.log_dir = log_dir
        self.class_names = class_names
    
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions
        y_pred_proba = self.model.predict(self.X_val, verbose=0)
        
        if len(y_pred_proba.shape) == 1 or y_pred_proba.shape[1] == 1:
            # Binary classification
            y_true = self.y_val
            y_pred = y_pred_proba.flatten()
            
            log_pr_curve_to_tensorboard(
                y_true=y_true,
                y_pred_proba=y_pred,
                log_dir=self.log_dir,
                step=epoch
            )
        else:
            # Multi-class classification
            y_true = np.argmax(self.y_val, axis=1)
            
            log_multiclass_pr_curves(
                y_true=y_true,
                y_pred_proba=y_pred_proba,
                class_names=self.class_names or [f'Class_{i}' for i in range(y_pred_proba.shape[1])],
                log_dir=self.log_dir,
                step=epoch
            )

# Usage during training
pr_callback = PRCurveCallback(
    X_val=X_val,
    y_val=y_val,
    log_dir='logs/pr_curves',
    class_names=['Cat', 'Dog', 'Bird']
)

model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[pr_callback, tensorboard_callback]
)

```

---

## 6. Practical Implementation

### Complete TensorBoard Integration Example

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime
import os

# ============================================
# 1. SETUP LOGGING DIRECTORIES
# ============================================

def create_log_structure(base_dir='logs'):
    """
    Create organized directory structure for TensorBoard logs
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base_dir, f'run_{timestamp}')
    
    directories = {
        'base': run_dir,
        'training': os.path.join(run_dir, 'training'),
        'validation': os.path.join(run_dir, 'validation'),
        'filters': os.path.join(run_dir, 'filters'),
        'activations': os.path.join(run_dir, 'activations'),
        'embeddings': os.path.join(run_dir, 'embeddings'),
        'pr_curves': os.path.join(run_dir, 'pr_curves')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

# ============================================
# 2. CUSTOM TENSORBOARD CALLBACK
# ============================================

class ComprehensiveTensorBoardCallback(keras.callbacks.Callback):
    """
    Custom callback that logs everything to TensorBoard:
    - Training/validation metrics
    - Filters
    - Activations
    - Embeddings
    - PR curves
    """
    def __init__(self, log_dirs, X_val, y_val, log_freq=5):
        super().__init__()
        self.log_dirs = log_dirs
        self.X_val = X_val
        self.y_val = y_val
        self.log_freq = log_freq  # Log visualizations every N epochs
        
        # Create file writers
        self.train_writer = tf.summary.create_file_writer(log_dirs['training'])
        self.val_writer = tf.summary.create_file_writer(log_dirs['validation'])
    
    def on_epoch_end(self, epoch, logs=None):
        # Log training metrics
        with self.train_writer.as_default():
            for name, value in logs.items():
                if 'val_' not in name:
                    tf.summary.scalar(name, value, step=epoch)
        
        # Log validation metrics
        with self.val_writer.as_default():
            for name, value in logs.items():
                if 'val_' in name:
                    tf.summary.scalar(name.replace('val_', ''), value, step=epoch)
        
        # Log visualizations periodically
        if (epoch + 1) % self.log_freq == 0:
            print(f"\n[Epoch {epoch+1}] Logging visualizations to TensorBoard...")
            
            # 1. Log filters
            self._log_filters(epoch)
            
            # 2. Log sample activations
            self._log_activations(epoch)
            
            # 3. Log embeddings
            self._log_embeddings(epoch)
            
            # 4. Log PR curves
            self._log_pr_curves(epoch)
    
    def _log_filters(self, epoch):
        """Log convolutional filters"""
        for layer in self.model.layers:
            if 'conv' in layer.name:
                filters = layer.get_weights()[0]  # (H, W, in_channels, out_channels)
                
                # Normalize
                f_min, f_max = filters.min(), filters.max()
                filters_norm = (filters - f_min) / (f_max - f_min + 1e-8)
                
                # Take first input channel
                filter_images = filters_norm[:, :, 0, :]  # (H, W, num_filters)
                filter_images = tf.expand_dims(filter_images, 0)  # (1, H, W, num_filters)
                
                # Log to TensorBoard
                file_writer = tf.summary.create_file_writer(self.log_dirs['filters'])
                with file_writer.as_default():
                    tf.summary.image(
                        f"{layer.name}_filters",
                        filter_images,
                        step=epoch,
                        max_outputs=64
                    )
    
    def _log_activations(self, epoch):
        """Log activation maps for sample images"""
        # Select a few sample images
        sample_indices = np.random.choice(len(self.X_val), size=5, replace=False)
        sample_images = self.X_val[sample_indices]
        
        # Get activations from each conv layer
        for layer in self.model.layers:
            if 'conv' in layer.name:
                activation_model = keras.Model(
                    inputs=self.model.input,
                    outputs=layer.output
                )
                activations = activation_model.predict(sample_images, verbose=0)
                
                # activations shape: (num_samples, H, W, num_filters)
                # Take first sample, first few filters
                activation_sample = activations[0, :, :, :16]  # First 16 filters
                activation_sample = tf.expand_dims(activation_sample, 0)
                
                # Log
                file_writer = tf.summary.create_file_writer(self.log_dirs['activations'])
                with file_writer.as_default():
                    tf.summary.image(
                        f"{layer.name}_activations",
                        activation_sample,
                        step=epoch,
                        max_outputs=16
                    )
    
    def _log_embeddings(self, epoch):
        """Log embeddings from penultimate layer"""
        # Get embedding layer (second to last layer)
        embedding_layer = self.model.layers[-2]
        embedding_model = keras.Model(
            inputs=self.model.input,
            outputs=embedding_layer.output
        )
        
        # Extract embeddings
        embeddings = embedding_model.predict(self.X_val[:1000], verbose=0)
        
        # Flatten if needed
        if len(embeddings.shape) > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
        
        # Log to TensorBoard Projector
        from tensorboard.plugins import projector
        
        embedding_var = tf.Variable(embeddings, name='embeddings')
        checkpoint = tf.train.Checkpoint(embedding=embedding_var)
        checkpoint_path = os.path.join(self.log_dirs['embeddings'], f'embedding_epoch_{epoch}.ckpt')
        checkpoint.save(checkpoint_path)
        
        # Save labels
        labels = np.argmax(self.y_val[:1000], axis=1) if len(self.y_val.shape) > 1 else self.y_val[:1000]
        metadata_path = os.path.join(self.log_dirs['embeddings'], 'metadata.tsv')
        with open(metadata_path, 'w') as f:
            f.write("Label\n")
            for label in labels:
                f.write(f"{label}\n")
        
        # Configure projector
        config = projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding_config.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(self.log_dirs['embeddings'], config)
    
    def _log_pr_curves(self, epoch):
        """Log Precision-Recall curves"""
        # Get predictions
        y_pred_proba = self.model.predict(self.X_val, verbose=0)
        
        if len(y_pred_proba.shape) == 1 or y_pred_proba.shape[1] == 1:
            # Binary classification
            y_true = self.y_val
            y_pred = y_pred_proba.flatten()
            
            log_pr_curve_to_tensorboard(
                y_true=y_true,
                y_pred_proba=y_pred,
                log_dir=self.log_dirs['pr_curves'],
                step=epoch
            )
        else:
            # Multi-class classification
            y_true = np.argmax(self.y_val, axis=1)
            
            log_multiclass_pr_curves(
                y_true=y_true,
                y_pred_proba=y_pred_proba,
                class_names=[f'Class_{i}' for i in range(y_pred_proba.shape[1])],
                log_dir=self.log_dirs['pr_curves'],
                step=epoch
            )

# ============================================
# 3. COMPLETE TRAINING PIPELINE
# ============================================

def train_with_comprehensive_logging(model, X_train, y_train, X_val, y_val, epochs=50):
    """
    Train model with comprehensive TensorBoard logging
    """
    # Create log directories
    log_dirs = create_log_structure()
    
    print(f"TensorBoard logs will be saved to: {log_dirs['base']}")
    print(f"Run: tensorboard --logdir {log_dirs['base']}")
    
    # Create callbacks
    callbacks = [
        # Standard TensorBoard callback for basic logging
        keras.callbacks.TensorBoard(
            log_dir=log_dirs['training'],
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        
        # Custom comprehensive callback
        ComprehensiveTensorBoardCallback(
            log_dirs=log_dirs,
            X_val=X_val,
            y_val=y_val,
            log_freq=5  # Log visualizations every 5 epochs
        ),
        
        # Model checkpoint
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(log_dirs['base'], 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history, log_dirs

# ============================================
# 4. EXAMPLE USAGE
# ============================================

# Load data (example)
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Create model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), name='conv2d_1'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_2'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_3'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu', name='dense_1'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax', name='output')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train with comprehensive logging
history, log_dirs = train_with_comprehensive_logging(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_test,
    y_val=y_test,
    epochs=50
)

print(f"\nTraining complete!")
print(f"Launch TensorBoard: tensorboard --logdir {log_dirs['base']}")
print(f"Open browser to: http://localhost:6006")

```

---

## 7. Best Practices

### When to Visualize

**During Development:**

- ✅ Visualize filters after first epoch (check if learning started)
- ✅ Visualize activations on validation set periodically
- ✅ Monitor embeddings every 10-20 epochs
- ✅ Check PR curves every epoch for imbalanced data

**During Debugging:**

- ✅ Visualize when validation loss stops decreasing
- ✅ Check activations when model predictions are wrong
- ✅ Inspect embeddings when classes are confused
- ✅ Analyze PR curves when metrics are poor

**Before Deployment:**

- ✅ Final filter visualization (ensure no dead filters)
- ✅ Embedding visualization on test set (check generalization)
- ✅ PR curves on production-representative data
- ✅ Grad-CAM on edge cases

### Visualization Frequency

```python
# Recommended logging frequencies
logging_config = {
    'metrics': 'every_batch',      # Loss, accuracy
    'histograms': 'every_epoch',   # Weight distributions
    'filters': 'every_5_epochs',   # Filter visualizations
    'activations': 'every_5_epochs',  # Activation maps
    'embeddings': 'every_10_epochs',  # Embedding projections
    'pr_curves': 'every_epoch',    # For imbalanced data
    'grad_cam': 'on_demand'        # Only when analyzing specific images
}

```

### Storage Management

TensorBoard logs can become large:

```python
# Estimate log sizes
estimates = {
    'metrics': '~1 MB per 100 epochs',
    'histograms': '~10 MB per 100 epochs',
    'filters': '~5 MB per visualization',
    'activations': '~20 MB per visualization',
    'embeddings': '~50 MB per projection (1000 samples)',
    'images': '~1 MB per 100 images'
}

# Best practices:
# 1. Log embeddings on subset (1000-5000 samples, not entire dataset)
# 2. Log activations for a few sample images (5-10), not all validation set
# 3. Archive old runs after experiments complete
# 4. Use --reload_multifile flag when launching TensorBoard for large logs

```

### Multi-Experiment Comparison

```python
# Organize experiments hierarchically
log_structure = """
logs/
├── experiment_1_baseline/
│   ├── run_1/
│   ├── run_2/
│   └── run_3/
├── experiment_2_deeper_model/
│   ├── run_1/
│   ├── run_2/
│   └── run_3/
└── experiment_3_augmentation/
    ├── run_1/
    └── run_2/
"""

# Launch TensorBoard to compare all experiments
# tensorboard --logdir logs/
# Or compare specific experiments:
# tensorboard --logdir_spec=baseline:logs/experiment_1_baseline,deep:logs/experiment_2_deeper_model

```

---

## 8. Common Pitfalls

### Pitfall 1: Not Normalizing Visualizations

**Problem:**

```python
# Bad: Filters with different ranges
filter_A: min=-2.5, max=0.8
filter_B: min=-0.1, max=0.15

# When plotted, filter_B appears all gray (no contrast)

```

**Solution:**

```python
# Good: Normalize each filter individually
def normalize_filter(filter_array):
    f_min, f_max = filter_array.min(), filter_array.max()
    return (filter_array - f_min) / (f_max - f_min + 1e-8)

# Apply to each filter separately
for i in range(num_filters):
    filter_norm = normalize_filter(filters[:, :, :, i])
    ax.imshow(filter_norm)

```

### Pitfall 2: Logging Too Much Data

**Problem:**

```python
# Bad: Log embeddings for entire dataset every epoch
embeddings = extract_embeddings(model, X_train)  # 50,000 samples × 512 dims
# File size: 50,000 × 512 × 4 bytes = 100 MB per epoch
# After 100 epochs: 10 GB of embedding logs!

```

**Solution:**

```python
# Good: Log subset of data
sample_size = 2000
indices = np.random.choice(len(X_train), sample_size, replace=False)
embeddings = extract_embeddings(model, X_train[indices])
# File size: 2000 × 512 × 4 bytes = 4 MB per epoch
# After 100 epochs: 400 MB (manageable)

```

### Pitfall 3: Misinterpreting t-SNE

**Problem:**

```python
# t-SNE visualization shows three clusters
# Student concludes: "My model learned three categories!"
# Reality: Dataset has 10 classes, model is confused

```

**Important Notes:**

- t-SNE can create clusters even from random data
- Distances between clusters are meaningless (only within-cluster distances matter)
- Different random seeds give different visualizations
- Perplexity parameter greatly affects results

**Solution:**

```python
# Always:
# 1. Color points by true labels to verify clusters match classes
# 2. Try multiple perplexity values (5, 30, 50)
# 3. Compare to PCA (does it show similar structure?)
# 4. Run multiple times with different random seeds

```

### Pitfall 4: Ignoring Dead Filters

**Problem:**

```python
# Student sees 50% of filters are black (all zeros)
# Student ignores this and continues training
# Result: Wasted model capacity, poor performance

```

**Solution:**

```python
# Detect dead filters
def count_dead_filters(model, layer_name, threshold=1e-6):
    layer = model.get_layer(layer_name)
    filters = layer.get_weights()[0]
    
    num_filters = filters.shape[-1]
    dead_count = 0
    
    for i in range(num_filters):
        filter_max = np.abs(filters[:, :, :, i]).max()
        if filter_max < threshold:
            dead_count += 1
    
    percent_dead = (dead_count / num_filters) * 100
    print(f"{layer_name}: {dead_count}/{num_filters} dead ({percent_dead:.1f}%)")
    
    return dead_count

# If >20% filters are dead:
# - Lower learning rate
# - Use He initialization for ReLU
# - Add batch normalization
# - Reduce model size

```

### Pitfall 5: Wrong Interpretation of Grad-CAM

**Problem:**

```python
# Grad-CAM highlights background, not the object
# Student thinks: "Model is broken!"
# Reality: Model learned spurious correlation (e.g., "boats appear on water")

```

**Correct Interpretation:**

```
Grad-CAM shows what the model uses for prediction, not what it should use

If Grad-CAM highlights wrong regions:
✓ Check if those regions correlate with the label in training data
✓ This reveals dataset bias, not necessarily model failure
✓ Consider data augmentation to remove spurious correlations

```

### Pitfall 6: Comparing Models Without Standardization

**Problem:**

```python
# Compare two models on embeddings
# Model A: embeddings mean=0.5, std=2.0
# Model B: embeddings mean=10.0, std=0.1
# Use same t-SNE parameters → Unfair comparison

```

**Solution:**

```python
# Standardize embeddings before visualization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
embeddings_A_scaled = scaler.fit_transform(embeddings_A)
embeddings_B_scaled = scaler.fit_transform(embeddings_B)

# Now apply t-SNE with same parameters

```

---

## Summary

### Key Takeaways

1. 
**Filter Visualization**

Shows what patterns CNN layers learn
Early layers: edges, colors
Deep layers: complex features
Watch for dead filters (all zeros)

2. 
**Activation Visualization**

Shows how the model responds to specific inputs
Grad-CAM reveals important image regions
Activation maximization shows ideal inputs for filters

3. 
**Embedding Visualization**

Reveals learned representations
Good embeddings: clear clusters, separated classes
Use t-SNE or UMAP for 2D visualization
TensorBoard Projector for interactive exploration

4. 
**PR Curves**

Essential for imbalanced data
Shows precision-recall trade-off
Average Precision (AP) summarizes performance
Log to TensorBoard for monitoring

### Recommended Workflow

```python
# 1. Setup comprehensive logging
log_dirs = create_log_structure()

# 2. Train with visualization callbacks
history = model.fit(
    X_train, y_train,
    callbacks=[
        TensorBoardCallback(log_dir=log_dirs['training']),
        FilterVisualizationCallback(log_freq=5),
        EmbeddingCallback(log_freq=10),
        PRCurveCallback(log_freq=1)
    ]
)

# 3. Launch TensorBoard
# tensorboard --logdir logs/

# 4. Monitor during training:
#    - Filters: Are features emerging?
#    - Activations: Are all layers active?
#    - Embeddings: Are classes separating?
#    - PR curves: Is performance improving?

# 5. Debug based on visualizations:
#    - Dead filters → Adjust learning rate
#    - Overlapping embeddings → Add capacity or features
#    - Poor PR curve → Address class imbalance

# 6. Before deployment:
#    - Final embedding visualization (test set)
#    - PR curve on production-like data
#    - Grad-CAM on edge cases

```

### Resources

**TensorBoard Documentation:**

- [https://www.tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard)

**Visualization Papers:**

- "Visualizing and Understanding Convolutional Networks" (Zeiler & Fergus, 2014)
- "Grad-CAM: Visual Explanations from Deep Networks" (Selvaraju et al., 2017)
- "How to Use t-SNE Effectively" (Wattenberg et al., 2016)

**Libraries:**

- TensorFlow: [https://www.tensorflow.org](https://www.tensorflow.org)
- Scikit-learn: [https://scikit-learn.org](https://scikit-learn.org)
- UMAP: [https://umap-learn.readthedocs.io](https://umap-learn.readthedocs.io)

---

**End of Lecture Notes**

*Vishlesan i-Hub IIT Patna × Masai School — AIM (AI & Machine Learning)*

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