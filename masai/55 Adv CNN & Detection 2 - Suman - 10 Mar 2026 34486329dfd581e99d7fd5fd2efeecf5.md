# 55. Adv CNN & Detection 2 - Suman - 10 Mar 2026

# Advanced CNN & Object Detection: SSD, YOLO

## PPT File: [Click Here](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/253ccc7a-2b15-44f0-867b-ddfea878e13b/CgHgilNuerXLEtgp.pdf)

**Program:** Vishlesan i-Hub IIT Patna x Masai School — AIM (AI & Machine Learning)

**Duration:** 3 hours (180 minutes)

**Topics:** Object Detection, R-CNN Family, YOLO, SSD, Anchor Boxes, NMS, mAP

---

## Session Structure

- **Part 1:** Object Detection Fundamentals & R-CNN Evolution (30 min)
- **Part 2:** YOLO Architecture Deep Dive (35 min)
- **Part 3:** SSD Multi-Scale Detection (35 min)
- **Part 4:** Anchor Boxes & Matching Strategy (25 min)
- **Part 5:** Non-Maximum Suppression (20 min)
- **Part 6:** Evaluation Metrics - Precision, Recall, mAP (35 min)

---

# PART 1: OBJECT DETECTION FUNDAMENTALS (30 minutes)

## 1.1 From Classification to Detection

### The Computer Vision Hierarchy

**Level 1: Image Classification**

```
Task: What is in this image?

Input: 224×224×3 image
Architecture: CNN (e.g., ResNet-50)
Output: Class probabilities [0.05, 0.02, 0.89, ...]

Example:
Image → ResNet → [Dog: 0.89, Cat: 0.05, Bird: 0.02, ...]
Answer: "Dog" ✓

Limitation: No spatial information
- Where is the dog?
- How big is the dog?
- How many dogs?

```

---

**Level 2: Object Localization**

```
Task: What + Where (single object)

Input: 224×224×3 image
Architecture: CNN + Regression head
Output: 
  - Class: "Dog"
  - Bounding box: [x, y, width, height]

Example:
Image → CNN → Class: "Dog", Box: [50, 80, 150, 200]

Limitation: Assumes ONE primary object
Can't handle multiple objects ✗

```

---

**Level 3: Object Detection**

```
Task: What + Where (multiple objects)

Input: Any size image
Architecture: Detection network (YOLO, SSD)
Output: List of detections
  - [class, confidence, x, y, w, h]

Example:
Image → Detector → [
  ("Dog", 0.95, 50, 80, 150, 200),
  ("Cat", 0.87, 300, 120, 100, 180),
  ("Person", 0.92, 180, 30, 200, 400)
]

Perfect! Multiple objects detected ✓

```

---

### Bounding Box Representations

**Format 1: (x, y, width, height)**

```
x, y: Top-left corner coordinates
width, height: Box dimensions

Example:
Box: [100, 50, 200, 150]
  Top-left: (100, 50)
  Bottom-right: (100+200, 50+150) = (300, 200)
  
Used in: YOLO training, COCO annotations

```

---

**Format 2: (x_min, y_min, x_max, y_max)**

```
x_min, y_min: Top-left corner
x_max, y_max: Bottom-right corner

Example:
Box: [100, 50, 300, 200]
  Width: 300 - 100 = 200
  Height: 200 - 50 = 150

Used in: PASCAL VOC, most visualization libraries

```

---

**Format 3: (center_x, center_y, width, height)**

```
center_x, center_y: Box center
width, height: Box dimensions

Example:
Box: [200, 125, 200, 150]
  Center: (200, 125)
  Top-left: (200-100, 125-75) = (100, 50)
  Bottom-right: (300, 200)

Used in: YOLO predictions, anchor boxes

```

---

**Conversion Between Formats:**

```python
def xywh_to_xyxy(box):
    """Convert (x, y, w, h) to (x_min, y_min, x_max, y_max)"""
    x, y, w, h = box
    return [x, y, x + w, y + h]

def xyxy_to_xywh(box):
    """Convert (x_min, y_min, x_max, y_max) to (x, y, w, h)"""
    x_min, y_min, x_max, y_max = box
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def xywh_to_cxcywh(box):
    """Convert (x, y, w, h) to (center_x, center_y, w, h)"""
    x, y, w, h = box
    return [x + w/2, y + h/2, w, h]

def cxcywh_to_xywh(box):
    """Convert (center_x, center_y, w, h) to (x, y, w, h)"""
    cx, cy, w, h = box
    return [cx - w/2, cy - h/2, w, h]

```

---

## 1.2 R-CNN Family Evolution

### R-CNN (2014): The Foundation

**Architecture:**

```
Input: Image

Step 1: Selective Search
  Algorithm generates ~2000 region proposals
  Based on:
    - Color similarity
    - Texture similarity
    - Size constraints
    - Shape compatibility
  
  Output: 2000 candidate boxes

Step 2: Warp Regions
  For each region:
    - Extract image patch
    - Resize to 227×227 (AlexNet input size)
    - Ignore aspect ratio (distortion!)

Step 3: CNN Feature Extraction
  For each warped region:
    - Pass through AlexNet
    - Extract fc7 features (4096-dim vector)
  
  Total: 2000 × AlexNet forward passes!

Step 4: SVM Classification
  For each region:
    - 4096-dim features → SVM classifier
    - Output: Class scores for C classes
    - Keep if score > threshold

Step 5: Bounding Box Regression
  For positive detections:
    - Refine box coordinates
    - Linear regression: Δx, Δy, Δw, Δh
    - Improves localization

Output: Filtered detections with refined boxes

```

---

**R-CNN Problems:**

```
1. Extremely Slow:
   - 2000 CNN forward passes per image
   - 47 seconds per image on GPU
   - Impossible for real-time ✗

2. Multi-Stage Training:
   - Train CNN (ImageNet)
   - Fine-tune CNN (detection data)
   - Train SVMs
   - Train box regressors
   - Cannot be trained end-to-end ✗

3. Large Storage:
   - Must cache features for all regions
   - Hundreds of GB for training set ✗

4. Fixed Region Proposals:
   - Selective Search is not learned
   - Suboptimal regions ✗

```

---

### Fast R-CNN (2015): Single CNN Pass

**Key Innovation:** Compute CNN features ONCE for entire image!

**Architecture:**

```
Input: Image + Region Proposals (still from Selective Search)

Step 1: Entire Image Through CNN
  Image → VGG-16 → Feature map (H×W×C)
  
  Example: 
    Input: 600×800×3
    Output: 37×50×512 (after conv5)
  
  ONE forward pass! ✓

Step 2: ROI Pooling
  For each region proposal:
    a) Project onto feature map
       If region is [160, 240, 320, 480] in original image
       Scale: 160/16=10, 240/16=15, 320/16=20, 480/16=30
       ROI on feature map: [10, 15, 20, 30]
    
    b) Divide ROI into 7×7 grid
       Each cell: max pooling
    
    c) Output: 7×7×512 feature tensor
       Then flatten to 7×7×512 = 25,088-dim vector

Step 3: Fully Connected Layers
  25,088-dim → fc6 (4096-dim) → fc7 (4096-dim)

Step 4: Two Heads
  Head 1: Classification
    fc7 → fc_cls → Softmax → C+1 class probabilities
    (+1 for background class)
  
  Head 2: Bounding Box Regression
    fc7 → fc_bbox → 4×C outputs
    For each class: Δx, Δy, Δw, Δh

Output: Class + refined box for each ROI

```

---

**ROI Pooling Detailed Example:**

```
ROI on feature map: [10, 15, 20, 30] (x, y, w, h)

Target output: 7×7 grid

Step 1: Divide ROI into 7×7 bins
  Bin width: 20/7 ≈ 2.86
  Bin height: 30/7 ≈ 4.29

Step 2: For each bin (i, j):
  x_start = 10 + i × 2.86
  y_start = 15 + j × 4.29
  x_end = 10 + (i+1) × 2.86
  y_end = 15 + (j+1) × 4.29
  
  Bin region: [x_start:x_end, y_start:y_end]

Step 3: Max pool within each bin
  Take maximum activation value
  Output: Single value per bin per channel

Result: 7×7×512 tensor (consistent size for any ROI!)

```

---

**Fast R-CNN Improvements:**

```
Speed: 47s → 2.3s per image (20× faster!) ✓

Single-stage training:
  Multi-task loss = Classification loss + Regression loss
  Train CNN, classifier, and regressor jointly ✓

Higher accuracy:
  mAP: 66.9% (vs 62.4% for R-CNN) ✓

Remaining bottleneck:
  Selective Search: ~2 seconds per image
  Still not real-time! ✗

```

---

### Faster R-CNN (2015): Learned Proposals

**Revolutionary Idea:** Use CNN to generate region proposals!

**Architecture:**

```
Input: Image

Stage 1: Convolutional Layers (Backbone)
  Image → VGG-16/ResNet → Feature map
  Example: 600×800×3 → 37×50×512

Stage 2: Region Proposal Network (RPN)
  Feature map → RPN → Region proposals
  
  RPN Architecture:
    Input: 37×50×512 feature map
    
    3×3 conv: 37×50×512 → 37×50×512
    (spatial context)
    
    Two 1×1 conv branches:
      Branch 1: Classification
        → 37×50×(2k) objectness scores
        k = number of anchors per location
        2 = object vs background
      
      Branch 2: Regression
        → 37×50×(4k) box coordinates
        4 = (Δx, Δy, Δw, Δh) per anchor
  
  Output: ~300 proposals (after NMS)

Stage 3: ROI Pooling + Detection
  Same as Fast R-CNN
  Proposals → ROI Pool → FC → Class + refined box

Total: Fully differentiable! Train end-to-end ✓

```

---

**Anchor Boxes in RPN:**

```
At each location on feature map:
Generate k anchor boxes with different:
  - Scales: [128², 256², 512²]
  - Aspect ratios: [1:1, 1:2, 2:1]

Example: 3 scales × 3 ratios = 9 anchors per location

For 37×50 feature map:
  Total anchors: 37 × 50 × 9 = 16,650

RPN predicts for each anchor:
  1. Objectness score: p(object)
  2. Box refinement: (Δx, Δy, Δw, Δh)

Anchor with high objectness → Proposal ✓

```

---

**Anchor Box Generation:**

```python
def generate_anchors(base_size=16, scales=[8, 16, 32], 
                     ratios=[0.5, 1, 2]):
    """
    Generate anchor boxes at a single location
    
    base_size: Stride of feature map (16 for VGG)
    scales: Scale factors
    ratios: Height/width ratios
    """
    anchors = []
    
    for scale in scales:
        for ratio in ratios:
            # Compute width and height
            h = base_size * scale
            w = h * ratio
            
            # Center at (0, 0)
            x_ctr = base_size / 2
            y_ctr = base_size / 2
            
            # Convert to (x, y, w, h)
            anchor = [
                x_ctr - w/2,  # x
                y_ctr - h/2,  # y
                w,            # width
                h             # height
            ]
            anchors.append(anchor)
    
    return anchors

# Example:
anchors = generate_anchors()
# Returns 9 anchors:
# Scale 128, ratio 1:1 → 128×128
# Scale 128, ratio 1:2 → 181×90
# Scale 128, ratio 2:1 → 90×181
# ... (6 more)

```

---

**RPN Training:**

```
For each anchor:

1. Assign label:
   Positive (object) if:
     - IoU with ground truth > 0.7
     - OR highest IoU for a ground truth
   
   Negative (background) if:
     - IoU with all ground truths < 0.3
   
   Ignore if:
     - 0.3 ≤ IoU ≤ 0.7 (ambiguous)

2. Compute loss:
   L_rpn = L_cls + L_reg
   
   L_cls: Binary cross-entropy
     p = predicted objectness
     y = label (1 for object, 0 for background)
     L_cls = -[y log(p) + (1-y) log(1-p)]
   
   L_reg: Smooth L1 loss on box coordinates
     Only for positive anchors!
     t = predicted offset (Δx, Δy, Δw, Δh)
     t* = target offset
     L_reg = smooth_L1(t - t*)

3. Sampling:
   Sample 256 anchors per image
   Keep 1:1 positive:negative ratio
   (Balance classes!)

```

---

**Faster R-CNN Performance:**

```
Speed: 2.3s → 0.2s per image (10× faster!) ✓
  = 5 FPS (frames per second)

Accuracy:
  mAP: 73.2% (PASCAL VOC 2007)
  Better than Fast R-CNN! ✓

End-to-end training:
  Single loss function
  Backprop through entire network ✓

Remaining issue:
  Still not real-time (need 30+ FPS)
  Two-stage process (RPN + Detection) ✗

```

---

# PART 2: YOLO ARCHITECTURE DEEP DIVE (35 minutes)

## 2.1 YOLO Philosophy

### Single-Shot Detection

**Key Insight:** Treat detection as a SINGLE regression problem!

```
Traditional (Faster R-CNN):
  Stage 1: Generate proposals (~300)
  Stage 2: Classify each proposal
  Total: Two forward passes

YOLO:
  Stage 1: Predict everything at once
  Total: ONE forward pass! ✓

"You Only Look Once" at the image!

```

---

### Grid-Based Detection

**Core Idea:**

```
Divide image into S×S grid
Each grid cell predicts:
  - B bounding boxes
  - Confidence for each box
  - C class probabilities

Example: S=7, B=2, C=20 (PASCAL VOC)
  Output tensor: 7 × 7 × 30
  Where 30 = B×5 + C = 2×5 + 20

```

---

**Grid Cell Responsibility:**

```
Rule: Grid cell (i, j) is responsible for object if:
  → Object's center falls in cell (i, j)

Example:
Image: 448×448
Grid: 7×7
Cell size: 448/7 = 64 pixels

Dog's bounding box center: (250, 180)
Responsible cell: (floor(250/64), floor(180/64)) = (3, 2)

Cell (3,2) must:
  1. Detect the dog
  2. Predict its class
  3. Localize its bounding box

Other cells: Predict background ✓

```

---

## 2.2 YOLO Architecture

### YOLOv1 Network Architecture

```
Input: 448×448×3 image

Convolutional Layers (Feature Extraction):
  Conv 1: 448×448×3   → 224×224×64   (7×7 conv, stride 2)
  MaxPool            → 112×112×64   (2×2, stride 2)
  
  Conv 2: 112×112×64  → 112×112×192  (3×3 conv)
  MaxPool            → 56×56×192    (2×2, stride 2)
  
  Conv 3-4: 56×56×192 → 56×56×512   (1×1 and 3×3 convs)
  MaxPool            → 28×28×512    (2×2, stride 2)
  
  Conv 5-8: 28×28×512 → 28×28×1024  (1×1 and 3×3 convs)
  MaxPool            → 14×14×1024   (2×2, stride 2)
  
  Conv 9-16: 14×14×1024 → 14×14×1024 (1×1 and 3×3 convs)
  MaxPool             → 7×7×1024    (2×2, stride 2)
  
  Conv 17-20: 7×7×1024 → 7×7×1024   (3×3 convs)
  Conv 21-24: 7×7×1024 → 7×7×1024   (3×3 convs)

Fully Connected Layers (Detection):
  FC 1: 7×7×1024 → 4096 (flatten + dense)
  FC 2: 4096 → 7×7×30   (reshape to grid)

Output: 7×7×30 tensor

```

---

**Output Tensor Interpretation:**

```
Output: 7×7×30

For each grid cell (i, j):
  30 values = [Box 1 (5), Box 2 (5), Classes (20)]

Box encoding (5 values):
  1. x: Center x (relative to cell)
  2. y: Center y (relative to cell)
  3. w: Width (relative to image)
  4. h: Height (relative to image)
  5. confidence: P(object) × IoU

Class encoding (20 values):
  P(Class_i | Object) for each class

Total predictions per image:
  7 × 7 × 2 = 98 bounding boxes

```

---

**Coordinate Encoding Details:**

```
Grid cell (i, j) predicts box with:
  x, y, w, h ∈ [0, 1]

Absolute coordinates:
  x_abs = (i + x) / S
  y_abs = (j + y) / S
  w_abs = w
  h_abs = h

Where S = grid size (7)

Example:
Cell (3, 2) predicts x=0.6, y=0.4, w=0.3, h=0.5

Absolute coordinates:
  x_abs = (3 + 0.6) / 7 = 0.514 (51.4% of image width)
  y_abs = (2 + 0.4) / 7 = 0.343 (34.3% of image height)
  w_abs = 0.3 (30% of image width)
  h_abs = 0.5 (50% of image height)

In pixels (448×448 image):
  x_pixel = 0.514 × 448 = 230
  y_pixel = 0.343 × 448 = 154
  w_pixel = 0.3 × 448 = 134
  h_pixel = 0.5 × 448 = 224

Box: [230, 154, 134, 224] in image coordinates ✓

```

---

## 2.3 YOLO Training

### Loss Function

**Multi-Part Loss:**

```
L_total = L_coord + L_conf + L_class

Where:
  L_coord: Localization loss (box coordinates)
  L_conf: Confidence loss (objectness)
  L_class: Classification loss

```

---

**1. Localization Loss (L_coord):**

```
For cells containing objects:
  Sum squared error on (x, y)
  Sum squared error on (√w, √h)

Why square root for w, h?
  Small deviations in large boxes should penalize less
  than same deviations in small boxes!

Formula:
L_coord = λ_coord × Σ Σ 1_ij^obj [(x_i - x̂_i)² + (y_i - ŷ_i)²]
          + λ_coord × Σ Σ 1_ij^obj [(√w_i - √ŵ_i)² + (√h_i - √ĥ_i)²]

Where:
  λ_coord = 5 (weight for coordinate loss)
  1_ij^obj = 1 if cell i contains object, 0 otherwise
  (x_i, y_i, w_i, h_i) = predicted box
  (x̂_i, ŷ_i, ŵ_i, ĥ_i) = ground truth box

Summed over:
  - All grid cells i
  - All boxes j in cell i

```

---

**2. Confidence Loss (L_conf):**

```
Two parts:
  a) Cells WITH objects: predict C = P(object) × IoU
  b) Cells WITHOUT objects: predict C = 0

Formula:
L_conf = Σ Σ 1_ij^obj (C_i - Ĉ_i)²
         + λ_noobj × Σ Σ 1_ij^noobj (C_i - Ĉ_i)²

Where:
  λ_noobj = 0.5 (weight for no-object cells)
  1_ij^obj = 1 if box j in cell i is responsible for object
  1_ij^noobj = 1 if box j in cell i has no object
  C_i = predicted confidence
  Ĉ_i = target confidence (P(object) × IoU for positives, 0 for negatives)

Why λ_noobj < 1?
  Most cells contain NO object (background)
  Without downweighting, would dominate loss!
  Typical: 49 cells, maybe 2-5 have objects
  Ratio: 44-47 background vs 2-5 foreground

```

---

**3. Classification Loss (L_class):**

```
For cells containing objects:
  Sum squared error on class probabilities

Formula:
L_class = Σ 1_i^obj Σ (p_i(c) - p̂_i(c))²
                      c∈classes

Where:
  1_i^obj = 1 if cell i contains object
  p_i(c) = predicted probability for class c
  p̂_i(c) = target probability (1 for true class, 0 otherwise)

Summed over:
  - All grid cells i
  - All classes c

```

---

**Complete Loss Function:**

```python
def yolo_loss(predictions, targets, λ_coord=5, λ_noobj=0.5):
    """
    predictions: (batch, S, S, B*5 + C)
    targets: (batch, S, S, B*5 + C)
    """
    S, B, C = 7, 2, 20
    
    # Split predictions
    pred_boxes = predictions[..., :B*5]  # (batch, S, S, 10)
    pred_classes = predictions[..., B*5:]  # (batch, S, S, 20)
    
    # Split targets
    target_boxes = targets[..., :B*5]
    target_classes = targets[..., B*5:]
    
    # Masks
    obj_mask = target_boxes[..., 4] > 0  # Cells with objects
    noobj_mask = ~obj_mask                # Cells without objects
    
    # Localization loss
    loc_loss = λ_coord * (
        (pred_boxes[..., 0:2] - target_boxes[..., 0:2])**2 +  # x, y
        (torch.sqrt(pred_boxes[..., 2:4]) - 
         torch.sqrt(target_boxes[..., 2:4]))**2               # w, h
    )
    loc_loss = (loc_loss * obj_mask).sum()
    
    # Confidence loss
    conf_loss_obj = ((pred_boxes[..., 4] - target_boxes[..., 4])**2 
                     * obj_mask).sum()
    conf_loss_noobj = λ_noobj * ((pred_boxes[..., 4] - 0)**2 
                                  * noobj_mask).sum()
    conf_loss = conf_loss_obj + conf_loss_noobj
    
    # Classification loss
    class_loss = ((pred_classes - target_classes)**2 
                  * obj_mask.unsqueeze(-1)).sum()
    
    # Total loss
    total_loss = loc_loss + conf_loss + class_loss
    
    return total_loss

```

---

## 2.4 YOLO Inference

**Forward Pass:**

```python
def yolo_inference(image, model, conf_threshold=0.25, 
                   iou_threshold=0.5):
    """
    image: Input image (448, 448, 3)
    model: Trained YOLO model
    conf_threshold: Confidence threshold
    iou_threshold: NMS threshold
    """
    # Step 1: Forward pass
    predictions = model(image)  # (7, 7, 30)
    
    # Step 2: Decode predictions
    boxes = []
    scores = []
    classes = []
    
    for i in range(7):  # Grid rows
        for j in range(7):  # Grid columns
            # Get cell predictions
            cell_pred = predictions[i, j]
            
            # For each box in cell
            for b in range(2):
                # Extract box
                box_idx = b * 5
                x, y, w, h, conf = cell_pred[box_idx:box_idx+5]
                
                # Convert to absolute coordinates
                x_abs = (j + x) / 7.0
                y_abs = (i + y) / 7.0
                
                # Get class probabilities
                class_probs = cell_pred[10:]
                
                # Multiply confidence by class prob
                class_scores = conf * class_probs
                class_idx = np.argmax(class_scores)
                score = class_scores[class_idx]
                
                # Filter by threshold
                if score > conf_threshold:
                    boxes.append([x_abs, y_abs, w, h])
                    scores.append(score)
                    classes.append(class_idx)
    
    # Step 3: Non-Maximum Suppression
    keep = nms(boxes, scores, iou_threshold)
    
    # Step 4: Return filtered detections
    final_boxes = [boxes[i] for i in keep]
    final_scores = [scores[i] for i in keep]
    final_classes = [classes[i] for i in keep]
    
    return final_boxes, final_scores, final_classes

```

---

**YOLO Performance (Original v1):**

```
Speed: 45 FPS (frames per second)
  Real-time on GPU! ✓

Accuracy:
  mAP@0.5: 63.4% (PASCAL VOC 2007)
  Lower than Faster R-CNN (73.2%) ✗

Strengths:
  ✓ Very fast (real-time)
  ✓ Sees entire image (better context)
  ✓ Less background false positives
  ✓ Generalizes well to new domains

Weaknesses:
  ✗ Struggles with small objects
  ✗ Struggles with close objects (grid limitation)
  ✗ Coarse localization (7×7 grid)
  ✗ Each cell predicts only one class

```

---

# PART 3: SSD MULTI-SCALE DETECTION (35 minutes)

## 3.1 SSD Motivation

### YOLO's Small Object Problem

```
Problem: 7×7 grid is coarse

Example: Birds in sky
Bird 1: 20×20 pixels at (100, 150)
Bird 2: 18×22 pixels at (120, 140)

Grid cell size: 448/7 = 64×64 pixels

Both birds in cell (1, 2)!
Cell can only predict ONE object ✗

Even if it predicts both boxes:
  Only ONE class probability per cell
  Can't label both as "bird"! ✗

```

---

**Solution: Multi-Scale Feature Maps**

```
Key Insight: Different layers see different scales!

Early layers (high resolution):
  38×38 feature map
  Small receptive field
  Good for SMALL objects ✓

Late layers (low resolution):
  3×3 feature map
  Large receptive field
  Good for LARGE objects ✓

SSD: Use BOTH!
  Make predictions from MULTIPLE layers
  Combine all predictions ✓

```

---

## 3.2 SSD Architecture

### Base Network: VGG-16

```
Input: 300×300×3 image

VGG-16 Backbone (Modified):
  Conv1_1: 300×300×3   → 300×300×64
  Conv1_2: 300×300×64  → 300×300×64
  MaxPool             → 150×150×64
  
  Conv2_1: 150×150×64  → 150×150×128
  Conv2_2: 150×150×128 → 150×150×128
  MaxPool             → 75×75×128
  
  Conv3_1: 75×75×128   → 75×75×256
  Conv3_2: 75×75×256   → 75×75×256
  Conv3_3: 75×75×256   → 75×75×256
  MaxPool             → 38×38×256
  
  Conv4_1: 38×38×256   → 38×38×512
  Conv4_2: 38×38×512   → 38×38×512
  Conv4_3: 38×38×512   → 38×38×512  ← Detection layer!
  MaxPool             → 19×19×512
  
  Conv5_1: 19×19×512   → 19×19×512
  Conv5_2: 19×19×512   → 19×19×512
  Conv5_3: 19×19×512   → 19×19×512
  MaxPool             → (no pooling, keep 19×19)
  
  FC6 → Conv6: 19×19×512 → 19×19×1024
  FC7 → Conv7: 19×19×1024 → 19×19×1024  ← Detection layer!

```

---

### Extra Feature Layers

```
Added on top of VGG-16:

Conv8_1: 19×19×1024 → 19×19×256  (1×1 conv)
Conv8_2: 19×19×256  → 10×10×512  (3×3 conv, stride 2) ← Detection!

Conv9_1: 10×10×512  → 10×10×128  (1×1 conv)
Conv9_2: 10×10×128  → 5×5×256    (3×3 conv, stride 2) ← Detection!

Conv10_1: 5×5×256   → 5×5×128    (1×1 conv)
Conv10_2: 5×5×128   → 3×3×256    (3×3 conv, stride 2) ← Detection!

Conv11_1: 3×3×256   → 3×3×128    (1×1 conv)
Conv11_2: 3×3×128   → 1×1×256    (3×3 conv)         ← Detection!

Total: 6 detection layers

```

---

### Detection Layers Summary

```
┌──────────┬────────────┬────────────┬────────────┬────────────┐
│  Layer   │    Size    │  Anchors/  │   Total    │  Object    │
│          │            │  Location  │  Anchors   │  Size      │
├──────────┼────────────┼────────────┼────────────┼────────────┤
│ Conv4_3  │  38×38×512 │     4      │   5,776    │  Small     │
│ Conv7    │  19×19×1024│     6      │   2,166    │  Medium    │
│ Conv8_2  │  10×10×512 │     6      │     600    │  Med-Large │
│ Conv9_2  │   5×5×256  │     6      │     150    │  Large     │
│ Conv10_2 │   3×3×256  │     4      │      36    │  V.Large   │
│ Conv11_2 │   1×1×256  │     4      │       4    │  Huge      │
├──────────┼────────────┼────────────┼────────────┼────────────┤
│  TOTAL   │     -      │     -      │   8,732    │  All       │
└──────────┴────────────┴────────────┴────────────┴────────────┘

```

---

## 3.3 Default Boxes (Anchors)

### Scale and Aspect Ratio Selection

**Scale Formula:**

```
For layer k (k = 1 to 6):

s_k = s_min + (s_max - s_min) × (k-1) / (m-1)

Where:
  s_min = 0.2 (20% of image)
  s_max = 0.9 (90% of image)
  m = 6 (number of detection layers)

Example:
  Layer 1 (Conv4_3): s_1 = 0.2 → 30 pixels (300×0.2/2)
  Layer 2 (Conv7):   s_2 = 0.34 → 60 pixels
  Layer 3 (Conv8_2): s_3 = 0.48 → 111 pixels
  Layer 4 (Conv9_2): s_4 = 0.62 → 162 pixels
  Layer 5 (Conv10_2):s_5 = 0.76 → 213 pixels
  Layer 6 (Conv11_2):s_6 = 0.9 → 264 pixels

```

---

**Aspect Ratios:**

```
For most layers: {1, 2, 3, 1/2, 1/3}
  → 5 aspect ratios

Plus additional 1:1 box with scale:
  s_k' = √(s_k × s_{k+1})

Total: 6 default boxes per location (for layers with 6 anchors)

Box dimensions:
  For aspect ratio a_r:
    width = s_k × √a_r
    height = s_k / √a_r

Example (Layer 2, s=60):
  Ratio 1:1 → 60×60
  Ratio 2:1 → 85×42
  Ratio 3:1 → 104×35
  Ratio 1:2 → 42×85
  Ratio 1:3 → 35×104
  Extra 1:1 → 81×81 (s_k' = √(60×111))

```

---

### Default Box Assignment

**Matching Strategy (Training):**

```
For each ground truth box:

Step 1: Find best matching default box
  Compute IoU with ALL default boxes
  Match to box with HIGHEST IoU

Step 2: Find additional matches
  All default boxes with IoU > 0.5
  Also matched to this ground truth

Step 3: Assign regression targets
  Matched boxes learn to regress to ground truth:
    Δx = (x_gt - x_anchor) / w_anchor
    Δy = (y_gt - y_anchor) / h_anchor
    Δw = log(w_gt / w_anchor)
    Δh = log(h_gt / h_anchor)

Step 4: Assign class label
  Matched boxes: true class
  Unmatched boxes: background class

```

---

**Example Matching:**

```
Ground truth: Dog at [150, 200, 100, 150]

Compute IoU with all 8732 default boxes:
  Default box 234: IoU = 0.82 (highest) → Match! ✓
  Default box 567: IoU = 0.61 → Match! ✓
  Default box 891: IoU = 0.53 → Match! ✓
  Default box 1024: IoU = 0.42 → No match ✗
  ...

3 default boxes matched to this dog!
All 3 learn to:
  - Classify as "dog"
  - Regress to exact location

```

---

## 3.4 SSD Training

### Loss Function

**Multi-Task Loss:**

```
L(x, c, l, g) = (L_conf(x, c) + α × L_loc(x, l, g)) / N

Where:
  x: Indicator for match (x_ij = 1 if box i matches gt j)
  c: Class confidences
  l: Predicted box locations
  g: Ground truth box locations
  α: Weight for localization loss (default: 1)
  N: Number of matched boxes

```

---

**1. Localization Loss:**

```
L_loc = Smooth L1 loss on box offsets

L_loc(x, l, g) = Σ Σ x_ij × smooth_L1(l_i - ĝ_j)
                 i∈Pos j

Where:
  ĝ_j = encoded ground truth:
    ĝ_j^cx = (g_j^cx - d_i^cx) / d_i^w
    ĝ_j^cy = (g_j^cy - d_i^cy) / d_i^h
    ĝ_j^w = log(g_j^w / d_i^w)
    ĝ_j^h = log(g_j^h / d_i^h)
  
  d_i = default box i
  g_j = ground truth j

Smooth L1:
  smooth_L1(x) = 0.5 × x²       if |x| < 1
                 |x| - 0.5      otherwise

```

---

**2. Confidence Loss:**

```
L_conf = Softmax loss over multiple classes

L_conf(x, c) = -Σ x_ij log(ĉ_i^p) - Σ log(ĉ_i^0)
               i∈Pos                i∈Neg

Where:
  ĉ_i^p = softmax confidence for true class p
  ĉ_i^0 = confidence for background class

  ĉ_i^p = exp(c_i^p) / Σ exp(c_i^j)
                        j

Hard negative mining applied to Neg set!

```

---

**Hard Negative Mining:**

```
Problem: 
  Positive: ~20-50 boxes
  Negative: ~8700 boxes
  Ratio: 1:400+ ✗

Solution:
  1. Calculate confidence loss for ALL negatives
  2. Sort negatives by loss (highest first)
  3. Keep top k negatives where k = 3 × #positives
  4. Discard rest

Example:
  Positives: 30 matched boxes
  All negatives: 8702 boxes
  
  Keep: 90 hardest negatives (top 90 by loss)
  Ratio: 30:90 = 1:3 ✓
  
  Hard negatives are confusing examples:
    - Objects that look similar to true class
    - Difficult backgrounds
    - Partial objects
  
  These are most informative for learning!

```

---

**Complete SSD Loss:**

```python
def ssd_loss(predictions, targets, α=1.0):
    """
    predictions: (batch, num_boxes, 4 + num_classes)
    targets: (batch, num_boxes, 4 + 1)
    """
    # Extract predictions
    pred_locs = predictions[..., :4]      # Box locations
    pred_confs = predictions[..., 4:]      # Class confidences
    
    # Extract targets
    target_locs = targets[..., :4]
    target_labels = targets[..., 4].long()
    
    # Positive mask
    pos_mask = target_labels > 0  # Not background
    num_pos = pos_mask.sum()
    
    # Localization loss (only positives)
    loc_loss = smooth_L1_loss(pred_locs[pos_mask], 
                               target_locs[pos_mask])
    
    # Confidence loss
    conf_loss = F.cross_entropy(pred_confs.view(-1, num_classes),
                                target_labels.view(-1),
                                reduction='none')
    
    # Hard negative mining
    conf_loss_pos = conf_loss[pos_mask].sum()
    conf_loss_neg = conf_loss[~pos_mask]
    
    # Keep top 3× negatives
    num_neg = min(3 * num_pos, conf_loss_neg.numel())
    conf_loss_neg, _ = conf_loss_neg.topk(num_neg)
    conf_loss = conf_loss_pos + conf_loss_neg.sum()
    
    # Total loss
    total_loss = (conf_loss + α * loc_loss) / num_pos
    
    return total_loss

```

---

**SSD Performance:**

```
SSD300 (300×300 input):
  Speed: 59 FPS
  mAP@0.5: 77.2% (PASCAL VOC 2007)
  
  Faster and more accurate than YOLO v1! ✓

SSD512 (512×512 input):
  Speed: 46 FPS (still real-time!)
  mAP@0.5: 79.8%
  
  Approaching Faster R-CNN accuracy! ✓

Strengths:
  ✓ Multi-scale predictions
  ✓ Better small object detection
  ✓ Higher accuracy than YOLO
  ✓ Still real-time

Weaknesses:
  ✗ More complex than YOLO
  ✗ Harder to train
  ✗ Many hyperparameters to tune

```

---

# PART 4: ANCHOR BOXES & MATCHING (25 minutes)

## 4.1 Anchor Box Design Principles

### Why Anchors Work

**Problem Without Anchors:**

```
Network must predict box from scratch:
  [x, y, w, h] = [?, ?, ?, ?]

For a car:
  Could be anywhere: x ∈ [0, image_width]
  Could be any size: w ∈ [10, image_width]

Search space: HUGE! ✗
Difficult for network to learn ✗

```

---

**Solution With Anchors:**

```
Pre-define anchor: [x_anchor, y_anchor, w_anchor, h_anchor]

Network predicts OFFSET:
  Δx, Δy: Small adjustments to position
  Δw, Δh: Scale factors for size

For a car with anchor [100, 150, 200, 100]:
  Predict: Δx=+5, Δy=-3, Δw=×1.1, Δh=×0.9
  Final box: [105, 147, 220, 90]

Search space: Small offsets! ✓
Much easier to learn! ✓

```

---

### Anchor Box Diversity

**Coverage Principle:**

```
Need anchors that cover:
  1. Different aspect ratios (tall, wide, square)
  2. Different scales (small, medium, large)

Example object distributions:
  People: Tall (aspect ratio ~1:2)
  Cars: Wide (aspect ratio ~2:1)
  Faces: Square (aspect ratio ~1:1)
  Buses: Large (scale ~400 pixels)
  Pedestrians: Medium (scale ~150 pixels)
  Traffic signs: Small (scale ~50 pixels)

Anchors should match these distributions! ✓

```

---

**K-Means Clustering for Anchor Selection:**

```python
def find_optimal_anchors(ground_truth_boxes, k=9):
    """
    Use K-Means to find k best anchor sizes
    
    ground_truth_boxes: (N, 4) array of [x, y, w, h]
    k: Number of anchors to generate
    """
    # Extract widths and heights
    widths = ground_truth_boxes[:, 2]
    heights = ground_truth_boxes[:, 3]
    boxes = np.column_stack([widths, heights])
    
    # K-Means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(boxes)
    
    # Cluster centers are our anchors
    anchors = kmeans.cluster_centers_
    
    # Sort by area
    areas = anchors[:, 0] * anchors[:, 1]
    anchors = anchors[np.argsort(areas)]
    
    return anchors

# Example output:
# [[ 15,  20],  # Small
#  [ 30,  40],  # Small-medium
#  [ 40,  80],  # Medium tall
#  [ 80,  40],  # Medium wide
#  [ 60,  60],  # Medium square
#  [100, 200],  # Large tall
#  [200, 100],  # Large wide
#  [150, 150],  # Large square
#  [300, 300]]  # Very large

```

---

## 4.2 Anchor Matching During Training

### Matching Strategies

**Strategy 1: Best Match (SSD)**

```
For each ground truth:
  Step 1: Find best matching anchor
    anchor* = argmax(IoU(anchor_i, gt))
    Match anchor* to this ground truth
  
  Step 2: Find threshold matches
    For all anchors:
      If IoU(anchor_i, gt) > 0.5:
        Match anchor_i to this ground truth

Ensures every ground truth has ≥1 match! ✓

```

---

**Strategy 2: Threshold Only (YOLO)**

```
For each anchor:
  If IoU(anchor, gt) > threshold:
    Match to ground truth with highest IoU
  Else:
    Negative (background)

Simpler but some ground truths might have 0 matches! ✗

```

---

**Strategy 3: Faster R-CNN (RPN)**

```
For each anchor:
  Positive if:
    IoU > 0.7 with any ground truth
    OR highest IoU for a ground truth
  
  Negative if:
    IoU < 0.3 with all ground truths
  
  Ignore if:
    0.3 ≤ IoU ≤ 0.7 (don't train on ambiguous)

Explicit handling of ambiguous cases! ✓

```

---

### Regression Target Encoding

**Parametrization:**

```
Given:
  Anchor box: [x_a, y_a, w_a, h_a]
  Ground truth: [x_gt, y_gt, w_gt, h_gt]

Encode targets as:
  t_x = (x_gt - x_a) / w_a
  t_y = (y_gt - y_a) / h_a
  t_w = log(w_gt / w_a)
  t_h = log(h_gt / h_a)

Why this encoding?

1. Position (t_x, t_y):
   Normalized by anchor width/height
   Makes offset invariant to anchor size! ✓

2. Size (t_w, t_h):
   Log space: multiplicative → additive
   log(2×w) = log(w) + log(2)
   Easier for network to learn! ✓

```

---

**Example Encoding:**

```
Anchor: [100, 100, 50, 50]
Ground truth: [110, 95, 60, 70]

Compute targets:
  t_x = (110 - 100) / 50 = +0.2
  t_y = (95 - 100) / 50 = -0.1
  t_w = log(60 / 50) = log(1.2) = +0.182
  t_h = log(70 / 50) = log(1.4) = +0.336

Network learns to predict:
  [0.2, -0.1, 0.182, 0.336]

Much easier than predicting:
  [110, 95, 60, 70] directly! ✓

```

---

**Decoding Predictions:**

```python
def decode_boxes(anchors, predictions):
    """
    anchors: (N, 4) [x, y, w, h]
    predictions: (N, 4) [t_x, t_y, t_w, t_h]
    """
    # Extract anchor coordinates
    x_a = anchors[:, 0]
    y_a = anchors[:, 1]
    w_a = anchors[:, 2]
    h_a = anchors[:, 3]
    
    # Extract predictions
    t_x = predictions[:, 0]
    t_y = predictions[:, 1]
    t_w = predictions[:, 2]
    t_h = predictions[:, 3]
    
    # Decode
    x = x_a + t_x * w_a
    y = y_a + t_y * h_a
    w = w_a * np.exp(t_w)
    h = h_a * np.exp(t_h)
    
    return np.column_stack([x, y, w, h])

```

---

# PART 5: NON-MAXIMUM SUPPRESSION (20 minutes)

## 5.1 NMS Algorithm

### Standard NMS

**Pseudocode:**

```
function NMS(boxes, scores, threshold):
    # boxes: List of [x, y, w, h]
    # scores: Confidence scores
    # threshold: IoU threshold
    
    # Step 1: Sort by score (descending)
    indices = argsort(scores, descending=True)
    
    keep = []
    
    while indices not empty:
        # Step 2: Pick highest score
        current = indices[0]
        keep.append(current)
        
        # Step 3: Remove from list
        indices = indices[1:]
        
        # Step 4: Compute IoU with remaining
        ious = [IoU(boxes[current], boxes[i]) 
                for i in indices]
        
        # Step 5: Remove high IoU boxes
        indices = [indices[i] 
                   for i in range(len(indices))
                   if ious[i] <= threshold]
    
    return keep

```

---

**Python Implementation:**

```python
def compute_iou(box1, box2):
    """
    box1, box2: [x, y, w, h]
    Returns: IoU value
    """
    # Convert to [x1, y1, x2, y2]
    x1_min, y1_min = box1[0], box1[1]
    x1_max, y1_max = box1[0] + box1[2], box1[1] + box1[3]
    
    x2_min, y2_min = box2[0], box2[1]
    x2_max, y2_max = box2[0] + box2[2], box2[1] + box2[3]
    
    # Intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    inter_width = max(0, xi_max - xi_min)
    inter_height = max(0, yi_max - yi_min)
    intersection = inter_width * inter_height
    
    # Union
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    # IoU
    iou = intersection / union if union > 0 else 0
    return iou

def nms(boxes, scores, iou_threshold=0.5):
    """
    boxes: List of [x, y, w, h]
    scores: List of confidence scores
    iou_threshold: IoU threshold for suppression
    """
    # Sort by score
    indices = np.argsort(scores)[::-1]
    
    keep = []
    
    while len(indices) > 0:
        # Pick highest score
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Compute IoU with remaining
        ious = np.array([
            compute_iou(boxes[current], boxes[i])
            for i in indices[1:]
        ])
        
        # Keep boxes with low IoU
        indices = indices[1:][ious <= iou_threshold]
    
    return keep

```

---

**Step-by-Step Example:**

```
Detections for "dog":
  Box A: [100, 150, 200, 180], score = 0.95
  Box B: [105, 148, 198, 182], score = 0.87
  Box C: [98, 152, 203, 179], score = 0.91
  Box D: [102, 151, 199, 181], score = 0.78

Threshold: 0.5

Step 1: Sort by score
  [A(0.95), C(0.91), B(0.87), D(0.78)]

Step 2: Keep A (highest)
  Remaining: [C, B, D]

Step 3: Compute IoU with A
  IoU(A, C) = 0.82 > 0.5 → Remove C ✗
  IoU(A, B) = 0.76 > 0.5 → Remove B ✗
  IoU(A, D) = 0.88 > 0.5 → Remove D ✗

Final: Keep [A]

Result: One box representing the dog! ✓

```

---

## 5.2 NMS Variants

### Soft-NMS

**Problem with Standard NMS:**

```
Crowded scenes with overlapping objects:

Box 1: Person at [100, 150, 80, 200], score = 0.9
Box 2: Person at [150, 140, 75, 210], score = 0.85

IoU(Box 1, Box 2) = 0.6 > 0.5

Standard NMS: Remove Box 2! ✗
Problem: Box 2 is a DIFFERENT person!

```

---

**Soft-NMS Solution:**

```
Instead of removing boxes:
  → Reduce their confidence scores

Algorithm:
  for each box in sorted order:
    for overlapping boxes:
      if IoU > threshold:
        # Don't remove, just reduce score
        score *= decay_function(IoU)

Decay functions:
  1. Linear: score *= (1 - IoU)
  2. Gaussian: score *= exp(-(IoU²) / σ)

High IoU → Low score (likely duplicate)
Low IoU → High score (likely separate object)

Boxes naturally sorted by confidence! ✓
Can keep more true positives ✓

```

---

**Soft-NMS Implementation:**

```python
def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5):
    """
    Soft-NMS with Gaussian decay
    """
    N = len(boxes)
    indices = np.arange(N)
    
    for i in range(N):
        # Find box with max score
        max_idx = i + np.argmax(scores[i:])
        
        # Swap with position i
        boxes[i], boxes[max_idx] = boxes[max_idx], boxes[i]
        scores[i], scores[max_idx] = scores[max_idx], scores[i]
        indices[i], indices[max_idx] = indices[max_idx], indices[i]
        
        # Compute IoU with remaining boxes
        for j in range(i + 1, N):
            iou = compute_iou(boxes[i], boxes[j])
            
            # Gaussian decay
            scores[j] *= np.exp(-(iou ** 2) / sigma)
    
    # Filter by score threshold
    keep = scores > 0.01
    
    return indices[keep], scores[keep]

```

---

### Class-Specific NMS

**Optimization:**

```
Standard NMS:
  Run on ALL boxes together
  Slow for many classes

Class-Specific NMS:
  Run NMS separately per class
  Much faster!

Example:
  1000 detections total
  - 200 "person"
  - 150 "car"
  - 100 "dog"
  - ...

Standard NMS:
  1000 boxes → O(1000²) = 1M comparisons

Class-Specific NMS:
  Person: 200² = 40K
  Car: 150² = 22.5K
  Dog: 100² = 10K
  Total: ~100K comparisons
  
  10× faster! ✓

Bonus: Can parallelize across classes ✓

```

---

# PART 6: EVALUATION METRICS (35 minutes)

## 6.1 Precision and Recall

### Definitions for Object Detection

**Confusion Matrix:**

```
For object detection at IoU threshold τ:

True Positive (TP):
  Detection with IoU(pred, gt) ≥ τ AND correct class

False Positive (FP):
  Detection with IoU(pred, gt) < τ OR wrong class

False Negative (FN):
  Ground truth object not detected

True Negative (TN):
  Not applicable for detection
  (Can't count all possible negative regions)

```

---

**Metrics:**

```
Precision = TP / (TP + FP)
  = "What fraction of detections are correct?"
  = "How precise are we?"

Recall = TP / (TP + FN)
  = "What fraction of objects did we find?"
  = "How complete is our detection?"

F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
  = Harmonic mean of precision and recall

```

---

**Example Calculation:**

```
Image with 10 dogs

Model outputs 12 detections:
  - 8 correct dogs (IoU > 0.5, class = dog) → TP = 8
  - 2 false alarms (IoU < 0.5 or wrong class) → FP = 2
  - 2 dogs missed → FN = 2

Precision = 8 / (8 + 2) = 8/10 = 0.80
  "80% of our detections were correct"

Recall = 8 / (8 + 2) = 8/10 = 0.80
  "We found 80% of all dogs"

F1 = 2 × (0.80 × 0.80) / (0.80 + 0.80) = 0.80

```

---

## 6.2 Precision-Recall Curve

### Varying Confidence Threshold

**Concept:**

```
Vary confidence threshold from 0 to 1:

Threshold = 0.9 (very strict):
  Few detections
  High precision (most are correct)
  Low recall (miss many objects)

Threshold = 0.3 (very loose):
  Many detections
  Low precision (many false positives)
  High recall (find most objects)

Plot: Precision (y) vs Recall (x)
Curve shows tradeoff! ✓

```

---

**Example PR Curve:**

```
Sort all detections by confidence:

Rank | Conf  | Correct? | TP | FP | Precision | Recall
-----|-------|----------|----|----|-----------|--------
  1  | 0.95  |   Yes    | 1  | 0  |   1.00    |  0.10
  2  | 0.87  |   Yes    | 2  | 0  |   1.00    |  0.20
  3  | 0.82  |   No     | 2  | 1  |   0.67    |  0.20
  4  | 0.78  |   Yes    | 3  | 1  |   0.75    |  0.30
  5  | 0.65  |   Yes    | 4  | 1  |   0.80    |  0.40
  6  | 0.61  |   No     | 4  | 2  |   0.67    |  0.40
  7  | 0.58  |   Yes    | 5  | 2  |   0.71    |  0.50
  8  | 0.52  |   Yes    | 6  | 2  |   0.75    |  0.60
  9  | 0.48  |   No     | 6  | 3  |   0.67    |  0.60
 10  | 0.45  |   Yes    | 7  | 3  |   0.70    |  0.70
 11  | 0.40  |   Yes    | 8  | 3  |   0.73    |  0.80
 12  | 0.35  |   No     | 8  | 4  |   0.67    |  0.80
 13  | 0.30  |   Yes    | 9  | 4  |   0.69    |  0.90
 14  | 0.25  |   Yes    | 10 | 4  |   0.71    |  1.00
 15  | 0.20  |   No     | 10 | 5  |   0.67    |  1.00

(Assuming 10 ground truth objects)

PR points: (0.1, 1.0), (0.2, 1.0), (0.3, 0.75), ...

```

---

## 6.3 Average Precision (AP)

### Computation Methods

**Method 1: 11-Point Interpolation (PASCAL VOC)**

```
Sample at 11 recall levels: {0.0, 0.1, 0.2, ..., 1.0}

For each recall level r:
  p_interp(r) = max(p) for all recall r' ≥ r

AP = (1/11) × Σ p_interp(r)
              r∈{0.0,0.1,...,1.0}

Example:
  p_interp(0.0) = 1.00 (max precision at recall ≥ 0)
  p_interp(0.1) = 1.00
  p_interp(0.2) = 1.00
  p_interp(0.3) = 0.75
  p_interp(0.4) = 0.80
  p_interp(0.5) = 0.75
  p_interp(0.6) = 0.75
  p_interp(0.7) = 0.73
  p_interp(0.8) = 0.73
  p_interp(0.9) = 0.69
  p_interp(1.0) = 0.71

AP = (1/11) × (1.00 + 1.00 + 1.00 + 0.75 + 0.80 + 
               0.75 + 0.75 + 0.73 + 0.73 + 0.69 + 0.71)
   = (1/11) × 8.91
   = 0.810

```

---

**Method 2: All Points (COCO)**

```
Use ALL unique recall values

AP = Σ (r_{n+1} - r_n) × p_interp(r_{n+1})
     n

Where:
  r_n = recall at rank n
  p_interp(r) = max precision for recall ≥ r

More accurate than 11-point! ✓

```

---

**Python Implementation:**

```python
def compute_ap(precision, recall):
    """
    Compute AP using all-point interpolation
    
    precision: Array of precision values
    recall: Array of recall values
    """
    # Append sentinel values
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    # Compute envelope (max precision at each recall)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Find recall differences
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Sum areas
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap

# Example usage:
precision = np.array([1.0, 1.0, 0.67, 0.75, 0.80, 0.67, 0.71])
recall = np.array([0.1, 0.2, 0.2, 0.3, 0.4, 0.4, 0.5])

ap = compute_ap(precision, recall)
print(f"AP: {ap:.3f}")

```

---

## 6.4 Mean Average Precision (mAP)

### mAP Calculation

**Definition:**

```
mAP = Mean of AP across all classes

mAP = (1/C) × Σ AP_c
              c=1 to C

Where:
  C = number of classes
  AP_c = Average Precision for class c

```

---

**Example Calculation:**

```
PASCAL VOC (20 classes):

Class          | AP@0.5
---------------|--------
Aeroplane      | 0.791
Bicycle        | 0.834
Bird           | 0.758
Boat           | 0.695
Bottle         | 0.623
Bus            | 0.857
Car            | 0.863
Cat            | 0.881
Chair          | 0.598
Cow            | 0.825
Dining table   | 0.712
Dog            | 0.846
Horse          | 0.868
Motorbike      | 0.838
Person         | 0.892
Potted plant   | 0.512
Sheep          | 0.779
Sofa           | 0.783
Train          | 0.875
TV/monitor     | 0.765

mAP@0.5 = (0.791 + 0.834 + ... + 0.765) / 20
        = 15.576 / 20
        = 0.779 (77.9%)

```

---

### mAP Variants

**mAP@0.5 (PASCAL VOC):**

```
IoU threshold = 0.5
Detection counts as TP if IoU ≥ 0.5

Easier metric:
  Box just needs 50% overlap
  More lenient on localization ✓

```

---

**mAP@0.75 (COCO):**

```
IoU threshold = 0.75
Detection counts as TP if IoU ≥ 0.75

Harder metric:
  Box needs 75% overlap
  Requires precise localization ✓

```

---

**mAP@[0.5:0.95] (COCO Primary):**

```
Average over 10 IoU thresholds:
  {0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95}

Compute AP at each threshold:
  AP@0.5, AP@0.55, ..., AP@0.95

mAP@[0.5:0.95] = (1/10) × Σ AP@threshold

Most comprehensive metric! ✓
Rewards both detection and localization ✓

Typical values:
  mAP@0.5: 60-80%
  mAP@[0.5:0.95]: 30-50%
  (Much lower due to strict thresholds!)

```

---

**Size-Specific mAP (COCO):**

```
Small objects: area < 32²
Medium objects: 32² ≤ area < 96²
Large objects: area ≥ 96²

Metrics:
  mAP_S: AP for small objects
  mAP_M: AP for medium objects
  mAP_L: AP for large objects

Example model performance:
  mAP overall: 42.5%
  mAP_S: 26.3% (struggles with small!)
  mAP_M: 46.2%
  mAP_L: 54.8%

Reveals model weaknesses! ✓

```

---

## Summary: Key Takeaways

### Object Detection Evolution

```
R-CNN (2014):
  Speed: 47 s/image
  Accuracy: 62.4% mAP
  Method: Selective Search + CNN per region

Fast R-CNN (2015):
  Speed: 2.3 s/image (20× faster)
  Accuracy: 66.9% mAP
  Method: CNN once + ROI pooling

Faster R-CNN (2015):
  Speed: 0.2 s/image (5 FPS)
  Accuracy: 73.2% mAP
  Method: RPN + ROI pooling

YOLO (2015):
  Speed: 45 FPS (real-time!)
  Accuracy: 63.4% mAP
  Method: Single-shot, grid-based

SSD (2016):
  Speed: 59 FPS
  Accuracy: 77.2% mAP
  Method: Multi-scale features

Breakthrough: Real-time + High accuracy! ✓

```

---

### YOLO vs SSD

```
YOLO:
  ✓ Simpler architecture
  ✓ Faster inference
  ✓ Better for large objects
  ✗ Struggles with small objects
  ✗ Lower accuracy

SSD:
  ✓ Multi-scale detection
  ✓ Better small object detection
  ✓ Higher accuracy
  ✗ More complex
  ✗ Harder to tune

Choose based on requirements:
  Speed critical? → YOLO
  Accuracy critical? → SSD
  Small objects? → SSD
  Large objects? → Either works

```

---

### Core Concepts

1. 
**Grid-based detection (YOLO)**

Divide image into cells
Each cell predicts boxes + classes
Simple but effective

2. 
**Multi-scale features (SSD)**

Different layers for different sizes
38×38 for small, 3×3 for large
Better coverage of object sizes

3. 
**Anchor boxes**

Pre-defined box templates
Predict offsets, not absolute values
Easier to train

4. 
**Non-Maximum Suppression**

Remove duplicate detections
Keep highest confidence per object
Essential post-processing

5. 
**mAP evaluation**

Precision-Recall curve → AP
Average across classes → mAP
Higher IoU threshold → harder metric

---

**End of Lecture Notes**

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