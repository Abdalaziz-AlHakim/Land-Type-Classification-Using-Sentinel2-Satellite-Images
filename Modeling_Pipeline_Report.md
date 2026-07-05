# EuroSAT Modeling & Optimization Pipeline — Summary Report
### Chunks 9–13: Data Loading → Model Architecture → Training → Learning Curves

---

## 1. Chunk 9 — Memory-Efficient Data Loading

### Problem
The full EuroSAT dataset (~19K images) would consume ~15 GB if loaded entirely into RAM, likely crashing a Kaggle kernel.

### Solution: Lazy Generators via `tf.data.Dataset`
Instead of loading everything at once, the pipeline uses **Python generators** wrapped in `tf.data.Dataset.from_generator()`. Images are read from disk **one batch at a time**, keeping RAM usage low.

| Component | Strategy | Why |
|---|---|---|
| **Train & Val sets** | Lazy generators (`from_generator`) | Too large for RAM; loaded on-the-fly per batch |
| **Test set** | Loaded into NumPy arrays in-memory | Small enough (~2–3K images); needed as a single array for `model.evaluate()` |

### Preprocessing Applied

| Dataset | Preprocessing Steps |
|---|---|
| **RGB images** | Resize to **224×224**, normalize with **ImageNet mean/std** (μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]). Training set also applies **augmentation** (horizontal/vertical flip, 90° rotation, brightness jitter) |
| **TIF (13-band)** | Keep at **64×64**, apply **per-band min-max normalization** to [0, 1] |

### Data Augmentation (RGB training only)
- Random horizontal flip (50% chance)
- Random vertical flip (50% chance)
- Random 90° rotation (0°/90°/180°/270°)
- Random brightness adjustment (factor ∈ [0.8, 1.2])

> **Key Design Choice:** Augmentation is applied only to the RGB training generator, not to validation or test data, to prevent data leakage and ensure fair evaluation.

---

## 2. Chunk 10 — TIF Model: Custom ResNet-50 Architecture

### Why Custom?
The 13-band multispectral (TIF) input has **13 channels**, not 3. Pretrained ImageNet models (designed for 3-channel RGB) cannot be directly reused, so a **custom ResNet-50-inspired** CNN is built from scratch.

### Architecture

```
Input (64×64×13)
  ↓
7×7 Conv (64 filters, stride 2) → BatchNorm → ReLU → MaxPool (3×3, stride 2)
  ↓
Stage 1: ConvBlock(64,64,256, stride=1) → 2× IdentityBlock(64,64,256)
  ↓
Stage 2: ConvBlock(128,128,512, stride=2) → 3× IdentityBlock(128,128,512)
  ↓
Stage 3: ConvBlock(256,256,1024, stride=2) → 2× IdentityBlock(256,256,1024)
  ↓
Global Average Pooling → Dense(512, ReLU) → Dropout(0.5) → Dense(10, Softmax)
```

### Key Components
- **Residual blocks:** Each block uses a 1×1 → 3×3 → 1×1 bottleneck pattern with skip connections, addressing the vanishing gradient problem.
- **Identity block:** Input dimensions = output dimensions (skip connection is a direct add).
- **Convolutional block:** Uses a strided convolution on both the main path and the shortcut to reduce spatial dimensions.
- **Dropout (0.5):** Regularization to prevent overfitting on the ~16K training images.

---

## 3. Chunk 11 — RGB Model: Transfer Learning with ResNet-50

### Strategy: Feature Extraction (Frozen Backbone)
The RGB model uses a **pretrained ResNet-50** (trained on ImageNet's 1.2M images) as a frozen feature extractor, with only a custom classification head trained on EuroSAT.

### Architecture

```
Input (224×224×3)
  ↓
ResNet-50 backbone (pretrained, FROZEN — all layers non-trainable)
  ↓
Global Average Pooling
  ↓
Dense(512, ReLU) → Dropout(0.5) → Dense(10, Softmax)
```

### Parameter Breakdown

| Category | Approx. Count |
|---|---|
| Total parameters | ~24M |
| Frozen (ResNet-50 backbone) | ~23.5M |
| Trainable (classification head) | ~270K |

> **Key Design Choice:** Freezing the backbone means training is **fast** (only ~270K parameters to update) and **less prone to overfitting** since the ImageNet features are general-purpose and highly transferable to satellite imagery.

---

## 4. Chunk 12 — Training Configuration & Execution

### Compilation Settings (both models)

| Setting | Value |
|---|---|
| **Optimizer** | Adam (lr = 1e-3) |
| **Loss** | Categorical Cross-Entropy |
| **Metric** | Accuracy |

### Training Callbacks

| Callback | Configuration | Purpose |
|---|---|---|
| **EarlyStopping** | `monitor='val_loss'`, `patience=8`, `restore_best_weights=True` | Stops training when validation loss stops improving; reverts to best weights |
| **ReduceLROnPlateau** | `monitor='val_loss'`, `factor=0.5`, `patience=4`, `min_lr=1e-7` | Halves the learning rate when validation loss plateaus for 4 epochs |
| **ModelCheckpoint** | `monitor='val_accuracy'`, `save_best_only=True` | Saves the best model (by validation accuracy) to disk as `.keras` file |

### Training Process
1. **TIF model** trains first on the 13-band lazy generator for up to **50 epochs**.
2. **RGB model** trains second on the RGB lazy generator for up to **50 epochs**.
3. Both models use the same callback suite; early stopping usually terminates training well before 50 epochs.

> **Optimization Strategy:** The combination of EarlyStopping + ReduceLROnPlateau implements an adaptive training schedule — the model trains at a high learning rate initially for fast convergence, then drops the LR for fine-grained optimization, and stops when no further improvement is observed.

---

## 5. Chunk 13 — Learning Curves Visualization

### What is Plotted
For each model, a **dual-panel figure** is generated:

| Panel | Content |
|---|---|
| **Left (Accuracy)** | Training accuracy vs. validation accuracy over epochs |
| **Right (Loss)** | Training loss vs. validation loss over epochs |

Both panels include:
- A **green dashed vertical line** marking the **best epoch** (epoch with highest validation accuracy).
- A legend distinguishing training (blue circles) vs. validation (red squares) curves.

### Diagnostic Value
The learning curves serve as the primary diagnostic tool for detecting:

| Pattern | Diagnosis | Implication |
|---|---|---|
| Train acc >> Val acc | **Overfitting** | Model memorizes training data; needs more regularization or data |
| Train acc ≈ Val acc, both low | **Underfitting** | Model is too simple or LR is too high |
| Both curves converge and plateau | **Good fit** | Model generalizes well |
| Val loss increases while train loss drops | **Overfitting onset** | EarlyStopping should trigger here |

### Summary Statistics Printed
After plotting, the function prints:
- Total epochs actually trained (before early stopping)
- Best epoch number
- Best validation accuracy achieved
- Final training and validation accuracy

---

## Pipeline Flow Summary

```
Chunk 9: Data Loading
  ├── RGB: lazy tf.data generator (224×224, ImageNet-normalized, augmented)
  └── TIF: lazy tf.data generator (64×64, per-band normalized)
          ↓
Chunk 10: TIF Model Definition
  └── Custom ResNet-50 (13-channel input, ~millions of params, trained from scratch)
          ↓
Chunk 11: RGB Model Definition
  └── Transfer Learning ResNet-50 (frozen backbone, ~270K trainable params)
          ↓
Chunk 12: Training
  ├── Adam optimizer (lr=1e-3), categorical cross-entropy
  ├── EarlyStopping (patience=8) + ReduceLROnPlateau (patience=4)
  └── ModelCheckpoint (saves best .keras file)
          ↓
Chunk 13: Learning Curves
  └── Accuracy & Loss plots → diagnose overfitting/underfitting
```
