# %% [markdown]
# # 🛰️ EuroSAT — Model Development & Performance Tuning Pipeline
# ### Milestone 2 & 3: RGB + TIF Classification Models
# ---
# **Continuation of**: `eurosat_eda_complete.py` (Milestone 1 — EDA & Preprocessing)
#
# **Milestone 2**: Build two classification models:
#   - TIF Model: Custom ResNet-50 inspired CNN for 13-band multispectral input
#   - RGB Model: Transfer Learning with pretrained ResNet-50 (ImageNet)
#
# **Milestone 3**: Evaluate, compare, and tune both models:
#   - Learning curves (accuracy & loss)
#   - Confusion matrices & classification reports
#   - RGB vs TIF comparison

# %% [markdown]
# ---
# ## 🔧 CHUNK 9 — SETUP, IMPORTS & DATA LOADING
# ---
# **What this does:**
# Before building any model, we need to:
# 1. Import TensorFlow/Keras (our deep learning framework)
# 2. Set up reproducibility (same results every run)
# 3. Load the dataset into efficient data pipelines
#
# **Why TensorFlow/Keras?**
# - Industry standard for image classification
# - Built-in support for pretrained models (ResNet-50)
# - `tf.data` pipelines handle large datasets efficiently without loading everything into RAM

# %%
# ============================================================
#  CHUNK 9 — SETUP, IMPORTS & DATA LOADING
# ============================================================
import os, sys, json, warnings, gc
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
import cv2

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical

# Metrics
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder

# Rasterio for multispectral .tif loading
try:
    import rasterio
except ImportError:
    os.system("pip install rasterio -q")
    import rasterio

# ── Suppress warnings & set seeds for reproducibility ──
warnings.filterwarnings("ignore")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ── Plot styling ──
sns.set_theme(style="darkgrid", palette="deep", font_scale=1.1)
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "figure.dpi": 120,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.family": "sans-serif",
})

# ── GPU check ──
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"✅ GPU detected: {gpus[0].name}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️  No GPU detected — training will be slow on CPU.")

print(f"TensorFlow version: {tf.__version__}")
print("✅ All imports successful.")

# %%
# ============================================================
#  CONFIGURATION — UPDATE THESE PATHS FOR YOUR ENVIRONMENT
# ============================================================
# If running on Colab with Google Drive:
# from google.colab import drive
# drive.mount('/content/drive')
# BASE_DIR = Path("/content/drive/MyDrive/path_to/archive")

BASE_DIR = Path("archive")  # <-- Update this path

RGB_DIR = BASE_DIR / "EuroSAT"
MS_DIR  = BASE_DIR / "EuroSATallBands"

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]
NUM_CLASSES = len(CLASS_NAMES)

# Sentinel-2 band indices (0-indexed)
B04_RED   = 3
B08_NIR   = 7
B03_GREEN = 2
B11_SWIR1 = 11

# Training configuration
BATCH_SIZE     = 32
EPOCHS         = 50      # EarlyStopping will cut this short if needed
RGB_IMG_SIZE   = (224, 224)  # ResNet-50 expects 224×224
TIF_IMG_SIZE   = (64, 64)   # Native Sentinel-2 resolution
LEARNING_RATE  = 1e-3

print(f"📂 RGB dir  : {RGB_DIR}")
print(f"📂 MS dir   : {MS_DIR}")
print(f"📋 Classes  : {NUM_CLASSES}")
print(f"⚙️  Batch    : {BATCH_SIZE} | Epochs: {EPOCHS} | LR: {LEARNING_RATE}")

# %%
# ============================================================
#  LOAD CSV SPLITS & LABEL ENCODING
# ============================================================
# ── Why split into Train/Val/Test? ──
# • Train: the model learns from this data
# • Validation: used to monitor the model DURING training (prevents overfitting)
# • Test: used AFTER training to get the final unbiased accuracy score
#
# We NEVER train on test or validation data — that would be data leakage.

def load_csv(base_dir, name):
    """Load a split CSV and return a DataFrame."""
    df = pd.read_csv(base_dir / name)
    df.columns = [c.strip() for c in df.columns]
    return df

# Load RGB splits
train_rgb_df = load_csv(RGB_DIR, "train.csv")
val_rgb_df   = load_csv(RGB_DIR, "validation.csv")
test_rgb_df  = load_csv(RGB_DIR, "test.csv")

# Load MS (TIF) splits
train_ms_df = load_csv(MS_DIR, "train.csv")
val_ms_df   = load_csv(MS_DIR, "validation.csv")
test_ms_df  = load_csv(MS_DIR, "test.csv")

# Label encoder
le = LabelEncoder()
le.fit(CLASS_NAMES)
label_to_int = {name: i for i, name in enumerate(CLASS_NAMES)}

print(f"📄 RGB  — Train: {len(train_rgb_df)} | Val: {len(val_rgb_df)} | Test: {len(test_rgb_df)}")
print(f"📄 TIF  — Train: {len(train_ms_df)}  | Val: {len(val_ms_df)}  | Test: {len(test_ms_df)}")
print(f"🏷️  Label mapping: {label_to_int}")

# %%
# ============================================================
#  DATA GENERATORS — RGB (for Transfer Learning ResNet-50)
# ============================================================
# ── Why use data generators? ──
# Loading ALL 27,000 images into RAM at once would use ~15 GB+.
# Instead, we load images in small batches (32 at a time).
#
# ── RGB Preprocessing Steps ──
# 1. Load image → resize to 224×224 (ResNet-50 input size)
# 2. Normalize using ImageNet stats (mean/std) — because ResNet-50
#    was trained on ImageNet with these exact statistics
# 3. Apply augmentation (flips, rotations) to training set ONLY

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def load_and_preprocess_rgb(filepath, label, augment=False):
    """Load an RGB image, resize, normalize with ImageNet stats."""
    img = Image.open(filepath).convert("RGB")
    img = img.resize(RGB_IMG_SIZE, Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0

    # Augmentation (training only)
    if augment:
        if np.random.rand() > 0.5:
            img = np.fliplr(img)          # Horizontal flip
        if np.random.rand() > 0.5:
            img = np.flipud(img)          # Vertical flip
        k = np.random.choice([0, 1, 2, 3])
        img = np.rot90(img, k)            # Random 90° rotation
        # Brightness adjustment
        factor = np.random.uniform(0.8, 1.2)
        img = np.clip(img * factor, 0, 1)

    # ImageNet normalization
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.astype(np.float32), label

def build_rgb_dataset(df, base_dir, augment=False, shuffle=True):
    """Build a tf.data.Dataset for RGB images."""
    filepaths = [str(base_dir / row["Filename"]) for _, row in df.iterrows()]
    labels = to_categorical([label_to_int[c] for c in df["ClassName"]], NUM_CLASSES)

    images = []
    valid_labels = []
    for fp, lbl in zip(filepaths, labels):
        try:
            img, _ = load_and_preprocess_rgb(fp, lbl, augment=augment)
            images.append(img)
            valid_labels.append(lbl)
        except Exception as e:
            continue  # Skip corrupted

    X = np.array(images, dtype=np.float32)
    Y = np.array(valid_labels, dtype=np.float32)

    if len(X) == 0:
        print(f"⚠️  WARNING: No images were loaded! Check that file paths in the CSV match the directory structure.")
        print(f"   Expected base dir: {base_dir}")
        print(f"   Sample path tried: {filepaths[0] if filepaths else 'N/A'}")

    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle and len(X) > 0:
        dataset = dataset.shuffle(buffer_size=len(X), seed=RANDOM_SEED)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset, X, Y

print("🔄 Loading RGB datasets (this may take a few minutes)...")
train_rgb_ds, X_train_rgb, Y_train_rgb = build_rgb_dataset(train_rgb_df, RGB_DIR, augment=True)
val_rgb_ds,   X_val_rgb,   Y_val_rgb   = build_rgb_dataset(val_rgb_df, RGB_DIR, augment=False, shuffle=False)
test_rgb_ds,  X_test_rgb,  Y_test_rgb  = build_rgb_dataset(test_rgb_df, RGB_DIR, augment=False, shuffle=False)

print(f"✅ RGB data loaded — Train: {X_train_rgb.shape} | Val: {X_val_rgb.shape} | Test: {X_test_rgb.shape}")

# %%
# ============================================================
#  DATA GENERATORS — TIF (for Custom ResNet-50)
# ============================================================
# ── TIF Preprocessing Steps ──
# 1. Load .tif file → 64×64×13 array (13 spectral bands)
# 2. Per-band normalization: each band scaled to [0, 1] by its own min/max
#    (because bands have wildly different value ranges: Blue 0–3000, NIR 0–8000)
# 3. No pretrained model exists for 13 bands, so we train from scratch

def load_and_preprocess_tif(filepath, label):
    """Load a multispectral .tif image and normalize per-band to [0, 1]."""
    with rasterio.open(filepath) as src:
        data = src.read()  # (13, H, W)
    img = np.transpose(data, (1, 2, 0)).astype(np.float32)  # (H, W, 13)

    # Per-band min-max normalization
    for b in range(img.shape[2]):
        bmin, bmax = img[:, :, b].min(), img[:, :, b].max()
        if bmax - bmin > 0:
            img[:, :, b] = (img[:, :, b] - bmin) / (bmax - bmin)
        else:
            img[:, :, b] = 0.0

    return img, label

def build_tif_dataset(df, base_dir, shuffle=True):
    """Build a tf.data.Dataset for TIF multispectral images."""
    filepaths = [str(base_dir / row["Filename"]) for _, row in df.iterrows()]
    labels = to_categorical([label_to_int[c] for c in df["ClassName"]], NUM_CLASSES)

    images = []
    valid_labels = []
    for fp, lbl in zip(filepaths, labels):
        try:
            img, _ = load_and_preprocess_tif(fp, lbl)
            images.append(img)
            valid_labels.append(lbl)
        except Exception:
            continue

    X = np.array(images, dtype=np.float32)
    Y = np.array(valid_labels, dtype=np.float32)

    if len(X) == 0:
        print(f"⚠️  WARNING: No TIF images were loaded! Check that file paths in the CSV match the directory structure.")
        print(f"   Expected base dir: {base_dir}")
        print(f"   Sample path tried: {filepaths[0] if filepaths else 'N/A'}")

    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle and len(X) > 0:
        dataset = dataset.shuffle(buffer_size=len(X), seed=RANDOM_SEED)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset, X, Y

print("🔄 Loading TIF datasets (this may take several minutes)...")
train_tif_ds, X_train_tif, Y_train_tif = build_tif_dataset(train_ms_df, MS_DIR)
val_tif_ds,   X_val_tif,   Y_val_tif   = build_tif_dataset(val_ms_df, MS_DIR, shuffle=False)
test_tif_ds,  X_test_tif,  Y_test_tif  = build_tif_dataset(test_ms_df, MS_DIR, shuffle=False)

print(f"✅ TIF data loaded — Train: {X_train_tif.shape} | Val: {X_val_tif.shape} | Test: {X_test_tif.shape}")

# %% [markdown]
# ---
# ## 🏗️ CHUNK 10 — TIF CLASSIFICATION MODEL (Custom ResNet-50 Inspired)
# ---
# **What this does:**
# Builds a **custom ResNet-50 inspired CNN** designed specifically for
# 13-band multispectral satellite imagery.
#
# **Why custom (not pretrained)?**
# - Pretrained ResNet-50 expects **3 channels** (RGB). Our TIF data has **13 channels**.
# - No public pretrained model exists for 13-band Sentinel-2 data.
# - We must train the entire network from scratch.
#
# **Architecture Overview:**
# ```
# Input (64×64×13)
#     ↓
# Initial Conv Layer (64 filters, 7×7, stride 2) + BatchNorm + ReLU + MaxPool
#     ↓
# Stage 1: 3 residual blocks [64 → 256 filters]  (spatial: 64×64)
#     ↓
# Stage 2: 4 residual blocks [128 → 512 filters]  (spatial: 32×32)
#     ↓
# Stage 3: 3 residual blocks [256 → 1024 filters] (spatial: 16×16)
#     ↓
# Global Average Pooling
#     ↓
# Dense (512 neurons) + Dropout (0.5)
#     ↓
# Output (10 classes, Softmax)
# ```
#
# **Key Concepts:**
# - **Residual Blocks**: Allow the network to learn "changes" instead of full
#   transformations. This prevents the vanishing gradient problem in deep networks.
# - **Identity Block**: Input dimensions match output → direct skip connection.
# - **Convolutional Block**: Input dimensions differ → uses 1×1 conv to match dimensions.
# - **BatchNorm**: Normalizes activations between layers → faster, more stable training.
# - **Global Average Pooling**: Replaces fully-connected layers → fewer parameters, less overfitting.

# %%
# ============================================================
#  CHUNK 10 — TIF MODEL: CUSTOM RESNET-50 INSPIRED CNN
# ============================================================

def identity_block(x, filters):
    """
    Identity Block: the skip connection goes directly from input to output
    because the dimensions match.

    Structure: Conv(1×1) → BN → ReLU → Conv(3×3) → BN → ReLU → Conv(1×1) → BN → Add → ReLU

    Args:
        x: input tensor
        filters: tuple of 3 filter sizes (f1, f2, f3)
    """
    f1, f2, f3 = filters
    shortcut = x  # Save input for skip connection

    # First 1×1 conv — reduce dimensions
    x = layers.Conv2D(f1, (1, 1), padding="valid")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Second 3×3 conv — main feature extraction
    x = layers.Conv2D(f2, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Third 1×1 conv — restore dimensions
    x = layers.Conv2D(f3, (1, 1), padding="valid")(x)
    x = layers.BatchNormalization()(x)

    # Skip connection: add input directly to output
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def convolutional_block(x, filters, strides=(2, 2)):
    """
    Convolutional Block: the skip connection uses a 1×1 conv to match dimensions
    because spatial size or depth changes.

    Structure: Conv(1×1,stride) → BN → ReLU → Conv(3×3) → BN → ReLU → Conv(1×1) → BN
               ↓ shortcut: Conv(1×1,stride) → BN
               Add → ReLU

    Args:
        x: input tensor
        filters: tuple of 3 filter sizes (f1, f2, f3)
        strides: stride for the first conv (controls spatial downsampling)
    """
    f1, f2, f3 = filters

    # Shortcut path — 1×1 conv to match output dimensions
    shortcut = layers.Conv2D(f3, (1, 1), strides=strides, padding="valid")(x)
    shortcut = layers.BatchNormalization()(shortcut)

    # Main path
    x = layers.Conv2D(f1, (1, 1), strides=strides, padding="valid")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(f2, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(f3, (1, 1), padding="valid")(x)
    x = layers.BatchNormalization()(x)

    # Add shortcut
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def build_tif_resnet50(input_shape=(64, 64, 13), num_classes=10):
    """
    Build a custom ResNet-50 inspired model for 13-band multispectral input.

    Architecture:
    - Initial conv (64 filters, 7×7, stride 2) + BatchNorm + ReLU + MaxPool
    - Stage 1: 1 conv block + 2 identity blocks (64→256)
    - Stage 2: 1 conv block + 3 identity blocks (128→512)
    - Stage 3: 1 conv block + 2 identity blocks (256→1024)
    - Global Average Pooling → Dense(512) → Dropout → Dense(10, Softmax)
    """
    inputs = layers.Input(shape=input_shape, name="tif_input")

    # ── Initial convolution ──
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same", name="initial_conv")(inputs)
    x = layers.BatchNormalization(name="initial_bn")(x)
    x = layers.Activation("relu", name="initial_relu")(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="initial_pool")(x)

    # ── Stage 1: 3 blocks [64 → 256 filters] ──
    x = convolutional_block(x, filters=(64, 64, 256), strides=(1, 1))
    x = identity_block(x, filters=(64, 64, 256))
    x = identity_block(x, filters=(64, 64, 256))

    # ── Stage 2: 4 blocks [128 → 512 filters] ──
    x = convolutional_block(x, filters=(128, 128, 512), strides=(2, 2))
    x = identity_block(x, filters=(128, 128, 512))
    x = identity_block(x, filters=(128, 128, 512))
    x = identity_block(x, filters=(128, 128, 512))

    # ── Stage 3: 3 blocks [256 → 1024 filters] ──
    x = convolutional_block(x, filters=(256, 256, 1024), strides=(2, 2))
    x = identity_block(x, filters=(256, 256, 1024))
    x = identity_block(x, filters=(256, 256, 1024))

    # ── Classification head ──
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(512, activation="relu", name="fc_512")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="TIF_CustomResNet50")
    return model


# Build the TIF model
tif_model = build_tif_resnet50(input_shape=(64, 64, 13), num_classes=NUM_CLASSES)
tif_model.summary()

print(f"\n🏗️ TIF Model built — Total parameters: {tif_model.count_params():,}")

# %% [markdown]
# ---
# ## 🔄 CHUNK 11 — RGB CLASSIFICATION MODEL (Transfer Learning ResNet-50)
# ---
# **What this does:**
# Uses a **pretrained ResNet-50** model (trained on ImageNet — 1.4 million images,
# 1000 classes) and adapts it for our 10 land-use classes.
#
# **Why Transfer Learning?**
# Instead of training from scratch (which needs millions of images), we:
# 1. **Reuse** a model that already learned to recognize edges, textures,
#    shapes, and objects from ImageNet
# 2. **Freeze** the pretrained layers (don't change their weights)
# 3. **Replace** only the final classification "head" with our own layers
# 4. **Fine-tune** just the head for our 10 land-use classes
#
# This is like hiring an experienced photographer and just teaching them
# to classify satellite images — they already know what edges, colors,
# and patterns look like.
#
# **Input: 224×224×3** (RGB resized to ResNet's expected size)
#
# **Architecture:**
# ```
# Input (224×224×3)
#     ↓
# Pretrained ResNet-50 (frozen — all layers keep their ImageNet weights)
#     ↓
# Global Average Pooling
#     ↓
# Dense (512 neurons, ReLU)
#     ↓
# Dropout (0.5)
#     ↓
# Output (10 classes, Softmax)
# ```

# %%
# ============================================================
#  CHUNK 11 — RGB MODEL: TRANSFER LEARNING WITH RESNET-50
# ============================================================

def build_rgb_resnet50(input_shape=(224, 224, 3), num_classes=10):
    """
    Build a transfer learning model using pretrained ResNet-50.

    Steps:
    1. Load ResNet-50 with ImageNet weights (exclude top classification layer)
    2. Freeze ALL base layers (they already know good features)
    3. Add our own classification head for 10 classes
    """
    # Load pretrained ResNet-50 (without the final 1000-class layer)
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,          # Remove the original 1000-class output
        input_shape=input_shape
    )

    # Freeze all base layers — we don't want to change pretrained weights
    # (they already contain excellent feature extractors)
    base_model.trainable = False

    # Build our custom classification head on top
    inputs = layers.Input(shape=input_shape, name="rgb_input")
    x = base_model(inputs, training=False)   # training=False keeps BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, activation="relu", name="fc_512")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="RGB_TransferResNet50")
    return model


# Build the RGB model
rgb_model = build_rgb_resnet50(input_shape=(*RGB_IMG_SIZE, 3), num_classes=NUM_CLASSES)
rgb_model.summary()

trainable_params = sum([tf.size(w).numpy() for w in rgb_model.trainable_weights])
total_params = rgb_model.count_params()
print(f"\n🔄 RGB Model built — Total: {total_params:,} | Trainable: {trainable_params:,} | Frozen: {total_params - trainable_params:,}")

# %% [markdown]
# ---
# ## ⚡ CHUNK 12 — TRAINING CONFIGURATION & EXECUTION
# ---
# **What this does:**
# Compile and train both models with appropriate settings.
#
# **Training Components Explained:**
#
# | Component | What It Does |
# |-----------|-------------|
# | **Adam Optimizer** | Adjusts learning rate automatically per parameter. Best general-purpose optimizer |
# | **Categorical Crossentropy** | Loss function for multi-class classification (10 classes) |
# | **EarlyStopping** | Stops training if validation loss stops improving → prevents overfitting |
# | **ReduceLROnPlateau** | Cuts learning rate in half when validation loss plateaus → finer tuning |
# | **ModelCheckpoint** | Saves the best model weights (by validation accuracy) to disk |
#
# **Why these callbacks matter:**
# - Without EarlyStopping, the model would keep training past the optimal point
#   and start memorizing the training data (overfitting)
# - ReduceLR helps escape loss plateaus where the model gets "stuck"
# - ModelCheckpoint ensures we keep the very best model even if later epochs are worse

# %%
# ============================================================
#  CHUNK 12 — COMPILE & TRAIN BOTH MODELS
# ============================================================
import time

# ── Compile TIF Model ──
tif_model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ── Compile RGB Model ──
rgb_model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ── Callbacks ──
def get_callbacks(model_name):
    """Create training callbacks for a given model."""
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,               # Stop if no improvement for 8 epochs
            restore_best_weights=True, # Go back to the best weights
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,               # Cut LR in half
            patience=4,               # Wait 4 epochs before reducing
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=f"best_{model_name}.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

print("=" * 70)
print("  TRAINING TIF MODEL (Custom ResNet-50)")
print("=" * 70)

tif_start = time.time()
tif_history = tif_model.fit(
    train_tif_ds,
    validation_data=val_tif_ds,
    epochs=EPOCHS,
    callbacks=get_callbacks("tif_model"),
    verbose=1
)
tif_train_time = time.time() - tif_start
print(f"\n⏱️ TIF training time: {tif_train_time:.1f} seconds ({tif_train_time/60:.1f} minutes)")

print("\n" + "=" * 70)
print("  TRAINING RGB MODEL (Transfer Learning ResNet-50)")
print("=" * 70)

rgb_start = time.time()
rgb_history = rgb_model.fit(
    train_rgb_ds,
    validation_data=val_rgb_ds,
    epochs=EPOCHS,
    callbacks=get_callbacks("rgb_model"),
    verbose=1
)
rgb_train_time = time.time() - rgb_start
print(f"\n⏱️ RGB training time: {rgb_train_time:.1f} seconds ({rgb_train_time/60:.1f} minutes)")

print("\n✅ Both models trained successfully!")

# %% [markdown]
# ---
# ## 📈 CHUNK 13 — TIF MODEL LEARNING CURVES (Milestone 3)
# ---
# **What this does:**
# Plots the **training history** — how accuracy and loss changed over each epoch.
#
# **How to read learning curves:**
# - **Training accuracy going up** = model is learning
# - **Validation accuracy going up** = model generalizes well to unseen data
# - **Gap between train and val** = overfitting (model memorized training data)
# - **Both curves plateauing** = model reached its capacity, more epochs won't help
#
# **What the loss curve tells us:**
# - **Loss going down** = model is improving
# - **Training loss much lower than val loss** = overfitting
# - **Both losses similar and low** = good generalization

# %%
# ============================================================
#  CHUNK 13 — TIF MODEL LEARNING CURVES
# ============================================================

def plot_learning_curves(history, model_name, save_prefix):
    """
    Plot training/validation accuracy & loss curves.

    Args:
        history: Keras training history object
        model_name: display name for the model
        save_prefix: filename prefix for saving plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    epochs_range = range(1, len(history.history["accuracy"]) + 1)

    # ── Accuracy Plot ──
    axes[0].plot(epochs_range, history.history["accuracy"],
                 "b-o", label="Training Accuracy", linewidth=2, markersize=4)
    axes[0].plot(epochs_range, history.history["val_accuracy"],
                 "r-s", label="Validation Accuracy", linewidth=2, markersize=4)
    # Mark best validation accuracy
    best_epoch = np.argmax(history.history["val_accuracy"]) + 1
    best_val_acc = max(history.history["val_accuracy"])
    axes[0].axvline(x=best_epoch, color="green", linestyle="--", alpha=0.6,
                    label=f"Best Epoch: {best_epoch}")
    axes[0].annotate(f"{best_val_acc:.4f}",
                     xy=(best_epoch, best_val_acc),
                     xytext=(best_epoch + 1, best_val_acc - 0.03),
                     arrowprops=dict(arrowstyle="->", color="green"),
                     fontsize=10, color="green", fontweight="bold")
    axes[0].set_title(f"{model_name} — Accuracy Over Epochs", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(frameon=True, fancybox=True, shadow=True)
    axes[0].grid(True, alpha=0.3)
    axes[0].spines[["top", "right"]].set_visible(False)

    # ── Loss Plot ──
    axes[1].plot(epochs_range, history.history["loss"],
                 "b-o", label="Training Loss", linewidth=2, markersize=4)
    axes[1].plot(epochs_range, history.history["val_loss"],
                 "r-s", label="Validation Loss", linewidth=2, markersize=4)
    axes[1].axvline(x=best_epoch, color="green", linestyle="--", alpha=0.6,
                    label=f"Best Epoch: {best_epoch}")
    axes[1].set_title(f"{model_name} — Loss Over Epochs", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(frameon=True, fancybox=True, shadow=True)
    axes[1].grid(True, alpha=0.3)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle(f"📈 {model_name} — Learning Curves", fontweight="bold", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print summary
    print(f"\n📊 {model_name} Training Summary:")
    print(f"   Total epochs trained:     {len(history.history['accuracy'])}")
    print(f"   Best epoch:               {best_epoch}")
    print(f"   Best val accuracy:        {best_val_acc:.4f}")
    print(f"   Final train accuracy:     {history.history['accuracy'][-1]:.4f}")
    print(f"   Final val accuracy:       {history.history['val_accuracy'][-1]:.4f}")
    print(f"   Final train loss:         {history.history['loss'][-1]:.4f}")
    print(f"   Final val loss:           {history.history['val_loss'][-1]:.4f}")

# Plot TIF learning curves
plot_learning_curves(tif_history, "TIF Model (Custom ResNet-50)", "tif_model")

# %% [markdown]
# ---
# ## 🔍 CHUNK 14 — TIF MODEL EVALUATION (Milestone 3)
# ---
# **What this does:**
# Evaluates the TIF model on the **test set** (data the model has NEVER seen).
#
# **Metrics Explained:**
# - **Accuracy**: % of images classified correctly overall
# - **Precision**: Of all images predicted as class X, how many were actually X?
#   (High precision = few false positives)
# - **Recall**: Of all actual class X images, how many did we find?
#   (High recall = few false negatives)
# - **F1-Score**: Harmonic mean of precision and recall (best single metric)
#
# **Confusion Matrix:**
# A 10×10 grid showing where the model gets confused.
# - Diagonal = correct predictions (we want this dark/high)
# - Off-diagonal = mistakes (model predicted column class but truth was row class)
# - Look for bright off-diagonal cells → systematic confusion between class pairs

# %%
# ============================================================
#  CHUNK 14 — TIF MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_test, Y_test, model_name, save_prefix, class_names):
    """
    Full evaluation: test accuracy, confusion matrix, classification report.

    Args:
        model: trained Keras model
        X_test: test images
        Y_test: one-hot encoded test labels
        model_name: display name
        save_prefix: filename prefix for saving plots
        class_names: list of class names
    """
    print("=" * 70)
    print(f"  EVALUATION: {model_name}")
    print("=" * 70)

    # ── Get predictions ──
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(Y_test, axis=1)

    # ── Overall metrics ──
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"\n  📊 Test Loss:     {test_loss:.4f}")
    print(f"  📊 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # ── Classification Report ──
    print(f"\n  📋 Classification Report:\n")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    # ── Confusion Matrix ──
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.8})
    axes[0].set_title(f"{model_name} — Confusion Matrix (Counts)", fontweight="bold")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].tick_params(axis="x", rotation=45)

    # Normalized (percentages per class)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], linewidths=0.5, linecolor="white",
                vmin=0, vmax=1, cbar_kws={"shrink": 0.8, "label": "Recall"})
    axes[1].set_title(f"{model_name} — Normalized Confusion Matrix", fontweight="bold")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].tick_params(axis="x", rotation=45)

    plt.suptitle(f"🔍 {model_name} — Test Set Evaluation", fontweight="bold", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── Per-class accuracy bar chart ──
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = sns.color_palette("husl", len(class_names))
    bars = ax.bar(class_names, per_class_acc * 100, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(y=test_acc * 100, color="red", linestyle="--", alpha=0.7,
               label=f"Overall Accuracy: {test_acc*100:.1f}%")
    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_title(f"{model_name} — Per-Class Accuracy", fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=True, fancybox=True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_per_class_accuracy.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── Identify most confused pairs ──
    print("\n  ⚠️  Most Confused Class Pairs:")
    np.fill_diagonal(cm, 0)  # Ignore correct predictions
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j] > 0:
                confused_pairs.append((class_names[i], class_names[j], cm[i, j]))
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    for true_cls, pred_cls, count in confused_pairs[:5]:
        print(f"    {true_cls:25s} → predicted as {pred_cls:25s}  ({count} times)")

    return test_acc, test_loss, y_pred, y_true


# Evaluate TIF model
tif_test_acc, tif_test_loss, tif_y_pred, tif_y_true = evaluate_model(
    tif_model, X_test_tif, Y_test_tif,
    "TIF Model (Custom ResNet-50)", "tif_model", CLASS_NAMES
)

# %% [markdown]
# ---
# ## 📈 CHUNK 15 — RGB MODEL LEARNING CURVES (Milestone 3)
# ---
# **What this does:**
# Same learning curve analysis as Chunk 13, but for the **RGB Transfer Learning model**.
#
# **What to compare:**
# - Does the RGB model converge faster than TIF? (Expected yes — it starts with
#   pretrained features, so it has a "head start")
# - Is the gap between train/val accuracy smaller? (Expected yes — frozen layers
#   act as regularization, preventing overfitting)
# - Does it reach a higher or lower final accuracy?

# %%
# ============================================================
#  CHUNK 15 — RGB MODEL LEARNING CURVES
# ============================================================
plot_learning_curves(rgb_history, "RGB Model (Transfer Learning ResNet-50)", "rgb_model")

# %% [markdown]
# ---
# ## 🔍 CHUNK 16 — RGB MODEL EVALUATION (Milestone 3)
# ---
# **What this does:**
# Same full evaluation as Chunk 14, but for the RGB model.
#
# **Expected Differences:**
# - RGB model only sees 3 channels → misses NIR, SWIR, Red Edge information
# - May struggle more with vegetation sub-types (AnnualCrop vs PermanentCrop)
#   because they rely on NIR differences
# - Should still perform well on visually distinct classes (SeaLake, Forest)

# %%
# ============================================================
#  CHUNK 16 — RGB MODEL EVALUATION
# ============================================================
rgb_test_acc, rgb_test_loss, rgb_y_pred, rgb_y_true = evaluate_model(
    rgb_model, X_test_rgb, Y_test_rgb,
    "RGB Model (Transfer Learning ResNet-50)", "rgb_model", CLASS_NAMES
)

# %% [markdown]
# ---
# ## ⚖️ CHUNK 17 — RGB vs TIF COMPARISON & FINAL SUMMARY
# ---
# **What this does:**
# A side-by-side comparison of both models across all key metrics.
#
# **Why two models?**
# - **RGB**: Widely available, works with transfer learning, fast to train
# - **TIF**: Uses all 13 spectral bands, captures invisible light information
#   (NIR, SWIR), potentially more accurate for remote sensing tasks
#
# **Trade-offs:**
# | Aspect | RGB (Transfer Learning) | TIF (Custom CNN) |
# |--------|------------------------|-------------------|
# | Input | 3 channels (visible light) | 13 channels (full spectrum) |
# | Training | Fast (only head trains) | Slow (trains from scratch) |
# | Data needed | Less (leverages ImageNet) | More (learns everything) |
# | Accuracy | Good for visual classes | Better for spectral separation |

# %%
# ============================================================
#  CHUNK 17 — RGB vs TIF COMPARISON & FINAL SUMMARY
# ============================================================
print("=" * 70)
print("  ⚖️  RGB vs TIF — COMPLETE MODEL COMPARISON")
print("=" * 70)

# ── Comparison Table ──
comparison_data = {
    "Metric": [
        "Architecture",
        "Input Shape",
        "Total Parameters",
        "Trainable Parameters",
        "Training Strategy",
        "Training Time",
        "Test Accuracy",
        "Test Loss",
        "Best For",
    ],
    "RGB Model (Transfer Learning)": [
        "ResNet-50 (Pretrained ImageNet)",
        f"{RGB_IMG_SIZE[0]}×{RGB_IMG_SIZE[1]}×3",
        f"{rgb_model.count_params():,}",
        f"{sum(tf.size(w).numpy() for w in rgb_model.trainable_weights):,}",
        "Fine-tuning (frozen base)",
        f"{rgb_train_time:.1f}s ({rgb_train_time/60:.1f} min)",
        f"{rgb_test_acc:.4f} ({rgb_test_acc*100:.2f}%)",
        f"{rgb_test_loss:.4f}",
        "General RGB photos, fast training",
    ],
    "TIF Model (Custom ResNet-50)": [
        "Custom ResNet-50 Inspired",
        f"{TIF_IMG_SIZE[0]}×{TIF_IMG_SIZE[1]}×13",
        f"{tif_model.count_params():,}",
        f"{tif_model.count_params():,} (all)",
        "Training from scratch",
        f"{tif_train_time:.1f}s ({tif_train_time/60:.1f} min)",
        f"{tif_test_acc:.4f} ({tif_test_acc*100:.2f}%)",
        f"{tif_test_loss:.4f}",
        "Specialized satellite multi-spectral",
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# ── Visual Comparison ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy comparison
models_names = ["RGB\n(Transfer Learning)", "TIF\n(Custom ResNet-50)"]
accuracies = [rgb_test_acc * 100, tif_test_acc * 100]
colors = ["#3498db", "#e74c3c"]

bars = axes[0].bar(models_names, accuracies, color=colors, edgecolor="white",
                   linewidth=1.5, width=0.5)
for bar, val in zip(bars, accuracies):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.2f}%", ha="center", fontsize=14, fontweight="bold")
axes[0].set_title("Test Accuracy Comparison", fontweight="bold", fontsize=14)
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_ylim(0, 105)
axes[0].spines[["top", "right"]].set_visible(False)

# Training time comparison
times = [rgb_train_time / 60, tif_train_time / 60]
bars = axes[1].bar(models_names, times, color=colors, edgecolor="white",
                   linewidth=1.5, width=0.5)
for bar, val in zip(bars, times):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{val:.1f} min", ha="center", fontsize=14, fontweight="bold")
axes[1].set_title("Training Time Comparison", fontweight="bold", fontsize=14)
axes[1].set_ylabel("Time (minutes)")
axes[1].spines[["top", "right"]].set_visible(False)

plt.suptitle("⚖️ RGB vs TIF — Model Comparison", fontweight="bold", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Final Summary ──
print("""
┌─────────────────────────────────────────────────────────────────────┐
│           🏆  MILESTONE 2 & 3 — COMPLETE SUMMARY                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ✅ Milestone 2 — Model Development:                               │
│     • TIF Model: Custom ResNet-50 inspired CNN (13-band input)     │
│     • RGB Model: Transfer Learning ResNet-50 (ImageNet pretrained) │
│     • Both models compiled with Adam optimizer + crossentropy loss │
│     • Training with EarlyStopping, ReduceLR, and ModelCheckpoint   │
│                                                                     │
│  ✅ Milestone 3 — Optimization & Performance Tuning:               │
│     • Learning curves plotted for both models                      │
│     • Confusion matrices (raw + normalized) generated              │
│     • Per-class accuracy, precision, recall, F1 computed           │
│     • Most confused class pairs identified                         │
│     • RGB vs TIF side-by-side comparison completed                 │
│                                                                     │
│  📁 Saved Artifacts:                                               │
│     • best_tif_model.keras     (best TIF model weights)            │
│     • best_rgb_model.keras     (best RGB model weights)            │
│     • tif_model_learning_curves.png                                │
│     • tif_model_confusion_matrix.png                               │
│     • rgb_model_learning_curves.png                                │
│     • rgb_model_confusion_matrix.png                               │
│     • model_comparison.png                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

# Save models
tif_model.save("tif_classification_model.keras")
rgb_model.save("rgb_classification_model.keras")
print("💾 Models saved: tif_classification_model.keras, rgb_classification_model.keras")
print("✅ Milestones 2 & 3 COMPLETE!")
