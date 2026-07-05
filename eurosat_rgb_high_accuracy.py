# %% [markdown]
# # 🛰️ EuroSAT — HIGH-ACCURACY RGB Classification Pipeline
# ### Target: ≥ 96% Test Accuracy (Baseline Friend: MobileNetV2 @ 94%)
# ---
#
# **Strategy:**
# 1. EfficientNetV2-S backbone (pretrained on ImageNet)
# 2. Two-phase training: frozen warmup → progressive fine-tuning
# 3. Advanced augmentation: CutMix + MixUp + strong color/geometry transforms
# 4. Generalized Mean (GeM) Pooling instead of simple Global Average Pooling
# 5. Label Smoothing loss + AdamW optimizer + Cosine LR Annealing
# 6. Test-Time Augmentation (TTA) for free accuracy boost at inference
#
# **Compatibility:** Same CSV splits and BASE_DIR as eurosat_model_pipeline.py

# %%
# ============================================================
#  SECTION 1 — IMPORTS, CONFIGURATION & REPRODUCIBILITY
# ============================================================
import os, sys, json, warnings, gc, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.utils import to_categorical

# Metrics
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder

# Suppress noisy warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Plot styling
sns.set_theme(style="darkgrid", palette="deep", font_scale=1.1)
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "figure.dpi": 120,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.family": "sans-serif",
})

# GPU setup
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"GPU detected: {gpus[0].name}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    MIXED_PRECISION = True
    print("Mixed precision (float16) ENABLED")
else:
    print("No GPU detected. Training will be slow on CPU.")
    MIXED_PRECISION = False

print(f"TensorFlow version: {tf.__version__}")
print("All imports successful.")

# %%
# ============================================================
#  CONFIGURATION — UPDATE BASE_DIR TO MATCH YOUR ENVIRONMENT
# ============================================================
# Kaggle/Colab: update BASE_DIR as needed
# e.g., BASE_DIR = Path("/kaggle/input/eurosat-dataset")

BASE_DIR = Path("archive")       # <-- same as eurosat_model_pipeline.py
RGB_DIR  = BASE_DIR / "EuroSAT"  # folder with class subfolders + train/val/test CSV

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]
NUM_CLASSES = len(CLASS_NAMES)

# Image & Training Hyperparameters
IMG_SIZE        = (224, 224)  # EfficientNetV2-S native input
BATCH_SIZE      = 32
PHASE_A_EPOCHS  = 15          # Head-only warmup (backbone frozen)
PHASE_B_EPOCHS  = 50          # Fine-tuning (EarlyStopping will cut short)
PHASE_A_LR      = 1e-3        # Head learning rate in Phase A
PHASE_B_HEAD_LR = 1e-4        # Head learning rate in Phase B
PHASE_B_BASE_LR = 1e-5        # Backbone LR in Phase B (10x smaller)
WEIGHT_DECAY    = 1e-4        # AdamW weight decay
LABEL_SMOOTHING_A = 0.10      # Label smoothing in Phase A
LABEL_SMOOTHING_B = 0.05      # Label smoothing in Phase B (model more stable)
UNFREEZE_LAYERS = 100         # Unfreeze top N backbone layers in Phase B

label_to_int = {name: i for i, name in enumerate(CLASS_NAMES)}

print(f"RGB dir  : {RGB_DIR}")
print(f"Classes  : {NUM_CLASSES}")
print(f"Batch    : {BATCH_SIZE} | Phase-A epochs: {PHASE_A_EPOCHS} | Phase-B epochs: {PHASE_B_EPOCHS}")

# %%
# ============================================================
#  SECTION 2 — LOAD CSV SPLITS
# ============================================================
def load_csv(base_dir, name):
    df = pd.read_csv(base_dir / name)
    df.columns = [c.strip() for c in df.columns]
    return df

train_df = load_csv(RGB_DIR, "train.csv")
val_df   = load_csv(RGB_DIR, "validation.csv")
test_df  = load_csv(RGB_DIR, "test.csv")

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# %%
# ============================================================
#  SECTION 3 — IMAGE LOADING & PREPROCESSING
# ============================================================
# EfficientNetV2 expects pixel values in [0, 255] float32.
# It applies its OWN internal normalization (include_preprocessing=True).
# Do NOT apply ImageNet mean/std manually.

def load_image(filepath):
    """Load and resize an RGB image to (224, 224). Returns float32 in [0, 255]."""
    img = Image.open(filepath).convert("RGB")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    return np.array(img, dtype=np.float32)  # range [0, 255]

def load_dataset_arrays(df, base_dir, desc="Loading"):
    """Load all images into NumPy arrays. Returns X (N,224,224,3), Y (N,10)."""
    filepaths = [str(base_dir / row["Filename"]) for _, row in df.iterrows()]
    labels    = to_categorical(
        [label_to_int[c] for c in df["ClassName"]], NUM_CLASSES
    )
    images, valid_labels = [], []
    for i, (fp, lbl) in enumerate(zip(filepaths, labels)):
        if i % 2000 == 0:
            print(f"  {desc}: {i}/{len(filepaths)}", end="\r")
        try:
            img = load_image(fp)
            images.append(img)
            valid_labels.append(lbl)
        except Exception:
            continue

    X = np.array(images, dtype=np.float32)
    Y = np.array(valid_labels, dtype=np.float32)
    print(f"  {desc}: {len(X)}/{len(filepaths)} images loaded")
    if len(X) == 0:
        raise RuntimeError(
            f"No images loaded from {base_dir}. "
            "Check that Filename column paths are correct."
        )
    return X, Y

print("Loading datasets (this may take a few minutes)...")
X_train, Y_train = load_dataset_arrays(train_df, RGB_DIR, "Train")
X_val,   Y_val   = load_dataset_arrays(val_df,   RGB_DIR, "Val  ")
X_test,  Y_test  = load_dataset_arrays(test_df,  RGB_DIR, "Test ")

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# %%
# ============================================================
#  SECTION 4 — ADVANCED AUGMENTATION: CutMix & MixUp
# ============================================================
# MixUp: blend two images linearly -- img = lambda*imgA + (1-lambda)*imgB
#        label = lambda*labelA + (1-lambda)*labelB
#        Forces network to predict smoothly between classes.
#
# CutMix: cut a rectangle from image B and paste onto image A.
#         Labels are mixed proportional to the pasted area.
#         Forces spatial feature learning.

def mixup(images, labels, alpha=0.4):
    """Apply MixUp augmentation to a batch."""
    batch_size = images.shape[0]
    lam = np.random.beta(alpha, alpha)
    indices = np.random.permutation(batch_size)
    images2 = images[indices]
    labels2 = labels[indices]
    mixed_images = lam * images + (1.0 - lam) * images2
    mixed_labels = lam * labels + (1.0 - lam) * labels2
    return mixed_images.astype(np.float32), mixed_labels.astype(np.float32)


def rand_bbox(height, width, lam):
    """Generate a random bounding box for CutMix."""
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(height * cut_ratio)
    cut_w = int(width  * cut_ratio)
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    x1 = np.clip(cx - cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y2 = np.clip(cy + cut_h // 2, 0, height)
    return x1, y1, x2, y2


def cutmix(images, labels, alpha=1.0):
    """Apply CutMix augmentation to a batch."""
    if hasattr(images, "numpy"):
        images = images.numpy()
    if hasattr(labels, "numpy"):
        labels = labels.numpy()
    batch_size, height, width = images.shape[0], images.shape[1], images.shape[2]
    lam = np.random.beta(alpha, alpha)
    indices = np.random.permutation(batch_size)
    images2 = images[indices]
    labels2 = labels[indices]
    x1, y1, x2, y2 = rand_bbox(height, width, lam)
    mixed_images = images.copy()
    mixed_images[:, y1:y2, x1:x2, :] = images2[:, y1:y2, x1:x2, :]
    actual_lam = 1.0 - ((x2 - x1) * (y2 - y1) / (width * height))
    mixed_labels = actual_lam * labels + (1.0 - actual_lam) * labels2
    return mixed_images.astype(np.float32), mixed_labels.astype(np.float32)


CUTMIX_PROB = 0.50  # 50% chance of CutMix per batch
MIXUP_PROB  = 0.30  # 30% chance of MixUp per batch (when CutMix is not applied)


def apply_batch_augmentation(images, labels):
    """Randomly apply CutMix or MixUp to a training batch."""
    if hasattr(images, "numpy"):
        images = images.numpy()
    if hasattr(labels, "numpy"):
        labels = labels.numpy()
    r = np.random.rand()
    if r < CUTMIX_PROB:
        return cutmix(images, labels, alpha=1.0)
    elif r < CUTMIX_PROB + MIXUP_PROB:
        return mixup(images, labels, alpha=0.4)
    return images, labels

print("CutMix + MixUp augmentation functions defined.")

# %%
# ============================================================
#  SECTION 5 — KERAS AUGMENTATION LAYERS (in-graph, GPU-accel)
# ============================================================
# These run inside the model graph during training on the GPU.

def build_augmentation_pipeline():
    """Build sequential augmentation applied during training."""
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),           # +/- 15% of 360 degrees
        layers.RandomZoom(0.10),               # +/- 10% zoom
        layers.RandomContrast(0.15),           # +/- 15% contrast change
        layers.RandomTranslation(0.05, 0.05),  # +/- 5% shift in x and y
    ], name="augmentation")

augmentation_pipeline = build_augmentation_pipeline()

# %%
# ============================================================
#  SECTION 6 — tf.data PIPELINE
# ============================================================

def make_dataset(X, Y, augment=False, shuffle=True):
    """
    Build a tf.data.Dataset from NumPy arrays.
    X: float32 images [0, 255], shape (N, 224, 224, 3)
    Y: one-hot labels, shape (N, 10)
    """
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=RANDOM_SEED)
    ds = ds.batch(BATCH_SIZE)

    if augment:
        def aug_fn(x, y):
            x = augmentation_pipeline(x, training=True)
            return x, y
        ds = ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(X_train, Y_train, augment=True,  shuffle=True)
val_ds   = make_dataset(X_val,   Y_val,   augment=False, shuffle=False)
test_ds  = make_dataset(X_test,  Y_test,  augment=False, shuffle=False)

print("tf.data pipelines built.")
print(f"  Train batches : {len(train_ds)}")
print(f"  Val   batches : {len(val_ds)}")
print(f"  Test  batches : {len(test_ds)}")

# %%
# ============================================================
#  SECTION 7 — GENERALIZED MEAN (GeM) POOLING LAYER
# ============================================================
# GeM Pooling: output = (mean(x^p))^(1/p), learnable exponent p.
# p=1 -> Global Average Pooling
# p->inf -> Global Max Pooling
# For satellite imagery p typically learns to ~3-6.
# More selective than GAP: focuses on dominant/discriminative features.
# Reference: Radenovic et al., TPAMI 2019

class GeMPooling(layers.Layer):
    """
    Generalized Mean Pooling.
    Learnable exponent p controls selectivity between
    Global Average Pooling (p=1) and Global Max Pooling (p->inf).
    """
    def __init__(self, p_init=3.0, p_trainable=True, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.p_init      = p_init
        self.p_trainable = p_trainable
        self.eps         = eps

    def build(self, input_shape):
        self.p = self.add_weight(
            name="gem_p",
            shape=(1,),
            initializer=tf.constant_initializer(self.p_init),
            trainable=self.p_trainable,
            dtype=tf.float32,
            constraint=tf.keras.constraints.NonNeg()
        )
        super().build(input_shape)

    def call(self, inputs):
        # Cast to float32 for numerical stability under mixed precision
        x = tf.cast(inputs, tf.float32)
        x = tf.clip_by_value(x, self.eps, tf.reduce_max(x))
        x = tf.pow(x, self.p)
        x = tf.reduce_mean(x, axis=[1, 2])  # spatial mean: (B, C)
        x = tf.pow(x, 1.0 / self.p)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "p_init": self.p_init,
            "p_trainable": self.p_trainable,
            "eps": self.eps
        })
        return config

print("GeMPooling layer defined.")

# %%
# ============================================================
#  SECTION 8 — MODEL ARCHITECTURE: EfficientNetV2-S + GeM Head
# ============================================================
# Architecture:
#   Input (224x224x3, float32 in [0,255])
#     -> EfficientNetV2-S backbone (pretrained ImageNet, initially FROZEN)
#     -> GeM Pooling (learnable exponent p, initialized at 3.0)
#     -> Dense(512, swish) -> BatchNorm -> Dropout(0.4)
#     -> Dense(256, swish) -> BatchNorm -> Dropout(0.3)
#     -> Dense(10, softmax, dtype=float32)
#
# Note: EfficientNetV2S includes its own preprocessing (include_preprocessing=True)
# so we do NOT subtract ImageNet mean/std manually.

def build_efficientnetv2s_model(num_classes=10, input_shape=(224, 224, 3)):
    """
    Build the high-accuracy RGB classification model.
    Returns (model, backbone) with backbone FROZEN for Phase A.
    """
    # Load pretrained backbone
    base_model = EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        include_preprocessing=True,  # built-in normalization applied automatically
    )
    base_model.trainable = False  # Freeze for Phase A

    # Build head
    inputs = layers.Input(shape=input_shape, name="rgb_input")
    x = base_model(inputs, training=False)  # inference mode: BN stays frozen

    # GeM Pooling
    x = GeMPooling(p_init=3.0, p_trainable=True, name="gem_pool")(x)

    # Dense block 1
    x = layers.Dense(512, name="fc_512")(x)
    x = layers.BatchNormalization(name="bn_512")(x)
    x = layers.Activation("swish", name="swish_512")(x)
    x = layers.Dropout(0.4, name="drop_512")(x)

    # Dense block 2
    x = layers.Dense(256, name="fc_256")(x)
    x = layers.BatchNormalization(name="bn_256")(x)
    x = layers.Activation("swish", name="swish_256")(x)
    x = layers.Dropout(0.3, name="drop_256")(x)

    # Output -- always float32 even with mixed precision
    outputs = layers.Dense(num_classes, activation="softmax",
                           dtype="float32", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs,
                         name="RGB_EfficientNetV2S_GeM")
    return model, base_model


def unfreeze_top_layers(model, base_model, n_layers=UNFREEZE_LAYERS):
    """Unfreeze the top n_layers of the backbone for fine-tuning."""
    base_model.trainable = True
    total = len(base_model.layers)
    freeze_until = total - n_layers
    for i, layer in enumerate(base_model.layers):
        layer.trainable = (i >= freeze_until)
    frozen   = sum(1 for l in base_model.layers if not l.trainable)
    unfrozen = sum(1 for l in base_model.layers if l.trainable)
    print(f"  Backbone layers -- Frozen: {frozen} | Unfrozen: {unfrozen}")
    return model


# Build the model
rgb_model, backbone = build_efficientnetv2s_model(
    num_classes=NUM_CLASSES,
    input_shape=(*IMG_SIZE, 3)
)
rgb_model.summary(line_length=100)

trainable = sum(tf.size(w).numpy() for w in rgb_model.trainable_weights)
total     = rgb_model.count_params()
print(f"\nModel built -- Total: {total:,} | Trainable (Phase A): {trainable:,}")

# %%
# ============================================================
#  SECTION 9 — COSINE LR SCHEDULE HELPER
# ============================================================

def cosine_lr_schedule(epoch, max_lr, warmup_epochs=2, total_epochs=50, min_lr=1e-8):
    """
    Cosine annealing learning rate schedule with linear warmup.
    Used per-epoch in the Phase B training loop.
    """
    if epoch < warmup_epochs:
        return max_lr * (epoch + 1) / warmup_epochs
    cos_epoch = epoch - warmup_epochs
    cos_total = total_epochs - warmup_epochs
    cos_inner = np.pi * cos_epoch / max(cos_total, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + np.cos(cos_inner))

print("Cosine LR schedule helper defined.")

# %%
# ============================================================
#  SECTION 10 — PHASE A: HEAD WARMUP (Backbone FROZEN)
# ============================================================
# Train only the classification head + GeM pooling layer.
# CutMix and MixUp are applied per-batch inside the custom training loop.
# Label smoothing = 0.10 to prevent overconfident predictions early on.

print("=" * 70)
print("  PHASE A -- HEAD WARMUP (Backbone FROZEN)")
print(f"  Optimizer : Adam(lr={PHASE_A_LR})")
print(f"  Loss      : CategoricalCrossentropy(label_smoothing={LABEL_SMOOTHING_A})")
print(f"  Epochs    : up to {PHASE_A_EPOCHS} (EarlyStopping patience=5)")
print("=" * 70)

optimizer_a = optimizers.Adam(learning_rate=PHASE_A_LR)
loss_fn_a   = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING_A)

phase_a_train_accs,   phase_a_val_accs   = [], []
phase_a_train_losses, phase_a_val_losses = [], []

best_val_acc_a  = 0.0
best_weights_a  = None
patience_ctr_a  = 0
PATIENCE_A      = 5

print(f"\nStarting Phase A training...")
phase_a_start = time.time()

for epoch in range(PHASE_A_EPOCHS):
    epoch_losses, epoch_accs = [], []

    for step, (x_batch, y_batch) in enumerate(train_ds):
        # Apply CutMix / MixUp stochastically
        x_np, y_np = apply_batch_augmentation(x_batch, y_batch)
        x_batch = tf.constant(x_np, dtype=tf.float32)
        y_batch = tf.constant(y_np, dtype=tf.float32)

        with tf.GradientTape() as tape:
            preds = rgb_model(x_batch, training=True)
            loss  = loss_fn_a(y_batch, preds)

        grads = tape.gradient(loss, rgb_model.trainable_weights)
        optimizer_a.apply_gradients(zip(grads, rgb_model.trainable_weights))

        acc = float(tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(preds, 1),
                         tf.argmax(tf.cast(y_batch, tf.float32), 1)),
                tf.float32
            )
        ))
        epoch_losses.append(float(loss))
        epoch_accs.append(acc)

    train_loss = np.mean(epoch_losses)
    train_acc  = np.mean(epoch_accs)

    # Validation pass
    val_losses_ep, val_accs_ep = [], []
    for x_vb, y_vb in val_ds:
        v_preds = rgb_model(x_vb, training=False)
        v_loss  = loss_fn_a(y_vb, v_preds)
        v_acc   = float(tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(v_preds, 1), tf.argmax(y_vb, 1)), tf.float32)
        ))
        val_losses_ep.append(float(v_loss))
        val_accs_ep.append(v_acc)

    val_loss = np.mean(val_losses_ep)
    val_acc  = np.mean(val_accs_ep)

    phase_a_train_accs.append(train_acc)
    phase_a_val_accs.append(val_acc)
    phase_a_train_losses.append(train_loss)
    phase_a_val_losses.append(val_loss)

    print(f"  Epoch {epoch+1:2d}/{PHASE_A_EPOCHS} -- "
          f"loss: {train_loss:.4f} | acc: {train_acc:.4f} | "
          f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

    if val_acc > best_val_acc_a:
        best_val_acc_a = val_acc
        best_weights_a = rgb_model.get_weights()
        patience_ctr_a = 0
        rgb_model.save("best_rgb_phase_a.keras")
        print(f"  --> New best val_acc: {best_val_acc_a:.4f} -- weights saved")
    else:
        patience_ctr_a += 1
        if patience_ctr_a >= PATIENCE_A:
            print(f"  EarlyStopping at epoch {epoch+1} (patience={PATIENCE_A})")
            break

rgb_model.set_weights(best_weights_a)
phase_a_time = time.time() - phase_a_start
print(f"\nPhase A done in {phase_a_time/60:.1f} min | Best val_acc: {best_val_acc_a:.4f}")

# %%
# ============================================================
#  SECTION 11 — PHASE B: PROGRESSIVE FINE-TUNING
# ============================================================
# Unfreeze top UNFREEZE_LAYERS of EfficientNetV2-S backbone.
# Use TWO separate AdamW optimizers:
#   - backbone optimizer: 10x smaller LR to avoid catastrophic forgetting
#   - head optimizer   : standard fine-tuning LR
# Both use cosine annealing from Phase B start.
# Label smoothing reduced to 0.05 (model is already calibrated from Phase A).

print("\n" + "=" * 70)
print("  PHASE B -- PROGRESSIVE FINE-TUNING")
print(f"  Backbone LR : {PHASE_B_BASE_LR}")
print(f"  Head LR     : {PHASE_B_HEAD_LR}")
print(f"  Optimizer   : AdamW(weight_decay={WEIGHT_DECAY})")
print(f"  Loss        : CategoricalCrossentropy(label_smoothing={LABEL_SMOOTHING_B})")
print(f"  Epochs      : up to {PHASE_B_EPOCHS} (EarlyStopping patience=10)")
print("=" * 70)

# Unfreeze top backbone layers
rgb_model = unfreeze_top_layers(rgb_model, backbone, n_layers=UNFREEZE_LAYERS)

# Separate trainable weight lists for dual-optimizer approach
backbone_var_names = {w.name for w in backbone.trainable_weights}
backbone_weights   = [w for w in rgb_model.trainable_weights
                      if w.name in backbone_var_names]
head_weights       = [w for w in rgb_model.trainable_weights
                      if w.name not in backbone_var_names]

print(f"  Backbone trainable params : {sum(tf.size(w).numpy() for w in backbone_weights):,}")
print(f"  Head trainable params     : {sum(tf.size(w).numpy() for w in head_weights):,}")

optimizer_backbone = optimizers.AdamW(
    learning_rate=PHASE_B_BASE_LR,
    weight_decay=WEIGHT_DECAY,
    name="adamw_backbone"
)
optimizer_head = optimizers.AdamW(
    learning_rate=PHASE_B_HEAD_LR,
    weight_decay=WEIGHT_DECAY,
    name="adamw_head"
)
loss_fn_b = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING_B)

phase_b_train_accs,   phase_b_val_accs   = [], []
phase_b_train_losses, phase_b_val_losses = [], []

best_val_acc_b  = best_val_acc_a
best_weights_b  = rgb_model.get_weights()
patience_ctr_b  = 0
PATIENCE_B      = 10

print(f"\nStarting Phase B training...")
phase_b_start = time.time()

for epoch in range(PHASE_B_EPOCHS):
    # Cosine annealing LR for this epoch
    new_head_lr = cosine_lr_schedule(
        epoch,
        max_lr=PHASE_B_HEAD_LR,
        warmup_epochs=2,
        total_epochs=PHASE_B_EPOCHS,
        min_lr=1e-8
    )
    new_base_lr = new_head_lr / 10.0
    optimizer_head.learning_rate.assign(new_head_lr)
    optimizer_backbone.learning_rate.assign(new_base_lr)

    epoch_losses, epoch_accs = [], []

    for step, (x_batch, y_batch) in enumerate(train_ds):
        x_np, y_np = apply_batch_augmentation(x_batch, y_batch)
        x_batch = tf.constant(x_np, dtype=tf.float32)
        y_batch = tf.constant(y_np, dtype=tf.float32)

        with tf.GradientTape() as tape:
            preds = rgb_model(x_batch, training=True)
            loss  = loss_fn_b(y_batch, preds)

        grads = tape.gradient(loss, rgb_model.trainable_weights)

        # Split gradients by variable group
        backbone_grads = [g for g, w in zip(grads, rgb_model.trainable_weights)
                          if w.name in backbone_var_names]
        head_grads     = [g for g, w in zip(grads, rgb_model.trainable_weights)
                          if w.name not in backbone_var_names]

        if backbone_weights and backbone_grads:
            optimizer_backbone.apply_gradients(zip(backbone_grads, backbone_weights))
        if head_weights and head_grads:
            optimizer_head.apply_gradients(zip(head_grads, head_weights))

        acc = float(tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(preds, 1),
                         tf.argmax(tf.cast(y_batch, tf.float32), 1)),
                tf.float32
            )
        ))
        epoch_losses.append(float(loss))
        epoch_accs.append(acc)

    train_loss = np.mean(epoch_losses)
    train_acc  = np.mean(epoch_accs)

    val_losses_ep, val_accs_ep = [], []
    for x_vb, y_vb in val_ds:
        v_preds = rgb_model(x_vb, training=False)
        v_loss  = loss_fn_b(y_vb, v_preds)
        v_acc   = float(tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(v_preds, 1), tf.argmax(y_vb, 1)), tf.float32)
        ))
        val_losses_ep.append(float(v_loss))
        val_accs_ep.append(v_acc)

    val_loss = np.mean(val_losses_ep)
    val_acc  = np.mean(val_accs_ep)

    phase_b_train_accs.append(train_acc)
    phase_b_val_accs.append(val_acc)
    phase_b_train_losses.append(train_loss)
    phase_b_val_losses.append(val_loss)

    print(f"  Epoch {epoch+1:2d}/{PHASE_B_EPOCHS} -- "
          f"loss: {train_loss:.4f} | acc: {train_acc:.4f} | "
          f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | "
          f"head_lr: {new_head_lr:.2e}")

    if val_acc > best_val_acc_b:
        best_val_acc_b = val_acc
        best_weights_b = rgb_model.get_weights()
        patience_ctr_b = 0
        rgb_model.save("best_rgb_phase_b.keras")
        print(f"  --> New best val_acc: {best_val_acc_b:.4f} -- weights saved")
    else:
        patience_ctr_b += 1
        if patience_ctr_b >= PATIENCE_B:
            print(f"  EarlyStopping at epoch {epoch+1} (patience={PATIENCE_B})")
            break

rgb_model.set_weights(best_weights_b)
phase_b_time = time.time() - phase_b_start
print(f"\nPhase B done in {phase_b_time/60:.1f} min | Best val_acc: {best_val_acc_b:.4f}")

# %%
# ============================================================
#  SECTION 12 — TEST-TIME AUGMENTATION (TTA)
# ============================================================
# TTA averages predictions over 6 augmented views of each test image.
# This is a zero-training-cost technique that typically gives +0.5-1.5%.
#
# Views:
#  0: Original
#  1: Horizontal flip
#  2: Vertical flip
#  3: H + V flip
#  4: 90 degree rotation
#  5: 90 degree rotation + horizontal flip

def tta_predict(model, X, batch_size=32):
    """
    Test-Time Augmentation: average softmax probs over 6 augmented views.
    Returns array of shape (N, num_classes).
    """
    n = len(X)
    all_probs = np.zeros((n, NUM_CLASSES), dtype=np.float32)

    views = [
        X,                            # 0: original
        X[:, :, ::-1, :],             # 1: horizontal flip
        X[:, ::-1, :, :],             # 2: vertical flip
        X[:, ::-1, ::-1, :],          # 3: H + V flip
        np.rot90(X, k=1, axes=(1,2)), # 4: 90 deg rotation
        np.rot90(X, k=1, axes=(1,2))[:, :, ::-1, :],  # 5: 90 deg + H flip
    ]

    for v_idx, X_view in enumerate(views):
        view_probs = []
        for i in range(0, n, batch_size):
            batch = X_view[i:i+batch_size]
            probs = model(tf.constant(batch, dtype=tf.float32), training=False).numpy()
            view_probs.append(probs)
        all_probs += np.concatenate(view_probs, axis=0)
        print(f"  TTA view {v_idx+1}/6 done", end="\r")

    all_probs /= len(views)
    print(f"\n  TTA complete over {len(views)} views.")
    return all_probs


print("\nRunning Test-Time Augmentation (6 views)...")
tta_start = time.time()
tta_probs = tta_predict(rgb_model, X_test, batch_size=BATCH_SIZE)
tta_preds = np.argmax(tta_probs, axis=1)
y_true    = np.argmax(Y_test, axis=1)
tta_acc   = accuracy_score(y_true, tta_preds)
print(f"TTA inference time: {time.time()-tta_start:.1f}s")
print(f"\nTTA Test Accuracy: {tta_acc:.4f} ({tta_acc*100:.2f}%)")

# Standard prediction (no TTA) for comparison
std_probs_list = []
for i in range(0, len(X_test), BATCH_SIZE):
    batch_probs = rgb_model(
        tf.constant(X_test[i:i+BATCH_SIZE], dtype=tf.float32),
        training=False
    ).numpy()
    std_probs_list.append(batch_probs)
std_probs = np.concatenate(std_probs_list, axis=0)
std_preds = np.argmax(std_probs, axis=1)
std_acc   = accuracy_score(y_true, std_preds)
print(f"Standard Test Accuracy (no TTA): {std_acc:.4f} ({std_acc*100:.2f}%)")
print(f"TTA boost: +{(tta_acc - std_acc)*100:.2f}%")

# %%
# ============================================================
#  SECTION 13 — LEARNING CURVES (Phase A + Phase B combined)
# ============================================================

def plot_learning_curves_combined(
    phase_a_accs, phase_a_losses, phase_a_val_accs, phase_a_val_losses,
    phase_b_accs, phase_b_losses, phase_b_val_accs, phase_b_val_losses
):
    total_a = len(phase_a_accs)
    total_b = len(phase_b_accs)
    epochs_a = list(range(1, total_a + 1))
    epochs_b = list(range(total_a + 1, total_a + total_b + 1))

    fig, axes = plt.subplots(1, 2, figsize=(22, 7))

    # Accuracy panel
    axes[0].plot(epochs_a, phase_a_accs,     "b-o", label="Train Acc (Phase A)", lw=2, ms=4)
    axes[0].plot(epochs_a, phase_a_val_accs, "b--s", label="Val Acc (Phase A)",  lw=2, ms=4)
    axes[0].plot(epochs_b, phase_b_accs,     "g-o", label="Train Acc (Phase B)", lw=2, ms=4)
    axes[0].plot(epochs_b, phase_b_val_accs, "g--s", label="Val Acc (Phase B)",  lw=2, ms=4)
    axes[0].axvline(x=total_a, color="orange", linestyle=":", lw=2,
                    label=f"Phase A->B (ep {total_a})")
    all_val = phase_a_val_accs + phase_b_val_accs
    all_ep  = epochs_a + epochs_b
    best_i  = int(np.argmax(all_val))
    axes[0].axvline(x=all_ep[best_i], color="red", linestyle="--", alpha=0.7,
                    label=f"Best ep {all_ep[best_i]} ({all_val[best_i]:.4f})")
    axes[0].set_title("EfficientNetV2-S -- Accuracy Over Epochs", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(frameon=True, fancybox=True, fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Loss panel
    axes[1].plot(epochs_a, phase_a_losses,     "b-o", label="Train Loss (Phase A)", lw=2, ms=4)
    axes[1].plot(epochs_a, phase_a_val_losses, "b--s", label="Val Loss (Phase A)",  lw=2, ms=4)
    axes[1].plot(epochs_b, phase_b_losses,     "g-o", label="Train Loss (Phase B)", lw=2, ms=4)
    axes[1].plot(epochs_b, phase_b_val_losses, "g--s", label="Val Loss (Phase B)",  lw=2, ms=4)
    axes[1].axvline(x=total_a, color="orange", linestyle=":", lw=2, label="Phase A->B")
    axes[1].set_title("EfficientNetV2-S -- Loss Over Epochs", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(frameon=True, fancybox=True, fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("EfficientNetV2-S -- Two-Phase Training Curves",
                 fontweight="bold", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig("efficientnetv2s_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: efficientnetv2s_learning_curves.png")

plot_learning_curves_combined(
    phase_a_train_accs, phase_a_train_losses, phase_a_val_accs, phase_a_val_losses,
    phase_b_train_accs, phase_b_train_losses, phase_b_val_accs, phase_b_val_losses
)

# %%
# ============================================================
#  SECTION 14 — FULL EVALUATION (Confusion Matrix + Report)
# ============================================================

def full_evaluation(y_true, y_pred, model_name, save_prefix, class_names):
    """Confusion matrix, classification report, per-class accuracy bar chart."""
    print("=" * 70)
    print(f"  EVALUATION: {model_name}")
    print("=" * 70)

    test_acc = accuracy_score(y_true, y_pred)
    print(f"\n  Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)

    # Dual confusion matrix plot
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.8})
    axes[0].set_title(f"{model_name}\nConfusion Matrix (Counts)", fontweight="bold")
    axes[0].set_xlabel("Predicted Label"); axes[0].set_ylabel("True Label")
    axes[0].tick_params(axis="x", rotation=45)

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], linewidths=0.5, linecolor="white",
                vmin=0, vmax=1, cbar_kws={"shrink": 0.8, "label": "Recall"})
    axes[1].set_title(f"{model_name}\nNormalized Confusion Matrix", fontweight="bold")
    axes[1].set_xlabel("Predicted Label"); axes[1].set_ylabel("True Label")
    axes[1].tick_params(axis="x", rotation=45)

    plt.suptitle(f"{model_name} -- Test Set Evaluation",
                 fontweight="bold", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_prefix}_confusion_matrix.png")

    # Per-class accuracy bar chart
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(16, 5))
    colors = sns.color_palette("viridis", len(class_names))
    bars = ax.bar(class_names, per_class_acc * 100, color=colors,
                  edgecolor="white", linewidth=0.8)
    ax.axhline(y=test_acc * 100, color="red", linestyle="--", alpha=0.8,
               label=f"Overall Accuracy: {test_acc*100:.2f}%")
    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_title(f"{model_name} -- Per-Class Accuracy", fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=True, fancybox=True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_per_class_accuracy.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_prefix}_per_class_accuracy.png")

    # Most confused pairs
    print("\n  Most Confused Class Pairs:")
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    confused = [
        (class_names[i], class_names[j], cm_copy[i, j])
        for i in range(len(class_names))
        for j in range(len(class_names))
        if cm_copy[i, j] > 0
    ]
    confused.sort(key=lambda x: x[2], reverse=True)
    for tc, pc, cnt in confused[:5]:
        print(f"    {tc:25s} -> predicted as {pc:25s} ({cnt} times)")

    return test_acc


print("\nEvaluating with TTA predictions...")
tta_final_acc = full_evaluation(
    y_true, tta_preds,
    "EfficientNetV2-S + GeM + TTA (6-view)",
    "efficientnetv2s_tta",
    CLASS_NAMES
)

print("\nEvaluating without TTA (standard)...")
std_final_acc = full_evaluation(
    y_true, std_preds,
    "EfficientNetV2-S + GeM (No TTA)",
    "efficientnetv2s_std",
    CLASS_NAMES
)

# %%
# ============================================================
#  SECTION 15 — COMPARISON vs BASELINE + SAVE FINAL MODEL
# ============================================================

BASELINE_MOBILENETV2_ACC = 0.94   # friend's reported accuracy

print("\n" + "=" * 70)
print("  FINAL COMPARISON -- EfficientNetV2-S vs Baselines")
print("=" * 70)

comparison = {
    "Model": [
        "ResNet-50 (frozen, no TTA)  [original pipeline]",
        "MobileNetV2                 [friend baseline]",
        "EfficientNetV2-S + GeM      [ours, no TTA]",
        "EfficientNetV2-S + GeM + TTA [ours, 6-view]",
    ],
    "Test Accuracy": [
        "~91-93% (reference)",
        f"{BASELINE_MOBILENETV2_ACC*100:.2f}%",
        f"{std_final_acc*100:.2f}%",
        f"{tta_final_acc*100:.2f}%",
    ],
    "vs MobileNetV2": [
        "--",
        "baseline",
        f"{(std_final_acc - BASELINE_MOBILENETV2_ACC)*100:+.2f}%",
        f"{(tta_final_acc - BASELINE_MOBILENETV2_ACC)*100:+.2f}%",
    ]
}

comp_df = pd.DataFrame(comparison)
print("\n" + comp_df.to_string(index=False))

# Bar chart comparison
fig, ax = plt.subplots(figsize=(14, 6))
model_labels = [
    "MobileNetV2\n(friend baseline)",
    "EfficientNetV2-S\n+ GeM (no TTA)",
    "EfficientNetV2-S\n+ GeM + TTA"
]
accs       = [BASELINE_MOBILENETV2_ACC * 100, std_final_acc * 100, tta_final_acc * 100]
colors_bar = ["#e74c3c", "#3498db", "#2ecc71"]
bars = ax.bar(model_labels, accs, color=colors_bar, edgecolor="white", linewidth=1.5, width=0.5)
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{val:.2f}%", ha="center", fontsize=13, fontweight="bold")
ax.set_title("RGB Model Accuracy Comparison", fontweight="bold", fontsize=14)
ax.set_ylabel("Test Accuracy (%)")
ax.set_ylim(85, 100)
ax.spines[["top", "right"]].set_visible(False)
ax.axhline(y=BASELINE_MOBILENETV2_ACC * 100, color="red", linestyle="--",
           alpha=0.5, label=f"MobileNetV2 baseline ({BASELINE_MOBILENETV2_ACC*100:.0f}%)")
ax.legend()
plt.tight_layout()
plt.savefig("model_accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: model_accuracy_comparison.png")

# Save final model
rgb_model.save("rgb_efficientnetv2s_final.keras")
print("\nFinal model saved: rgb_efficientnetv2s_final.keras")

total_time = phase_a_time + phase_b_time
print(f"\nTotal training time: {total_time/60:.1f} minutes")
print(f"  Phase A (warmup)    : {phase_a_time/60:.1f} min")
print(f"  Phase B (fine-tune) : {phase_b_time/60:.1f} min")

print("""
+----------------------------------------------------------------------+
|        HIGH-ACCURACY RGB PIPELINE -- COMPLETE                       |
+----------------------------------------------------------------------+
|                                                                      |
|  Model       : EfficientNetV2-S + GeM Pooling                       |
|  Phase A     : Head warmup (frozen backbone, up to 15 epochs)       |
|  Phase B     : Fine-tuning (top 100 layers, AdamW + cosine LR)     |
|  Augmentation: CutMix + MixUp + RandomFlip/Zoom/Contrast           |
|  TTA         : 6-view averaging for free accuracy boost             |
|  Label smooth: 0.10 (Phase A) -> 0.05 (Phase B)                    |
|                                                                      |
|  Saved Artifacts:                                                    |
|    best_rgb_phase_a.keras              (Phase A best model)         |
|    best_rgb_phase_b.keras              (Phase B best model)         |
|    rgb_efficientnetv2s_final.keras     (final saved model)          |
|    efficientnetv2s_learning_curves.png                               |
|    efficientnetv2s_tta_confusion_matrix.png                          |
|    efficientnetv2s_std_confusion_matrix.png                          |
|    efficientnetv2s_tta_per_class_accuracy.png                        |
|    model_accuracy_comparison.png                                     |
+----------------------------------------------------------------------+
""")
