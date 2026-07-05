# %% [markdown]
# # EuroSAT - HIGH-ACCURACY COMBINED PIPELINE (Local PC Edition)
# ### RGB + Multispectral (13-Band) -- Optimized for 16 GB RAM / 4 GB VRAM
# ---
# Hardware target: Intel i7-9850H | 16 GB RAM | NVIDIA Quadro T1000 (4 GB)
#
# Key optimizations vs. Kaggle version:
#   - Lazy tf.data generators  -> no RAM spike (train never loaded into memory)
#   - Batch size 16            -> fits within 4 GB VRAM comfortably
#   - GPU memory cap 3800 MB   -> leaves headroom for Windows + other processes
#   - Mixed precision float16  -> halves VRAM use; Quadro T1000 fully supports it
#   - Unfreeze only top 60     -> safe for 4 GB during fine-tuning
#   - TTA on CPU batches       -> only test set (small) ever in RAM
#   - gc.collect between phases -> prevents RAM creep over long runs
#
# | Component           | Original                      | This Pipeline                         |
# |---------------------|-------------------------------|---------------------------------------|
# | RGB backbone        | ResNet-50 (frozen only)       | EfficientNetV2-S, two-phase fine-tune |
# | RGB pooling         | GlobalAveragePooling          | Generalized Mean (GeM) Pooling        |
# | RGB augmentation    | Flip+rotate+brightness        | + CutMix + MixUp + Zoom + Contrast    |
# | RGB optimizer       | Adam                          | AdamW + cosine LR annealing           |
# | RGB inference       | Single pass                   | 6-view Test-Time Augmentation (TTA)   |
# | TIF architecture    | 3-stage custom ResNet-50      | 4-stage ResNet + spectral indices     |
# | TIF input channels  | 13                            | 15 (+ NDVI + NDWI)                    |
# | TIF pooling         | GlobalAveragePooling          | GeM Pooling                           |
# | TIF augmentation    | None                          | Spatial flips + band dropout          |
# | TIF optimizer       | Adam                          | AdamW + label smoothing + cosine LR   |
# | TIF inference       | Single pass                   | 5-view Test-Time Augmentation (TTA)   |

# %%
# ============================================================
#  CHUNK 1 -- IMPORTS & GPU SETUP
# ============================================================
import os, sys, json, warnings, gc, time
from pathlib import Path

# ── VS Code: to remove the red squiggles under imports ────────────────────
# Press Ctrl+Shift+P -> "Python: Select Interpreter"
# Choose:  .\eurosat_env\Scripts\python.exe
# (the virtual environment we created in the LTC folder)
# ──────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import rasterio
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

sns.set_theme(style="darkgrid", palette="deep", font_scale=1.1)
plt.rcParams.update({
    "figure.figsize": (14, 6), "figure.dpi": 100,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "font.family": "sans-serif",
})

# ── GPU: cap memory so Windows does not get starved ──
# ── GPU / CPU SETUP ──
# TensorFlow >= 2.11 dropped native Windows GPU support.
# On this machine (Windows + Python 3.13) training runs on CPU using
# oneDNN + AVX2 instructions (the i7-9850H has both -- still fast).
#
# To use the Quadro T1000 GPU you have two options (outside this script):
#   Option A: Install Python 3.10 + TF 2.10 (last version with Windows GPU)
#             pip install tensorflow==2.10.0
#   Option B: Use WSL2 (Windows Subsystem for Linux) with TF + CUDA
#
# For now: CPU mode with oneDNN acceleration.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"           # suppress info logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"           # keep oneDNN (AVX2 speedup)
os.environ["OMP_NUM_THREADS"] = "12"               # i7-9850H has 12 logical cores
os.environ["TF_NUM_INTEROP_THREADS"] = "6"         # 6 physical cores
os.environ["TF_NUM_INTRAOP_THREADS"] = "12"        # 12 threads for matmuls

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEMORY_LIMIT_MB)]
        )
        print(f"GPU: {gpus[0].name}  (memory capped at {GPU_MEMORY_LIMIT_MB} MB)")
    except RuntimeError as e:
        print(f"GPU memory cap skipped: {e}")
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    MIXED_PRECISION = True
    HAS_GPU = True
    print("Mixed precision (float16) ENABLED")
else:
    print("No GPU detected -- running on CPU (i7-9850H + oneDNN + AVX2)")
    print("  Using 12 threads | oneDNN acceleration enabled")
    print("  Tip: For GPU support, install Python 3.10 + TF 2.10")
    MIXED_PRECISION = False
    HAS_GPU = False
    # Set CPU parallelism explicitly
    tf.config.threading.set_inter_op_parallelism_threads(6)
    tf.config.threading.set_intra_op_parallelism_threads(12)

print(f"TensorFlow: {tf.__version__}")
print("All imports OK.")


# %%
# ============================================================
#  CHUNK 2 -- PATHS & HYPERPARAMETERS
# ============================================================
# Folder structure on this machine:
#   c:\Users\pc\OneDrive\Desktop\DEPI\
#     LTC\                        <- this script lives here
#     archive\
#       EuroSAT\                  <- RGB JPGs + train/validation/test.csv
#       EuroSATallBands\          <- TIF files + train/validation/test.csv

# Absolute paths -- no need to change
BASE_DIR   = Path(r"c:\Users\pc\OneDrive\Desktop\DEPI\archive")
OUTPUT_DIR = Path(r"c:\Users\pc\OneDrive\Desktop\DEPI\LTC\outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RGB_DIR = BASE_DIR / "EuroSAT"
MS_DIR  = BASE_DIR / "EuroSATallBands"

# Sanity check -- will error early with a clear message if paths are wrong
if not RGB_DIR.exists():
    raise FileNotFoundError(f"RGB folder not found: {RGB_DIR}")
if not MS_DIR.exists():
    raise FileNotFoundError(f"TIF folder not found: {MS_DIR}")

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]
NUM_CLASSES  = len(CLASS_NAMES)
label_to_int = {name: i for i, name in enumerate(CLASS_NAMES)}

# Sentinel-2 band indices (0-indexed in 13-band TIF array)
B04_RED   = 3    # Red
B08_NIR   = 7    # NIR broad
B03_GREEN = 2    # Green
B11_SWIR1 = 11   # SWIR 1

# ── Image sizes ──
RGB_IMG_SIZE   = (224, 224)   # EfficientNetV2-S native input
TIF_IMG_SIZE   = (64, 64)     # Native Sentinel-2 patch size
TIF_N_CHANNELS = 15           # 13 original bands + NDVI + NDWI

# ── Batch size: 16 works on 4 GB VRAM (or CPU) ──
BATCH_SIZE = 16

# ── Training epochs (CPU-tuned; EarlyStopping cuts short if converged) ──
# i7-9850H timing: ~5-10 min/epoch for RGB, ~2-4 min/epoch for TIF
RGB_PHASE_A_EPOCHS  = 10       # head warmup   (~50-100 min)
RGB_PHASE_B_EPOCHS  = 30       # fine-tuning   (~150-300 min)
RGB_PHASE_A_LR      = 1e-3
RGB_PHASE_B_HEAD_LR = 1e-4
RGB_PHASE_B_BASE_LR = 1e-5
RGB_UNFREEZE_LAYERS = 60       # top 60 layers -- safe for 4 GB

TIF_EPOCHS  = 40               # (~80-160 min)
TIF_LR      = 1e-3

# ── Shared training settings ──
WEIGHT_DECAY      = 1e-4
LABEL_SMOOTH_A    = 0.10
LABEL_SMOOTH_B    = 0.05
LABEL_SMOOTH_TIF  = 0.08
CUTMIX_PROB       = 0.50
MIXUP_PROB        = 0.30

def save_fig(filename):
    plt.savefig(OUTPUT_DIR / filename, dpi=120, bbox_inches="tight")

def load_csv(base_dir, split_name):
    """Load a CSV split, drop any index columns, strip whitespace from headers."""
    df = pd.read_csv(base_dir / f"{split_name}.csv")
    df.columns = [c.strip() for c in df.columns]
    # RGB CSVs have an extra 'Unnamed: 0' index column -- drop it
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df

print(f"RGB dir  : {RGB_DIR}")
print(f"TIF dir  : {MS_DIR}")
print(f"Output   : {OUTPUT_DIR}")
print(f"Batch    : {BATCH_SIZE} | Classes: {NUM_CLASSES}")
print(f"RGB Phase-A epochs: {RGB_PHASE_A_EPOCHS} | Phase-B epochs: {RGB_PHASE_B_EPOCHS}")
print(f"TIF epochs        : {TIF_EPOCHS}")


# %%
# ============================================================
#  CHUNK 3 -- LOAD CSV SPLITS
# ============================================================
train_rgb_df = load_csv(RGB_DIR, "train")
val_rgb_df   = load_csv(RGB_DIR, "validation")
test_rgb_df  = load_csv(RGB_DIR, "test")

train_ms_df  = load_csv(MS_DIR, "train")
val_ms_df    = load_csv(MS_DIR, "validation")
test_ms_df   = load_csv(MS_DIR, "test")

print(f"RGB -- Train: {len(train_rgb_df)} | Val: {len(val_rgb_df)} | Test: {len(test_rgb_df)}")
print(f"TIF -- Train: {len(train_ms_df)} | Val: {len(val_ms_df)} | Test: {len(test_ms_df)}")

# %%
# ============================================================
#  CHUNK 4 -- AUGMENTATION FUNCTIONS
# ============================================================
# CutMix, MixUp operate on batches.  TIF spatial aug + band dropout
# are applied per-image inside the generator (before batching).
# All numpy operations; works in TF2 eager mode when you call .numpy()
# on tensor batches inside the training loop.

def mixup(images, labels, alpha=0.4):
    """Linearly blend two images and their labels (MixUp)."""
    if hasattr(images, "numpy"): images = images.numpy()
    if hasattr(labels, "numpy"): labels = labels.numpy()
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(images.shape[0])
    return (lam * images + (1.0 - lam) * images[idx]).astype(np.float32),            (lam * labels + (1.0 - lam) * labels[idx]).astype(np.float32)

def rand_bbox(h, w, lam):
    cw = int(w * np.sqrt(1.0 - lam))
    ch = int(h * np.sqrt(1.0 - lam))
    cx, cy = np.random.randint(w), np.random.randint(h)
    return (np.clip(cx - cw//2, 0, w), np.clip(cy - ch//2, 0, h),
            np.clip(cx + cw//2, 0, w), np.clip(cy + ch//2, 0, h))

def cutmix(images, labels, alpha=1.0):
    """Cut-and-paste patch from one image into another (CutMix)."""
    if hasattr(images, "numpy"): images = images.numpy()
    if hasattr(labels, "numpy"): labels = labels.numpy()
    bs, h, w = images.shape[0], images.shape[1], images.shape[2]
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(bs)
    x1, y1, x2, y2 = rand_bbox(h, w, lam)
    out = images.copy()
    out[:, y1:y2, x1:x2, :] = images[idx, y1:y2, x1:x2, :]
    actual_lam = 1.0 - ((x2-x1)*(y2-y1)/(w*h))
    return out.astype(np.float32),            (actual_lam * labels + (1.0-actual_lam)*labels[idx]).astype(np.float32)

def apply_rgb_batch_aug(images, labels):
    """Stochastically apply CutMix or MixUp to an RGB training batch."""
    if hasattr(images, "numpy"): images = images.numpy()
    if hasattr(labels, "numpy"): labels = labels.numpy()
    r = np.random.rand()
    if r < CUTMIX_PROB:
        return cutmix(images, labels, alpha=1.0)
    elif r < CUTMIX_PROB + MIXUP_PROB:
        return mixup(images, labels, alpha=0.4)
    return images, labels

def apply_tif_spatial_aug(img):
    """Random H/V flip + 90-degree rotation for a (H,W,C) TIF image."""
    if np.random.rand() > 0.5: img = img[:, ::-1, :]   # H flip
    if np.random.rand() > 0.5: img = img[::-1, :, :]   # V flip
    k = np.random.choice([0, 1, 2, 3])
    img = np.rot90(img, k=k, axes=(0, 1))
    return np.ascontiguousarray(img)

def apply_tif_band_dropout(img, drop_prob=0.10):
    """Zero one random band with drop_prob -- prevents over-reliance on any band."""
    if np.random.rand() < drop_prob:
        b = np.random.randint(0, 13)   # only original 13 bands, not derived
        img = img.copy()
        img[:, :, b] = 0.0
    return img

print("Augmentation functions defined.")

# %%
# ============================================================
#  CHUNK 5 -- GENERALIZED MEAN (GeM) POOLING LAYER
# ============================================================
# GeM(x) = (mean(x^p))^(1/p), with learnable p.
# p=1 -> GAP.  p->inf -> GMP.  Remote sensing tasks typically
# converge to p~3-6, giving more weight to dominant features.

class GeMPooling(layers.Layer):
    def __init__(self, p_init=3.0, p_trainable=True, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.p_init = p_init
        self.p_trainable = p_trainable
        self.eps = eps

    def build(self, input_shape):
        self.p = self.add_weight(
            name="gem_p", shape=(1,),
            initializer=tf.constant_initializer(self.p_init),
            trainable=self.p_trainable, dtype=tf.float32,
            constraint=tf.keras.constraints.NonNeg()
        )
        super().build(input_shape)

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        x = tf.clip_by_value(x, self.eps, tf.reduce_max(x))
        x = tf.pow(x, self.p)
        x = tf.reduce_mean(x, axis=[1, 2])
        x = tf.pow(x, 1.0 / self.p)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"p_init": self.p_init, "p_trainable": self.p_trainable, "eps": self.eps})
        return cfg

print("GeMPooling layer defined.")

# %%
# ============================================================
#  CHUNK 6 -- RGB DATA PIPELINE  (Memory-efficient lazy loading)
# ============================================================
# WHY LAZY?
#   19K RGB images at 224x224 = ~11 GB in RAM.
#   With 16 GB total RAM you'd be left with only 5 GB for OS + TF.
#   Solution: tf.data.from_generator() reads images one batch at a time
#   from disk, keeping RAM usage well under 2 GB at all times.
#
# ONLY the test set (~2700 images = ~1.6 GB) is loaded into numpy
# arrays -- needed for TTA (Test-Time Augmentation) at inference.
#
# NOTE: EfficientNetV2S uses include_preprocessing=True, so images
#       must be in [0, 255] float32 -- do NOT apply ImageNet mean/std.

def load_rgb_image(filepath):
    """Load one RGB image, resize to 224x224. Returns float32 in [0, 255]."""
    img = Image.open(filepath).convert("RGB")
    img = img.resize(RGB_IMG_SIZE, Image.BILINEAR)
    return np.array(img, dtype=np.float32)

# Single-image augmentation applied inside the generator (before batching)
def aug_single_rgb(img):
    """Light per-image augmentation applied inside the generator."""
    if np.random.rand() > 0.5: img = img[:, ::-1, :]   # H flip
    if np.random.rand() > 0.5: img = img[::-1, :, :]   # V flip
    k = np.random.choice([0, 1, 2, 3])
    img = np.rot90(img, k=k, axes=(0, 1))               # 90-degree rotation
    # Brightness jitter: random scale in [0.85, 1.15]
    img = np.clip(img * np.random.uniform(0.85, 1.15), 0, 255)
    return np.ascontiguousarray(img, dtype=np.float32)

# Keras in-graph augmentation (GPU-accelerated, applied after batching)
rgb_aug_pipeline = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.08),
    layers.RandomContrast(0.12),
], name="rgb_keras_aug")

def make_rgb_generator(df, base_dir, augment=False, shuffle=True):
    """
    Returns a Python generator that yields (image, one_hot_label) pairs.
    Images are loaded from disk on demand -- no full array in RAM.
    """
    filepaths  = [str(base_dir / row["Filename"]) for _, row in df.iterrows()]
    int_labels = [label_to_int[c] for c in df["ClassName"]]
    n = len(filepaths)

    def gen():
        indices = list(range(n))
        if shuffle:
            np.random.shuffle(indices)
        for idx in indices:
            try:
                img = load_rgb_image(filepaths[idx])
                if augment:
                    img = aug_single_rgb(img)
                lbl = to_categorical(int_labels[idx], NUM_CLASSES).astype(np.float32)
                yield img, lbl
            except Exception:
                continue
    return gen, n

def make_rgb_lazy_ds(df, base_dir, augment=False, shuffle=True):
    """Build a lazy tf.data.Dataset for RGB images."""
    gen_fn, n = make_rgb_generator(df, base_dir, augment=augment, shuffle=shuffle)
    ds = tf.data.Dataset.from_generator(
        gen_fn,
        output_signature=(
            tf.TensorSpec(shape=(*RGB_IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(NUM_CLASSES,), dtype=tf.float32),
        )
    )
    ds = ds.batch(BATCH_SIZE)
    if augment:
        def keras_aug(x, y):
            return rgb_aug_pipeline(x, training=True), y
        ds = ds.map(keras_aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE), n

def load_rgb_test_numpy(df, base_dir):
    """
    Load only the TEST set into numpy arrays for TTA evaluation.
    (~2700 images x 224x224x3 x 4 bytes = ~1.6 GB -- fits in 16 GB RAM)
    """
    filepaths  = [str(base_dir / row["Filename"]) for _, row in df.iterrows()]
    int_labels = [label_to_int[c] for c in df["ClassName"]]
    images, labels = [], []
    for i, (fp, lbl_int) in enumerate(zip(filepaths, int_labels)):
        if i % 500 == 0:
            print(f"  Loading test RGB: {i}/{len(filepaths)}", end="\r")
        try:
            images.append(load_rgb_image(fp))
            labels.append(lbl_int)
        except Exception:
            continue
    X = np.array(images, dtype=np.float32)
    Y = to_categorical(labels, NUM_CLASSES).astype(np.float32)
    print(f"  Test RGB loaded: {X.shape}  ({X.nbytes/1e9:.2f} GB)")
    return X, Y

print("Building RGB lazy datasets (no RAM spike)...")
train_rgb_ds, n_train_rgb = make_rgb_lazy_ds(train_rgb_df, RGB_DIR, augment=True,  shuffle=True)
val_rgb_ds,   n_val_rgb   = make_rgb_lazy_ds(val_rgb_df,   RGB_DIR, augment=False, shuffle=False)
print(f"  Train: {n_train_rgb} samples (lazy) | Val: {n_val_rgb} samples (lazy)")

print("Loading test RGB into memory for TTA...")
X_test_rgb, Y_test_rgb = load_rgb_test_numpy(test_rgb_df, RGB_DIR)
gc.collect()
print("RGB pipelines ready.")

# %%
# ============================================================
#  CHUNK 7 -- TIF DATA PIPELINE  (15 channels + lazy loading)
# ============================================================
# New features vs. original:
#   - NDVI = (NIR - Red) / (NIR + Red + eps)   [vegetation index]
#   - NDWI = (Green - NIR) / (Green + NIR + eps) [water/moisture index]
#   Both scaled to [0, 1] and appended as channels 14 and 15.
#   - Training images get spatial augmentation + band dropout per-image
#   - Lazy generator: only test set loaded into numpy for TTA

def compute_spectral_indices(img):
    """Add NDVI and NDWI to a per-band-normalized (H,W,13) image. Returns (H,W,15)."""
    nir   = img[:, :, B08_NIR].astype(np.float64)
    red   = img[:, :, B04_RED].astype(np.float64)
    green = img[:, :, B03_GREEN].astype(np.float64)
    ndvi  = (nir - red)   / (nir + red   + 1e-8)
    ndwi  = (green - nir) / (green + nir + 1e-8)
    # Scale from [-1,1] to [0,1]
    ndvi_s = ((ndvi + 1.0) / 2.0)[:, :, np.newaxis]
    ndwi_s = ((ndwi + 1.0) / 2.0)[:, :, np.newaxis]
    return np.concatenate([img, ndvi_s, ndwi_s], axis=-1).astype(np.float32)

def load_tif_image(filepath, augment=False):
    """
    Load one .tif file.
    1. Read 13-band array (H,W,13)
    2. Per-band min-max normalization to [0,1]
    3. Optional: band dropout + spatial augmentation
    4. Resize to TIF_IMG_SIZE (64x64) if needed
    5. Append NDVI + NDWI -> (64,64,15)
    Returns float32 array of shape (64,64,15).
    """
    with rasterio.open(filepath) as src:
        data = src.read()                               # (13, H, W)
    img = np.transpose(data, (1, 2, 0)).astype(np.float32)  # (H, W, 13)

    # Per-band min-max normalization
    for b in range(img.shape[2]):
        bmin, bmax = img[:, :, b].min(), img[:, :, b].max()
        img[:, :, b] = (img[:, :, b] - bmin) / (bmax - bmin + 1e-8)

    # Augmentation (train only)
    if augment:
        img = apply_tif_band_dropout(img, drop_prob=0.10)
        img = apply_tif_spatial_aug(img)

    # Resize if spatial size differs from target
    if img.shape[:2] != TIF_IMG_SIZE:
        bands = [cv2.resize(img[:, :, b], TIF_IMG_SIZE, interpolation=cv2.INTER_LINEAR)
                 for b in range(img.shape[2])]
        img = np.stack(bands, axis=-1)

    return compute_spectral_indices(img)   # (64, 64, 15)

def make_tif_generator(df, base_dir, augment=False, shuffle=True):
    """Python generator yielding (tif_image, one_hot_label) pairs."""
    filepaths  = [str(base_dir / row["Filename"]) for _, row in df.iterrows()]
    int_labels = [label_to_int[c] for c in df["ClassName"]]
    n = len(filepaths)

    def gen():
        indices = list(range(n))
        if shuffle:
            np.random.shuffle(indices)
        for idx in indices:
            try:
                img = load_tif_image(filepaths[idx], augment=augment)
                lbl = to_categorical(int_labels[idx], NUM_CLASSES).astype(np.float32)
                yield img, lbl
            except Exception:
                continue
    return gen, n

def make_tif_lazy_ds(df, base_dir, augment=False, shuffle=True):
    """Build a lazy tf.data.Dataset for TIF images."""
    gen_fn, n = make_tif_generator(df, base_dir, augment=augment, shuffle=shuffle)
    ds = tf.data.Dataset.from_generator(
        gen_fn,
        output_signature=(
            tf.TensorSpec(shape=(*TIF_IMG_SIZE, TIF_N_CHANNELS), dtype=tf.float32),
            tf.TensorSpec(shape=(NUM_CLASSES,), dtype=tf.float32),
        )
    )
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE), n

def load_tif_test_numpy(df, base_dir):
    """Load only test TIF images into numpy for TTA (~660 MB -- fine for 16 GB RAM)."""
    filepaths  = [str(base_dir / row["Filename"]) for _, row in df.iterrows()]
    int_labels = [label_to_int[c] for c in df["ClassName"]]
    images, labels = [], []
    for i, (fp, lbl_int) in enumerate(zip(filepaths, int_labels)):
        if i % 500 == 0:
            print(f"  Loading test TIF: {i}/{len(filepaths)}", end="\r")
        try:
            images.append(load_tif_image(fp, augment=False))
            labels.append(lbl_int)
        except Exception:
            continue
    X = np.array(images, dtype=np.float32)
    Y = to_categorical(labels, NUM_CLASSES).astype(np.float32)
    print(f"  Test TIF loaded: {X.shape}  ({X.nbytes/1e6:.0f} MB)")
    return X, Y

print("Building TIF lazy datasets (no RAM spike)...")
train_tif_ds, n_train_tif = make_tif_lazy_ds(train_ms_df, MS_DIR, augment=True,  shuffle=True)
val_tif_ds,   n_val_tif   = make_tif_lazy_ds(val_ms_df,   MS_DIR, augment=False, shuffle=False)
print(f"  Train: {n_train_tif} samples (lazy) | Val: {n_val_tif} samples (lazy)")

print("Loading test TIF into memory for TTA...")
X_test_tif, Y_test_tif = load_tif_test_numpy(test_ms_df, MS_DIR)
gc.collect()
print("TIF pipelines ready.")

# %% [markdown]
# ---
# ## CHUNK 8 -- RGB MODEL: EfficientNetV2-S + GeM + Phase A Head Warmup
# ---

# %%
# ============================================================
#  CHUNK 8 -- RGB MODEL ARCHITECTURE & PHASE A WARMUP
# ============================================================
# Architecture:
#   Input (224x224x3, float32 [0,255])
#     -> EfficientNetV2-S (include_preprocessing=True, initially FROZEN)
#     -> GeM Pooling (learnable p, init=3.0)
#     -> Dense(512, swish) -> BatchNorm -> Dropout(0.4)
#     -> Dense(256, swish) -> BatchNorm -> Dropout(0.3)
#     -> Dense(10, softmax, dtype=float32)   <- float32 output even in fp16 mode

def build_rgb_model(num_classes=10, input_shape=(224, 224, 3)):
    """Build EfficientNetV2-S + GeM head. Backbone frozen for Phase A."""
    base = EfficientNetV2S(
        include_top=False, weights="imagenet",
        input_shape=input_shape, include_preprocessing=True,
    )
    base.trainable = False

    inp = layers.Input(shape=input_shape, name="rgb_input")
    x   = base(inp, training=False)
    x   = GeMPooling(p_init=3.0, p_trainable=True, name="gem_pool")(x)
    x   = layers.Dense(512, name="fc_512")(x)
    x   = layers.BatchNormalization(name="bn_512")(x)
    x   = layers.Activation("swish", name="swish_512")(x)
    x   = layers.Dropout(0.4, name="drop_512")(x)
    x   = layers.Dense(256, name="fc_256")(x)
    x   = layers.BatchNormalization(name="bn_256")(x)
    x   = layers.Activation("swish", name="swish_256")(x)
    x   = layers.Dropout(0.3, name="drop_256")(x)
    out = layers.Dense(num_classes, activation="softmax",
                       dtype="float32", name="output")(x)
    return models.Model(inp, out, name="RGB_EfficientNetV2S_GeM"), base

def unfreeze_top(model, base, n=RGB_UNFREEZE_LAYERS):
    """Unfreeze top n layers of the backbone for Phase B."""
    base.trainable = True
    freeze_until = len(base.layers) - n
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= freeze_until)
    frozen   = sum(1 for l in base.layers if not l.trainable)
    unfrozen = sum(1 for l in base.layers if l.trainable)
    print(f"  Backbone -- Frozen: {frozen} | Unfrozen: {unfrozen}")
    return model

def cosine_lr(epoch, max_lr, warmup=2, total=40, min_lr=1e-8):
    """Cosine annealing with linear warmup."""
    if epoch < warmup:
        return max_lr * (epoch + 1) / warmup
    t = (epoch - warmup) / max(total - warmup, 1)
    return float(min_lr + 0.5 * (max_lr - min_lr) * (1.0 + np.cos(np.pi * t)))

rgb_model, rgb_backbone = build_rgb_model(NUM_CLASSES, (*RGB_IMG_SIZE, 3))
total_rgb  = rgb_model.count_params()
train_rgb_p = sum(tf.size(w).numpy() for w in rgb_model.trainable_weights)
print(f"RGB Model -- Total params: {total_rgb:,} | Trainable (Phase A): {train_rgb_p:,}")

# ── PHASE A: HEAD WARMUP ──
print("\n" + "=" * 65)
print("  PHASE A -- RGB HEAD WARMUP (backbone frozen)")
print(f"  Optimizer: Adam(lr={RGB_PHASE_A_LR}) | Label smooth: {LABEL_SMOOTH_A}")
print(f"  Max epochs: {RGB_PHASE_A_EPOCHS} (EarlyStopping patience=5)")
print("=" * 65)

optimizer_a = optimizers.Adam(learning_rate=RGB_PHASE_A_LR)
loss_fn_a   = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH_A)

rgb_a_train_acc, rgb_a_val_acc   = [], []
rgb_a_train_loss, rgb_a_val_loss = [], []
best_val_acc_a  = 0.0
best_weights_a  = None
patience_a      = 0
PATIENCE_A      = 5

rgb_a_start = time.time()
for epoch in range(RGB_PHASE_A_EPOCHS):
    ep_losses, ep_accs = [], []

    for step, (xb, yb) in enumerate(train_rgb_ds):
        # CutMix / MixUp on current batch
        xn, yn = apply_rgb_batch_aug(xb, yb)
        xb_t   = tf.constant(xn, dtype=tf.float32)
        yb_t   = tf.constant(yn, dtype=tf.float32)
        with tf.GradientTape() as tape:
            pred = rgb_model(xb_t, training=True)
            loss = loss_fn_a(yb_t, pred)
        grads = tape.gradient(loss, rgb_model.trainable_weights)
        optimizer_a.apply_gradients(zip(grads, rgb_model.trainable_weights))
        acc = float(tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(pred,1), tf.argmax(tf.cast(yb_t,tf.float32),1)),
            tf.float32)))
        ep_losses.append(float(loss)); ep_accs.append(acc)
        if (step+1) % 50 == 0:
            print(f"  ep{epoch+1} step{step+1} loss={np.mean(ep_losses):.4f} acc={np.mean(ep_accs):.4f}", end="\r")

    vl_list, va_list = [], []
    for xv, yv in val_rgb_ds:
        vp = rgb_model(xv, training=False)
        vl_list.append(float(loss_fn_a(yv, vp)))
        va_list.append(float(tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(vp,1), tf.argmax(yv,1)), tf.float32))))

    tl, ta = np.mean(ep_losses), np.mean(ep_accs)
    vl, va = np.mean(vl_list), np.mean(va_list)
    rgb_a_train_acc.append(ta); rgb_a_val_acc.append(va)
    rgb_a_train_loss.append(tl); rgb_a_val_loss.append(vl)
    elapsed = (time.time() - rgb_a_start) / 60
    print(f"  Epoch {epoch+1:2d}/{RGB_PHASE_A_EPOCHS} | loss:{tl:.4f} acc:{ta:.4f} | "
          f"val_loss:{vl:.4f} val_acc:{va:.4f} | {elapsed:.1f} min elapsed")

    if va > best_val_acc_a:
        best_val_acc_a = va; best_weights_a = rgb_model.get_weights(); patience_a = 0
        rgb_model.save(str(OUTPUT_DIR / "best_rgb_phase_a.keras"))
        print(f"  --> Best val_acc: {best_val_acc_a:.4f}  [saved]")
    else:
        patience_a += 1
        if patience_a >= PATIENCE_A:
            print(f"  EarlyStopping at epoch {epoch+1} (patience={PATIENCE_A})")
            break

rgb_model.set_weights(best_weights_a)
rgb_a_time = time.time() - rgb_a_start
print(f"Phase A done in {rgb_a_time/60:.1f} min | Best val_acc: {best_val_acc_a:.4f}")
gc.collect()

# %%
# ============================================================
#  CHUNK 9 -- RGB PHASE B: PROGRESSIVE FINE-TUNING
# ============================================================
# Unfreeze top RGB_UNFREEZE_LAYERS (default 60) of EfficientNetV2-S.
# Use TWO separate AdamW optimizers:
#   - backbone: 10x smaller LR  -> avoids catastrophic forgetting
#   - head    : standard fine-tuning LR
# Head LR follows cosine annealing schedule.

print("\n" + "=" * 65)
print(f"  PHASE B -- RGB FINE-TUNING (top {RGB_UNFREEZE_LAYERS} backbone layers)")
print(f"  Head LR: {RGB_PHASE_B_HEAD_LR} | Backbone LR: {RGB_PHASE_B_BASE_LR}")
print(f"  Optimizer: AdamW(weight_decay={WEIGHT_DECAY}) | Label smooth: {LABEL_SMOOTH_B}")
print(f"  Max epochs: {RGB_PHASE_B_EPOCHS} (EarlyStopping patience=10)")
print("=" * 65)

rgb_model = unfreeze_top(rgb_model, rgb_backbone, n=RGB_UNFREEZE_LAYERS)

bb_names  = {w.name for w in rgb_backbone.trainable_weights}
bb_w      = [w for w in rgb_model.trainable_weights if w.name in bb_names]
hd_w      = [w for w in rgb_model.trainable_weights if w.name not in bb_names]

opt_bb = optimizers.AdamW(learning_rate=RGB_PHASE_B_BASE_LR, weight_decay=WEIGHT_DECAY)
opt_hd = optimizers.AdamW(learning_rate=RGB_PHASE_B_HEAD_LR, weight_decay=WEIGHT_DECAY)
loss_fn_b = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH_B)

print(f"  Backbone trainable params : {sum(tf.size(w).numpy() for w in bb_w):,}")
print(f"  Head trainable params     : {sum(tf.size(w).numpy() for w in hd_w):,}")

rgb_b_train_acc, rgb_b_val_acc   = [], []
rgb_b_train_loss, rgb_b_val_loss = [], []
best_val_acc_b  = best_val_acc_a
best_weights_b  = rgb_model.get_weights()
patience_b      = 0
PATIENCE_B      = 10

rgb_b_start = time.time()
for epoch in range(RGB_PHASE_B_EPOCHS):
    new_hd_lr = cosine_lr(epoch, RGB_PHASE_B_HEAD_LR,
                          warmup=2, total=RGB_PHASE_B_EPOCHS, min_lr=1e-8)
    opt_hd.learning_rate.assign(new_hd_lr)
    opt_bb.learning_rate.assign(new_hd_lr / 10.0)

    ep_losses, ep_accs = [], []
    for step, (xb, yb) in enumerate(train_rgb_ds):
        xn, yn = apply_rgb_batch_aug(xb, yb)
        xb_t   = tf.constant(xn, dtype=tf.float32)
        yb_t   = tf.constant(yn, dtype=tf.float32)
        with tf.GradientTape() as tape:
            pred = rgb_model(xb_t, training=True)
            loss = loss_fn_b(yb_t, pred)
        grads   = tape.gradient(loss, rgb_model.trainable_weights)
        bb_grads = [g for g,w in zip(grads, rgb_model.trainable_weights) if w.name in bb_names]
        hd_grads = [g for g,w in zip(grads, rgb_model.trainable_weights) if w.name not in bb_names]
        if bb_w and bb_grads: opt_bb.apply_gradients(zip(bb_grads, bb_w))
        if hd_w and hd_grads: opt_hd.apply_gradients(zip(hd_grads, hd_w))
        acc = float(tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(pred,1), tf.argmax(tf.cast(yb_t,tf.float32),1)),
            tf.float32)))
        ep_losses.append(float(loss)); ep_accs.append(acc)
        if (step+1) % 50 == 0:
            print(f"  ep{epoch+1} step{step+1} loss={np.mean(ep_losses):.4f} acc={np.mean(ep_accs):.4f}", end="\r")

    vl_list, va_list = [], []
    for xv, yv in val_rgb_ds:
        vp = rgb_model(xv, training=False)
        vl_list.append(float(loss_fn_b(yv, vp)))
        va_list.append(float(tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(vp,1), tf.argmax(yv,1)), tf.float32))))
    tl, ta = np.mean(ep_losses), np.mean(ep_accs)
    vl, va = np.mean(vl_list), np.mean(va_list)
    rgb_b_train_acc.append(ta); rgb_b_val_acc.append(va)
    rgb_b_train_loss.append(tl); rgb_b_val_loss.append(vl)
    elapsed = (time.time() - rgb_b_start) / 60
    print(f"  Epoch {epoch+1:2d}/{RGB_PHASE_B_EPOCHS} | loss:{tl:.4f} acc:{ta:.4f} | "
          f"val_loss:{vl:.4f} val_acc:{va:.4f} | lr:{new_hd_lr:.1e} | {elapsed:.1f} min")

    if va > best_val_acc_b:
        best_val_acc_b = va; best_weights_b = rgb_model.get_weights(); patience_b = 0
        rgb_model.save(str(OUTPUT_DIR / "best_rgb_phase_b.keras"))
        print(f"  --> Best val_acc: {best_val_acc_b:.4f}  [saved]")
    else:
        patience_b += 1
        if patience_b >= PATIENCE_B:
            print(f"  EarlyStopping at epoch {epoch+1}")
            break

rgb_model.set_weights(best_weights_b)
rgb_b_time = time.time() - rgb_b_start
rgb_total_time = rgb_a_time + rgb_b_time
print(f"Phase B done in {rgb_b_time/60:.1f} min | Best val_acc: {best_val_acc_b:.4f}")
print(f"Total RGB training: {rgb_total_time/60:.1f} min")
gc.collect()

# %%
# ============================================================
#  CHUNK 10 -- RGB TEST-TIME AUGMENTATION (6 views) + EVALUATION
# ============================================================

def tta_predict(model, X, n_views=6, bs=BATCH_SIZE):
    """6-view TTA: original, H-flip, V-flip, HV-flip, 90-rot, 90-rot+H-flip."""
    probs = np.zeros((len(X), NUM_CLASSES), dtype=np.float32)
    views = [
        X,
        X[:, :, ::-1, :],
        X[:, ::-1, :, :],
        X[:, ::-1, ::-1, :],
        np.rot90(X, k=1, axes=(1, 2)),
        np.rot90(X, k=1, axes=(1, 2))[:, :, ::-1, :],
    ]
    for vi, Xv in enumerate(views):
        batch_probs = []
        for i in range(0, len(Xv), bs):
            p = model(tf.constant(Xv[i:i+bs], dtype=tf.float32), training=False).numpy()
            batch_probs.append(p)
        probs += np.concatenate(batch_probs, axis=0)
        print(f"  TTA view {vi+1}/{len(views)}", end="\r")
    probs /= len(views)
    print(f"  TTA complete ({len(views)} views).")
    return probs

def evaluate_model(y_true, y_pred, test_acc, model_name, save_prefix):
    """Confusion matrix, classification report, per-class accuracy bar chart."""
    print("=" * 65)
    print(f"  {model_name}")
    print(f"  Test Accuracy: {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print("=" * 65)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[0], linewidths=0.5, cbar_kws={"shrink": 0.8})
    axes[0].set_title(f"{model_name}\nConfusion Matrix", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], vmin=0, vmax=1, cbar_kws={"shrink": 0.8, "label": "Recall"})
    axes[1].set_title(f"{model_name}\nNormalized Confusion Matrix", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    save_fig(f"{save_prefix}_confusion_matrix.png")
    plt.show()

    pca = cm.diagonal() / cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(CLASS_NAMES, pca * 100,
                  color=sns.color_palette("viridis", NUM_CLASSES),
                  edgecolor="white", linewidth=0.8)
    ax.axhline(y=test_acc*100, color="red", linestyle="--",
               label=f"Overall: {test_acc*100:.2f}%")
    for bar, val in zip(bars, pca):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{val*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_title(f"{model_name} -- Per-Class Accuracy", fontweight="bold")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 110)
    ax.tick_params(axis="x", rotation=45); ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    save_fig(f"{save_prefix}_per_class_accuracy.png")
    plt.show()

    print("  Most confused pairs:")
    cm_c = cm.copy(); np.fill_diagonal(cm_c, 0)
    pairs = [(CLASS_NAMES[i], CLASS_NAMES[j], cm_c[i,j])
             for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if cm_c[i,j]>0]
    for tc, pc, cnt in sorted(pairs, key=lambda x:-x[2])[:5]:
        print(f"    {tc:25s} -> {pc:25s} ({int(cnt)})")

def plot_two_phase_curves(a_acc, a_loss, a_va, a_vl, b_acc, b_loss, b_va, b_vl, title, prefix):
    na, nb = len(a_acc), len(b_acc)
    ea, eb = list(range(1, na+1)), list(range(na+1, na+nb+1))
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    for ax, train_a, val_a, train_b, val_b, ylabel in [
        (axes[0], a_acc,  a_va,  b_acc,  b_va,  "Accuracy"),
        (axes[1], a_loss, a_vl,  b_loss, b_vl,  "Loss"),
    ]:
        ax.plot(ea, train_a, "b-o", label="Train A", lw=2, ms=3)
        ax.plot(ea, val_a,   "b--s", label="Val A",  lw=2, ms=3)
        ax.plot(eb, train_b, "g-o", label="Train B", lw=2, ms=3)
        ax.plot(eb, val_b,   "g--s", label="Val B",  lw=2, ms=3)
        ax.axvline(x=na, color="orange", ls=":", lw=2, label=f"A->B (ep{na})")
        ax.set_title(f"{title} -- {ylabel}", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    save_fig(f"{prefix}_learning_curves.png")
    plt.show()

print("Running RGB TTA (6 views)...")
rgb_tta_probs = tta_predict(rgb_model, X_test_rgb)
y_true_rgb    = np.argmax(Y_test_rgb, axis=1)
rgb_tta_preds = np.argmax(rgb_tta_probs, axis=1)
rgb_tta_acc   = accuracy_score(y_true_rgb, rgb_tta_preds)

rgb_std_probs = np.concatenate([
    rgb_model(tf.constant(X_test_rgb[i:i+BATCH_SIZE], dtype=tf.float32),
              training=False).numpy()
    for i in range(0, len(X_test_rgb), BATCH_SIZE)
], axis=0)
rgb_std_preds = np.argmax(rgb_std_probs, axis=1)
rgb_std_acc   = accuracy_score(y_true_rgb, rgb_std_preds)

print(f"\nRGB std accuracy : {rgb_std_acc:.4f} ({rgb_std_acc*100:.2f}%)")
print(f"RGB TTA accuracy : {rgb_tta_acc:.4f} ({rgb_tta_acc*100:.2f}%)")
print(f"TTA boost        : +{(rgb_tta_acc-rgb_std_acc)*100:.2f}%")

plot_two_phase_curves(
    rgb_a_train_acc, rgb_a_train_loss, rgb_a_val_acc, rgb_a_val_loss,
    rgb_b_train_acc, rgb_b_train_loss, rgb_b_val_acc, rgb_b_val_loss,
    "EfficientNetV2-S (RGB)", "rgb_efficientnetv2s"
)
evaluate_model(y_true_rgb, rgb_tta_preds, rgb_tta_acc,
               "RGB: EfficientNetV2-S + GeM + 6-view TTA", "rgb_efficientnetv2s_tta")

rgb_model.save(str(OUTPUT_DIR / "rgb_efficientnetv2s_final.keras"))
print("RGB model saved.")

# Free GPU memory before TIF training
del rgb_aug_pipeline
gc.collect()
tf.keras.backend.clear_session()

# Rebuild model from saved weights for later comparison
rgb_model_saved_path = str(OUTPUT_DIR / "rgb_efficientnetv2s_final.keras")
print(f"RGB done. Model at: {rgb_model_saved_path}")

# %% [markdown]
# ---
# ## CHUNK 11 -- IMPROVED TIF MODEL (4-Stage ResNet + GeM + 15 channels)
# ---

# %%
# ============================================================
#  CHUNK 11 -- TIF MODEL ARCHITECTURE
# ============================================================
# Input: (64, 64, 15) -- 13 original bands + NDVI + NDWI
# Architecture (4 stages, deeper than original 3-stage ResNet):
#   Stem:    64x64 -> 16x16 (2x stride conv + maxpool)
#   Stage 1: 16x16x256  (stride=1, no spatial reduction)
#   Stage 2: 8x8x512    (stride=2)
#   Stage 3: 4x4x1024   (stride=2)
#   Stage 4: 2x2x2048   (stride=2) -- NEW vs original
#   GeM:     2048-dim vector
#   Head:    Dense(512)->BN->swish->Dropout(0.4)->Dense(256)->...->Dense(10)
#
# Note: input is only 64x64, so VRAM usage is small even with 4 stages.

def _id_block(x, filters):
    """Identity residual block -- input/output dimensions match."""
    f1, f2, f3 = filters
    sc = x
    x = layers.Conv2D(f1, (1,1), padding="valid")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(f2, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(f3, (1,1), padding="valid")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, sc])
    x = layers.Activation("relu")(x)
    return x

def _conv_block(x, filters, strides=(2,2)):
    """Convolutional residual block -- spatial size changes via stride."""
    f1, f2, f3 = filters
    sc = layers.Conv2D(f3, (1,1), strides=strides, padding="valid")(x)
    sc = layers.BatchNormalization()(sc)
    x = layers.Conv2D(f1, (1,1), strides=strides, padding="valid")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(f2, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(f3, (1,1), padding="valid")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, sc])
    x = layers.Activation("relu")(x)
    return x

def build_tif_model(input_shape=(64, 64, 15), num_classes=10):
    inp = layers.Input(shape=input_shape, name="tif_input")

    # Stem: 64x64 -> 32x32 -> 16x16
    x = layers.Conv2D(64, (7,7), strides=(2,2), padding="same", name="stem_conv")(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)
    x = layers.MaxPooling2D((3,3), strides=(2,2), padding="same", name="stem_pool")(x)
    # -> 16x16x64

    # Stage 1: 16x16 -> 16x16 (stride=1), 64->256
    x = _conv_block(x, (64, 64, 256), strides=(1,1))
    x = _id_block(x,   (64, 64, 256))
    x = _id_block(x,   (64, 64, 256))

    # Stage 2: 16x16 -> 8x8, 128->512
    x = _conv_block(x, (128, 128, 512))
    x = _id_block(x,   (128, 128, 512))
    x = _id_block(x,   (128, 128, 512))
    x = _id_block(x,   (128, 128, 512))

    # Stage 3: 8x8 -> 4x4, 256->1024
    x = _conv_block(x, (256, 256, 1024))
    x = _id_block(x,   (256, 256, 1024))
    x = _id_block(x,   (256, 256, 1024))
    x = _id_block(x,   (256, 256, 1024))
    x = _id_block(x,   (256, 256, 1024))
    x = _id_block(x,   (256, 256, 1024))

    # Stage 4 (NEW): 4x4 -> 2x2, 512->2048
    x = _conv_block(x, (512, 512, 2048))
    x = _id_block(x,   (512, 512, 2048))
    x = _id_block(x,   (512, 512, 2048))

    # GeM Pooling
    x = GeMPooling(p_init=3.0, p_trainable=True, name="gem_pool")(x)

    # Head
    x = layers.Dense(512, name="fc_512")(x)
    x = layers.BatchNormalization(name="bn_512")(x)
    x = layers.Activation("swish", name="swish_512")(x)
    x = layers.Dropout(0.4, name="drop_512")(x)
    x = layers.Dense(256, name="fc_256")(x)
    x = layers.BatchNormalization(name="bn_256")(x)
    x = layers.Activation("swish", name="swish_256")(x)
    x = layers.Dropout(0.3, name="drop_256")(x)
    out = layers.Dense(num_classes, activation="softmax",
                       dtype="float32", name="output")(x)

    return models.Model(inp, out, name="TIF_ImprovedResNet_GeM")

tif_model = build_tif_model(input_shape=(*TIF_IMG_SIZE, TIF_N_CHANNELS), num_classes=NUM_CLASSES)
tif_model.summary(line_length=100)
print(f"\nTIF Model -- Total params: {tif_model.count_params():,}")

# %%
# ============================================================
#  CHUNK 12 -- TIF TRAINING (AdamW + Label Smoothing + Cosine LR + MixUp)
# ============================================================

print("\n" + "=" * 65)
print("  TRAINING IMPROVED TIF MODEL")
print(f"  Optimizer: AdamW(lr={TIF_LR}, wd={WEIGHT_DECAY}) | "
      f"Label smooth: {LABEL_SMOOTH_TIF}")
print(f"  Max epochs: {TIF_EPOCHS} (EarlyStopping patience=10)")
print("=" * 65)

# Re-define TIF_LR here in case clear_session wiped the scope
TIF_LR = 1e-3

opt_tif     = optimizers.AdamW(learning_rate=TIF_LR, weight_decay=WEIGHT_DECAY)
loss_fn_tif = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH_TIF)

tif_train_acc, tif_val_acc   = [], []
tif_train_loss, tif_val_loss = [], []
best_tif_acc     = 0.0
best_tif_weights = None
patience_tif     = 0
PATIENCE_TIF     = 10

tif_start = time.time()
for epoch in range(TIF_EPOCHS):
    new_lr = cosine_lr(epoch, TIF_LR, warmup=3, total=TIF_EPOCHS, min_lr=1e-8)
    opt_tif.learning_rate.assign(new_lr)

    ep_losses, ep_accs = [], []
    for step, (xb, yb) in enumerate(train_tif_ds):
        # MixUp for TIF (spatial aug already done in generator)
        xn, yn = mixup(xb, yb, alpha=0.3)
        xb_t   = tf.constant(xn, dtype=tf.float32)
        yb_t   = tf.constant(yn, dtype=tf.float32)
        with tf.GradientTape() as tape:
            pred = tif_model(xb_t, training=True)
            loss = loss_fn_tif(yb_t, pred)
        grads = tape.gradient(loss, tif_model.trainable_weights)
        opt_tif.apply_gradients(zip(grads, tif_model.trainable_weights))
        acc = float(tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(pred,1), tf.argmax(tf.cast(yb_t,tf.float32),1)),
            tf.float32)))
        ep_losses.append(float(loss)); ep_accs.append(acc)
        if (step+1) % 50 == 0:
            print(f"  ep{epoch+1} step{step+1} loss={np.mean(ep_losses):.4f} acc={np.mean(ep_accs):.4f}", end="\r")

    vl_list, va_list = [], []
    for xv, yv in val_tif_ds:
        vp = tif_model(xv, training=False)
        vl_list.append(float(loss_fn_tif(yv, vp)))
        va_list.append(float(tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(vp,1), tf.argmax(yv,1)), tf.float32))))
    tl, ta = np.mean(ep_losses), np.mean(ep_accs)
    vl, va = np.mean(vl_list), np.mean(va_list)
    tif_train_acc.append(ta); tif_val_acc.append(va)
    tif_train_loss.append(tl); tif_val_loss.append(vl)
    elapsed = (time.time() - tif_start) / 60
    print(f"  Epoch {epoch+1:2d}/{TIF_EPOCHS} | loss:{tl:.4f} acc:{ta:.4f} | "
          f"val_loss:{vl:.4f} val_acc:{va:.4f} | lr:{new_lr:.1e} | {elapsed:.1f} min")

    if va > best_tif_acc:
        best_tif_acc = va; best_tif_weights = tif_model.get_weights(); patience_tif = 0
        tif_model.save(str(OUTPUT_DIR / "best_tif_improved.keras"))
        print(f"  --> Best val_acc: {best_tif_acc:.4f}  [saved]")
    else:
        patience_tif += 1
        if patience_tif >= PATIENCE_TIF:
            print(f"  EarlyStopping at epoch {epoch+1}")
            break

tif_model.set_weights(best_tif_weights)
tif_train_time = time.time() - tif_start
print(f"TIF training done in {tif_train_time/60:.1f} min | Best val_acc: {best_tif_acc:.4f}")
gc.collect()

# %%
# ============================================================
#  CHUNK 13 -- TIF TTA (5 views) + EVALUATION
# ============================================================

def tta_predict_tif(model, X, bs=BATCH_SIZE):
    """5-view TTA: original, H-flip, V-flip, HV-flip, 90-rot."""
    probs = np.zeros((len(X), NUM_CLASSES), dtype=np.float32)
    views = [
        X,
        X[:, :, ::-1, :],
        X[:, ::-1, :, :],
        X[:, ::-1, ::-1, :],
        np.rot90(X, k=1, axes=(1, 2)),
    ]
    for vi, Xv in enumerate(views):
        batch_probs = []
        for i in range(0, len(Xv), bs):
            p = model(tf.constant(Xv[i:i+bs], dtype=tf.float32), training=False).numpy()
            batch_probs.append(p)
        probs += np.concatenate(batch_probs, axis=0)
        print(f"  TIF TTA view {vi+1}/{len(views)}", end="\r")
    probs /= len(views)
    print(f"  TIF TTA complete ({len(views)} views).")
    return probs

def plot_single_curves(acc, loss, val_acc, val_loss, title, prefix):
    ep = list(range(1, len(acc)+1))
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    axes[0].plot(ep, acc,     "b-o", label="Train Acc", lw=2, ms=3)
    axes[0].plot(ep, val_acc, "r--s", label="Val Acc",  lw=2, ms=3)
    bi = int(np.argmax(val_acc))
    axes[0].axvline(x=ep[bi], color="green", ls="--", alpha=0.7,
                    label=f"Best ep{ep[bi]} ({val_acc[bi]:.4f})")
    axes[0].set_title(f"{title} -- Accuracy", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].spines[["top","right"]].set_visible(False)
    axes[1].plot(ep, loss,     "b-o", label="Train Loss", lw=2, ms=3)
    axes[1].plot(ep, val_loss, "r--s", label="Val Loss",  lw=2, ms=3)
    axes[1].axvline(x=ep[bi], color="green", ls="--", alpha=0.7)
    axes[1].set_title(f"{title} -- Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    save_fig(f"{prefix}_learning_curves.png")
    plt.show()

print("Running TIF TTA (5 views)...")
tif_tta_probs = tta_predict_tif(tif_model, X_test_tif)
y_true_tif    = np.argmax(Y_test_tif, axis=1)
tif_tta_preds = np.argmax(tif_tta_probs, axis=1)
tif_tta_acc   = accuracy_score(y_true_tif, tif_tta_preds)

tif_std_probs = np.concatenate([
    tif_model(tf.constant(X_test_tif[i:i+BATCH_SIZE], dtype=tf.float32),
              training=False).numpy()
    for i in range(0, len(X_test_tif), BATCH_SIZE)
], axis=0)
tif_std_preds = np.argmax(tif_std_probs, axis=1)
tif_std_acc   = accuracy_score(y_true_tif, tif_std_preds)

print(f"\nTIF std accuracy : {tif_std_acc:.4f} ({tif_std_acc*100:.2f}%)")
print(f"TIF TTA accuracy : {tif_tta_acc:.4f} ({tif_tta_acc*100:.2f}%)")
print(f"TTA boost        : +{(tif_tta_acc-tif_std_acc)*100:.2f}%")

plot_single_curves(
    tif_train_acc, tif_train_loss, tif_val_acc, tif_val_loss,
    "TIF Improved ResNet (15 channels)", "tif_improved"
)
evaluate_model(y_true_tif, tif_tta_preds, tif_tta_acc,
               "TIF: Improved ResNet + GeM + 5-view TTA", "tif_improved_tta")

tif_model.save(str(OUTPUT_DIR / "tif_improved_final.keras"))
print("TIF model saved.")
gc.collect()

# %% [markdown]
# ---
# ## CHUNK 14 -- FINAL 4-WAY COMPARISON
# ---

# %%
# ============================================================
#  CHUNK 14 -- FINAL COMPARISON + SUMMARY
# ============================================================
# Update the two reference constants below with your actual scores
# from the ORIGINAL pipeline (eurosat_model_pipeline.py / eurosat_kaggle_pipeline.py).

ORIG_RGB_ACC = 0.919   # <-- replace with actual result from original pipeline
ORIG_TIF_ACC = 0.940   # <-- replace with actual result from original pipeline

print("=" * 70)
print("  4-WAY MODEL COMPARISON")
print("=" * 70)

comparison = {
    "Model": [
        "Original RGB  (ResNet-50, frozen)          [reference]",
        "Improved RGB  (EfficientNetV2-S + TTA)     [this run ]",
        "Original TIF  (3-stage ResNet, 13ch)       [reference]",
        "Improved TIF  (4-stage ResNet, 15ch + TTA) [this run ]",
    ],
    "Accuracy": [
        f"{ORIG_RGB_ACC*100:.2f}%",
        f"{rgb_tta_acc*100:.2f}%",
        f"{ORIG_TIF_ACC*100:.2f}%",
        f"{tif_tta_acc*100:.2f}%",
    ],
    "Delta": [
        "--",
        f"{(rgb_tta_acc-ORIG_RGB_ACC)*100:+.2f}%",
        "--",
        f"{(tif_tta_acc-ORIG_TIF_ACC)*100:+.2f}%",
    ],
    "Key improvements": [
        "Frozen backbone, GAP, basic aug, Adam",
        "Fine-tuned, GeM, CutMix/MixUp, AdamW, TTA",
        "3 stages, 13ch, GAP, no aug, Adam",
        "4 stages, 15ch, GeM, band-dropout, MixUp, AdamW, TTA",
    ]
}
import pandas as pd
print("\n" + pd.DataFrame(comparison).to_string(index=False))

# Bar chart
fig, ax = plt.subplots(figsize=(14, 6))
labels_bar = [
    "RGB\nOriginal\n(ResNet-50)",
    "RGB\nImproved\n(EffNetV2-S)",
    "TIF\nOriginal\n(3-stage)",
    "TIF\nImproved\n(4-stage)",
]
accs_bar   = [ORIG_RGB_ACC*100, rgb_tta_acc*100, ORIG_TIF_ACC*100, tif_tta_acc*100]
colors_bar = ["#95a5a6", "#2ecc71", "#e67e22", "#3498db"]
bars = ax.bar(labels_bar, accs_bar, color=colors_bar,
              edgecolor="white", linewidth=1.5, width=0.55)
for bar, val in zip(bars, accs_bar):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15,
            f"{val:.2f}%", ha="center", fontsize=13, fontweight="bold")
ax.set_title("4-Way Model Accuracy Comparison\n(Original vs Improved -- RGB and TIF)",
             fontweight="bold", fontsize=13)
ax.set_ylabel("Test Accuracy (%)"); ax.set_ylim(85, 100)
ax.axhline(y=ORIG_RGB_ACC*100, color="gray", ls="--", alpha=0.4, lw=1)
ax.axhline(y=ORIG_TIF_ACC*100, color="gray", ls="--", alpha=0.4, lw=1)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
save_fig("4way_model_comparison.png")
plt.show()

total_time = rgb_total_time + tif_train_time
print(f"""
+------------------------------------------------------------------+
|         HIGH-ACCURACY COMBINED PIPELINE -- FINAL SUMMARY        |
|         Hardware: i7-9850H | 16 GB RAM | Quadro T1000 4 GB      |
+------------------------------------------------------------------+
|                                                                  |
|  RGB MODEL (EfficientNetV2-S + GeM):                            |
|    Phase A : Head warmup, backbone frozen, label smooth 0.10    |
|    Phase B : Top-{RGB_UNFREEZE_LAYERS} layers, AdamW, cosine LR               |
|    Aug     : CutMix + MixUp + Flip + Zoom + Contrast + Translate|
|    TTA     : 6-view average                                      |
|    Accuracy: {rgb_tta_acc*100:.2f}%  (delta vs orig: {(rgb_tta_acc-ORIG_RGB_ACC)*100:+.2f}%)             |
|                                                                  |
|  TIF MODEL (4-stage ResNet + GeM + 15 channels):               |
|    Input   : 13 bands + NDVI + NDWI  (15 channels total)       |
|    Aug     : Spatial flips + band dropout + MixUp               |
|    Opt     : AdamW + cosine LR + label smooth 0.08             |
|    TTA     : 5-view average                                      |
|    Accuracy: {tif_tta_acc*100:.2f}%  (delta vs orig: {(tif_tta_acc-ORIG_TIF_ACC)*100:+.2f}%)             |
|                                                                  |
|  Total training time: {total_time/60:.0f} min                               |
|                                                                  |
|  Saved to: outputs/                                             |
|    best_rgb_phase_a.keras                                       |
|    best_rgb_phase_b.keras                                       |
|    rgb_efficientnetv2s_final.keras                              |
|    best_tif_improved.keras                                      |
|    tif_improved_final.keras                                     |
|    (plus PNG plots for all evaluation charts)                   |
+------------------------------------------------------------------+
""")

