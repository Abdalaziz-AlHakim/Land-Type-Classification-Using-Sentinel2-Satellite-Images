import sys, importlib

required = [
    ("tensorflow", "TensorFlow"),
    ("numpy",      "NumPy"),
    ("pandas",     "Pandas"),
    ("matplotlib", "Matplotlib"),
    ("seaborn",    "Seaborn"),
    ("sklearn",    "scikit-learn"),
    ("PIL",        "Pillow"),
    ("cv2",        "OpenCV"),
    ("rasterio",   "Rasterio"),
]

all_ok = True
for mod, label in required:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "OK")
        print(f"  [OK]  {label}: {ver}")
    except ImportError:
        print(f"  [MISS] {label} -- NOT INSTALLED")
        all_ok = False

import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(f"\n  GPUs detected: {len(gpus)}")
for g in gpus:
    print(f"    {g.name}")
if not gpus:
    print("  No GPU found -- will run on CPU (slow but works)")

sys.exit(0 if all_ok else 1)
