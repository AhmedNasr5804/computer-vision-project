"""
Phase 3 + Phase 4 — Build the truncated U-Net and train it on the new binary
masks. Pre-loads all images/masks into RAM for fast iteration (no disk I/O
inside the training loop).
"""
import sys, io, json, time, re, random
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
def log(*a, **kw):
    print(*a, **kw); sys.stdout.flush()

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)

ROOT = Path("D:/Spring forth year/Computer Vision/project")
DS = ROOT / "fine_tuning" / "lane_fine_tuning" / "segmentation_dataset"
IM = DS / "images"; MK = DS / "masks_binary"
ART = ROOT / "artifacts"
H = W = 128

# --------------------------------------------------------------------------
# 1) Split source-frame stems 70/15/15, stratified by direction
# --------------------------------------------------------------------------
def base_stem(stem): return re.sub(r"_dup\d+$", "", stem)
all_paths = sorted(IM.glob("*.jpg"))
by_dir = {"left_left_01": [], "right_right_01": [], "straight_straight_02": []}
for p in all_paths:
    by_dir[base_stem(p.stem).split("_f")[0]].append(p)

def split_stems(paths):
    stems = sorted({base_stem(p.stem) for p in paths})
    rng = random.Random(SEED); rng.shuffle(stems)
    n = len(stems); n_tr = int(0.70 * n); n_va = int(0.15 * n)
    return set(stems[:n_tr]), set(stems[n_tr:n_tr + n_va]), set(stems[n_tr + n_va:])

train_stems, val_stems, test_stems = set(), set(), set()
for d, paths in by_dir.items():
    tr, va, te = split_stems(paths)
    train_stems |= tr; val_stems |= va; test_stems |= te
    log(f"{d}: train_stems={len(tr)} val_stems={len(va)} test_stems={len(te)}")

def to_split(paths, stems): return [p for p in paths if base_stem(p.stem) in stems]
train_paths = [p for d in by_dir.values() for p in to_split(d, train_stems)]
val_paths   = [p for d in by_dir.values() for p in to_split(d, val_stems)]
test_paths  = [p for d in by_dir.values() for p in to_split(d, test_stems)]
log(f"images: train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}")

# --------------------------------------------------------------------------
# 2) Pre-load all splits into RAM (uint8 -> float32 once)
# --------------------------------------------------------------------------
def load_split(paths, name):
    n = len(paths)
    X = np.zeros((n, H, W, 3), dtype=np.float32)
    Y = np.zeros((n, H, W, 1), dtype=np.float32)
    t0 = time.time()
    for i, p in enumerate(paths):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        X[i] = img.astype(np.float32) / 255.0
        mask = cv2.imread(str(MK / (p.stem + ".png")), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        Y[i, ..., 0] = (mask > 127).astype(np.float32)
        if (i + 1) % 1500 == 0:
            log(f"  loaded {name} {i+1}/{n} (elapsed {time.time()-t0:.0f}s)")
    log(f"loaded {name}: X={X.shape} Y={Y.shape} in {time.time()-t0:.1f}s | mean coverage = {Y.mean():.3f}")
    return X, Y

X_tr, Y_tr = load_split(train_paths, "train")
X_va, Y_va = load_split(val_paths,   "val")
X_te, Y_te = load_split(test_paths,  "test")

# --------------------------------------------------------------------------
# 3) Truncated U-Net (~12k params, 2 enc + 2 dec)
# --------------------------------------------------------------------------
def build_truncated_unet(h=H, w=W, c=3):
    inp = tf.keras.Input(shape=(h, w, c), name="image")
    e1 = tf.keras.layers.Conv2D(8, 3, padding="same", use_bias=False)(inp)
    e1 = tf.keras.layers.BatchNormalization()(e1); e1 = tf.keras.layers.ReLU()(e1)
    p1 = tf.keras.layers.MaxPool2D()(e1)
    e2 = tf.keras.layers.Conv2D(16, 3, padding="same", use_bias=False)(p1)
    e2 = tf.keras.layers.BatchNormalization()(e2); e2 = tf.keras.layers.ReLU()(e2)
    p2 = tf.keras.layers.MaxPool2D()(e2)
    b = tf.keras.layers.Conv2D(24, 3, padding="same", use_bias=False)(p2)
    b = tf.keras.layers.BatchNormalization()(b); b = tf.keras.layers.ReLU()(b)
    u2 = tf.keras.layers.UpSampling2D()(b)
    c2 = tf.keras.layers.Concatenate()([u2, e2])
    d2 = tf.keras.layers.Conv2D(16, 3, padding="same", use_bias=False)(c2)
    d2 = tf.keras.layers.BatchNormalization()(d2); d2 = tf.keras.layers.ReLU()(d2)
    u1 = tf.keras.layers.UpSampling2D()(d2)
    c1 = tf.keras.layers.Concatenate()([u1, e1])
    d1 = tf.keras.layers.Conv2D(8, 3, padding="same", use_bias=False)(c1)
    d1 = tf.keras.layers.BatchNormalization()(d1); d1 = tf.keras.layers.ReLU()(d1)
    out = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="mask")(d1)
    return tf.keras.Model(inp, out, name="lane_truncated_unet")

def dice_loss(y_true, y_pred, smooth=1.0):
    yt = tf.cast(y_true, tf.float32); yp = tf.cast(y_pred, tf.float32)
    inter = tf.reduce_sum(yt * yp, axis=[1, 2, 3])
    s = tf.reduce_sum(yt, axis=[1, 2, 3]) + tf.reduce_sum(yp, axis=[1, 2, 3])
    return 1.0 - (2 * inter + smooth) / (s + smooth)

def bce_dice(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred), axis=[1, 2])
    return bce + dice_loss(y_true, y_pred)

def iou_metric(y_true, y_pred):
    yt = tf.cast(y_true > 0.5, tf.float32); yp = tf.cast(y_pred > 0.5, tf.float32)
    inter = tf.reduce_sum(yt * yp, axis=[1, 2, 3])
    union = tf.reduce_sum(yt, axis=[1, 2, 3]) + tf.reduce_sum(yp, axis=[1, 2, 3]) - inter
    return tf.reduce_mean((inter + 1e-7) / (union + 1e-7))

def dice_metric(y_true, y_pred):
    yt = tf.cast(y_true > 0.5, tf.float32); yp = tf.cast(y_pred > 0.5, tf.float32)
    inter = tf.reduce_sum(yt * yp, axis=[1, 2, 3])
    s = tf.reduce_sum(yt, axis=[1, 2, 3]) + tf.reduce_sum(yp, axis=[1, 2, 3])
    return tf.reduce_mean((2 * inter + 1e-7) / (s + 1e-7))

model = build_truncated_unet()
log(f"params: {model.count_params()}")
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=bce_dice, metrics=[iou_metric, dice_metric])

cbs = [
    tf.keras.callbacks.EarlyStopping(monitor="val_iou_metric", mode="max",
                                     patience=6, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3,
                                         factor=0.5, min_lr=1e-5, verbose=1),
]

log("Starting training (max 30 epochs, EarlyStopping patience=6)...")
t0 = time.time()
h = model.fit(X_tr, Y_tr, batch_size=32, epochs=30, validation_data=(X_va, Y_va),
              callbacks=cbs, verbose=2)
t_train = time.time() - t0
log(f"train wall-clock: {t_train:.1f}s ({len(h.history['loss'])} epochs)")

test_results = model.evaluate(X_te, Y_te, batch_size=32, verbose=0)
log(f"test: loss={test_results[0]:.4f} iou={test_results[1]:.4f} dice={test_results[2]:.4f}")

model.save(str(ART / "lane_truncated.keras"))
log(f"Saved: {ART / 'lane_truncated.keras'}")

# Plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(h.history["loss"], label="train"); axes[0].plot(h.history["val_loss"], label="val")
axes[0].set_title("BCE+Dice loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(h.history["iou_metric"], label="train"); axes[1].plot(h.history["val_iou_metric"], label="val")
axes[1].set_title("IoU"); axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_ylim(0, 1)
axes[2].plot(h.history["dice_metric"], label="train"); axes[2].plot(h.history["val_dice_metric"], label="val")
axes[2].set_title("Dice"); axes[2].legend(); axes[2].grid(alpha=0.3); axes[2].set_ylim(0, 1)
plt.suptitle(f"Phase 4 — Truncated U-Net ({model.count_params()} params) on new binary masks", y=1.02)
plt.tight_layout()
plt.savefig(ART / "lane_phase4_train_curves.png", dpi=140, bbox_inches="tight")
plt.close()

# 8 random test predictions
idx = np.random.RandomState(SEED).choice(len(X_te), 8, replace=False)
preds = model.predict(X_te[idx], verbose=0)
fig, axes = plt.subplots(8, 3, figsize=(9, 22))
for r, k in enumerate(idx):
    axes[r, 0].imshow(X_te[k]); axes[r, 0].set_title("image", fontsize=8); axes[r, 0].axis("off")
    axes[r, 1].imshow(Y_te[k, ..., 0], cmap="gray", vmin=0, vmax=1)
    axes[r, 1].set_title("gt (binary)", fontsize=8); axes[r, 1].axis("off")
    axes[r, 2].imshow(preds[r, ..., 0], cmap="gray", vmin=0, vmax=1)
    axes[r, 2].set_title("pred (sigmoid)", fontsize=8); axes[r, 2].axis("off")
plt.suptitle("Phase 4 — Truncated U-Net test predictions", y=1.0, fontsize=11)
plt.tight_layout()
plt.savefig(ART / "lane_phase4_test_preds.png", dpi=140, bbox_inches="tight")
plt.close()

summary = {
    "n_params": int(model.count_params()),
    "split_counts": {
        "train": len(train_paths), "val": len(val_paths), "test": len(test_paths),
        "train_source_frames": len(train_stems),
        "val_source_frames":   len(val_stems),
        "test_source_frames":  len(test_stems),
    },
    "best_val_iou":  float(max(h.history["val_iou_metric"])),
    "best_val_dice": float(max(h.history["val_dice_metric"])),
    "epochs_run":    len(h.history["loss"]),
    "train_seconds": t_train,
    "test_loss":  float(test_results[0]),
    "test_iou":   float(test_results[1]),
    "test_dice":  float(test_results[2]),
    "gate_passed": float(test_results[1]) >= 0.70,
}
with open(ART / "lane_phase4_results.json", "w") as f:
    json.dump(summary, f, indent=2)
log(json.dumps(summary, indent=2))
