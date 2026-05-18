"""
Phase 6 — Extend from binary to 3-class segmentation (background / lane-left /
lane-right). Same truncated U-Net architecture but the final head is
Conv(3, 1x1, softmax). Loss: CCE + multi-class Dice.

3-class mask generation: median-column-x bisection of the binary lane region.
This gives the model a meaningful left/right partition tied to where the
lane's bisecting axis falls in the frame.
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
M3 = DS / "masks_3class"; M3.mkdir(exist_ok=True)
ART = ROOT / "artifacts"
H = W = 128; N_CLASSES = 3

# --------------------------------------------------------------------------
# 1) Generate 3-class masks (median-x bisection)
# --------------------------------------------------------------------------
def make_3class_mask(bm):
    h, w = bm.shape
    lane = bm > 127
    if lane.sum() == 0:
        return np.zeros_like(bm, dtype=np.uint8)
    _, xs = np.where(lane)
    cx = int(np.median(xs))
    out = np.zeros((h, w), dtype=np.uint8)
    left  = lane & (np.arange(w)[None, :] < cx)
    right = lane & ~left
    out[left]  = 1
    out[right] = 2
    return out

bin_paths = sorted(MK.glob("*.png"))
todo = [bp for bp in bin_paths if not (M3 / bp.name).exists()]
log(f"3-class masks to generate: {len(todo)} of {len(bin_paths)}")
t0 = time.time()
for i, bp in enumerate(todo):
    bm = cv2.imread(str(bp), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(M3 / bp.name), make_3class_mask(bm))
    if (i + 1) % 2000 == 0:
        log(f"  generated {i+1}/{len(todo)} (elapsed {time.time()-t0:.0f}s)")
log(f"3-class mask generation done in {time.time()-t0:.1f}s")

# --------------------------------------------------------------------------
# 2) Same split as Phase 4
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

def to_split(paths, stems): return [p for p in paths if base_stem(p.stem) in stems]
train_paths = [p for d in by_dir.values() for p in to_split(d, train_stems)]
val_paths   = [p for d in by_dir.values() for p in to_split(d, val_stems)]
test_paths  = [p for d in by_dir.values() for p in to_split(d, test_stems)]
log(f"split: train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}")

# --------------------------------------------------------------------------
# 3) Pre-load into RAM as one-hot masks
# --------------------------------------------------------------------------
def load_split(paths, name):
    n = len(paths)
    X = np.zeros((n, H, W, 3), dtype=np.float32)
    Y = np.zeros((n, H, W, N_CLASSES), dtype=np.float32)
    t0 = time.time()
    eye = np.eye(N_CLASSES, dtype=np.float32)
    for i, p in enumerate(paths):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        X[i] = img.astype(np.float32) / 255.0
        mask = cv2.imread(str(M3 / (p.stem + ".png")), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        Y[i] = eye[mask]
        if (i + 1) % 2000 == 0:
            log(f"  loaded {name} {i+1}/{n} (elapsed {time.time()-t0:.0f}s)")
    log(f"loaded {name}: X={X.shape} Y={Y.shape} in {time.time()-t0:.1f}s "
        f"(class shares: bg={Y[...,0].mean():.3f} L={Y[...,1].mean():.3f} R={Y[...,2].mean():.3f})")
    return X, Y

X_tr, Y_tr = load_split(train_paths, "train")
X_va, Y_va = load_split(val_paths,   "val")
X_te, Y_te = load_split(test_paths,  "test")

# --------------------------------------------------------------------------
# 4) Truncated U-Net w/ 3-class softmax head
# --------------------------------------------------------------------------
def build_truncated_unet_3c(h=H, w=W, c=3, nc=N_CLASSES):
    inp = tf.keras.Input(shape=(h, w, c))
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
    out = tf.keras.layers.Conv2D(nc, 1, activation="softmax")(d1)
    return tf.keras.Model(inp, out, name="lane_truncated_unet_3c")

def multi_dice(y_true, y_pred, smooth=1.0):
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2])   # (B,C)
    s = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
    return 1.0 - tf.reduce_mean((2 * inter + smooth) / (s + smooth), axis=-1)

def cce_dice(y_true, y_pred):
    cce = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred), axis=[1, 2])
    return cce + multi_dice(y_true, y_pred)

def make_iou(cls):
    def fn(y_true, y_pred):
        yt = tf.cast(y_true[..., cls] > 0.5, tf.float32)
        yp = tf.cast(tf.argmax(y_pred, axis=-1) == cls, tf.float32)
        inter = tf.reduce_sum(yt * yp, axis=[1, 2])
        union = tf.reduce_sum(yt, axis=[1, 2]) + tf.reduce_sum(yp, axis=[1, 2]) - inter
        return tf.reduce_mean((inter + 1e-7) / (union + 1e-7))
    fn.__name__ = f"iou_c{cls}"
    return fn

model = build_truncated_unet_3c()
log(f"params: {model.count_params()}")
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=cce_dice, metrics=[make_iou(0), make_iou(1), make_iou(2)])

cbs = [
    tf.keras.callbacks.EarlyStopping(monitor="val_iou_c1", mode="max",
                                     patience=6, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3,
                                         factor=0.5, min_lr=1e-5, verbose=1),
]
log("Starting 3-class training (max 30 epochs)...")
t0 = time.time()
h = model.fit(X_tr, Y_tr, batch_size=32, epochs=30, validation_data=(X_va, Y_va),
              callbacks=cbs, verbose=2)
t_train = time.time() - t0

test_metrics = model.evaluate(X_te, Y_te, batch_size=32, verbose=0)
log(f"test: loss={test_metrics[0]:.4f} iou_bg={test_metrics[1]:.4f} "
    f"iou_left={test_metrics[2]:.4f} iou_right={test_metrics[3]:.4f}")

model.save(str(ART / "lane_truncated_3class.keras"))

# PTQ-int8
rep_idx = np.random.RandomState(SEED).choice(len(X_tr), 200, replace=False)
REP = X_tr[rep_idx]
def rep_gen():
    for i in range(len(REP)):
        yield [REP[i:i+1].astype(np.float32)]

conv = tf.lite.TFLiteConverter.from_keras_model(model)
fp32_bytes = conv.convert()
OUT_FP32 = ART / "lane_truncated_3class.tflite"
OUT_FP32.write_bytes(fp32_bytes)

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_gen
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
int8_bytes = conv.convert()
OUT_INT8 = ART / "lane_truncated_3class_int8.tflite"
OUT_INT8.write_bytes(int8_bytes)
log(f"3class FP32: {OUT_FP32.stat().st_size/1024:.1f} KB | "
    f"INT8: {OUT_INT8.stat().st_size/1024:.1f} KB "
    f"(compression {OUT_FP32.stat().st_size/OUT_INT8.stat().st_size:.2f}x)")

# Plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(h.history["loss"], label="train"); axes[0].plot(h.history["val_loss"], label="val")
axes[0].set_title("CCE+Dice loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
for cls, name in enumerate(["background", "lane-left", "lane-right"]):
    axes[1].plot(h.history[f"val_iou_c{cls}"], label=f"val {name}")
axes[1].axhline(0.6, color="orange", linestyle="--", linewidth=0.6, label="gate (0.60)")
axes[1].set_title("Per-class IoU (val)"); axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_ylim(0, 1)
plt.suptitle(f"Phase 6 — 3-class Truncated U-Net ({model.count_params()} params)", y=1.02)
plt.tight_layout()
plt.savefig(ART / "lane_phase6_train_curves.png", dpi=140, bbox_inches="tight")
plt.close()

idx = np.random.RandomState(SEED).choice(len(X_te), 8, replace=False)
preds_oh = model.predict(X_te[idx], verbose=0)
preds = np.argmax(preds_oh, axis=-1)
gt = np.argmax(Y_te[idx], axis=-1)
fig, axes = plt.subplots(8, 3, figsize=(9, 22))
for r in range(8):
    axes[r, 0].imshow(X_te[idx[r]]); axes[r, 0].set_title("image", fontsize=8); axes[r, 0].axis("off")
    axes[r, 1].imshow(gt[r], cmap="viridis", vmin=0, vmax=2)
    axes[r, 1].set_title("gt (0=bg 1=L 2=R)", fontsize=8); axes[r, 1].axis("off")
    axes[r, 2].imshow(preds[r], cmap="viridis", vmin=0, vmax=2)
    axes[r, 2].set_title("pred", fontsize=8); axes[r, 2].axis("off")
plt.suptitle("Phase 6 — 3-class test predictions", y=1.0, fontsize=11)
plt.tight_layout()
plt.savefig(ART / "lane_phase6_test_preds.png", dpi=140, bbox_inches="tight")
plt.close()

summary = {
    "n_params": int(model.count_params()),
    "epochs_run": len(h.history["loss"]),
    "train_seconds": t_train,
    "test_loss": float(test_metrics[0]),
    "test_iou_bg":    float(test_metrics[1]),
    "test_iou_left":  float(test_metrics[2]),
    "test_iou_right": float(test_metrics[3]),
    "test_mean_iou":  float(np.mean(test_metrics[1:4])),
    "fp32_size_kb": OUT_FP32.stat().st_size / 1024,
    "int8_size_kb": OUT_INT8.stat().st_size / 1024,
    "gate_passed": (float(test_metrics[2]) >= 0.6) and (float(test_metrics[3]) >= 0.6),
}
with open(ART / "lane_phase6_results.json", "w") as f:
    json.dump(summary, f, indent=2)
log(json.dumps(summary, indent=2))
