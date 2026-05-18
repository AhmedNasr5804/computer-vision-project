"""
Phase 1a — Sanity overfit on the ORIGINAL SegFormer masks.

Hypothesis from the doctor: the SegFormer-Cityscapes masks are rubbish on
this domain (dark tape on light fabric, no real road context). If even a
tiny CNN cannot overfit a 50-image subset with these masks, the masks are
unfit for training and must be regenerated (Phase 2).

Subject:
  - 50 random images stratified across the 3 source videos (left_01, right_01, straight_02)
  - Original 720x1280 binary masks downsampled to 128x128
  - A 4,941-parameter U-Net-lite, no augmentation, no regularization
  - BCE + Dice loss, Adam lr=1e-3, 60 epochs

Pass criterion for sanity:
  - train_iou_final >= 0.95  -> masks have signal
  - train_iou_final <  0.70  -> masks are rubbish (expected here)
"""
import os, sys, io, json, random
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

SEED = 42
random.seed(SEED); np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

ROOT = Path("D:/Spring forth year/Computer Vision/project")
DS = ROOT / "fine_tuning" / "lane_fine_tuning" / "segmentation_dataset"
IMG_DIR = DS / "images"
MASK_DIR = DS / "masks"
ART = ROOT / "artifacts"

# Stratified subset: 17/17/16 per direction
N_PER = [17, 17, 16]
PREFIXES = ["left_left_01", "right_right_01", "straight_straight_02"]

def collect_subset():
    subset = []
    for prefix, n in zip(PREFIXES, N_PER):
        imgs = sorted([p for p in IMG_DIR.glob(f"{prefix}_*.jpg")])
        random.shuffle(imgs)
        subset.extend(imgs[:n])
    return subset

SUBSET = collect_subset()
print(f"subset: {len(SUBSET)} images")

H = W = 128
X = np.zeros((len(SUBSET), H, W, 3), dtype=np.float32)
Y = np.zeros((len(SUBSET), H, W, 1), dtype=np.float32)
for i, ip in enumerate(SUBSET):
    img = cv2.imread(str(ip))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    X[i] = img.astype(np.float32) / 255.0

    mp = MASK_DIR / (ip.stem + ".png")
    mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    Y[i, ..., 0] = (mask > 127).astype(np.float32)

print(f"X={X.shape}, Y={Y.shape}, mean mask coverage={Y.mean():.4f}")

# Sanity-tiny U-Net (~5k params)
def build_tiny_unet(h=H, w=W, c=3):
    inp = tf.keras.Input(shape=(h, w, c))
    # encoder
    e1 = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inp)
    p1 = tf.keras.layers.MaxPool2D()(e1)
    e2 = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(p1)
    p2 = tf.keras.layers.MaxPool2D()(e2)
    # bottleneck
    b = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(p2)
    # decoder
    u2 = tf.keras.layers.UpSampling2D()(b)
    d2 = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(
        tf.keras.layers.Concatenate()([u2, e2])
    )
    u1 = tf.keras.layers.UpSampling2D()(d2)
    d1 = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(
        tf.keras.layers.Concatenate()([u1, e1])
    )
    out = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(d1)
    return tf.keras.Model(inp, out)

def dice_loss(y_true, y_pred, smooth=1.0):
    yt = tf.cast(y_true, tf.float32)
    yp = tf.cast(y_pred, tf.float32)
    inter = tf.reduce_sum(yt * yp, axis=[1, 2, 3])
    s = tf.reduce_sum(yt, axis=[1, 2, 3]) + tf.reduce_sum(yp, axis=[1, 2, 3])
    return 1.0 - (2 * inter + smooth) / (s + smooth)

def bce_dice(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce, axis=[1, 2])  # reduce H,W -> [batch]
    return bce + dice_loss(y_true, y_pred)

def iou_metric(y_true, y_pred):
    yt = tf.cast(y_true > 0.5, tf.float32)
    yp = tf.cast(y_pred > 0.5, tf.float32)
    inter = tf.reduce_sum(yt * yp, axis=[1, 2, 3])
    union = tf.reduce_sum(yt, axis=[1, 2, 3]) + tf.reduce_sum(yp, axis=[1, 2, 3]) - inter
    return tf.reduce_mean((inter + 1e-7) / (union + 1e-7))

model = build_tiny_unet()
print(f"model params: {model.count_params()}")
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=bce_dice, metrics=[iou_metric])

h = model.fit(X, Y, batch_size=8, epochs=60, verbose=0)

final_loss = float(h.history["loss"][-1])
final_iou = float(h.history["iou_metric"][-1])
print(f"\nFinal train loss = {final_loss:.4f}")
print(f"Final train IoU  = {final_iou:.4f}")

# Save curves
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(h.history["loss"], color="#cc2222")
axes[0].set_title("Train BCE+Dice loss"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
axes[0].grid(alpha=0.3)
axes[1].plot(h.history["iou_metric"], color="#2266aa")
axes[1].set_title("Train IoU"); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("IoU")
axes[1].axhline(0.95, color="green", linestyle="--", linewidth=0.6, label="overfit pass (0.95)")
axes[1].axhline(0.70, color="orange", linestyle="--", linewidth=0.6, label="rubbish threshold (0.70)")
axes[1].set_ylim(0, 1); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
plt.suptitle("Phase 1a — Sanity overfit on ORIGINAL SegFormer masks (50 imgs)", y=1.02)
plt.tight_layout()
plt.savefig(ART / "lane_phase1a_overfit_original.png", dpi=140, bbox_inches="tight")
plt.close()

# Also save a 4-tile sample showing image / mask / prediction
preds = model.predict(X[:4], verbose=0)
fig, axes = plt.subplots(4, 3, figsize=(8, 11))
for r in range(4):
    axes[r, 0].imshow(X[r]); axes[r, 0].set_title("image"); axes[r, 0].axis("off")
    axes[r, 1].imshow(Y[r, ..., 0], cmap="gray", vmin=0, vmax=1)
    axes[r, 1].set_title(f"original mask\n({100*Y[r].mean():.1f}% nonzero)"); axes[r, 1].axis("off")
    axes[r, 2].imshow(preds[r, ..., 0], cmap="gray", vmin=0, vmax=1)
    axes[r, 2].set_title("model prediction"); axes[r, 2].axis("off")
plt.suptitle("Phase 1a — image / SegFormer mask / model output", y=1.0)
plt.tight_layout()
plt.savefig(ART / "lane_phase1a_samples.png", dpi=140, bbox_inches="tight")
plt.close()

verdict = ("MASKS HAVE SIGNAL" if final_iou >= 0.95
           else "MASKS ARE RUBBISH" if final_iou < 0.70
           else "INCONCLUSIVE")

report = {
    "subset_size": len(SUBSET),
    "subset_per_direction": dict(zip(["left", "right", "straight"], N_PER)),
    "n_params": int(model.count_params()),
    "epochs": 60,
    "final_train_loss": final_loss,
    "final_train_iou": final_iou,
    "mean_mask_coverage": float(Y.mean()),
    "verdict": verdict,
}
with open(ART / "lane_phase1a_results.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n=== VERDICT: {verdict} ===")
print(json.dumps(report, indent=2))
