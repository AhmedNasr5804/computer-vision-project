"""
Regenerate the two paper figures that reference the fine-tune set, now using
the 32 new captures (20260517_*) and omitting the original 12 family photos
(20260515_*) per privacy ask.

Figures emitted into paper/figures/:
  - fig_eye_phone_detect.png    one open/closed pair from the new 32
  - fig_eye_app_finetune.png    before/after P(open) bars for the 32 new

Also prints the zero-shot and post-finetune accuracies computed on the new-32
subset so the paper text can quote consistent numbers.
"""
import os, json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

ROOT = Path("D:/Spring forth year/Computer Vision/project")
ART  = ROOT / "artifacts"
FIG  = ROOT / "paper" / "figures"
FT_DIR = ROOT / "fine_tuning" / "CEW_fine_tuning"

with open(ART / "eye_finetune_labels.json") as f:
    lbl_map = json.load(f)

# Keep only the new 32 captures
NEW_NAMES = sorted([n for n in lbl_map.keys() if n.startswith("20260517_")])
print(f"new captures: {len(NEW_NAMES)}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def auto_orient_face(rgb):
    best = (0, None, None)
    for k in range(4):
        rot = np.rot90(rgb, k=k)
        gray = cv2.cvtColor(rot, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60,60))
        if len(faces)==0: continue
        a = max(w*h for x,y,w,h in faces)
        if a > best[0]: best=(a,rot,faces)
    return best

def crop_100(rot, faces):
    if rot is None or len(faces)==0: return None
    x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
    pad = int(0.05*max(w,h))
    x0=max(0,x-pad); y0=max(0,y-pad)
    x1=min(rot.shape[1],x+w+pad); y1=min(rot.shape[0],y+h+pad)
    return cv2.resize(rot[y0:y1, x0:x1], (100,100))

# --- Load both models ---
base = tf.keras.models.load_model(str(ART / "eye_winner.keras"))
ft   = tf.keras.models.load_model(str(ART / "eye_winner_finetuned.keras"))
H,W,C = base.input_shape[1:]
print(f"model input: ({H},{W},{C})")

def prep(crop):
    img = cv2.resize(crop, (W,H))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    return (gray[..., None].astype(np.float32) / 255.0)

# --- Build paired arrays for the new 32 ---
pairs = []  # (name, label_int, x, full_rot, bbox)
for n in NEW_NAMES:
    fp = FT_DIR / n
    img = cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB)
    area, rot, faces = auto_orient_face(img)
    c = crop_100(rot, faces)
    if c is None: continue
    y = 1 if lbl_map[n] == "open" else 0
    bb = max(faces, key=lambda f: f[2]*f[3]) if faces is not None else None
    pairs.append((n, y, prep(c), rot, bb))

print(f"usable pairs (new 32): {len(pairs)}")

Xnew = np.stack([p[2] for p in pairs])
ynew = np.array([p[1] for p in pairs])

p_before = base.predict(Xnew, verbose=0)   # before fine-tune
p_after  = ft.predict(Xnew,   verbose=0)   # after fine-tune
acc_before = float((p_before.argmax(axis=1) == ynew).mean())
acc_after  = float((p_after .argmax(axis=1) == ynew).mean())
print(f"new-32 zero-shot accuracy (base winner):     {acc_before:.4f}")
print(f"new-32 accuracy after fine-tune:             {acc_after:.4f}")

# ==========================================================================
# 1) fig_eye_phone_detect.png — one open/closed pair, with face-detect bbox
#    (pick the first open and the first closed from the new 32)
# ==========================================================================
first_open  = next(p for p in pairs if p[1] == 1)
first_closed = next(p for p in pairs if p[1] == 0)

fig, axes = plt.subplots(1, 2, figsize=(8, 5))
for ax, p, lbl in [(axes[0], first_open, "open"), (axes[1], first_closed, "closed")]:
    n, y, x, rot, bb = p
    view = rot.copy()
    if bb is not None:
        bx,by,bw,bh = bb
        cv2.rectangle(view, (bx,by), (bx+bw, by+bh), (0,200,0), 8)
    ax.imshow(view)
    ax.set_title(f"{lbl}  ({n})", fontsize=9)
    ax.axis("off")
plt.suptitle("S24-Ultra fine-tune capture pair (after orientation correction + Haar face detect)",
             y=0.97, fontsize=11)
plt.tight_layout()
plt.savefig(FIG / "fig_eye_phone_detect.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"wrote {FIG/'fig_eye_phone_detect.png'}")

# ==========================================================================
# 2) fig_eye_app_finetune.png — per-sample P(open) before vs after for new 32
# ==========================================================================
fig, ax = plt.subplots(figsize=(14, 4.5))
n = len(pairs)
xs = np.arange(n)
width = 0.4
p_open_before = p_before[:, 1]
p_open_after  = p_after [:, 1]

bars1 = ax.bar(xs - width/2, p_open_before, width, label="before fine-tune", color="#aaaaaa")
bars2 = ax.bar(xs + width/2, p_open_after,  width, label="after fine-tune",  color="#4477aa")

# Truth markers above the bar pairs
for i, (name, y_true, *_ ) in enumerate(pairs):
    marker_y = 1.06
    if y_true == 1:
        ax.scatter(i, marker_y, marker='o', s=40, c='#22aa22', edgecolors='black', linewidths=0.5, zorder=5)
    else:
        ax.scatter(i, marker_y, marker='s', s=40, c='#cc2222', edgecolors='black', linewidths=0.5, zorder=5)

ax.axhline(0.5, color='black', linestyle='--', linewidth=0.6, alpha=0.5)
ax.set_ylim(0, 1.15)
ax.set_xlim(-0.7, n - 0.3)
ax.set_xticks(xs)
ax.set_xticklabels([str(i+1) for i in xs], fontsize=7)
ax.set_xlabel("capture index (new S24-Ultra fine-tune set, n=32)")
ax.set_ylabel("P(open)")
ax.set_title("Eye fine-tune: per-sample P(open) before vs after on the 32 newly-captured photos")
ax.legend(loc="upper right", fontsize=9)

# Legend for truth markers
ax.scatter([], [], marker='o', s=40, c='#22aa22', edgecolors='black', linewidths=0.5, label="truth: open")
ax.scatter([], [], marker='s', s=40, c='#cc2222', edgecolors='black', linewidths=0.5, label="truth: closed")
ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig(FIG / "fig_eye_app_finetune.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"wrote {FIG/'fig_eye_app_finetune.png'}")

# ==========================================================================
# Report (so the paper text can match the figure exactly)
# ==========================================================================
out = {
    "new32_zero_shot_acc": acc_before,
    "new32_after_finetune_acc": acc_after,
    "new32_n_open":  int(ynew.sum()),
    "new32_n_closed": int(len(ynew) - ynew.sum()),
}
with open(ART / "eye07_new32_summary.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nReport:", json.dumps(out, indent=2))
