"""
Phase 2 — Regenerate the binary lane masks classically.

The SegFormer-Cityscapes masks fail because the Pi rig is dark tape on
light fabric, not a real road. The dark/light contrast is ideal for
Otsu thresholding. Recipe:

  1. cv2.imread(..., IMREAD_GRAYSCALE)
  2. Gaussian blur, k=5  (anti-alias the tape edges)
  3. Otsu threshold + INV  (tape is darker than fabric -> invert)
  4. Morphological opening 3x3 x1  (kill salt noise)
  5. Connected-component filter: drop blobs < 200 px

Output:
  - fine_tuning/lane_fine_tuning/segmentation_dataset/masks_binary/*.png (10500 files)
  - paper/figures/fig_lane_masks_v2.png  (4-row x 4-col: image | old mask | new mask | overlay)
  - artifacts/lane_phase2_summary.json
"""
import sys, io, json, time, random
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import cv2
import matplotlib.pyplot as plt

random.seed(42)

ROOT = Path("D:/Spring forth year/Computer Vision/project")
DS = ROOT / "fine_tuning" / "lane_fine_tuning" / "segmentation_dataset"
IM = DS / "images"
OLD_M = DS / "masks"
NEW_M = DS / "masks_binary"
NEW_M.mkdir(exist_ok=True)
FIG = ROOT / "paper" / "figures"
ART = ROOT / "artifacts"

def gen_mask(img_gray, min_blob_px=200):
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    # Connected component filter
    num, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    cleaned = np.zeros_like(th)
    for k in range(1, num):  # skip background label 0
        if stats[k, cv2.CC_STAT_AREA] >= min_blob_px:
            cleaned[labels == k] = 255
    return cleaned

# Pass 1: process all 10,500
images = sorted(IM.glob("*.jpg"))
print(f"processing {len(images)} images")
t0 = time.time()
old_nonzero, new_nonzero = [], []
for i, ip in enumerate(images):
    g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
    new = gen_mask(g)
    out = NEW_M / (ip.stem + ".png")
    cv2.imwrite(str(out), new)
    # sample stats
    if i % 250 == 0:
        old = cv2.imread(str(OLD_M / (ip.stem + ".png")), cv2.IMREAD_GRAYSCALE)
        old_nz = 100 * np.count_nonzero(old > 127) / old.size
        new_nz = 100 * np.count_nonzero(new > 127) / new.size
        old_nonzero.append(old_nz); new_nonzero.append(new_nz)
        if i % 1000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(1, i + 1) * (len(images) - i)
            print(f"  [{i}/{len(images)}] {ip.name} old={old_nz:.2f}% new={new_nz:.2f}% "
                  f"(elapsed={elapsed:.0f}s eta={eta:.0f}s)")

t_total = time.time() - t0
print(f"finished in {t_total:.1f}s ({t_total/len(images)*1000:.1f} ms/img)")
print(f"old nonzero%: min={min(old_nonzero):.2f}, median={np.median(old_nonzero):.2f}, "
      f"max={max(old_nonzero):.2f}")
print(f"new nonzero%: min={min(new_nonzero):.2f}, median={np.median(new_nonzero):.2f}, "
      f"max={max(new_nonzero):.2f}")

# Pass 2: build the comparison figure
# Pick 4 images, one per direction + one extra
random.seed(42)
sample_imgs = []
for prefix in ["left_left_01", "right_right_01", "straight_straight_02"]:
    cands = [p for p in images if p.name.startswith(prefix) and "dup" not in p.name]
    sample_imgs.append(random.choice(cands))
# 4th: a randomly chosen augmented frame
augs = [p for p in images if "dup" in p.name]
sample_imgs.append(random.choice(augs))

fig, axes = plt.subplots(len(sample_imgs), 4, figsize=(14, 3.5 * len(sample_imgs)))
for r, ip in enumerate(sample_imgs):
    img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
    old = cv2.imread(str(OLD_M / (ip.stem + ".png")), cv2.IMREAD_GRAYSCALE)
    new = cv2.imread(str(NEW_M / (ip.stem + ".png")), cv2.IMREAD_GRAYSCALE)
    # Overlay: green for tape pixels
    overlay = img.copy()
    overlay[new > 127] = [60, 220, 60]
    blend = (0.55 * img + 0.45 * overlay).astype(np.uint8)
    axes[r, 0].imshow(img); axes[r, 0].set_title(f"image: {ip.stem[:30]}", fontsize=8); axes[r, 0].axis("off")
    axes[r, 1].imshow(old, cmap="gray")
    axes[r, 1].set_title(f"SegFormer mask\n({100*np.count_nonzero(old>127)/old.size:.2f}% lane)", fontsize=8); axes[r, 1].axis("off")
    axes[r, 2].imshow(new, cmap="gray")
    axes[r, 2].set_title(f"new Otsu mask\n({100*np.count_nonzero(new>127)/new.size:.2f}% lane)", fontsize=8); axes[r, 2].axis("off")
    axes[r, 3].imshow(blend); axes[r, 3].set_title("overlay (new mask)", fontsize=8); axes[r, 3].axis("off")
plt.suptitle("Pi lane masks: SegFormer (broken on this domain) vs classical Otsu+morphology",
             y=1.0, fontsize=11)
plt.tight_layout()
out_png = FIG / "fig_lane_masks_v2.png"
plt.savefig(out_png, dpi=140, bbox_inches="tight")
plt.close()
print(f"wrote {out_png}")

# Summary
summary = {
    "n_images_processed": len(images),
    "ms_per_image": t_total / len(images) * 1000.0,
    "old_mask_nonzero_pct": {
        "min": float(min(old_nonzero)),
        "median": float(np.median(old_nonzero)),
        "max": float(max(old_nonzero)),
    },
    "new_mask_nonzero_pct": {
        "min": float(min(new_nonzero)),
        "median": float(np.median(new_nonzero)),
        "max": float(max(new_nonzero)),
    },
    "recipe": "grayscale -> GaussianBlur(5) -> Otsu+INV -> open(3x3,1) -> drop blobs<200px",
}
with open(ART / "lane_phase2_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
