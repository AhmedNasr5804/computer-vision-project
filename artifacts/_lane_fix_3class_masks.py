"""
Regenerate masks_3class/*.png with viewable values 0/127/255 (instead of
the original 0/1/2) so they can be inspected with any image viewer.

The training pipeline maps these back to label indices on read via a LUT,
so accuracy is unchanged.
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np
import cv2

DS = Path("D:/Spring forth year/Computer Vision/project/fine_tuning/"
          "lane_fine_tuning/segmentation_dataset")
SRC = DS / "masks_3class"
files = sorted(SRC.glob("*.png"))
print(f"updating {len(files)} 3-class masks: 0/1/2 -> 0/127/255")

# LUT: index -> viewable value
WRITE_LUT = np.array([0, 127, 255], dtype=np.uint8)

t0 = time.time()
n_updated = 0
for i, fp in enumerate(files):
    m = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
    u = np.unique(m)
    if set(u.tolist()).issubset({0, 127, 255}):
        continue  # already viewable
    if not set(u.tolist()).issubset({0, 1, 2}):
        print(f"  unexpected values in {fp.name}: {u}")
        continue
    m_view = WRITE_LUT[m]
    cv2.imwrite(str(fp), m_view)
    n_updated += 1
    if (i + 1) % 2000 == 0:
        print(f"  [{i+1}/{len(files)}] (updated so far: {n_updated})")

print(f"done in {time.time()-t0:.1f}s; updated {n_updated} files")

# Quick verification
sample = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
print(f"verify {files[0].name}: unique={np.unique(sample)}")
