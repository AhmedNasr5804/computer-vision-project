"""
Re-run of notebook 07 + the PTQ-int8 cell of notebook 08, now on the 44-photo
fine-tune set instead of the original 12. Generates:

  artifacts/eye_winner_finetuned.keras      (FP32 keras)
  artifacts/eye_winner_finetuned.tflite     (FP32 TFLite — for sanity)
  artifacts/eye_winner_finetuned_int8.tflite (PTQ-int8 — what ships to the app)

Reports:
  - per-photo before/after probabilities (saved to artifacts/eye07_before_after_44.png)
  - fine-tune accuracy on the 44 photos
  - retained accuracy on the original CEW test split
  - int8 accuracy parity vs FP32
"""

import os, json, time, sys
from pathlib import Path
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import cv2
import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
import random; random.seed(SEED); np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

ROOT = Path("D:/Spring forth year/Computer Vision/project")
ART = ROOT / "artifacts"
FT_DIR = ROOT / "fine_tuning" / "CEW_fine_tuning"
LABELS_JSON = ART / "eye_finetune_labels.json"
WINNER_KERAS = ART / "eye_winner.keras"
SPLIT_JSON = ART / "eye_split.json"

OUT_KERAS = ART / "eye_winner_finetuned.keras"
OUT_FP32_TFLITE = ART / "eye_winner_finetuned.tflite"
OUT_INT8_TFLITE = ART / "eye_winner_finetuned_int8.tflite"

# -------------------------------------------------------------------------
# 1) Load labels and photos
# -------------------------------------------------------------------------
with open(LABELS_JSON) as f:
    lbl_map = json.load(f)
print(f"Loaded {len(lbl_map)} labels — open: {sum(1 for v in lbl_map.values() if v=='open')}, "
      f"closed: {sum(1 for v in lbl_map.values() if v=='closed')}")

FT_FILES = sorted([f for f in FT_DIR.iterdir() if f.suffix.lower()==".jpg"])
print(f"{len(FT_FILES)} jpg files on disk")

# -------------------------------------------------------------------------
# 2) Face crop with auto-orientation (same recipe as notebook 07)
# -------------------------------------------------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def auto_orient_face(rgb):
    best = (0, None, None)
    for k in range(4):
        rot = np.rot90(rgb, k=k)
        gray = cv2.cvtColor(rot, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            continue
        area = max(w*h for x,y,w,h in faces)
        if area > best[0]:
            best = (area, rot, faces)
    return best

def crop_face_100(rot, faces, target=(100,100)):
    if rot is None or len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    pad = int(0.05 * max(w, h))
    x0 = max(0, x-pad); y0 = max(0, y-pad)
    x1 = min(rot.shape[1], x+w+pad); y1 = min(rot.shape[0], y+h+pad)
    crop = rot[y0:y1, x0:x1]
    return cv2.resize(crop, target)

results = []
for fp in FT_FILES:
    img = cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB)
    area, rot, faces = auto_orient_face(img)
    crop = crop_face_100(rot, faces)
    results.append((fp, crop))

n_with_face = sum(1 for _, c in results if c is not None)
print(f"face detected in {n_with_face} / {len(FT_FILES)} captures")

# -------------------------------------------------------------------------
# 3) Load winner, prep model-shape inputs
# -------------------------------------------------------------------------
model = tf.keras.models.load_model(str(WINNER_KERAS))
H, W, C = model.input_shape[1:]
print(f"winner: input shape = ({H}, {W}, {C})")

def prep_for_model(crop_100):
    img = cv2.resize(crop_100, (W, H))
    if C == 1:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        return (gray[..., None].astype(np.float32) / 255.0)
    else:
        return img.astype(np.float32)

pairs = []  # (name, x, y_label_int)
for fp, c in results:
    if c is None: continue
    lbl = lbl_map.get(fp.name)
    if lbl not in ("open", "closed"): continue
    pairs.append((fp.name, prep_for_model(c), 1 if lbl=="open" else 0))

print(f"usable pairs: {len(pairs)}")
Xft = np.stack([p[1] for p in pairs])
yft = np.array([p[2] for p in pairs])
print(f"  X={Xft.shape}, y={yft.shape}, class balance: open={int(yft.sum())}, closed={int(len(yft)-yft.sum())}")

# -------------------------------------------------------------------------
# 4) Fine-tune
# -------------------------------------------------------------------------
probs_before = model.predict(Xft, verbose=0)
acc_before = float((probs_before.argmax(axis=1) == yft).mean())
print(f"accuracy on 44 photos BEFORE fine-tune (using base winner): {acc_before:.4f}")

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
t0 = time.time()
h = model.fit(Xft, yft, epochs=20, batch_size=4, verbose=0)
secs = time.time() - t0
probs_after = model.predict(Xft, verbose=0)
acc_after = float((probs_after.argmax(axis=1) == yft).mean())
print(f"fine-tuned on {len(pairs)} captures in {secs:.1f}s — final train_acc={h.history['accuracy'][-1]:.4f}")
print(f"accuracy on 44 photos AFTER fine-tune: {acc_after:.4f}")

model.save(str(OUT_KERAS))
print(f"Saved: {OUT_KERAS.name}")

# -------------------------------------------------------------------------
# 5) Retained accuracy on the canonical CEW test split
# -------------------------------------------------------------------------
with open(SPLIT_JSON) as f: SPLIT = json.load(f)
Xte = []; yte = []
for r in SPLIT["test"]:
    p = ROOT / r["path"]
    if not p.exists():
        # try absolute
        p = Path(r["path"])
        if not p.exists(): continue
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (W, H))
    img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)
    if C == 1:
        Xte.append(img[..., None].astype(np.float32) / 255.0)
    else:
        Xte.append(np.stack([img,img,img], axis=-1).astype(np.float32))
    yte.append(r["label"])
Xte = np.stack(Xte); yte = np.array(yte)
cew_acc = float((model.predict(Xte, verbose=0).argmax(axis=1) == yte).mean())
print(f"CEW test-set accuracy after fine-tune (domain stability): {cew_acc:.4f}")

# -------------------------------------------------------------------------
# 6) Export FP32 TFLite (sanity baseline)
# -------------------------------------------------------------------------
conv = tf.lite.TFLiteConverter.from_keras_model(model)
fp32_bytes = conv.convert()
OUT_FP32_TFLITE.write_bytes(fp32_bytes)
print(f"Saved {OUT_FP32_TFLITE.name}: {OUT_FP32_TFLITE.stat().st_size/1024:.1f} KB")

# -------------------------------------------------------------------------
# 7) PTQ-int8: representative dataset = 200 CEW train+val images
# -------------------------------------------------------------------------
rep_recs = (SPLIT["train"] + SPLIT["val"])[:200]
REP = []
for r in rep_recs:
    p = ROOT / r["path"]
    if not p.exists(): continue
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (W, H))
    img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)
    if C == 1:
        REP.append(img[..., None].astype(np.float32) / 255.0)
    else:
        REP.append(np.stack([img,img,img], axis=-1).astype(np.float32))
REP = np.stack(REP)
print(f"rep dataset: {REP.shape}")

def rep_gen():
    for i in range(len(REP)):
        yield [REP[i:i+1].astype(np.float32)]

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_gen
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
int8_bytes = conv.convert()
OUT_INT8_TFLITE.write_bytes(int8_bytes)
print(f"Saved {OUT_INT8_TFLITE.name}: {OUT_INT8_TFLITE.stat().st_size/1024:.1f} KB")

# -------------------------------------------------------------------------
# 8) Verify PTQ-int8 parity on CEW test split + 44 photos
# -------------------------------------------------------------------------
def bench(tfl_path, X, y):
    interp = tf.lite.Interpreter(model_path=str(tfl_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    correct = 0
    for x, label in zip(X, y):
        xq = x[None]
        if inp["dtype"] == np.int8:
            xq = (xq / max(in_scale, 1e-9) + in_zp).round().clip(-128, 127).astype(np.int8)
        else:
            xq = xq.astype(np.float32)
        interp.set_tensor(inp["index"], xq)
        interp.invoke()
        out_t = interp.get_tensor(out["index"])
        if out_t.dtype == np.int8:
            out_t = (out_t.astype(np.float32) - out_zp) * out_scale
        if out_t.argmax() == label:
            correct += 1
    return correct / len(X)

acc_int8_cew = bench(OUT_INT8_TFLITE, Xte, yte)
acc_int8_ft  = bench(OUT_INT8_TFLITE, Xft, yft)
print(f"PTQ-int8 — CEW test accuracy: {acc_int8_cew:.4f}")
print(f"PTQ-int8 — 44-photo accuracy: {acc_int8_ft:.4f}")

# -------------------------------------------------------------------------
# 9) Save report
# -------------------------------------------------------------------------
report = {
    "n_finetune_samples": len(pairs),
    "n_open": int(yft.sum()),
    "n_closed": int(len(yft) - yft.sum()),
    "train_seconds": secs,
    "final_train_acc": float(h.history["accuracy"][-1]),
    "ft_acc_before_finetune_fp32": acc_before,
    "ft_acc_after_finetune_fp32": acc_after,
    "cew_test_acc_after_finetune_fp32": cew_acc,
    "ft_acc_after_finetune_int8": acc_int8_ft,
    "cew_test_acc_after_finetune_int8": acc_int8_cew,
    "fp32_tflite_size_kb": OUT_FP32_TFLITE.stat().st_size/1024,
    "int8_tflite_size_kb": OUT_INT8_TFLITE.stat().st_size/1024,
}
with open(ART / "eye07_finetune_results.json", "w") as f:
    json.dump(report, f, indent=2)
print("\n=== REPORT ===")
print(json.dumps(report, indent=2))
