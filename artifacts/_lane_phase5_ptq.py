"""
Phase 5 — PTQ-int8 after Phase 4 training.

Doctor's note: "we should train it before the optimization to preserve the
model features then optimize." So FP32 first (Phase 4), int8 here.

Recipe (identical to the eye notebook 08 recipe that succeeded):
  - representative dataset: 200 train-split images (random sample)
  - tf.lite.Optimize.DEFAULT
  - target spec: TFLITE_BUILTINS_INT8
  - inference_input_type / output_type: int8

Verification:
  - Re-evaluate IoU + Dice on the full test set
  - Pass if int8 IoU is within 2 pp of FP32 IoU
"""
import sys, io, json, random, re
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import cv2
import tensorflow as tf

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)

ROOT = Path("D:/Spring forth year/Computer Vision/project")
DS = ROOT / "fine_tuning" / "lane_fine_tuning" / "segmentation_dataset"
IM = DS / "images"; MK = DS / "masks_binary"
ART = ROOT / "artifacts"

MODEL_KERAS = ART / "lane_truncated.keras"
OUT_INT8 = ART / "lane_truncated_int8.tflite"
OUT_FP32 = ART / "lane_truncated.tflite"

H = W = 128

# Reconstruct the same split deterministically
def base_stem(stem): return re.sub(r"_dup\d+$", "", stem)
all_paths = sorted(IM.glob("*.jpg"))
by_dir = {"left_left_01": [], "right_right_01": [], "straight_straight_02": []}
for p in all_paths:
    pre = base_stem(p.stem).split("_f")[0]
    by_dir[pre].append(p)

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
test_paths  = [p for d in by_dir.values() for p in to_split(d, test_stems)]
print(f"reconstructed split: train={len(train_paths)} test={len(test_paths)}")

def load_pair(ip):
    img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    mask = cv2.imread(str(MK / (Path(ip).stem + ".png")), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return img.astype(np.float32) / 255.0, (mask > 127).astype(np.float32)

# Representative dataset: 200 random training images
rep_paths = random.sample(train_paths, 200)
REP = np.stack([load_pair(p)[0] for p in rep_paths])
print(f"rep dataset: {REP.shape}")

# Load the FP32 keras model
model = tf.keras.models.load_model(str(MODEL_KERAS), compile=False)
print(f"loaded {MODEL_KERAS.name} — {model.count_params()} params")

# Export FP32 TFLite (sanity baseline)
conv = tf.lite.TFLiteConverter.from_keras_model(model)
fp32_bytes = conv.convert()
OUT_FP32.write_bytes(fp32_bytes)
print(f"FP32: {OUT_FP32.name} {OUT_FP32.stat().st_size/1024:.1f} KB")

# Export PTQ-int8 TFLite
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
OUT_INT8.write_bytes(int8_bytes)
print(f"INT8: {OUT_INT8.name} {OUT_INT8.stat().st_size/1024:.1f} KB "
      f"(compression vs FP32 = {OUT_FP32.stat().st_size/OUT_INT8.stat().st_size:.2f}x)")

# Evaluate both on the test split
def evaluate_tflite(tfl_path):
    interp = tf.lite.Interpreter(model_path=str(tfl_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]; out_scale, out_zp = out["quantization"]
    ious = []; dices = []
    for p in test_paths:
        img, gt = load_pair(p)
        x = img[None]
        if inp["dtype"] == np.int8:
            x = (x / max(in_scale, 1e-9) + in_zp).round().clip(-128, 127).astype(np.int8)
        else:
            x = x.astype(np.float32)
        interp.set_tensor(inp["index"], x); interp.invoke()
        y = interp.get_tensor(out["index"])[0, ..., 0]
        if out["dtype"] == np.int8:
            y = (y.astype(np.float32) - out_zp) * out_scale
        yp = (y > 0.5).astype(np.float32); yt = gt.astype(np.float32)
        inter = float((yt * yp).sum()); s = float(yt.sum() + yp.sum())
        union = s - inter
        ious.append((inter + 1e-7) / (union + 1e-7))
        dices.append((2 * inter + 1e-7) / (s + 1e-7))
    return float(np.mean(ious)), float(np.mean(dices))

print("\nEvaluating on test split...")
fp32_iou, fp32_dice = evaluate_tflite(OUT_FP32)
int8_iou, int8_dice = evaluate_tflite(OUT_INT8)
print(f"  FP32 TFLite: IoU={fp32_iou:.4f}, Dice={fp32_dice:.4f}")
print(f"  PTQ-int8   : IoU={int8_iou:.4f}, Dice={int8_dice:.4f}")
print(f"  delta IoU  : {(int8_iou - fp32_iou)*100:+.2f} pp")

summary = {
    "fp32_size_kb": OUT_FP32.stat().st_size / 1024,
    "int8_size_kb": OUT_INT8.stat().st_size / 1024,
    "compression_x": OUT_FP32.stat().st_size / OUT_INT8.stat().st_size,
    "fp32_test_iou": fp32_iou, "fp32_test_dice": fp32_dice,
    "int8_test_iou": int8_iou, "int8_test_dice": int8_dice,
    "iou_delta_pp": (int8_iou - fp32_iou) * 100.0,
    "gate_passed": abs(int8_iou - fp32_iou) <= 0.02,
}
with open(ART / "lane_phase5_results.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
