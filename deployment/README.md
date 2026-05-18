# Deployment artifacts — CIE-552 Computer Vision Term Project

Ready-to-ship int8 models for both pipelines, with usage guides and a paste-able Claude Code prompt for the Raspberry Pi side.

| Pipeline | Target | Model file | Size | Reference |
|---|---|---|---|---|
| **Eye** (open / closed) | Samsung S24 Ultra (Android) — also runs on any tflite-runtime host | `eye/eye_winner.tflite` | 107 KB | `paper/paper.tex` §4.7 |
| **Lane** (bg / lane-left / lane-right) | Raspberry Pi 4B + Pi Camera v2 | `lane/lane_truncated_3class.tflite` | 27.0 KB | `paper/paper.tex` §4.6, §4.7 |
| **Lane** (binary lane mask) | Raspberry Pi 4B + Pi Camera v2 (simpler downstream) | `lane/lane_truncated_binary.tflite` | 26.7 KB | `paper/paper.tex` §4.6 |

All three were post-training int8 quantized with a 200-image representative dataset (see `artifacts/_lane_phase5_ptq.py` / `artifacts/_finetune_44.py`). Per-model accuracy deltas vs the FP32 source are sub-pp:

| Model | FP32 metric | int8 metric | Δ |
|---|---|---|---|
| Eye `lw_wide` fine-tuned | 76.4% CEW + 100% on 44 photos | 76.4% CEW + 100% on 44 photos | 0.00 pp |
| Lane truncated U-Net (binary) | 0.9969 IoU | 0.9969 IoU | −0.003 pp |
| Lane truncated U-Net (3-class) | mean IoU 0.9931 | mean IoU 0.9931 | 0.00 pp |

## What's where

```
deployment/
├── README.md                     <- this file
├── eye/
│   ├── README.md                 <- model specs + integration notes
│   └── eye_winner.tflite         <- 107 KB int8
└── lane/
    ├── README.md                 <- model specs + paste-able Claude Code prompt for RPi
    ├── lane_truncated_3class.tflite   <- 27.0 KB int8 (primary)
    └── lane_truncated_binary.tflite   <- 26.7 KB int8 (fallback)
```

## Quick start

**On the S24 Ultra (eye)**: open `android_app/` in Android Studio, run on the connected phone. Or `cd android_app && ./gradlew assembleDebug && adb install -r app/build/outputs/apk/debug/app-debug.apk`. The model is already bundled in `app/src/main/assets/eye_winner.tflite`.

**On the Raspberry Pi 4B (lane)**: `cd deployment/lane`, install prerequisites (`tflite-runtime`, `picamera2`, `python3-opencv` — full list in `lane/README.md`), then start `claude code` in that directory and paste the prompt from `lane/README.md`. Claude Code will write `lane_live.py` for you and start the live demo.

## Subscribing to the live eye state from the Pi

The Android app publishes its smoothed eye-state predictions to the Firebase
Realtime Database at `https://lab1-f7c43-default-rtdb.firebaseio.com` under
`/eye_monitor`. Field schema is documented in `android_app/README.md`.

To consume the feed from the Pi (or any Python host), install
`firebase-admin` or use a plain HTTP poll. The HTTP-poll path needs no
credentials and works under the open test rules:

```python
import time, requests
URL = "https://lab1-f7c43-default-rtdb.firebaseio.com/eye_monitor.json"

while True:
    try:
        data = requests.get(URL, timeout=2).json()  # dict or None
        if data:
            print(f"{data['state']:7}  "
                  f"P(open)={data['p_open']:.2f}  "
                  f"fps={data['fps']:.1f}  "
                  f"age_ms={int(time.time()*1000) - data['timestamp']}")
    except Exception as e:
        print(f"poll failed: {e}")
    time.sleep(0.1)   # 10 Hz, matches the publisher
```

For an event-driven subscription (a callback every time `eye_monitor`
changes), use the Firebase REST event-stream endpoint with the `Accept:
text/event-stream` header:

```python
import requests, json
URL = "https://lab1-f7c43-default-rtdb.firebaseio.com/eye_monitor.json"
r = requests.get(URL, headers={"Accept": "text/event-stream"}, stream=True)
for line in r.iter_lines(decode_unicode=True):
    if line.startswith("data:"):
        event = json.loads(line[5:].strip())
        if event["data"] is not None:
            print(event["data"])
```

Either path is what the `lane_live.py` script (built by Claude Code via the
prompt in `deployment/lane/README.md`) can use to react to the driver's eye
state — e.g. drop confidence on the lane prediction when the driver is
flagged CLOSED, or write that state into `car_telemetry/latest/eye_state`
alongside the existing telemetry.

## Reproducing the models from scratch

Every model in this folder is reproducible from the repo. Scripts are in `artifacts/`:

```
Eye:
  artifacts/_finetune_44.py             <- fine-tune + PTQ-int8 (~30 s on CPU)

Lane (binary):
  artifacts/_lane_phase2_regen_masks.py <- regenerate Otsu masks (~3.5 min)
  artifacts/_lane_phase3_4_train.py     <- train truncated U-Net   (~10 min)
  artifacts/_lane_phase5_ptq.py         <- PTQ-int8 conversion     (~30 s)

Lane (3-class):
  artifacts/_lane_phase6_3class.py      <- generate 3-class masks + train + PTQ-int8
                                           (~14 min total)
  artifacts/_lane_fix_3class_masks.py   <- one-shot remap of 0/1/2 -> 0/127/255
                                           so masks become viewable in image viewers
```

Run from the repo root with `python -u artifacts/<script>.py`. Each script writes its outputs to `artifacts/` and the paper figures land in `paper/figures/`.

## Why these models and not the larger candidates

For the eye pipeline the model-selection algorithm picked `lw_wide` (97 k parameters) over MobileNetV2 fine-tune (2.26 M parameters) because the latter is 88× larger and 0.011 *less* accurate on CEW (Table II, `paper/paper.tex`). For the lane pipeline the supervisor's review of the original deployment plan led us to drop the TuSimple-trained model entirely on the Pi side and train a $12{,}627$-parameter shallow segmentation network from scratch on regenerated Pi-domain supervision masks — see §4.6 of `paper/paper.tex` for the full data audit (the original SegFormer masks have median $0.8\%$ nonzero coverage and cannot be overfit even with a $7$ k-parameter U-Net).
