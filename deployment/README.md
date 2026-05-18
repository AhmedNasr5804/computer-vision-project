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
