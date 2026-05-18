# Eye-state classifier — deployment artifact

`eye_winner.tflite` is the post-training int8 quantized version of the `lw_wide` lightweight winner from the model-selection notebook (`notebooks/eye/06_eye_model_selection.ipynb`), fine-tuned on 44 S24-Ultra capture photos (Section 4.7 of `paper/paper.tex`).

| Field | Value |
|---|---|
| File | `eye_winner.tflite` |
| Size | 107 KB |
| Parameters | 97,538 |
| Input shape | `(1, 64, 64, 1)` |
| Input dtype | `int8` |
| Input quantization | `scale = 0.003922`, `zero_point = -128` (= float / 255 exactly) |
| Output shape | `(1, 2)` |
| Output dtype | `int8` |
| Output quantization | `scale = 0.003906`, `zero_point = -128` |
| Output channels | `[P(closed), P(open)]` (softmax) |

## Primary deployment: Samsung Galaxy S24 Ultra (Android)

The Kotlin Android application that wraps this model is in `android_app/` at the repo root — including the pipeline (CameraX → ML Kit face detect → Kotlin CLAHE → this model → EMA smoothing + Schmitt-trigger hysteresis → UI) and the Gradle build configuration.

On the S24 Ultra (Snapdragon 8 Gen 3, XNNPACK 4-thread CPU): **0.126 ms / frame** inference (TFLite Benchmark APK, 100 timed runs after 50 warmup).

## Secondary deployment: any tflite-runtime host (laptop, Raspberry Pi, etc.)

If you want to run this model outside the Android app — say on a webcam-attached laptop or a Raspberry Pi for testing — the same Claude Code prompt template from `deployment/lane/README.md` works. Adapt it by changing:

- model path → `eye_winner.tflite`
- input → `64×64×1` grayscale (BT.601 from RGB), with **CLAHE clip=2.0, tileGridSize=(8,8)** applied before quantization (this preprocessing is critical — without CLAHE the model biases strongly toward CLOSED, see `android_app/app/src/main/java/com/cv552/eyedemo/EyeClassifier.kt` for the exact recipe)
- output → 2-vector softmax `[P(closed), P(open)]`
- a face detector ahead of the model (ML Kit on Android; for Python use `mediapipe` or `opencv` Haar cascade)

## How this model was trained

See `paper/paper.tex` Section 4.7 (the "Eye fine-tune" subsection) and `artifacts/_finetune_44.py` for the exact 44-photo fine-tune recipe and PTQ-int8 conversion. The unfine-tuned canonical-CEW winner is also in `artifacts/eye_winner.tflite` (108 KB int8) if you need a non-personally-fine-tuned baseline.
