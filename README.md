# Edge-Deployable Driver Assistance Vision Project

[![Status](https://img.shields.io/badge/status-complete-success?style=flat-square)](./README.md)
[![Platform](https://img.shields.io/badge/platform-Android%2015%20%7C%20Raspberry%20Pi%204-blue?style=flat-square)](./README.md)
[![Python](https://img.shields.io/badge/python-3.13-blue?style=flat-square)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?style=flat-square)](https://www.tensorflow.org/)
[![TFLite](https://img.shields.io/badge/TFLite-int8%20PTQ-orange?style=flat-square)](https://ai.google.dev/edge/litert)
[![Kotlin](https://img.shields.io/badge/Kotlin-2.0.20-7F52FF?style=flat-square&logo=kotlin&logoColor=white)](https://kotlinlang.org/)
[![Android%20SDK](https://img.shields.io/badge/Android%20SDK-35-3DDC84?style=flat-square&logo=android&logoColor=white)](https://developer.android.com/)
[![Paper](https://img.shields.io/badge/report-IEEE%20paper-lightgrey?style=flat-square&logo=latex)](paper/paper.tex)
[![License](https://img.shields.io/badge/license-Academic-yellow?style=flat-square)](./README.md#license-and-academic-context)

> A complete CIE-552 Computer Vision term project implementing, comparing, and **deploying** two
> edge-AI pipelines:
>
> - **Eye-state classification** (open vs. closed) — deployed as a live Kotlin Android app on a real **Samsung Galaxy S24 Ultra**, running at **0.126 ms / inference** (≈7,900 fps theoretical) with PTQ-int8 on the CPU's XNNPACK backend.
> - **Lane-direction classification** (left / right / straight / stop) — targeted for **Raspberry Pi 4 Model B**, projected at **~3.1 ms / inference** (~320 fps).

The repo is organized as a fully reproducible chain of notebooks, JSON manifests, and code, going from data exploration all the way through to a working APK installed on the demonstration phone.

## Table of Contents

1. [Highlights](#highlights)
2. [Live Android Demo](#live-android-demo)
3. [High-Level Results](#high-level-results)
4. [Repository Layout](#repository-layout)
5. [Experiment Workflow](#experiment-workflow)
6. [Quick Start — Notebooks](#quick-start--notebooks)
7. [Quick Start — Android App](#quick-start--android-app)
8. [Build the Paper](#build-the-paper)
9. [Artifacts and Traceability](#artifacts-and-traceability)
10. [Key Findings Worth Reading](#key-findings-worth-reading)
11. [Reproducibility Notes](#reproducibility-notes)
12. [License and Academic Context](#license-and-academic-context)

## Highlights

| | |
|---|---|
| 📱 **Live mobile app** | Kotlin + CameraX + ML Kit face detection + TFLite, full-screen selfie demo with confidence bars and bounding box |
| 🔬 **End-to-end pipeline** | Classical CV → custom CNN → transfer learning → controlled experiments → model selection → quantized export → on-device benchmark |
| ⚡ **Measured on real hardware** | Eye model benchmarked on a physical Samsung Galaxy S24 Ultra (SM-S928B) across CPU/GPU/NNAPI delegates; lane numbers calibrated against published RPi-4 benchmarks |
| 📈 **Honest deployment findings** | Documented the 0.907 → 0.583 zero-shot accuracy drop when moving from CEW to phone captures, and a 12-photo personal fine-tune that closes it without catastrophic forgetting |
| 📄 **IEEE paper** | 9-page paper compiles from `paper/paper.tex` with 23 figures auto-generated from the JSON manifests |

## Live Android Demo

The `android_app/` folder is a complete Android Studio project. Build it with the included Gradle wrapper and `adb install -r` the resulting APK; the app uses the **front-facing camera**, runs the int8 winner model in real time, and renders **OPEN / CLOSED** + confidence bars + a green face-box overlay.

**Pipeline:**

```
Front camera ─► CameraX ImageAnalysis ─► ML Kit face detect (FAST) ─►
   Square crop +30 % pad −10 % y-shift  ─►  BT.601 grey + 8×8 CLAHE (2-pass clip) ─►
       TFLite lw_wide (64×64 FP32)  ─►  EMA α=0.25 + Schmitt hysteresis ─►  UI label
```

The end-to-end latency on the S24 Ultra is roughly **1 ms** per frame, dominated by ML Kit's face detection — the classifier itself takes 0.13 ms. The app ships the *fine-tuned* model (per-subject accuracy 100 % on the 12 phone captures, CEW test accuracy 86.5 %), not the raw CEW-only winner.

See [`android_app/README.md`](android_app/README.md) for full build / install instructions and the file-by-file source tour.

## High-Level Results

### Eye pipeline (Samsung S24 Ultra target)

| Stage | Accuracy | F1 | Params | Notes |
|---|---|---|---|---|
| Classical HOG + SVM | 0.874 | 0.876 | 14 k | CLAHE + HOG features |
| Custom CNN (from scratch, no pretrained) | 0.896 | 0.898 | 25.8 k | ~0.6 MB |
| MobileNetV2-finetune (best TL candidate) | 0.885 | 0.888 | 2.26 M | |
| **`lw_wide` — model-selection winner** | **0.907** | **0.910** | **97.5 k** | 106 KB int8 TFLite |
| Same model deployed on real S24 Ultra CPU (PTQ-int8 + XNNPACK, 4 threads) | — | — | — | **0.126 ms / inference** |
| Personal-fine-tuned model in the shipping APK | 1.000 (n=12 phone) / 0.865 (CEW test) | — | 97.5 k | 383 KB FP32 TFLite |

### Lane pipeline (Raspberry Pi 4 target)

| Stage | Accuracy (4-class) | F1 | Params |
|---|---|---|---|
| Classical Canny + Hough + SVM | 0.505 | 0.575 | — |
| Custom CNN (from scratch) | 0.702 | 0.659 | 86 k |
| MobileNetV2-finetune (best TL) | 0.776 | 0.779 | 711 k |
| **`lw_baseline` — model-selection winner** | **0.744** | **0.775** | **86 k** |
| After 10-epoch fine-tune on the Raspberry-Pi camera split | 1.000 (on Pi test split — caveats in paper) | — | 86 k |
| Projected on RPi 4 (PTQ-int8 + XNNPACK, 4 threads) | — | — | — / **~3.1 ms** |

Exact numbers live in `artifacts/*.json` — see `artifacts/DESCRIPTION.md`.

## Repository Layout

```
.
├── README.md                         (this file)
├── android_app/                      Kotlin Android Studio project — the live demo
│   ├── README.md                     build / install instructions
│   ├── app/src/main/                 Kotlin sources, manifest, assets/eye_winner.tflite
│   └── gradle/, gradlew.bat, ...     Gradle wrapper
├── notebooks/                        Staged reproducible pipelines
│   ├── 00_setup_environment.ipynb
│   ├── eye/   01..08                 Eye pipeline (data → classical → CNN → TL → exp → select → finetune → deploy)
│   ├── lane/  01..08                 Lane pipeline (same staging)
│   └── 09_paper_figures.ipynb        Aggregates every figure used in the paper
├── artifacts/                        117 files: trained models, JSON manifests, figures, deployment outputs
│   └── DESCRIPTION.md                file-by-file map
├── datasets/                         CEW + TuSimple sources (not committed; see datasets/DESCRIPTION.md)
├── fine_tuning/                      S24 captures + Pi-camera frames (not committed)
├── paper/                            IEEE paper
│   ├── paper.tex                     1019 lines, 9 pages, 23 unique figures
│   ├── figures/                      33 PNGs, all auto-generated from JSON
│   ├── summary.md                    plain-text numerical roll-up
│   ├── table1_eye.csv, table1_lane.csv
│   └── DESCRIPTION.md
└── tools/                            Android benchmark APK and helper binaries
```

Every directory has its own `DESCRIPTION.md` documenting role, file naming, and re-generation commands.

## Experiment Workflow

### 1. Environment setup

Open `notebooks/00_setup_environment.ipynb` once. It pins seeds, prints library versions, and visualises one sample per source folder so any path / encoding / cropping mistake is caught before any training begins.

### 2. Eye pipeline (in order)

| # | Notebook | Output |
|---|---|---|
| 01 | `eye/01_eye_data_exploration.ipynb` | `eye_split.json`, class-balance plots, per-channel histograms |
| 02 | `eye/02_eye_classical_baseline.ipynb` | HOG+SVM model, ROC, confusion |
| 03 | `eye/03_eye_cnn_baseline.ipynb` | `eye03_cnn_baseline.keras` + training curves |
| 04 | `eye/04_eye_transfer_learning.ipynb` | 4 TL candidates (MobileNetV2/V3-Small × frozen/finetune) |
| 05 | `eye/05_eye_improved_experiments.ipynb` | Exp-A aug, Exp-B color, Exp-C robustness, Exp-D lightweight |
| 06 | `eye/06_eye_model_selection.ipynb` | Pareto frontier, weighted score, **`eye_winner.{keras,tflite}`** |
| 07 | `eye/07_eye_finetune.ipynb` | Personal domain-adaptation demo with the 12 phone captures |
| 08 | `eye/08_eye_mobile_deployment.ipynb` | PTQ-int8, QAT, S24 Ultra paste-back benchmarks |

### 3. Lane pipeline (in order)

Mirrors the eye pipeline. The two notable differences:

- `lane/02_lane_classical_baseline.ipynb` uses **Canny + Hough lines + per-side feature statistics** (vs HOG for the eye pipeline).
- `lane/08_lane_rpi_deployment.ipynb` reports **simulated** RPi 4 numbers calibrated against published benchmarks (DeepEdgeBench, PyTorch RPi tutorial, Frigate OpenVINO discussion) — to be replaced by measured numbers when the hardware is available.

### 4. Paper figures

`notebooks/09_paper_figures.ipynb` reads every JSON manifest in `artifacts/` and regenerates every PNG referenced in `paper.tex`. Re-run after any experiment change and the paper auto-updates.

## Quick Start — Notebooks

### Option A: Interactive (Jupyter / VS Code)

1. Open `notebooks/00_setup_environment.ipynb`, run all cells.
2. Run eye notebooks 01 → 08 in order.
3. Run lane notebooks 01 → 08 in order.
4. Run `notebooks/09_paper_figures.ipynb`.

### Option B: Headless (PowerShell)

```powershell
# 0) Setup
jupyter nbconvert --to notebook --execute --inplace notebooks/00_setup_environment.ipynb

# 1) Eye pipeline
1..8 | ForEach-Object {
    $name = Get-ChildItem notebooks/eye/$($_.ToString('00'))_*.ipynb
    jupyter nbconvert --to notebook --execute --inplace $name.FullName
}

# 2) Lane pipeline
1..8 | ForEach-Object {
    $name = Get-ChildItem notebooks/lane/$($_.ToString('00'))_*.ipynb
    jupyter nbconvert --to notebook --execute --inplace $name.FullName
}

# 3) Paper figures
jupyter nbconvert --to notebook --execute --inplace notebooks/09_paper_figures.ipynb
```

## Quick Start — Android App

You need **Android Studio JBR (JDK 21)** + **Android SDK 35** + **adb**. With those installed and an S24 Ultra (or any Android 11+ device) connected via USB-debug:

```powershell
Set-Location android_app
.\gradlew.bat assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell pm grant com.cv552.eyedemo android.permission.CAMERA
adb shell am start -W -n com.cv552.eyedemo/.MainActivity
```

First build downloads Gradle 8.10.2, AGP 8.7, CameraX, ML Kit, TFLite (~120 MB total) — takes ~15 minutes; subsequent builds are ~30 seconds. Full instructions: [`android_app/README.md`](android_app/README.md).

## Build the Paper

```powershell
Set-Location paper
pdflatex paper.tex
pdflatex paper.tex
```

(Two passes so cross-references settle; no `bibtex` step — the bibliography is inline.)

> ⚠ **Windows**: if `pdflatex` complains *"I can't write on file `paper.pdf`"*, close any PDF viewer (Adobe / Edge / your IDE preview / VS Code) that has `paper.pdf` open. Sumatra is the only common Windows viewer that doesn't take a write-lock.

## Artifacts and Traceability

`artifacts/` is the single source of truth for:

- **Splits**: `eye_split.json`, `lane_split.json` — exact JSON dump of which file is in train/val/test
- **Metrics**: `eye{02..08}_*.json`, `lane{02..08}_*.json` — every numerical claim in the paper traces to one of these
- **Models**: `*.keras` (trainable), `*.tflite` (deployment), `*.xml/.bin` (OpenVINO IR for the lane pipeline)
- **Manifests**: `eye08_deployment.json`, `lane08_deployment.json` — full latency / size / accuracy variants per device
- **Figures**: every PNG referenced from `paper/figures/` is derived from these JSONs

See `artifacts/DESCRIPTION.md` for the per-file map.

## Key Findings Worth Reading

The full discussion is in `paper/paper.tex` (Section "Discussion"); the headline findings that this codebase establishes are:

1. **CEW does not generalize zero-shot to a contemporary smartphone selfie**: the canonical winner drops from 0.907 (CEW test) to 0.583 (12 S24-Ultra captures of six different subjects). The failure cases are systematic — oblique gaze, beards, headphones, hair visible — all under-represented in CEW.
2. **A 12-photo, 3-minute personal fine-tune fixes it**: 240 augmented samples, 23 epochs, Adam lr 2e-4 → 1.000 per-subject accuracy + 0.865 retained on the CEW test split. No architectural change, no extra data labels.
3. **NNAPI is a placebo on Samsung OneUI**: the runtime accepts `--use_nnapi=true`, constructs the delegate, then silently falls back to XNNPACK. Reaching the Hexagon NPU on a Snapdragon 8 Gen 3 phone running OneUI requires the QNN delegate, not NNAPI.
4. **For tiny models, GPU is slower than CPU**: 1.100 ms (Adreno) vs 0.126 ms (4-thread XNNPACK) — OpenCL launch overhead dominates at this model size.
5. **Class-weighted multi-class training can collapse with short patience**: we observed and document a `0.5334` majority-class fallback that affected 6 of 9 lane Exp-B/Exp-D candidates until we extended EarlyStopping patience to 12 and added a 5e-4 warmup LR.
6. **A naive Kotlin CLAHE re-implementation is not faithful to OpenCV**: a single-pass clip+redistribute gives 44-pixel mean error against `cv2.createCLAHE` and is enough to bias the deployed model toward CLOSED at >95 %. A two-pass clip + bilinear inter-tile interpolation closes the gap.

## Reproducibility Notes

- **Seed = 42** everywhere. Pinned in `notebooks/00_setup_environment.ipynb`. Every other notebook re-applies it in cell 1.
- **Pinned library versions**: TensorFlow 2.20, PyTorch 2.6.0+cu124, OpenCV 4.13, scikit-learn 1.7.2, albumentations 2.0.8, scikit-image 0.26, MiKTeX 24.1 for the paper. See `notebooks/00_setup_environment.ipynb` for the exact `pip list` output captured at session start.
- **Hardware used during development**: laptop (Intel i7 + RTX 3060 + Windows 11) + Samsung Galaxy S24 Ultra (SM-S928B, OneUI on Android 15) over `adb`. Lane numbers projected from x86 against published RPi-4 benchmarks until a physical Pi is available.
- **TensorFlow GPU caveat**: TF ≥ 2.11 on native Windows is **CPU-only**. PyTorch retains CUDA. We use PyTorch for SegFormer mask generation (GPU) and TensorFlow for everything else (CPU).

## License and Academic Context

This repository is the submission for **CIE-552 Computer Vision — Term Project**, Zewail City of Science, Technology and Innovation. The CEW and TuSimple datasets are subject to their original licenses; please consult them before redistribution. All other content (notebook code, Kotlin sources, paper text, figures generated from our pipelines) is provided for academic / educational use.

**Authors**: Ahmed Mohamed Elsaid (202200294), Abdelhady Mohamed (202201172).
