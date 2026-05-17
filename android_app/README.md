# Eye Demo — Android Live Application

[![Kotlin](https://img.shields.io/badge/Kotlin-2.0.20-7F52FF?style=flat-square&logo=kotlin&logoColor=white)](https://kotlinlang.org/)
[![Min%20SDK](https://img.shields.io/badge/min%20SDK-24-3DDC84?style=flat-square&logo=android&logoColor=white)](https://developer.android.com/)
[![Target%20SDK](https://img.shields.io/badge/target%20SDK-35-3DDC84?style=flat-square&logo=android&logoColor=white)](https://developer.android.com/)
[![AGP](https://img.shields.io/badge/AGP-8.7.0-blue?style=flat-square)](https://developer.android.com/build)
[![Gradle](https://img.shields.io/badge/Gradle-8.10.2-02303A?style=flat-square&logo=gradle&logoColor=white)](https://gradle.org/)
[![TFLite](https://img.shields.io/badge/TFLite-2.16.1-orange?style=flat-square)](https://ai.google.dev/edge/litert)
[![ML%20Kit](https://img.shields.io/badge/ML%20Kit-face--detection%2016.1.7-FF6F00?style=flat-square)](https://developers.google.com/ml-kit/vision/face-detection)

A live Android demo of the **eye-state classifier** (open vs. closed) from the parent CIE-552 project. It runs the int8 winner model in real-time against the front-facing camera, draws a green bounding box around the detected face, and displays a smoothed OPEN / CLOSED label with per-class probability bars.

The bundled model (`assets/eye_winner.tflite`, 383 KB FP32 — int8 export is available alongside in `artifacts/`) is the **personally fine-tuned** version: 100 % accuracy on the 12 phone captures + 86.5 % retained on the canonical CEW test split.

## Pipeline

```
Front camera ─► CameraX ImageAnalysis (YUV → RGB upright)
   ─► ML Kit face detect (FAST mode, no landmarks)
   ─► Square crop, +30 % padding, −10 % y-shift (CEW composition)
   ─► BT.601 grayscale + 8×8 CLAHE (two-pass clip, bilinear inter-tile)
   ─► TFLite lw_wide (64×64 FP32, 4-thread XNNPACK)
   ─► EMA smoother α = 0.25  +  Schmitt-trigger hysteresis (0.62 / 0.38)
   ─► UI: green bbox + label + confidence bars
```

End-to-end latency on the S24 Ultra: ~1 ms median per frame (ML Kit face detection dominates; classifier itself is 0.13 ms).

## Requirements

| | |
|---|---|
| **JDK** | OpenJDK 21 (Android Studio JBR — shipped with AS) |
| **Android SDK** | platforms 35, build-tools 36.1.0 |
| **Gradle wrapper** | 8.10.2 (downloaded automatically on first build) |
| **adb** | from Android SDK platform-tools |
| **Device** | Android 7.0+ (minSdk 24) with a front camera. Tested on **Samsung Galaxy S24 Ultra (SM-S928B, OneUI on Android 15)** |

## Build & Install

```powershell
# From repo root
Set-Location android_app

# Set JAVA_HOME to Android Studio's bundled JBR (one-time)
$env:JAVA_HOME = "D:\program files\Android\Android Studio\jbr"

# Adjust local.properties if your Android SDK is not at the default path

# Debug build (first build: ~15 min — downloads Gradle + deps;
# subsequent builds: ~30 s)
.\gradlew.bat assembleDebug

# Install on the connected device
adb install -r app/build/outputs/apk/debug/app-debug.apk

# Grant camera permission without the on-device prompt
adb shell pm grant com.cv552.eyedemo android.permission.CAMERA

# Launch
adb shell am start -W -n com.cv552.eyedemo/.MainActivity
```

> If you hit `INSTALL_FAILED_VERIFICATION_FAILURE` on Samsung, **Auto Blocker** is intercepting unknown installs. Tap **Allow** on the on-device prompt that appears, then re-run `adb install -r`. No need to disable Auto Blocker globally.

## Project Structure

```
android_app/
├── settings.gradle.kts          Single-module Gradle setup
├── build.gradle.kts             Top-level plugin declarations (AGP 8.7, Kotlin 2.0.20)
├── gradle.properties            JVM args, AndroidX flags
├── local.properties             sdk.dir (adjust if your SDK lives elsewhere)
├── gradlew.bat                  Windows wrapper
├── gradle/wrapper/
│   ├── gradle-wrapper.jar       43 KB
│   └── gradle-wrapper.properties
└── app/
    ├── build.gradle.kts         Dependencies: CameraX, ML Kit, TFLite
    └── src/main/
        ├── AndroidManifest.xml
        ├── assets/
        │   └── eye_winner.tflite     Personally fine-tuned int8 winner (383 KB)
        ├── java/com/cv552/eyedemo/
        │   ├── MainActivity.kt       CameraX wiring + ML Kit face detect + UI
        │   ├── EyeClassifier.kt      TFLite wrapper, BT.601 grayscale, Kotlin CLAHE
        │   ├── FaceOverlayView.kt    Green bbox overlay
        │   └── Smoother.kt           EMA + Schmitt hysteresis
        └── res/
            ├── layout/activity_main.xml    Preview + overlay + status panel + 2 probability bars
            ├── values/themes.xml           Fullscreen Material theme
            └── values/strings.xml
```

## Configuration knobs

| Parameter | Default | Where to change | Notes |
|---|---|---|---|
| EMA alpha (smoothing strength) | 0.25 | `MainActivity.kt`, `Smoother(alpha = 0.25f)` | Lower = more responsive but jitter, higher = more stable but lagged |
| Hysteresis thresholds | 0.62 / 0.38 | `MainActivity.kt`, `Smoother(high=…, low=…)` | Narrower band = quicker flips; wider = more stable label |
| Face crop padding | +30 % each side | `MainActivity.kt`, `sideRaw = max(bb.width, bb.height) * 1.30f` | Tightens / loosens what the model sees |
| Face crop y-shift | −10 % side | `MainActivity.kt`, `cyAdj = cy - 0.10f * side` | Where the eyes land in the crop |
| Debug PNG dump | every 90 frames | `EyeClassifier.kt`, `DEBUG_SAVE_EVERY = 90` | Set to `0` to disable |

## Debugging on-device

The classifier dumps every Nth face-crop + post-CLAHE grayscale to `/sdcard/Android/data/com.cv552.eyedemo/files/debug/` (private external dir). Pull and inspect:

```powershell
adb pull /sdcard/Android/data/com.cv552.eyedemo/files/debug debug_pulls
```

The log tag is `EyeDemo` — view live with:
```powershell
adb logcat -s EyeDemo:I
```

Each frame logs `raw=[P(closed), P(open)] norm=[…] -> LABEL (n.n ms)`.

## Known limitations

1. **Out-of-distribution subjects**: the bundled model was fine-tuned on 12 photos covering 6 subjects. New faces (different ethnicity, beards, headphones, very different lighting) may still bias toward CLOSED. The fix is to add ~10 photos of the new subject and re-run `eye/07_eye_finetune.ipynb`, then `cp artifacts/eye_winner_finetuned.tflite app/src/main/assets/eye_winner.tflite` and rebuild.
2. **Gaze direction**: looking down at a laptop screen below the phone reduces P(open). Hold the phone at eye level.
3. **NNAPI is a no-op on Samsung OneUI**: the runtime accepts the flag but falls back to XNNPACK on this device. CPU-XNNPACK is already the optimum here (1.05× slower than the OpenVINO ARM plugin would be in theory).

## License & Attribution

Part of the CIE-552 Computer Vision Term Project. Bundled libraries are under their respective licenses (Apache 2.0 for TFLite, CameraX, AndroidX; Apache 2.0 for ML Kit face detection model). The shipping model is derived from training on the **CEW** dataset (research use) plus 12 author-captured photos.
