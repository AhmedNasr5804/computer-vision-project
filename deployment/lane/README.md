# Lane segmentation — Raspberry Pi deployment

This directory contains the **deployment artifacts** for the lane pipeline from the CIE-552 project. Both models are post-training int8 quantized truncated U-Nets trained on the regenerated binary lane masks (Section 4.6 of `paper/paper.tex`).

| File | Purpose | Size | Output |
|---|---|---|---|
| `lane_truncated_3class.tflite` | **Primary deployment target** — 3-class segmentation (background / lane-left / lane-right) | 27.0 KB | `(1, 128, 128, 3)` int8 softmax |
| `lane_truncated_binary.tflite` | Simpler 1-class lane vs. background — useful if your downstream code only needs the lane mask | 26.7 KB | `(1, 128, 128, 1)` int8 sigmoid |

Both share the same input contract:

| Field | Value |
|---|---|
| Input shape | `(1, 128, 128, 3)` |
| Input dtype | `int8` |
| Input quantization | `scale = 0.002753`, `zero_point = -128` |
| Output quantization (both) | `scale = 0.003906`, `zero_point = -128` |
| Preprocessing | Resize Pi-Camera frame to 128×128, divide by 255, then quantize: `q = round(x/scale + zero_point)` clipped to `[-128, 127]` |

## Hardware target

- Raspberry Pi 4 Model B (4 GB) + Pi Camera Module v2 (Sony IMX219, 8 MP)
- Raspberry Pi OS Bookworm (64-bit) — the only version that ships `picamera2` by default
- Display attached, OR `--headless` mode that writes to `lane_demo.mp4`

## Prerequisites (install on the Pi)

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-pip libatlas-base-dev
pip3 install --break-system-packages tflite-runtime numpy
```

`tflite-runtime` (not full TensorFlow) is the right runtime for the Pi — much smaller and ships with the XNNPACK CPU delegate enabled by default.

## Claude Code prompt — paste into a Claude Code session running on the Pi

Open a terminal on the RPi, `cd` into the directory containing `lane_truncated_3class.tflite`, and run `claude code`. Paste the following prompt verbatim — Claude Code will write `lane_live.py` and run it for you.

> ```
> Build a Python 3 script called lane_live.py that runs a live lane-segmentation
> demo on a Raspberry Pi 4 with the Pi Camera Module v2.
>
> Model: lane_truncated_3class.tflite (an int8 TFLite model in the current directory).
>   - Input  : shape (1, 128, 128, 3), dtype int8, quant scale=0.002753 zp=-128
>     (the float input is the Pi-Camera frame resized to 128x128, RGB, divided by
>      255 to land in [0, 1], then quantized via q = round(x/scale + zp), clipped
>      to [-128, 127])
>   - Output : shape (1, 128, 128, 3), dtype int8, quant scale=0.003906 zp=-128
>     The three channels are the softmax probabilities for
>     [class 0 = background, class 1 = lane-left, class 2 = lane-right].
>     Dequantize via p = (q - zp) * scale, then argmax along channel axis.
>
> Pipeline:
>   1. Initialize the Pi Camera at 640x480 RGB888 using picamera2. Start it.
>   2. Load the model with tflite_runtime.interpreter.Interpreter (4 threads).
>      Read input/output details, including dtype and quantization parameters,
>      from the interpreter (do NOT hard-code them — verify they match the
>      header comment).
>   3. In a loop:
>      a. Capture one frame as a numpy array.
>      b. Resize to 128x128 (use cv2.INTER_AREA), divide by 255, quantize.
>      c. interpreter.set_tensor + invoke. Measure inference time with
>         time.perf_counter() (ns) around the invoke() call only.
>      d. Read the output tensor, dequantize, argmax over channel axis to get a
>         128x128 uint8 class map (values 0/1/2).
>      e. Build a colored overlay at 640x480: background = transparent,
>         lane-left = red (255, 0, 0), lane-right = green (0, 255, 0).
>         Use cv2.resize on the class map with INTER_NEAREST to upscale to
>         640x480 before coloring.
>      f. Blend the overlay with the original frame at alpha=0.5 (use
>         cv2.addWeighted).
>      g. Compute FPS as a running EMA with alpha=0.1 of (1.0 / wall_dt),
>         where wall_dt is the time between two consecutive iterations.
>      h. Draw two lines of text in the top-left of the frame using
>         cv2.putText (font HERSHEY_SIMPLEX, scale 0.6, thickness 2, color
>         (255, 255, 255), outlined by a thicker black version drawn first):
>            line 1: f"FPS: {fps:5.1f}"
>            line 2: f"Inference: {inference_ms:5.2f} ms"
>      i. cv2.imshow("Lane", blended) — and break out of the loop on key 'q'.
>   4. On exit, release the camera and destroy windows.
>
> Add a --headless command-line flag that, when set, writes lane_demo.mp4
> at 30 fps using cv2.VideoWriter (mp4v codec) instead of cv2.imshow.
> Stop after 30 seconds in headless mode.
>
> Once the script is written, run it (in non-headless mode) and report:
>   - the median inference_ms across the first ~150 frames,
>   - the steady-state FPS,
>   - any errors observed,
>   - the picamera2 / tflite_runtime versions detected.
>
> Important details:
>   - Use BGR throughout (picamera2 returns RGB888; convert with cv2.cvtColor
>     to BGR before cv2.imshow / cv2.VideoWriter, and back to RGB before the
>     model — the model was trained on RGB).
>   - Do not import tensorflow; use tflite_runtime only.
>   - Pre-allocate numpy buffers outside the loop for the resized image and
>     the quantized input to avoid per-frame allocation.
>   - If picamera2 fails to initialize (e.g., no camera detected), print a
>     clear error and exit cleanly.
> ```

When Claude Code finishes you should see a window showing the camera feed with red/green lane overlays and FPS/inference text in the corner. Expected ranges on a Pi 4B (TFLite + XNNPACK, 4 threads, int8):

| Metric | Expected |
|---|---|
| Inference per frame | 5–15 ms |
| End-to-end FPS | 25–40 |
| CPU utilization | 60–90% (single core dominant) |

## Binary mode

If your downstream code only needs the lane mask (no left/right split), use `lane_truncated_binary.tflite` instead. The Claude Code prompt above works almost unchanged — just substitute the model name, change the output shape to `(1, 128, 128, 1)`, drop the argmax (use `output > 0` on the dequantized sigmoid output), and color the lane region a single color.

## Troubleshooting

- **`ImportError: tflite_runtime`** — install it: `pip3 install --break-system-packages tflite-runtime`. On a 64-bit Pi OS this pulls the pre-built ARM64 wheel automatically.
- **`picamera2` fails with `[0:00:00.123456789] [123] ERROR`** — the Camera CSI ribbon is reversed or the `dtoverlay=imx219` line is missing from `/boot/firmware/config.txt`.
- **FPS below 10** — make sure you are running int8, not FP32. Check `interpreter.get_input_details()[0]["dtype"]` is `np.int8`, not `np.float32`. Also verify you're using the 4-thread XNNPACK delegate (default with tflite-runtime).
- **Overlay looks misaligned** — confirm the upscale uses `cv2.INTER_NEAREST`; bilinear interpolation will dilate the lane region by one pixel of every boundary and look fuzzy.

## How this model was trained (so you can re-train)

See `paper/paper.tex` Section 4.6 (`sec:lane_rebuild`) and the reproducible scripts in `artifacts/`:

1. `_lane_phase2_regen_masks.py` — classical Otsu+morphology mask regeneration
2. `_lane_phase3_4_train.py` — truncated U-Net training (binary)
3. `_lane_phase6_3class.py` — three-class extension
4. `_lane_phase5_ptq.py` — post-training int8 quantization
