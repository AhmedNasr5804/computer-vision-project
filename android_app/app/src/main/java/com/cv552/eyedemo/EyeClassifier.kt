package com.cv552.eyedemo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

/**
 * Wraps `assets/eye_winner.tflite` (the §06 winner — `lw_wide`).
 *
 * Model schema (verified by inspecting the .tflite on the workstation):
 *   - input  shape  (1, 64, 64, 1)
 *   - input  dtype  float32   (model is "weight-only" quantized)
 *   - output shape  (1, 2)
 *   - output dtype  float32   — softmax over [P(closed), P(open)]
 *
 * Training preprocessing (from notebook eye/01 and eye/03):
 *   1. cv2.imread(path, IMREAD_GRAYSCALE)              -> grayscale (BT.601)
 *   2. cv2.resize(64x64)
 *   3. cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply()
 *   4. divide by 255 -> [0,1] float
 *
 * The runtime pipeline below mirrors this exactly: BT.601 grayscale,
 * full CLAHE with bilinear interpolation between 8x8 tiles, then /255.
 */
class EyeClassifier(context: Context) {

    private val interpreter: Interpreter
    private val inputW: Int
    private val inputH: Int

    // Reusable buffers (avoid GC pressure in the camera loop)
    private val inBuffer: ByteBuffer
    private val outBuffer = Array(1) { FloatArray(2) }

    // Pre-allocated scratch space for CLAHE
    private val grayBuf: IntArray
    private val claheBuf: IntArray

    // Histogram scratch
    private val hist = IntArray(256)
    private val mapping: Array<Array<IntArray>>

    init {
        val mappedModel = FileUtil.loadMappedFile(context, MODEL_FILE)
        val opts = Interpreter.Options().apply { setNumThreads(4) }
        interpreter = Interpreter(mappedModel, opts)

        val inTensor = interpreter.getInputTensor(0)
        val outTensor = interpreter.getOutputTensor(0)
        val inShape = inTensor.shape() // expected [1, 64, 64, 1]
        inputH = inShape[1]
        inputW = inShape[2]
        // 4 bytes per float
        inBuffer = ByteBuffer.allocateDirect(inputW * inputH * 4).order(ByteOrder.nativeOrder())

        grayBuf = IntArray(inputW * inputH)
        claheBuf = IntArray(inputW * inputH)

        val tilesX = inputW / TILE
        val tilesY = inputH / TILE
        mapping = Array(tilesY) { Array(tilesX) { IntArray(256) } }

        Log.i(TAG, "Model loaded. input=(${inShape.joinToString()}) " +
                "inDtype=${inTensor.dataType().name} outDtype=${outTensor.dataType().name}")
    }

    fun inputSize() = Pair(inputW, inputH)

    private var debugSaveCounter = 0
    private var debugDir: File? = null
    fun setDebugDir(dir: File?) { debugDir = dir }

    fun classify(faceCrop: Bitmap): Result {
        // 1) resize to model input size
        val resized = if (faceCrop.width == inputW && faceCrop.height == inputH)
            faceCrop
        else
            Bitmap.createScaledBitmap(faceCrop, inputW, inputH, /*filter*/ true)

        // Save crop + post-CLAHE input every Nth call for offline inspection
        debugSaveCounter++
        val doSave = DEBUG_SAVE && debugDir != null && debugSaveCounter % DEBUG_SAVE_EVERY == 0

        // 2) BT.601 grayscale (matches OpenCV cv2.IMREAD_GRAYSCALE)
        // Read all pixels in one call (much faster than per-pixel getPixel).
        val pixels = IntArray(inputW * inputH)
        resized.getPixels(pixels, 0, inputW, 0, 0, inputW, inputH)
        for (i in pixels.indices) {
            val px = pixels[i]
            val r = (px shr 16) and 0xff
            val g = (px shr 8) and 0xff
            val b = px and 0xff
            grayBuf[i] = (0.299f * r + 0.587f * g + 0.114f * b).toInt().coerceIn(0, 255)
        }

        // 3) CLAHE (tile 8x8, clipLimit=2.0) — matches the OpenCV defaults used in training
        applyCLAHE(grayBuf, inputW, inputH, claheBuf)

        // 4) write float32 input in [0, 1]
        inBuffer.rewind()
        for (i in claheBuf.indices) {
            inBuffer.putFloat(claheBuf[i] / 255f)
        }
        inBuffer.rewind()

        if (doSave) {
            try {
                val dir = debugDir!!
                dir.mkdirs()
                val stamp = System.currentTimeMillis()
                // Save the (already-resized) face crop bitmap as the model sees it
                FileOutputStream(File(dir, "crop_${stamp}.png")).use {
                    resized.compress(Bitmap.CompressFormat.PNG, 100, it)
                }
                // Save the post-CLAHE grayscale image
                val claheBmp = Bitmap.createBitmap(inputW, inputH, Bitmap.Config.ARGB_8888)
                val pix = IntArray(claheBuf.size) { i ->
                    val v = claheBuf[i]
                    Color.argb(255, v, v, v)
                }
                claheBmp.setPixels(pix, 0, inputW, 0, 0, inputW, inputH)
                FileOutputStream(File(dir, "clahe_${stamp}.png")).use {
                    claheBmp.compress(Bitmap.CompressFormat.PNG, 100, it)
                }
                claheBmp.recycle()
                Log.i(TAG, "DEBUG dumped crop_$stamp.png + clahe_$stamp.png in $dir")
            } catch (e: Exception) {
                Log.w(TAG, "debug save failed: $e")
            }
        }

        // 5) inference
        val t0 = System.nanoTime()
        interpreter.run(inBuffer, outBuffer)
        val tNs = System.nanoTime() - t0

        val pClosedRaw = outBuffer[0][0]
        val pOpenRaw = outBuffer[0][1]
        val sum = (pClosedRaw + pOpenRaw).coerceAtLeast(1e-6f)
        val pClosed = pClosedRaw / sum
        val pOpen = pOpenRaw / sum

        // Debug — logged at INFO so it shows in logcat -s EyeDemo:*
        if (DEBUG_LOG) {
            Log.i(TAG, "raw=[%.4f, %.4f]  norm=[%.4f, %.4f]  -> %s (%.1f ms)".format(
                pClosedRaw, pOpenRaw, pClosed, pOpen,
                if (pOpen >= pClosed) "OPEN" else "CLOSED", tNs / 1e6
            ))
        }
        return Result(pClosed = pClosed, pOpen = pOpen, latencyMs = tNs / 1_000_000.0)
    }

    /**
     * CLAHE with bilinear interpolation between tile centers.
     * Faithful to cv2.createCLAHE(clipLimit, (tile,tile)).apply():
     *   1. for each TILE×TILE block, compute histogram, clip ABOVE limit
     *   2. redistribute the excess uniformly across ALL 256 bins (this can cause
     *      some bins to exceed the limit again — OpenCV iterates this once more,
     *      so we mimic that with a second pass of clip+redistribute)
     *   3. build the per-tile mapping (cumulative histogram -> [0,255])
     *   4. for every pixel, bilinearly interpolate the four nearest tile mappings
     */
    private fun applyCLAHE(src: IntArray, w: Int, h: Int, dst: IntArray) {
        val tilesX = w / TILE
        val tilesY = h / TILE
        val tileSize = TILE * TILE
        // OpenCV: clipLimit is multiplied by tileSize/256 and floored, with a minimum of 1.
        val limit = max(1, (CLIP_LIMIT * tileSize / 256f).toInt())

        // step 1+2+3: per-tile mapping
        for (ty in 0 until tilesY) {
            for (tx in 0 until tilesX) {
                // histogram
                for (i in 0 until 256) hist[i] = 0
                val y0 = ty * TILE; val x0 = tx * TILE
                for (y in y0 until y0 + TILE) {
                    val row = y * w
                    for (x in x0 until x0 + TILE) {
                        hist[src[row + x]]++
                    }
                }
                // Two-pass clip+redistribute — matches OpenCV's behavior more closely
                repeat(2) {
                    var excess = 0
                    for (i in 0 until 256) {
                        if (hist[i] > limit) { excess += hist[i] - limit; hist[i] = limit }
                    }
                    if (excess == 0) return@repeat
                    val perBin = excess / 256
                    var remainder = excess - perBin * 256
                    for (i in 0 until 256) {
                        hist[i] += perBin
                        if (remainder > 0) { hist[i]++; remainder-- }
                    }
                }
                // CDF -> mapping
                var cum = 0
                val m = mapping[ty][tx]
                for (i in 0 until 256) {
                    cum += hist[i]
                    m[i] = ((cum * 255L) / tileSize).toInt().coerceIn(0, 255)
                }
            }
        }

        // step 3: bilinear interpolation across tiles
        val half = TILE / 2f
        for (y in 0 until h) {
            val tyf = ((y + 0.5f) / TILE - 0.5f).coerceIn(0f, (tilesY - 1).toFloat())
            val ty0 = tyf.toInt()
            val ty1 = min(ty0 + 1, tilesY - 1)
            val fy = tyf - ty0
            val row = y * w
            for (x in 0 until w) {
                val txf = ((x + 0.5f) / TILE - 0.5f).coerceIn(0f, (tilesX - 1).toFloat())
                val tx0 = txf.toInt()
                val tx1 = min(tx0 + 1, tilesX - 1)
                val fx = txf - tx0
                val g = src[row + x]
                val v00 = mapping[ty0][tx0][g]
                val v01 = mapping[ty0][tx1][g]
                val v10 = mapping[ty1][tx0][g]
                val v11 = mapping[ty1][tx1][g]
                val top = (1 - fx) * v00 + fx * v01
                val bot = (1 - fx) * v10 + fx * v11
                dst[row + x] = ((1 - fy) * top + fy * bot).toInt().coerceIn(0, 255)
            }
        }
    }

    fun close() = interpreter.close()

    data class Result(val pClosed: Float, val pOpen: Float, val latencyMs: Double) {
        val label: String get() = if (pOpen >= pClosed) "OPEN" else "CLOSED"
        val confidence: Float get() = max(pClosed, pOpen)
    }

    companion object {
        private const val TAG = "EyeDemo"
        private const val MODEL_FILE = "eye_winner.tflite"
        private const val TILE = 8
        private const val CLIP_LIMIT = 2.0f
        private const val DEBUG_LOG = true
        private const val DEBUG_SAVE = true
        private const val DEBUG_SAVE_EVERY = 90   // ~3 sec at 30 fps
    }
}
