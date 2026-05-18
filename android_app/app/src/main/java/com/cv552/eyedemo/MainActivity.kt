package com.cv552.eyedemo

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.RectF
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.cv552.eyedemo.databinding.ActivityMainBinding
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var classifier: EyeClassifier
    private lateinit var firebase: FirebaseClient
    private val smoother = Smoother(alpha = 0.25f, high = 0.62f, low = 0.38f)
    // Running EMA of frame rate (Hz). Updated whenever the analyzer fires.
    private var fpsEma: Double = 0.0
    private var lastFrameNs: Long = 0L
    private val faceDetector by lazy {
        FaceDetection.getClient(
            FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                .setMinFaceSize(0.20f)
                .build()
        )
    }

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else {
            Toast.makeText(this, R.string.permission_denied, Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        classifier = EyeClassifier(this)
        // Tell classifier where to dump debug PNGs (app-private external dir,
        // pullable with `adb pull /sdcard/Android/data/com.cv552.eyedemo/files/debug`)
        classifier.setDebugDir(java.io.File(getExternalFilesDir(null), "debug"))
        firebase = FirebaseClient(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        val (w, h) = classifier.inputSize()
        binding.labelText.text = "loading model… ${w}x${h}"
        binding.latencyText.text = "tap allow to grant camera"

        if (ContextCompat.checkSelfPermission(
                this, Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val cameraProvider = providerFuture.get()

            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
                .also { it.setSurfaceProvider(binding.previewView.surfaceProvider) }

            val analyzer = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, FaceAnalyzer())
                }

            val selector = CameraSelector.DEFAULT_FRONT_CAMERA
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, selector, preview, analyzer)
                binding.labelText.text = getString(R.string.waiting_face)
                binding.latencyText.text = ""
            } catch (e: Exception) {
                Log.e(TAG, "bind failed", e)
                binding.labelText.text = "Camera bind failed"
                binding.latencyText.text = e.message.orEmpty()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private inner class FaceAnalyzer : ImageAnalysis.Analyzer {
        @SuppressLint("UnsafeOptInUsageError")
        override fun analyze(imageProxy: ImageProxy) {
            val mediaImage = imageProxy.image
            if (mediaImage == null) { imageProxy.close(); return }

            val rotation = imageProxy.imageInfo.rotationDegrees
            val input = InputImage.fromMediaImage(mediaImage, rotation)

            val pipelineStart = System.nanoTime()
            // Update running FPS estimate from wall-clock between analyzer calls.
            if (lastFrameNs != 0L) {
                val dt = (pipelineStart - lastFrameNs) / 1e9
                if (dt > 0.0) {
                    val inst = 1.0 / dt
                    fpsEma = if (fpsEma == 0.0) inst else 0.9 * fpsEma + 0.1 * inst
                }
            }
            lastFrameNs = pipelineStart

            faceDetector.process(input)
                .addOnSuccessListener { faces ->
                    if (faces.isEmpty()) {
                        smoother.reset()
                        firebase.push("UNKNOWN", 0f, 0f, 0.0, fpsEma)
                        runOnUiThread {
                            binding.overlayView.setBoxNormalized(null)
                            binding.labelText.text = getString(R.string.waiting_face)
                            binding.latencyText.text = ""
                            binding.probOpen.progress = 0
                            binding.probClosed.progress = 0
                        }
                        return@addOnSuccessListener
                    }

                    val face = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }!!
                    val bb = face.boundingBox

                    // Rotate the analyzer image to upright orientation so the bitmap is
                    // in display coordinates, then crop the face rect.
                    val frameBmp = imageProxyToBitmap(imageProxy, rotation)
                    if (frameBmp == null) { imageProxy.close(); return@addOnSuccessListener }

                    val W = frameBmp.width; val H = frameBmp.height
                    // ML Kit's bbox is tighter than CEW's face crops. Expand to a square
                    // centered on the face, with ~30 % padding (asymmetric: a bit more on
                    // top so the eyes land in the upper-third of the crop, like CEW).
                    val cx = (bb.left + bb.right) / 2f
                    val cy = (bb.top + bb.bottom) / 2f
                    val sideRaw = maxOf(bb.width(), bb.height()) * 1.30f
                    val side = sideRaw.coerceAtMost(min(W, H).toFloat())
                    // Shift the box centre upward by ~10 % of side so eyes are in
                    // the upper third of the resulting square (matches CEW).
                    val cyAdj = cy - 0.10f * side
                    var left = (cx - side / 2f).toInt()
                    var top = (cyAdj - side / 2f).toInt()
                    var right = left + side.toInt()
                    var bottom = top + side.toInt()
                    // Clamp to image bounds, preserving box size by sliding (not shrinking)
                    if (left < 0) { right -= left; left = 0 }
                    if (top < 0) { bottom -= top; top = 0 }
                    if (right > W) { left -= right - W; right = W }
                    if (bottom > H) { top -= bottom - H; bottom = H }
                    left = left.coerceIn(0, W - 1)
                    top = top.coerceIn(0, H - 1)
                    right = right.coerceIn(left + 1, W)
                    bottom = bottom.coerceIn(top + 1, H)
                    val faceCrop = Bitmap.createBitmap(frameBmp, left, top, right - left, bottom - top)

                    val result = classifier.classify(faceCrop)
                    val pipelineMs = (System.nanoTime() - pipelineStart) / 1_000_000.0

                    // Temporal EMA smoothing + Schmitt-trigger hysteresis on the OPEN probability
                    val smoothed = smoother.update(result.pOpen)

                    // Push the smoothed observation to Firebase for the Pi to read.
                    // The client throttles internally to ~10 Hz; safe to call every frame.
                    firebase.push(
                        state = smoothed.label,
                        pOpen = smoothed.sOpen,
                        pClosed = smoothed.sClosed,
                        latencyMs = result.latencyMs,
                        fps = fpsEma,
                    )

                    val normRect = RectF(
                        left.toFloat() / W,
                        top.toFloat() / H,
                        right.toFloat() / W,
                        bottom.toFloat() / H,
                    )

                    runOnUiThread {
                        binding.overlayView.setBoxNormalized(normRect)
                        // The headline label tracks the smoothed/hysteresis state, but the
                        // confidence shown is computed from the smoothed signal so it doesn't
                        // bounce around when the state is held by hysteresis.
                        val displayConf = (maxOf(smoothed.sOpen, smoothed.sClosed) * 100f).toInt()
                        binding.labelText.text = "%s  %d%%".format(smoothed.label, displayConf)
                        binding.latencyText.text =
                            "raw [%.2f/%.2f] · smooth [%.2f/%.2f] · %.1f ms".format(
                                result.pClosed, result.pOpen,
                                smoothed.sClosed, smoothed.sOpen,
                                pipelineMs,
                            )
                        binding.probOpen.progress = (smoothed.sOpen * 100f).toInt()
                        binding.probClosed.progress = (smoothed.sClosed * 100f).toInt()
                    }
                }
                .addOnFailureListener { e ->
                    Log.e(TAG, "face detect failed", e)
                }
                .addOnCompleteListener {
                    imageProxy.close()
                }
        }
    }

    /** Convert ImageProxy (YUV) -> Bitmap, rotated to the display upright orientation. */
    private fun imageProxyToBitmap(imageProxy: ImageProxy, rotationDegrees: Int): Bitmap? {
        return try {
            val yBuffer = imageProxy.planes[0].buffer
            val uBuffer = imageProxy.planes[1].buffer
            val vBuffer = imageProxy.planes[2].buffer
            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()
            val nv21 = ByteArray(ySize + uSize + vSize)
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)
            val yuvImage = android.graphics.YuvImage(
                nv21, android.graphics.ImageFormat.NV21,
                imageProxy.width, imageProxy.height, null,
            )
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(
                android.graphics.Rect(0, 0, imageProxy.width, imageProxy.height),
                80, out,
            )
            val jpeg = out.toByteArray()
            val bmp = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size)
            if (rotationDegrees == 0) bmp
            else {
                val m = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
                Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
            }
        } catch (e: Exception) {
            Log.e(TAG, "imageProxyToBitmap failed", e); null
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        classifier.close()
        faceDetector.close()
    }

    companion object {
        private const val TAG = "EyeDemo"
    }
}
