package com.cv552.eyedemo

import android.annotation.SuppressLint
import android.content.Context
import android.os.Build
import android.provider.Settings
import android.util.Log
import com.google.firebase.FirebaseApp
import com.google.firebase.FirebaseOptions
import com.google.firebase.database.DatabaseReference
import com.google.firebase.database.FirebaseDatabase
import com.google.firebase.database.ServerValue

/**
 * Pushes live eye-state predictions to the lab1-f7c43 Firebase Realtime
 * Database, under a new top-level node `eye_monitor` (separate from the
 * existing `car_telemetry/latest/eye_state` field so the two pipelines do
 * not overwrite each other).
 *
 * Schema:
 *   eye_monitor/
 *     state       : "OPEN" | "CLOSED" | "UNKNOWN"
 *     p_open      : double in [0, 1]   (smoothed P(open) from the EMA)
 *     p_closed    : double in [0, 1]   (smoothed P(closed))
 *     latency_ms  : double             (single-frame model inference time)
 *     fps         : double             (running EMA of frame rate)
 *     timestamp   : long (server time, ms since epoch — ServerValue.TIMESTAMP)
 *     device_id   : string             ("<Build.MODEL>_<8-char ANDROID_ID>")
 *
 * Throttled to ~10 Hz (one write per 100 ms) so the database is not flooded
 * by the 30-fps camera analyzer.
 *
 * Init is programmatic via FirebaseOptions so the project compiles without a
 * google-services.json (we only have the web-app config from Firebase
 * console, which the Realtime Database SDK accepts).
 */
class FirebaseClient(context: Context) {

    private val ref: DatabaseReference?
    private val deviceId: String
    @Volatile private var lastPushNs: Long = 0L

    init {
        val appCtx = context.applicationContext
        val ok = ensureFirebaseInitialised(appCtx)
        ref = if (ok) {
            FirebaseDatabase.getInstance(DATABASE_URL).getReference("eye_monitor")
        } else null

        @SuppressLint("HardwareIds")
        val androidId = (Settings.Secure.getString(
            appCtx.contentResolver, Settings.Secure.ANDROID_ID
        ) ?: "unknown").take(8)
        deviceId = "${Build.MODEL.replace(" ", "_")}_$androidId"

        Log.i(TAG, "Firebase ${if (ok) "initialised" else "NOT initialised"}; " +
                "node=eye_monitor device=$deviceId")
    }

    /**
     * Push one observation. Throttled internally to one write per
     * [MIN_PUSH_INTERVAL_NS] (10 Hz default); calls that arrive sooner are
     * dropped silently. Force a write by passing [force]=true.
     */
    fun push(
        state: String,
        pOpen: Float,
        pClosed: Float,
        latencyMs: Double,
        fps: Double,
        force: Boolean = false,
    ) {
        val r = ref ?: return
        val nowNs = System.nanoTime()
        if (!force && nowNs - lastPushNs < MIN_PUSH_INTERVAL_NS) return
        lastPushNs = nowNs

        val payload = mapOf(
            "state" to state,
            "p_open" to pOpen.toDouble(),
            "p_closed" to pClosed.toDouble(),
            "latency_ms" to latencyMs,
            "fps" to fps,
            "timestamp" to ServerValue.TIMESTAMP,
            "device_id" to deviceId,
        )
        r.setValue(payload)
            .addOnFailureListener { e -> Log.w(TAG, "push failed: ${e.message}") }
    }

    private fun ensureFirebaseInitialised(context: Context): Boolean {
        return try {
            // FirebaseApp is a singleton per-name; only call initializeApp once.
            if (FirebaseApp.getApps(context).any { it.name == FirebaseApp.DEFAULT_APP_NAME }) {
                return true
            }
            val options = FirebaseOptions.Builder()
                .setApiKey("AIzaSyDGsJjDkg2YTs-n7RgbqprQ3XhDR8dhek0")
                .setApplicationId("1:1026461634964:web:689deb454072b495153f81")
                .setProjectId("lab1-f7c43")
                .setDatabaseUrl(DATABASE_URL)
                .setStorageBucket("lab1-f7c43.firebasestorage.app")
                .setGcmSenderId("1026461634964")
                .build()
            FirebaseApp.initializeApp(context, options)
            true
        } catch (e: Exception) {
            Log.e(TAG, "Firebase init failed: ${e.message}", e); false
        }
    }

    companion object {
        private const val TAG = "EyeDemo"
        private const val DATABASE_URL = "https://lab1-f7c43-default-rtdb.firebaseio.com"
        private const val MIN_PUSH_INTERVAL_NS = 100_000_000L   // 100 ms -> 10 Hz max
    }
}
