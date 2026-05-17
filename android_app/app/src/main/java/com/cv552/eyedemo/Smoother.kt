package com.cv552.eyedemo

/**
 * Exponential-moving-average smoother with Schmitt-trigger hysteresis.
 *
 * - EMA: `s = alpha * x + (1-alpha) * s_prev`
 *   With alpha=0.25 the effective window is ~7 frames, which at 30 fps is ~230 ms —
 *   long enough to kill per-frame jitter, short enough to feel responsive.
 *
 * - Hysteresis (Schmitt trigger):
 *   * When the current displayed state is CLOSED, switch to OPEN only when smoothed P(open) > HIGH.
 *   * When the current displayed state is OPEN,   switch to CLOSED only when smoothed P(open) < LOW.
 *   This prevents the label from flickering when the model output is near 0.5.
 */
class Smoother(
    private val alpha: Float = 0.25f,
    private val high: Float = 0.62f,
    private val low: Float = 0.38f,
) {
    private var sOpen: Float = 0.5f
    private var hasSample = false
    var state: State = State.UNKNOWN
        private set

    enum class State { UNKNOWN, OPEN, CLOSED }

    /**
     * Push a new raw P(open), return the smoothed P(open) and the (possibly updated) state.
     */
    fun update(pOpen: Float): Smoothed {
        if (!hasSample) { sOpen = pOpen; hasSample = true }
        else { sOpen = alpha * pOpen + (1f - alpha) * sOpen }

        state = when (state) {
            State.OPEN     -> if (sOpen < low)  State.CLOSED else State.OPEN
            State.CLOSED   -> if (sOpen > high) State.OPEN   else State.CLOSED
            State.UNKNOWN  -> when {
                sOpen > high -> State.OPEN
                sOpen < low  -> State.CLOSED
                else         -> State.UNKNOWN
            }
        }
        return Smoothed(sOpen = sOpen, state = state)
    }

    /** Called when the face is lost — reset to unknown so we don't carry stale state. */
    fun reset() {
        hasSample = false
        sOpen = 0.5f
        state = State.UNKNOWN
    }

    data class Smoothed(val sOpen: Float, val state: State) {
        val sClosed: Float get() = 1f - sOpen
        val label: String get() = when (state) {
            State.OPEN -> "OPEN"; State.CLOSED -> "CLOSED"; State.UNKNOWN -> "…"
        }
    }
}
