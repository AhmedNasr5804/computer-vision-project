package com.cv552.eyedemo

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/**
 * Transparent overlay that draws a green bounding box on the detected face,
 * mapped from analyzer-image coordinates to view coordinates.
 */
class FaceOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0,
) : View(context, attrs, defStyleAttr) {

    private var box: RectF? = null
    private val paint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }

    /** rect is in normalized 0..1 coordinates over the camera-preview surface. */
    fun setBoxNormalized(rect: RectF?) {
        box = rect
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val b = box ?: return
        val r = RectF(
            b.left * width,
            b.top * height,
            b.right * width,
            b.bottom * height,
        )
        canvas.drawRect(r, paint)
    }
}
