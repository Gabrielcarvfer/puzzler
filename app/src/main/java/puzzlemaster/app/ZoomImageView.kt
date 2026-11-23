package puzzlemaster.app

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.DashPathEffect
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View

/**
 * A photo you can pinch-zoom, drag and double-tap. Marks are drawn in screen-thin lines on top:
 * each piece's outline and a dashed ring around (not over) the exact piece.
 */
class ZoomImageView(context: Context, private val bmp: Bitmap, private val marks: List<Mark>) : View(context) {
    /** [outline]: x, y pairs in 0..1 of the photo; [cx], [cy]: the exact piece (0..1); [r]: ring radius, fraction of the photo's long side. */
    class Mark(val outline: FloatArray, val cx: Float, val cy: Float, val r: Float, val color: Int)

    var showMarks = true
        set(v) { field = v; invalidate() }
    private val m = Matrix()
    private var fitted = false
    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 3f }
    private val ring = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 4f; pathEffect = DashPathEffect(floatArrayOf(18f, 12f), 0f) }
    private val bmpPaint = Paint(Paint.FILTER_BITMAP_FLAG)

    private fun scaleNow(): Float { val v = FloatArray(9); m.getValues(v); return v[Matrix.MSCALE_X] }
    private fun minScale() = minOf(width.toFloat() / bmp.width, height.toFloat() / bmp.height)

    /** Starts zoomed so the marked pieces fill about half the view. */
    private fun fit() {
        val k0 = minScale()
        m.reset(); m.postScale(k0, k0); m.postTranslate((width - bmp.width * k0) / 2, (height - bmp.height * k0) / 2)
        if (marks.isEmpty()) return
        val xs = marks.map { it.cx * bmp.width }; val ys = marks.map { it.cy * bmp.height }
        val r = marks.maxOf { it.r } * maxOf(bmp.width, bmp.height)
        val spanX = (xs.max() - xs.min()) + 6 * r; val spanY = (ys.max() - ys.min()) + 6 * r
        val k = minOf(width / spanX, height / spanY).coerceIn(k0, k0 * 6)
        val cx = (xs.max() + xs.min()) / 2; val cy = (ys.max() + ys.min()) / 2
        m.reset(); m.postScale(k, k); m.postTranslate(width / 2f - cx * k, height / 2f - cy * k)
        clamp()
    }

    private fun clamp() {
        val v = FloatArray(9); m.getValues(v)
        val k = v[Matrix.MSCALE_X]; val w = bmp.width * k; val h = bmp.height * k
        var tx = v[Matrix.MTRANS_X]; var ty = v[Matrix.MTRANS_Y]
        tx = if (w <= width) (width - w) / 2 else tx.coerceIn(width - w, 0f)
        ty = if (h <= height) (height - h) / 2 else ty.coerceIn(height - h, 0f)
        v[Matrix.MTRANS_X] = tx; v[Matrix.MTRANS_Y] = ty; m.setValues(v)
    }

    private val scaler = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(d: ScaleGestureDetector): Boolean {
            val k = scaleNow(); val f = d.scaleFactor.coerceIn(minScale() / k, minScale() * 12 / k)
            m.postScale(f, f, d.focusX, d.focusY); clamp(); invalidate(); return true
        }
    })
    private val gestures = GestureDetector(context, object : GestureDetector.SimpleOnGestureListener() {
        override fun onScroll(e1: MotionEvent?, e2: MotionEvent, dx: Float, dy: Float): Boolean { m.postTranslate(-dx, -dy); clamp(); invalidate(); return true }
        override fun onDoubleTap(e: MotionEvent): Boolean {
            val k = scaleNow(); val target = if (k < minScale() * 2.5f) minScale() * 4 else minScale()
            m.postScale(target / k, target / k, e.x, e.y); clamp(); invalidate(); return true
        }
    })

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(e: MotionEvent): Boolean {
        parent?.requestDisallowInterceptTouchEvent(true)
        scaler.onTouchEvent(e); if (!scaler.isInProgress) gestures.onTouchEvent(e)
        return true
    }

    override fun onDraw(c: Canvas) {
        if (!fitted && width > 0) { fit(); fitted = true }
        c.drawBitmap(bmp, m, bmpPaint)
        if (!showMarks) return
        val pts = FloatArray(2)
        fun map(x: Float, y: Float): FloatArray { pts[0] = x * bmp.width; pts[1] = y * bmp.height; m.mapPoints(pts); return pts }
        val k = scaleNow()
        for (mk in marks) {
            line.color = mk.color; ring.color = mk.color
            line.setShadowLayer(6f, 0f, 0f, mk.color)
            val p = Path(); val n = mk.outline.size / 2
            for (i in 0 until n) { val q = map(mk.outline[2 * i], mk.outline[2 * i + 1]); if (i == 0) p.moveTo(q[0], q[1]) else p.lineTo(q[0], q[1]) }
            p.close(); c.drawPath(p, line)
            val q = map(mk.cx, mk.cy)
            c.drawCircle(q[0], q[1], mk.r * maxOf(bmp.width, bmp.height) * k, ring)
        }
    }
}
