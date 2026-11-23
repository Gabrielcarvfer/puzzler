package puzzlemaster.app

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.graphics.RectF
import android.view.MotionEvent
import android.view.View
import kotlin.math.hypot

/** Shows a photo with four draggable corner handles (photo pixel coordinates). */
class CornerEditor(context: Context, private val photo: Bitmap, initial: List<PointF>) : View(context) {
    val corners = initial.map { PointF(it.x, it.y) }.toMutableList()
    private val dst = RectF()
    private var drag = -1
    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 5f; color = 0xFFFF7A1A.toInt(); setShadowLayer(10f, 0f, 0f, 0xFFFF7A1A.toInt()) }
    private val handle = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 6f; color = 0xFF3CE8FF.toInt(); setShadowLayer(10f, 0f, 0f, 0xFF3CE8FF.toInt()) }
    private val zoomBorder = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 4f; color = Color.WHITE }

    private fun scale() = minOf(width.toFloat() / photo.width, height.toFloat() / photo.height)
    private fun toView(p: PointF): PointF { val k = scale(); return PointF(dst.left + p.x * k, dst.top + p.y * k) }

    override fun onDraw(c: Canvas) {
        val k = scale()
        val w = photo.width * k; val h = photo.height * k
        dst.set((width - w) / 2, (height - h) / 2, (width + w) / 2, (height + h) / 2)
        c.drawColor(Color.BLACK)
        c.drawBitmap(photo, null, dst, null)
        val path = Path()
        corners.forEachIndexed { i, p -> val v = toView(p); if (i == 0) path.moveTo(v.x, v.y) else path.lineTo(v.x, v.y) }
        path.close(); c.drawPath(path, line)
        for (p in corners) { val v = toView(p); c.drawCircle(v.x, v.y, 36f, handle) }
        // Magnifier while dragging, so the corner can be placed exactly under the finger.
        if (drag >= 0) {
            val p = corners[drag]; val r = 0.04f * maxOf(photo.width, photo.height)
            val src = android.graphics.Rect((p.x - r).toInt(), (p.y - r).toInt(), (p.x + r).toInt(), (p.y + r).toInt())
            val box = RectF(20f, 20f, 380f, 380f)
            c.drawBitmap(photo, src, box, null); c.drawRect(box, zoomBorder)
            c.drawLine(200f, 170f, 200f, 230f, handle); c.drawLine(170f, 200f, 230f, 200f, handle)
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(e: MotionEvent): Boolean {
        val k = scale()
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                drag = corners.indices.minBy { val v = toView(corners[it]); hypot(v.x - e.x, v.y - e.y) }
                val v = toView(corners[drag]); if (hypot(v.x - e.x, v.y - e.y) > 150) drag = -1
            }
            MotionEvent.ACTION_MOVE -> if (drag >= 0) {
                corners[drag] = PointF(((e.x - dst.left) / k).coerceIn(0f, photo.width - 1f), ((e.y - dst.top) / k).coerceIn(0f, photo.height - 1f))
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> drag = -1
        }
        invalidate(); return true
    }
}
