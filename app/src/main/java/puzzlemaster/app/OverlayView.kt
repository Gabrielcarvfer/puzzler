package puzzlemaster.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.view.View
import puzzlemaster.core.LivePipeline
import kotlin.math.max

/** A confirmed pair: the two joint points in rectified-image coordinates. */
class ShownPair(val ax: Double, val ay: Double, val bx: Double, val by: Double, val index: Int)

/** What to draw: one pass's pieces plus the pairs confirmed over the last passes. */
class LiveView(
    val out: LivePipeline.Out,
    val pairs: List<ShownPair>,
    /** Rectified -> camera frame homography (row-major), or null if the frame was not warped. */
    val back: DoubleArray?,
    val frameW: Int, val frameH: Int,
)

/**
 * Draws over the live preview (same FILL_CENTER mapping as PreviewView): every detected piece's
 * outline, edge pieces in orange with their straight side in pink, corner pieces in red with a "C",
 * and a coloured line joining the two edges of every matching pair.
 */
class OverlayView(context: Context) : View(context) {
    @Volatile var live: LiveView? = null
        set(value) { field = value; postInvalidate() }

    private val thin = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 3f; color = Color.rgb(0, 230, 255) }
    private val faint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 2f; color = Color.argb(170, 255, 255, 255) }
    private val thick = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 7f }
    private val link = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 10f; strokeCap = Paint.Cap.ROUND }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val label = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = 44f; isFakeBoldText = true; setShadowLayer(6f, 0f, 0f, Color.BLACK) }
    private val colors = intArrayOf(0xFF00E676.toInt(), 0xFFFFEA00.toInt(), 0xFFFF4081.toInt(), 0xFFD500F9.toInt(), 0xFF76FF03.toInt(), 0xFFFFFFFF.toInt())

    private var scale = 1f; private var offX = 0f; private var offY = 0f
    private var h: DoubleArray? = null

    private fun vx(x: Double, y: Double): Float {
        val m = h ?: return (x * scale + offX).toFloat()
        val w = m[6] * x + m[7] * y + m[8]
        return (((m[0] * x + m[1] * y + m[2]) / w) * scale + offX).toFloat()
    }
    private fun vy(x: Double, y: Double): Float {
        val m = h ?: return (y * scale + offY).toFloat()
        val w = m[6] * x + m[7] * y + m[8]
        return (((m[3] * x + m[4] * y + m[5]) / w) * scale + offY).toFloat()
    }

    override fun onDraw(canvas: Canvas) {
        val v = live ?: return
        h = v.back
        scale = max(width.toFloat() / v.frameW, height.toFloat() / v.frameH)
        offX = (width - v.frameW * scale) / 2; offY = (height - v.frameH * scale) / 2
        for ((xs, ys) in v.out.loose) {
            val path = Path()
            path.moveTo(vx(xs[0], ys[0]), vy(xs[0], ys[0]))
            var i = 3
            while (i < xs.size) { path.lineTo(vx(xs[i], ys[i]), vy(xs[i], ys[i])); i += 3 }
            path.close()
            canvas.drawPath(path, faint)
        }
        for (p in v.out.pieces) {
            val path = Path()
            path.moveTo(vx(p.xs[0], p.ys[0]), vy(p.xs[0], p.ys[0]))
            var i = 2
            while (i < p.xs.size) { path.lineTo(vx(p.xs[i], p.ys[i]), vy(p.xs[i], p.ys[i])); i += 2 }
            path.close()
            val flat = v.out.border[p.id]
            if (flat == null) { canvas.drawPath(path, thin); continue }
            thick.color = if (flat.size >= 2) Color.RED else Color.rgb(255, 140, 0)
            canvas.drawPath(path, thick)
            thick.color = Color.MAGENTA
            for (e in flat) canvas.drawLine(vx(e.startX, e.startY), vy(e.startX, e.startY), vx(e.endX, e.endY), vy(e.endX, e.endY), thick)
            if (flat.size >= 2) canvas.drawText("C", vx(p.centroidX, p.centroidY) - label.measureText("C") / 2, vy(p.centroidX, p.centroidY) + 15f, label)
        }
        for (pr in v.pairs) {
            val c = colors[pr.index % colors.size]
            link.color = c; fill.color = c
            val x1 = vx(pr.ax, pr.ay); val y1 = vy(pr.ax, pr.ay); val x2 = vx(pr.bx, pr.by); val y2 = vy(pr.bx, pr.by)
            canvas.drawLine(x1, y1, x2, y2, link)
            canvas.drawCircle(x1, y1, 16f, fill); canvas.drawCircle(x2, y2, 16f, fill)
        }
    }
}
