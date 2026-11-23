package puzzlemaster.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.view.View
import puzzlemaster.core.Edge
import puzzlemaster.core.Piece
import kotlin.math.max

/** A catalogued piece visible in this frame. */
class SeenPiece(val piece: Piece, val entryId: Int, val sightings: Int, val flat: List<Edge>)

/** A solved pair with at least one of its pieces in view. [b]/[edgeB] are null when the partner is not in view. */
class SeenPair(val a: Piece, val edgeA: Edge, val b: Piece?, val edgeB: Edge?, val index: Int)

/** A found pair as two small pictures; (ax, ay) and (bx, by) mark the joining sides inside each picture. */
class PairThumb(val a: android.graphics.Bitmap, val ax: Float, val ay: Float, val b: android.graphics.Bitmap, val bx: Float, val by: Float, val index: Int)

/** One table photo with the outlines (x, y pairs, 0..1 of the photo) of the pieces to find in it. */
class WherePanel(val bmp: android.graphics.Bitmap, val outlines: List<FloatArray>)
class WhereView(val title: String, val panels: List<WherePanel>, val index: Int)

class LiveView(
    val pieces: List<SeenPiece>,
    val loose: List<Pair<DoubleArray, DoubleArray>>,
    val pairs: List<SeenPair>,
    /** Analysed image -> camera frame homography (row-major), or null if the frame was not warped. */
    val back: DoubleArray?,
    val frameW: Int, val frameH: Int,
)

/**
 * Draws over the live preview (same FILL_CENTER mapping as PreviewView): the outline of every
 * catalogued piece in view (orange = edge piece with its straight side in pink, red "C" = corner),
 * and for every solved pair a thick coloured outline on its pieces, joined by a line between the two
 * edges that fit when both are in view.
 */
class OverlayView(context: Context) : View(context) {
    @Volatile var live: LiveView? = null
        set(value) { field = value; postInvalidate() }
    /** Found pairs, shown as pictures under the status text whether or not the pieces are in view. */
    @Volatile var pairThumbs: List<PairThumb> = emptyList()
        set(value) { field = value; postInvalidate() }

    private val thin = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 3f; color = 0xFF3CE8FF.toInt(); setShadowLayer(8f, 0f, 0f, 0xFF3CE8FF.toInt()) }
    private val faint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 2f; color = Color.argb(120, 60, 232, 255) }
    private val thick = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 7f; setShadowLayer(10f, 0f, 0f, Color.WHITE) }
    private val link = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 12f; strokeCap = Paint.Cap.ROUND }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val label = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF3CE8FF.toInt(); textSize = 30f; typeface = android.graphics.Typeface.create(android.graphics.Typeface.MONOSPACE, android.graphics.Typeface.BOLD); setShadowLayer(6f, 0f, 0f, 0xFF3CE8FF.toInt()) }
    private val colors = intArrayOf(0xFFFF7A1A.toInt(), 0xFF3CE8FF.toInt(), 0xFFFFFFFF.toInt(), 0xFFFFB066.toInt(), 0xFF9AF2FF.toInt())

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

    private fun path(xs: DoubleArray, ys: DoubleArray, step: Int): Path {
        val p = Path()
        p.moveTo(vx(xs[0], ys[0]), vy(xs[0], ys[0]))
        var i = step
        while (i < xs.size) { p.lineTo(vx(xs[i], ys[i]), vy(xs[i], ys[i])); i += step }
        p.close()
        return p
    }

    /** When set, the "where is it" view covers the preview. */
    @Volatile var where: WhereView? = null
        set(value) { field = value; postInvalidate() }
    /** Called with a found pair's number when its picture at the top is tapped. */
    var onPairTap: ((Int) -> Unit)? = null
    private val thumbRects = ArrayList<Pair<android.graphics.RectF, Int>>()
    private val dim = Paint().apply { color = Color.argb(240, 5, 8, 12) }

    private fun drawWhere(canvas: Canvas, v: WhereView) {
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), dim)
        label.textSize = 34f
        canvas.drawText(v.title, 20f, 90f, label)
        val c = colors[v.index % colors.size]
        val top = 120f; val bottom = height - 40f
        val each = (bottom - top) / v.panels.size
        for ((i, p) in v.panels.withIndex()) {
            val rw = width - 32f; val rh = each - 16f
            val k = minOf(rw / p.bmp.width, rh / p.bmp.height)
            val w = p.bmp.width * k; val h = p.bmp.height * k
            val x0 = (width - w) / 2; val y0 = top + i * each + (each - h) / 2
            canvas.drawBitmap(p.bmp, null, android.graphics.RectF(x0, y0, x0 + w, y0 + h), null)
            val centres = ArrayList<FloatArray>()
            for (o in p.outlines) {
                val path = Path()
                var cx = 0f; var cy = 0f; var r = 0f
                val n = o.size / 2
                for (t in 0 until n) { val x = x0 + o[2 * t] * w; val y = y0 + o[2 * t + 1] * h; if (t == 0) path.moveTo(x, y) else path.lineTo(x, y); cx += x; cy += y }
                path.close(); cx /= n; cy /= n
                for (t in 0 until n) r = maxOf(r, kotlin.math.hypot(x0 + o[2 * t] * w - cx, y0 + o[2 * t + 1] * h - cy))
                thick.color = c; thick.strokeWidth = 6f
                canvas.drawPath(path, thick)
                thick.strokeWidth = 5f
                canvas.drawCircle(cx, cy, r * 1.9f, thick)
                centres.add(floatArrayOf(cx, cy))
            }
            if (centres.size == 2) { link.color = c; link.strokeWidth = 6f; canvas.drawLine(centres[0][0], centres[0][1], centres[1][0], centres[1][1], link); link.strokeWidth = 12f }
            thick.strokeWidth = 7f
        }
    }

    override fun onTouchEvent(event: android.view.MotionEvent): Boolean {
        if (event.action != android.view.MotionEvent.ACTION_UP) return true
        if (where != null) { where = null; return true }
        for ((r, idx) in thumbRects) if (r.contains(event.x, event.y)) { onPairTap?.invoke(idx); return true }
        return true
    }

    private fun drawThumbs(canvas: Canvas) {
        val list = pairThumbs
        if (list.isEmpty()) return
        val size = (width - 16f * 2 - 24f * 2) / 6f   // three pairs across
        var x = 16f; val y = 350f
        thumbRects.clear()
        for (t in list.take(3)) {
            thumbRects.add(android.graphics.RectF(x, y, x + 2 * size, y + size) to t.index)
            val c = colors[t.index % colors.size]
            fill.color = c; thick.color = c
            for ((bmp, px, py, dx) in listOf(Quad(t.a, t.ax, t.ay, 0f), Quad(t.b, t.bx, t.by, size))) {
                val r = android.graphics.RectF(x + dx, y, x + dx + size, y + size)
                canvas.drawBitmap(bmp, null, r, null)
                canvas.drawCircle(r.left + px * size / bmp.width, r.top + py * size / bmp.height, 11f, fill)
            }
            canvas.drawRect(x, y, x + 2 * size, y + size, thick)
            label.textSize = 34f
            canvas.drawText("${t.index + 1}", x + 8f, y + 36f, label)
            x += 2 * size + 24f
        }
    }
    private data class Quad(val bmp: android.graphics.Bitmap, val px: Float, val py: Float, val dx: Float)

    override fun onDraw(canvas: Canvas) {
        where?.let { drawWhere(canvas, it); return }
        drawThumbs(canvas)
        val v = live ?: return
        h = v.back
        scale = max(width.toFloat() / v.frameW, height.toFloat() / v.frameH)
        offX = (width - v.frameW * scale) / 2; offY = (height - v.frameH * scale) / 2
        for ((xs, ys) in v.loose) canvas.drawPath(path(xs, ys, 3), faint)
        for (s in v.pieces) {
            val p = s.piece
            if (s.flat.isEmpty()) canvas.drawPath(path(p.xs, p.ys, 2), thin)
            else {
                thick.color = if (s.flat.size >= 2) Color.WHITE else 0xFFFF7A1A.toInt()
                canvas.drawPath(path(p.xs, p.ys, 2), thick)
                thick.color = 0xFFFF7A1A.toInt()
                for (e in s.flat) canvas.drawLine(vx(e.startX, e.startY), vy(e.startX, e.startY), vx(e.endX, e.endY), vy(e.endX, e.endY), thick)
            }
            label.textSize = 24f
            val t = if (s.flat.size >= 2) "C" else "×${s.sightings}"
            canvas.drawText(t, vx(p.centroidX, p.centroidY) - label.measureText(t) / 2, vy(p.centroidX, p.centroidY) + 8f, label)
        }
        for (pr in v.pairs) {
            val c = colors[pr.index % colors.size]
            thick.color = c; link.color = c; fill.color = c
            canvas.drawPath(path(pr.a.xs, pr.a.ys, 2), thick)
            val x1 = vx(pr.edgeA.midX, pr.edgeA.midY); val y1 = vy(pr.edgeA.midX, pr.edgeA.midY)
            canvas.drawCircle(x1, y1, 18f, fill)
            val b = pr.b; val eb = pr.edgeB
            if (b != null && eb != null) {
                canvas.drawPath(path(b.xs, b.ys, 2), thick)
                val x2 = vx(eb.midX, eb.midY); val y2 = vy(eb.midX, eb.midY)
                canvas.drawLine(x1, y1, x2, y2, link)
                canvas.drawCircle(x2, y2, 18f, fill)
            }
            label.textSize = 34f
            val t = "${pr.index + 1}"
            canvas.drawText(t, x1 - label.measureText(t) / 2, y1 - 26f, label)
        }
    }
}
