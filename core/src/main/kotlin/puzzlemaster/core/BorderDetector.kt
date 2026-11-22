package puzzlemaster.core

import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/**
 * Finds border pieces directly from their straight side, without needing a clean outline of the
 * whole piece: line segments about one piece-side long, not part of a longer straight structure
 * (box edges, table edge, the assembled frame).
 */
object BorderDetector {
    class Segment(val x1: Double, val y1: Double, val x2: Double, val y2: Double) {
        val length get() = hypot(x2 - x1, y2 - y1)
        val angle get() = atan2(y2 - y1, x2 - x1)
        val midX get() = (x1 + x2) / 2
        val midY get() = (y1 + y2) / 2
    }

    /** [gray]: 8-bit image; [side]: expected piece side length in pixels. */
    fun find(gray: Mat, side: Double): List<Segment> {
        val lsd = Imgproc.createLineSegmentDetector()
        val lines = Mat()
        lsd.detect(gray, lines)
        val all = ArrayList<Segment>()
        for (i in 0 until lines.rows()) {
            val v = lines.get(i, 0)
            all.add(Segment(v[0], v[1], v[2], v[3]))
        }
        // Group collinear pieces of the same straight structure (LSD splits long lines).
        fun collinear(a: Segment, b: Segment): Boolean {
            var da = abs(a.angle - b.angle) % Math.PI
            if (da > Math.PI / 2) da = Math.PI - da
            if (da > 0.06) return false
            // distance of b's midpoint from a's infinite line
            val nx = -sin(a.angle); val ny = cos(a.angle)
            return abs((b.midX - a.x1) * nx + (b.midY - a.y1) * ny) < 4.0
        }
        val candidates = all.filter { it.length in side * 0.65..side * 1.35 }
        return candidates.filter { c ->
            // Total extent of the straight structure this segment belongs to (projected on its direction).
            val ux = cos(c.angle); val uy = sin(c.angle)
            var lo = 0.0; var hi = c.length
            for (o in all) {
                if (o === c || !collinear(c, o)) continue
                for ((px, py) in listOf(o.x1 to o.y1, o.x2 to o.y2)) {
                    val t = (px - c.x1) * ux + (py - c.y1) * uy
                    // only count parts that continue the line (gaps < half a side)
                    if (t < lo && t > lo - side * 0.5) lo = t
                    if (t > hi && t < hi + side * 0.5) hi = t
                }
            }
            hi - lo < side * 1.5
        }
    }
}
