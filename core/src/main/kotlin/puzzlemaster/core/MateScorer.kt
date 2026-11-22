package puzzlemaster.core

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Decides whether a tab and a blank physically fit, by actually fitting the two curves together.
 *
 * Edge B is flipped into edge A's frame (as if the pieces were joined) and rigidly aligned with a few
 * ICP steps (closest point on the other curve, so small corner-localisation errors don't matter).
 * The score is the 90th-percentile gap between the curves over the middle of the edge, relative to
 * the edge length: the knob is where edges differ, and an average over the whole edge would be
 * diluted by the straight shoulders that every edge shares.
 */
class MateScorer(val cfg: Config = Config()) {
    data class Config(
        /** Max relative difference of the two corner-to-corner lengths. */
        val maxLengthDiff: Double = 0.07,
        /** Max difference of knob height / knob position (fractions of edge length) before fitting. */
        val maxBulgeDiff: Double = 0.07,
        val maxPositionDiff: Double = 0.09,
        /** Part of the edge (from each end) ignored in the fit: corners are rounded and unreliable. */
        val trim: Double = 0.08,
        val icpIterations: Int = 4,
    )

    class Mate(
        /** 90th percentile gap / edge length. */
        val gap: Double,
        val meanGap: Double,
        val lengthDiff: Double,
    )

    fun compatible(a: Edge, b: Edge): Boolean =
        a.pieceId != b.pieceId &&
            ((a.type == EdgeType.TAB && b.type == EdgeType.BLANK) || (a.type == EdgeType.BLANK && b.type == EdgeType.TAB))

    /** Returns null when the edges cannot be mates (wrong types, lengths, knob size/position). */
    fun fit(a: Edge, b: Edge): Mate? {
        if (!compatible(a, b)) return null
        val la = a.length; val lb = b.length
        val l = (la + lb) / 2
        val lengthDiff = abs(la - lb) / l
        if (lengthDiff > cfg.maxLengthDiff) return null
        val n = a.cx.size
        // B as it sits when joined to A: reversed, x mirrored along the chord, y negated.
        val bx = DoubleArray(n) { lb - b.cx[n - 1 - it] + (la - lb) / 2 }
        val by = DoubleArray(n) { -b.cy[n - 1 - it] }
        // Cheap rejects: knob height and knob position along the edge.
        var ia = 0; var ib = 0
        for (i in 1 until n) { if (abs(a.cy[i]) > abs(a.cy[ia])) ia = i; if (abs(by[i]) > abs(by[ib])) ib = i }
        if (abs(abs(a.cy[ia]) - abs(by[ib])) / l > cfg.maxBulgeDiff) return null
        if (abs(a.cx[ia] - bx[ib]) / l > cfg.maxPositionDiff) return null

        val k0 = max(1, (n * cfg.trim).toInt()); val k1 = n - k0
        val m = k1 - k0
        val qx = DoubleArray(m); val qy = DoubleArray(m)
        // ICP: move B onto A.
        repeat(cfg.icpIterations) {
            for (i in k0 until k1) closest(a.cx[i], a.cy[i], bx, by, qx, qy, i - k0)
            // Rigid transform taking q (on B) to a.
            var sax = 0.0; var say = 0.0; var sqx = 0.0; var sqy = 0.0
            for (i in 0 until m) { sax += a.cx[i + k0]; say += a.cy[i + k0]; sqx += qx[i]; sqy += qy[i] }
            sax /= m; say /= m; sqx /= m; sqy /= m
            var dot = 0.0; var crs = 0.0
            for (i in 0 until m) {
                val ux = qx[i] - sqx; val uy = qy[i] - sqy
                val vx = a.cx[i + k0] - sax; val vy = a.cy[i + k0] - say
                dot += ux * vx + uy * vy; crs += ux * vy - uy * vx
            }
            val th = atan2(crs, dot); val c = cos(th); val s = sin(th)
            for (i in 0 until n) {
                val ux = bx[i] - sqx; val uy = by[i] - sqy
                bx[i] = c * ux - s * uy + sax; by[i] = s * ux + c * uy + say
            }
        }
        // Symmetric gaps: A samples to curve B, and B samples to curve A.
        val d = DoubleArray(2 * m)
        for (i in k0 until k1) {
            closest(a.cx[i], a.cy[i], bx, by, qx, qy, 0); d[i - k0] = hypot(a.cx[i] - qx[0], a.cy[i] - qy[0])
            closest(bx[i], by[i], a.cx, a.cy, qx, qy, 0); d[m + i - k0] = hypot(bx[i] - qx[0], by[i] - qy[0])
        }
        var mean = 0.0; for (v in d) mean += v; mean /= d.size
        d.sort()
        return Mate(d[(d.size * 0.9).toInt()] / l, mean / l, lengthDiff)
    }

    /** Closest point to (px, py) on the polyline; result into ox[o], oy[o]. */
    private fun closest(px: Double, py: Double, xs: DoubleArray, ys: DoubleArray, ox: DoubleArray, oy: DoubleArray, o: Int) {
        var best = Double.MAX_VALUE; var bx = 0.0; var by = 0.0
        for (j in 0 until xs.size - 1) {
            val x0 = xs[j]; val y0 = ys[j]
            val dx = xs[j + 1] - x0; val dy = ys[j + 1] - y0
            val len2 = dx * dx + dy * dy
            val t = if (len2 < 1e-12) 0.0 else (((px - x0) * dx + (py - y0) * dy) / len2).coerceIn(0.0, 1.0)
            val cx = x0 + t * dx; val cy = y0 + t * dy
            val d2 = (px - cx) * (px - cx) + (py - cy) * (py - cy)
            if (d2 < best) { best = d2; bx = cx; by = cy }
        }
        ox[o] = bx; oy[o] = by
    }

    @Suppress("unused") private fun unusedMath() = sqrt(min(1.0, 2.0))
}
