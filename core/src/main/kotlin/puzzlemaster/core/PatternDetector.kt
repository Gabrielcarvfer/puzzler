package puzzlemaster.core

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Finds pieces by pattern matching instead of segmentation.
 *
 * Commercial puzzles are cut with a standard die, so a handful of outline shapes repeat over the
 * whole puzzle. Templates of those shapes are learned from cleanly detected pieces, then searched for
 * in the image's edge map at every position and rotation (oriented chamfer matching): a template
 * fits where its outline lies on image edges running in the same direction. Touching pieces, print
 * texture and shadows don't break this, because the search only asks "is this known shape here?".
 */
class PatternDetector(val cfg: Config = Config()) {
    data class Config(
        /** Piece size (pixels) at the search resolution. */
        val searchPieceSize: Double = 48.0,
        val templatePoints: Int = 72,
        val coarseRotationStep: Double = 10.0,
        val positionStride: Int = 2,
        /** Chamfer truncation distance (search pixels). */
        val tau: Double = 3.0,
        val orientationWeight: Double = 0.6,
        /** Accept a fit below this cost (0 = perfect, ~1.6 = nothing there). */
        val maxCost: Double = 0.55,
        val maxTemplates: Int = 12,
    )

    /** Outline relative to its centroid, in original-image pixels. */
    class Template(val xs: DoubleArray, val ys: DoubleArray, val signature: String, val scale: Double)

    class Stats(var templates: Int = 0, var peaks: Int = 0, var accepted: Int = 0, var ms: Long = 0)

    /** Learns the die's outline shapes from good pieces (e.g. those found by [EdgeSegmenter]). */
    fun learnTemplates(pieces: List<Piece>): List<Template> {
        val good = pieces.filter { it.quality >= 0.35 }.sortedByDescending { it.quality }
        val perClass = HashMap<String, Int>()
        val out = ArrayList<Template>()
        for (p in good) {
            val sig = signature(p)
            if ((perClass[sig] ?: 0) >= 2) continue
            perClass[sig] = (perClass[sig] ?: 0) + 1
            out.add(Template(DoubleArray(p.xs.size) { p.xs[it] - p.centroidX }, DoubleArray(p.ys.size) { p.ys[it] - p.centroidY }, sig, p.scale))
            if (out.size >= cfg.maxTemplates) break
        }
        return out
    }

    /** Tab/blank/flat pattern around the piece, rotation-invariant (lexicographically smallest rotation). */
    private fun signature(p: Piece): String {
        val c = p.edges.map { when (it.type) { EdgeType.TAB -> 'T'; EdgeType.BLANK -> 'B'; EdgeType.FLAT -> 'F' } }.joinToString("")
        return (0 until 4).map { c.substring(it) + c.substring(0, it) }.min()
    }

    fun detect(img: RgbImage, templates: List<Template>, detector: PieceDetector, stats: Stats = Stats()): List<Piece> {
        val t0 = System.currentTimeMillis()
        if (templates.isEmpty()) return emptyList()
        stats.templates = templates.size
        val pieceScale = templates.map { it.scale }.sorted()[templates.size / 2]
        val ws = min(1.0, cfg.searchPieceSize / pieceScale)
        val w = (img.width * ws).toInt(); val h = (img.height * ws).toInt()

        // Edge map (non-maximum-suppressed gradient) with edge orientation.
        val lum = FloatArray(w * h)
        val inv = 1 / ws
        for (y in 0 until h) for (x in 0 until w) {
            val sx0 = (x * inv).toInt(); val sy0 = (y * inv).toInt()
            val sx1 = min(img.width, max(sx0 + 1, ((x + 1) * inv).toInt())); val sy1 = min(img.height, max(sy0 + 1, ((y + 1) * inv).toInt()))
            var s = 0f; var n = 0
            var yy = sy0
            while (yy < sy1) { var xx = sx0; while (xx < sx1) { val q = img[xx, yy]; s += 0.299f * ((q shr 16) and 0xFF) + 0.587f * ((q shr 8) and 0xFF) + 0.114f * (q and 0xFF); n++; xx++ }; yy++ }
            lum[y * w + x] = s / n
        }
        val mag = FloatArray(w * h); val ang = FloatArray(w * h)
        for (y in 1 until h - 1) for (x in 1 until w - 1) {
            val i = y * w + x
            val gx = (lum[i - w + 1] + 2 * lum[i + 1] + lum[i + w + 1]) - (lum[i - w - 1] + 2 * lum[i - 1] + lum[i + w - 1])
            val gy = (lum[i + w - 1] + 2 * lum[i + w] + lum[i + w + 1]) - (lum[i - w - 1] + 2 * lum[i - w] + lum[i - w + 1])
            mag[i] = sqrt(gx * gx + gy * gy); ang[i] = atan2(gy, gx)
        }
        val sorted = mag.copyOf(); sorted.sort()
        val thr = 0.15f * sorted[(sorted.size * 0.99).toInt()]
        val edge = BooleanArray(w * h)
        for (y in 1 until h - 1) for (x in 1 until w - 1) {
            val i = y * w + x
            val m = mag[i]
            if (m < thr) continue
            val dx = cos(ang[i]).roundToInt(); val dy = sin(ang[i]).roundToInt()
            if (m >= mag[i + dy * w + dx] && m >= mag[i - dy * w - dx]) edge[i] = true
        }
        // Distance transform carrying the nearest edge's orientation.
        val big = 1 shl 20
        val dist = IntArray(w * h) { if (edge[it]) 0 else big }
        val near = IntArray(w * h) { if (edge[it]) it else -1 }
        fun relax(i: Int, j: Int, c: Int) { if (dist[j] + c < dist[i]) { dist[i] = dist[j] + c; near[i] = near[j] } }
        for (y in 0 until h) for (x in 0 until w) {
            val i = y * w + x
            if (x > 0) relax(i, i - 1, 3)
            if (y > 0) { relax(i, i - w, 3); if (x > 0) relax(i, i - w - 1, 4); if (x < w - 1) relax(i, i - w + 1, 4) }
        }
        for (y in h - 1 downTo 0) for (x in w - 1 downTo 0) {
            val i = y * w + x
            if (x < w - 1) relax(i, i + 1, 3)
            if (y < h - 1) { relax(i, i + w, 3); if (x < w - 1) relax(i, i + w + 1, 4); if (x > 0) relax(i, i + w - 1, 4) }
        }
        val dt = FloatArray(w * h) { min(cfg.tau, dist[it] / 3.0).toFloat() }
        val nearAng = FloatArray(w * h) { if (near[it] >= 0) ang[near[it]] else 0f }

        // Template point sets at search scale for every coarse rotation.
        val m = cfg.templatePoints
        class Pose(val t: Int, val rot: Double, val dx: IntArray, val dy: IntArray, val nAng: FloatArray, val order: IntArray)
        fun pose(ti: Int, rotDeg: Double): Pose {
            val tp = templates[ti]
            val (rx, ry) = PieceDetector.resampleClosed(tp.xs, tp.ys, m)
            val r = rotDeg * PI / 180; val c = cos(r); val s = sin(r)
            val dx = IntArray(m); val dy = IntArray(m); val na = FloatArray(m)
            for (k in 0 until m) {
                val x = (c * rx[k] - s * ry[k]) * ws; val y = (s * rx[k] + c * ry[k]) * ws
                dx[k] = x.roundToInt(); dy[k] = y.roundToInt()
                val a = (k - 1 + m) % m; val b = (k + 1) % m
                val tx = rx[b] - rx[a]; val ty = ry[b] - ry[a]
                na[k] = (atan2(tx.toDouble(), -ty.toDouble()) + r).toFloat() // normal direction
            }
            // Spread evaluation order so early exit sees the whole outline.
            val order = IntArray(m) { (it * 17) % m }
            return Pose(ti, rotDeg, dx, dy, na, order)
        }
        val poses = ArrayList<Pose>()
        var rot = 0.0
        while (rot < 360) { for (ti in templates.indices) poses.add(pose(ti, rot)); rot += cfg.coarseRotationStep }
        val radius = (pieceScale * ws * 0.9).toInt() + 2

        fun cost(p: Pose, cx: Int, cy: Int, limit: Double): Double {
            var sum = 0.0
            for ((n, kk) in p.order.withIndex()) {
                val x = cx + p.dx[kk]; val y = cy + p.dy[kk]
                if (x < 0 || y < 0 || x >= w || y >= h) { sum += 1 + cfg.orientationWeight; continue }
                val i = y * w + x
                val d = dt[i]
                sum += d / cfg.tau + cfg.orientationWeight * (if (d >= cfg.tau) 1.0 else abs(sin(nearAng[i] - p.nAng[kk]).toDouble()))
                if (n == 15 && sum / 16 > limit * 1.5) return Double.MAX_VALUE
                if (n == 35 && sum / 36 > limit * 1.25) return Double.MAX_VALUE
            }
            return sum / m
        }

        // Coarse search.
        val st = cfg.positionStride
        val gw = w / st; val gh = h / st
        val best = FloatArray(gw * gh) { Float.MAX_VALUE }
        val bestPose = IntArray(gw * gh) { -1 }
        for (gy in 0 until gh) for (gx in 0 until gw) {
            val cx = gx * st; val cy = gy * st
            if (cx < radius / 2 || cy < radius / 2 || cx >= w - radius / 2 || cy >= h - radius / 2) continue
            // Quick reject: a piece centre is surrounded by edges at about the piece radius.
            var bv = cfg.maxCost * 1.15
            var bp = -1
            for ((pi, p) in poses.withIndex()) {
                val c = cost(p, cx, cy, bv)
                if (c < bv) { bv = c; bp = pi }
            }
            if (bp >= 0) { best[gy * gw + gx] = bv.toFloat(); bestPose[gy * gw + gx] = bp }
        }
        // Peaks with non-maximum suppression.
        val peaks = ArrayList<Int>()
        for (i in best.indices) if (bestPose[i] >= 0) peaks.add(i)
        peaks.sortBy { best[it] }
        val chosen = ArrayList<Int>()
        val minSep = pieceScale * ws * 0.6 / st
        for (i in peaks) {
            val x = i % gw; val y = i / gw
            if (chosen.any { val dx = it % gw - x; val dy = it / gw - y; dx * dx + dy * dy < minSep * minSep }) continue
            chosen.add(i)
        }
        stats.peaks = chosen.size

        // Refine each peak (rotation +-6 deg, position +-st), then fit at full resolution.
        val lumFull = OutlineRefiner.luminance(img)
        val out = ArrayList<Piece>()
        for (i in chosen) {
            val p0 = poses[bestPose[i]]
            var bc = Double.MAX_VALUE; var bx = 0; var by = 0; var br = p0.rot
            var r = p0.rot - 6
            while (r <= p0.rot + 6) {
                val p = pose(p0.t, r)
                for (yy in -st..st) for (xx in -st..st) {
                    val c = cost(p, (i % gw) * st + xx, (i / gw) * st + yy, Double.MAX_VALUE)
                    if (c < bc) { bc = c; bx = (i % gw) * st + xx; by = (i / gw) * st + yy; br = r }
                }
                r += 2.0
            }
            if (bc > cfg.maxCost) continue
            val tp = templates[p0.t]
            val rr = br * PI / 180; val c = cos(rr); val s = sin(rr)
            val ox = DoubleArray(tp.xs.size) { (c * tp.xs[it] - s * tp.ys[it]) + (bx + 0.5) / ws }
            val oy = DoubleArray(tp.ys.size) { (s * tp.xs[it] + c * tp.ys[it]) + (by + 0.5) / ws }
            // Snap the template outline onto the actual piece edge (the real piece differs slightly).
            val (rx, ry) = OutlineRefiner.refine(ox, oy, lumFull, img.width, img.height, 2.5 / ws)
            val piece = detector.analyzeContour(rx, ry, img, out.size) ?: continue
            if (!detector.isPlausible(piece)) continue
            out.add(piece)
        }
        stats.accepted = out.size
        stats.ms = System.currentTimeMillis() - t0
        return out
    }
}
