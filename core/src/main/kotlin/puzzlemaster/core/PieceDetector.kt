package puzzlemaster.core

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

enum class EdgeType { FLAT, TAB, BLANK }

/**
 * One side of a piece, between two consecutive corners.
 * Canonical frame: start corner at the origin, end corner on the +x axis, piece interior on +y.
 */
class Edge(
    val pieceId: Int,
    val index: Int,
    val type: EdgeType,
    /** Corner-to-corner distance, pixels. */
    val length: Double,
    /** Arc-length resampled points in the canonical frame (pixels). */
    val cx: DoubleArray,
    val cy: DoubleArray,
    /** Colours sampled just inside the piece along the edge, r,g,b triples, start to end. */
    val colors: FloatArray,
    /** Image-space geometry for display. */
    val startX: Double, val startY: Double,
    val endX: Double, val endY: Double,
    /** Image-space point on the contour at the middle of the edge (tip of a tab / bottom of a blank). */
    val midX: Double, val midY: Double,
    /**
     * Picture strip just inside the edge: [stripRows] rows (increasing inset) x [stripLen] samples
     * along the edge (start to end) x RGB. Used to check the picture continues across the joint.
     */
    val strip: FloatArray = FloatArray(0),
    val stripRows: Int = 0,
    val stripLen: Int = 0,
    /** The owning piece's colour palette (see [Piece.palette]). */
    val palette: FloatArray = FloatArray(0),
) {
    fun withPalette(p: FloatArray) = Edge(pieceId, index, type, length, cx, cy, colors, startX, startY, endX, endY, midX, midY, strip, stripRows, stripLen, p)

    /** Height of the tab/blank relative to the edge length (positive = outward). */
    val bulge: Double get() = cy.maxOf { -it }.let { out -> val inward = cy.maxOf { it }; if (out >= inward) out / length else -inward / length }
}

class Piece(
    val id: Int,
    /** Smoothed closed contour in image coordinates. */
    val xs: DoubleArray,
    val ys: DoubleArray,
    /** Indices into xs/ys of the four corners, in contour order. */
    val cornerIdx: IntArray,
    val edges: List<Edge>,
    val centroidX: Double,
    val centroidY: Double,
    /** Mean side length in pixels. */
    val scale: Double,
    val area: Double,
    /** Fit of the 4-corner piece model (higher is better). */
    val quality: Double = 0.0,
    /** Chromaticity histogram of the whole piece surface (lighting-invariant colour mix), sums to 1. */
    val palette: FloatArray = FloatArray(0),
) {
    val flatCount: Int get() = edges.count { it.type == EdgeType.FLAT }
}

/** Finds puzzle pieces in an image and describes their four edges. */
class PieceDetector(val cfg: Config = Config()) {
    data class Config(
        val segmentation: Segmenter.Config = Segmenter.Config(),
        /** Points per edge descriptor. */
        val edgeSamples: Int = 48,
        /** Colour samples per edge. */
        val colorSamples: Int = 16,
        /** |bulge| below this fraction of edge length means a flat (border) edge. */
        val flatThreshold: Double = 0.12,
        /** Colour sampling inset as a fraction of piece scale. */
        val colorInset: Double = 0.06,
        /** Picture strip insets (fraction of piece size) and samples along each edge. */
        val stripInsets: DoubleArray = doubleArrayOf(0.025, 0.05, 0.08),
        val stripSamples: Int = 32,
        val skipBorderTouching: Boolean = true,
        /** Compute the whole-piece colour palette (skip for fast screening). */
        val computePalette: Boolean = true,
        /** Contour resampling cap (lower = faster, coarser). */
        val maxContourPoints: Int = 1200,
        /** Tab/blank depth limit as a fraction of side length. */
        val maxBulge: Double = 0.6,
        val minRelativeSize: Double = 0.65,
        val maxRelativeSize: Double = 1.5,
    )

    private var lastCornerScore = 0.0

    /** Last corner candidates (debugging aid, not thread safe). */
    @Volatile var debugCandidates: List<Int> = emptyList()
    @Volatile var debugContour: Pair<DoubleArray, DoubleArray>? = null

    class Result(val segmentation: Segmentation, val pieces: List<Piece>, val rejected: Int)

    fun detect(img: RgbImage, firstId: Int = 0): Result {
        val seg = Segmenter.segment(img, cfg.segmentation)
        val pieces = ArrayList<Piece>()
        var rejected = 0
        var id = firstId
        for (c in seg.components) {
            if (cfg.skipBorderTouching && c.touchesBorder) { rejected++; continue }
            val raw = traceContour(seg.labels, seg.width, seg.height, c.label, c.startX, c.startY)
            val piece = analyzeContour(raw.first, raw.second, img, id)
            if (piece == null || !isPlausible(piece)) rejected++ else { pieces.add(piece); id++ }
        }
        // Size consistency: all pieces of one puzzle are about the same size, so fragments (part of
        // a piece separated from the table) and clumps (touching pieces) stand out.
        if (pieces.size >= 5) {
            val med = pieces.map { it.scale }.sorted()[pieces.size / 2]
            val before = pieces.size
            pieces.retainAll { it.scale in med * cfg.minRelativeSize..med * cfg.maxRelativeSize }
            rejected += before - pieces.size
        }
        return Result(seg, pieces, rejected)
    }

    /** Chromaticity (r, g share of r+g+b) histogram over the piece interior, sampled on a grid. */
    private fun palette(px: DoubleArray, py: DoubleArray, img: RgbImage): FloatArray {
        val bins = PALETTE_BINS
        val h = FloatArray(bins * bins)
        val minX = px.min(); val maxX = px.max(); val minY = py.min(); val maxY = py.max()
        val step = max(1.0, max(maxX - minX, maxY - minY) / 28)
        // Shrink the polygon slightly so the cut edge / shadow isn't sampled.
        val cx = px.average(); val cy = py.average()
        val sx = DoubleArray(px.size) { cx + (px[it] - cx) * 0.85 }; val sy = DoubleArray(py.size) { cy + (py[it] - cy) * 0.85 }
        var n = 0
        var y = minY + step / 2
        while (y < maxY) {
            // Scanline crossings.
            val xs = ArrayList<Double>()
            for (i in sx.indices) {
                val j = (i + 1) % sx.size
                if ((sy[i] <= y && sy[j] > y) || (sy[j] <= y && sy[i] > y)) xs.add(sx[i] + (y - sy[i]) / (sy[j] - sy[i]) * (sx[j] - sx[i]))
            }
            xs.sort()
            var k = 0
            while (k + 1 < xs.size) {
                var x = xs[k] + step / 2
                while (x < xs[k + 1]) {
                    val ix = x.toInt(); val iy = y.toInt()
                    if (ix in 0 until img.width && iy in 0 until img.height) {
                        val q = img[ix, iy]
                        val r = (q shr 16) and 0xFF; val g = (q shr 8) and 0xFF; val b = q and 0xFF
                        val sum = r + g + b + 1.0
                        val br = ((r / sum) * bins * 1.5).toInt().coerceIn(0, bins - 1)
                        val bg = ((g / sum) * bins * 1.5).toInt().coerceIn(0, bins - 1)
                        h[br * bins + bg] += 1f; n++
                    }
                    x += step
                }
                k += 2
            }
            y += step
        }
        if (n > 0) for (i in h.indices) h[i] /= n
        return h
    }

    /** Shape sanity: real tabs/blanks are 15–45% of the side; junk (chairs, prints) is not. */
    fun isPlausible(p: Piece): Boolean {
        for (e in p.edges) {
            var out = 0.0; var inn = 0.0
            for (y in e.cy) { if (-y > out) out = -y; if (y > inn) inn = y }
            if (out > cfg.maxBulge * e.length || inn > cfg.maxBulge * e.length) return false
            // A tab or blank, not both.
            if (out > 0.25 * e.length && inn > 0.25 * e.length) return false
        }
        val sides = p.edges.map { it.length }
        return sides.max() < 2.2 * sides.min()
    }

    /** Builds a piece from a closed boundary polygon (any orientation, any point density). */
    fun analyzeContour(rawX: DoubleArray, rawY: DoubleArray, img: RgbImage?, id: Int): Piece? {
        if (rawX.size < 40) return null
        var xs = rawX
        var ys = rawY
        if (signedArea(xs, ys) < 0) { xs = xs.reversedArray(); ys = ys.reversedArray() }
        // Resample to roughly 1 point per pixel, then smooth away pixel staircase.
        val perimeter = perimeter(xs, ys)
        val n = perimeter.roundToInt().coerceIn(min(200, cfg.maxContourPoints), cfg.maxContourPoints)
        var (px, py) = resampleClosed(xs, ys, n)
        val smoothW = max(1, n / 200)
        repeat(2) { val s = smoothClosed(px, py, smoothW); px = s.first; py = s.second }

        debugContour = px to py
        val corners = findCorners(px, py) ?: return null
        val quality = lastCornerScore
        val area = signedArea(px, py)
        var cxSum = 0.0; var cySum = 0.0
        for (i in 0 until n) { cxSum += px[i]; cySum += py[i] }
        val centroidX = cxSum / n
        val centroidY = cySum / n
        var scale = 0.0
        for (e in 0 until 4) {
            val a = corners[e]; val b = corners[(e + 1) % 4]
            scale += hypot(px[b] - px[a], py[b] - py[a])
        }
        scale /= 4
        val edges = (0 until 4).map { e -> buildEdge(px, py, corners[e], corners[(e + 1) % 4], id, e, scale, img) }
        val palette = if (img != null && cfg.computePalette) palette(px, py, img) else FloatArray(0)
        val pieceEdges = if (palette.isNotEmpty()) edges.map { it.withPalette(palette) } else edges
        return Piece(id, px, py, corners, pieceEdges, centroidX, centroidY, scale, area, quality, palette)
    }

    private fun buildEdge(px: DoubleArray, py: DoubleArray, from: Int, to: Int, pieceId: Int, index: Int, scale: Double, img: RgbImage?): Edge {
        val n = px.size
        val count = ((to - from + n) % n) + 1
        val ex = DoubleArray(count) { px[(from + it) % n] }
        val ey = DoubleArray(count) { py[(from + it) % n] }
        val sx = ex[0]; val sy = ey[0]
        val dx = ex[count - 1] - sx; val dy = ey[count - 1] - sy
        val len = hypot(dx, dy)
        val ux = dx / len; val uy = dy / len
        // Rotate into canonical frame: x along the chord, y = left-hand normal (interior side).
        val canX = DoubleArray(count) { (ex[it] - sx) * ux + (ey[it] - sy) * uy }
        val canY = DoubleArray(count) { -(ex[it] - sx) * uy + (ey[it] - sy) * ux }
        val (rx, ry) = resampleOpen(canX, canY, cfg.edgeSamples)
        var maxOut = 0.0; var maxIn = 0.0
        for (y in ry) { if (-y > maxOut) maxOut = -y; if (y > maxIn) maxIn = y }
        val type = when {
            max(maxOut, maxIn) < cfg.flatThreshold * len -> EdgeType.FLAT
            maxOut > maxIn -> EdgeType.TAB
            else -> EdgeType.BLANK
        }
        // Colours just inside the boundary.
        val colors = FloatArray(cfg.colorSamples * 3)
        val (ix, iy) = resampleOpen(ex, ey, cfg.colorSamples + 2)
        if (img != null) {
            val inset = cfg.colorInset * scale
            val r = max(1, (scale * 0.02).roundToInt())
            for (k in 0 until cfg.colorSamples) {
                val j = k + 1
                var tx = ix[j + 1] - ix[j - 1]; var ty = iy[j + 1] - iy[j - 1]
                val tl = hypot(tx, ty).coerceAtLeast(1e-9); tx /= tl; ty /= tl
                img.samplePatch(ix[j] - ty * inset, iy[j] + tx * inset, r, colors, k * 3)
            }
        }
        // Mid point: the contour point farthest from the chord (tab tip), or chord midpoint for flat edges.
        var midX = sx + dx / 2; var midY = sy + dy / 2
        if (type != EdgeType.FLAT) {
            var best = -1.0
            for (i in 0 until count) {
                val d = abs(canY[i])
                if (d > best) { best = d; midX = ex[i]; midY = ey[i] }
            }
        }
        // Picture strip: a few rows at increasing inset, many samples along the edge.
        val rows = cfg.stripInsets.size
        val sl = cfg.stripSamples
        val strip = FloatArray(rows * sl * 3)
        if (img != null) {
            val (qx, qy) = resampleOpen(ex, ey, sl + 2)
            val r = max(0, (scale * 0.01).roundToInt())
            for (k in 0 until sl) {
                val j = k + 1
                var tx = qx[j + 1] - qx[j - 1]; var ty = qy[j + 1] - qy[j - 1]
                val tl = hypot(tx, ty).coerceAtLeast(1e-9); tx /= tl; ty /= tl
                for ((ri, ins) in cfg.stripInsets.withIndex()) {
                    val d = ins * scale
                    img.samplePatch(qx[j] - ty * d, qy[j] + tx * d, r, strip, (ri * sl + k) * 3)
                }
            }
        }
        return Edge(pieceId, index, type, len, rx, ry, colors, sx, sy, ex[count - 1], ey[count - 1], midX, midY, strip, rows, sl)
    }

    /**
     * Corner detection: candidates are convex points where the contour turns ~90 degrees between two
     * nearly straight arms; then the 4-combination forming the most rectangle-like quadrilateral wins.
     */
    private fun findCorners(px: DoubleArray, py: DoubleArray): IntArray? {
        val n = px.size
        val k = max(4, n / 28)
        val response = DoubleArray(n) { Double.MAX_VALUE }
        for (i in 0 until n) {
            val a = (i - k + n) % n; val b = (i + k) % n
            val v1x = px[a] - px[i]; val v1y = py[a] - py[i]
            val v2x = px[b] - px[i]; val v2y = py[b] - py[i]
            val l1 = hypot(v1x, v1y); val l2 = hypot(v2x, v2y)
            if (l1 < 1e-6 || l2 < 1e-6) continue
            val cross = v1x * v2y - v1y * v2x
            if (cross >= 0) continue // concave (or straight)
            val ang = Math.toDegrees(acos(((v1x * v2x + v1y * v2y) / (l1 * l2)).coerceIn(-1.0, 1.0)))
            if (ang < 45 || ang > 135) continue
            val s1 = armDeviation(px, py, i, a, -1, k) / l1
            val s2 = armDeviation(px, py, i, b, 1, k) / l2
            // Arms must be roughly as long as their arc (straight), not curled like a tab knob.
            response[i] = abs(ang - 90) / 90 + 4 * (s1 + s2)
        }
        // Non-maximum suppression.
        val cand = ArrayList<Int>()
        val win = max(2, k / 2)
        for (i in 0 until n) {
            val r = response[i]
            if (r > 1.2) continue
            var isMin = true
            for (d in -win..win) {
                if (d == 0) continue
                val j = (i + d + n) % n
                if (response[j] < r || (response[j] == r && j < i)) { isMin = false; break }
            }
            if (isMin) cand.add(i)
        }
        if (cand.size < 4) return null
        cand.sortBy { response[it] }
        val c = cand.take(14).sorted()
        debugCandidates = c
        val contourArea = signedArea(px, py)
        var bestScore = Double.NEGATIVE_INFINITY
        var best: IntArray? = null
        val m = c.size
        val minGap = n / 12
        val qx = DoubleArray(4); val qy = DoubleArray(4)
        for (i0 in 0 until m) for (i1 in i0 + 1 until m) for (i2 in i1 + 1 until m) for (i3 in i2 + 1 until m) {
            val idx = intArrayOf(c[i0], c[i1], c[i2], c[i3])
            var ok = true
            for (e in 0 until 4) {
                val gap = (idx[(e + 1) % 4] - idx[e] + n) % n
                if (gap < minGap) { ok = false; break }
            }
            if (!ok) continue
            for (e in 0 until 4) { qx[e] = px[idx[e]]; qy[e] = py[idx[e]] }
            val qa = signedArea(qx, qy)
            if (qa <= 0) continue
            var angDev = 0.0
            val side = DoubleArray(4)
            for (e in 0 until 4) {
                val p = (e + 3) % 4; val nx = (e + 1) % 4
                val ax = qx[p] - qx[e]; val ay = qy[p] - qy[e]
                val bx = qx[nx] - qx[e]; val by = qy[nx] - qy[e]
                val ang = Math.toDegrees(acos(((ax * bx + ay * by) / (hypot(ax, ay) * hypot(bx, by))).coerceIn(-1.0, 1.0)))
                angDev += abs(ang - 90) / 90
                side[e] = hypot(bx, by)
            }
            angDev /= 4
            val sideDev = (abs(1 - side[0] / side[2]) + abs(1 - side[1] / side[3])) / 2
            val aspect = min(side[0] + side[2], side[1] + side[3]) / max(side[0] + side[2], side[1] + side[3])
            var resp = 0.0
            for (e in 0 until 4) resp += response[idx[e]]
            val score = qa / contourArea - 2.0 * angDev - 1.0 * sideDev - 0.5 * (1 - aspect) - 0.3 * resp / 4
            if (score > bestScore) { bestScore = score; best = idx }
        }
        val b = best ?: return null
        lastCornerScore = bestScore
        // Sanity: reject blobs (touching pieces, hands, debris) that don't look like one piece.
        for (e in 0 until 4) { qx[e] = px[b[e]]; qy[e] = py[b[e]] }
        val ratio = signedArea(qx, qy) / contourArea
        if (bestScore < -0.2 || ratio < 0.45 || ratio > 1.6) return null
        return b
    }

    /** Max distance of the contour points between i and j (walking in [dir]) from the chord i–j. */
    private fun armDeviation(px: DoubleArray, py: DoubleArray, i: Int, j: Int, dir: Int, k: Int): Double {
        val n = px.size
        val dx = px[j] - px[i]; val dy = py[j] - py[i]
        val len = hypot(dx, dy)
        var worst = 0.0
        for (s in 1 until k) {
            val t = (i + dir * s + n) % n
            val d = abs((px[t] - px[i]) * dy - (py[t] - py[i]) * dx) / len
            if (d > worst) worst = d
        }
        return worst
    }

    companion object {
        const val PALETTE_BINS = 8

        /** 1 - Bhattacharyya coefficient: 0 = identical colour mix, 1 = no overlap. */
        fun paletteDistance(a: FloatArray, b: FloatArray): Double {
            if (a.isEmpty() || b.size != a.size) return 0.0
            var bc = 0.0
            for (i in a.indices) bc += kotlin.math.sqrt((a[i] * b[i]).toDouble())
            return (1 - bc).coerceIn(0.0, 1.0)
        }

        private val DX = intArrayOf(1, 1, 0, -1, -1, -1, 0, 1)
        private val DY = intArrayOf(0, 1, 1, 1, 0, -1, -1, -1)

        private fun dirOf(dx: Int, dy: Int): Int {
            for (d in 0 until 8) if (DX[d] == dx && DY[d] == dy) return d
            error("not a neighbour")
        }

        /** Moore-neighbour boundary tracing of one labelled component. Returns pixel-centre coordinates. */
        fun traceContour(labels: IntArray, w: Int, h: Int, id: Int, sx: Int, sy: Int): Pair<DoubleArray, DoubleArray> {
            fun inside(x: Int, y: Int) = x in 0 until w && y in 0 until h && labels[y * w + x] == id
            val xs = ArrayList<Int>()
            val ys = ArrayList<Int>()
            var cx = sx; var cy = sy
            var back = 4 // came from the west
            var firstDir = -1
            val limit = 4 * w * h
            var steps = 0
            while (steps++ < limit) {
                var moved = -1
                for (k in 1..8) {
                    val d = (back + k) % 8
                    if (inside(cx + DX[d], cy + DY[d])) { moved = d; break }
                }
                if (moved < 0) { xs.add(cx); ys.add(cy); break } // isolated pixel
                if (cx == sx && cy == sy) {
                    if (firstDir == moved) break
                    if (firstDir < 0) firstDir = moved
                }
                xs.add(cx); ys.add(cy)
                val bd = (moved + 7) % 8
                val bx = cx + DX[bd]; val by = cy + DY[bd]
                cx += DX[moved]; cy += DY[moved]
                back = dirOf(bx - cx, by - cy)
            }
            return DoubleArray(xs.size) { xs[it] + 0.5 } to DoubleArray(ys.size) { ys[it] + 0.5 }
        }

        fun signedArea(xs: DoubleArray, ys: DoubleArray): Double {
            var a = 0.0
            val n = xs.size
            for (i in 0 until n) {
                val j = (i + 1) % n
                a += xs[i] * ys[j] - xs[j] * ys[i]
            }
            return a / 2
        }

        fun perimeter(xs: DoubleArray, ys: DoubleArray): Double {
            var p = 0.0
            for (i in xs.indices) { val j = (i + 1) % xs.size; p += hypot(xs[j] - xs[i], ys[j] - ys[i]) }
            return p
        }

        fun resampleClosed(xs: DoubleArray, ys: DoubleArray, n: Int): Pair<DoubleArray, DoubleArray> {
            val cx = xs + xs[0]
            val cy = ys + ys[0]
            val (rx, ry) = resampleOpen(cx, cy, n + 1)
            return rx.copyOf(n) to ry.copyOf(n)
        }

        /** Uniform arc-length resampling of an open polyline to [n] points including both endpoints. */
        fun resampleOpen(xs: DoubleArray, ys: DoubleArray, n: Int): Pair<DoubleArray, DoubleArray> {
            val m = xs.size
            val cum = DoubleArray(m)
            for (i in 1 until m) cum[i] = cum[i - 1] + hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
            val total = cum[m - 1]
            val ox = DoubleArray(n); val oy = DoubleArray(n)
            var seg = 0
            for (k in 0 until n) {
                val target = if (n == 1) 0.0 else total * k / (n - 1)
                while (seg < m - 2 && cum[seg + 1] < target) seg++
                val segLen = cum[seg + 1] - cum[seg]
                val t = if (segLen < 1e-12) 0.0 else ((target - cum[seg]) / segLen).coerceIn(0.0, 1.0)
                ox[k] = xs[seg] + (xs[seg + 1] - xs[seg]) * t
                oy[k] = ys[seg] + (ys[seg + 1] - ys[seg]) * t
            }
            return ox to oy
        }

        fun smoothClosed(xs: DoubleArray, ys: DoubleArray, w: Int): Pair<DoubleArray, DoubleArray> {
            val n = xs.size
            val ox = DoubleArray(n); val oy = DoubleArray(n)
            val norm = 2.0 * w + 1
            for (i in 0 until n) {
                var sx = 0.0; var sy = 0.0
                for (d in -w..w) { val j = (i + d + n) % n; sx += xs[j]; sy += ys[j] }
                ox[i] = sx / norm; oy[i] = sy / norm
            }
            return ox to oy
        }

        @Suppress("unused")
        fun rms(a: DoubleArray): Double = sqrt(a.sumOf { it * it } / a.size)
    }
}
