package puzzlemaster.core.synth

import puzzlemaster.core.RgbImage
import kotlin.math.PI
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.random.Random

/**
 * Generates a realistic jigsaw puzzle: a jittered grid of corners, classic knob-shaped tabs with
 * random size/position/skew on every interior edge, and a colourful procedural picture.
 * Used for functional tests; no real images required.
 */
/**
 * @param standardDie like commercial puzzles: every tab is one of a few fixed shapes on a regular
 *   grid, so shape alone cannot tell which pieces are neighbours.
 */
class SyntheticPuzzle(val cols: Int, val rows: Int, val pieceSize: Double = 100.0, seed: Long = 1, val standardDie: Boolean = false) {
    private val rnd = Random(seed)
    private val cornerX = Array(rows + 1) { DoubleArray(cols + 1) }
    private val cornerY = Array(rows + 1) { DoubleArray(cols + 1) }

    /** Horizontal edges: [r][c] goes from corner (r,c) to (r,c+1). */
    private val hEdges: Array<Array<EdgeCurve>>
    /** Vertical edges: [r][c] goes from corner (r,c) to (r+1,c). */
    private val vEdges: Array<Array<EdgeCurve>>

    private val picture = Picture(rnd, cols * pieceSize, rows * pieceSize)

    class EdgeCurve(val xs: DoubleArray, val ys: DoubleArray)

    init {
        val j = (if (standardDie) 0.01 else 0.06) * pieceSize
        for (r in 0..rows) for (c in 0..cols) {
            val jx = if (c == 0 || c == cols) 0.0 else rnd.nextDouble(-j, j)
            val jy = if (r == 0 || r == rows) 0.0 else rnd.nextDouble(-j, j)
            cornerX[r][c] = c * pieceSize + jx
            cornerY[r][c] = r * pieceSize + jy
        }
        hEdges = Array(rows + 1) { r -> Array(cols) { c -> curve(r, c, r, c + 1, r == 0 || r == rows) } }
        vEdges = Array(rows) { r -> Array(cols + 1) { c -> curve(r, c, r + 1, c, c == 0 || c == cols) } }
    }

    private fun curve(r0: Int, c0: Int, r1: Int, c1: Int, border: Boolean): EdgeCurve {
        val ax = cornerX[r0][c0]; val ay = cornerY[r0][c0]
        val bx = cornerX[r1][c1]; val by = cornerY[r1][c1]
        if (border) return EdgeCurve(doubleArrayOf(ax, bx), doubleArrayOf(ay, by))
        // Classic jigsaw edge as 3 cubic Beziers in (u along edge, v across edge) coordinates.
        val flip = if (rnd.nextBoolean()) 1.0 else -1.0
        val t: Double; val a: Double; val b: Double; val c: Double; val d: Double; val e: Double
        if (standardDie) {
            // Two die shapes, tiny manufacturing noise.
            val v = rnd.nextInt(2)
            t = if (v == 0) 0.10 else 0.11
            a = 0.0; b = if (v == 0) 0.0 else 0.015; c = 0.0; d = 0.0; e = 0.0
        } else {
            t = rnd.nextDouble(0.085, 0.12)
            val jit = 0.04
            a = rnd.nextDouble(-jit, jit); b = rnd.nextDouble(-jit, jit); c = rnd.nextDouble(-jit, jit)
            d = rnd.nextDouble(-jit, jit); e = rnd.nextDouble(-jit, jit)
        }
        val p = arrayOf(
            0.0 to 0.0, 0.2 to a, 0.5 + b + d to -t + c, 0.5 - t + b to t + c,
            0.5 - 2 * t + b - d to 3 * t + c, 0.5 + 2 * t + b - d to 3 * t + c, 0.5 + t + b to t + c,
            0.5 + b + d to -t + c, 0.8 to e, 1.0 to 0.0,
        )
        val us = ArrayList<Double>(); val vs = ArrayList<Double>()
        val seg = 40
        for (s in 0 until 3) {
            val q0 = p[s * 3]; val q1 = p[s * 3 + 1]; val q2 = p[s * 3 + 2]; val q3 = p[s * 3 + 3]
            for (k in (if (s == 0) 0 else 1)..seg) {
                val tt = k.toDouble() / seg
                val m = 1 - tt
                val w0 = m * m * m; val w1 = 3 * m * m * tt; val w2 = 3 * m * tt * tt; val w3 = tt * tt * tt
                us.add(w0 * q0.first + w1 * q1.first + w2 * q2.first + w3 * q3.first)
                vs.add(w0 * q0.second + w1 * q1.second + w2 * q2.second + w3 * q3.second)
            }
        }
        val dx = bx - ax; val dy = by - ay
        // Normal (−dy, dx) has the same length as the edge.
        val xs = DoubleArray(us.size) { ax + us[it] * dx - flip * vs[it] * dy }
        val ys = DoubleArray(us.size) { ay + us[it] * dy + flip * vs[it] * dx }
        return EdgeCurve(xs, ys)
    }

    /** Piece outline in puzzle coordinates: top (TL→TR), right, bottom, left. Clockwise on screen (y down). */
    fun outline(r: Int, c: Int): Pair<DoubleArray, DoubleArray> {
        val xs = ArrayList<Double>(); val ys = ArrayList<Double>()
        fun add(e: EdgeCurve, reversed: Boolean) {
            val n = e.xs.size
            for (k in 0 until n - 1) { // drop last point: it is the next edge's first
                val i = if (reversed) n - 1 - k else k
                xs.add(e.xs[i]); ys.add(e.ys[i])
            }
        }
        add(hEdges[r][c], false)
        add(vEdges[r][c + 1], false)
        add(hEdges[r + 1][c], true)
        add(vEdges[r][c], true)
        return xs.toDoubleArray() to ys.toDoubleArray()
    }

    /** Points of one side (0=top,1=right,2=bottom,3=left) in clockwise outline order. */
    fun side(r: Int, c: Int, s: Int): Pair<DoubleArray, DoubleArray> {
        val (e, rev) = when (s) {
            0 -> hEdges[r][c] to false
            1 -> vEdges[r][c + 1] to false
            2 -> hEdges[r + 1][c] to true
            else -> vEdges[r][c] to true
        }
        return if (rev) e.xs.reversedArray() to e.ys.reversedArray() else e.xs to e.ys
    }

    /** Corners TL, TR, BR, BL in puzzle coordinates. */
    fun corners(r: Int, c: Int): List<Pair<Double, Double>> = listOf(
        cornerX[r][c] to cornerY[r][c], cornerX[r][c + 1] to cornerY[r][c + 1],
        cornerX[r + 1][c + 1] to cornerY[r + 1][c + 1], cornerX[r + 1][c] to cornerY[r + 1][c],
    )

    fun color(x: Double, y: Double): Int = picture.color(x, y)

    /** Placement of one piece in a target image: rotate by [angle] about the piece centre, scale, then translate. */
    class Placement(val r: Int, val c: Int, val angle: Double, val scale: Double, val tx: Double, val ty: Double, val cx: Double, val cy: Double) {
        fun toImage(x: Double, y: Double): Pair<Double, Double> {
            val dx = (x - cx) * scale; val dy = (y - cy) * scale
            return (cos(angle) * dx - sin(angle) * dy + tx) to (sin(angle) * dx + cos(angle) * dy + ty)
        }
        fun toPuzzle(x: Double, y: Double): Pair<Double, Double> {
            val dx = x - tx; val dy = y - ty
            val ux = cos(angle) * dx + sin(angle) * dy; val uy = -sin(angle) * dx + cos(angle) * dy
            return (ux / scale + cx) to (uy / scale + cy)
        }
    }

    fun pieceCenter(r: Int, c: Int): Pair<Double, Double> {
        val k = corners(r, c)
        return k.sumOf { it.first } / 4 to k.sumOf { it.second } / 4
    }

    /**
     * Draws the piece into [img] with 2x2 supersampling (anti-aliased boundary), lighting [gain]
     * and per-pixel noise of amplitude [noise].
     */
    fun render(img: RgbImage, pl: Placement, rnd: Random, gain: Double = 1.0, noise: Int = 4) {
        val (ox, oy) = outline(pl.r, pl.c)
        val n = ox.size
        val ix = DoubleArray(n); val iy = DoubleArray(n)
        for (i in 0 until n) { val (a, b) = pl.toImage(ox[i], oy[i]); ix[i] = a; iy[i] = b }
        val minY = max(0, floor(iy.min()).toInt()); val maxY = min(img.height - 1, ceil(iy.max()).toInt())
        val minX = max(0, floor(ix.min()).toInt()); val maxX = min(img.width - 1, ceil(ix.max()).toInt())
        val w = maxX - minX + 1
        val cover = IntArray(w)
        val crossings = DoubleArray(64)
        for (y in minY..maxY) {
            java.util.Arrays.fill(cover, 0)
            for (sub in 0 until 2) {
                val sy = y + 0.25 + 0.5 * sub
                var nc = 0
                var cr = crossings
                for (i in 0 until n) {
                    val j = if (i == n - 1) 0 else i + 1
                    val y0 = iy[i]; val y1 = iy[j]
                    if ((y0 <= sy && y1 > sy) || (y1 <= sy && y0 > sy)) {
                        if (nc == cr.size) cr = cr.copyOf(nc * 2)
                        cr[nc++] = ix[i] + (sy - y0) / (y1 - y0) * (ix[j] - ix[i])
                    }
                }
                java.util.Arrays.sort(cr, 0, nc)
                var k = 0
                while (k + 1 < nc) {
                    val xa = cr[k]; val xb = cr[k + 1]
                    for (sxI in 0 until 2) {
                        // sample x positions x + 0.25, x + 0.75
                        val startX = max(minX, ceil(xa - 0.25 - 0.5 * sxI).toInt())
                        var x = startX
                        while (x <= maxX) {
                            val sx = x + 0.25 + 0.5 * sxI
                            if (sx >= xb) break
                            if (sx >= xa) cover[x - minX]++
                            x++
                        }
                    }
                    k += 2
                }
            }
            for (x in minX..maxX) {
                val cv = cover[x - minX]
                if (cv == 0) continue
                val (px, py) = pl.toPuzzle(x + 0.5, y + 0.5)
                val col = color(px, py)
                val bg = img[x, y]
                val f = cv / 4.0
                fun ch(sh: Int): Int {
                    val fg = ((col shr sh) and 0xFF) * gain + rnd.nextInt(-noise, noise + 1)
                    return (fg * f + ((bg shr sh) and 0xFF) * (1 - f)).toInt()
                }
                img[x, y] = RgbImage.argb(ch(16), ch(8), ch(0))
            }
        }
    }

    /** Fills an image with a plain background with some noise and a gentle lighting gradient. */
    fun background(w: Int, h: Int, color: Int, rnd: Random, noise: Int = 5): RgbImage {
        val img = RgbImage(w, h)
        val r = (color shr 16) and 0xFF; val g = (color shr 8) and 0xFF; val b = color and 0xFF
        for (y in 0 until h) for (x in 0 until w) {
            val l = 1.0 + 0.08 * (x.toDouble() / w - 0.5) + 0.05 * (y.toDouble() / h - 0.5)
            img[x, y] = RgbImage.argb(
                (r * l).toInt() + rnd.nextInt(-noise, noise + 1),
                (g * l).toInt() + rnd.nextInt(-noise, noise + 1),
                (b * l).toInt() + rnd.nextInt(-noise, noise + 1),
            )
        }
        return img
    }

    /** Procedural "photo": smooth colour gradients plus blobs and stripes, so neighbouring pieces share colours. */
    private class Picture(rnd: Random, val w: Double, val h: Double) {
        private val waves = List(12) {
            doubleArrayOf(
                rnd.nextDouble(0.5, 6.0) * 2 * PI / w * (if (rnd.nextBoolean()) 1 else -1),
                rnd.nextDouble(0.5, 6.0) * 2 * PI / h * (if (rnd.nextBoolean()) 1 else -1),
                rnd.nextDouble(0.0, 2 * PI),
                rnd.nextDouble(12.0, 35.0),
                rnd.nextInt(3).toDouble(),
            )
        }
        private val blobs = List(max(8, (w * h / 40000).toInt())) {
            doubleArrayOf(rnd.nextDouble(w), rnd.nextDouble(h), rnd.nextDouble(30.0, 250.0), rnd.nextDouble(-90.0, 90.0), rnd.nextDouble(-90.0, 90.0), rnd.nextDouble(-90.0, 90.0))
        }
        private val grid = BlobGrid(blobs, w, h)

        fun color(x: Double, y: Double): Int {
            val c = doubleArrayOf(128.0, 120.0, 110.0)
            for (wv in waves) c[wv[4].toInt()] += wv[3] * sin(wv[0] * x + wv[1] * y + wv[2])
            for (b in grid.near(x, y)) {
                val d = hypot(x - b[0], y - b[1]) / b[2]
                if (d < 1) { val f = (1 - d * d); c[0] += b[3] * f; c[1] += b[4] * f; c[2] += b[5] * f }
            }
            // Fine texture (like printed detail) so colour sampling is not trivially smooth.
            val tex = 12 * sin(x * 0.21) * sin(y * 0.17)
            return RgbImage.argb((c[0] + tex).toInt(), (c[1] - tex).toInt(), (c[2] + tex / 2).toInt())
        }
    }

    private class BlobGrid(blobs: List<DoubleArray>, w: Double, h: Double) {
        private val cell = 250.0
        private val gw = ceil(w / cell).toInt() + 1
        private val gh = ceil(h / cell).toInt() + 1
        private val cells = Array(gw * gh) { ArrayList<DoubleArray>() }
        init {
            for (b in blobs) {
                val x0 = max(0, ((b[0] - b[2]) / cell).toInt()); val x1 = min(gw - 1, ((b[0] + b[2]) / cell).toInt())
                val y0 = max(0, ((b[1] - b[2]) / cell).toInt()); val y1 = min(gh - 1, ((b[1] + b[2]) / cell).toInt())
                for (gy in y0..y1) for (gx in x0..x1) cells[gy * gw + gx].add(b)
            }
        }
        private val empty = ArrayList<DoubleArray>()
        fun near(x: Double, y: Double): List<DoubleArray> {
            val gx = (x / cell).toInt(); val gy = (y / cell).toInt()
            if (gx < 0 || gy < 0 || gx >= gw || gy >= gh) return empty
            return cells[gy * gw + gx]
        }
    }
}
