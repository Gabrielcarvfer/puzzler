package puzzlemaster.core

import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.DMatch
import org.opencv.core.KeyPoint
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.features2d.FlannBasedMatcher
import org.opencv.features2d.SIFT
import org.opencv.imgproc.Imgproc
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/**
 * Finds neighbours through the box picture: every catalogued piece is located on the picture by its
 * print (SIFT features, one rigid placement), the cutting grid is fitted to those positions (the box
 * print need not show exactly the puzzle's area), and pieces in neighbouring cells are pairs.
 * Edge shape is not used to decide: measured on real photos it does not separate neighbours from
 * other pieces of this die.
 */
class BoxPlacer(val poster: Mat, val cols: Int = 40, val rows: Int = 25) {
    /** [sideDirs]: where the entry's four sides point in the view that was matched ([theta] rotates that view onto the box). */
    class Hit(val entry: PieceCatalogue.Entry, val tx: Double, val ty: Double, val theta: Double, val inliers: Int, val sideDirs: Array<DoubleArray>)
    class Grid(val cw: Double, val ch: Double, val offX: Double, val offY: Double, val fitted: Boolean)
    class Placed(val hit: Hit, val col: Int, val row: Int, /** entry side index facing north, east, south, west */ val facing: IntArray)
    class Pair(
        val a: PieceCatalogue.Entry, val sideA: Int, val b: PieceCatalogue.Entry, val sideB: Int,
        val colA: Int, val rowA: Int, val colB: Int, val rowB: Int,
        /** True when the two facing sides are a tab against a blank, as measured. */
        val sidesAgree: Boolean,
        val support: Int,
    )

    private val pk: Array<KeyPoint>
    private val flann = FlannBasedMatcher.create()
    /** Expected size of one piece on the poster (pixels), for sanity-checking a placement. */
    private val nominal = (poster.cols().toDouble() / cols + poster.rows().toDouble() / rows) / 2

    init {
        val gray = Mat(); Imgproc.cvtColor(poster, gray, Imgproc.COLOR_BGR2GRAY)
        val kp = MatOfKeyPoint(); val d = Mat()
        SIFT.create(0, 3, 0.02, 10.0, 1.6).detectAndCompute(gray, Mat(), kp, d)
        pk = kp.toArray()
        flann.add(listOf(d)); flann.train()
    }

    val posterFeatures: Int get() = pk.size
    /** Expected size of one piece on the poster (pixels). */
    val nominalPiece: Double get() = nominal
    fun posterPoint(i: Int): Point = pk[i].pt

    /** Ratio-test matches of [desc] (query) against the poster's features (train). */
    fun matchFeatures(desc: Mat, ratio: Float): List<DMatch> {
        if (desc.rows() < 2) return emptyList()
        // The FLANN matcher object is shared: one search at a time.
        val knn = ArrayList<MatOfDMatch>(); synchronized(flann) { flann.knnMatch(desc, knn, 2) }
        val good = ArrayList<DMatch>()
        for (k in knn) { val m = k.toArray(); if (m.size == 2 && m[0].distance < ratio * m[1].distance) good.add(m[0]) }
        return good
    }

    /**
     * Where this piece's print appears on the box picture, or null when it can't be told. Every stored
     * view of the piece is tried and the best-supported placement wins.
     */
    fun locate(e: PieceCatalogue.Entry, minInliers: Int = 4, ratio: Float = 0.85f, firstViewOnly: Boolean = false): Hit? {
        var best: Hit? = null
        for (v in if (firstViewOnly) e.views().take(1) else e.views()) {
            val h = locateView(e, v, minInliers, ratio) ?: continue
            if (best == null || h.inliers > best.inliers) best = h
        }
        return best
    }

    private fun locateView(e: PieceCatalogue.Entry, v: PieceCatalogue.View, minInliers: Int, ratio: Float): Hit? {
        if (v.descriptors.rows() < 6) return null
        val knn = ArrayList<MatOfDMatch>(); synchronized(flann) { flann.knnMatch(v.descriptors, knn, 2) }
        val good = ArrayList<DMatch>()
        for (k in knn) { val m = k.toArray(); if (m.size == 2 && m[0].distance < ratio * m[1].distance) good.add(m[0]) }
        if (good.size < minInliers) return null
        val from = MatOfPoint2f(*good.map { v.keypoints[it.queryIdx] }.toTypedArray())
        val to = MatOfPoint2f(*good.map { pk[it.trainIdx].pt }.toTypedArray())
        val inl = Mat()
        val a = Calib3d.estimateAffinePartial2D(from, to, inl, Calib3d.RANSAC, 4.0, 2000, 0.99, 10)
        if (a.empty()) return null
        val n = Core.countNonZero(inl)
        val sc = hypot(a.get(0, 0)[0], a.get(1, 0)[0])
        if (n < minInliers || sc < nominal * 0.75 || sc > nominal * 1.35) return null
        return Hit(e, a.get(0, 2)[0], a.get(1, 2)[0], atan2(a.get(1, 0)[0], a.get(0, 0)[0]), n, v.sideDirs)
    }

    /** Cell size and offset that put the located pieces' centres on cell centres. */
    fun fitGrid(hits: List<Hit>): Grid {
        val pw = poster.cols().toDouble(); val ph = poster.rows().toDouble()
        if (hits.size < 10) return Grid(pw / cols, ph / rows, 0.0, 0.0, false)
        fun fit(v: List<Double>, nom: Double): DoubleArray {
            var best = doubleArrayOf(nom, 0.0); var bs = -1e9
            var p = nom * 0.95
            while (p <= nom * 1.05) {
                var f = 0.0
                while (f < 1.0) {
                    var sc = 0.0
                    for (x in v) sc += cos(2 * PI * (x / p - f - 0.5))
                    if (sc > bs) { bs = sc; best = doubleArrayOf(p, f) }
                    f += 0.02
                }
                p += nom * 0.002
            }
            return best
        }
        val fx = fit(hits.map { it.tx }, pw / cols); val fy = fit(hits.map { it.ty }, ph / rows)
        // Phase as an offset of at most half a cell either way.
        fun off(f: Double, p: Double) = (if (f > 0.5) f - 1 else f) * p
        return Grid(fx[0], fy[0], off(fx[1], fx[0]), off(fy[1], fy[0]), true)
    }

    /** Cell and orientation of each located piece; when two pieces claim a cell, the better-supported one keeps it. */
    fun place(hits: List<Hit>, g: Grid): List<Placed> {
        val dirs = arrayOf(doubleArrayOf(0.0, -1.0), doubleArrayOf(1.0, 0.0), doubleArrayOf(0.0, 1.0), doubleArrayOf(-1.0, 0.0))
        val byCell = HashMap<Int, Placed>()
        for (h in hits.sortedByDescending { it.inliers }) {
            val col = Math.floor((h.tx - g.offX) / g.cw).toInt(); val row = Math.floor((h.ty - g.offY) / g.ch).toInt()
            if (col !in 0 until cols || row !in 0 until rows) continue
            val e = h.entry
            val facing = IntArray(4) { d ->
                (0 until 4).maxBy { s ->
                    val x = cos(h.theta) * h.sideDirs[s][0] - sin(h.theta) * h.sideDirs[s][1]
                    val y = sin(h.theta) * h.sideDirs[s][0] + cos(h.theta) * h.sideDirs[s][1]
                    x * dirs[d][0] + y * dirs[d][1]
                }
            }
            if (facing.toSet().size != 4) continue
            byCell.putIfAbsent(row * cols + col, Placed(h, col, row, facing))
        }
        return byCell.values.toList()
    }

    /** Pieces placed in neighbouring cells. Pairs whose facing sides agree come first, then by support. */
    fun pairs(placed: List<Placed>): List<Pair> {
        val out = ArrayList<Pair>()
        for (i in placed.indices) for (j in i + 1 until placed.size) {
            val a = placed[i]; val b = placed[j]
            val dc = b.col - a.col; val dr = b.row - a.row
            if (abs(dc) + abs(dr) != 1) continue
            val d = if (dr == -1) 0 else if (dc == 1) 1 else if (dr == 1) 2 else 3
            val sa = a.facing[d]; val sb = b.facing[(d + 2) % 4]
            val ta = a.hit.entry.sides[sa].type; val tb = b.hit.entry.sides[sb].type
            val agree = (ta == EdgeType.TAB && tb == EdgeType.BLANK) || (ta == EdgeType.BLANK && tb == EdgeType.TAB)
            out.add(Pair(a.hit.entry, sa, b.hit.entry, sb, a.col, a.row, b.col, b.row, agree, minOf(a.hit.inliers, b.hit.inliers)))
        }
        return out.sortedWith(compareByDescending<Pair> { it.sidesAgree }.thenByDescending { it.support })
    }

    /** A patch of the box picture ([size] px square, 5 x 5 cells) with A's cell in green and B's in magenta. */
    fun patch(p: Pair, g: Grid, size: Int): Mat = patch(p.colA, p.rowA, p.colB, p.rowB, g, size)

    fun patch(colA: Int, rowA: Int, colB: Int, rowB: Int, g: Grid, size: Int): Mat {
        val wpx = (5 * g.cw).toInt(); val hpx = (5 * g.ch).toInt()
        val x = (g.offX + (minOf(colA, colB) - 2) * g.cw).toInt().coerceIn(0, poster.cols() - wpx - 1)
        val y = (g.offY + (minOf(rowA, rowB) - 2) * g.ch).toInt().coerceIn(0, poster.rows() - hpx - 1)
        val out = Mat(); Imgproc.resize(poster.submat(Rect(x, y, wpx, hpx)), out, Size(size.toDouble(), size.toDouble()))
        val kx = size / (5 * g.cw); val ky = size / (5 * g.ch)
        fun box(col: Int, row: Int, c: Scalar) = Imgproc.rectangle(out,
            Point((g.offX + col * g.cw - x) * kx, (g.offY + row * g.ch - y) * ky),
            Point((g.offX + (col + 1) * g.cw - x) * kx, (g.offY + (row + 1) * g.ch - y) * ky), c, 3)
        box(colA, rowA, Scalar(255.0, 232.0, 60.0)); box(colB, rowB, Scalar(26.0, 122.0, 255.0)) // A cyan, B orange (BGR)
        return out
    }
}
