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
import org.opencv.core.Size
import org.opencv.features2d.FlannBasedMatcher
import org.opencv.features2d.SIFT
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.ln

/**
 * Finds where the pieces in a camera frame belong on the box picture, without segmenting them.
 *
 * This is the standard recipe for recognising known planar pictures in clutter: SIFT features of the
 * frame are matched against the box picture, and matches are grouped into pieces by geometric
 * consistency: all matches of one piece agree on a single similarity transform (position, rotation,
 * scale), and all pieces in a frame share about the same scale. Touching or overlapping pieces and
 * a tilted camera don't matter; plain pieces (sky) simply produce no matches.
 */
class FeatureLocator(posterBgr: Mat, val cols: Int, val rows: Int) {
    data class Config(
        val ratio: Float = 0.8f,
        val minInliers: Int = 5,
        val ransacThreshold: Double = 4.0,
        /** Allowed deviation of a piece's scale from the frame's median (log ratio). */
        val scaleTolerance: Double = 0.18,
        /** Minimum picture agreement (normalised correlation) of the chosen cell. */
        val minNcc: Double = 0.2,
        /** Max SIFT features per frame (0 = unlimited). */
        val maxFeatures: Int = 0,
        /** Inliers a group needs to take part in estimating the frame's scale. */
        val scaleInliers: Int = 8,
    )

    class Located(
        val col: Int, val row: Int,
        val inliers: Int,
        /** Picture agreement of the chosen cell, -1..1. */
        val ncc: Double,
        /** Poster -> frame similarity (a, b, tx, c, d, ty). */
        val a: DoubleArray,
        /** The piece's cell outline in the frame, and its centre. */
        val quad: Array<Point>,
        val cx: Double, val cy: Double,
        val angle: Double, val scale: Double,
    )

    class Pair(val a: Located, val b: Located)

    class Result(val pieces: List<Located>, val pairs: List<Pair>, val features: Int, val matches: Int, val detectMs: Long, val matchMs: Long, val groupMs: Long)

    val cell: Double = posterBgr.cols().toDouble() / cols
    private val posterGray = Mat().also { Imgproc.cvtColor(posterBgr, it, Imgproc.COLOR_BGR2GRAY) }
    private val pk: Array<KeyPoint>
    private val matcher = FlannBasedMatcher.create()
    /** Poster at [SMALL] px per cell, padded by one cell, for the per-cell picture check. */
    private val small: Mat

    init {
        val kp = MatOfKeyPoint(); val d = Mat()
        SIFT.create(0, 3, 0.02, 10.0, 1.6).detectAndCompute(posterGray, Mat(), kp, d)
        pk = kp.toArray()
        matcher.add(listOf(d)); matcher.train()
        val s = Mat(); Imgproc.resize(posterGray, s, Size(cols * SMALL.toDouble(), rows * SMALL.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        small = Mat(); Core.copyMakeBorder(s, small, SMALL, SMALL, SMALL, SMALL, Core.BORDER_REPLICATE)
    }

    val posterFeatures: Int get() = pk.size

    fun locate(frameBgr: Mat, cfg: Config = Config()): Result {
        val t0 = System.currentTimeMillis()
        val gray = Mat(); Imgproc.cvtColor(frameBgr, gray, Imgproc.COLOR_BGR2GRAY)
        val fkM = MatOfKeyPoint(); val fd = Mat()
        SIFT.create(cfg.maxFeatures, 3, 0.02, 10.0, 1.6).detectAndCompute(gray, Mat(), fkM, fd)
        val fk = fkM.toArray()
        val t1 = System.currentTimeMillis()
        if (fk.size < 8) return Result(emptyList(), emptyList(), fk.size, 0, t1 - t0, 0, 0)
        val knn = ArrayList<MatOfDMatch>()
        matcher.knnMatch(fd, knn, 2)
        val good = ArrayList<DMatch>()
        for (k in knn) { val m = k.toArray(); if (m.size == 2 && m[0].distance < cfg.ratio * m[1].distance) good.add(m[0]) }
        val t2 = System.currentTimeMillis()
        val n = good.size
        val fx = DoubleArray(n) { fk[good[it].queryIdx].pt.x }; val fy = DoubleArray(n) { fk[good[it].queryIdx].pt.y }
        val px = DoubleArray(n) { pk[good[it].trainIdx].pt.x }; val py = DoubleArray(n) { pk[good[it].trainIdx].pt.y }
        val sc = DoubleArray(n) { fk[good[it].queryIdx].size.toDouble() / pk[good[it].trainIdx].size }
        val used = BooleanArray(n)
        val order = (0 until n).sortedBy { good[it].distance }

        class Group(val a: DoubleArray, val idx: IntArray, val scale: Double, val angle: Double)
        val groups = ArrayList<Group>()
        for (i in order) {
            if (used[i]) continue
            var near = (0 until n).filter { !used[it] && hypot(px[it] - px[i], py[it] - py[i]) < cell }
            if (near.size < 4) continue
            val s = near.map { sc[it] }.sorted()[near.size / 2]
            near = near.filter { hypot(fx[it] - fx[i], fy[it] - fy[i]) < cell * s * 1.6 }
            if (near.size < 4) continue
            val from = MatOfPoint2f(*near.map { Point(px[it], py[it]) }.toTypedArray())
            val to = MatOfPoint2f(*near.map { Point(fx[it], fy[it]) }.toTypedArray())
            val inl = Mat()
            val am = Calib3d.estimateAffinePartial2D(from, to, inl, Calib3d.RANSAC, cfg.ransacThreshold, 2000, 0.99, 10)
            if (am.empty()) continue
            val mask = ByteArray(near.size); inl.get(0, 0, mask)
            val ins = near.filterIndexed { k, _ -> mask[k].toInt() != 0 }
            if (ins.size < 4) continue
            val mx = ins.sumOf { px[it] } / ins.size; val my = ins.sumOf { py[it] } / ins.size
            if (ins.maxOf { hypot(px[it] - mx, py[it] - my) } < cell * 0.15) continue // all on one spot
            val a = DoubleArray(6); am.get(0, 0, a)
            for (k in ins) used[k] = true
            groups.add(Group(a, ins.toIntArray(), hypot(a[0], a[3]), atan2(a[3], a[0])))
        }
        if (groups.isEmpty()) return Result(emptyList(), emptyList(), fk.size, n, t1 - t0, t2 - t1, System.currentTimeMillis() - t2)
        // The frame's scale comes from well-supported groups only; weak groups must then agree with it.
        var strong = groups.filter { it.idx.size >= cfg.scaleInliers }
        if (strong.size < 2) strong = groups.filter { it.idx.size >= cfg.scaleInliers - 2 }
        if (strong.size < 2) return Result(emptyList(), emptyList(), fk.size, n, t1 - t0, t2 - t1, System.currentTimeMillis() - t2)
        val med = strong.map { it.scale }.sorted()[strong.size / 2]

        val out = ArrayList<Located>()
        for (g in groups.sortedByDescending { it.idx.size }) {
            if (g.idx.size < cfg.minInliers || abs(ln(g.scale / med)) > cfg.scaleTolerance) continue
            // Start from the cell holding most inliers, then let the picture decide among its neighbours.
            val votes = HashMap<Int, Int>()
            for (k in g.idx) { val c = (px[k] / cell).toInt().coerceIn(0, cols - 1); val r = (py[k] / cell).toInt().coerceIn(0, rows - 1); votes.merge(r * cols + c, 1, Int::plus) }
            val start = votes.maxBy { it.value }.key
            val (best, ncc) = refineCell(gray, g.a, start % cols, start / cols)
            if (ncc < cfg.minNcc) continue
            val col = best % cols; val row = best / cols
            fun map(x: Double, y: Double) = Point(g.a[0] * x + g.a[1] * y + g.a[2], g.a[3] * x + g.a[4] * y + g.a[5])
            val quad = arrayOf(map(col * cell, row * cell), map((col + 1) * cell, row * cell), map((col + 1) * cell, (row + 1) * cell), map(col * cell, (row + 1) * cell))
            val c = map((col + 0.5) * cell, (row + 0.5) * cell)
            // One result per physical piece.
            if (out.any { hypot(it.cx - c.x, it.cy - c.y) < cell * med * 0.6 }) continue
            // A cell belongs to one piece: groups are visited strongest first, so the first claim wins.
            if (out.any { it.col == col && it.row == row }) continue
            out.add(Located(col, row, g.idx.size, ncc, g.a, quad, c.x, c.y, g.angle, g.scale))
        }
        // Pairs: neighbouring cells, and not already joined (joined pieces share one transform).
        val pairs = ArrayList<Pair>()
        for (i in out.indices) for (j in i + 1 until out.size) {
            val a = out[i]; val b = out[j]
            if (abs(a.col - b.col) + abs(a.row - b.row) != 1) continue
            val ex = a.a[0] * (b.col + 0.5) * cell + a.a[1] * (b.row + 0.5) * cell + a.a[2]
            val ey = a.a[3] * (b.col + 0.5) * cell + a.a[4] * (b.row + 0.5) * cell + a.a[5]
            if (hypot(ex - b.cx, ey - b.cy) < cell * med * 0.35) continue // already assembled
            pairs.add(Pair(a, b))
        }
        return Result(out, pairs, fk.size, n, t1 - t0, t2 - t1, System.currentTimeMillis() - t2)
    }

    /**
     * The frame is warped into poster coordinates around the piece (3x3 cells) and each cell is
     * correlated with the box picture: the piece covers its own cell completely, while neighbouring
     * cells show table or other pieces. Returns (cell index, correlation).
     */
    private fun refineCell(frameGray: Mat, a: DoubleArray, col: Int, row: Int): kotlin.Pair<Int, Double> {
        val k = SMALL / cell
        // poster -> patch: p' = k * (p - origin); frame -> poster is the inverse of a.
        val det = a[0] * a[4] - a[1] * a[3]
        val i0 = a[4] / det; val i1 = -a[1] / det; val i3 = -a[3] / det; val i4 = a[0] / det
        val i2 = -(i0 * a[2] + i1 * a[5]); val i5 = -(i3 * a[2] + i4 * a[5])
        val ox = (col - 1) * cell; val oy = (row - 1) * cell
        val m = Mat(2, 3, org.opencv.core.CvType.CV_64F)
        m.put(0, 0, k * i0, k * i1, k * (i2 - ox), k * i3, k * i4, k * (i5 - oy))
        val patch = Mat(); Imgproc.warpAffine(frameGray, patch, m, Size(3.0 * SMALL, 3.0 * SMALL), Imgproc.INTER_AREA)
        var best = row * cols + col; var bestV = -2.0; var ownV = -2.0
        val inset = SMALL / 8; val sz = SMALL - 2 * inset
        for (dr in 0..2) for (dc in 0..2) {
            val c = col - 1 + dc; val r = row - 1 + dr
            if (c < 0 || r < 0 || c >= cols || r >= rows) continue
            val pa = patch.submat(Rect(dc * SMALL + inset, dr * SMALL + inset, sz, sz))
            val pb = small.submat(Rect((c + 1) * SMALL + inset, (r + 1) * SMALL + inset, sz, sz))
            val res = Mat(); Imgproc.matchTemplate(pa, pb, res, Imgproc.TM_CCOEFF_NORMED)
            val v = res.get(0, 0)[0]
            if (!v.isFinite()) continue
            if (dr == 1 && dc == 1) ownV = v
            if (v > bestV) { bestV = v; best = r * cols + c }
        }
        // Move off the cell the features voted for only when a neighbour is clearly the better fit.
        if (best != row * cols + col && bestV < ownV + 0.15) return (row * cols + col) to ownV
        return best to bestV
    }

    companion object { private const val SMALL = 32 }
}
