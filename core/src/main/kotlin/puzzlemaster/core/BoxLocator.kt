package puzzlemaster.core

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Finds where pieces go on the box picture.
 *
 * The box picture is rectified once from its 4 corners. Each piece is straightened into an upright
 * square via its own 4 corners and searched on the picture in 4 rotations: first a fast, coarse
 * search over the whole picture, then a precise, masked comparison (piece pixels only) around the
 * best few candidates. Pieces that land in neighbouring cells are neighbours in the puzzle.
 *
 * Uses the OpenCV Java API (desktop jar in tests, Android AAR in the app). OpenCV must be loaded by
 * the caller.
 */
class BoxLocator(val poster: Mat, val cols: Int, val rows: Int, val cell: Int) {
    class Placement(
        val piece: Piece,
        /** Grid position of the piece's centre (fractional cells; integer part = column/row index). */
        val col: Double, val row: Double,
        /** Quarter turns that bring the straightened piece to the picture's orientation. */
        val rot: Int,
        val score: Double,
        /** Score margin over the best different location. */
        val margin: Double,
        /** Top candidate spots (col, row, score), best first, including the chosen one. */
        val alternatives: List<Triple<Double, Double, Double>> = emptyList(),
    ) {
        val confident get() = score > 0.5 && margin > 0.08
    }

    /** [edgeA]/[edgeB]: the joining edges when verified by edge matching; [score] lower = better. */
    class Pair(val a: Placement, val b: Placement, val edgeA: Edge? = null, val edgeB: Edge? = null, val score: Double = 0.0)

    private val coarseScale = 0.5
    private val posterSmall: Mat = Mat().also { Imgproc.resize(poster, it, Size(), coarseScale, coarseScale, Imgproc.INTER_AREA) }

    /** Locates one piece. [photo] is the (white-balanced, BGR) image the piece was detected in. */
    fun locate(photo: Mat, piece: Piece): Placement? {
        val c = cell.toDouble()
        val s0 = c * 0.5; val s1 = c * 1.5; val sz = cell * 2
        val src = MatOfPoint2f(*piece.cornerIdx.map { Point(piece.xs[it], piece.ys[it]) }.toTypedArray())
        val h = Imgproc.getPerspectiveTransform(src, MatOfPoint2f(Point(s0, s0), Point(s1, s0), Point(s1, s1), Point(s0, s1)))
        val patch = Mat(); Imgproc.warpPerspective(photo, patch, h, Size(sz.toDouble(), sz.toDouble()))
        val outline = MatOfPoint2f(*Array(piece.xs.size) { Point(piece.xs[it], piece.ys[it]) })
        val wo = MatOfPoint2f(); Core.perspectiveTransform(outline, wo, h)
        val mask = Mat.zeros(Size(sz.toDouble(), sz.toDouble()), CvType.CV_8UC1)
        Imgproc.fillPoly(mask, listOf(MatOfPoint(*wo.toArray())), Scalar(255.0))
        Imgproc.erode(mask, mask, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0)))
        if (Core.countNonZero(mask) < sz * sz / 6) return null
        // Coarse template: non-piece pixels (table seen through blanks) filled with the piece's mean colour.
        val mean = Core.mean(patch, mask)
        val filled = Mat(sz, sz, patch.type(), mean); patch.copyTo(filled, mask)

        class Cand(val rot: Int, val x: Int, val y: Int, val v: Double)
        val cands = ArrayList<Cand>()
        for (rot in 0 until 4) {
            val t = rotate(filled, rot)
            val ts = Mat(); Imgproc.resize(t, ts, Size(), coarseScale, coarseScale, Imgproc.INTER_AREA)
            val res = Mat(); Imgproc.matchTemplate(posterSmall, ts, res, Imgproc.TM_CCOEFF_NORMED)
            repeat(4) {
                val mm = Core.minMaxLoc(res)
                cands.add(Cand(rot, (mm.maxLoc.x / coarseScale).toInt(), (mm.maxLoc.y / coarseScale).toInt(), mm.maxVal))
                Imgproc.circle(res, mm.maxLoc, (c * coarseScale * 0.8).toInt(), Scalar(-1.0), -1)
            }
        }
        cands.sortByDescending { it.v }
        // Precise masked comparison around the best candidates.
        class Fine(val rot: Int, val x: Double, val y: Double, val v: Double)
        val fines = ArrayList<Fine>()
        for (cd in cands.take(8)) {
            val t = rotate(patch, cd.rot); val tm = rotate(mask, cd.rot)
            val tm3 = Mat(); Imgproc.cvtColor(tm, tm3, Imgproc.COLOR_GRAY2BGR)
            val pad = cell / 2
            val x0 = max(0, cd.x - pad); val y0 = max(0, cd.y - pad)
            val x1 = min(poster.cols(), cd.x + sz + pad); val y1 = min(poster.rows(), cd.y + sz + pad)
            if (x1 - x0 < sz || y1 - y0 < sz) continue
            val res = Mat(); Imgproc.matchTemplate(poster.submat(Rect(x0, y0, x1 - x0, y1 - y0)), t, res, Imgproc.TM_CCOEFF_NORMED, tm3)
            Core.patchNaNs(res, -1.0)
            val mm = Core.minMaxLoc(res)
            fines.add(Fine(cd.rot, x0 + mm.maxLoc.x, y0 + mm.maxLoc.y, mm.maxVal))
        }
        if (fines.isEmpty()) return null
        fines.sortByDescending { it.v }
        val best = fines[0]
        val other = fines.drop(1).firstOrNull { hypot(it.x - best.x, it.y - best.y) > c * 0.7 }?.v ?: -1.0
        // Template top-left + 1 cell = centre of the piece's square.
        val alts = ArrayList<Triple<Double, Double, Double>>()
        for (f in fines) {
            if (alts.any { hypot(it.first - (f.x / c + 1.0), it.second - (f.y / c + 1.0)) < 0.7 }) continue
            alts.add(Triple(f.x / c + 1.0, f.y / c + 1.0, f.v))
            if (alts.size == 3) break
        }
        return Placement(piece, best.x / c + 1.0, best.y / c + 1.0, best.rot, best.v, best.v - other, alts)
    }

    /**
     * Pairs from box positions + edge verification: two pieces are candidates when any of their
     * candidate spots are neighbouring cells; the pair is kept when some edge of one fits some edge
     * of the other (shape gate + picture continuity). Best first, each piece edge used once.
     */
    fun verifiedPairs(placements: List<Placement>, matcher: EdgeMatcher = EdgeMatcher(), minSpotScore: Double = 0.45, maxEdgeScore: Double = 0.35): List<Pair> {
        class Cand(val a: Placement, val b: Placement, val ea: Edge, val eb: Edge, val score: Double)
        val cands = ArrayList<Cand>()
        for (i in placements.indices) for (j in i + 1 until placements.size) {
            val a = placements[i]; val b = placements[j]
            var adjacent = 0.0
            for (ha in a.alternatives) for (hb in b.alternatives) {
                if (ha.third < minSpotScore || hb.third < minSpotScore) continue
                val dc = abs(ha.first - hb.first); val dr = abs(ha.second - hb.second)
                if ((abs(dc - 1) < 0.4 && dr < 0.4) || (abs(dr - 1) < 0.4 && dc < 0.4)) adjacent = max(adjacent, (ha.third + hb.third) / 2)
            }
            if (adjacent == 0.0) continue
            var best: Cand? = null
            for (ea in a.piece.edges) for (eb in b.piece.edges) {
                val sc = matcher.score(ea, eb)
                if (sc.isFinite() && sc < maxEdgeScore && (best == null || sc - adjacent < best.score)) best = Cand(a, b, ea, eb, sc - adjacent)
            }
            best?.let { cands.add(it) }
        }
        cands.sortBy { it.score }
        val used = HashSet<Edge>()
        val out = ArrayList<Pair>()
        for (c in cands) {
            if (c.ea in used || c.eb in used) continue
            used.add(c.ea); used.add(c.eb)
            out.add(Pair(c.a, c.b, c.ea, c.eb, c.score))
        }
        return out
    }

    /**
     * Strict pairs for live scanning: the two pieces' *best* spots are neighbouring cells, both
     * placements are strong and clearly ahead of their runner-up, and some edge pair fits.
     */
    fun livePairs(placements: List<Placement>, matcher: EdgeMatcher = EdgeMatcher(), minScore: Double = 0.65, minMargin: Double = 0.04, maxEdgeScore: Double = 0.3): List<Pair> {
        val strong = placements.filter { it.score >= minScore && it.margin >= minMargin }
        val out = ArrayList<Pair>()
        for (i in strong.indices) for (j in i + 1 until strong.size) {
            val a = strong[i]; val b = strong[j]
            val dc = abs(a.col - b.col); val dr = abs(a.row - b.row)
            if (!((abs(dc - 1) < 0.35 && dr < 0.35) || (abs(dr - 1) < 0.35 && dc < 0.35))) continue
            var best: Pair? = null
            for (ea in a.piece.edges) for (eb in b.piece.edges) {
                val sc = matcher.score(ea, eb)
                if (sc.isFinite() && sc < maxEdgeScore && (best == null || sc < best.score)) best = Pair(a, b, ea, eb, sc)
            }
            best?.let { out.add(it) }
        }
        return out.sortedBy { it.score }
    }

    /** Confident pieces placed in neighbouring cells. */
    fun pairs(placements: List<Placement>): List<Pair> {
        val conf = placements.filter { it.confident }
        val out = ArrayList<Pair>()
        for (i in conf.indices) for (j in i + 1 until conf.size) {
            val a = conf[i]; val b = conf[j]
            val dc = abs(a.col - b.col); val dr = abs(a.row - b.row)
            if ((abs(dc - 1) < 0.35 && dr < 0.35) || (abs(dr - 1) < 0.35 && dc < 0.35)) out.add(Pair(a, b))
        }
        return out.sortedByDescending { min(it.a.margin, it.b.margin) }
    }

    private fun rotate(m: Mat, rot: Int): Mat {
        val o = Mat()
        when (rot) {
            0 -> m.copyTo(o)
            1 -> Core.rotate(m, o, Core.ROTATE_90_CLOCKWISE)
            2 -> Core.rotate(m, o, Core.ROTATE_180)
            else -> Core.rotate(m, o, Core.ROTATE_90_COUNTERCLOCKWISE)
        }
        return o
    }

    companion object {
        /** Gray-world white balance, so the box photo and table photos agree in colour. */
        fun grayWorld(m: Mat): Mat {
            val mean = Core.mean(m)
            val g = (mean.`val`[0] + mean.`val`[1] + mean.`val`[2]) / 3
            val ch = ArrayList<Mat>(); Core.split(m, ch)
            for (i in 0 until 3) ch[i].convertTo(ch[i], -1, g / max(1.0, mean.`val`[i]), 0.0)
            val out = Mat(); Core.merge(ch, out); return out
        }

        /**
         * Finds the box picture's four corners in a photo (the largest bright-edged quadrilateral), or
         * null. Corners are returned clockwise from top-left, in the photo's pixel coordinates.
         */
        fun detectCorners(photoBgr: Mat): List<Point>? {
            val k = 1000.0 / maxOf(photoBgr.cols(), photoBgr.rows())
            val small = Mat(); Imgproc.resize(photoBgr, small, Size(photoBgr.cols() * k, photoBgr.rows() * k), 0.0, 0.0, Imgproc.INTER_AREA)
            val gray = Mat(); Imgproc.cvtColor(small, gray, Imgproc.COLOR_BGR2GRAY)
            Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)
            val area = small.cols() * small.rows().toDouble()
            var best: List<Point>? = null; var bestArea = 0.15 * area
            for ((lo, hi) in listOf(30.0 to 90.0, 50.0 to 150.0, 15.0 to 45.0)) {
                val edges = Mat(); Imgproc.Canny(gray, edges, lo, hi)
                Imgproc.dilate(edges, edges, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0)))
                val cs = ArrayList<org.opencv.core.MatOfPoint>()
                Imgproc.findContours(edges, cs, Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
                for (c in cs) {
                    val c2 = MatOfPoint2f(*c.toArray())
                    val hull = org.opencv.core.MatOfInt(); Imgproc.convexHull(c, hull)
                    val hp = MatOfPoint2f(*hull.toArray().map { c.toArray()[it] }.toTypedArray())
                    val approx = MatOfPoint2f(); Imgproc.approxPolyDP(hp, approx, 0.02 * Imgproc.arcLength(hp, true), true)
                    if (approx.rows() != 4) continue
                    val a = Imgproc.contourArea(approx)
                    if (a <= bestArea || a > 0.98 * area) continue
                    bestArea = a; best = approx.toArray().map { Point(it.x / k, it.y / k) }
                    c2.release()
                }
            }
            val b = best ?: return null
            val tl = b.minBy { it.x + it.y }; val br = b.maxBy { it.x + it.y }
            val tr = b.maxBy { it.x - it.y }; val bl = b.minBy { it.x - it.y }
            return listOf(tl, tr, br, bl)
        }

        /**
         * Builds the reference from a photo of the box picture and its 4 corners (any order).
         * [pieceCount] and [aspect] (width / height of the finished puzzle) set the grid.
         */
        fun fromPhoto(boxPhotoBgr: Mat, corners: List<Point>, pieceCount: Int = 1000, aspect: Double = 683.0 / 480.0, cell: Int = 32): BoxLocator {
            val tl = corners.minBy { it.x + it.y }; val br = corners.maxBy { it.x + it.y }
            val tr = corners.maxBy { it.x - it.y }; val bl = corners.minBy { it.x - it.y }
            val cols = Math.round(kotlin.math.sqrt(pieceCount * aspect)).toInt()
            val rows = Math.round(pieceCount.toDouble() / cols).toInt()
            val w = cols * cell; val h = Math.round(w / aspect).toInt()
            val m = Imgproc.getPerspectiveTransform(
                MatOfPoint2f(tl, tr, br, bl),
                MatOfPoint2f(Point(0.0, 0.0), Point(w.toDouble(), 0.0), Point(w.toDouble(), h.toDouble()), Point(0.0, h.toDouble())),
            )
            val poster = Mat(); Imgproc.warpPerspective(boxPhotoBgr, poster, m, Size(w.toDouble(), h.toDouble()))
            return BoxLocator(grayWorld(poster), cols, rows, cell)
        }
    }
}
