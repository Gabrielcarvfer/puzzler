package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Test
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File

/**
 * Locates each detected piece on the box picture: the piece is straightened into an upright square via
 * its 4 corners and searched on the rectified poster (normalised cross-correlation, 4 rotations).
 * Border pieces (with a flat edge) must land on the poster's outer ring: an automatic accuracy check.
 */
class BoxMatchTest {
    private val cols = 38
    private val cell = (System.getProperty("pm.cell") ?: "48").toInt() // poster pixels per piece cell

    /** Gray-world white balance: scale each channel so its mean equals the overall mean. */
    private fun grayWorld(m: Mat): Mat {
        val mean = Core.mean(m)
        val g = (mean.`val`[0] + mean.`val`[1] + mean.`val`[2]) / 3
        val ch = ArrayList<Mat>(); Core.split(m, ch)
        for (i in 0 until 3) ch[i].convertTo(ch[i], -1, g / mean.`val`[i].coerceAtLeast(1.0), 0.0)
        val out = Mat(); Core.merge(ch, out); return out
    }

    private fun loadPoster(): Mat? {
        val f = File("src/test/resources/box/box.jpg").takeIf { it.exists() } ?: return null
        val img = Imgcodecs.imread(f.path, Imgcodecs.IMREAD_REDUCED_COLOR_2)
        // Poster = large saturated region on grey cardboard.
        val hsv = Mat(); Imgproc.cvtColor(img, hsv, Imgproc.COLOR_BGR2HSV)
        val ch = ArrayList<Mat>(); Core.split(hsv, ch)
        val mask = Mat(); Imgproc.threshold(ch[1], mask, 115.0, 255.0, Imgproc.THRESH_BINARY)
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(25.0, 25.0)))
        val contours = ArrayList<MatOfPoint>(); Imgproc.findContours(mask, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        val big = contours.maxBy { Imgproc.contourArea(it) }
        val c2f = MatOfPoint2f(*big.toArray())
        val approx = MatOfPoint2f()
        var eps = 0.01
        do { Imgproc.approxPolyDP(c2f, approx, eps * Imgproc.arcLength(c2f, true), true); eps += 0.005 } while (approx.toArray().size > 4 && eps < 0.1)
        var q = approx.toArray()
        // Fallback: corners read off the preview (half-resolution coordinates).
        if (q.size != 4 || Imgproc.contourArea(MatOfPoint(*q)) > 0.95 * img.cols() * img.rows())
            q = arrayOf(Point(127.0, 215.0), Point(3977.0, 205.0), Point(3990.0, 2974.0), Point(119.0, 2966.0))
        val tl = q.minBy { it.x + it.y }; val br = q.maxBy { it.x + it.y }
        val tr = q.maxBy { it.x - it.y }; val bl = q.minBy { it.x - it.y }
        val w = cols * cell
        val h = Math.round(w * 480.0 / 683.0).toInt()
        val M = Imgproc.getPerspectiveTransform(MatOfPoint2f(tl, tr, br, bl), MatOfPoint2f(Point(0.0, 0.0), Point(w.toDouble(), 0.0), Point(w.toDouble(), h.toDouble()), Point(0.0, h.toDouble())))
        val posterRaw = Mat(); Imgproc.warpPerspective(img, posterRaw, M, Size(w.toDouble(), h.toDouble()))
        val poster = grayWorld(posterRaw)
        println("BOXMATCH poster ${w}x$h (${cols}x${h / cell} cells), corners $tl $tr $br $bl")
        return poster
    }

    class Placement(val pieceId: Int, val col: Double, val row: Double, val rot: Int, val score: Double, val second: Double)

    @Test
    fun locatePiecesOnBox() {
        if (!File("src/test/resources/box/box.jpg").exists()) return
        if (figureDir()?.let { File(it, ".slow").exists() } != true) return // exploratory, minutes long
        OpenCV.loadLocally()
        val poster = loadPoster() ?: return
        val rows = poster.rows() / cell
        val photos = File("src/test/resources/map").listFiles { f -> f.name.endsWith(".jpg") }?.sorted() ?: return
        val reg = MosaicTest.register(photos)
        val canvasPos = HashMap<Int, Pair<Double, Double>>()
        val detector = PieceDetector()
        var total = 0; var borderPieces = 0; var borderOk = 0
        val results = ArrayList<Pair<Placement, Mat>>()
        for ((pi, f) in photos.withIndex()) {
            val mRaw = Imgcodecs.imread(f.path, Imgcodecs.IMREAD_REDUCED_COLOR_2)
            val m = grayWorld(mRaw)
            val buf = ByteArray(m.cols() * m.rows() * 3); m.get(0, 0, buf)
            val px = IntArray(m.cols() * m.rows()) { val k = it * 3; (0xFF shl 24) or ((buf[k + 2].toInt() and 0xFF) shl 16) or ((buf[k + 1].toInt() and 0xFF) shl 8) or (buf[k].toInt() and 0xFF) }
            val pieces = EdgeSegmenter().findPieces(RgbImage(m.cols(), m.rows(), px), detector)
            for (p in pieces) {
                // Whole piece (tabs included) warped so its corner square maps to [cell/2, 3cell/2].
                val s0 = cell * 0.5; val s1 = cell * 1.5; val S = cell * 2
                val src = MatOfPoint2f(*p.cornerIdx.map { Point(p.xs[it], p.ys[it]) }.toTypedArray())
                val H = Imgproc.getPerspectiveTransform(src, MatOfPoint2f(Point(s0, s0), Point(s1, s0), Point(s1, s1), Point(s0, s1)))
                val patch = Mat(); Imgproc.warpPerspective(m, patch, H, Size(S.toDouble(), S.toDouble()))
                // Mask: piece outline warped the same way, shrunk to avoid the cut edge and shadow.
                val outl = MatOfPoint2f(*Array(p.xs.size) { Point(p.xs[it], p.ys[it]) })
                val wo = MatOfPoint2f(); Core.perspectiveTransform(outl, wo, H)
                val mask = Mat.zeros(Size(S.toDouble(), S.toDouble()), CvType.CV_8UC1)
                Imgproc.fillPoly(mask, listOf(MatOfPoint(*wo.toArray().map { Point(it.x, it.y) }.toTypedArray())), Scalar(255.0))
                Imgproc.erode(mask, mask, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0)))
                val inset = 0
                val core = patch
                var best = -2.0; var bestLoc = Point(); var bestRot = 0; var bestMap: Mat? = null
                for (rot in 0 until 4) {
                    val t = Mat(); val tm = Mat()
                    fun r(a: Mat, b: Mat) { when (rot) { 0 -> a.copyTo(b); 1 -> Core.rotate(a, b, Core.ROTATE_90_CLOCKWISE); 2 -> Core.rotate(a, b, Core.ROTATE_180); else -> Core.rotate(a, b, Core.ROTATE_90_COUNTERCLOCKWISE) } }
                    r(core, t); r(mask, tm)
                    val tm3 = Mat(); Imgproc.cvtColor(tm, tm3, Imgproc.COLOR_GRAY2BGR)
                    val res = Mat(); Imgproc.matchTemplate(poster, t, res, Imgproc.TM_CCOEFF_NORMED, tm3)
                    Core.patchNaNs(res, -1.0)
                    val mm = Core.minMaxLoc(res)
                    if (mm.maxVal > best) { best = mm.maxVal; bestLoc = mm.maxLoc; bestRot = rot; bestMap = res }
                }
                // Second-best peak at least one cell away (uniqueness).
                val res = bestMap!!
                Imgproc.circle(res, bestLoc, cell, Scalar(-1.0), -1)
                val second = Core.minMaxLoc(res).maxVal
                // Template top-left + cell = centre of the piece's square.
                val pl = Placement(p.id + pi * 10000, bestLoc.x / cell + 1.0, bestLoc.y / cell + 1.0, bestRot, best, second)
                total++
                if (p.flatCount > 0) {
                    borderPieces++
                    val onRing = pl.col < 1.2 || pl.col > cols - 1.2 || pl.row < 1.2 || pl.row > rows - 1.2
                    if (onRing) borderOk++
                }
                results.add(pl to patch.submat(Rect(cell / 2, cell / 2, cell, cell)))
                reg.toRef[pi]?.let { h ->
                    val toC = Mat(); Core.gemm(reg.canvasT, h, 1.0, Mat(), 0.0, toC)
                    val o = MatOfPoint2f(); Core.perspectiveTransform(MatOfPoint2f(Point(p.centroidX / 2, p.centroidY / 2)), o, toC)
                    canvasPos[pl.pieceId] = o.toArray()[0].x to o.toArray()[0].y
                }
            }
            println("BOXMATCH photo $pi: ${pieces.size} pieces")
        }
        // Consistency: the same physical piece seen in two photos must land on the same poster spot.
        val byId = results.associate { it.first.pieceId to it.first }
        val ids = canvasPos.keys.toList()
        var dup = 0; var agree = 0; var dupConf = 0; var agreeConf = 0
        for (i in ids.indices) for (j in i + 1 until ids.size) {
            val a = ids[i]; val b = ids[j]
            if (a / 10000 == b / 10000) continue // same photo
            val (ax, ay) = canvasPos.getValue(a); val (bx, by) = canvasPos.getValue(b)
            if (kotlin.math.hypot(ax - bx, ay - by) > 25) continue // ~half a piece on the map
            val pa = byId.getValue(a); val pb = byId.getValue(b)
            val same = kotlin.math.hypot(pa.col - pb.col, pa.row - pb.row) < 0.6
            dup++; if (same) agree++
            val conf = minOf(pa.score - pa.second, pb.score - pb.second) > 0.08
            if (conf) { dupConf++; if (same) agreeConf++ }
        }
        println("BOXMATCH consistency: same piece in 2 photos -> same poster spot: $agree/$dup; among confident (margin>0.08): $agreeConf/$dupConf")
        val confident = results.count { it.first.score > 0.5 && it.first.score - it.first.second > 0.1 }
        println("BOXMATCH located $total pieces; confident (score>0.5, margin>0.1): $confident; border pieces on outer ring: $borderOk/$borderPieces")
        println("BOXMATCH score distribution: " + results.map { it.first.score }.sorted().let { s -> listOf(0.1, 0.25, 0.5, 0.75, 0.9).joinToString { "p%.0f=%.2f".format(it * 100, s[(s.size * it).toInt().coerceAtMost(s.size - 1)]) } })
        figureDir()?.let { dir ->
            val vis = poster.clone()
            for ((pl, _) in results) {
                val ok = pl.score > 0.5 && pl.score - pl.second > 0.1
                val c = if (ok) Scalar(0.0, 255.0, 0.0) else Scalar(0.0, 0.0, 255.0)
                Imgproc.rectangle(vis, Point((pl.col - 0.5) * cell, (pl.row - 0.5) * cell), Point((pl.col + 0.5) * cell, (pl.row + 0.5) * cell), c, 2)
            }
            Imgcodecs.imwrite(File(dir, "box_placements.jpg").path, vis)
            // Montage: piece patch next to the poster at its found location (first 24 confident).
            val tiles = results.filter { it.first.score > 0.5 }.take(24)
            val mont = Mat.zeros(Size(cell * 2.0 * 6 * 3, cell * 3.0 * ((tiles.size + 5) / 6)), CvType.CV_8UC3)
            for ((k, t) in tiles.withIndex()) {
                val (pl, patch) = t
                val big = Mat(); Imgproc.resize(patch, big, Size(cell * 3.0, cell * 3.0))
                val x0 = (k % 6) * cell * 6; val y0 = (k / 6) * cell * 3
                big.copyTo(mont.submat(Rect(x0, y0, cell * 3, cell * 3)))
                val r = Rect(((pl.col - 0.5) * cell).toInt().coerceIn(0, poster.cols() - cell), ((pl.row - 0.5) * cell).toInt().coerceIn(0, poster.rows() - cell), cell, cell)
                val pc = Mat(); Imgproc.resize(poster.submat(r), pc, Size(cell * 3.0, cell * 3.0))
                val rp = Mat()
                when (pl.rot) { 0 -> pc.copyTo(rp); 1 -> Core.rotate(pc, rp, Core.ROTATE_90_COUNTERCLOCKWISE); 2 -> Core.rotate(pc, rp, Core.ROTATE_180); else -> Core.rotate(pc, rp, Core.ROTATE_90_CLOCKWISE) }
                rp.copyTo(mont.submat(Rect(x0 + cell * 3, y0, cell * 3, cell * 3)))
            }
            Imgcodecs.imwrite(File(dir, "box_pairs.jpg").path, mont)
        }
    }
}
