package puzzlemaster.core

import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.DMatch
import org.opencv.core.KeyPoint
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.features2d.FlannBasedMatcher
import org.opencv.features2d.SIFT
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot

/**
 * Places every piece-sized outline of a photo straight onto the box picture, without needing a clean
 * four-corner piece model: the print inside the outline is matched to the box and one rigid placement is
 * fitted per outline. Sightings accumulate per box cell, so a piece seen in many photos is one cell, and
 * pairs are simply neighbouring occupied cells.
 */
class BoxMap(val placer: BoxPlacer) {
    class Sighting(val photo: String, val tag: Int, val xs: DoubleArray, val ys: DoubleArray, val boxX: Double, val boxY: Double, val inliers: Int, val clean: Boolean,
        /** Matched print points on the box (inliers), and the same points in the photo. */
        val boxPts: Array<Point> = emptyArray(), val photoPts: Array<Point> = emptyArray())
    /** Everything seen at roughly one box position. */
    class Spot(var boxX: Double, var boxY: Double) {
        val sightings = ArrayList<Sighting>()
        val support: Int get() = sightings.sumOf { it.inliers }
        val best: Sighting get() = sightings.maxBy { it.inliers }
    }

    val spots = ArrayList<Spot>()
    /** Every placed sighting, for [cells]. */
    val all = ArrayList<Sighting>()
    private val sift = ThreadLocal.withInitial { SIFT.create(0, 3, 0.03, 10.0, 1.6) }
    var minInliers = 5
    var ratio = 0.85f
    var centreFromOutline = false
    /** Timing of the last [locate] (diagnostics). */
    @Volatile var lastTiming = ""

    /**
     * [image]: the analysed (squared-up) photo; [outlines]: closed outlines in its coordinates;
     * [pieceSize]: typical piece size there (pixels). Returns how many outlines were placed.
     */
    fun observe(photo: String, image: Mat, outlines: List<Pair<DoubleArray, DoubleArray>>, pieceSize: Double, cleanCount: Int = 0,
                tag: Int = 0, toOutput: ((Double, Double) -> Point)? = null): Int {
        val found = locate(photo, image, outlines, pieceSize, cleanCount, tag, toOutput)
        addAll(found)
        return found.size
    }

    /** Adds sightings from [locate] (thread-safe). */
    fun addAll(found: List<Sighting>) = synchronized(this) { for (sg in found) { all.add(sg); add(sg) } }

    /**
     * The placing part of [observe], without changing the map: safe to run for several photos in
     * parallel; hand the result to [addAll].
     */
    fun locate(photo: String, image: Mat, outlines: List<Pair<DoubleArray, DoubleArray>>, pieceSize: Double, cleanCount: Int = 0,
               tag: Int = 0, toOutput: ((Double, Double) -> Point)? = null): List<Sighting> {
        val found = ArrayList<Sighting>()
        val tStart = System.currentTimeMillis()
        fun conv(x: Double, y: Double) = toOutput?.invoke(x, y) ?: Point(x, y)
        if (outlines.isEmpty() || pieceSize <= 0) return found
        // Features per outline, on its own small crop: running SIFT over the whole photo with a mask
        // still builds the full-resolution pyramid (10+ s per photo on the phone).
        val gray = Mat(); Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY)
        val kpList = ArrayList<KeyPoint>(); val owner = ArrayList<Int>(); val descs = ArrayList<Mat>()
        val ker = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, org.opencv.core.Size(5.0, 5.0))
        for ((i, o) in outlines.withIndex()) {
            val x0 = o.first.min().toInt().coerceAtLeast(0); val y0 = o.second.min().toInt().coerceAtLeast(0)
            val x1 = o.first.max().toInt().coerceAtMost(gray.cols() - 1); val y1 = o.second.max().toInt().coerceAtMost(gray.rows() - 1)
            if (x1 - x0 < 16 || y1 - y0 < 16) continue
            val roi = org.opencv.core.Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1)
            val mask = Mat.zeros(roi.height, roi.width, CvType.CV_8UC1)
            Imgproc.fillPoly(mask, listOf(MatOfPoint(*Array(o.first.size) { Point(o.first[it] - x0, o.second[it] - y0) })), Scalar(255.0))
            Imgproc.erode(mask, mask, ker)
            val kpM = MatOfKeyPoint(); val d = Mat()
            sift.get().detectAndCompute(gray.submat(roi), mask, kpM, d)
            mask.release()
            val ks = kpM.toArray()
            if (ks.isEmpty() || d.empty()) continue
            for (k in ks) { k.pt = Point(k.pt.x + x0, k.pt.y + y0); kpList.add(k); owner.add(i) }
            descs.add(d)
        }
        gray.release()
        if (kpList.isEmpty()) return found
        val desc = Mat(); Core.vconcat(descs, desc)
        val kps = kpList.toTypedArray()
        val tSift = System.currentTimeMillis()
        val matches = placer.matchFeatures(desc, ratio)
        lastTiming = "sift ${tSift - tStart} ms (${kps.size} features), match ${System.currentTimeMillis() - tSift} ms"
        // Group matches by outline.
        val byOutline = HashMap<Int, ArrayList<DMatch>>()
        for (m in matches) byOutline.getOrPut(owner[m.queryIdx]) { ArrayList() }.add(m)
        val expect = placer.nominalPiece / pieceSize
        for ((i, ms) in byOutline) {
            if (ms.size < minInliers) continue
            val from = MatOfPoint2f(*ms.map { kps[it.queryIdx].pt }.toTypedArray())
            val to = MatOfPoint2f(*ms.map { placer.posterPoint(it.trainIdx) }.toTypedArray())
            val inl = Mat()
            val a = Calib3d.estimateAffinePartial2D(from, to, inl, Calib3d.RANSAC, 4.0, 2000, 0.99, 10)
            if (a.empty()) continue
            val n = Core.countNonZero(inl)
            val sc = hypot(a.get(0, 0)[0], a.get(1, 0)[0])
            if (n < minInliers || sc < expect * 0.7 || sc > expect * 1.4) continue
            val o = outlines[i]
            val bx: Double; val by: Double
            val clean = i < cleanCount
            if (centreFromOutline || clean) {
                val mo = Imgproc.moments(MatOfPoint2f(*Array(o.first.size) { Point(o.first[it], o.second[it]) }))
                if (mo.m00 <= 0) continue
                val cx = mo.m10 / mo.m00; val cy = mo.m01 / mo.m00
                bx = a.get(0, 0)[0] * cx + a.get(0, 1)[0] * cy + a.get(0, 2)[0]
                by = a.get(1, 0)[0] * cx + a.get(1, 1)[0] * cy + a.get(1, 2)[0]
            } else {
                // Median of the matched print points on the box: the outline may be partial or merged,
                // but the matched print is where the piece is.
                val t = to.toArray(); val xs = ArrayList<Double>(); val ys = ArrayList<Double>()
                for (j in t.indices) if (inl.get(j, 0)[0] != 0.0) { xs.add(t[j].x); ys.add(t[j].y) }
                xs.sort(); ys.sort(); bx = xs[xs.size / 2]; by = ys[ys.size / 2]
            }
            val tt = to.toArray(); val ff = from.toArray()
            val bp = ArrayList<Point>(); val pp = ArrayList<Point>()
            for (j in tt.indices) if (inl.get(j, 0)[0] != 0.0) { bp.add(tt[j]); pp.add(ff[j]) }
            // Stored outline and print points are in output coordinates (e.g. the upright photo, normalised).
            val step = maxOf(1, o.first.size / 80)
            val ox = ArrayList<Double>(); val oy = ArrayList<Double>()
            var t = 0
            while (t < o.first.size) { val q = conv(o.first[t], o.second[t]); ox.add(q.x); oy.add(q.y); t += step }
            val sg = Sighting(photo, tag, ox.toDoubleArray(), oy.toDoubleArray(), bx, by, n, clean, bp.toTypedArray(), pp.map { conv(it.x, it.y) }.toTypedArray())
            found.add(sg)
        }
        return found
    }

    private fun add(s: Sighting) {
        val r = 0.45 * placer.nominalPiece
        val near = spots.minByOrNull { hypot(it.boxX - s.boxX, it.boxY - s.boxY) }
        val spot = if (near != null && hypot(near.boxX - s.boxX, near.boxY - s.boxY) < r) near else Spot(s.boxX, s.boxY).also { spots.add(it) }
        spot.sightings.add(s)
        // Support-weighted position.
        val w = spot.sightings.sumOf { it.inliers }.toDouble()
        spot.boxX = spot.sightings.sumOf { it.boxX * it.inliers } / w
        spot.boxY = spot.sightings.sumOf { it.boxY * it.inliers } / w
    }

    /** Spots with at least this much support, as BoxPlacer hits for grid fitting (one per spot). */
    fun grid(minSupport: Int = 6): BoxPlacer.Grid = placer.fitGrid(spots.filter { it.support >= minSupport }.map { pseudoHit(it) })

    private fun pseudoHit(s: Spot) = BoxPlacer.Hit(DUMMY, s.boxX, s.boxY, 0.0, s.support, Array(4) { doubleArrayOf(0.0, 0.0) })

    class CellPair(val a: Spot, val b: Spot, val colA: Int, val rowA: Int, val colB: Int, val rowB: Int)

    /** Neighbouring occupied cells; the best-supported spot holds each cell. */
    fun pairs(g: BoxPlacer.Grid, minSupport: Int = 6): List<CellPair> {
        val cell = HashMap<Pair<Int, Int>, Spot>()
        for (s in spots.filter { it.support >= minSupport }.sortedByDescending { it.support }) {
            val c = Math.floor((s.boxX - g.offX) / g.cw).toInt(); val r = Math.floor((s.boxY - g.offY) / g.ch).toInt()
            if (c in 0 until placer.cols && r in 0 until placer.rows) cell.putIfAbsent(c to r, s)
        }
        val out = ArrayList<CellPair>()
        for ((k, s) in cell) for (d in listOf(1 to 0, 0 to 1)) {
            val o = cell[(k.first + d.first) to (k.second + d.second)] ?: continue
            out.add(CellPair(s, o, k.first, k.second, k.first + d.first, k.second + d.second))
        }
        return out.sortedByDescending { minOf(it.a.support, it.b.support) }
    }

    fun occupiedCells(g: BoxPlacer.Grid, minSupport: Int = 6): Int =
        spots.filter { it.support >= minSupport }.map { Math.floor((it.boxX - g.offX) / g.cw).toInt() to Math.floor((it.boxY - g.offY) / g.ch).toInt() }.toSet().size

    /** One box cell and what was seen there. [votes]: matched print points that fell in this cell. */
    class Cell(val col: Int, val row: Int) {
        val seen = ArrayList<Pair<Sighting, Int>>()
        val votes: Int get() = seen.sumOf { it.second }
        /** The sighting that shows this piece best, and the photo point of its print in this cell. */
        val best: Sighting get() = seen.maxBy { it.second }.first
    }

    /**
     * Assigns sightings to cells by voting: each matched print point votes for the cell it lands in.
     * A sighting supports every cell that got at least [minVotes] of its points and [minShare] of them,
     * so an outline holding two touching pieces supports both cells.
     */
    fun cells(g: BoxPlacer.Grid, minVotes: Int = 3, minShare: Double = 0.25): Map<Pair<Int, Int>, Cell> {
        val out = HashMap<Pair<Int, Int>, Cell>()
        for (sg in all) {
            val cnt = HashMap<Pair<Int, Int>, Int>()
            for (p in sg.boxPts) {
                val c = Math.floor((p.x - g.offX) / g.cw).toInt(); val r = Math.floor((p.y - g.offY) / g.ch).toInt()
                if (c in 0 until placer.cols && r in 0 until placer.rows) cnt.merge(c to r, 1, Int::plus)
            }
            for ((k, v) in cnt) if (v >= minVotes && v >= minShare * sg.boxPts.size) out.getOrPut(k) { Cell(k.first, k.second) }.seen.add(sg to v)
        }
        return out
    }

    /** Grid fitted to clean pieces only (their outline centres are exact), falling back to all spots. */
    fun cleanGrid(): BoxPlacer.Grid {
        val clean = all.filter { it.clean }
        val hits = (if (clean.size >= 10) clean.map { it.boxX to it.boxY } else spots.map { it.boxX to it.boxY })
            .map { BoxPlacer.Hit(DUMMY, it.first, it.second, 0.0, 1, Array(4) { doubleArrayOf(0.0, 0.0) }) }
        return placer.fitGrid(hits)
    }

    /** Where, in the sighting's output coordinates, the print that voted for this cell is (median point). */
    fun markerOf(sg: Sighting, col: Int, row: Int, g: BoxPlacer.Grid): Point {
        val xs = ArrayList<Double>(); val ys = ArrayList<Double>()
        for (j in sg.boxPts.indices) {
            val c = Math.floor((sg.boxPts[j].x - g.offX) / g.cw).toInt(); val r = Math.floor((sg.boxPts[j].y - g.offY) / g.ch).toInt()
            if (c == col && r == row) { xs.add(sg.photoPts[j].x); ys.add(sg.photoPts[j].y) }
        }
        if (xs.isEmpty()) sg.photoPts.forEach { xs.add(it.x); ys.add(it.y) }
        xs.sort(); ys.sort()
        return Point(xs[xs.size / 2], ys[ys.size / 2])
    }

    /**
     * True when the two cells' pieces are already joined on the table: in some photo showing both, B sits
     * where A's own box-to-photo mapping says B's cell should be.
     */
    fun joined(a: Cell, b: Cell, g: BoxPlacer.Grid): Boolean {
        for ((sa, _) in a.seen) for ((sb, _) in b.seen) {
            if (sa.photo != sb.photo || sa.boxPts.size < 3) continue
            val aff = org.opencv.calib3d.Calib3d.estimateAffinePartial2D(MatOfPoint2f(*sa.boxPts), MatOfPoint2f(*sa.photoPts))
            if (aff.empty()) continue
            fun map(x: Double, y: Double) = Point(aff.get(0, 0)[0] * x + aff.get(0, 1)[0] * y + aff.get(0, 2)[0], aff.get(1, 0)[0] * x + aff.get(1, 1)[0] * y + aff.get(1, 2)[0])
            val pa = map(g.offX + (a.col + 0.5) * g.cw, g.offY + (a.row + 0.5) * g.ch)
            val pb = map(g.offX + (b.col + 0.5) * g.cw, g.offY + (b.row + 0.5) * g.ch)
            val mb = markerOf(sb, b.col, b.row, g)
            val step = hypot(pb.x - pa.x, pb.y - pa.y)
            if (hypot(mb.x - pb.x, mb.y - pb.y) < 0.4 * step) return true
        }
        return false
    }

    fun clear() { spots.clear(); all.clear() }

    fun save(f: java.io.File) {
        val tmp = java.io.File(f.path + ".tmp")
        java.io.DataOutputStream(java.io.BufferedOutputStream(tmp.outputStream())).use { o ->
            o.writeInt(MAGIC); o.writeInt(all.size)
            for (s in all) {
                o.writeUTF(s.photo); o.writeInt(s.tag); o.writeInt(s.inliers); o.writeBoolean(s.clean)
                o.writeDouble(s.boxX); o.writeDouble(s.boxY)
                o.writeInt(s.xs.size); for (i in s.xs.indices) { o.writeFloat(s.xs[i].toFloat()); o.writeFloat(s.ys[i].toFloat()) }
                o.writeInt(s.boxPts.size)
                for (i in s.boxPts.indices) { o.writeFloat(s.boxPts[i].x.toFloat()); o.writeFloat(s.boxPts[i].y.toFloat()); o.writeFloat(s.photoPts[i].x.toFloat()); o.writeFloat(s.photoPts[i].y.toFloat()) }
            }
        }
        tmp.renameTo(f)
    }

    /** Replaces the contents with a saved map; false if the file is missing or of another format. */
    fun load(f: java.io.File): Boolean {
        if (!f.exists()) return false
        return try {
            java.io.DataInputStream(java.io.BufferedInputStream(f.inputStream())).use { i ->
                if (i.readInt() != MAGIC) return false
                clear()
                repeat(i.readInt()) {
                    val photo = i.readUTF(); val tag = i.readInt(); val n = i.readInt(); val clean = i.readBoolean()
                    val bx = i.readDouble(); val by = i.readDouble()
                    val m = i.readInt(); val xs = DoubleArray(m); val ys = DoubleArray(m)
                    for (k in 0 until m) { xs[k] = i.readFloat().toDouble(); ys[k] = i.readFloat().toDouble() }
                    val q = i.readInt(); val bp = arrayOfNulls<Point>(q); val pp = arrayOfNulls<Point>(q)
                    for (k in 0 until q) { bp[k] = Point(i.readFloat().toDouble(), i.readFloat().toDouble()); pp[k] = Point(i.readFloat().toDouble(), i.readFloat().toDouble()) }
                    val sg = Sighting(photo, tag, xs, ys, bx, by, n, clean, bp.requireNoNulls(), pp.requireNoNulls())
                    all.add(sg); add(sg)
                }
            }
            true
        } catch (e: Exception) { clear(); false }
    }

    companion object {
        private const val MAGIC = 0x504D4231
        private val DUMMY = PieceCatalogue.Entry(-1, Mat(), emptyArray(), emptyArray())
    }
}
