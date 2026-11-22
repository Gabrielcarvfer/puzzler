package puzzlemaster.core

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * One live pass: (squared-up) frame -> piece outlines -> edge/corner pieces -> pairs that fit.
 *
 * 1. [TableSegmenter] gives one blob per isolated piece (touching pieces merge and are dropped by the
 *    piece-shape check: we report what we can see clearly).
 * 2. Thickness correction: seen at an angle, a piece's silhouette is its printed face *plus* the
 *    visible cardboard side wall on the camera side. The face is recovered as
 *    silhouette AND (silhouette shifted away from the camera by thickness * tan(view angle)).
 * 3. The outline is snapped to the strongest brightness edge nearby and described by
 *    [PieceDetector] (4 corners, tab / blank / flat sides).
 * 4. Pairs: mutual best [MateScorer] fits, clearly better than each edge's runner-up.
 */
class LivePipeline(val cfg: Config = Config()) {
    data class Config(
        /** Piece thickness / piece side length (about 1.9 mm / 18 mm for a 1000-piece puzzle). */
        val thicknessRatio: Double = 0.105,
        /** Outline snap range as a fraction of piece size (0 = off). */
        val refineRange: Double = 0.02,
        /** Pair acceptance: 90th-percentile gap between the fitted edges / edge length. */
        val maxGap: Double = 0.022,
        /** Best gap must be at most this fraction of the runner-up's for both edges. */
        val maxRatio: Double = 0.75,
        /** Whole-piece colour mix gate and weight of picture continuity in the ranking. */
        val paletteGate: Double = 0.75,
        val pictureWeight: Double = 0.02,
        /** A flat side deviates less than this fraction of its length from a straight line. */
        val borderFlatness: Double = 0.05,
        val minQuality: Double = 0.25,
        val sizePerRadius: Double = 2.8,
        /** Pairs are only reported for pieces at least this big (pixels): below it, shape is noise. */
        val minPairSize: Double = 170.0,
    )

    /** Where the camera is relative to the rectified image (for the thickness correction). */
    class Lean(val nadirX: Double, val nadirY: Double, val focal: Double)

    class Pair(val a: Edge, val b: Edge, val gap: Double, val runnerUp: Double, val picture: Double)

    class Out(
        val pieces: List<Piece>,
        /** Flat (border) sides per piece id. */
        val border: Map<Int, List<Edge>>,
        val pairs: List<Pair>,
        /** Outlines of piece-sized blobs whose corners could not be worked out (drawn, not analysed). */
        val loose: List<kotlin.Pair<DoubleArray, DoubleArray>>,
        val blobs: Int,
        /** Why blobs were not accepted as pieces: (x, y, reason). */
        val rejects: List<Triple<Double, Double, String>>,
        val bodyRadius: Double,
        val segMs: Long, val pieceMs: Long, val matchMs: Long,
    )

    private val detector = PieceDetector(PieceDetector.Config(edgeSamples = 96))
    private val scorer = MateScorer()
    private val picture = EdgeMatcher()

    fun process(bgr: Mat, valid: Mat? = null, lean: Lean? = null): Out {
        val t0 = System.currentTimeMillis()
        val seg = TableSegmenter.segment(bgr, valid)
        val blobs = seg.blobs
        val t1 = System.currentTimeMillis()
        val w = bgr.cols(); val h = bgr.rows()
        // ARGB + luminance views for the Kotlin piece analysis.
        val raw = ByteArray(w * h * 3); bgr.get(0, 0, raw)
        val px = IntArray(w * h) { val k = it * 3; (0xFF shl 24) or ((raw[k + 2].toInt() and 0xFF) shl 16) or ((raw[k + 1].toInt() and 0xFF) shl 8) or (raw[k].toInt() and 0xFF) }
        val img = RgbImage(w, h, px)
        val lum = if (cfg.refineRange > 0) OutlineRefiner.luminance(img) else null

        val usable = blobs.filter { !it.touchesBorder }
        // Piece size estimate: median blob (a piece's area is close to its side length squared).
        // A piece is about 2.8 body radii across (measured on real frames).
        val sizeGuess = if (seg.bodyRadius > 0) seg.bodyRadius * cfg.sizePerRadius else if (usable.isEmpty()) 0.0 else sqrt(usable.map { it.area }.sorted()[usable.size / 2])
        val pieces = ArrayList<Piece>()
        val rejects = ArrayList<Triple<Double, Double, String>>()
        val loose = ArrayList<kotlin.Pair<DoubleArray, DoubleArray>>()
        fun rej(b: TableSegmenter.Blob, why: String) { val r = Imgproc.boundingRect(b.contour); rejects.add(Triple(r.x + r.width / 2.0, r.y + r.height / 2.0, why)) }
        for (b in blobs) if (b.touchesBorder) rej(b, "border")
        for (b in usable) {
            if (b.area > 2.6 * sizeGuess * sizeGuess) { rej(b, "big"); continue }
            if (b.area < 0.35 * sizeGuess * sizeGuess) continue // fragments
            val pts = faceOutline(b, lean, sizeGuess) ?: continue
            var xs = DoubleArray(pts.size) { pts[it].x + 0.5 }; var ys = DoubleArray(pts.size) { pts[it].y + 0.5 }
            if (lum != null) { val r = OutlineRefiner.refine(xs, ys, lum, w, h, (cfg.refineRange * sizeGuess).coerceAtLeast(1.5)); xs = r.first; ys = r.second }
            val p = detector.analyzeContour(xs, ys, img, pieces.size)
            if (p == null) { rej(b, "corners"); loose.add(xs to ys); continue }
            if (p.quality < cfg.minQuality) { rej(b, "q%.2f".format(p.quality)); loose.add(xs to ys); continue }
            if (!detector.isPlausible(p)) { rej(b, "shape"); loose.add(xs to ys); continue }
            pieces.add(p)
        }
        // All pieces of one puzzle have about the same size.
        var kept: List<Piece> = pieces
        if (pieces.size >= 4) {
            val med = pieces.map { it.scale }.sorted()[pieces.size / 2]
            kept = pieces.filter { it.scale in med * 0.7..med * 1.4 }
        }
        val border = HashMap<Int, List<Edge>>()
        for (p in kept) {
            val flat = p.edges.filter { e -> e.type == EdgeType.FLAT && e.cy.maxOf { abs(it) } < cfg.borderFlatness * e.length }
            if (flat.isNotEmpty()) border[p.id] = flat
        }
        val t2 = System.currentTimeMillis()
        val pairs = match(kept)
        return Out(kept, border, pairs, loose, blobs.size, rejects, seg.bodyRadius, t1 - t0, t2 - t1, System.currentTimeMillis() - t2)
    }

    /** Blob outline with the visible side wall removed (see class comment). */
    private fun faceOutline(b: TableSegmenter.Blob, lean: Lean?, pieceSize: Double): Array<Point>? {
        val pts = b.contour.toArray()
        if (lean == null || pieceSize <= 0) return pts
        val m = Imgproc.moments(b.contour)
        val cx = m.m10 / m.m00; val cy = m.m01 / m.m00
        // Top face is displaced away from the nadir by (p - nadir) * thickness / cameraHeight,
        // and thickness / cameraHeight = thicknessRatio * pieceSize / focal.
        val k = cfg.thicknessRatio * pieceSize / lean.focal
        val dx = ((cx - lean.nadirX) * k).roundToInt(); val dy = ((cy - lean.nadirY) * k).roundToInt()
        if (dx == 0 && dy == 0) return pts
        val r = Imgproc.boundingRect(b.contour)
        val pad = maxOf(abs(dx), abs(dy)) + 2
        val mw = r.width + 2 * pad; val mh = r.height + 2 * pad
        val mask = Mat.zeros(mh, mw, CvType.CV_8UC1)
        val off = Point((pad - r.x).toDouble(), (pad - r.y).toDouble())
        Imgproc.drawContours(mask, listOf(b.contour), 0, Scalar(255.0), -1, Imgproc.LINE_8, Mat(), 0, off)
        val shifted = Mat.zeros(mh, mw, CvType.CV_8UC1)
        val srcRect = Rect(maxOf(0, -dx), maxOf(0, -dy), mw - abs(dx), mh - abs(dy))
        val dstRect = Rect(maxOf(0, dx), maxOf(0, dy), mw - abs(dx), mh - abs(dy))
        mask.submat(srcRect).copyTo(shifted.submat(dstRect))
        Core.bitwise_and(mask, shifted, mask)
        val cs = ArrayList<MatOfPoint>()
        Imgproc.findContours(mask, cs, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE)
        val big = cs.maxByOrNull { Imgproc.contourArea(it) } ?: return null
        return big.toArray().map { Point(it.x - off.x, it.y - off.y) }.toTypedArray()
    }

    private fun match(pieces: List<Piece>): List<Pair> {
        val edges = pieces.filter { it.scale >= cfg.minPairSize }.flatMap { p -> p.edges.filter { it.type != EdgeType.FLAT } }
        val n = edges.size
        val best = IntArray(n) { -1 }
        val bestS = DoubleArray(n) { Double.MAX_VALUE }
        val second = DoubleArray(n) { Double.MAX_VALUE }
        val gaps = HashMap<Long, Double>()
        fun offer(i: Int, j: Int, s: Double) {
            if (s < bestS[i]) { second[i] = bestS[i]; bestS[i] = s; best[i] = j } else if (s < second[i]) second[i] = s
        }
        for (i in 0 until n) for (j in i + 1 until n) {
            val a = edges[i]; val b = edges[j]
            if (!scorer.compatible(a, b)) continue
            if (PieceDetector.paletteDistance(a.palette, b.palette) > cfg.paletteGate) continue
            val m = scorer.fit(a, b) ?: continue
            // Ranking score: the gap, nudged by how well the picture continues across the joint.
            val s = m.gap + cfg.pictureWeight * picture.pictureDistance(a, b)
            gaps[i.toLong() * n + j] = m.gap
            offer(i, j, s); offer(j, i, s)
        }
        val out = ArrayList<Pair>()
        for (i in 0 until n) {
            val j = best[i]
            if (j <= i || best[j] != i) continue
            val gap = gaps.getValue(i.toLong() * n + j)
            if (gap > cfg.maxGap) continue
            val runner = minOf(second[i], second[j])
            if (bestS[i] > cfg.maxRatio * runner) continue
            out.add(Pair(edges[i], edges[j], gap, runner, picture.pictureDistance(edges[i], edges[j])))
        }
        // One joint per pair of pieces.
        val seen = HashSet<Long>()
        return out.sortedBy { it.gap }.filter { seen.add(minOf(it.a.pieceId, it.b.pieceId) * 100000L + maxOf(it.a.pieceId, it.b.pieceId)) }
    }

    @Suppress("unused") private fun unused() = hypot(1.0, 1.0)
}
