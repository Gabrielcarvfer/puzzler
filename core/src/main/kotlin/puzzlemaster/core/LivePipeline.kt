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
        /** Blobs above this many piece areas are split; above [splitUpTo] they are ignored. */
        val splitAbove: Double = 1.45,
        val splitUpTo: Double = 400.0,
        /** A neck is at most this wide (fraction of piece size) with at least [neckMinPart] of a piece on each side. */
        val neckWidth: Double = 0.24,
        val neckMinPart: Double = 0.5,
        /** Along one boundary, a neck's two sides are at least this far apart (fraction of piece size). */
        val neckMinArc: Double = 2.2,
        /** Cutting rounds: maximum neck width per round, as fractions of piece size. */
        val neckRounds: List<Double> = listOf(0.06, 0.11, 0.16, 0.21),
        /** Enclosed gaps smaller than this (fraction of a piece's area) are ignored when cutting. */
        val minGapArea: Double = 0.06,
        val segmentation: TableSegmenter.Config = TableSegmenter.Config(),
        /** Snap reduced-resolution outlines to the full-resolution edge. */
        val hiResRefine: Boolean = true,
        val sizePerRadius: Double = 2.8,
        val cornerSmooth: Int = 0,
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

    /** Debug hook: called for every piece-sized outline that was not accepted (window image, outline, reason). */
    var debugReject: ((RgbImage, DoubleArray, DoubleArray, String) -> Unit)? = null
    /** Debug hook: called for every accepted piece (coordinates in the full image). */
    var debugAccept: ((Piece) -> Unit)? = null

    private val detector = PieceDetector(PieceDetector.Config(edgeSamples = 96, cornerSmooth = cfg.cornerSmooth))
    private val scorer = MateScorer()
    private val picture = EdgeMatcher()

    fun process(bgr: Mat, valid: Mat? = null, lean: Lean? = null): Out {
        val t0 = System.currentTimeMillis()
        val seg = TableSegmenter.segment(bgr, valid, cfg.segmentation)
        val blobs = seg.blobs
        val t1 = System.currentTimeMillis()
        val w = bgr.cols(); val h = bgr.rows()
        // ARGB + luminance views for the Kotlin piece analysis.
        val raw = ByteArray(w * h * 3); bgr.get(0, 0, raw)
        val px = IntArray(w * h) { val k = it * 3; (0xFF shl 24) or ((raw[k + 2].toInt() and 0xFF) shl 16) or ((raw[k + 1].toInt() and 0xFF) shl 8) or (raw[k].toInt() and 0xFF) }
        val img = RgbImage(w, h, px)
        val lum = if (cfg.refineRange > 0) OutlineRefiner.luminance(img) else null

        // Piece size from the unsplit blobs (single pieces dominate), then split only what is too big.
        // Piece size estimate: median blob (a piece's area is close to its side length squared).
        // A piece is about 2.8 body radii across (measured on real frames).
        val sizeGuess = pieceSize(seg.raw.filter { !it.touchesBorder }, if (seg.bodyRadius > 0) seg.bodyRadius * cfg.sizePerRadius else 0.0)
        val usable = candidates(seg, sizeGuess, valid)
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

    /**
     * Piece side length from the blobs themselves: single pieces are the most common blob size, while
     * fragments are smaller and clumps or assembled strips are bigger. Finds the densest cluster of
     * blob areas (within a factor ~1.8), preferring larger clusters' sizes over specks.
     */
    private fun pieceSize(blobs: List<TableSegmenter.Blob>, fallback: Double): Double {
        val r = blobs.map { sqrt(it.area) }.sorted()
        if (r.size < 4) return fallback
        var best = 0.0; var bestScore = 0.0
        var hi = 0
        for (lo in r.indices) {
            while (hi < r.size && r[hi] <= r[lo] * 1.35) hi++
            val n = hi - lo
            val score = n * r[lo]
            if (n >= 3 && score > bestScore) { bestScore = score; best = r[lo + n / 2] }
        }
        // A piece's area is about 0.72 of its side length squared (tabs minus blanks, rounded corners).
        return if (best > 0) best / 0.85 else fallback
    }

    /**
     * Final piece candidates: the mask's regions after touching pieces have been cut apart at their
     * necks ([cutNecks]). Regions cut off by the photo's edge are dropped.
     */
    private fun candidates(seg: TableSegmenter.Result, size: Double, valid: Mat?): List<TableSegmenter.Blob> {
        val mask = seg.mask
        if (size <= 0 || mask == null) return seg.blobs.filter { !it.touchesBorder }
        // Narrowest contacts first, re-reading the boundaries after each round: once a tab's knob has
        // been freed from the neighbour it touched, its own neck no longer looks like a join.
        var cut: Mat = mask
        for (f in cfg.neckRounds) cut = cutNecks(cut, size, f * size)
        return TableSegmenter.blobsOf(cut, valid, seg.minArea).filter { !it.touchesBorder }
    }

    /**
     * Two touching pieces are joined by a neck: a place where the mask is narrow between two stretches
     * of boundary that are far apart along the boundary, or that belong to different boundaries (the
     * outside and an enclosed gap of table, or two gaps). A tab's own neck doesn't qualify: the way
     * around its knob is short. Every such neck inside a clump is cut with a thin line.
     */
    private fun cutNecks(mask: Mat, size: Double, maxCut: Double): Mat {
        val pieceArea = 0.72 * size * size
        val contours = ArrayList<MatOfPoint>(); val hier = Mat()
        Imgproc.findContours(mask, contours, hier, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_NONE)
        if (contours.isEmpty()) return mask
        val hi = IntArray(contours.size * 4); hier.get(0, 0, hi)
        val w = mask.cols(); val h = mask.rows()
        val mk = ByteArray(w * h); mask.get(0, 0, mk)
        fun inside(x: Double, y: Double): Boolean { val xi = x.toInt(); val yi = y.toInt(); return xi in 0 until w && yi in 0 until h && mk[yi * w + xi].toInt() != 0 }
        val minArc = cfg.neckMinArc * size
        val step = maxOf(1, (size * 0.03).toInt())
        // Boundary samples of every clump: x, y, boundary id, position along it, boundary length.
        class P(val x: Double, val y: Double, val c: Int, val t: Int, val len: Int)
        val pts = ArrayList<P>()
        for (ci in contours.indices) {
            val parent = hi[ci * 4 + 3]
            val outer = if (parent < 0) ci else parent
            if (Imgproc.contourArea(contours[outer]) < cfg.splitAbove * pieceArea) continue
            // Small holes are print colours close to the table's, not gaps between pieces.
            if (parent >= 0 && Imgproc.contourArea(contours[ci]) < cfg.minGapArea * pieceArea) continue
            val a = contours[ci].toArray()
            var t = 0
            while (t < a.size) { pts.add(P(a[t].x, a[t].y, ci, t, a.size)); t += step }
        }
        if (pts.size < 4) return mask
        // Spatial grid for near-neighbour search.
        val cell = maxCut.coerceAtLeast(2.0)
        val grid = HashMap<Long, ArrayList<Int>>()
        fun key(gx: Int, gy: Int) = gx.toLong() * 100003L + gy
        for ((i, p) in pts.withIndex()) grid.getOrPut(key((p.x / cell).toInt(), (p.y / cell).toInt())) { ArrayList() }.add(i)
        class Cut(val a: P, val b: P, val d: Double)
        val cuts = ArrayList<Cut>()
        for ((i, p) in pts.withIndex()) {
            val gx = (p.x / cell).toInt(); val gy = (p.y / cell).toInt()
            for (dy in -1..1) for (dx in -1..1) for (j in grid[key(gx + dx, gy + dy)] ?: continue) {
                if (j <= i) continue
                val q = pts[j]
                val d = hypot(p.x - q.x, p.y - q.y)
                if (d > maxCut) continue
                if (p.c == q.c) { val along = kotlin.math.abs(p.t - q.t); if (minOf(along, p.len - along) < minArc) continue }
                // The cut must run through the pieces, not across open table.
                var ok = true
                for (k in 1..4) { val f = k / 5.0; if (!inside(p.x + (q.x - p.x) * f, p.y + (q.y - p.y) * f)) { ok = false; break } }
                if (ok) cuts.add(Cut(p, q, d))
            }
        }
        if (cuts.isEmpty()) return mask
        cuts.sortBy { it.d }
        val out = mask.clone()
        val taken = ArrayList<DoubleArray>()
        for (c in cuts) {
            val mx = (c.a.x + c.b.x) / 2; val my = (c.a.y + c.b.y) / 2
            if (taken.any { hypot(it[0] - mx, it[1] - my) < 0.6 * maxCut }) continue
            taken.add(doubleArrayOf(mx, my))
            // Extend a little past both ends so the line really separates the two sides.
            val l = c.d.coerceAtLeast(1e-6); val ux = (c.b.x - c.a.x) / l; val uy = (c.b.y - c.a.y) / l
            Imgproc.line(out, Point(c.a.x - 2 * ux, c.a.y - 2 * uy), Point(c.b.x + 2 * ux, c.b.y + 2 * uy), Scalar(0.0), 2)
        }
        return out
    }

    /**
     * Two pieces that touch are joined by a neck: a place where two far-apart stretches of the outline
     * come close together, with at least most of a piece on either side. (A tab's own neck doesn't
     * qualify: the knob beyond it is far too small.) The blob is cut at its narrowest such neck, and
     * the two parts are cut again while they are still bigger than a piece.
     */
    private fun splitAtNecks(pts: Array<Point>, size: Double, pieceArea: Double, depth: Int): List<Array<Point>> {
        val n = pts.size
        val total = polyArea(pts, 0, n)
        if (depth > 120 || total < cfg.splitAbove * pieceArea || n < 40) return listOf(pts)
        // Search on a subsample, cut on the full outline.
        // Sample about every 5% of a piece's size, at most 2000 points.
        val step = maxOf(1, minOf((size * 0.05).toInt(), n / 300), n / 2000)
        val idx = IntArray((n + step - 1) / step) { it * step }
        val m = idx.size
        // Prefix sums of the shoelace terms over the full outline.
        val pre = DoubleArray(n + 1)
        for (t in 0 until n) { val a = pts[t]; val b = pts[(t + 1) % n]; pre[t + 1] = pre[t] + (a.x * b.y - b.x * a.y) }
        fun arcArea(i: Int, j: Int): Double { // polygon: outline from i to j, closed by the chord j -> i
            val a = pts[i]; val b = pts[j]
            return kotlin.math.abs(pre[j] - pre[i] + (b.x * a.y - a.x * b.y)) / 2
        }
        val maxCut = cfg.neckWidth * size
        val minPart = cfg.neckMinPart * pieceArea
        var best = Double.MAX_VALUE; var bi = -1; var bj = -1
        val poly = org.opencv.core.MatOfPoint2f(*pts)
        for (u in 0 until m) for (v in u + 1 until m) {
            val i = idx[u]; val j = idx[v]
            val d = hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y)
            if (d >= best || d > maxCut) continue
            val a = arcArea(i, j)
            if (a < minPart || total - a < minPart) continue
            // The cut must run through the blob, not across a gap.
            val mid = Point((pts[i].x + pts[j].x) / 2, (pts[i].y + pts[j].y) / 2)
            if (d > 3 && Imgproc.pointPolygonTest(poly, mid, false) < 0) continue
            best = d; bi = i; bj = j
        }
        if (bi < 0) return listOf(pts)
        val a = Array(bj - bi + 1) { pts[bi + it] }
        val b = Array(n - (bj - bi) + 1) { pts[(bj + it) % n] }
        return splitAtNecks(a, size, pieceArea, depth + 1) + splitAtNecks(b, size, pieceArea, depth + 1)
    }

    private fun polyArea(pts: Array<Point>, from: Int, to: Int): Double {
        var s = 0.0
        for (t in from until to) { val a = pts[t]; val b = pts[if (t + 1 == to) from else t + 1]; s += a.x * b.y - b.x * a.y }
        return kotlin.math.abs(s) / 2
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

    /**
     * High-resolution variant for big stills: pieces are found on a copy reduced by [factor] (fast,
     * little memory), then each piece's outline is refined and measured on the full-resolution image
     * inside its own small window. Coordinates in the result are full-resolution.
     */
    fun processHiRes(full: Mat, valid: Mat?, lean: Lean?, factor: Int): Out {
        if (factor <= 1) return process(full, valid, lean)
        val t0 = System.currentTimeMillis()
        val small = Mat(); Imgproc.resize(full, small, org.opencv.core.Size(full.cols() / factor.toDouble(), full.rows() / factor.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val validSmall = valid?.let { v -> Mat().also { Imgproc.resize(v, it, small.size(), 0.0, 0.0, Imgproc.INTER_NEAREST) } }
        val seg = TableSegmenter.segment(small, validSmall, cfg.segmentation)
        val t1 = System.currentTimeMillis()
        val sizeSmall = pieceSize(seg.raw.filter { !it.touchesBorder }, if (seg.bodyRadius > 0) seg.bodyRadius * cfg.sizePerRadius else 0.0)
        val usable = candidates(seg, sizeSmall, validSmall)
        val leanSmall = lean?.let { Lean(it.nadirX / factor, it.nadirY / factor, it.focal / factor) }
        val pieces = ArrayList<Piece>()
        val loose = ArrayList<kotlin.Pair<DoubleArray, DoubleArray>>()
        val rejects = ArrayList<Triple<Double, Double, String>>()
        val w = full.cols(); val h = full.rows()
        fun note(b: TableSegmenter.Blob, why: String) { val r = Imgproc.boundingRect(b.contour); rejects.add(Triple((r.x + r.width / 2.0) * factor, (r.y + r.height / 2.0) * factor, why)) }
        for (b in seg.blobs) if (b.touchesBorder) note(b, "edge-of-photo")
        for (b in usable) {
            if (b.area > 2.6 * sizeSmall * sizeSmall) { note(b, "big %.1f".format(b.area / (sizeSmall * sizeSmall))); continue }
            if (b.area < 0.35 * sizeSmall * sizeSmall) { if (b.area > 0.08 * sizeSmall * sizeSmall) note(b, "small %.2f".format(b.area / (sizeSmall * sizeSmall))); continue }
            val pts = faceOutline(b, leanSmall, sizeSmall) ?: continue
            // Window around the piece in the full image.
            val fx = DoubleArray(pts.size) { (pts[it].x + 0.5) * factor }; val fy = DoubleArray(pts.size) { (pts[it].y + 0.5) * factor }
            val margin = (0.12 * sizeSmall * factor).toInt() + 4
            val x0 = (fx.min().toInt() - margin).coerceAtLeast(0); val y0 = (fy.min().toInt() - margin).coerceAtLeast(0)
            val x1 = (fx.max().toInt() + margin).coerceAtMost(w - 1); val y1 = (fy.max().toInt() + margin).coerceAtMost(h - 1)
            val rw = x1 - x0 + 1; val rh = y1 - y0 + 1
            if (rw < 16 || rh < 16) continue
            val roi = full.submat(Rect(x0, y0, rw, rh))
            val raw = ByteArray(rw * rh * 3)
            if (roi.isContinuous) roi.get(0, 0, raw) else roi.clone().get(0, 0, raw)
            val px = IntArray(rw * rh) { val k = it * 3; (0xFF shl 24) or ((raw[k + 2].toInt() and 0xFF) shl 16) or ((raw[k + 1].toInt() and 0xFF) shl 8) or (raw[k].toInt() and 0xFF) }
            val img = RgbImage(rw, rh, px)
            var xs = DoubleArray(fx.size) { fx[it] - x0 }; var ys = DoubleArray(fy.size) { fy[it] - y0 }
            // The reduced outline is only accurate to about one reduced pixel: snap it to the real edge.
            if (cfg.hiResRefine) {
                val r = OutlineRefiner.refine(xs, ys, OutlineRefiner.luminance(img), rw, rh, (factor * 1.5).coerceAtLeast(cfg.refineRange * sizeSmall * factor))
                xs = r.first; ys = r.second
            }
            val p = detector.analyzeContour(xs, ys, img, pieces.size)
            val gx = DoubleArray(xs.size) { xs[it] + x0 }; val gy = DoubleArray(ys.size) { ys[it] + y0 }
            if (p == null || p.quality < cfg.minQuality || !detector.isPlausible(p)) {
                debugReject?.invoke(img, xs, ys, if (p == null) "corners" else if (p.quality < cfg.minQuality) "q%.2f".format(p.quality) else "shape")
                rejects.add(Triple(gx.average(), gy.average(), if (p == null) "corners" else if (p.quality < cfg.minQuality) "q%.2f".format(p.quality) else "shape"))
                loose.add(gx to gy); continue
            }
            pieces.add(translate(p, x0.toDouble(), y0.toDouble()))
        }
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
        return Out(kept, border, emptyList(), loose, seg.blobs.size, rejects, seg.bodyRadius * factor, t1 - t0, System.currentTimeMillis() - t1, 0)
    }

    private fun translate(p: Piece, dx: Double, dy: Double): Piece = Piece(
        p.id, DoubleArray(p.xs.size) { p.xs[it] + dx }, DoubleArray(p.ys.size) { p.ys[it] + dy }, p.cornerIdx,
        p.edges.map { e -> Edge(e.pieceId, e.index, e.type, e.length, e.cx, e.cy, e.colors, e.startX + dx, e.startY + dy, e.endX + dx, e.endY + dy, e.midX + dx, e.midY + dy, e.strip, e.stripRows, e.stripLen, e.palette) },
        p.centroidX + dx, p.centroidY + dy, p.scale, p.area, p.quality, p.palette,
    )
}
