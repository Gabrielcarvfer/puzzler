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
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.SIFT
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot

/**
 * A growing catalogue of every piece seen so far.
 *
 * A piece is recognised again by its printed picture (SIFT features inside its outline, verified by
 * a similarity transform), so it may be moved, rotated or seen from another angle. Each sighting adds
 * one more measurement of its four sides; the catalogue keeps their average, which is more precise
 * than any single view. Pairs are then solved on the averaged shapes.
 */
class PieceCatalogue(val cfg: Config = Config()) {
    data class Config(
        val minFeatures: Int = 8,
        val ratio: Float = 0.8f,
        val minInliers: Int = 6,
        /** Shape fallback: all four sides must agree within this gap (fraction of edge length). */
        val shapeReId: Double = 0.022,
        /** Duplicate merging: worst side gap allowed on shape alone / when the pictures also match. */
        val mergeTight: Double = 0.016,
        val mergeLoose: Double = 0.03,
        /** Samples per side (must match the detector's edgeSamples). */
        val samples: Int = 96,
        /** Pair acceptance on averaged shapes. */
        val maxGap: Double = 0.016,
        val maxRatio: Double = 0.8,
        /** Pairs whose picture differs more than this across the joint are not considered (1 = off). */
        val maxColour: Double = 1.0,
        /** Views of its print kept per piece, and the minimum number of photos between two kept views. */
        val maxViews: Int = 4,
        val viewSpacing: Int = 3,
    )

    class Side(val type: EdgeType, val x: DoubleArray, val y: DoubleArray, var length: Double, var n: Int, var colors: FloatArray)

    class Entry(val id: Int, val descriptors: Mat, val keypoints: Array<Point>, val sides: Array<Side>) {
        var sightings = 1
        /** Unit vectors from the piece centre to each side's midpoint, in the entry's own frame. */
        lateinit var sideDirs: Array<DoubleArray>
        /** Where it was last seen (frame coordinates), in which pass, and each side's direction there. */
        var lastX = 0.0; var lastY = 0.0; var lastPass = -100
        var lastDirs: Array<DoubleArray>? = null
        /** Further views of the same piece (other angles, light, sharpness), besides the first one above. */
        val extraViews = ArrayList<View>()
        var lastViewPass = -100
        /** All stored views of the piece's print, the first sighting first. */
        fun views(): List<View> = listOf(View(descriptors, keypoints, sideDirs)) + extraViews
    }

    /**
     * One view of a piece's printed picture: its features, their positions relative to the piece's
     * centre (in units of piece size), and where the entry's four sides point in that view.
     */
    class View(val descriptors: Mat, val keypoints: Array<Point>, val sideDirs: Array<DoubleArray>)

    class Sighting(val piece: Piece, val entry: Entry, val isNew: Boolean, /** piece.edges[k] is entry.sides[map[k]] */ val map: IntArray)

    /** [colour]: how different the picture is on the two sides of the joint, 0 (same) to 1; -1 if unknown. */
    class Pair(val a: Entry, val sideA: Int, val b: Entry, val sideB: Int, val gap: Double, val runnerUp: Double, val colour: Double = -1.0)

    val entries = ArrayList<Entry>()
    private var passNo = 0
    private var nextId = 0

    /** Tells the catalogue where [e] is expected in the next frame (e.g. from registering two photos). */
    fun hintPosition(e: Entry, x: Double, y: Double, dirs: Array<DoubleArray>) { e.lastX = x; e.lastY = y; e.lastDirs = dirs; e.lastPass = passNo }

    /**
     * The camera moved between the previous frame and the next one: [h] (row-major 3x3) maps previous
     * frame coordinates to the new frame. Recently seen pieces keep their identity by position.
     */
    fun applyCameraMotion(h: DoubleArray) {
        fun map(x: Double, y: Double): DoubleArray { val w = h[6] * x + h[7] * y + h[8]; return doubleArrayOf((h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w) }
        for (e in entries) {
            if (passNo - e.lastPass > 3) continue
            val dirs = e.lastDirs ?: continue
            val c = map(e.lastX, e.lastY)
            val nd = Array(4) { i ->
                val q = map(e.lastX + 50 * dirs[i][0], e.lastY + 50 * dirs[i][1])
                val dx = q[0] - c[0]; val dy = q[1] - c[1]; val l = hypot(dx, dy).coerceAtLeast(1e-9)
                doubleArrayOf(dx / l, dy / l)
            }
            e.lastX = c[0]; e.lastY = c[1]; e.lastDirs = nd
        }
    }

    private fun merge(e: Entry, p: Piece, map: IntArray, k: Double) {
        for (i in 0 until 4) {
            val sd = e.sides[map[i]]; val ed = p.edges[i]
            // Running average, capped so the entry keeps adapting.
            val n = minOf(sd.n, 30)
            for (t in 0 until cfg.samples) { sd.x[t] = (sd.x[t] * n + ed.cx[t] * k) / (n + 1); sd.y[t] = (sd.y[t] * n + ed.cy[t] * k) / (n + 1) }
            sd.length = (sd.length * n + ed.length * k) / (n + 1); sd.n++
            if (sd.colors.size == ed.colors.size) for (t in sd.colors.indices) sd.colors[t] = ((sd.colors[t] * n + ed.colors[t]) / (n + 1)).toFloat()
            else if (sd.colors.isEmpty()) sd.colors = ed.colors.copyOf()
        }
        e.sightings++
    }

    /**
     * Keeps another view of a recognised piece (at most [Config.maxViews] in total, not from
     * back-to-back photos). When full, the new view replaces the poorest extra view if it is richer.
     */
    private fun addView(e: Entry, desc: Mat, pts: Array<Point>, dirs: Array<DoubleArray>, map: IntArray) {
        if (desc.rows() < cfg.minFeatures || passNo - e.lastViewPass < cfg.viewSpacing) return
        val vd = Array(4) { DoubleArray(2) }
        for (i in 0 until 4) vd[map[i]] = dirs[i]
        val v = View(desc, pts, vd)
        if (e.extraViews.size < cfg.maxViews - 1) e.extraViews.add(v)
        else {
            val worst = e.extraViews.indices.minBy { e.extraViews[it].descriptors.rows() }
            if (desc.rows() > e.extraViews[worst].descriptors.rows() * 1.2) e.extraViews[worst] = v else return
        }
        e.lastViewPass = passNo
    }

    private fun touch(e: Entry, p: Piece, map: IntArray, dirs: Array<DoubleArray>) {
        e.lastX = p.centroidX; e.lastY = p.centroidY; e.lastPass = passNo
        val ld = Array(4) { DoubleArray(2) }
        for (i in 0 until 4) ld[map[i]] = dirs[i]
        e.lastDirs = ld
    }
    private val sift = SIFT.create(0, 3, 0.03, 10.0, 1.6)
    private val bf = BFMatcher.create(Core.NORM_L2, false)
    private val scorer = MateScorer()

    /** Adds this frame's cleanly outlined pieces: each is matched to an existing entry or becomes a new one. */
    fun observe(frameBgr: Mat, pieces: List<Piece>): List<Sighting> {
        val gray = Mat(); Imgproc.cvtColor(frameBgr, gray, Imgproc.COLOR_BGR2GRAY)
        val out = ArrayList<Sighting>()
        passNo++
        for (p in pieces) {
            // Features on the piece's face only.
            val minX = p.xs.min().toInt().coerceAtLeast(0); val minY = p.ys.min().toInt().coerceAtLeast(0)
            val maxX = p.xs.max().toInt().coerceAtMost(gray.cols() - 1); val maxY = p.ys.max().toInt().coerceAtMost(gray.rows() - 1)
            if (maxX - minX < 16 || maxY - minY < 16) continue
            val roi = Rect(minX, minY, maxX - minX + 1, maxY - minY + 1)
            val mask = Mat.zeros(roi.height, roi.width, CvType.CV_8UC1)
            Imgproc.fillPoly(mask, listOf(MatOfPoint(*Array(p.xs.size) { Point(p.xs[it] - minX, p.ys[it] - minY) })), Scalar(255.0))
            Imgproc.erode(mask, mask, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, org.opencv.core.Size(5.0, 5.0)))
            val kpM = MatOfKeyPoint(); val desc = Mat()
            sift.detectAndCompute(gray.submat(roi), mask, kpM, desc)
            val kps: Array<KeyPoint> = kpM.toArray()
            // Keypoints relative to the piece centre, in units of piece size.
            val pts = Array(kps.size) { Point((kps[it].pt.x + minX - p.centroidX) / p.scale, (kps[it].pt.y + minY - p.centroidY) / p.scale) }
            val dirs = Array(4) { k -> val e = p.edges[k]; val dx = (e.startX + e.endX) / 2 - p.centroidX; val dy = (e.startY + e.endY) / 2 - p.centroidY; val l = hypot(dx, dy); doubleArrayOf(dx / l, dy / l) }

            val k = 100.0 / p.scale
            // 1) Same place as in a recent pass (camera on a tripod, piece not moved): same piece.
            val here = entries.filter { passNo - it.lastPass <= 3 && it.lastDirs != null && hypot(it.lastX - p.centroidX, it.lastY - p.centroidY) < 0.25 * p.scale }
                .minByOrNull { hypot(it.lastX - p.centroidX, it.lastY - p.centroidY) }
            if (here != null && p.edges.all { it.cx.size == cfg.samples }) {
                val ld = here.lastDirs!!
                val map = IntArray(4) { i -> (0 until 4).maxBy { j -> dirs[i][0] * ld[j][0] + dirs[i][1] * ld[j][1] } }
                if (map.toSet().size == 4 && (0 until 4).all { p.edges[it].type == here.sides[map[it]].type }) {
                    merge(here, p, map, k); touch(here, p, map, dirs); addView(here, desc, pts, dirs, map)
                    out.add(Sighting(p, here, false, map))
                    continue
                }
            }
            // A piece exactly where a known piece was, but whose sides were read differently this time
            // (corners misjudged): a bad measurement of the same piece, not a new piece. Skip it.
            if (here != null) { here.lastPass = passNo; continue }
            // 2) Recognised by its printed picture.
            var best: Entry? = null; var bestInl = 0; var bestRot = 0.0
            if (kps.size >= cfg.minFeatures) for (e in entries) {
                if (e.descriptors.rows() < cfg.minFeatures) continue
                val knn = ArrayList<MatOfDMatch>(); bf.knnMatch(desc, e.descriptors, knn, 2)
                val good = ArrayList<DMatch>()
                for (k in knn) { val m = k.toArray(); if (m.size == 2 && m[0].distance < cfg.ratio * m[1].distance) good.add(m[0]) }
                if (good.size < cfg.minInliers) continue
                val from = MatOfPoint2f(*good.map { pts[it.queryIdx] }.toTypedArray())
                val to = MatOfPoint2f(*good.map { e.keypoints[it.trainIdx] }.toTypedArray())
                val inl = Mat()
                val a = Calib3d.estimateAffinePartial2D(from, to, inl, Calib3d.RANSAC, 0.06, 2000, 0.99, 10)
                if (a.empty()) continue
                val n = Core.countNonZero(inl)
                val s = hypot(a.get(0, 0)[0], a.get(1, 0)[0])
                if (n > bestInl && s in 0.8..1.25) { bestInl = n; best = e; bestRot = atan2(a.get(1, 0)[0], a.get(0, 0)[0]) }
            }
            if (best != null && bestInl >= cfg.minInliers && p.edges.all { it.cx.size == cfg.samples }) {
                val c = kotlin.math.cos(bestRot); val s = kotlin.math.sin(bestRot)
                val map = IntArray(4) { i ->
                    val rx = c * dirs[i][0] - s * dirs[i][1]; val ry = s * dirs[i][0] + c * dirs[i][1]
                    (0 until 4).maxBy { j -> rx * best.sideDirs[j][0] + ry * best.sideDirs[j][1] }
                }
                if (map.toSet().size == 4 && (0 until 4).all { p.edges[it].type == best.sides[map[it]].type }) {
                    merge(best, p, map, k); touch(best, p, map, dirs); addView(best, desc, pts, dirs, map)
                    out.add(Sighting(p, best, false, map))
                    continue
                }
            }
            if (p.edges.any { it.cx.size != cfg.samples }) continue
            // Fallback for plain pieces: same sequence of side types and all four side shapes agree.
            val shapeHit = reIdByShape(p, k)
            if (shapeHit != null) {
                val (e, rot) = shapeHit
                val map = IntArray(4) { (it + rot) % 4 }
                merge(e, p, map, k); touch(e, p, map, dirs); addView(e, desc, pts, dirs, map)
                out.add(Sighting(p, e, false, map))
                continue
            }
            val entry = Entry(nextId++, desc, pts, Array(4) { i -> val e = p.edges[i]; Side(e.type, DoubleArray(cfg.samples) { e.cx[it] * k }, DoubleArray(cfg.samples) { e.cy[it] * k }, e.length * k, 1, e.colors) })
            entry.sideDirs = dirs
            touch(entry, p, intArrayOf(0, 1, 2, 3), dirs)
            entries.add(entry)
            out.add(Sighting(p, entry, true, intArrayOf(0, 1, 2, 3)))
        }
        return out
    }

    private val loose = MateScorer(MateScorer.Config(maxLengthDiff = 0.05, maxBulgeDiff = 0.05, maxPositionDiff = 0.06))

    /** Entry whose four sides (at some rotation) all match this piece's sides; must be the only such entry. */
    private fun reIdByShape(p: Piece, k: Double): kotlin.Pair<Entry, Int>? {
        var hit: kotlin.Pair<Entry, Int>? = null; var hits = 0
        for (e in entries) for (rot in 0 until 4) {
            if ((0 until 4).any { p.edges[it].type != e.sides[(it + rot) % 4].type }) continue
            var worst = 0.0
            for (i in 0 until 4) {
                val ed = p.edges[i]
                if (ed.type == EdgeType.FLAT) { if (kotlin.math.abs(ed.length * k - e.sides[(i + rot) % 4].length) > 4.0) { worst = 1.0; break }; continue }
                // Compare with the stored side by mating against its mirror image.
                val sd = e.sides[(i + rot) % 4]; val n = sd.x.size
                val mirror = Edge(-1, 0, if (sd.type == EdgeType.TAB) EdgeType.BLANK else EdgeType.TAB, sd.length, DoubleArray(n) { sd.length - sd.x[n - 1 - it] }, DoubleArray(n) { -sd.y[n - 1 - it] }, sd.colors, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                val probe = Edge(-2, 0, ed.type, ed.length * k, DoubleArray(n) { ed.cx[it] * k }, DoubleArray(n) { ed.cy[it] * k }, ed.colors, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                val g = loose.fit(probe, mirror)?.gap ?: 1.0
                if (g > worst) worst = g
                if (worst > cfg.shapeReId) break
            }
            if (worst <= cfg.shapeReId) { hits++; hit = e to rot }
        }
        return if (hits == 1) hit else null
    }

    private fun sideGap(a: Side, b: Side): Double {
        if (a.type != b.type) return 1.0
        if (a.type == EdgeType.FLAT) return if (kotlin.math.abs(a.length - b.length) > 4.0) 1.0 else 0.0
        val n = b.x.size
        val mirror = Edge(-1, 0, if (b.type == EdgeType.TAB) EdgeType.BLANK else EdgeType.TAB, b.length, DoubleArray(n) { b.length - b.x[n - 1 - it] }, DoubleArray(n) { -b.y[n - 1 - it] }, b.colors, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        val probe = Edge(-2, 0, a.type, a.length, a.x, a.y, a.colors, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return loose.fit(probe, mirror)?.gap ?: 1.0
    }

    /** Number of picture features of [a] and [b] that agree on one rigid placement (0 if either has too few). */
    private fun pictureInliers(a: Entry, b: Entry): Int {
        if (a.descriptors.rows() < cfg.minFeatures || b.descriptors.rows() < cfg.minFeatures) return 0
        val knn = ArrayList<MatOfDMatch>(); bf.knnMatch(a.descriptors, b.descriptors, knn, 2)
        val good = ArrayList<DMatch>()
        for (k in knn) { val m = k.toArray(); if (m.size == 2 && m[0].distance < cfg.ratio * m[1].distance) good.add(m[0]) }
        if (good.size < cfg.minInliers) return 0
        val inl = Mat()
        val t = Calib3d.estimateAffinePartial2D(MatOfPoint2f(*good.map { a.keypoints[it.queryIdx] }.toTypedArray()), MatOfPoint2f(*good.map { b.keypoints[it.trainIdx] }.toTypedArray()), inl, Calib3d.RANSAC, 0.06, 2000, 0.99, 10)
        return if (t.empty()) 0 else Core.countNonZero(inl)
    }

    /**
     * Merges entries that are the same physical piece catalogued twice: all four side shapes agree
     * (closely, or loosely when the printed pictures also match). Their measurements are pooled.
     * Returns (kept id, removed id) for every merge.
     */
    fun mergeDuplicates(): List<kotlin.Pair<Int, Int>> {
        class Cand(val a: Entry, val b: Entry, val rot: Int, val worst: Double)
        val cands = ArrayList<Cand>()
        for (i in entries.indices) for (j in i + 1 until entries.size) {
            val a = entries[i]; val b = entries[j]
            for (rot in 0 until 4) {
                if ((0 until 4).any { a.sides[it].type != b.sides[(it + rot) % 4].type }) continue
                var worst = 0.0
                for (k in 0 until 4) { worst = maxOf(worst, sideGap(a.sides[k], b.sides[(k + rot) % 4])); if (worst > cfg.mergeLoose) break }
                if (worst > cfg.mergeLoose) continue
                if (worst <= cfg.mergeTight || pictureInliers(a, b) >= cfg.minInliers) cands.add(Cand(a, b, rot, worst))
            }
        }
        cands.sortBy { it.worst }
        val gone = HashSet<Int>(); val out = ArrayList<kotlin.Pair<Int, Int>>()
        for (c in cands) {
            if (c.a.id in gone || c.b.id in gone) continue
            for (k in 0 until 4) {
                val sa = c.a.sides[k]; val sb = c.b.sides[(k + c.rot) % 4]
                val wa = minOf(sa.n, 30).toDouble(); val wb = minOf(sb.n, 30).toDouble()
                for (t in sa.x.indices) { sa.x[t] = (sa.x[t] * wa + sb.x[t] * wb) / (wa + wb); sa.y[t] = (sa.y[t] * wa + sb.y[t] * wb) / (wa + wb) }
                sa.length = (sa.length * wa + sb.length * wb) / (wa + wb); sa.n += sb.n
            }
            c.a.sightings += c.b.sightings
            // b's views join a's: b's side k' is a's side k where k' = (k + rot) % 4.
            for (v in c.b.views()) if (c.a.extraViews.size < cfg.maxViews - 1 && v.descriptors.rows() >= cfg.minFeatures)
                c.a.extraViews.add(View(v.descriptors, v.keypoints, Array(4) { k -> v.sideDirs[(k + c.rot) % 4] }))
            if (c.b.lastPass > c.a.lastPass) {
                // Keep tracking at the most recent position; b's side k' is a's side k where k' = (k + rot) % 4.
                c.a.lastPass = c.b.lastPass; c.a.lastX = c.b.lastX; c.a.lastY = c.b.lastY
                c.b.lastDirs?.let { bd -> c.a.lastDirs = Array(4) { k -> bd[(k + c.rot) % 4] } }
            }
            gone.add(c.b.id); out.add(c.a.id to c.b.id)
        }
        entries.removeAll { it.id in gone }
        return out
    }

    /** How well side [sideA] of [a] mates with side [sideB] of [b] (gap / edge length), or null if they can't. */
    fun mateGap(a: Entry, sideA: Int, b: Entry, sideB: Int): Double? = scorer.fit(edgeOf(a, sideA), edgeOf(b, sideB))?.gap

    private fun edgeOf(e: Entry, i: Int): Edge { val s = e.sides[i]; return Edge(e.id, i, s.type, s.length, s.x, s.y, s.colors, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) }

    /** Mutual-best tab/blank fits on the averaged shapes, clearly ahead of each side's runner-up. */
    fun solve(minSightings: Int = 1): List<Pair> {
        val sides = entries.filter { it.sightings >= minSightings }.flatMap { e -> (0 until 4).filter { e.sides[it].type != EdgeType.FLAT }.map { e to it } }
        val n = sides.size
        val best = IntArray(n) { -1 }; val bestG = DoubleArray(n) { Double.MAX_VALUE }; val second = DoubleArray(n) { Double.MAX_VALUE }
        fun offer(i: Int, j: Int, g: Double) { if (g < bestG[i]) { second[i] = bestG[i]; bestG[i] = g; best[i] = j } else if (g < second[i]) second[i] = g }
        for (i in 0 until n) for (j in i + 1 until n) {
            // The picture continues across a joint: sides whose colours along the cut differ a lot can't be partners.
            if (cfg.maxColour < 1.0) { val c = colourAcross(sides[i].first.sides[sides[i].second], sides[j].first.sides[sides[j].second]); if (c > cfg.maxColour) continue }
            val g = scorer.fit(edgeOf(sides[i].first, sides[i].second), edgeOf(sides[j].first, sides[j].second))?.gap ?: continue
            offer(i, j, g); offer(j, i, g)
        }
        val out = ArrayList<Pair>()
        for (i in 0 until n) {
            val j = best[i]
            if (j <= i || best[j] != i || bestG[i] > cfg.maxGap) continue
            val runner = minOf(second[i], second[j])
            if (bestG[i] > cfg.maxRatio * runner) continue
            out.add(Pair(sides[i].first, sides[i].second, sides[j].first, sides[j].second, bestG[i], runner, colourAcross(sides[i].first.sides[sides[i].second], sides[j].first.sides[sides[j].second])))
        }
        return out.sortedBy { it.gap }
    }

    /**
     * Colour difference across a would-be joint, 0..1: the colours just inside side A, compared with
     * those just inside side B in reverse order (the two sides face each other). Each side is first
     * scaled to the same overall brightness, since the two pieces may have been photographed under
     * different light. Returns -1 when either side has no colours stored.
     */
    fun colourAcross(a: Side, b: Side): Double {
        val k = a.colors.size / 3
        if (k == 0 || b.colors.size != a.colors.size) return -1.0
        var ma = 0.0; var mb = 0.0
        for (v in a.colors) ma += v; for (v in b.colors) mb += v
        ma = (ma / a.colors.size).coerceAtLeast(8.0); mb = (mb / b.colors.size).coerceAtLeast(8.0)
        var d = 0.0
        for (i in 0 until k) { val j = k - 1 - i; for (c in 0 until 3) d += abs(a.colors[i * 3 + c] / ma - b.colors[j * 3 + c] / mb) }
        return (d / (k * 3) / 2).coerceIn(0.0, 1.0)
    }

    /** Writes the whole catalogue to [file] (atomic: temp file + rename). */
    fun save(file: java.io.File) {
        val tmp = java.io.File(file.path + ".tmp")
        java.io.DataOutputStream(java.io.BufferedOutputStream(tmp.outputStream())).use { o ->
            o.writeInt(0x504D4338); o.writeInt(entries.size)
            for (e in entries) {
                o.writeInt(e.sightings)
                for (d in e.sideDirs) { o.writeDouble(d[0]); o.writeDouble(d[1]) }
                for (sd in e.sides) {
                    o.writeInt(sd.type.ordinal); o.writeInt(sd.n); o.writeDouble(sd.length); o.writeInt(sd.x.size)
                    for (v in sd.x) o.writeDouble(v); for (v in sd.y) o.writeDouble(v)
                    o.writeInt(sd.colors.size); for (v in sd.colors) o.writeFloat(v)
                }
                o.writeInt(e.keypoints.size); for (k in e.keypoints) { o.writeDouble(k.x); o.writeDouble(k.y) }
                val rows = e.descriptors.rows(); val cols = e.descriptors.cols()
                o.writeInt(rows); o.writeInt(cols)
                if (rows > 0) { val f = FloatArray(rows * cols); e.descriptors.get(0, 0, f); for (v in f) o.writeFloat(v) }
                o.writeInt(e.extraViews.size)
                for (v in e.extraViews) {
                    for (d in v.sideDirs) { o.writeDouble(d[0]); o.writeDouble(d[1]) }
                    o.writeInt(v.keypoints.size); for (k in v.keypoints) { o.writeDouble(k.x); o.writeDouble(k.y) }
                    val r = v.descriptors.rows(); val c = v.descriptors.cols()
                    o.writeInt(r); o.writeInt(c)
                    if (r > 0) { val f = FloatArray(r * c); v.descriptors.get(0, 0, f); for (x in f) o.writeFloat(x) }
                }
            }
        }
        tmp.renameTo(file)
    }

    /** Replaces the catalogue with the contents of [file]; returns false if it is missing or unreadable. */
    fun load(file: java.io.File): Boolean {
        if (!file.exists()) return false
        return try {
            val loaded = ArrayList<Entry>()
            java.io.DataInputStream(java.io.BufferedInputStream(file.inputStream())).use { i ->
                if (i.readInt() != 0x504D4338) return false
                repeat(i.readInt()) { id ->
                    val sightings = i.readInt()
                    val dirs = Array(4) { doubleArrayOf(i.readDouble(), i.readDouble()) }
                    val sides = Array(4) {
                        val type = EdgeType.entries[i.readInt()]; val n = i.readInt(); val len = i.readDouble(); val m = i.readInt()
                        val x = DoubleArray(m) { i.readDouble() }; val y = DoubleArray(m) { i.readDouble() }
                        val col = FloatArray(i.readInt()) { i.readFloat() }
                        Side(type, x, y, len, n, col)
                    }
                    val kp = Array(i.readInt()) { Point(i.readDouble(), i.readDouble()) }
                    val rows = i.readInt(); val cols = i.readInt()
                    val desc = if (rows > 0) Mat(rows, cols, CvType.CV_32F).also { m -> m.put(0, 0, FloatArray(rows * cols) { i.readFloat() }) } else Mat()
                    val entry = Entry(id, desc, kp, sides).also { it.sightings = sightings; it.sideDirs = dirs }
                    repeat(i.readInt()) {
                        val vd = Array(4) { doubleArrayOf(i.readDouble(), i.readDouble()) }
                        val vk = Array(i.readInt()) { Point(i.readDouble(), i.readDouble()) }
                        val r = i.readInt(); val c = i.readInt()
                        val vm = if (r > 0) Mat(r, c, CvType.CV_32F).also { m -> m.put(0, 0, FloatArray(r * c) { i.readFloat() }) } else Mat()
                        entry.extraViews.add(View(vm, vk, vd))
                    }
                    loaded.add(entry)
                }
            }
            entries.clear(); entries.addAll(loaded); nextId = loaded.size; true
        } catch (e: Exception) { false }
    }

    fun clear() { entries.clear(); nextId = 0 }
}
