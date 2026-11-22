package puzzlemaster.core

import java.util.PriorityQueue
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Shape/edge-driven piece finder. No background colour model.
 *
 * Pieces are 3D: their cut sides and shadows draw a strong, continuous edge around every piece,
 * even when pieces touch or lie on top of each other. The image is over-segmented along edges
 * (watershed superpixels), then neighbouring regions are merged weakest-boundary-first. Picture
 * details inside a piece merge early; a piece's outline survives longest. Every intermediate region
 * of plausible size is tested against the puzzle-piece shape model (4 corners, tabs/blanks), and the
 * best-fitting non-overlapping regions are returned.
 */
class EdgeSegmenter(val cfg: Config = Config()) {
    data class Config(
        /** Working resolution (long side). */
        val workSize: Int = 1600,
        /** Watershed seed cell size (pixels at working resolution). */
        val seedCell: Int = 7,
        val blurSigma: Double = 1.4,
        /** Candidate piece area as a fraction of the image. */
        val minAreaFraction: Double = 0.0004,
        val maxAreaFraction: Double = 0.04,
        /** Outline must be this much stronger than the strongest edge merged inside the region. */
        /** Mean outline edge strength (0..255) a piece must have: cut edges + shadows are strong. */
        val minOuter: Double = 140.0,
        /** Minimum 4-corner model fit for a reported piece. */
        val minQuality: Double = 0.3,
        val refine: Boolean = true,
        /** Debug: print the history of the region containing this (x, y) in image coordinates. */
        val debugPoint: Pair<Int, Int>? = null,
    )

    class Stats(var regions: Int = 0, var candidates: Int = 0, var evaluated: Int = 0, var ms: Long = 0)

    /** Returns piece outlines in original image coordinates. */
    fun findPieces(img: RgbImage, detector: PieceDetector, stats: Stats = Stats()): List<Piece> {
        val t0 = System.currentTimeMillis()
        // Coarse screening detector (fast); winners are re-analysed at full precision at the end.
        val fullDetector = detector
        @Suppress("NAME_SHADOWING") val detector = PieceDetector(detector.cfg.copy(maxContourPoints = 240, computePalette = false))
        val f = max(1.0, max(img.width, img.height).toDouble() / cfg.workSize)
        val w = (img.width / f).toInt(); val h = (img.height / f).toInt()
        val g = gradient(img, w, h, f)
        val labels = IntArray(w * h)
        val nRegions = watershed(g, w, h, labels)
        stats.regions = nRegions

        // Region state.
        val area = IntArray(nRegions + 1)
        val pixels = arrayOfNulls<IntList>(nRegions + 1)
        for (i in 0 until w * h) { val l = labels[i]; area[l]++; (pixels[l] ?: IntList().also { pixels[l] = it }).add(i) }
        val adj = Array(nRegions + 1) { HashMap<Int, DoubleArray>(8) } // neighbour -> [sum, count]
        fun addEdge(a: Int, b: Int, v: Double) {
            // One shared stats array per boundary, referenced from both sides.
            val e = adj[a].getOrPut(b) { DoubleArray(2).also { adj[b][a] = it } }; e[0] += v; e[1] += 1.0
        }
        for (y in 0 until h) for (x in 0 until w) {
            val i = y * w + x; val a = labels[i]
            if (x + 1 < w) { val b = labels[i + 1]; if (a != b) addEdge(a, b, max(g[i], g[i + 1]).toDouble()) }
            if (y + 1 < h) { val b = labels[i + w]; if (a != b) addEdge(a, b, max(g[i], g[i + w]).toDouble()) }
        }
        val parent = IntArray(nRegions + 1) { it }
        fun find(x: Int): Int { var r = x; while (parent[r] != r) r = parent[r]; var c = x; while (parent[c] != r) { val n = parent[c]; parent[c] = r; c = n }; return r }
        val maxInternal = DoubleArray(nRegions + 1)
        val version = IntArray(nRegions + 1)

        class Merge(val s: Double, val a: Int, val b: Int, val va: Int, val vb: Int)
        val pq = PriorityQueue<Merge>(compareBy { it.s })
        for (a in 1..nRegions) for ((b, e) in adj[a]) if (a < b) pq.add(Merge(e[0] / e[1], a, b, 0, 0))

        val minA = (cfg.minAreaFraction * w * h).toInt()
        val maxA = (cfg.maxAreaFraction * w * h).toInt()
        class Cand(val piece: Piece, val score: Double, val px: IntArray)
        val cands = ArrayList<Cand>()

        while (pq.isNotEmpty()) {
            val m = pq.poll()
            if (version[m.a] != m.va || version[m.b] != m.vb || find(m.a) != m.a || find(m.b) != m.b) continue
            val newArea = area[m.a] + area[m.b]
            if (newArea > maxA) continue // regions that big are never pieces; leave them
            // Merge smaller into larger (adjacency + pixel lists).
            val (keep, gone) = if (adj[m.a].size >= adj[m.b].size) m.a to m.b else m.b to m.a
            parent[gone] = keep
            area[keep] = newArea
            maxInternal[keep] = max(max(maxInternal[keep], maxInternal[gone]), m.s)
            val pk = pixels[keep]!!; val pg = pixels[gone]!!
            if (pk.size >= pg.size) { pk.addAll(pg); pixels[gone] = null } else { pg.addAll(pk); pixels[keep] = pg; pixels[gone] = null }
            adj[keep].remove(gone)
            for ((c, e) in adj[gone]) {
                if (c == keep) continue
                val ec = adj[keep].getOrPut(c) { DoubleArray(2) }; ec[0] += e[0]; ec[1] += e[1]
                adj[c].remove(gone); adj[c][keep] = ec
            }
            adj[gone].clear()
            version[keep]++; version[gone]++
            for ((c, e) in adj[keep]) pq.add(Merge(e[0] / e[1], keep, c, version[keep], version[c]))
            val dbg = cfg.debugPoint?.let { (dx, dy) -> find(labels[(dy / f).toInt() * w + (dx / f).toInt()]) == keep } ?: false
            if (dbg && newArea >= minA / 4) {
                var oS = 0.0; var oC = 0.0
                for (e in adj[keep].values) { oS += e[0]; oC += e[1] }
                val p = if (newArea >= minA) evaluate(pixels[keep]!!.toArray(), w, h, f, img, detector) else null
                println("dbg merge s=%.1f area=%d (minA %d) outer=%.1f maxInt=%.1f piece=%s q=%s".format(m.s, newArea, minA, oS / oC, maxInternal[keep], p != null, p?.quality))
            }
            // Candidate test.
            if (newArea >= minA) {
                stats.candidates++
                var outSum = 0.0; var outCnt = 0.0
                for (e in adj[keep].values) { outSum += e[0]; outCnt += e[1] }
                val outer = if (outCnt > 0) outSum / outCnt else 0.0
                if (outer >= cfg.minOuter) {
                    val px = pixels[keep]!!.toArray()
                    val piece = evaluate(px, w, h, f, img, detector)
                    stats.evaluated++
                    if (piece != null) cands.add(Cand(piece, piece.quality + 0.5 * outer / 255, px))
                }
            }
        }

        // Best-first non-overlapping selection.
        cands.sortByDescending { it.score }
        val taken = BooleanArray(w * h)
        val out = ArrayList<Piece>()
        for (c in cands) {
            var overlap = 0
            for (p in c.px) if (taken[p]) overlap++
            if (overlap > c.px.size * 0.1) continue
            for (p in c.px) taken[p] = true
            out.add(c.piece)
        }
        if (out.size >= 5) {
            val med = out.map { it.scale }.sorted()[out.size / 2]
            out.retainAll { it.scale in med * detector.cfg.minRelativeSize..med * detector.cfg.maxRelativeSize }
        }
        stats.ms = System.currentTimeMillis() - t0
        // Full-resolution refinement: snap each outline point to the strongest edge along its normal.
        val lum = FloatArray(img.width * img.height) { val q = img.pixels[it]; 0.299f * ((q shr 16) and 0xFF) + 0.587f * ((q shr 8) and 0xFF) + 0.114f * (q and 0xFF) }
        val refined = ArrayList<Piece>()
        for (p in out) {
            val (rx, ry) = if (cfg.refine) OutlineRefiner.refine(p.xs, p.ys, lum, img.width, img.height, 1.5 * f) else p.xs to p.ys
            val r = fullDetector.analyzeContour(rx, ry, img, refined.size) ?: run { if (cfg.minQuality < -1) println("drop: no corners at %.0f,%.0f".format(p.centroidX, p.centroidY)); null } ?: continue
            if (r.quality < cfg.minQuality || !fullDetector.isPlausible(r)) { if (cfg.minQuality < -1) println("drop q=%.2f plausible=%s at %.0f,%.0f".format(r.quality, fullDetector.isPlausible(r), r.centroidX, r.centroidY)); continue }
            refined.add(r)
        }
        return refined
    }

    /** Traces the region's outer outline and fits the piece model. */
    private fun evaluate(px: IntArray, w: Int, h: Int, f: Double, img: RgbImage, detector: PieceDetector): Piece? {
        var minX = w; var minY = h; var maxX = 0; var maxY = 0
        for (p in px) { val x = p % w; val y = p / w; minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y) }
        val bw = maxX - minX + 3; val bh = maxY - minY + 3
        // Quick shape prefilters: compactness and aspect ratio.
        if (px.size < 0.3 * (bw - 2) * (bh - 2)) return null
        if (max(bw, bh) > 2.5 * min(bw, bh)) return null
        val mask = IntArray(bw * bh)
        var start = Int.MAX_VALUE
        for (p in px) { val i = (p / w - minY + 1) * bw + (p % w - minX + 1); mask[i] = 1; if (i < start) start = i }
        val (cx, cy) = PieceDetector.traceContour(mask, bw, bh, 1, start % bw, start / bw)
        for (i in cx.indices) { cx[i] = (cx[i] + minX - 1) * f; cy[i] = (cy[i] + minY - 1) * f }
        val piece = detector.analyzeContour(cx, cy, img, 0) ?: return null
        if (!detector.isPlausible(piece)) return null
        // Solidity: a piece's area is close to its corner quadrilateral's area.
        return piece
    }

    /** Blurred colour gradient magnitude, quantised to 0..255. */
    private fun gradient(img: RgbImage, w: Int, h: Int, f: Double): IntArray {
        val ch = Array(3) { FloatArray(w * h) }
        for (y in 0 until h) for (x in 0 until w) {
            // Box-average the source pixels covered by this working pixel.
            val sx0 = (x * f).toInt(); val sy0 = (y * f).toInt()
            val sx1 = min(img.width, max(sx0 + 1, ((x + 1) * f).toInt())); val sy1 = min(img.height, max(sy0 + 1, ((y + 1) * f).toInt()))
            var r = 0f; var gg = 0f; var b = 0f; var n = 0
            var yy = sy0
            while (yy < sy1) { var xx = sx0; while (xx < sx1) { val p = img[xx, yy]; r += (p shr 16) and 0xFF; gg += (p shr 8) and 0xFF; b += p and 0xFF; n++; xx++ }; yy++ }
            r /= n; gg /= n; b /= n
            val i = y * w + x
            ch[0][i] = 0.299f * r + 0.587f * gg + 0.114f * b // luminance (edges, shadows)
            ch[1][i] = 0.5f * (r - gg)
            ch[2][i] = 0.5f * ((r + gg) / 2 - b)
        }
        for (c in ch) blur(c, w, h, cfg.blurSigma)
        val mag = FloatArray(w * h)
        for (y in 1 until h - 1) for (x in 1 until w - 1) {
            val i = y * w + x
            var s = 0f
            for (c in ch) {
                val gx = (c[i - w + 1] + 2 * c[i + 1] + c[i + w + 1]) - (c[i - w - 1] + 2 * c[i - 1] + c[i + w - 1])
                val gy = (c[i + w - 1] + 2 * c[i + w] + c[i + w + 1]) - (c[i - w - 1] + 2 * c[i - w] + c[i - w + 1])
                s += gx * gx + gy * gy
            }
            mag[i] = sqrt(s)
        }
        val sorted = mag.copyOf(); sorted.sort()
        val p99 = max(1f, sorted[(sorted.size * 0.99).toInt()])
        return IntArray(w * h) { min(255, (mag[it] / p99 * 255).toInt()) }
    }

    private fun blur(a: FloatArray, w: Int, h: Int, sigma: Double) {
        val r = max(1, (sigma * 2.5).toInt())
        val k = FloatArray(2 * r + 1) { val d = it - r; kotlin.math.exp(-d * d / (2 * sigma * sigma)).toFloat() }
        val ks = k.sum(); for (i in k.indices) k[i] /= ks
        val tmp = FloatArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            var s = 0f
            for (d in -r..r) s += k[d + r] * a[y * w + (x + d).coerceIn(0, w - 1)]
            tmp[y * w + x] = s
        }
        for (y in 0 until h) for (x in 0 until w) {
            var s = 0f
            for (d in -r..r) s += k[d + r] * tmp[(y + d).coerceIn(0, h - 1) * w + x]
            a[y * w + x] = s
        }
    }

    /** Seeded watershed (priority flood). Seeds: the lowest-gradient pixel of each grid cell. */
    private fun watershed(g: IntArray, w: Int, h: Int, labels: IntArray): Int {
        val buckets = Array(256) { IntList() }
        var n = 0
        val c = cfg.seedCell
        for (cy in 0 until h step c) for (cx in 0 until w step c) {
            var best = -1; var bv = Int.MAX_VALUE
            for (y in cy until min(h, cy + c)) for (x in cx until min(w, cx + c)) { val v = g[y * w + x]; if (v < bv) { bv = v; best = y * w + x } }
            n++
            labels[best] = n
            buckets[bv].add(best)
        }
        var level = 0
        while (level < 256) {
            val b = buckets[level]
            if (b.size == 0) { level++; continue }
            val i = b.pop()
            val x = i % w; val y = i / w
            val l = labels[i]
            fun visit(j: Int) {
                if (labels[j] != 0) return
                labels[j] = l
                val v = max(g[j], level)
                buckets[v].add(j)
                if (v < level) level = v
            }
            if (x > 0) visit(i - 1)
            if (x < w - 1) visit(i + 1)
            if (y > 0) visit(i - w)
            if (y < h - 1) visit(i + w)
        }
        return n
    }

    private class IntList {
        var a = IntArray(16); var size = 0
        fun add(v: Int) { if (size == a.size) a = a.copyOf(size * 2); a[size++] = v }
        fun pop(): Int = a[--size]
        fun addAll(o: IntList) { if (size + o.size > a.size) a = a.copyOf(max(a.size * 2, size + o.size)); System.arraycopy(o.a, 0, a, size, o.size); size += o.size }
        fun toArray(): IntArray = a.copyOf(size)
    }
}
