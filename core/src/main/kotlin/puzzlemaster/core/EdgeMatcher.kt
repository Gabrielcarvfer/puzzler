package puzzlemaster.core

import java.util.concurrent.Executors
import java.util.concurrent.Future
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

class EdgeMatch(val a: Edge, val b: Edge, val score: Double)

/**
 * Scores how well two edges fit together. Lower is better.
 *
 * Shape: edge B is walked backwards and rotated 180 degrees so it should lie exactly on edge A;
 * the residual after a best rigid (Procrustes) alignment is the shape error.
 * Colour: the picture is continuous across the cut, so colours just inside A should equal the
 * colours just inside B at the mirrored positions.
 */
class EdgeMatcher(val cfg: Config = Config()) {
    data class Config(
        val lengthWeight: Double = 0.6,
        val colorWeight: Double = 0.35,
        /** In-view matches scoring worse than this are not reported. */
        val maxScore: Double = 0.10,
        /** Picture-continuity mode (for standard-die puzzles): shape gates, picture decides. */
        val pictureMode: Boolean = true,
        /** In-view limit for picture-mode scores (calibrated on real photos: true pair 0.23, false 0.29). */
        val maxPictureScore: Double = 0.27,
        val shapeGate: Double = 0.06,
        val lengthGate: Double = 0.10,
        val shapeWeightInPicture: Double = 0.5,
        val structureWeight: Double = 0.15,
        val paletteGate: Double = 0.6,
        val paletteWeight: Double = 0.2,
        /** Shape residual limit (fraction of edge length), regardless of colour. */
        val maxShape: Double = 0.045,
        /** Best partner must score at most this fraction of the runner-up (ambiguity rejection). */
        val ratio: Double = 0.8,
    )

    fun compatible(a: Edge, b: Edge): Boolean =
        a.pieceId != b.pieceId &&
            ((a.type == EdgeType.TAB && b.type == EdgeType.BLANK) || (a.type == EdgeType.BLANK && b.type == EdgeType.TAB))

    fun score(a: Edge, b: Edge): Double {
        if (!compatible(a, b)) return Double.POSITIVE_INFINITY
        val s = (a.length + b.length) / 2
        val shape = shapeError(a, b) / s
        val len = abs(a.length - b.length) / s
        if (!cfg.pictureMode || a.stripLen == 0 || b.stripLen == 0) {
            return shape + cfg.lengthWeight * len + cfg.colorWeight * colorDistance(a, b)
        }
        // Standard dies repeat the same tab shapes: shape only gates, the picture decides.
        if (shape > cfg.shapeGate || len > cfg.lengthGate) return Double.POSITIVE_INFINITY
        // Whole-piece colour mix: neighbours in the picture share a palette.
        val pal = PieceDetector.paletteDistance(a.palette, b.palette)
        if (pal > cfg.paletteGate) return Double.POSITIVE_INFINITY
        return pictureDistance(a, b) + cfg.shapeWeightInPicture * shape + cfg.paletteWeight * pal
    }

    /**
     * How well the printed picture continues across the joint: the strips just inside both edges
     * (B reversed) should look alike. Each strip is normalised by its own brightness so that lighting
     * differences between the two pieces don't count; colour ratios and the pattern along the edge do.
     */
    fun pictureDistance(a: Edge, b: Edge): Double {
        val sl = a.stripLen
        if (sl == 0 || b.stripLen != sl || a.stripRows != b.stripRows) return 1.0
        val rows = a.stripRows
        fun mean(st: FloatArray): Double { var m = 0.0; for (v in st) m += v; return (m / st.size).coerceAtLeast(8.0) }
        val ma = mean(a.strip); val mb = mean(b.strip)
        var d = 0.0
        var wsum = 0.0
        for (r in 0 until rows) {
            // Rows nearest the cut matter most.
            val w = 1.0 / (r + 1)
            for (i in 0 until sl) {
                val j = sl - 1 - i
                val ia = (r * sl + i) * 3; val ib = (r * sl + j) * 3
                for (c in 0 until 3) d += w * abs(a.strip[ia + c] / ma - b.strip[ib + c] / mb)
                wsum += w * 3
            }
        }
        val colour = d / wsum
        // Pattern along the edge: correlation of the nearest-row brightness profiles.
        var sa = 0.0; var sb = 0.0
        val la = DoubleArray(sl) { i -> val k = i * 3; (a.strip[k] + a.strip[k + 1] + a.strip[k + 2]) / ma }
        val lb = DoubleArray(sl) { i -> val k = (sl - 1 - i) * 3; (b.strip[k] + b.strip[k + 1] + b.strip[k + 2]) / mb }
        for (i in 0 until sl) { sa += la[i]; sb += lb[i] }
        sa /= sl; sb /= sl
        var cov = 0.0; var va = 0.0; var vb = 0.0
        for (i in 0 until sl) { cov += (la[i] - sa) * (lb[i] - sb); va += (la[i] - sa) * (la[i] - sa); vb += (lb[i] - sb) * (lb[i] - sb) }
        val texture = kotlin.math.sqrt(va * vb) / sl
        val corr = if (texture > 1e-3) cov / kotlin.math.sqrt(va * vb) else 0.0
        // Only trust the pattern term when both strips actually have a pattern.
        val structure = if (texture > 1e-3) (1 - corr) / 2 * minOf(1.0, texture / 0.01) else 0.0
        return colour + cfg.structureWeight * structure
    }

    fun shapeError(a: Edge, b: Edge): Double {
        val n = a.cx.size
        // B mapped into A's frame: reversed order, (x, y) -> (Lb - x, -y), centred on A's chord.
        var sax = 0.0; var say = 0.0; var sbx = 0.0; var sby = 0.0
        val bx = DoubleArray(n); val by = DoubleArray(n)
        val shift = (a.length - b.length) / 2
        for (i in 0 until n) {
            val j = n - 1 - i
            bx[i] = b.length - b.cx[j] + shift
            by[i] = -b.cy[j]
            sax += a.cx[i]; say += a.cy[i]; sbx += bx[i]; sby += by[i]
        }
        sax /= n; say /= n; sbx /= n; sby /= n
        var dot = 0.0; var crs = 0.0
        for (i in 0 until n) {
            val ax = a.cx[i] - sax; val ay = a.cy[i] - say
            val px = bx[i] - sbx; val py = by[i] - sby
            dot += px * ax + py * ay
            crs += px * ay - py * ax
        }
        val th = atan2(crs, dot)
        val c = cos(th); val s = sin(th)
        var err = 0.0
        for (i in 0 until n) {
            val px = bx[i] - sbx; val py = by[i] - sby
            val rx = c * px - s * py; val ry = s * px + c * py
            val dx = a.cx[i] - sax - rx; val dy = a.cy[i] - say - ry
            err += dx * dx + dy * dy
        }
        return sqrt(err / n)
    }

    /** Mean absolute colour difference in [0, 1]. */
    fun colorDistance(a: Edge, b: Edge): Double {
        val k = a.colors.size / 3
        if (k == 0 || b.colors.size != a.colors.size) return 0.0
        var d = 0.0
        for (i in 0 until k) {
            val j = k - 1 - i
            d += abs(a.colors[i * 3] - b.colors[j * 3]) + abs(a.colors[i * 3 + 1] - b.colors[j * 3 + 1]) + abs(a.colors[i * 3 + 2] - b.colors[j * 3 + 2])
        }
        return d / (k * 3 * 255.0)
    }

    /**
     * Matching for the pieces currently in view: every edge's best partner is found and only
     * mutual-best pairs below [Config.maxScore] are kept. Sorted best first.
     */
    fun matchInView(pieces: List<Piece>): List<EdgeMatch> {
        val edges = pieces.flatMap { p -> p.edges.filter { it.type != EdgeType.FLAT } }
        val n = edges.size
        val bestOf = IntArray(n) { -1 }
        val bestScore = DoubleArray(n) { Double.POSITIVE_INFINITY }
        val second = DoubleArray(n) { Double.POSITIVE_INFINITY }
        fun offer(i: Int, j: Int, sc: Double) {
            if (sc < bestScore[i]) { second[i] = bestScore[i]; bestScore[i] = sc; bestOf[i] = j } else if (sc < second[i]) second[i] = sc
        }
        val scores = HashMap<Long, Double>()
        for (i in 0 until n) for (j in i + 1 until n) {
            val sc = score(edges[i], edges[j])
            if (sc == Double.POSITIVE_INFINITY) continue
            scores[i.toLong() * n + j] = sc
            offer(i, j, sc); offer(j, i, sc)
        }
        val out = ArrayList<EdgeMatch>()
        for (i in 0 until n) {
            val j = bestOf[i]
            if (j <= i || bestOf[j] != i || bestScore[i] > (if (cfg.pictureMode) cfg.maxPictureScore else cfg.maxScore)) continue
            if (bestScore[i] > cfg.ratio * second[i] || bestScore[i] > cfg.ratio * second[j]) continue
            val s = (edges[i].length + edges[j].length) / 2
            if (shapeError(edges[i], edges[j]) / s > (if (cfg.pictureMode) cfg.shapeGate else cfg.maxShape)) continue
            out.add(EdgeMatch(edges[i], edges[j], bestScore[i]))
        }
        // A piece pair can only share one edge: keep the best one.
        val seen = HashSet<Long>()
        return out.sortedBy { it.score }.filter { m ->
            val lo = minOf(m.a.pieceId, m.b.pieceId).toLong(); val hi = maxOf(m.a.pieceId, m.b.pieceId).toLong()
            seen.add(lo * 1_000_000 + hi)
        }
    }
}

/**
 * Index for searching a large collection (thousands of pieces) for an edge's partner.
 * A cheap coarse descriptor prunes candidates, then the full [EdgeMatcher.score] ranks the survivors.
 */
class EdgeIndex(edges: List<Edge>, private val matcher: EdgeMatcher = EdgeMatcher(), private val coarsePoints: Int = 8) {
    private val tabs = edges.filter { it.type == EdgeType.TAB }
    private val blanks = edges.filter { it.type == EdgeType.BLANK }
    private val dims = coarsePoints * 2 + 1 + 6
    private val tabVec = Array(tabs.size) { coarse(tabs[it], false) }
    private val blankVec = Array(blanks.size) { coarse(blanks[it], true) }

    /**
     * Coarse vector: a few shape points relative to the chord midpoint (scaled by edge length),
     * the length, and two average colours. Blanks are stored already mirrored, so that a tab and its
     * true partner have nearly identical vectors.
     */
    private fun coarse(e: Edge, mirror: Boolean): DoubleArray {
        val v = DoubleArray(dims)
        val n = e.cx.size
        val s = e.length
        for (k in 0 until coarsePoints) {
            val i = (k * (n - 1)) / (coarsePoints - 1)
            if (!mirror) {
                v[2 * k] = (e.cx[i] - s / 2) / s
                v[2 * k + 1] = e.cy[i] / s
            } else {
                val j = n - 1 - i
                v[2 * k] = (s / 2 - e.cx[j]) / s
                v[2 * k + 1] = -e.cy[j] / s
            }
        }
        v[coarsePoints * 2] = s / 100.0
        val k = e.colors.size / 3
        if (k > 0) {
            val half = k / 2
            for (h in 0 until 2) {
                // For mirrored edges the halves swap so they line up with the partner.
                val range = if ((h == 0) != mirror) 0 until half else half until k
                for (c in 0 until 3) {
                    var sum = 0.0
                    for (i in range) sum += e.colors[i * 3 + c]
                    v[coarsePoints * 2 + 1 + h * 3 + c] = 0.6 * sum / (range.count() * 255.0)
                }
            }
        }
        return v
    }

    /** Best [k] partners for [query], best first. */
    fun query(query: Edge, k: Int = 5, prefilter: Int = 40): List<EdgeMatch> {
        val (pool, vecs, qv) = when (query.type) {
            EdgeType.TAB -> Triple(blanks, blankVec, coarse(query, false))
            EdgeType.BLANK -> Triple(tabs, tabVec, coarse(query, true))
            EdgeType.FLAT -> return emptyList()
        }
        val keep = minOf(prefilter, pool.size)
        val heapIdx = IntArray(keep)
        val heapD = DoubleArray(keep) { Double.POSITIVE_INFINITY }
        for (i in pool.indices) {
            if (pool[i].pieceId == query.pieceId) continue
            val v = vecs[i]
            var d = 0.0
            var t = 0
            val worst = heapD[keep - 1]
            while (t < dims) { val x = v[t] - qv[t]; d += x * x; if (d >= worst) break; t++ }
            if (d >= worst) continue
            // insertion into small sorted array
            var p = keep - 1
            while (p > 0 && heapD[p - 1] > d) { heapD[p] = heapD[p - 1]; heapIdx[p] = heapIdx[p - 1]; p-- }
            heapD[p] = d; heapIdx[p] = i
        }
        val out = ArrayList<EdgeMatch>(keep)
        for (p in 0 until keep) {
            if (heapD[p] == Double.POSITIVE_INFINITY) break
            val e = pool[heapIdx[p]]
            out.add(EdgeMatch(query, e, matcher.score(query, e)))
        }
        out.sortBy { it.score }
        return out.take(k)
    }

    /** Runs [query] for many edges using all cores. */
    fun queryAll(queries: List<Edge>, k: Int = 5, threads: Int = Runtime.getRuntime().availableProcessors()): List<List<EdgeMatch>> {
        val pool = Executors.newFixedThreadPool(threads)
        try {
            val chunk = (queries.size + threads - 1) / threads
            val futures = ArrayList<Future<List<List<EdgeMatch>>>>()
            for (t in 0 until threads) {
                val from = t * chunk
                val to = minOf(queries.size, from + chunk)
                if (from >= to) break
                futures.add(pool.submit<List<List<EdgeMatch>>> { (from until to).map { query(queries[it], k) } })
            }
            return futures.flatMap { it.get() }
        } finally {
            pool.shutdown()
        }
    }
}
