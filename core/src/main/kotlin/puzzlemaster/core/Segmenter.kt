package puzzlemaster.core

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class Component(
    val label: Int,
    val area: Int,
    val minX: Int, val minY: Int, val maxX: Int, val maxY: Int,
    /** First pixel in raster order (top-most, then left-most); its west neighbour is background. */
    val startX: Int, val startY: Int,
    val touchesBorder: Boolean,
)

class Segmentation(
    val width: Int,
    val height: Int,
    /** 0 = background, otherwise component label. */
    val labels: IntArray,
    val components: List<Component>,
    val backgroundColor: Int,
    val threshold: Int,
)

/**
 * Separates puzzle pieces from a plain, contrasting background (table, felt, sheet of paper).
 * The background colour is estimated from the image border, each pixel's colour distance to it is
 * thresholded with Otsu's method, then cleaned up with morphology and hole filling.
 */
object Segmenter {
    data class Config(
        val minThreshold: Int = 18,
        val minAreaFraction: Double = 0.0004,
        val minArea: Int = 300,
        /** Model lighting variation with a grid of local background colours. */
        val localBackground: Boolean = false,
        /** Background colour/noise from the image border (true) or from the dominant scene colour. */
        val borderBackground: Boolean = true,
    )

    fun segment(img: RgbImage, cfg: Config = Config()): Segmentation {
        val w = img.width
        val h = img.height
        // Background = the dominant colour of the scene (the table), modelled per grid cell so that
        // uneven lighting across the table is handled. Objects at the image border (chairs, the box)
        // don't influence it, unlike a border-based estimate.
        // Border-based estimate only if the border really is one uniform surface; otherwise (chairs,
        // window, box at the edge of the photo) use the dominant scene colour.
        val useBorder = cfg.borderBackground && borderIsUniform(img)
        val bgModel = BackgroundModel(img, cfg.localBackground || !useBorder, if (useBorder) estimateBackground(img) else null)
        val bg = bgModel.global

        // Colour distance to the local background; chroma counts as much as brightness.
        val dist = ByteArray(w * h)
        val hist = IntArray(256)
        for (y in 0 until h) {
            bgModel.row(y)
            for (x in 0 until w) {
                val i = y * w + x
                val p = img.pixels[i]
                val dr = ((p shr 16) and 0xFF) - bgModel.rowR[x]
                val dg = ((p shr 8) and 0xFF) - bgModel.rowG[x]
                val db = (p and 0xFF) - bgModel.rowB[x]
                val mean = (dr + dg + db) / 3
                val chroma = abs(dr - mean) + abs(dg - mean) + abs(db - mean)
                val d = (abs(mean) + chroma).coerceAtMost(255)
                dist[i] = d.toByte()
                hist[d]++
            }
        }
        // The background is the largest region, so a low percentile of the distances measures its
        // noise (wood grain, sensor noise). Threshold a few times above that, capped by Otsu.
        val t = if (useBorder) {
            val borderD = ArrayList<Int>()
            val step = max(1, (w + h) / 600)
            for (x in 0 until w step step) { borderD.add(dist[x].toInt() and 0xFF); borderD.add(dist[(h - 1) * w + x].toInt() and 0xFF) }
            for (y in 0 until h step step) { borderD.add(dist[y * w].toInt() and 0xFF); borderD.add(dist[y * w + w - 1].toInt() and 0xFF) }
            borderD.sort()
            val p98 = borderD[(borderD.size * 98) / 100]
            max(cfg.minThreshold, minOf((p98 * 1.5 + 10).toInt(), otsu(hist, w * h).coerceAtLeast(cfg.minThreshold)))
        } else {
            var cum = 0
            var p30 = 0
            for (v in 0 until 256) { cum += hist[v]; if (cum >= w * h * 0.3) { p30 = v; break } }
            max(cfg.minThreshold, minOf(3 * p30 + 14, otsu(hist, w * h).coerceAtLeast(cfg.minThreshold)))
        }

        var mask = ByteArray(w * h)
        for (i in 0 until w * h) if ((dist[i].toInt() and 0xFF) > t) mask[i] = 1
        mask = erode(mask, w, h); mask = dilate(mask, w, h) // open: remove speckles
        mask = dilate(mask, w, h); mask = erode(mask, w, h) // close: fill small gaps
        fillHoles(mask, w, h)

        val minArea = max(cfg.minArea, (cfg.minAreaFraction * w * h).toInt())
        val (labels, comps) = label(mask, w, h, minArea)
        return Segmentation(w, h, labels, comps, bg, t)
    }

    /** Per-cell background colour, bilinearly interpolated per row on demand. */
    private class BackgroundModel(img: RgbImage, local: Boolean, fixed: Int?) {
        val w = img.width
        val h = img.height
        val global: Int
        private val gx: Int
        private val gy: Int
        private val cell: Double
        private val cr: Array<DoubleArray>
        private val cg: Array<DoubleArray>
        private val cb: Array<DoubleArray>
        val rowR = IntArray(w); val rowG = IntArray(w); val rowB = IntArray(w)

        init {
            val step = max(1, kotlin.math.sqrt(w.toDouble() * h / 60000).toInt())
            val bins = IntArray(4096)
            for (y in 0 until h step step) for (x in 0 until w step step) {
                val p = img[x, y]
                bins[(((p shr 20) and 0xF) shl 8) or (((p shr 12) and 0xF) shl 4) or ((p shr 4) and 0xF)]++
            }
            val mode = bins.indices.maxBy { bins[it] }
            var sr = 0.0; var sg = 0.0; var sb = 0.0; var n = 0
            for (y in 0 until h step step) for (x in 0 until w step step) {
                val p = img[x, y]
                if ((((p shr 20) and 0xF) shl 8) or (((p shr 12) and 0xF) shl 4) or ((p shr 4) and 0xF) == mode) {
                    sr += (p shr 16) and 0xFF; sg += (p shr 8) and 0xFF; sb += p and 0xFF; n++
                }
            }
            var gr = sr / n; var gg = sg / n; var gb = sb / n
            if (fixed != null) { gr = ((fixed shr 16) and 0xFF).toDouble(); gg = ((fixed shr 8) and 0xFF).toDouble(); gb = (fixed and 0xFF).toDouble() }
            global = RgbImage.argb(gr.toInt(), gg.toInt(), gb.toInt())

            cell = if (local) max(w, h) / 12.0 else max(w, h).toDouble() + 1
            gx = kotlin.math.ceil(w / cell).toInt(); gy = kotlin.math.ceil(h / cell).toInt()
            cr = Array(gy) { DoubleArray(gx) { Double.NaN } }
            cg = Array(gy) { DoubleArray(gx) { Double.NaN } }
            cb = Array(gy) { DoubleArray(gx) { Double.NaN } }
            val s2 = max(1, step / 2)
            for (cy in 0 until gy) for (cx in 0 until gx) {
                val x0 = (cx * cell).toInt(); val x1 = min(w, ((cx + 1) * cell).toInt())
                val y0 = (cy * cell).toInt(); val y1 = min(h, ((cy + 1) * cell).toInt())
                var r = 0.0; var g = 0.0; var b = 0.0; var k = 0; var tot = 0
                for (y in y0 until y1 step s2) for (x in x0 until x1 step s2) {
                    val p = img[x, y]; tot++
                    val pr = (p shr 16) and 0xFF; val pg = (p shr 8) and 0xFF; val pb = p and 0xFF
                    // Background-like: close in hue to the table, brightness may differ (lighting).
                    val l = (pr + pg + pb) / 3.0; val gl = (gr + gg + gb) / 3.0
                    if (abs((pr - l) - (gr - gl)) + abs((pg - l) - (gg - gl)) + abs((pb - l) - (gb - gl)) < 40 && abs(l - gl) < 90) {
                        r += pr; g += pg; b += pb; k++
                    }
                }
                if (local && k > 0.15 * tot && k > 4) { cr[cy][cx] = r / k; cg[cy][cx] = g / k; cb[cy][cx] = b / k }
            }
            // Cells without enough table pixels borrow from neighbours.
            var missing = true
            var guard = 0
            while (missing && guard++ < gx + gy) {
                missing = false
                for (cy in 0 until gy) for (cx in 0 until gx) if (cr[cy][cx].isNaN()) {
                    var r = 0.0; var g = 0.0; var b = 0.0; var k = 0
                    for ((dx, dy) in listOf(1 to 0, -1 to 0, 0 to 1, 0 to -1)) {
                        val nx = cx + dx; val ny = cy + dy
                        if (nx in 0 until gx && ny in 0 until gy && !cr[ny][nx].isNaN()) { r += cr[ny][nx]; g += cg[ny][nx]; b += cb[ny][nx]; k++ }
                    }
                    if (k > 0) { cr[cy][cx] = r / k; cg[cy][cx] = g / k; cb[cy][cx] = b / k } else missing = true
                }
            }
            for (cy in 0 until gy) for (cx in 0 until gx) if (cr[cy][cx].isNaN()) { cr[cy][cx] = gr; cg[cy][cx] = gg; cb[cy][cx] = gb }
        }

        fun row(y: Int) {
            val fy = ((y + 0.5) / cell - 0.5).coerceIn(0.0, gy - 1.0)
            val y0 = fy.toInt(); val y1 = min(gy - 1, y0 + 1); val ty = fy - y0
            for (x in 0 until w) {
                val fx = ((x + 0.5) / cell - 0.5).coerceIn(0.0, gx - 1.0)
                val x0 = fx.toInt(); val x1 = min(gx - 1, x0 + 1); val tx = fx - x0
                fun bil(c: Array<DoubleArray>) = ((c[y0][x0] * (1 - tx) + c[y0][x1] * tx) * (1 - ty) + (c[y1][x0] * (1 - tx) + c[y1][x1] * tx) * ty).toInt()
                rowR[x] = bil(cr); rowG[x] = bil(cg); rowB[x] = bil(cb)
            }
        }
    }

    private fun borderIsUniform(img: RgbImage): Boolean {
        val bg = estimateBackground(img)
        val w = img.width; val h = img.height
        val step = max(1, (w + h) / 400)
        val d = ArrayList<Int>()
        fun add(x: Int, y: Int) {
            val p = img[x, y]
            d.add(abs(((p shr 16) and 0xFF) - ((bg shr 16) and 0xFF)) + abs(((p shr 8) and 0xFF) - ((bg shr 8) and 0xFF)) + abs((p and 0xFF) - (bg and 0xFF)))
        }
        for (x in 0 until w step step) { add(x, 0); add(x, h - 1) }
        for (y in 0 until h step step) { add(0, y); add(w - 1, y) }
        d.sort()
        return d[d.size * 85 / 100] < 75
    }

    private fun estimateBackground(img: RgbImage): Int {
        val rs = ArrayList<Int>()
        val gs = ArrayList<Int>()
        val bs = ArrayList<Int>()
        val w = img.width
        val h = img.height
        val step = max(1, (w + h) / 400)
        fun add(x: Int, y: Int) {
            val p = img[x, y]
            rs.add((p shr 16) and 0xFF); gs.add((p shr 8) and 0xFF); bs.add(p and 0xFF)
        }
        for (x in 0 until w step step) { add(x, 0); add(x, h - 1); add(x, 1.coerceAtMost(h - 1)); add(x, (h - 2).coerceAtLeast(0)) }
        for (y in 0 until h step step) { add(0, y); add(w - 1, y); add(1.coerceAtMost(w - 1), y); add((w - 2).coerceAtLeast(0), y) }
        rs.sort(); gs.sort(); bs.sort()
        return RgbImage.argb(rs[rs.size / 2], gs[gs.size / 2], bs[bs.size / 2])
    }

    private fun otsu(hist: IntArray, total: Int): Int {
        var sum = 0.0
        for (i in 0 until 256) sum += i.toDouble() * hist[i]
        var sumB = 0.0
        var wB = 0
        var best = 0.0
        var threshold = 0
        for (i in 0 until 256) {
            wB += hist[i]
            if (wB == 0) continue
            val wF = total - wB
            if (wF == 0) break
            sumB += i.toDouble() * hist[i]
            val mB = sumB / wB
            val mF = (sum - sumB) / wF
            val between = wB.toDouble() * wF * (mB - mF) * (mB - mF)
            if (between > best) { best = between; threshold = i }
        }
        return threshold
    }

    private fun erode(m: ByteArray, w: Int, h: Int): ByteArray {
        val out = ByteArray(w * h)
        for (y in 1 until h - 1) {
            val row = y * w
            for (x in 1 until w - 1) {
                val i = row + x
                if (m[i].toInt() == 1 && m[i - 1].toInt() == 1 && m[i + 1].toInt() == 1 &&
                    m[i - w].toInt() == 1 && m[i + w].toInt() == 1
                ) out[i] = 1
            }
        }
        return out
    }

    private fun dilate(m: ByteArray, w: Int, h: Int): ByteArray {
        val out = ByteArray(w * h)
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                val i = row + x
                if (m[i].toInt() == 1 ||
                    (x > 0 && m[i - 1].toInt() == 1) || (x < w - 1 && m[i + 1].toInt() == 1) ||
                    (y > 0 && m[i - w].toInt() == 1) || (y < h - 1 && m[i + w].toInt() == 1)
                ) out[i] = 1
            }
        }
        return out
    }

    /** Any background region not connected to the image border becomes foreground. */
    private fun fillHoles(m: ByteArray, w: Int, h: Int) {
        val reached = BooleanArray(w * h)
        val stack = IntArray(w * h)
        var sp = 0
        fun push(i: Int) {
            if (!reached[i] && m[i].toInt() == 0) { reached[i] = true; stack[sp++] = i }
        }
        for (x in 0 until w) { push(x); push((h - 1) * w + x) }
        for (y in 0 until h) { push(y * w); push(y * w + w - 1) }
        while (sp > 0) {
            val i = stack[--sp]
            val x = i % w
            val y = i / w
            if (x > 0) push(i - 1)
            if (x < w - 1) push(i + 1)
            if (y > 0) push(i - w)
            if (y < h - 1) push(i + w)
        }
        for (i in 0 until w * h) if (m[i].toInt() == 0 && !reached[i]) m[i] = 1
    }

    private fun label(m: ByteArray, w: Int, h: Int, minArea: Int): Pair<IntArray, List<Component>> {
        val labels = IntArray(w * h)
        val stack = IntArray(w * h)
        val comps = ArrayList<Component>()
        var next = 1
        for (start in 0 until w * h) {
            if (m[start].toInt() == 0 || labels[start] != 0) continue
            val id = next++
            var sp = 0
            stack[sp++] = start
            labels[start] = id
            var area = 0
            var minX = w; var minY = h; var maxX = 0; var maxY = 0
            val members = ArrayList<Int>()
            while (sp > 0) {
                val i = stack[--sp]
                members.add(i)
                area++
                val x = i % w
                val y = i / w
                if (x < minX) minX = x; if (x > maxX) maxX = x
                if (y < minY) minY = y; if (y > maxY) maxY = y
                if (x > 0 && m[i - 1].toInt() == 1 && labels[i - 1] == 0) { labels[i - 1] = id; stack[sp++] = i - 1 }
                if (x < w - 1 && m[i + 1].toInt() == 1 && labels[i + 1] == 0) { labels[i + 1] = id; stack[sp++] = i + 1 }
                if (y > 0 && m[i - w].toInt() == 1 && labels[i - w] == 0) { labels[i - w] = id; stack[sp++] = i - w }
                if (y < h - 1 && m[i + w].toInt() == 1 && labels[i + w] == 0) { labels[i + w] = id; stack[sp++] = i + w }
            }
            if (area < minArea) {
                for (i in members) labels[i] = -1 // too small: mark as ignored
                continue
            }
            val touches = minX == 0 || minY == 0 || maxX == w - 1 || maxY == h - 1
            comps.add(Component(id, area, minX, minY, maxX, maxY, start % w, start / w, touches))
        }
        for (i in labels.indices) if (labels[i] < 0) labels[i] = 0
        return labels to comps
    }
}
