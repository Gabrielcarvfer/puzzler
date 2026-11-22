package puzzlemaster.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Instance segmentation of puzzle pieces with MobileSAM (Segment Anything), so that touching pieces
 * and difficult backgrounds still produce one outline per piece.
 *
 * The image is processed in square tiles at native resolution (pieces stay large in the model's
 * 1024 px input). Prompt points are placed on non-background pixels, one decoder call per point;
 * points already covered by an accepted mask are skipped, so cost scales with the number of pieces.
 * Each accepted mask is traced and handed to [PieceDetector.analyzeContour].
 */
class SamSegmenter(
    encoderModel: ByteArray,
    decoderModel: ByteArray,
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment(),
    options: OrtSession.SessionOptions = OrtSession.SessionOptions(),
    decoderOptions: OrtSession.SessionOptions = OrtSession.SessionOptions().apply { setIntraOpNumThreads(1) },
    /** Concurrent decoder calls. */
    private val parallel: Int = 4,
) : AutoCloseable {
    private val pool = java.util.concurrent.Executors.newFixedThreadPool(parallel)
    data class Config(
        /** Source pixels per tile side (scaled to the model's 1024 input). */
        val tileSize: Int = 1024,
        val overlap: Double = 0.25,
        /** Prompt grid spacing in source pixels; 0 = automatic (half the estimated piece size). */
        val pointSpacing: Int = 0,
        val minIou: Float = 0.80f,
        val maxPromptsPerTile: Int = 48,
    )

    private val encoder = env.createSession(encoderModel, options)
    private val decoder = env.createSession(decoderModel, decoderOptions)
    private val input = 1024
    private val lowRes = 256

    class Stats(var classic: Int = 0, var tiles: Int = 0, var prompts: Int = 0, var masks: Int = 0, var encoderMs: Long = 0, var decoderMs: Long = 0)

    /**
     * Finds pieces inside [region] (x0, y0, x1, y1; defaults to the whole image).
     * [onProgress] receives (tilesDone, tilesTotal).
     */
    fun detect(
        img: RgbImage,
        detector: PieceDetector,
        cfg: Config = Config(),
        region: IntArray = intArrayOf(0, 0, img.width, img.height),
        stats: Stats = Stats(),
        onProgress: (Int, Int) -> Unit = { _, _ -> },
    ): List<Piece> {
        // Foreground proposal (anything that is not the table) only decides where to put prompts.
        val seg = Segmenter.segment(img, detector.cfg.segmentation.copy(minArea = 30, minAreaFraction = 0.0))
        val covered = BooleanArray(img.width * img.height)
        // Classic detector first: isolated pieces are found cheaply and accurately. MobileSAM is only
        // prompted on what is left (clumps of touching pieces, blobs the classic detector rejected).
        val classic = detector.detect(img).pieces.filter { it.centroidX >= region[0] && it.centroidX < region[2] && it.centroidY >= region[1] && it.centroidY < region[3] }
        val pieces = ArrayList<Piece>()
        for (p in classic) {
            pieces.add(detector.analyzeContour(p.xs, p.ys, img, pieces.size) ?: continue)
            fillPolygon(p.xs, p.ys, covered, img.width, img.height, grow = 0.08 * p.scale)
        }
        stats.classic = pieces.size
        val spacing = if (cfg.pointSpacing > 0) cfg.pointSpacing else {
            val est = if (classic.size >= 2) classic.map { it.scale }.sorted()[classic.size / 2] else (region[2] - region[0]) / 10.0
            max(8, (est / 2).toInt())
        }
        val tiles = tiles(region, cfg)
        for ((ti, t) in tiles.withIndex()) {
            onProgress(ti, tiles.size)
            processTile(img, t[0], t[1], min(cfg.tileSize, img.width - t[0]), min(cfg.tileSize, img.height - t[1]), seg, covered, detector, pieces, cfg.copy(pointSpacing = spacing), stats)
            stats.tiles++
        }
        onProgress(tiles.size, tiles.size)
        // Size consistency (fragments, merged pairs, junk).
        if (pieces.size >= 5) {
            val med = pieces.map { it.scale }.sorted()[pieces.size / 2]
            pieces.retainAll { it.scale in med * detector.cfg.minRelativeSize..med * detector.cfg.maxRelativeSize }
        }
        return pieces
    }

    private fun tiles(r: IntArray, cfg: Config): List<IntArray> {
        val step = (cfg.tileSize * (1 - cfg.overlap)).toInt()
        val out = ArrayList<IntArray>()
        var y = r[1]
        while (true) {
            var x = r[0]
            val ty = min(y, max(r[1], r[3] - cfg.tileSize))
            while (true) {
                val tx = min(x, max(r[0], r[2] - cfg.tileSize))
                out.add(intArrayOf(max(0, tx), max(0, ty)))
                if (tx + cfg.tileSize >= r[2]) break
                x += step
            }
            if (ty + cfg.tileSize >= r[3]) break
            y += step
        }
        return out.distinctBy { it[0] * 100000 + it[1] }
    }

    private fun processTile(
        img: RgbImage, x0: Int, y0: Int, tw: Int, th: Int, seg: Segmentation, covered: BooleanArray,
        detector: PieceDetector, pieces: MutableList<Piece>, cfg: Config, stats: Stats,
    ) {
        // Prompts: roughly one per piece, at the "deepest" foreground points (local maxima of the
        // distance to the background). Narrow necks between touching pieces split clumps into
        // separate maxima, so each piece of a clump still gets its own prompt.
        val prompts = piecePrompts(img, x0, y0, tw, th, seg, covered, cfg)
        if (prompts.isEmpty()) return
        val s = input.toDouble() / max(tw, th)
        val iw = (tw * s).roundToInt(); val ih = (th * s).roundToInt()
        val buf = FloatBuffer.allocate(iw * ih * 3)
        for (y in 0 until ih) for (x in 0 until iw) {
            val p = img[min(img.width - 1, x0 + (x / s).toInt()), min(img.height - 1, y0 + (y / s).toInt())]
            buf.put(((p shr 16) and 0xFF).toFloat()); buf.put(((p shr 8) and 0xFF).toFloat()); buf.put((p and 0xFF).toFloat())
        }
        buf.rewind()
        val t0 = System.currentTimeMillis()
        val embedding: FloatArray
        OnnxTensor.createTensor(env, buf, longArrayOf(ih.toLong(), iw.toLong(), 3)).use { inp ->
            encoder.run(mapOf("input_image" to inp)).use { out ->
                embedding = ((out[0] as OnnxTensor).floatBuffer).let { fb -> FloatArray(fb.remaining()).also { fb.get(it) } }
            }
        }
        stats.encoderMs += System.currentTimeMillis() - t0

        val t1 = System.currentTimeMillis()
        OnnxTensor.createTensor(env, FloatBuffer.wrap(embedding), longArrayOf(1, 256, 64, 64)).use { emb ->
            OnnxTensor.createTensor(env, FloatBuffer.allocate(lowRes * lowRes), longArrayOf(1, 1, lowRes.toLong(), lowRes.toLong())).use { maskIn ->
                OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(0f)), longArrayOf(1)).use { hasMask ->
                    OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(ih.toFloat(), iw.toFloat())), longArrayOf(2)).use { origSize ->
                        // Rounds of up to [parallel] well-separated, uncovered prompts decoded concurrently;
                        // results are merged serially so coverage stays consistent.
                        val remaining = ArrayList(prompts)
                        while (true) {
                            val batch = ArrayList<Pair<Int, Int>>()
                            val it = remaining.iterator()
                            while (it.hasNext() && batch.size < parallel) {
                                val (px, py) = it.next()
                                if (covered[py * img.width + px]) { it.remove(); continue }
                                if (batch.any { (bx, by) -> kotlin.math.abs(bx - px) < 2 * cfg.pointSpacing && kotlin.math.abs(by - py) < 2 * cfg.pointSpacing }) continue
                                batch.add(px to py); it.remove()
                            }
                            if (batch.isEmpty()) break
                            stats.prompts += batch.size
                            val results = batch.map { (px, py) ->
                                pool.submit<Pair<FloatArray, FloatArray>> {
                                    val coords = floatArrayOf(((px - x0 + 0.5) * s).toFloat(), ((py - y0 + 0.5) * s).toFloat(), 0f, 0f)
                                    OnnxTensor.createTensor(env, FloatBuffer.wrap(coords), longArrayOf(1, 2, 2)).use { pc ->
                                        OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(1f, -1f)), longArrayOf(1, 2)).use { pl ->
                                            decoder.run(
                                                mapOf(
                                                    "image_embeddings" to emb, "point_coords" to pc, "point_labels" to pl,
                                                    "mask_input" to maskIn, "has_mask_input" to hasMask, "orig_im_size" to origSize,
                                                ),
                                                setOf("iou_predictions", "low_res_masks"),
                                            ).use { out ->
                                                val iou = (out.get("iou_predictions").get() as OnnxTensor).floatBuffer
                                                val low = (out.get("low_res_masks").get() as OnnxTensor).floatBuffer
                                                FloatArray(iou.remaining()).also { a -> iou.get(a) } to FloatArray(low.remaining()).also { a -> low.get(a) }
                                            }
                                        }
                                    }
                                }
                            }.map { f -> f.get() }
                            for ((k, res) in results.withIndex()) {
                                val (px, py) = batch[k]
                                if (covered[py * img.width + px]) continue
                                val (iou, lowArr) = res
                                val low = FloatBuffer.wrap(lowArr)
                                // Candidate masks from best IoU down; first one that is a plausible piece wins.
                                var accepted = false
                                for (m in iou.indices.sortedByDescending { iou[it] }) {
                                    if (iou[m] < cfg.minIou) break
                                    val piece = maskToPiece(low, m, px, py, x0, y0, s, iw, ih, img, detector, covered, pieces.size) ?: continue
                                    stats.masks++
                                    pieces.add(piece)
                                    accepted = true
                                    break
                                }
                                // Not a piece here: don't prompt this spot again.
                                if (!accepted) {
                                    val r = cfg.pointSpacing
                                    for (yy in max(0, py - r)..min(img.height - 1, py + r)) for (xx in max(0, px - r)..min(img.width - 1, px + r)) covered[yy * img.width + xx] = true
                                }
                            }
                        }
                    }
                }
            }
        }
        stats.decoderMs += System.currentTimeMillis() - t1
    }

    private fun piecePrompts(img: RgbImage, x0: Int, y0: Int, tw: Int, th: Int, seg: Segmentation, covered: BooleanArray, cfg: Config): List<Pair<Int, Int>> {
        val w = tw; val h = th
        val big = 1 shl 20
        val d = IntArray(w * h)
        for (y in 0 until h) for (x in 0 until w) d[y * w + x] = if (seg.labels[(y0 + y) * img.width + x0 + x] != 0 && !covered[(y0 + y) * img.width + x0 + x]) big else 0
        // Two-pass chamfer distance (3-4 weights).
        for (y in 0 until h) for (x in 0 until w) {
            val i = y * w + x
            if (d[i] == 0) continue
            var v = d[i]
            if (x > 0) v = min(v, d[i - 1] + 3)
            if (y > 0) { v = min(v, d[i - w] + 3); if (x > 0) v = min(v, d[i - w - 1] + 4); if (x < w - 1) v = min(v, d[i - w + 1] + 4) }
            if (x == 0 || y == 0 || x == w - 1 || y == h - 1) v = min(v, 3)
            d[i] = v
        }
        for (y in h - 1 downTo 0) for (x in w - 1 downTo 0) {
            val i = y * w + x
            if (d[i] == 0) continue
            var v = d[i]
            if (x < w - 1) v = min(v, d[i + 1] + 3)
            if (y < h - 1) { v = min(v, d[i + w] + 3); if (x < w - 1) v = min(v, d[i + w + 1] + 4); if (x > 0) v = min(v, d[i + w - 1] + 4) }
            d[i] = v
        }
        // Local maxima, deepest first, with non-maximum suppression at ~half a piece.
        // A piece's inscribed radius is ~0.3 of its size (= 2 x spacing); deeper points belong to big
        // non-piece regions (box, chairs), shallower ones to slivers. Chamfer units are 3 per pixel.
        val est = 2.0 * cfg.pointSpacing
        val minDepth = (3 * 0.12 * est).toInt()
        val maxDepth = (3 * 0.6 * est).toInt()
        val cand = ArrayList<Int>()
        val step = max(1, cfg.pointSpacing / 6)
        for (y in step until h - step step step) for (x in step until w - step step step) {
            val i = y * w + x
            val v = d[i]
            if (v < minDepth || v > maxDepth) continue
            if (v >= d[i - step] && v >= d[i + step] && v >= d[i - step * w] && v >= d[i + step * w]) cand.add(i)
        }
        cand.sortByDescending { d[it] }
        val out = ArrayList<Pair<Int, Int>>()
        val r = cfg.pointSpacing
        for (i in cand) {
            val x = i % w; val y = i / w
            if (out.any { (ox, oy) -> (ox - x0 - x) * (ox - x0 - x) + (oy - y0 - y) * (oy - y0 - y) < r * r }) continue
            out.add(x0 + x to y0 + y)
            if (out.size >= cfg.maxPromptsPerTile) break
        }
        return out
    }

    /** Scanline-fills a polygon (slightly enlarged about its centroid) into [mask]. */
    private fun fillPolygon(xs: DoubleArray, ys: DoubleArray, mask: BooleanArray, w: Int, h: Int, grow: Double) {
        val n = xs.size
        val cx = xs.average(); val cy = ys.average()
        val gx = DoubleArray(n); val gy = DoubleArray(n)
        for (i in 0 until n) {
            val dx = xs[i] - cx; val dy = ys[i] - cy
            val l = kotlin.math.hypot(dx, dy).coerceAtLeast(1e-6)
            gx[i] = xs[i] + dx / l * grow; gy[i] = ys[i] + dy / l * grow
        }
        val y0 = max(0, gy.min().toInt()); val y1 = min(h - 1, gy.max().toInt() + 1)
        val xsx = DoubleArray(n)
        for (y in y0..y1) {
            val sy = y + 0.5
            var k = 0
            for (i in 0 until n) {
                val j = (i + 1) % n
                if ((gy[i] <= sy && gy[j] > sy) || (gy[j] <= sy && gy[i] > sy)) xsx[k++] = gx[i] + (sy - gy[i]) / (gy[j] - gy[i]) * (gx[j] - gx[i])
            }
            java.util.Arrays.sort(xsx, 0, k)
            var q = 0
            while (q + 1 < k) {
                for (x in max(0, xsx[q].toInt())..min(w - 1, xsx[q + 1].toInt())) mask[y * w + x] = true
                q += 2
            }
        }
    }

    /** Upsamples one low-res mask around the prompt, traces it and checks it is a piece. */
    private fun maskToPiece(
        low: FloatBuffer, m: Int, px: Int, py: Int, x0: Int, y0: Int, s: Double, iw: Int, ih: Int,
        img: RgbImage, detector: PieceDetector, covered: BooleanArray, id: Int,
    ): Piece? {
        val off = m * lowRes * lowRes
        val f = lowRes / input.toDouble() // input px -> low-res px
        // Flood fill in low-res space from the prompt to get the component's bounding box.
        val lx = ((px - x0 + 0.5) * s * f).toInt().coerceIn(0, lowRes - 1)
        val ly = ((py - y0 + 0.5) * s * f).toInt().coerceIn(0, lowRes - 1)
        if (low.get(off + ly * lowRes + lx) <= 0f) return null
        val seen = BooleanArray(lowRes * lowRes)
        val stack = IntArray(lowRes * lowRes)
        var sp = 0
        stack[sp++] = ly * lowRes + lx; seen[ly * lowRes + lx] = true
        var minX = lx; var maxX = lx; var minY = ly; var maxY = ly; var area = 0
        val lw = (iw * f).toInt(); val lh = (ih * f).toInt()
        while (sp > 0) {
            val i = stack[--sp]; area++
            val x = i % lowRes; val y = i / lowRes
            minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y)
            for ((dx, dy) in NEIGHBOURS) {
                val nx = x + dx; val ny = y + dy
                if (nx < 0 || ny < 0 || nx >= lw || ny >= lh) continue
                val j = ny * lowRes + nx
                if (!seen[j] && low.get(off + j) > 0f) { seen[j] = true; stack[sp++] = j }
            }
        }
        // Reject tiny specks, huge regions (table, box) and masks cut by the tile edge
        // (an overlapping tile sees that piece whole).
        if (area < 12 || area > lowRes * lowRes / 6) return null
        if (minX == 0 || minY == 0 || maxX >= lw - 1 || maxY >= lh - 1) return null

        // Upsample the bounding box to source pixels with bilinear interpolation of the logits.
        val k = 1.0 / (f * s) // low-res px -> source px
        val bx0 = ((minX - 1) * k).toInt(); val by0 = ((minY - 1) * k).toInt()
        val bw = ((maxX - minX + 3) * k).toInt() + 1; val bh = ((maxY - minY + 3) * k).toInt() + 1
        val labels = IntArray(bw * bh)
        for (y in 0 until bh) for (x in 0 until bw) {
            val gx = (bx0 + x + 0.5) / k - 0.5; val gy = (by0 + y + 0.5) / k - 0.5
            val ix = gx.toInt().coerceIn(0, lowRes - 2); val iy = gy.toInt().coerceIn(0, lowRes - 2)
            val tx = (gx - ix).coerceIn(0.0, 1.0); val ty = (gy - iy).coerceIn(0.0, 1.0)
            val v = (low.get(off + iy * lowRes + ix) * (1 - tx) + low.get(off + iy * lowRes + ix + 1) * tx) * (1 - ty) +
                (low.get(off + (iy + 1) * lowRes + ix) * (1 - tx) + low.get(off + (iy + 1) * lowRes + ix + 1) * tx) * ty
            if (v > 0) labels[y * bw + x] = 1
        }
        // Keep the component containing the prompt.
        val sx = (px - x0 - bx0).coerceIn(0, bw - 1); val sy = (py - y0 - by0).coerceIn(0, bh - 1)
        if (labels[sy * bw + sx] == 0) return null
        val comp = IntArray(bw * bh)
        sp = 0
        val st2 = IntArray(bw * bh)
        st2[sp++] = sy * bw + sx; comp[sy * bw + sx] = 2
        var startIdx = sy * bw + sx
        while (sp > 0) {
            val i = st2[--sp]
            if (i < startIdx) startIdx = i
            val x = i % bw; val y = i / bw
            for ((dx, dy) in NEIGHBOURS) {
                val nx = x + dx; val ny = y + dy
                if (nx < 0 || ny < 0 || nx >= bw || ny >= bh) continue
                val j = ny * bw + nx
                if (labels[j] == 1 && comp[j] == 0) { comp[j] = 2; st2[sp++] = j }
            }
        }
        val (cx, cy) = PieceDetector.traceContour(comp, bw, bh, 2, startIdx % bw, startIdx / bw)
        val ox = (x0 + bx0).toDouble(); val oy = (y0 + by0).toDouble()
        for (i in cx.indices) { cx[i] += ox; cy[i] += oy }
        val piece = detector.analyzeContour(cx, cy, img, id) ?: return null
        if (!detector.isPlausible(piece)) return null
        markCovered(comp, bw, bh, x0 + bx0, y0 + by0, img, covered)
        return piece
    }

    private fun markCovered(comp: IntArray, bw: Int, bh: Int, ox: Int, oy: Int, img: RgbImage, covered: BooleanArray) {
        for (y in 0 until bh) for (x in 0 until bw) {
            if (comp[y * bw + x] == 0) continue
            val gx = ox + x; val gy = oy + y
            if (gx in 0 until img.width && gy in 0 until img.height) covered[gy * img.width + gx] = true
        }
    }

    override fun close() {
        pool.shutdown(); encoder.close(); decoder.close()
    }

    companion object {
        private val NEIGHBOURS = listOf(1 to 0, -1 to 0, 0 to 1, 0 to -1)
    }
}
