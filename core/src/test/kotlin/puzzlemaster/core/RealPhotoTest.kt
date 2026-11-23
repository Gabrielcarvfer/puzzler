package puzzlemaster.core

import org.junit.Test
import java.io.File
import javax.imageio.ImageIO

/**
 * Runs the pipeline on real frames captured by the app (Freeze saves them; copied into
 * src/test/resources/real). Writes annotated figures to ./test/real_<name>.png.
 */
class RealPhotoTest {
    @Test
    fun analyseCapturedFrames() {
        val dir = File("src/test/resources/real")
        val files = dir.listFiles { f -> f.name.endsWith(".png") }?.sorted() ?: return
        for (f in files) {
            val bi = ImageIO.read(f)
            val img = RgbImage(bi.width, bi.height, bi.getRGB(0, 0, bi.width, bi.height, null, 0, bi.width))
            val t0 = System.nanoTime()
            val det = PieceDetector(PieceDetector.Config(segmentation = Segmenter.Config(minAreaFraction = 0.00003))).detect(img)
            val matches = EdgeMatcher().matchInView(det.pieces)
            val ms = (System.nanoTime() - t0) / 1e6
            println("${f.name}: ${img.width}x${img.height}, threshold ${det.segmentation.threshold}, components ${det.segmentation.components.size}, pieces ${det.pieces.size}, rejected ${det.rejected}, matches ${matches.size} (%.0f ms)".format(ms))
            val em = EdgeMatcher()
            for (m in matches.take(10)) println("   #${m.a.pieceId}.${m.a.index}(${m.a.type}) <-> #${m.b.pieceId}.${m.b.index}(${m.b.type}) score %.3f shape %.3f len %.3f color %.3f".format(m.score, em.shapeError(m.a, m.b) / ((m.a.length + m.b.length) / 2), kotlin.math.abs(m.a.length - m.b.length) / ((m.a.length + m.b.length) / 2), em.colorDistance(m.a, m.b)))
            if (det.pieces.size <= 6) for (a in det.pieces) for (b in det.pieces) if (a.id < b.id) for (ea in a.edges) for (eb in b.edges) {
                val sc = em.score(ea, eb)
                if (sc.isFinite()) println("     all #${a.id}.${ea.index}(${ea.type}) #${b.id}.${eb.index}(${eb.type}) score %.3f shape %.3f".format(sc, em.shapeError(ea, eb) / ((ea.length + eb.length) / 2)))
            }
            figureDir()?.let { out ->
                SceneTest.drawOverlay(img, det.pieces, matches, null, File(out, "real_${f.nameWithoutExtension}.png"))
                // Segmentation mask: what the detector considers "piece" (white) vs background.
                val mask = java.awt.image.BufferedImage(img.width, img.height, java.awt.image.BufferedImage.TYPE_INT_RGB)
                val ok = det.pieces.size
                for (i in det.segmentation.labels.indices) {
                    val l = det.segmentation.labels[i]
                    mask.setRGB(i % img.width, i / img.width, if (l == 0) 0 else 0xFFFFFF)
                }
                ImageIO.write(mask, "png", File(out, "real_${f.nameWithoutExtension}_mask.png"))
                check(ok >= 0)
            }
        }
    }
}

class RealPhotoSamTest {
    @Test
    fun samOnCapturedFrames() {
        val models = File(System.getenv("PM_MODELS") ?: return)
        val enc = File(models, "mobile_sam_image_encoder.onnx").readBytes()
        val dec = File(models, "sam_mask_decoder_multi.onnx").readBytes()
        val files = File("src/test/resources/real").listFiles { f -> f.name.startsWith(System.getenv("PM_SAM_PREFIX") ?: "hd_20261122_151") && f.name.endsWith(".png") }?.sorted() ?: return
        SamSegmenter(enc, dec).use { sam ->
            for (f in files) {
                val bi = ImageIO.read(f)
                val img = RgbImage(bi.width, bi.height, bi.getRGB(0, 0, bi.width, bi.height, null, 0, bi.width))
                val detector = PieceDetector(PieceDetector.Config(segmentation = Segmenter.Config(minAreaFraction = 0.00003)))
                val stats = SamSegmenter.Stats()
                val t0 = System.nanoTime()
                // Same setup as the app's Scan HD: whole frame, one tile.
                val pieces = sam.detect(img, detector, SamSegmenter.Config(tileSize = maxOf(img.width, img.height)), stats = stats)
                val matches = EdgeMatcher().matchInView(pieces)
                println("SAM ${f.name}: ${img.width}x${img.height}, ${pieces.size} pieces, ${matches.size} matches | classic ${stats.classic}, tiles ${stats.tiles}, prompts ${stats.prompts}, encoder ${stats.encoderMs} ms, decoder ${stats.decoderMs} ms, total %.1f s".format((System.nanoTime() - t0) / 1e9))
                figureDir()?.let { SceneTest.drawOverlay(img, pieces, matches.take(5), null, File(it, "sam_${f.nameWithoutExtension}.png")) }
            }
        }
    }
}

/** Regressions from real photos where the user confirmed the suggested pair does NOT fit. */
class RealNonMatchTest {
    private fun load(name: String): RgbImage? {
        val f = File("src/test/resources/real/$name").takeIf { it.exists() } ?: return null
        val bi = ImageIO.read(f)
        return RgbImage(bi.width, bi.height, bi.getRGB(0, 0, bi.width, bi.height, null, 0, bi.width))
    }

    @Test
    fun twoPiecesThatDoNotFit() {
        val img = load("frame_20261122_150029.png") ?: return
        val pieces = PieceDetector().detect(img).pieces
        org.junit.Assert.assertTrue("both pieces detected", pieces.size >= 2)
        org.junit.Assert.assertEquals("no false pair", 0, EdgeMatcher().matchInView(pieces).size)
    }

    @Test
    fun twoPiecesThatDoFit() {
        val img = load("frame_20261122_150344.png") ?: return
        val pieces = PieceDetector().detect(img).pieces
        // Left piece (x~480) and middle piece (x~780) connect.
        val left = pieces.single { it.centroidX in 400.0..560.0 && it.centroidY in 650.0..800.0 }
        val mid = pieces.single { it.centroidX in 700.0..860.0 && it.centroidY in 650.0..800.0 }
        val m = EdgeMatcher().matchInView(pieces)
        println("real pair: " + m.joinToString { "#${it.a.pieceId}-#${it.b.pieceId} %.3f color %.3f quality %.2f/%.2f".format(it.score, EdgeMatcher().colorDistance(it.a, it.b), pieces.first { p -> p.id == it.a.pieceId }.quality, pieces.first { p -> p.id == it.b.pieceId }.quality) })
        org.junit.Assert.assertTrue("true pair found", m.any { setOf(it.a.pieceId, it.b.pieceId) == setOf(left.id, mid.id) })
        org.junit.Assert.assertTrue("nothing else", m.size == 1)
    }
}

class RealPhotoEdgeTest {
    @Test
    fun edgeMethodOnCapturedFrames() {
        val files = File("src/test/resources/real").listFiles { f -> f.name.endsWith(".png") }?.sorted() ?: return
        for (f in files) {
            val bi = ImageIO.read(f)
            val img = RgbImage(bi.width, bi.height, bi.getRGB(0, 0, bi.width, bi.height, null, 0, bi.width))
            val detector = PieceDetector()
            val stats = EdgeSegmenter.Stats()
            val pieces = EdgeSegmenter().findPieces(img, detector, stats)
            val matches = EdgeMatcher().matchInView(pieces)
            println("EDGE ${f.name}: ${pieces.size} pieces, ${matches.size} matches | regions ${stats.regions}, candidates ${stats.candidates}, evaluated ${stats.evaluated}, ${stats.ms} ms")
            figureDir()?.let { SceneTest.drawOverlay(img, pieces, matches, null, File(it, "edge_${f.nameWithoutExtension}.png")) }
        }
    }
}

class EdgeDebugTest {
    @Test
    fun debugLeftPiece() {
        val f = File("src/test/resources/real/frame_20261122_150344.png").takeIf { it.exists() } ?: return
        val bi = ImageIO.read(f)
        val img = RgbImage(bi.width, bi.height, bi.getRGB(0, 0, bi.width, bi.height, null, 0, bi.width))
        EdgeSegmenter(EdgeSegmenter.Config(debugPoint = 500 to 730)).findPieces(img, PieceDetector())
    }
}

class MatchInspectTest {
    /** For each suggested pair: crops of both pieces (matched edge thick) + the two edge profiles overlaid. */
    @Test
    fun inspectSuggestedPairs() {
        val out = figureDir() ?: return
        val f = File("src/test/resources/real/hd_20261122_152527.png").takeIf { it.exists() } ?: return
        val bi = ImageIO.read(f)
        val img = RgbImage(bi.width, bi.height, bi.getRGB(0, 0, bi.width, bi.height, null, 0, bi.width))
        val pieces = EdgeSegmenter().findPieces(img, PieceDetector())
        val em = EdgeMatcher()
        val matches = em.matchInView(pieces)
        val figs = ArrayList<java.awt.image.BufferedImage>()
        fun crop(p: Piece): RgbImage {
            val r = (p.scale * 1.1).toInt()
            val x0 = (p.centroidX - r).toInt().coerceIn(0, img.width - 1); val y0 = (p.centroidY - r).toInt().coerceIn(0, img.height - 1)
            val w = minOf(2 * r, img.width - x0); val h = minOf(2 * r, img.height - y0)
            val c = RgbImage(w, h); for (y in 0 until h) for (x in 0 until w) c[x, y] = img[x0 + x, y0 + y]
            return c
        }
        fun shifted(p: Piece, c: RgbImage): Piece {
            val r = (p.scale * 1.1).toInt()
            val dx = (p.centroidX - r).toInt().coerceIn(0, img.width - 1).toDouble(); val dy = (p.centroidY - r).toInt().coerceIn(0, img.height - 1).toDouble()
            return PieceDetector().analyzeContour(DoubleArray(p.xs.size) { p.xs[it] - dx }, DoubleArray(p.ys.size) { p.ys[it] - dy }, c, p.id)!!
        }
        for (m in matches) {
            val pa = pieces.first { it.id == m.a.pieceId }; val pb = pieces.first { it.id == m.b.pieceId }
            val ca = crop(pa); val cb = crop(pb)
            val sa = shifted(pa, ca); val sb = shifted(pb, cb)
            val ea = sa.edges.minBy { kotlin.math.hypot(it.midX - (m.a.midX - (pa.centroidX - (pa.scale * 1.1).toInt())), it.midY - (m.a.midY - (pa.centroidY - (pa.scale * 1.1).toInt()))) }
            val eb = sb.edges.minBy { kotlin.math.hypot(it.midX - (m.b.midX - (pb.centroidX - (pb.scale * 1.1).toInt())), it.midY - (m.b.midY - (pb.centroidY - (pb.scale * 1.1).toInt()))) }
            figs.add(renderPieceFigure(ca, sa, "#${pa.id} score %.3f".format(m.score), ea))
            figs.add(renderPieceFigure(cb, sb, "#${pb.id} shape %.3f color %.3f".format(em.shapeError(m.a, m.b) / m.a.length, em.colorDistance(m.a, m.b)), eb))
            // Profile overlay: A in white, B mirrored (as it would sit when joined) in red.
            val s = 300; val prof = java.awt.image.BufferedImage(s, s, java.awt.image.BufferedImage.TYPE_INT_RGB)
            val g = prof.createGraphics(); g.stroke = java.awt.BasicStroke(3f)
            val L = m.a.length; val k = (s - 40) / L
            fun drawEdge(xs: DoubleArray, ys: DoubleArray, col: java.awt.Color) { g.color = col; for (i in 1 until xs.size) g.drawLine((20 + xs[i - 1] * k).toInt(), (s / 2 + ys[i - 1] * k).toInt(), (20 + xs[i] * k).toInt(), (s / 2 + ys[i] * k).toInt()) }
            drawEdge(m.a.cx, m.a.cy, java.awt.Color.WHITE)
            val n = m.b.cx.size
            drawEdge(DoubleArray(n) { m.b.length - m.b.cx[n - 1 - it] + (L - m.b.length) / 2 }, DoubleArray(n) { -m.b.cy[n - 1 - it] }, java.awt.Color.RED)
            figs.add(prof)
        }
        saveMontage(figs, 3, File(out, "inspect_pairs.png"))
    }
}

class TruePairEdgeTest {
    @Test
    fun truePairWithEdgePipeline() {
        val f = File("src/test/resources/real/frame_20261122_150344.png").takeIf { it.exists() } ?: return
        val bi = ImageIO.read(f)
        val img = RgbImage(bi.width, bi.height, bi.getRGB(0, 0, bi.width, bi.height, null, 0, bi.width))
        val pieces = EdgeSegmenter(EdgeSegmenter.Config(minQuality = -10.0, refine = System.getenv("PM_NOREFINE") == null)).findPieces(img, PieceDetector())
        val em = EdgeMatcher()
        for (p in pieces) println("TP piece #${p.id} at %.0f,%.0f q %.2f".format(p.centroidX, p.centroidY, p.quality))
        val left = pieces.firstOrNull { it.centroidX in 400.0..560.0 && it.centroidY in 650.0..800.0 }
        val mid = pieces.firstOrNull { it.centroidX in 700.0..860.0 && it.centroidY in 650.0..800.0 }
        if (left == null || mid == null) { println("TP: pieces not found"); return }
        for (a in left.edges) for (b in mid.edges) {
            val sc = em.score(a, b)
            if (sc.isFinite()) println("TP edge ${a.index}-${b.index} score %.3f shape %.3f color %.3f".format(sc, em.shapeError(a, b) / a.length, em.colorDistance(a, b)))
        }
    }
}

class RealPhotoPatternTest {
    @Test
    fun patternDetectionOnCapturedFrames() {
        val files = File("src/test/resources/real").listFiles { f -> f.name.endsWith(".png") }?.sorted() ?: return
        for (f in files) {
            val bi = ImageIO.read(f)
            val img = RgbImage(bi.width, bi.height, bi.getRGB(0, 0, bi.width, bi.height, null, 0, bi.width))
            val detector = PieceDetector()
            val seedPieces = EdgeSegmenter().findPieces(img, detector)
            val pd = PatternDetector()
            val templates = pd.learnTemplates(seedPieces)
            val stats = PatternDetector.Stats()
            val pieces = pd.detect(img, templates, detector, stats)
            val matches = EdgeMatcher().matchInView(pieces)
            println("PATTERN ${f.name}: seeds ${seedPieces.size}, templates ${stats.templates} (${templates.map { it.signature }.distinct()}), peaks ${stats.peaks}, pieces ${pieces.size}, matches ${matches.size}, ${stats.ms} ms")
            figureDir()?.let { SceneTest.drawOverlay(img, pieces, matches, null, File(it, "pattern_${f.nameWithoutExtension}.png")) }
        }
    }
}

class FullResEdgeTest {
    @Test
    fun fullResolutionOnLatestScan() {
        val f = File("src/test/resources/real/hd_20261122_154338.png").takeIf { it.exists() } ?: return
        val bi = ImageIO.read(f)
        val img = RgbImage(bi.width, bi.height, bi.getRGB(0, 0, bi.width, bi.height, null, 0, bi.width))
        for (ws in listOf(1600, 2400)) {
            val st = EdgeSegmenter.Stats()
            val pieces = EdgeSegmenter(EdgeSegmenter.Config(workSize = ws)).findPieces(img, PieceDetector(), st)
            val matches = EdgeMatcher().matchInView(pieces)
            println("FULLRES workSize $ws: ${pieces.size} pieces, ${matches.size} matches, ${st.ms} ms")
            figureDir()?.let { SceneTest.drawOverlay(img, pieces, matches, null, File(it, "fullres_${ws}.png")) }
        }
    }
}

/** Dev helper: crops regions listed in test/crops.txt ("file x y w h name" per line) to test/crop_<name>.png. */
class CropTool {
    @Test
    fun crop() {
        val out = figureDir() ?: return
        val spec = File(out, "crops.txt").takeIf { it.exists() } ?: return
        for (line in spec.readLines().filter { it.isNotBlank() }) {
            val (file, x, y, w, h, name) = line.trim().split(Regex("\\s+"))
            val bi = ImageIO.read(File("src/test/resources/real/$file"))
            ImageIO.write(bi.getSubimage(x.toInt(), y.toInt(), w.toInt(), h.toInt()), "png", File(out, "crop_$name.png"))
        }
    }
}
private operator fun <T> List<T>.component6() = this[5]

/**
 * Ground truth from assembled pieces in the user's photos: two detected pieces whose edges lie on top
 * of each other are physically joined, hence true neighbours. Measures how the matcher ranks them.
 */
class JoinedPairsTest {
    private fun imagePoints(e: Edge): Pair<DoubleArray, DoubleArray> {
        val ux = (e.endX - e.startX) / e.length; val uy = (e.endY - e.startY) / e.length
        val n = e.cx.size
        return DoubleArray(n) { e.startX + e.cx[it] * ux - e.cy[it] * uy } to DoubleArray(n) { e.startY + e.cx[it] * uy + e.cy[it] * ux }
    }

    @Test
    fun evaluateOnJoinedPieces() {
        val files = File("src/test/resources/real").listFiles { f -> f.name.startsWith("hd_") && f.name.endsWith(".png") }?.sorted() ?: return
        val em = EdgeMatcher()
        var pairs = 0; var top1 = 0; var top3 = 0
        val trueScores = ArrayList<Double>(); val impostorScores = ArrayList<Double>()
        for (f in files) {
            val bi = ImageIO.read(f)
            val img = RgbImage(bi.width, bi.height, bi.getRGB(0, 0, bi.width, bi.height, null, 0, bi.width))
            val pieces = EdgeSegmenter().findPieces(img, PieceDetector())
            val edges = pieces.flatMap { it.edges }.filter { it.type != EdgeType.FLAT }
            val pts = edges.associateWith { imagePoints(it) }
            val joined = ArrayList<Pair<Edge, Edge>>()
            for (i in edges.indices) for (j in i + 1 until edges.size) {
                val a = edges[i]; val b = edges[j]
                if (a.pieceId == b.pieceId) continue
                if (kotlin.math.hypot(a.midX - b.midX, a.midY - b.midY) > a.length * 0.3) continue
                val (ax, ay) = pts.getValue(a); val (bx, by) = pts.getValue(b)
                var sum = 0.0
                for (k in ax.indices) { var best = Double.MAX_VALUE; for (q in bx.indices) best = minOf(best, kotlin.math.hypot(ax[k] - bx[q], ay[k] - by[q])); sum += best }
                if (sum / ax.size < 0.04 * a.length) joined.add(a to b)
            }
            for ((a, b) in joined) {
                val cands = edges.filter { em.compatible(a, it) }.map { it to em.score(a, it) }.filter { it.second.isFinite() }.sortedBy { it.second }
                val rank = cands.indexOfFirst { it.first === b }
                val ts = em.score(a, b)
                println("JOINED ${f.name} #${a.pieceId}-#${b.pieceId} types ${a.type}/${b.type} true score %.3f shape %.3f rank %d of %d, best other %.3f".format(
                    ts, em.shapeError(a, b) / a.length, rank + 1, cands.size, cands.firstOrNull { it.first !== b }?.second ?: Double.NaN))
                pairs++
                if (rank == 0) top1++
                if (rank in 0..2) top3++
                if (ts.isFinite()) trueScores.add(ts)
                cands.firstOrNull { it.first !== b }?.let { impostorScores.add(it.second) }
            }
            figureDir()?.let { dir ->
                SceneTest.drawOverlay(img, pieces, joined.map { EdgeMatch(it.first, it.second, 0.0) }, joined.map { true }, File(dir, "joined_${f.nameWithoutExtension}.png"))
            }
        }
        println("JOINED total: $pairs true pairs, top-1 $top1, top-3 $top3; true scores ${trueScores.sorted().map { "%.2f".format(it) }}; best impostors ${impostorScores.sorted().take(15).map { "%.2f".format(it) }}")
    }
}

class StripDebugTest {
    @Test
    fun stripRegion() {
        val f = File("src/test/resources/real/hd_20261122_154338.png").takeIf { it.exists() } ?: return
        val full = ImageIO.read(f)
        val bi = full.getSubimage(1700, 450, 1000, 800)
        val img = RgbImage(bi.width, bi.height, bi.getRGB(0, 0, bi.width, bi.height, null, 0, bi.width))
        val st = EdgeSegmenter.Stats()
        val pieces = EdgeSegmenter(EdgeSegmenter.Config(workSize = 1000, minAreaFraction = 0.01, maxAreaFraction = 0.2)).findPieces(img, PieceDetector(), st)
        println("STRIP ${pieces.size} pieces, evaluated ${st.evaluated}")
        figureDir()?.let { SceneTest.drawOverlay(img, pieces, emptyList(), null, File(it, "strip_debug.png")) }
    }
}

class PaletteCalibrationTest {
    @Test
    fun realPairPalettes() {
        val f = File("src/test/resources/real/frame_20261122_150344.png").takeIf { it.exists() } ?: return
        val g = File("src/test/resources/real/frame_20261122_150029.png")
        for ((file, label) in listOf(f to "TRUE pair photo", g to "FALSE pair photo")) {
            val bi = ImageIO.read(file)
            val img = RgbImage(bi.width, bi.height, bi.getRGB(0, 0, bi.width, bi.height, null, 0, bi.width))
            val pieces = PieceDetector().detect(img).pieces
            for (a in pieces) for (b in pieces) if (a.id < b.id) println("PALETTE $label #${a.id}-#${b.id}: %.3f".format(PieceDetector.paletteDistance(a.palette, b.palette)))
        }
    }
}
