package puzzlemaster.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import puzzlemaster.core.synth.SyntheticPuzzle
import kotlin.math.PI
import kotlin.math.hypot
import kotlin.random.Random

class PieceDetectorTest {
    private val detector = PieceDetector()

    private fun renderAlone(p: SyntheticPuzzle, r: Int, c: Int, angle: Double, scale: Double, rnd: Random): Pair<RgbImage, SyntheticPuzzle.Placement> {
        val size = (p.pieceSize * scale * 2.0).toInt()
        val (cx, cy) = p.pieceCenter(r, c)
        val pl = SyntheticPuzzle.Placement(r, c, angle, scale, size / 2.0, size / 2.0, cx, cy)
        val img = p.background(size, size, RgbImage.argb(30, 70, 40), rnd)
        p.render(img, pl, rnd)
        return img to pl
    }

    private fun debugDump(img: RgbImage, name: String, piece: Piece?) {
        val dir = System.getenv("PM_DEBUG")?.takeIf { it.isNotEmpty() } ?: return
        val s = 3
        val bi = java.awt.image.BufferedImage(img.width * s, img.height * s, java.awt.image.BufferedImage.TYPE_INT_RGB)
        val g = bi.createGraphics()
        val src = java.awt.image.BufferedImage(img.width, img.height, java.awt.image.BufferedImage.TYPE_INT_RGB)
        src.setRGB(0, 0, img.width, img.height, img.pixels, 0, img.width)
        g.drawImage(src, 0, 0, img.width * s, img.height * s, null)
        val (xs, ys) = detector.debugContour ?: return
        g.color = java.awt.Color.WHITE
        for (i in xs.indices) g.fillRect((xs[i] * s).toInt(), (ys[i] * s).toInt(), 1, 1)
        g.color = java.awt.Color.YELLOW
        for (i in detector.debugCandidates) g.drawOval((xs[i] * s).toInt() - 5, (ys[i] * s).toInt() - 5, 10, 10)
        g.color = java.awt.Color.RED
        piece?.cornerIdx?.forEach { i -> g.fillOval((xs[i] * s).toInt() - 4, (ys[i] * s).toInt() - 4, 8, 8) }
        javax.imageio.ImageIO.write(bi, "png", java.io.File(dir, "$name.png"))
    }

    @Test
    fun detectsCornersAndEdgeTypes() {
        val p = SyntheticPuzzle(6, 5, seed = 7)
        val rnd = Random(3)
        var cornerErr = 0.0
        var n = 0
        var typeOk = 0
        var failures = 0
        val figs = ArrayList<java.awt.image.BufferedImage>()
        for (r in 0 until p.rows) for (c in 0 until p.cols) {
            val (img, pl) = renderAlone(p, r, c, rnd.nextDouble(2 * PI), 0.9, rnd)
            val res = detector.detect(img)
            if (res.pieces.size != 1) {
                println("piece ($r,$c): ${res.pieces.size} pieces, rejected ${res.rejected}, comps ${res.segmentation.components.map { it.area }} thr ${res.segmentation.threshold}")
                failures++
                debugDump(img, "fail_${r}_$c", null)
                continue
            }
            val piece = res.pieces[0]
            val truth = p.corners(r, c).map { pl.toImage(it.first, it.second) }
            for (k in piece.cornerIdx) {
                val d = truth.minOf { hypot(it.first - piece.xs[k], it.second - piece.ys[k]) }
                cornerErr = maxOf(cornerErr, d / piece.scale)
            }
            val ids = p.identifyEdges(piece, pl)
            figs.add(renderPieceFigure(img, piece, "($r,$c)"))
            if (ids.values.toSet().size != 4) {
                println("piece ($r,$c): wrong corners")
                failures++
                debugDump(img, "corners_${r}_$c", piece)
                continue
            }
            for (e in piece.edges) {
                n++
                if (e.type == p.trueType(ids.getValue(e))) typeOk++
            }
        }
        figureDir()?.let { saveMontage(figs, 6, java.io.File(it, "single_pieces.png")) }
        println("max corner error ${percent(cornerErr)} of piece size, edge types correct $typeOk/$n")
        assertEquals(0, failures)
        assertTrue("corner error $cornerErr", cornerErr < 0.06)
        assertEquals(n, typeOk)
    }
}
