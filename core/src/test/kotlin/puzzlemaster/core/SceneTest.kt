package puzzlemaster.core

import org.junit.Assert.assertTrue
import org.junit.Test
import puzzlemaster.core.synth.SyntheticPuzzle
import java.awt.BasicStroke
import java.awt.Color
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import kotlin.math.PI
import kotlin.math.hypot
import kotlin.random.Random

/** End-to-end: a photo-like scene of scattered, rotated pieces -> detection -> in-view matching. */
class SceneTest {
    class Scene(val img: RgbImage, val placements: List<SyntheticPuzzle.Placement>)

    private fun scene(p: SyntheticPuzzle, w: Int, h: Int, scale: Double, rnd: Random): Scene {
        val img = p.background(w, h, RgbImage.argb(28, 66, 42), rnd)
        val cells = ArrayList<Pair<Int, Int>>()
        for (r in 0 until p.rows) for (c in 0 until p.cols) cells.add(r to c)
        cells.shuffle(rnd)
        val cell = p.pieceSize * scale * 1.75
        val gx = (w / cell).toInt(); val gy = (h / cell).toInt()
        require(gx * gy >= cells.size) { "scene too small" }
        val slots = (0 until gx * gy).shuffled(rnd)
        val pls = cells.mapIndexed { i, (r, c) ->
            val s = slots[i]
            val (cx, cy) = p.pieceCenter(r, c)
            val tx = (s % gx + 0.5) * cell + rnd.nextDouble(-0.08, 0.08) * cell + (w - gx * cell) / 2
            val ty = (s / gx + 0.5) * cell + rnd.nextDouble(-0.08, 0.08) * cell + (h - gy * cell) / 2
            SyntheticPuzzle.Placement(r, c, rnd.nextDouble(2 * PI), scale, tx, ty, cx, cy)
        }
        for (pl in pls) p.render(img, pl, rnd, gain = rnd.nextDouble(0.95, 1.05))
        return Scene(img, pls)
    }

    @Test
    fun findsMatchesBetweenPiecesInView() {
        var totalTrue = 0; var totalReported = 0; var totalCorrect = 0
        for ((i, seed) in listOf(11L, 12L, 13L).withIndex()) {
            val p = SyntheticPuzzle(6, 5, seed = seed)
            val rnd = Random(seed)
            val sc = scene(p, 1600, 1200, 0.9, rnd)
            val det = PieceDetector().detect(sc.img)
            // Which placement does each detected piece belong to? (nearest centre)
            val owner = det.pieces.associateWith { pc ->
                sc.placements.minBy { pl -> val (x, y) = pl.toImage(pl.cx, pl.cy); hypot(x - pc.centroidX, y - pc.centroidY) }
            }
            val truth = HashMap<Edge, TrueEdge>()
            for (pc in det.pieces) truth.putAll(p.identifyEdges(pc, owner.getValue(pc)))
            val matches = EdgeMatcher().matchInView(det.pieces)
            val correct = matches.map { m -> p.neighbour(truth.getValue(m.a)) == truth.getValue(m.b) }
            val trueLinks = det.pieces.sumOf { pc -> pc.edges.count { p.neighbour(truth.getValue(it)) != null } } / 2
            println("scene $seed: detected ${det.pieces.size}/${sc.placements.size} pieces, reported ${matches.size} matches, ${correct.count { it }} correct, of $trueLinks true connections")
            assertTrue(det.pieces.size == sc.placements.size)
            totalTrue += trueLinks; totalReported += matches.size; totalCorrect += correct.count { it }
            figureDir()?.let { dir -> drawOverlay(sc.img, det.pieces, matches, correct, File(dir, "scene_$i.png")) }
        }
        val precision = totalCorrect.toDouble() / totalReported
        val recall = totalCorrect.toDouble() / totalTrue
        println("in-view matching: precision ${percent(precision)}, recall ${percent(recall)}")
        assertTrue("precision $precision", precision >= 0.9)
        assertTrue("recall $recall", recall >= 0.6)
    }

    companion object {
        fun drawOverlay(img: RgbImage, pieces: List<Piece>, matches: List<EdgeMatch>, correct: List<Boolean>?, out: File) {
            val bi = BufferedImage(img.width, img.height, BufferedImage.TYPE_INT_RGB)
            bi.setRGB(0, 0, img.width, img.height, img.pixels, 0, img.width)
            val g = bi.createGraphics()
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
            val typeColor = mapOf(EdgeType.FLAT to Color(200, 200, 200), EdgeType.TAB to Color(255, 160, 0), EdgeType.BLANK to Color(0, 200, 255))
            for (pc in pieces) {
                val n = pc.xs.size
                for (e in 0 until 4) {
                    g.color = typeColor.getValue(pc.edges[e].type)
                    g.stroke = BasicStroke(2.5f)
                    val from = pc.cornerIdx[e]; val to = pc.cornerIdx[(e + 1) % 4]
                    var i = from
                    while (i != to) { val j = (i + 1) % n; g.drawLine(pc.xs[i].toInt(), pc.ys[i].toInt(), pc.xs[j].toInt(), pc.ys[j].toInt()); i = j }
                }
                g.color = Color.RED
                for (k in pc.cornerIdx) g.fillOval(pc.xs[k].toInt() - 5, pc.ys[k].toInt() - 5, 10, 10)
                g.color = Color.WHITE
                g.drawString("#${pc.id}", pc.centroidX.toInt() - 8, pc.centroidY.toInt() + 4)
            }
            g.stroke = BasicStroke(3f)
            for ((i, m) in matches.withIndex()) {
                g.color = when (correct?.get(i)) { true -> Color(60, 255, 60); false -> Color(255, 40, 40); null -> Color.GREEN }
                g.drawLine(m.a.midX.toInt(), m.a.midY.toInt(), m.b.midX.toInt(), m.b.midY.toInt())
                g.fillOval(m.a.midX.toInt() - 4, m.a.midY.toInt() - 4, 8, 8)
                g.fillOval(m.b.midX.toInt() - 4, m.b.midY.toInt() - 4, 8, 8)
            }
            g.color = Color.WHITE
            g.drawString("orange=tab  cyan=blank  grey=flat  red dots=corners  green line=correct match  red line=wrong match", 10, 20)
            ImageIO.write(bi, "png", out)
        }
    }
}
