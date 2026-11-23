package puzzlemaster.core

import org.junit.Assert.assertTrue
import org.junit.Test
import puzzlemaster.core.synth.SyntheticPuzzle
import java.util.concurrent.Executors
import kotlin.math.PI
import kotlin.random.Random

/**
 * Whole-collection search: every piece of an N-piece puzzle is "photographed" separately (own random
 * rotation, lighting, noise), detected from pixels, and then each edge's true partner must be found
 * among all edges of all N pieces.
 */
class ScaleTest {
    private fun run(cols: Int, rows: Int, minTop1: Double, minTop5: Double, standardDie: Boolean = false) {
        val t0 = System.nanoTime()
        val p = SyntheticPuzzle(cols, rows, seed = cols * 1000L + rows, standardDie = standardDie)
        val detector = PieceDetector()
        val threads = Runtime.getRuntime().availableProcessors()
        val exec = Executors.newFixedThreadPool(threads)
        val cells = (0 until rows).flatMap { r -> (0 until cols).map { c -> r to c } }
        val results = try {
            cells.map { (r, c) ->
                exec.submit<Triple<Piece?, SyntheticPuzzle.Placement, RgbImage>> {
                    val rnd = Random(r * 100_003L + c)
                    val scale = 0.9
                    val size = (p.pieceSize * scale * 1.9).toInt()
                    val (cx, cy) = p.pieceCenter(r, c)
                    val pl = SyntheticPuzzle.Placement(r, c, rnd.nextDouble(2 * PI), scale, size / 2.0, size / 2.0, cx, cy)
                    val img = p.background(size, size, RgbImage.argb(28, 66, 42), rnd)
                    p.render(img, pl, rnd, gain = rnd.nextDouble(0.95, 1.05))
                    val det = detector.detect(img, firstId = r * cols + c)
                    Triple(det.pieces.singleOrNull(), pl, img)
                }
            }.map { it.get() }
        } finally { exec.shutdown() }
        val tDetect = (System.nanoTime() - t0) / 1e9

        val detected = results.filter { it.first != null }
        val imageOf = detected.associate { it.first!!.id to it.third }
        val truth = HashMap<Edge, TrueEdge>()
        val byTruth = HashMap<TrueEdge, Edge>()
        var typeErrors = 0
        for ((pc, pl) in detected) {
            val ids = p.identifyEdges(pc!!, pl)
            for ((e, t) in ids) {
                truth[e] = t; byTruth[t] = e
                if (e.type != p.trueType(t)) typeErrors++
            }
        }
        val all = detected.flatMap { it.first!!.edges }
        val queries = all.filter { it.type != EdgeType.FLAT && byTruth[p.neighbour(truth.getValue(it))] != null }
        val t1 = System.nanoTime()
        val index = EdgeIndex(all)
        val answers = index.queryAll(queries, k = 5)
        val tMatch = (System.nanoTime() - t1) / 1e9
        var top1 = 0; var top5 = 0
        val wrong = ArrayList<java.awt.image.BufferedImage>()
        val right = ArrayList<java.awt.image.BufferedImage>()
        val pieceOf = detected.associate { it.first!!.id to it.first!! }
        for ((q, ans) in queries.zip(answers)) {
            val partner = byTruth.getValue(p.neighbour(truth.getValue(q))!!)
            val rank = ans.indexOfFirst { it.b === partner }
            // Figures: query | matcher's best pick | true partner (thick edge = the edge in question)
            val list = if (rank == 0) right else wrong
            if (ans.isNotEmpty() && list.size < 3 * 12) {
                val best = ans[0].b
                list.add(renderPieceFigure(imageOf.getValue(q.pieceId), pieceOf[q.pieceId], "query ${truth[q]}", q))
                list.add(renderPieceFigure(imageOf.getValue(best.pieceId), pieceOf[best.pieceId], "best %.3f %s".format(ans[0].score, if (rank == 0) "OK" else "WRONG"), best))
                list.add(renderPieceFigure(imageOf.getValue(partner.pieceId), pieceOf[partner.pieceId], "true partner, rank ${if (rank < 0) ">5" else rank + 1}", partner))
            }
            if (rank == 0) top1++
            if (rank in 0..4) top5++
        }
        val n = cols * rows
        figureDir()?.let {
            saveMontage(right, 6, java.io.File(it, "scale_${n}_correct.png"))
            saveMontage(wrong, 6, java.io.File(it, "scale_${n}_mistakes.png"))
            saveMontage(detected.take(48).map { (pc, _, img) -> renderPieceFigure(img, pc, "#${pc!!.id}") }, 8, java.io.File(it, "scale_${n}_samples.png"))
        }
        val r1 = top1.toDouble() / queries.size; val r5 = top5.toDouble() / queries.size
        println(
            (if (standardDie) "standard die " else "") + "%5d pieces: detected %d (%s), edge-type errors %d, %d edge queries, top-1 %s, top-5 %s | render+detect %.1fs, match %.1fs"
                .format(n, detected.size, percent(detected.size.toDouble() / n), typeErrors, queries.size, percent(r1), percent(r5), tDetect, tMatch),
        )
        assertTrue("detection rate", detected.size >= n * 0.99)
        assertTrue("top-1 $r1", r1 >= minTop1)
        assertTrue("top-5 $r5", r5 >= minTop5)
    }

    @Test fun pieces100() = run(10, 10, 0.97, 0.98)
    @Test fun pieces500() = run(25, 20, 0.95, 0.97)
    @Test fun pieces1000() = run(40, 25, 0.95, 0.97)
    @Test fun pieces2000() = run(50, 40, 0.94, 0.97)
    @Test fun pieces4000() = run(80, 50, 0.93, 0.97)
    @Test fun pieces8000() = run(100, 80, 0.92, 0.97)

    // Standard die: shape is nearly useless, the picture must identify neighbours.
    @Test fun standardDie500() = run(25, 20, 0.97, 0.98, standardDie = true)
    @Test fun standardDie2000() = run(50, 40, 0.95, 0.97, standardDie = true)
}
