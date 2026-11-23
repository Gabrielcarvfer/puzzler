package puzzlemaster.core

import org.junit.Test
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.imgcodecs.Imgcodecs
import java.io.File
import javax.imageio.ImageIO

/** Piece map of the whole table from several photos: detect per photo, place on the mosaic, merge, match globally. */
class MapTest {
    class MapPiece(val photo: Int, val piece: Piece, val mx: Double, val my: Double, val outlineX: DoubleArray, val outlineY: DoubleArray)

    @Test
    fun buildPieceMap() {
        val files = File("src/test/resources/map").listFiles { f -> f.name.endsWith(".jpg") }?.sorted() ?: return
        if (files.size < 2) return
        val reg = MosaicTest.register(files)
        val detector = PieceDetector()
        val all = ArrayList<MapPiece>()
        var nextId = 0
        for ((pi, f) in files.withIndex()) {
            val h = reg.toRef[pi] ?: continue
            val m = Imgcodecs.imread(f.path, Imgcodecs.IMREAD_REDUCED_COLOR_2) // 3072x4096
            val w = m.cols(); val hh = m.rows()
            val buf = ByteArray(w * hh * 3); m.get(0, 0, buf)
            val px = IntArray(w * hh) { val k = it * 3; (0xFF shl 24) or ((buf[k + 2].toInt() and 0xFF) shl 16) or ((buf[k + 1].toInt() and 0xFF) shl 8) or (buf[k].toInt() and 0xFF) }
            val img = RgbImage(w, hh, px)
            val t0 = System.currentTimeMillis()
            val pieces = EdgeSegmenter().findPieces(img, detector)
            // photo (half res) -> registration (quarter res) -> canvas
            val toCanvas = Mat(); Core.gemm(reg.canvasT, h, 1.0, Mat(), 0.0, toCanvas)
            fun map(xs: DoubleArray, ys: DoubleArray): Pair<DoubleArray, DoubleArray> {
                val src = MatOfPoint2f(*Array(xs.size) { Point(xs[it] / 2, ys[it] / 2) })
                val dst = MatOfPoint2f(); Core.perspectiveTransform(src, dst, toCanvas)
                val a = dst.toArray()
                return DoubleArray(a.size) { a[it].x } to DoubleArray(a.size) { a[it].y }
            }
            for (p in pieces) {
                val (ox, oy) = map(p.xs, p.ys)
                val (cx, cy) = map(doubleArrayOf(p.centroidX), doubleArrayOf(p.centroidY))
                // Re-id globally (edges keep photo-local geometry, which is fine for matching).
                val g = detector.analyzeContour(p.xs, p.ys, img, nextId++) ?: continue
                all.add(MapPiece(pi, g, cx[0], cy[0], ox, oy))
            }
            println("MAP photo $pi: ${pieces.size} pieces (${System.currentTimeMillis() - t0} ms)")
        }
        // Merge duplicates seen in overlapping photos: keep the best-quality instance.
        val canvasScale = all.map { val p = it.piece; PieceDetector.perimeter(it.outlineX, it.outlineY) / PieceDetector.perimeter(p.xs, p.ys) * p.scale }.sorted().let { it[it.size / 2] }
        val kept = ArrayList<MapPiece>()
        for (mp in all.sortedByDescending { it.piece.quality }) {
            if (kept.any { kotlin.math.hypot(it.mx - mp.mx, it.my - mp.my) < canvasScale * 0.5 }) continue
            kept.add(mp)
        }
        println("MAP total detections ${all.size}, unique pieces ${kept.size}, piece size on map %.0f px".format(canvasScale))
        // Global matching across the whole map.
        val matcher = EdgeMatcher()
        val matches = matcher.matchInView(kept.map { it.piece })
        println("MAP matches ${matches.size}: " + matches.take(15).joinToString { "%.3f".format(it.score) })

        figureDir()?.let { dir ->
            val canvas = ImageIO.read(File(dir, "mosaic.jpg"))
            val g = canvas.createGraphics()
            g.setRenderingHint(java.awt.RenderingHints.KEY_ANTIALIASING, java.awt.RenderingHints.VALUE_ANTIALIAS_ON)
            g.stroke = java.awt.BasicStroke(2f)
            val byId = kept.associateBy { it.piece.id }
            for (mp in kept) {
                g.color = java.awt.Color(0, 230, 255)
                val n = mp.outlineX.size
                for (i in 0 until n) { val j = (i + 1) % n; g.drawLine(mp.outlineX[i].toInt(), mp.outlineY[i].toInt(), mp.outlineX[j].toInt(), mp.outlineY[j].toInt()) }
            }
            g.stroke = java.awt.BasicStroke(4f)
            for ((k, m) in matches.withIndex()) {
                val a = byId[m.a.pieceId] ?: continue; val b = byId[m.b.pieceId] ?: continue
                g.color = java.awt.Color.getHSBColor((k * 0.13f) % 1f, 1f, 1f)
                g.drawLine(a.mx.toInt(), a.my.toInt(), b.mx.toInt(), b.my.toInt())
                g.fillOval(a.mx.toInt() - 7, a.my.toInt() - 7, 14, 14); g.fillOval(b.mx.toInt() - 7, b.my.toInt() - 7, 14, 14)
            }
            ImageIO.write(canvas, "jpg", File(dir, "piece_map.jpg"))
        }
    }
}
