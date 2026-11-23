package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Assert.assertTrue
import org.junit.Test
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File

/**
 * Real-photo test of [BoxLocator]: pieces from 6 overlapping table photos are placed on the box
 * picture. Accuracy check without labels: the same physical piece seen in two photos must land on
 * the same spot.
 */
class BoxLocatorTest {
    @Test
    fun placesPiecesConsistently() {
        val boxFile = File("src/test/resources/box/box.jpg").takeIf { it.exists() } ?: return
        val photos = File("src/test/resources/map").listFiles { f -> f.name.endsWith(".jpg") }?.sorted() ?: return
        OpenCV.loadLocally()
        val box = Imgcodecs.imread(boxFile.path, Imgcodecs.IMREAD_REDUCED_COLOR_2)
        val loc = BoxLocator.fromPhoto(box, listOf(Point(127.0, 215.0), Point(3977.0, 205.0), Point(3990.0, 2974.0), Point(119.0, 2966.0)))
        val reg = MosaicTest.register(photos)
        val detector = PieceDetector()
        val all = ArrayList<Triple<Int, BoxLocator.Placement, Pair<Double, Double>>>()
        var locateMs = 0L
        for ((pi, f) in photos.withIndex()) {
            val m = BoxLocator.grayWorld(Imgcodecs.imread(f.path, Imgcodecs.IMREAD_REDUCED_COLOR_2))
            val buf = ByteArray(m.cols() * m.rows() * 3); m.get(0, 0, buf)
            val px = IntArray(m.cols() * m.rows()) { val k = it * 3; (0xFF shl 24) or ((buf[k + 2].toInt() and 0xFF) shl 16) or ((buf[k + 1].toInt() and 0xFF) shl 8) or (buf[k].toInt() and 0xFF) }
            val pieces = EdgeSegmenter().findPieces(RgbImage(m.cols(), m.rows(), px), detector)
            val toC = Mat(); Core.gemm(reg.canvasT, reg.toRef[pi], 1.0, Mat(), 0.0, toC)
            val t0 = System.currentTimeMillis()
            for (p in pieces) {
                val pl = loc.locate(m, p) ?: continue
                val o = MatOfPoint2f(); Core.perspectiveTransform(MatOfPoint2f(Point(p.centroidX / 2, p.centroidY / 2)), o, toC)
                all.add(Triple(pi, pl, o.toArray()[0].x to o.toArray()[0].y))
            }
            locateMs += System.currentTimeMillis() - t0
        }
        var dup = 0; var agree = 0; var dupC = 0; var agreeC = 0
        for (i in all.indices) for (j in i + 1 until all.size) {
            val (pa, a, ca) = all[i]; val (pb, b, cb) = all[j]
            if (pa == pb || kotlin.math.hypot(ca.first - cb.first, ca.second - cb.second) > 25) continue
            val same = kotlin.math.hypot(a.col - b.col, a.row - b.row) < 0.6
            dup++; if (same) agree++
            if (a.confident && b.confident) { dupC++; if (same) agreeC++ }
        }
        val conf = all.count { it.second.confident }
        println("LOCATOR ${all.size} placements, ${conf} confident, %.0f ms/piece; same piece twice -> same spot: $agree/$dup, confident: $agreeC/$dupC".format(locateMs.toDouble() / all.size))
        figureDir()?.let { dir ->
            val vis = loc.poster.clone()
            for ((_, pl, _) in all.filter { it.second.confident }) {
                Imgproc.rectangle(vis, Point((pl.col - 0.5) * loc.cell, (pl.row - 0.5) * loc.cell), Point((pl.col + 0.5) * loc.cell, (pl.row + 0.5) * loc.cell), Scalar(0.0, 255.0, 0.0), 2)
            }
            Imgcodecs.imwrite(File(dir, "locator_confident.jpg").path, vis)
        }
        assertTrue("confident placements must agree across photos ($agreeC/$dupC)", dupC == 0 || agreeC >= 0.85 * dupC)
    }
}

class ExportPosterTest {
    @Test
    fun exportPoster() {
        val boxFile = File("src/test/resources/box/box.jpg").takeIf { it.exists() } ?: return
        val dir = figureDir() ?: return
        OpenCV.loadLocally()
        val box = Imgcodecs.imread(boxFile.path, Imgcodecs.IMREAD_REDUCED_COLOR_2)
        val loc = BoxLocator.fromPhoto(box, listOf(Point(127.0, 215.0), Point(3977.0, 205.0), Point(3990.0, 2974.0), Point(119.0, 2966.0)))
        Imgcodecs.imwrite(File(dir, "box_poster.png").path, loc.poster)
        println("POSTER ${loc.cols}x${loc.rows} cell ${loc.cell}")
    }
}

class VerifiedPairsTest {
    /** Runs the app's scan pipeline on the user's latest scan photo and draws the pairs. */
    @Test
    fun pairsOnScan() {
        val boxFile = File("src/test/resources/box/box.jpg").takeIf { it.exists() } ?: return
        val scan = File("src/test/resources/real").listFiles { f -> f.name.startsWith("scan_") && f.name.endsWith(".jpg") }?.maxOrNull() ?: return
        OpenCV.loadLocally()
        val loc = BoxLocator.fromPhoto(Imgcodecs.imread(boxFile.path, Imgcodecs.IMREAD_REDUCED_COLOR_2), listOf(Point(127.0, 215.0), Point(3977.0, 205.0), Point(3990.0, 2974.0), Point(119.0, 2966.0)))
        val m0 = Imgcodecs.imread(scan.path)
        val buf = ByteArray(m0.cols() * m0.rows() * 3); m0.get(0, 0, buf)
        val px = IntArray(m0.cols() * m0.rows()) { val k = it * 3; (0xFF shl 24) or ((buf[k + 2].toInt() and 0xFF) shl 16) or ((buf[k + 1].toInt() and 0xFF) shl 8) or (buf[k].toInt() and 0xFF) }
        val img = RgbImage(m0.cols(), m0.rows(), px)
        val pieces = EdgeSegmenter().findPieces(img, PieceDetector())
        val m = BoxLocator.grayWorld(m0)
        val pls = pieces.mapNotNull { loc.locate(m, it) }
        val pairs = loc.verifiedPairs(pls)
        println("PAIRS ${scan.name}: ${pieces.size} pieces, ${pls.size} placed, ${pairs.size} verified pairs: " + pairs.take(10).joinToString { "#${it.a.piece.id}-#${it.b.piece.id} %.2f".format(it.score) })
        figureDir()?.let { dir ->
            SceneTest.drawOverlay(img, pieces, pairs.map { EdgeMatch(it.edgeA!!, it.edgeB!!, it.score) }, null, File(dir, "pairs_${scan.nameWithoutExtension}.png"))
        }
    }
}

class MapPairsTest {
    /** Whole-table pairing: all 6 photos -> unique pieces on the map -> box placements -> verified pairs. */
    @Test
    fun pairsAcrossWholeTable() {
        val boxFile = File("src/test/resources/box/box.jpg").takeIf { it.exists() } ?: return
        val photos = File("src/test/resources/map").listFiles { f -> f.name.endsWith(".jpg") }?.sorted() ?: return
        OpenCV.loadLocally()
        val loc = BoxLocator.fromPhoto(Imgcodecs.imread(boxFile.path, Imgcodecs.IMREAD_REDUCED_COLOR_2), listOf(Point(127.0, 215.0), Point(3977.0, 205.0), Point(3990.0, 2974.0), Point(119.0, 2966.0)))
        val reg = MosaicTest.register(photos)
        class MP(val photo: Int, val pl: BoxLocator.Placement, val mx: Double, val my: Double, val ox: DoubleArray = DoubleArray(0), val oy: DoubleArray = DoubleArray(0))
        val all = ArrayList<MP>()
        var nextId = 0
        for ((pi, f) in photos.withIndex()) {
            val m = BoxLocator.grayWorld(Imgcodecs.imread(f.path, Imgcodecs.IMREAD_REDUCED_COLOR_2))
            val buf = ByteArray(m.cols() * m.rows() * 3); m.get(0, 0, buf)
            val px = IntArray(m.cols() * m.rows()) { val k = it * 3; (0xFF shl 24) or ((buf[k + 2].toInt() and 0xFF) shl 16) or ((buf[k + 1].toInt() and 0xFF) shl 8) or (buf[k].toInt() and 0xFF) }
            val img = RgbImage(m.cols(), m.rows(), px)
            val det = PieceDetector()
            val pieces = EdgeSegmenter().findPieces(img, det).mapNotNull { det.analyzeContour(it.xs, it.ys, img, nextId++) }
            val toC = Mat(); Core.gemm(reg.canvasT, reg.toRef[pi], 1.0, Mat(), 0.0, toC)
            for (p in pieces) {
                val pl = loc.locate(m, p) ?: continue
                val o = MatOfPoint2f(); Core.perspectiveTransform(MatOfPoint2f(Point(p.centroidX / 2, p.centroidY / 2)), o, toC)
                val ol = MatOfPoint2f(); Core.perspectiveTransform(MatOfPoint2f(*Array(p.xs.size) { Point(p.xs[it] / 2, p.ys[it] / 2) }), ol, toC)
                val oa = ol.toArray()
                all.add(MP(pi, pl, o.toArray()[0].x, o.toArray()[0].y, DoubleArray(oa.size) { oa[it].x }, DoubleArray(oa.size) { oa[it].y }))
            }
        }
        // Unique pieces: keep the best-scoring sighting.
        val uniq = ArrayList<MP>()
        for (mp in all.sortedByDescending { it.pl.score }) if (uniq.none { kotlin.math.hypot(it.mx - mp.mx, it.my - mp.my) < 25 }) uniq.add(mp)
        val pairs = loc.verifiedPairs(uniq.map { it.pl })
        println("MAPPAIRS ${all.size} sightings, ${uniq.size} unique pieces, ${pairs.size} verified pairs")
        for ((k, pr) in pairs.take(16).withIndex()) println("MAPPAIRS #${k + 1}: A score %.2f margin %.2f conf=%s | B score %.2f margin %.2f conf=%s | pair score %.2f".format(pr.a.score, pr.a.margin, pr.a.confident, pr.b.score, pr.b.margin, pr.b.confident, pr.score))
        val strict = loc.verifiedPairs(uniq.map { it.pl }.filter { it.confident })
        println("MAPPAIRS strict (both confident): ${strict.size}")
        figureDir()?.let { dir ->
            val canvas = javax.imageio.ImageIO.read(File(dir, "mosaic.jpg"))
            val g = canvas.createGraphics(); g.stroke = java.awt.BasicStroke(5f)
            val pos = uniq.associateBy { it.pl.piece.id }
            for ((k, pr) in pairs.withIndex()) {
                val a = pos.getValue(pr.a.piece.id); val b = pos.getValue(pr.b.piece.id)
                g.color = java.awt.Color.getHSBColor((k * 0.13f) % 1f, 1f, 1f)
                g.drawLine(a.mx.toInt(), a.my.toInt(), b.mx.toInt(), b.my.toInt())
                g.fillOval(a.mx.toInt() - 8, a.my.toInt() - 8, 16, 16); g.fillOval(b.mx.toInt() - 8, b.my.toInt() - 8, 16, 16)
            }
            javax.imageio.ImageIO.write(canvas, "jpg", File(dir, "map_pairs.jpg"))
            // Numbered version (same numbering as pair_sheet.png), first 16 pairs only.
            val num = javax.imageio.ImageIO.read(File(dir, "mosaic.jpg"))
            val ng = num.createGraphics()
            ng.setRenderingHint(java.awt.RenderingHints.KEY_ANTIALIASING, java.awt.RenderingHints.VALUE_ANTIALIAS_ON)
            ng.font = java.awt.Font("SansSerif", java.awt.Font.BOLD, 34)
            for ((k, pr) in pairs.take(16).withIndex()) {
                val a = pos.getValue(pr.a.piece.id); val b = pos.getValue(pr.b.piece.id)
                val col = java.awt.Color.getHSBColor((k * 0.17f) % 1f, 1f, 1f)
                ng.color = col; ng.stroke = java.awt.BasicStroke(4f)
                for (mp in listOf(a, b)) for (i in mp.ox.indices) { val j = (i + 1) % mp.ox.size; ng.drawLine(mp.ox[i].toInt(), mp.oy[i].toInt(), mp.ox[j].toInt(), mp.oy[j].toInt()) }
                ng.stroke = java.awt.BasicStroke(2f, java.awt.BasicStroke.CAP_ROUND, java.awt.BasicStroke.JOIN_ROUND, 1f, floatArrayOf(10f, 8f), 0f)
                ng.drawLine(a.mx.toInt(), a.my.toInt(), b.mx.toInt(), b.my.toInt())
                for (mp in listOf(a, b)) {
                    val t = "${k + 1}"
                    ng.color = java.awt.Color.BLACK; ng.fillOval(mp.mx.toInt() - 24, mp.my.toInt() - 24, 48, 48)
                    ng.color = col; ng.drawString(t, mp.mx.toInt() - ng.fontMetrics.stringWidth(t) / 2, mp.my.toInt() + 12)
                }
            }
            javax.imageio.ImageIO.write(num, "jpg", File(dir, "pairs_on_table.jpg"))
            // Close-up evidence for the first pairs: both pieces side by side + their box spots.
            val tiles = ArrayList<java.awt.image.BufferedImage>()
            val posterBi = java.awt.image.BufferedImage(loc.poster.cols(), loc.poster.rows(), java.awt.image.BufferedImage.TYPE_3BYTE_BGR)
            val posterRgb = Mat(); Imgproc.cvtColor(loc.poster, posterRgb, Imgproc.COLOR_BGR2RGB)
            val pb = ByteArray(loc.poster.cols() * loc.poster.rows() * 3); posterRgb.get(0, 0, pb)
            posterBi.raster.setDataElements(0, 0, loc.poster.cols(), loc.poster.rows(), pb)
            for (pr in pairs.take(16)) {
                val c0 = minOf(pr.a.col, pr.b.col) - 1; val r0 = minOf(pr.a.row, pr.b.row) - 1
                val x = (c0 * loc.cell).toInt().coerceIn(0, loc.poster.cols() - 4 * loc.cell); val y = (r0 * loc.cell).toInt().coerceIn(0, loc.poster.rows() - 4 * loc.cell)
                val crop = posterBi.getSubimage(x, y, 4 * loc.cell, 4 * loc.cell)
                val t = java.awt.image.BufferedImage(4 * loc.cell * 2, 4 * loc.cell * 2, java.awt.image.BufferedImage.TYPE_INT_RGB)
                val tg = t.createGraphics(); tg.drawImage(crop, 0, 0, t.width, t.height, null)
                tg.stroke = java.awt.BasicStroke(3f)
                for ((pl, col) in listOf(pr.a to java.awt.Color.GREEN, pr.b to java.awt.Color.MAGENTA)) {
                    tg.color = col
                    tg.drawRect(((pl.col - 0.5) * loc.cell - x).toInt() * 2, ((pl.row - 0.5) * loc.cell - y).toInt() * 2, loc.cell * 2, loc.cell * 2)
                }
                tiles.add(t)
            }
            saveMontage(tiles, 4, File(dir, "map_pairs_box.png"))
            // Pair sheet: piece A crop | piece B crop | box spots, plus the pair number.
            val photoMats = photos.map { Imgcodecs.imread(it.path, Imgcodecs.IMREAD_REDUCED_COLOR_2) }
            val byPiece = uniq.associateBy { it.pl.piece.id }
            fun crop(mp: MP, color: java.awt.Color): java.awt.image.BufferedImage {
                val p = mp.pl.piece; val m = photoMats[mp.photo]
                val r = (p.scale * 1.0).toInt()
                val x0 = (p.centroidX - r).toInt().coerceIn(0, m.cols() - 2 * r); val y0 = (p.centroidY - r).toInt().coerceIn(0, m.rows() - 2 * r)
                val sub = Mat(); Imgproc.cvtColor(m.submat(org.opencv.core.Rect(x0, y0, 2 * r, 2 * r)), sub, Imgproc.COLOR_BGR2RGB)
                val b = ByteArray(sub.cols() * sub.rows() * 3); sub.get(0, 0, b)
                val bi = java.awt.image.BufferedImage(sub.cols(), sub.rows(), java.awt.image.BufferedImage.TYPE_3BYTE_BGR)
                bi.raster.setDataElements(0, 0, sub.cols(), sub.rows(), b)
                val out = java.awt.image.BufferedImage(240, 240, java.awt.image.BufferedImage.TYPE_INT_RGB)
                val g2 = out.createGraphics(); g2.drawImage(bi, 0, 0, 240, 240, null)
                g2.color = color; g2.stroke = java.awt.BasicStroke(3f)
                val k = 240.0 / (2 * r)
                for (i in p.xs.indices) { val j = (i + 1) % p.xs.size; g2.drawLine(((p.xs[i] - x0) * k).toInt(), ((p.ys[i] - y0) * k).toInt(), ((p.xs[j] - x0) * k).toInt(), ((p.ys[j] - y0) * k).toInt()) }
                return out
            }
            val sheet = ArrayList<java.awt.image.BufferedImage>()
            for ((k, pr) in pairs.take(16).withIndex()) {
                val a = byPiece.getValue(pr.a.piece.id); val b = byPiece.getValue(pr.b.piece.id)
                val row = java.awt.image.BufferedImage(240 * 3, 240, java.awt.image.BufferedImage.TYPE_INT_RGB)
                val g2 = row.createGraphics()
                g2.drawImage(crop(a, java.awt.Color.GREEN), 0, 0, null)
                g2.drawImage(crop(b, java.awt.Color.MAGENTA), 240, 0, null)
                g2.drawImage(tiles.getOrNull(k) ?: java.awt.image.BufferedImage(1, 1, 1), 480, 0, 240, 240, null)
                g2.color = java.awt.Color.WHITE; g2.font = java.awt.Font("SansSerif", java.awt.Font.BOLD, 28)
                g2.drawString("${k + 1}", 8, 32)
                g2.font = java.awt.Font("SansSerif", java.awt.Font.BOLD, 16)
                g2.drawString("row ${pr.a.row.toInt() + 1} col ${pr.a.col.toInt() + 1}", 488, 230)
                sheet.add(row)
            }
            saveMontage(sheet, 2, File(dir, "pair_sheet.png"))
        }
    }
}
