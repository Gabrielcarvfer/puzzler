package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Test
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File

class BorderFinderTest {
    @Test
    fun edgeAndCornerPieces() {
        val model = File(System.getenv("PM_MODELS") ?: return, "detr.onnx").takeIf { it.exists() } ?: return
        val dir = figureDir() ?: return
        OpenCV.loadLocally()
        val finder = BorderFinder(model.readBytes())
        val dbg = File(dir, "bfdebug").apply { deleteRecursively(); mkdirs() }
        val files = listOf("border/border1.jpg")
        for (name in files) {
            val f = File("src/test/resources/$name").takeIf { it.exists() } ?: continue
            val img = Imgcodecs.imread(f.path, Imgcodecs.IMREAD_COLOR)
            finder.debugDir = if (name.startsWith("real")) dbg else null
            val t0 = System.currentTimeMillis()
            val found = finder.find(img, BorderFinder.Config(tilesAcross = (System.getenv("PM_TILES") ?: "6").toInt()))
            val ms = System.currentTimeMillis() - t0
            println("BORDERFIND ${f.name}: ${found.size} pieces, outlines ${found.count { it.outline != null }}, edge ${found.count { it.isEdge && !it.isCorner }}, corner ${found.count { it.isCorner }}, $ms ms")
            for (p in found) {
                val col = when { p.isCorner -> Scalar(0.0, 0.0, 255.0); p.isEdge -> Scalar(0.0, 140.0, 255.0); else -> Scalar(0.0, 200.0, 0.0) }
                if (p.outline != null) Imgproc.polylines(img, listOf(MatOfPoint(*p.outline)), true, col, if (p.isEdge) 4 else 1)
                else Imgproc.rectangle(img, p.box, Scalar(200.0, 200.0, 200.0), 1)
                for ((a, b) in p.flatSegments) Imgproc.line(img, a, b, Scalar(255.0, 0.0, 255.0), 6)
                if (p.isCorner) Imgproc.putText(img, "C", Point(p.box.x + p.box.width / 3.0, p.box.y + p.box.height / 1.5), Imgproc.FONT_HERSHEY_SIMPLEX, 2.0, Scalar(255.0, 255.0, 255.0), 5)
            }
            Imgcodecs.imwrite(File(dir, "borderfind_${f.nameWithoutExtension}.jpg").path, img)
            // Contact sheets of ALL detected pieces (for labelling): number, crop, flagged = orange frame.
            run {
                val per = 60; val ts = 130
                for (sheetNo in 0 until (found.size + per - 1) / per) {
                    val part = found.drop(sheetNo * per).take(per)
                    val sheet = org.opencv.core.Mat.zeros(((part.size + 9) / 10) * ts, 10 * ts, img.type())
                    part.forEachIndexed { k, p ->
                        val raw = Imgcodecs.imread(f.path)
                        val m = maxOf(p.box.width, p.box.height) * 3 / 4
                        val x0 = maxOf(0, p.box.x - m / 3); val y0 = maxOf(0, p.box.y - m / 3)
                        val r = org.opencv.core.Rect(x0, y0, minOf(raw.cols() - x0, p.box.width + 2 * m / 3), minOf(raw.rows() - y0, p.box.height + 2 * m / 3))
                        val t = org.opencv.core.Mat(); Imgproc.resize(raw.submat(r), t, org.opencv.core.Size(ts.toDouble(), ts.toDouble()))
                        if (p.isEdge) Imgproc.rectangle(t, Point(1.0, 1.0), Point(ts - 2.0, ts - 2.0), Scalar(0.0, 140.0, 255.0), 4)
                        Imgproc.putText(t, "${sheetNo * per + k}", Point(4.0, 20.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255.0, 255.0, 255.0), 2)
                        t.copyTo(sheet.submat(org.opencv.core.Rect((k % 10) * ts, (k / 10) * ts, ts, ts)))
                    }
                    Imgcodecs.imwrite(File(dir, "label_${sheetNo}.jpg").path, sheet)
                }
                File(dir, "label_boxes.txt").writeText(found.withIndex().joinToString("\n") { (i, p) -> "$i ${p.box.x} ${p.box.y} ${p.box.width} ${p.box.height} ${p.isEdge}" })
            }
            // Close-ups of every flagged piece.
            val hits = found.filter { it.isEdge }
            val tiles = hits.map { p ->
                val r = org.opencv.core.Rect(p.box.x - p.box.width / 2, p.box.y - p.box.height / 2, p.box.width * 2, p.box.height * 2)
                val rr = org.opencv.core.Rect(maxOf(0, r.x), maxOf(0, r.y), minOf(img.cols() - maxOf(0, r.x), r.width), minOf(img.rows() - maxOf(0, r.y), r.height))
                val t = org.opencv.core.Mat(); Imgproc.resize(img.submat(rr), t, org.opencv.core.Size(220.0, 220.0)); t
            }
            if (tiles.isNotEmpty()) {
                val rows = (tiles.size + 5) / 6
                val sheet = org.opencv.core.Mat.zeros(rows * 220, 6 * 220, img.type())
                tiles.forEachIndexed { k, t -> t.copyTo(sheet.submat(org.opencv.core.Rect((k % 6) * 220, (k / 6) * 220, 220, 220))); Imgproc.putText(sheet, "${k + 1}", Point((k % 6) * 220 + 6.0, (k / 6) * 220 + 28.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255.0, 255.0, 255.0), 2) }
                Imgcodecs.imwrite(File(dir, "borderhits_${f.nameWithoutExtension}.jpg").path, sheet)
            }
        }
        finder.close()
    }
}
