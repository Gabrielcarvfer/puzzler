package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Test
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File

/**
 * Replays a scanning session pulled from the phone (../session/photos: photo + sensor data), through
 * exactly the code the app runs, then merges duplicates and solves. Writes test/session_pairs.jpg
 * (both pieces of every proposed pair side by side) and overlays of a few photos.
 */
class SessionReplayTest {
    class Crop(val img: Mat, val sideMid: Array<Point>, val sightings: Int)

    @Test
    fun replay() {
        val dir = figureDir() ?: return
        val photos = File("../session/photos").listFiles { f -> f.name.endsWith(".jpg") }?.sorted() ?: return
        if (photos.isEmpty()) return
        OpenCV.loadLocally()
        val hue = System.getenv("PM_HUE") != "0"
        val session = ScanSession(pipelineCfg = LivePipeline.Config(segmentation = TableSegmenter.Config(hueFusion = hue)))
        val crops = HashMap<Int, Crop>()
        var shown = 0
        session.debugHook = { image, pass ->
            for (s in pass.sightings) {
                // Keep one upright-ish crop per catalogue entry (latest sighting), with the entry's side midpoints.
                val p = s.piece
                val r = (p.scale * 0.95).toInt()
                val x0 = (p.centroidX - r).toInt().coerceIn(0, image.cols() - 2); val y0 = (p.centroidY - r).toInt().coerceIn(0, image.rows() - 2)
                val w = minOf(2 * r, image.cols() - x0); val h = minOf(2 * r, image.rows() - y0)
                val c = Mat(); Imgproc.resize(image.submat(Rect(x0, y0, w, h)), c, Size(260.0, 260.0))
                val k = 260.0 / (2 * r)
                Imgproc.polylines(c, listOf(MatOfPoint(*Array(p.xs.size) { Point((p.xs[it] - x0) * k, (p.ys[it] - y0) * k) })), true, Scalar(255.0, 230.0, 0.0), 1)
                val mids = Array(4) { Point() }
                for (i in 0 until 4) mids[s.map[i]] = Point((p.edges[i].midX - x0) * k, (p.edges[i].midY - y0) * k)
                crops[s.entry.id] = Crop(c, mids, s.entry.sightings)
            }
            if (shown < 4 && pass.sightings.size >= 4) {
                Imgcodecs.imwrite(File(dir, "session_view_${if (hue) "hue" else "base"}_${shown++}.jpg").path, LivePipelineTest.draw(image, pass.out))
            }
        }
        var newTotal = 0; var recTotal = 0; var clean = 0; var measurable = 0; var ms = 0L
        for (f in photos) {
            val meta = File(f.path.removeSuffix(".jpg") + ".txt").readLines().associate { l -> l.substringBefore(" ") to l.substringAfter(" ") }
            val g = meta.getValue("gravity").split(" ").map { it.toFloat() }.toFloatArray()
            val pass = session.process(f.readBytes(), meta.getValue("rotation").toInt(), g, meta.getValue("focalPerLongSide").toDouble())
            newTotal += pass.sightings.count { it.isNew }; recTotal += pass.sightings.count { !it.isNew }
            clean += pass.out.pieces.size; measurable += pass.measurable; ms += pass.pipelineMs + pass.observeMs
        }
        val cat = session.catalogue
        println("SESSION hueFusion=$hue ${photos.size} photos: free-lying pieces outlined $clean (measurable $measurable), new entries $newTotal, recognised $recTotal, %.1f s/photo".format(ms / 1000.0 / photos.size))
        val merges = cat.mergeDuplicates()
        val e = cat.entries
        println("SESSION catalogue ${e.size} pieces after ${merges.size} merges; measured once ${e.count { it.sightings == 1 }}, 2-4 times ${e.count { it.sightings in 2..4 }}, 5+ times ${e.count { it.sightings >= 5 }}; with a flat side ${e.count { en -> en.sides.any { it.type == EdgeType.FLAT } }}")
        val pairs = cat.solve()
        println("SESSION pairs ${pairs.size}: " + pairs.take(20).joinToString { "#${it.a.id}(x${it.a.sightings})-#${it.b.id}(x${it.b.sightings}) %.2f%%/%.2f%% colour %.2f".format(100 * it.gap, 100 * it.runnerUp, it.colour) })
        // Pair sheet.
        val rows = pairs.take(24).mapIndexedNotNull { i, p ->
            val a = crops[p.a.id] ?: return@mapIndexedNotNull null; val b = crops[p.b.id] ?: return@mapIndexedNotNull null
            val row = Mat.zeros(260, 530, a.img.type())
            a.img.copyTo(row.submat(Rect(0, 0, 260, 260))); b.img.copyTo(row.submat(Rect(270, 0, 260, 260)))
            Imgproc.circle(row, a.sideMid[p.sideA], 9, Scalar(0.0, 255.0, 0.0), -1)
            Imgproc.circle(row, Point(b.sideMid[p.sideB].x + 270, b.sideMid[p.sideB].y), 9, Scalar(0.0, 255.0, 0.0), -1)
            Imgproc.putText(row, "${i + 1}  gap %.1f%%  colour %.2f".format(100 * p.gap, p.colour), Point(6.0, 22.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255.0, 255.0, 255.0), 2)
            row
        }
        if (rows.isNotEmpty()) {
            val colsN = 3
            val sheet = Mat.zeros(((rows.size + colsN - 1) / colsN) * 270, colsN * 540, rows[0].type())
            rows.forEachIndexed { i, r -> r.copyTo(sheet.submat(Rect((i % colsN) * 540, (i / colsN) * 270, 530, 260))) }
            Imgcodecs.imwrite(File(dir, "session_pairs_${if (hue) "hue" else "base"}.jpg").path, sheet)
        }
    }
}
