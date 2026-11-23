package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Test
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File

/** One session photo, with every intermediate picture written out (mask, split, outlines + reject reasons). */
class SessionDebugTest {
    @Test
    fun onePhoto() {
        val dir = figureDir() ?: return
        val photos = File("../session/photos").listFiles { f -> f.name.endsWith(".jpg") }?.sorted() ?: return
        val f = photos.getOrNull((System.getenv("PM_PHOTO") ?: "12").toInt()) ?: return
        OpenCV.loadLocally()
        val meta = File(f.path.removeSuffix(".jpg") + ".txt").readLines().associate { l -> l.substringBefore(" ") to l.substringAfter(" ") }
        val g = meta.getValue("gravity").split(" ").map { it.toFloat() }.toFloatArray()
        val raw = Imgcodecs.imdecode(MatOfByte(*f.readBytes()), Imgcodecs.IMREAD_COLOR or Imgcodecs.IMREAD_IGNORE_ORIENTATION)
        val up = Mat()
        when (meta.getValue("rotation").toInt()) { 90 -> Core.rotate(raw, up, Core.ROTATE_90_CLOCKWISE); 180 -> Core.rotate(raw, up, Core.ROTATE_180); 270 -> Core.rotate(raw, up, Core.ROTATE_90_COUNTERCLOCKWISE); else -> raw.copyTo(up) }
        val view = Rectifier.squareUp(up, g, meta.getValue("focalPerLongSide").toDouble() * maxOf(up.cols(), up.rows()), maxTilt = 70.0, maxOut = 1.0)
        val factor = maxOf(1, Math.round(maxOf(view.image.cols(), view.image.rows()) / 2700.0).toInt())
        val small = Mat(); Imgproc.resize(view.image, small, Size(view.image.cols() / factor.toDouble(), view.image.rows() / factor.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val vs = view.valid?.let { v -> Mat().also { Imgproc.resize(v, it, small.size(), 0.0, 0.0, Imgproc.INTER_NEAREST) } }
        val seg = TableSegmenter.segment(small, vs, debug = { n, m -> Imgcodecs.imwrite(File(dir, "dbg_$n.png").path, m) })
        val out = LivePipeline(LivePipeline.Config(hiResRefine = System.getenv("PM_NOREFINE") == null)).processHiRes(view.image, view.valid, LivePipeline.Lean(view.nadirX, view.nadirY, view.focal), factor)
        println("DBG ${f.name}: tilt %.0f, factor $factor, blobs ${seg.blobs.size} (touching border ${seg.blobs.count { it.touchesBorder }}), body radius %.0f -> piece size %.0f; pieces ${out.pieces.size}, unanalysed ${out.loose.size}, rejects ${out.rejects.groupingBy { it.third.take(2) }.eachCount()}".format(view.tiltDeg, seg.bodyRadius * factor, seg.bodyRadius * factor * 2.8))
        val areas = seg.blobs.filter { !it.touchesBorder }.map { Math.sqrt(it.area) * factor }.sorted()
        println("DBG blob sizes (sqrt area, full-res px): " + areas.joinToString(" ") { "%.0f".format(it) })
        val vis = LivePipelineTest.draw(view.image, out)
        for ((xs, ys) in out.loose) Imgproc.polylines(vis, listOf(org.opencv.core.MatOfPoint(*Array(xs.size) { org.opencv.core.Point(xs[it], ys[it]) })), true, org.opencv.core.Scalar(0.0, 255.0, 255.0), 4)
        val sm = Mat(); Imgproc.resize(vis, sm, Size(), 0.5, 0.5); Imgcodecs.imwrite(File(dir, "dbg_view.jpg").path, sm)
    }
}
