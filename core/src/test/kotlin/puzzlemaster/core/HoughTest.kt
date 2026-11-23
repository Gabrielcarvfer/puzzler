package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Test
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File

/**
 * Experiment: Hough circles on the round tab/blank knob heads, which on a standard die have one fixed
 * radius. Knob radius is calibrated from the pieces the edge detector already found.
 */
class HoughTest {
    @Test
    fun knobCircles() {
        val f = File("src/test/resources/map").listFiles { it.name.endsWith(".jpg") }?.sorted()?.getOrNull(3) ?: return
        OpenCV.loadLocally()
        val m = Imgcodecs.imread(f.path, Imgcodecs.IMREAD_REDUCED_COLOR_2)
        val buf = ByteArray(m.cols() * m.rows() * 3); m.get(0, 0, buf)
        val px = IntArray(m.cols() * m.rows()) { val k = it * 3; (0xFF shl 24) or ((buf[k + 2].toInt() and 0xFF) shl 16) or ((buf[k + 1].toInt() and 0xFF) shl 8) or (buf[k].toInt() and 0xFF) }
        val img = RgbImage(m.cols(), m.rows(), px)
        val pieces = EdgeSegmenter().findPieces(img, PieceDetector())
        val scale = pieces.map { it.scale }.sorted()[pieces.size / 2]
        // Knob head radius: fit a circle to each tab tip region (points within 0.15 L of the tip).
        val radii = ArrayList<Double>()
        for (p in pieces) for (e in p.edges) {
            if (e.type == EdgeType.FLAT) continue
            val tipY = if (e.type == EdgeType.TAB) e.cy.min() else e.cy.max()
            val pts = e.cx.indices.filter { kotlin.math.abs(e.cy[it] - tipY) < 0.17 * e.length }
            if (pts.size < 5) continue
            // Algebraic circle fit (Kasa).
            var sx = 0.0; var sy = 0.0; for (i in pts) { sx += e.cx[i]; sy += e.cy[i] }; sx /= pts.size; sy /= pts.size
            var suu = 0.0; var svv = 0.0; var suv = 0.0; var suuu = 0.0; var svvv = 0.0; var suvv = 0.0; var svuu = 0.0
            for (i in pts) { val u = e.cx[i] - sx; val v = e.cy[i] - sy; suu += u * u; svv += v * v; suv += u * v; suuu += u * u * u; svvv += v * v * v; suvv += u * v * v; svuu += v * u * u }
            val det = suu * svv - suv * suv
            if (kotlin.math.abs(det) < 1e-9) continue
            val uc = (0.5 * (suuu + suvv) * svv - 0.5 * (svvv + svuu) * suv) / det
            val vc = (0.5 * (svvv + svuu) * suu - 0.5 * (suuu + suvv) * suv) / det
            radii.add(kotlin.math.sqrt(uc * uc + vc * vc + (suu + svv) / pts.size))
        }
        radii.sort()
        val r = radii[radii.size / 2]
        println("HOUGH piece size %.0f px, knob radius median %.1f px (p25 %.1f, p75 %.1f), %d knobs measured".format(scale, r, radii[radii.size / 4], radii[radii.size * 3 / 4], radii.size))
        val g = Mat(); Imgproc.cvtColor(m, g, Imgproc.COLOR_BGR2GRAY)
        Imgproc.medianBlur(g, g, 5)
        val circles = Mat()
        val t0 = System.currentTimeMillis()
        Imgproc.HoughCircles(g, circles, Imgproc.HOUGH_GRADIENT_ALT, 1.5, r * 1.2, 120.0, 0.75, (r * 0.8).toInt(), (r * 1.25).toInt())
        val n = circles.cols()
        println("HOUGH ${n} circles in ${System.currentTimeMillis() - t0} ms")
        // How many circles lie on knobs of already-detected pieces (validation), and how many elsewhere.
        var onKnown = 0
        val vis = m.clone()
        for (i in 0 until n) {
            val c = circles.get(0, i); val cx = c[0]; val cy = c[1]; val cr = c[2]
            val near = pieces.any { p -> p.edges.any { e -> e.type != EdgeType.FLAT && kotlin.math.hypot(e.midX - cx, e.midY - cy) < cr * 1.6 } }
            if (near) onKnown++
            Imgproc.circle(vis, Point(cx, cy), cr.toInt(), if (near) Scalar(0.0, 255.0, 0.0) else Scalar(0.0, 255.0, 255.0), 3)
        }
        for (p in pieces) for (e in p.edges) if (e.type != EdgeType.FLAT) Imgproc.circle(vis, Point(e.midX, e.midY), 6, Scalar(0.0, 0.0, 255.0), -1)
        println("HOUGH circles on knobs of detected pieces: $onKnown / $n; detected pieces' knobs: ${pieces.sumOf { p -> p.edges.count { it.type != EdgeType.FLAT } }}")
        figureDir()?.let { Imgcodecs.imwrite(File(it, "hough.jpg").path, vis) }
    }
}
