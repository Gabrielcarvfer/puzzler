package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Test
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File

class BorderDetectorTest {
    @Test
    fun straightSidesOnTablePhotos() {
        val photos = File("src/test/resources/map").listFiles { f -> f.name.endsWith(".jpg") }?.sorted() ?: return
        val dir = figureDir() ?: return
        OpenCV.loadLocally()
        for ((i, f) in photos.withIndex().filter { it.index in listOf(1, 4) }) {
            val m = Imgcodecs.imread(f.path, Imgcodecs.IMREAD_REDUCED_COLOR_2)
            val buf = ByteArray(m.cols() * m.rows() * 3); m.get(0, 0, buf)
            val px = IntArray(m.cols() * m.rows()) { val k = it * 3; (0xFF shl 24) or ((buf[k + 2].toInt() and 0xFF) shl 16) or ((buf[k + 1].toInt() and 0xFF) shl 8) or (buf[k].toInt() and 0xFF) }
            val pieces = EdgeSegmenter().findPieces(RgbImage(m.cols(), m.rows(), px), PieceDetector())
            val side = pieces.map { it.scale }.sorted().let { it[it.size / 2] }
            val g = Mat(); Imgproc.cvtColor(m, g, Imgproc.COLOR_BGR2GRAY)
            val t0 = System.currentTimeMillis()
            val segs = BorderDetector.find(g, side)
            println("BORDER photo $i: side %.0f px, ${segs.size} straight sides, ${System.currentTimeMillis() - t0} ms".format(side))
            for (s in segs) Imgproc.line(m, Point(s.x1, s.y1), Point(s.x2, s.y2), Scalar(0.0, 140.0, 255.0), 6)
            Imgcodecs.imwrite(File(dir, "border_$i.jpg").path, m)
        }
    }
}
