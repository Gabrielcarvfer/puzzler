package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Test
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File

/**
 * Exports real piece patches for training/evaluating the placement model: every detected piece from
 * the table photos, straightened via its 4 corners into a 128x128 image (corner square = centre 64x64),
 * pixels outside the outline set to grey. Also writes the NCC placement and map position (for
 * cross-photo consistency) to patches.csv.
 */
class ExportPatchesTest {
    @Test
    fun exportRealPatches() {
        val boxFile = File("src/test/resources/box/box.jpg").takeIf { it.exists() } ?: return
        val photos = File("src/test/resources/map").listFiles { f -> f.name.endsWith(".jpg") }?.sorted() ?: return
        val out = File(figureDir() ?: return, "patches").apply { mkdirs() }
        OpenCV.loadLocally()
        val loc = BoxLocator.fromPhoto(Imgcodecs.imread(boxFile.path, Imgcodecs.IMREAD_REDUCED_COLOR_2), listOf(Point(127.0, 215.0), Point(3977.0, 205.0), Point(3990.0, 2974.0), Point(119.0, 2966.0)))
        val reg = MosaicTest.register(photos)
        val csv = StringBuilder("id,photo,mapx,mapy,col,row,rot,score,margin,confident\n")
        var id = 0
        for ((pi, f) in photos.withIndex()) {
            val raw = Imgcodecs.imread(f.path, Imgcodecs.IMREAD_REDUCED_COLOR_2)
            val m = BoxLocator.grayWorld(raw)
            val buf = ByteArray(m.cols() * m.rows() * 3); m.get(0, 0, buf)
            val px = IntArray(m.cols() * m.rows()) { val k = it * 3; (0xFF shl 24) or ((buf[k + 2].toInt() and 0xFF) shl 16) or ((buf[k + 1].toInt() and 0xFF) shl 8) or (buf[k].toInt() and 0xFF) }
            val pieces = EdgeSegmenter().findPieces(RgbImage(m.cols(), m.rows(), px), PieceDetector())
            val toC = Mat(); Core.gemm(reg.canvasT, reg.toRef[pi], 1.0, Mat(), 0.0, toC)
            for (p in pieces) {
                // Straighten (raw colours: the model must learn to cope with the room light).
                val h = Imgproc.getPerspectiveTransform(MatOfPoint2f(*p.cornerIdx.map { Point(p.xs[it], p.ys[it]) }.toTypedArray()),
                    MatOfPoint2f(Point(32.0, 32.0), Point(96.0, 32.0), Point(96.0, 96.0), Point(32.0, 96.0)))
                val patch = Mat(); Imgproc.warpPerspective(raw, patch, h, Size(128.0, 128.0))
                val wo = MatOfPoint2f(); Core.perspectiveTransform(MatOfPoint2f(*Array(p.xs.size) { Point(p.xs[it], p.ys[it]) }), wo, h)
                val mask = Mat.zeros(Size(128.0, 128.0), CvType.CV_8UC1)
                Imgproc.fillPoly(mask, listOf(MatOfPoint(*wo.toArray())), Scalar(255.0))
                Imgproc.erode(mask, mask, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0)))
                val grey = Mat(128, 128, patch.type(), Scalar(128.0, 128.0, 128.0)); patch.copyTo(grey, mask)
                Imgcodecs.imwrite(File(out, "p$id.png").path, grey)
                Imgcodecs.imwrite(File(out, "m$id.png").path, mask)
                val pl = loc.locate(m, p)
                val o = MatOfPoint2f(); Core.perspectiveTransform(MatOfPoint2f(Point(p.centroidX / 2, p.centroidY / 2)), o, toC)
                val c = o.toArray()[0]
                csv.append("$id,$pi,%.1f,%.1f,%.3f,%.3f,%d,%.3f,%.3f,%s\n".format(java.util.Locale.US, c.x, c.y, pl?.col ?: -1.0, pl?.row ?: -1.0, pl?.rot ?: -1, pl?.score ?: 0.0, pl?.margin ?: 0.0, pl?.confident ?: false))
                id++
            }
        }
        File(out, "patches.csv").writeText(csv.toString())
        // Rectified poster at 64 px per cell for training.
        val boxFull = Imgcodecs.imread(boxFile.path)
        val big = BoxLocator.fromPhoto(boxFull, listOf(Point(254.0, 430.0), Point(7954.0, 410.0), Point(7980.0, 5948.0), Point(238.0, 5932.0)), cell = 64)
        Imgcodecs.imwrite(File(out, "poster64.png").path, big.poster)
        println("EXPORT $id patches, poster ${big.poster.cols()}x${big.poster.rows()} (${big.cols}x${big.rows})")
    }
}
