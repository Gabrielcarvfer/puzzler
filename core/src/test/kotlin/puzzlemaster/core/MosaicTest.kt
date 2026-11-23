package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Test
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.DMatch
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File

/**
 * Builds a flat map of the table from several overlapping photos: ORB features, pairwise matching,
 * RANSAC homographies, all photos mapped into the frame of a reference photo.
 */
class MosaicTest {
    class Registration(val imgs: List<Mat>, val toRef: Array<Mat?>, val canvasT: Mat, val canvasW: Int, val canvasH: Int)

    @Test
    fun buildMosaic() {
        val dir = File("src/test/resources/map")
        val files = dir.listFiles { f -> f.name.endsWith(".jpg") }?.sorted() ?: return
        if (files.size < 2) return
        register(files)
    }

    companion object {
    fun register(files: List<File>): Registration {
        OpenCV.loadLocally()
        // Work at 1/4 resolution (2048 px) for registration.
        val imgs = files.map { Imgcodecs.imread(it.path, Imgcodecs.IMREAD_REDUCED_COLOR_4) }
        println("MOSAIC images: " + imgs.joinToString { "${it.cols()}x${it.rows()}" })
        val orb = ORB.create(6000)
        val kps = ArrayList<MatOfKeyPoint>(); val descs = ArrayList<Mat>()
        for (m in imgs) {
            val g = Mat(); Imgproc.cvtColor(m, g, Imgproc.COLOR_BGR2GRAY)
            val kp = MatOfKeyPoint(); val d = Mat()
            orb.detectAndCompute(g, Mat(), kp, d)
            kps.add(kp); descs.add(d)
        }
        val matcher = BFMatcher.create(Core.NORM_HAMMING, false)
        val n = imgs.size
        // Pairwise homographies (j -> i) with inlier counts.
        val H = Array(n) { arrayOfNulls<Mat>(n) }
        val inl = Array(n) { IntArray(n) }
        for (i in 0 until n) for (j in 0 until n) {
            if (i == j) continue
            val knn = ArrayList<MatOfDMatch>()
            matcher.knnMatch(descs[j], descs[i], knn, 2)
            val good = ArrayList<DMatch>()
            for (k in knn) { val a = k.toArray(); if (a.size == 2 && a[0].distance < 0.75 * a[1].distance) good.add(a[0]) }
            if (good.size < 30) continue
            val kj = kps[j].toArray(); val ki = kps[i].toArray()
            val src = MatOfPoint2f(*good.map { kj[it.queryIdx].pt }.toTypedArray())
            val dst = MatOfPoint2f(*good.map { ki[it.trainIdx].pt }.toTypedArray())
            val mask = Mat()
            val h = Calib3d.findHomography(src, dst, Calib3d.RANSAC, 4.0, mask)
            if (h.empty()) continue
            val c = Core.countNonZero(mask)
            inl[i][j] = c
            if (c >= 40) H[i][j] = h
        }
        for (i in 0 until n) println("MOSAIC inliers from $i: " + (0 until n).joinToString(" ") { "%4d".format(inl[i][it]) })
        // Reference = image with most total inliers; chain others via best path (BFS by inlier weight).
        val ref = (0 until n).maxBy { i -> inl[i].sum() }
        val toRef = arrayOfNulls<Mat>(n); toRef[ref] = Mat.eye(3, 3, CvType.CV_64F)
        var changed = true
        while (changed) {
            changed = false
            for (j in 0 until n) if (toRef[j] == null) {
                val via = (0 until n).filter { toRef[it] != null && H[it][j] != null }.maxByOrNull { inl[it][j] } ?: continue
                val m = Mat(); Core.gemm(toRef[via], H[via][j], 1.0, Mat(), 0.0, m); toRef[j] = m; changed = true
            }
        }
        println("MOSAIC reference $ref, registered ${toRef.count { it != null }}/$n")
        // Canvas bounds.
        var minX = 0.0; var minY = 0.0; var maxX = imgs[ref].cols().toDouble(); var maxY = imgs[ref].rows().toDouble()
        for (j in 0 until n) {
            val h = toRef[j] ?: continue
            val c = MatOfPoint2f(Point(0.0, 0.0), Point(imgs[j].cols().toDouble(), 0.0), Point(imgs[j].cols().toDouble(), imgs[j].rows().toDouble()), Point(0.0, imgs[j].rows().toDouble()))
            val o = MatOfPoint2f(); Core.perspectiveTransform(c, o, h)
            for (p in o.toArray()) { minX = minOf(minX, p.x); minY = minOf(minY, p.y); maxX = maxOf(maxX, p.x); maxY = maxOf(maxY, p.y) }
        }
        val scale = minOf(1.0, 4000.0 / maxOf(maxX - minX, maxY - minY))
        val T = Mat.eye(3, 3, CvType.CV_64F); T.put(0, 0, scale); T.put(1, 1, scale); T.put(0, 2, -minX * scale); T.put(1, 2, -minY * scale)
        val canvas = Mat.zeros(Size((maxX - minX) * scale, (maxY - minY) * scale), CvType.CV_8UC3)
        for (j in 0 until n) {
            val h = toRef[j] ?: continue
            val m = Mat(); Core.gemm(T, h, 1.0, Mat(), 0.0, m)
            val warped = Mat(); Imgproc.warpPerspective(imgs[j], warped, m, canvas.size())
            val g = Mat(); Imgproc.cvtColor(warped, g, Imgproc.COLOR_BGR2GRAY)
            val mask = Mat(); Imgproc.threshold(g, mask, 0.0, 255.0, Imgproc.THRESH_BINARY)
            warped.copyTo(canvas, mask)
            // Outline of each photo's footprint.
            val c = MatOfPoint2f(Point(0.0, 0.0), Point(imgs[j].cols().toDouble(), 0.0), Point(imgs[j].cols().toDouble(), imgs[j].rows().toDouble()), Point(0.0, imgs[j].rows().toDouble()))
            val o = MatOfPoint2f(); Core.perspectiveTransform(c, o, m)
            val pts = o.toArray()
            for (k in 0 until 4) Imgproc.line(canvas, pts[k], pts[(k + 1) % 4], Scalar(0.0, 255.0, 255.0), 4)
            Imgproc.putText(canvas, "$j", pts[0], Imgproc.FONT_HERSHEY_SIMPLEX, 3.0, Scalar(0.0, 255.0, 255.0), 6)
        }
        figureDir()?.let { Imgcodecs.imwrite(File(it, "mosaic.jpg").path, canvas) }
        println("MOSAIC canvas ${canvas.cols()}x${canvas.rows()}")
        return Registration(imgs, toRef, T, canvas.cols(), canvas.rows())
    }
    }
}
