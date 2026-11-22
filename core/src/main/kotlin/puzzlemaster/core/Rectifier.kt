package puzzlemaster.core

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Turns a tilted camera frame into a straight-down view of the table.
 *
 * The gravity sensor gives "up" in device coordinates; rotating a virtual camera so that it looks
 * along gravity is a pure rotation, i.e. the homography H = K R K^-1 (the table is flat, so this is
 * exact up to scale). Also reports the nadir (the pixel straight below the camera) and the focal
 * length of the rectified view, which the pipeline needs to undo the "visible side wall" effect of
 * the pieces' thickness.
 */
object Rectifier {
    class View(
        /** Rectified image (up to maxOut times the input size per side). */
        val image: Mat,
        /** 255 where the rectified image has real pixels. */
        val valid: Mat?,
        /** Rectified -> original frame homography, row-major 3x3; null when no warp was applied. */
        val back: DoubleArray?,
        val nadirX: Double, val nadirY: Double,
        /** Focal length (pixels) of the rectified view. */
        val focal: Double,
        val tiltDeg: Double,
    )

    /**
     * [upright]: portrait-upright frame (x right, y down). [gravity]: TYPE_GRAVITY sensor values in
     * device coordinates. [focalPx]: focal length in pixels of [upright].
     */
    fun squareUp(upright: Mat, gravity: FloatArray, focalPx: Double, minTilt: Double = 2.0, maxTilt: Double = 60.0, maxOut: Double = 1.25): View {
        val w = upright.cols(); val h = upright.rows()
        val cx = w / 2.0; val cy = h / 2.0
        val gn = sqrt((gravity[0] * gravity[0] + gravity[1] * gravity[1] + gravity[2] * gravity[2]).toDouble())
        // "Up" in camera coordinates (x right, y down, z forward out of the back of the phone).
        val nx = gravity[0] / gn; val ny = -gravity[1] / gn; val nz = -gravity[2] / gn
        val th = acos((-nz).coerceIn(-1.0, 1.0))
        val tilt = Math.toDegrees(th)
        if (tilt < minTilt || tilt > maxTilt) return View(upright, null, null, cx, cy, focalPx, tilt)
        // Rotation taking n to (0, 0, -1): axis n x t, angle acos(n . t).
        var ax = -ny; var ay = nx
        val an = sqrt(ax * ax + ay * ay).coerceAtLeast(1e-9); ax /= an; ay /= an
        val c = cos(th); val s = sin(th); val t = 1 - c
        val r = doubleArrayOf(
            t * ax * ax + c, t * ax * ay, s * ay,
            t * ax * ay, t * ay * ay + c, -s * ax,
            -s * ay, s * ax, c,
        )
        val k = Mat(3, 3, CvType.CV_64F); k.put(0, 0, focalPx, 0.0, cx, 0.0, focalPx, cy, 0.0, 0.0, 1.0)
        val rm = Mat(3, 3, CvType.CV_64F); rm.put(0, 0, *r)
        val hm = Mat(); Core.gemm(k, rm, 1.0, Mat(), 0.0, hm)
        Core.gemm(hm, k.inv(), 1.0, Mat(), 0.0, hm)
        // Fit the warped frame into an image of the same size.
        val corners = MatOfPoint2f(Point(0.0, 0.0), Point(w.toDouble(), 0.0), Point(w.toDouble(), h.toDouble()), Point(0.0, h.toDouble()))
        val wc = MatOfPoint2f(); Core.perspectiveTransform(corners, wc, hm)
        val pts = wc.toArray()
        val minX = pts.minOf { it.x }; val maxX = pts.maxOf { it.x }; val minY = pts.minOf { it.y }; val maxY = pts.maxOf { it.y }
        // Keep the frame's resolution (scale 1 at the centre) unless the result would get too big.
        val sc = minOf(1.0, maxOut * w / (maxX - minX), maxOut * h / (maxY - minY))
        val ow = Math.ceil((maxX - minX) * sc).toInt(); val oh = Math.ceil((maxY - minY) * sc).toInt()
        val size = org.opencv.core.Size(ow.toDouble(), oh.toDouble())
        val sm = Mat(3, 3, CvType.CV_64F); sm.put(0, 0, sc, 0.0, -minX * sc, 0.0, sc, -minY * sc, 0.0, 0.0, 1.0)
        val full = Mat(); Core.gemm(sm, hm, 1.0, Mat(), 0.0, full)
        val out = Mat(); Imgproc.warpPerspective(upright, out, full, size, Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0, 0.0))
        val ones = Mat(h, w, CvType.CV_8UC1, Scalar(255.0))
        val valid = Mat(); Imgproc.warpPerspective(ones, valid, full, size, Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, Scalar(0.0))
        val inv = DoubleArray(9); full.inv().get(0, 0, inv)
        // The virtual camera looks straight down: its principal point is the nadir.
        return View(out, valid, inv, sc * (cx - minX), sc * (cy - minY), focalPx * sc, tilt)
    }
}
