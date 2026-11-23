package puzzlemaster.core

import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.KeyPoint
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

/**
 * Everything that happens to one photo, shared by the phone app and the laptop replay tests so both
 * behave identically: decode -> upright -> squared up with gravity -> free-lying pieces outlined and
 * measured at full resolution -> camera motion since the previous photo -> catalogue updated.
 */
class ScanSession(val catalogue: PieceCatalogue = PieceCatalogue(), val cfg: Config = Config(), pipelineCfg: LivePipeline.Config = LivePipeline.Config()) {
    data class Config(
        /** Smallest piece (pixels across) whose sides are measured for the catalogue. */
        val minPiecePx: Double = 150.0,
        /** Pieces are found on a copy reduced to about this many pixels on the long side. */
        val searchSize: Double = 2700.0,
        val maxTilt: Double = 70.0,
    )

    class Pass(
        val out: LivePipeline.Out,
        val sightings: List<PieceCatalogue.Sighting>,
        /** Analysed image -> upright photo homography (row-major), or null if the photo was not warped. */
        val back: DoubleArray?,
        val frameW: Int, val frameH: Int,
        val measurable: Int,
        val pieceSize: Double,
        val tiltDeg: Double,
        val pipelineMs: Long, val observeMs: Long,
    )

    /** Test/debug hook: called with the analysed (squared-up) image before it is released. */
    var debugHook: ((Mat, Pass) -> Unit)? = null

    /** One pipeline per thread: photos can be analysed in parallel (see [process]). */
    private val pipelines = ThreadLocal.withInitial { LivePipeline(pipelineCfg) }
    internal val pipeline: LivePipeline get() = pipelines.get()
    /** Guards everything that changes shared state: camera tracking, the catalogue, [debugHook]. */
    private val lock = Any()

    /** Runs [f] while no photo is being committed (for catalogue maintenance from other threads). */
    fun <T> exclusive(f: () -> T): T = synchronized(lock, f)

    /**
     * Called from the analysing thread, in parallel with other photos, with the analysed image and a
     * pass that has no sightings yet (for per-photo work that doesn't touch the catalogue).
     */
    var analysisHook: ((Mat, Pass) -> Unit)? = null
    private val orb = ORB.create(3000)
    private val matcher = BFMatcher.create(Core.NORM_HAMMING, false)
    private var prevKp: Array<KeyPoint>? = null
    private var prevDesc: Mat? = null

    /**
     * [rotationDegrees]: clockwise rotation that makes the JPEG upright. [gravity]: gravity sensor
     * values at capture. [focalPerLongSide]: focal length divided by the photo's long side (zoom included).
     */
    fun process(jpeg: ByteArray, rotationDegrees: Int, gravity: FloatArray, focalPerLongSide: Double): Pass {
        val t0 = System.currentTimeMillis()
        val raw = Imgcodecs.imdecode(MatOfByte(*jpeg), Imgcodecs.IMREAD_COLOR or Imgcodecs.IMREAD_IGNORE_ORIENTATION)
        val up = Mat()
        when (rotationDegrees) {
            90 -> Core.rotate(raw, up, Core.ROTATE_90_CLOCKWISE)
            180 -> Core.rotate(raw, up, Core.ROTATE_180)
            270 -> Core.rotate(raw, up, Core.ROTATE_90_COUNTERCLOCKWISE)
            else -> raw.copyTo(up)
        }
        raw.release()
        val frameW = up.cols(); val frameH = up.rows()
        val view = Rectifier.squareUp(up, gravity, focalPerLongSide * maxOf(frameW, frameH), maxTilt = cfg.maxTilt, maxOut = 1.0)
        val factor = maxOf(1, Math.round(maxOf(view.image.cols(), view.image.rows()) / cfg.searchSize).toInt())
        val out = pipeline.processHiRes(view.image, view.valid, LivePipeline.Lean(view.nadirX, view.nadirY, view.focal), factor)
        val t1 = System.currentTimeMillis()
        val measurable = out.pieces.filter { it.scale >= cfg.minPiecePx }
        val size = if (out.pieces.isEmpty()) 0.0 else out.pieces.map { it.scale }.sorted()[out.pieces.size / 2]
        analysisHook?.invoke(view.image, Pass(out, emptyList(), view.back, frameW, frameH, measurable.size, size, view.tiltDeg, t1 - t0, 0))
        // Shared state: one photo at a time.
        val pass = synchronized(lock) {
            val t2 = System.currentTimeMillis()
            trackCamera(view.image)
            val sightings = catalogue.observe(view.image, measurable)
            val p = Pass(out, sightings, view.back, frameW, frameH, measurable.size, size, view.tiltDeg, t1 - t0, System.currentTimeMillis() - t2)
            debugHook?.invoke(view.image, p)
            p
        }
        if (view.image !== up) view.image.release()
        view.valid?.release(); up.release()
        return pass
    }

    /** Camera motion since the previous photo (matching the table itself), handed to the catalogue. */
    private fun trackCamera(image: Mat) {
        val f = 1000.0 / maxOf(image.cols(), image.rows())
        val small = Mat(); Imgproc.resize(image, small, Size(), f, f, Imgproc.INTER_AREA)
        val gray = Mat(); Imgproc.cvtColor(small, gray, Imgproc.COLOR_BGR2GRAY)
        val kpM = MatOfKeyPoint(); val desc = Mat()
        orb.detectAndCompute(gray, Mat(), kpM, desc)
        val kp = kpM.toArray()
        val pk = prevKp; val pd = prevDesc
        prevKp = kp; prevDesc = desc
        if (pk == null || pd == null || pd.empty() || desc.empty()) return
        val knn = ArrayList<MatOfDMatch>()
        matcher.knnMatch(pd, desc, knn, 2)
        val good = knn.map { it.toArray() }.filter { it.size == 2 && it[0].distance < 0.75f * it[1].distance }.map { it[0] }
        if (good.size < 25) return
        val src = MatOfPoint2f(*good.map { Point(pk[it.queryIdx].pt.x / f, pk[it.queryIdx].pt.y / f) }.toTypedArray())
        val dst = MatOfPoint2f(*good.map { Point(kp[it.trainIdx].pt.x / f, kp[it.trainIdx].pt.y / f) }.toTypedArray())
        val mask = Mat()
        val h = Calib3d.findHomography(src, dst, Calib3d.RANSAC, 6.0, mask)
        if (h.empty() || Core.countNonZero(mask) < 20) return
        val a = DoubleArray(9); h.get(0, 0, a)
        catalogue.applyCameraMotion(a)
    }
}
