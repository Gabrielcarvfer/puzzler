package puzzlemaster.core

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Separates pieces from the table, fast (OpenCV native).
 *
 * The table is the dominant colour of the scene. A first, global model of it gives a rough mask; the
 * table colour is then re-estimated *locally* from the unmasked pixels (so a lamp hotspot or a light
 * gradient across the table doesn't matter) and every pixel is compared against its local table
 * colour in Lab space, with brightness counting less than colour so soft shadows mostly stay table.
 */
object TableSegmenter {
    class Blob(val contour: MatOfPoint, val area: Double, val touchesBorder: Boolean)

    /** [bodyRadius]: inscribed radius of a typical piece body in pixels (0 if unknown). */
    /** [raw]: blobs before touching pieces are split apart. */
    class Result(val blobs: List<Blob>, val bodyRadius: Double, val raw: List<Blob> = blobs, val mask: Mat? = null, val minArea: Double = 0.0)

    data class Config(
        val lightnessWeight: Double = 0.5,
        val chromaWeight: Double = 1.5,
        val minThreshold: Double = 18.0,
        val maxThreshold: Double = 42.0,
        /** Table candidates: chroma distance to the dominant colour, and max local gradient. */
        val tableChromaRange: Double = 40.0,
        /** A smooth wood-coloured area must be at least this fraction of the image to count as table. */
        val tableMinAreaFraction: Double = 0.004,
        /** Light variations broader than this fraction of the image's long side are removed before thresholding. */
        val glareScale: Double = 0.22,
        /** Fuse hue evidence into the distance map (see segment()). Hue units are OpenCV's 0..180. */
        val hueFusion: Boolean = false,
        val hueFar: Double = 12.0,
        val hueNear: Double = 7.0,
        val hueBonus: Double = 45.0,
        val huePenalty: Double = 12.0,
        val tableSmoothness: Double = 14.0,
        val splitTouching: Boolean = true,
        /** Core = deeper inside the mask than this fraction of a typical piece's inscribed radius. */
        val coreDepth: Double = 0.6,
        /** Smallest blob, as a fraction of the image area. */
        val minAreaFraction: Double = 0.00008,
    )

    /** [valid]: optional 8-bit mask of real pixels (after rectification). [debug] receives the distance map. */
    fun segment(bgr: Mat, valid: Mat? = null, cfg: Config = Config(), debug: ((String, Mat) -> Unit)? = null): Result {
        val w = bgr.cols(); val h = bgr.rows()
        val blurred = Mat(); Imgproc.GaussianBlur(bgr, blurred, Size(3.0, 3.0), 0.0)
        val lab = Mat(); Imgproc.cvtColor(blurred, lab, Imgproc.COLOR_BGR2Lab)

        // ---- Global table colour: mode of the chroma histogram on a small copy. ----
        val f = 480.0 / max(w, h)
        val small = Mat(); Imgproc.resize(lab, small, Size(), f, f, Imgproc.INTER_AREA)
        val sv = if (valid != null) Mat().also { Imgproc.resize(valid, it, small.size(), 0.0, 0.0, Imgproc.INTER_NEAREST) } else null
        val sw = small.cols(); val sh = small.rows()
        val px = ByteArray(sw * sh * 3); small.get(0, 0, px)
        val vm = sv?.let { ByteArray(sw * sh).also { b -> it.get(0, 0, b) } }
        val hist = IntArray(32 * 32)
        for (i in 0 until sw * sh) {
            if (vm != null && vm[i].toInt() == 0) continue
            hist[((px[i * 3 + 1].toInt() and 0xFF) shr 3) * 32 + ((px[i * 3 + 2].toInt() and 0xFF) shr 3)]++
        }
        val mode = hist.indices.maxBy { hist[it] }
        val ma = mode / 32; val mb = mode % 32
        val ls = ArrayList<Int>(); val las = ArrayList<Int>(); val lbs = ArrayList<Int>()
        for (i in 0 until sw * sh) {
            if (vm != null && vm[i].toInt() == 0) continue
            val a = px[i * 3 + 1].toInt() and 0xFF; val b = px[i * 3 + 2].toInt() and 0xFF
            if (abs((a shr 3) - ma) <= 1 && abs((b shr 3) - mb) <= 1) { ls.add(px[i * 3].toInt() and 0xFF); las.add(a); lbs.add(b) }
        }
        ls.sort(); las.sort(); lbs.sort()
        val g = doubleArrayOf(ls[ls.size / 2].toDouble(), las[las.size / 2].toDouble(), lbs[lbs.size / 2].toDouble())

        // ---- Table candidates on the small copy: roughly wood-coloured AND smooth (pieces are
        // textured and have edges; glare on the table is smooth and stays a candidate). ----
        val small32 = Mat(); small.convertTo(small32, CvType.CV_32FC3)
        val chans = ArrayList<Mat>(); Core.split(small32, chans)
        val gx = Mat(); val gy = Mat(); val grad = Mat()
        Imgproc.Sobel(chans[0], gx, CvType.CV_32F, 1, 0); Imgproc.Sobel(chans[0], gy, CvType.CV_32F, 0, 1)
        Core.magnitude(gx, gy, grad)
        Imgproc.blur(grad, grad, Size(5.0, 5.0))
        val gb = FloatArray(sw * sh); grad.get(0, 0, gb)
        // "Smooth" is relative to the photo: brighter light makes the wood grain stronger. The calmest
        // 30% of the image is table (or plain piece interiors); allow a bit more texture than that.
        val smoothLimit = run {
            val sorted = gb.copyOf(); sorted.sort()
            (1.7 * sorted[(sorted.size * 0.3).toInt()]).coerceIn(cfg.tableSmoothness, 3.0 * cfg.tableSmoothness)
        }
        val table = Mat(sh, sw, CvType.CV_32FC1)
        val tb = FloatArray(sw * sh)
        for (i in 0 until sw * sh) {
            val d = abs((px[i * 3 + 1].toInt() and 0xFF) - g[1]) + abs((px[i * 3 + 2].toInt() and 0xFF) - g[2])
            tb[i] = if (d < cfg.tableChromaRange && gb[i] < smoothLimit && (vm == null || vm[i].toInt() != 0)) 1f else 0f
        }
        table.put(0, 0, tb)
        // Only large connected smooth areas count as table: wood-coloured bits inside pieces don't,
        // while lamp or flash glare (smooth, big, but much brighter and paler) does.
        run {
            val t8 = Mat(); table.convertTo(t8, CvType.CV_8UC1, 255.0)
            val labels = Mat(); val stats = Mat(); val cents = Mat()
            val n = Imgproc.connectedComponentsWithStats(t8, labels, stats, cents, 8, CvType.CV_32S)
            val minArea = cfg.tableMinAreaFraction * sw * sh
            val keep = BooleanArray(n)
            for (i in 1 until n) keep[i] = stats.get(i, Imgproc.CC_STAT_AREA)[0] >= minArea
            val lb = IntArray(sw * sh); labels.get(0, 0, lb)
            for (i in 0 until sw * sh) if (!keep[lb[i]]) tb[i] = 0f
            table.put(0, 0, tb)
        }
        Imgproc.erode(table, table, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0)))
        debug?.let { d -> val t8 = Mat(); table.convertTo(t8, CvType.CV_8UC1, 255.0); d("table", t8); val g8 = Mat(); grad.convertTo(g8, CvType.CV_8UC1, 4.0); d("grad", g8) }

        // ---- Local table colour: normalised blur of the candidates, fine scale first, then coarse. ----
        val bgChans = ArrayList<Mat>()
        val kFine = (max(sw, sh) / 16) or 1; val kCoarse = (max(sw, sh) / 5) or 1
        val denF = Mat(); Imgproc.blur(table, denF, Size(kFine.toDouble(), kFine.toDouble()))
        val denC = Mat(); Imgproc.blur(table, denC, Size(kCoarse.toDouble(), kCoarse.toDouble()))
        val noFine = Mat(); Core.compare(denF, Scalar(0.08), noFine, Core.CMP_LT)
        val noCoarse = Mat(); Core.compare(denC, Scalar(0.02), noCoarse, Core.CMP_LT)
        for ((ci, c) in chans.withIndex()) {
            val masked = Mat(); Core.multiply(c, table, masked)
            val numF = Mat(); Imgproc.blur(masked, numF, Size(kFine.toDouble(), kFine.toDouble()))
            val numC = Mat(); Imgproc.blur(masked, numC, Size(kCoarse.toDouble(), kCoarse.toDouble()))
            val bg = Mat(); Core.divide(numF, denF, bg)
            val bgC = Mat(); Core.divide(numC, denC, bgC)
            bgC.setTo(Scalar(g[ci]), noCoarse)
            bgC.copyTo(bg, noFine)
            bgChans.add(bg)
        }
        val bgSmall = Mat(); Core.merge(bgChans, bgSmall)
        Imgproc.GaussianBlur(bgSmall, bgSmall, Size(9.0, 9.0), 0.0)
        val bgFull = Mat(); Imgproc.resize(bgSmall, bgFull, lab.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)

        // ---- Distance to the local table colour. ----
        val lab32 = Mat(); lab.convertTo(lab32, CvType.CV_32FC3)
        val diff = Mat(); Core.absdiff(lab32, bgFull, diff)
        val dch = ArrayList<Mat>(); Core.split(diff, dch)
        val dist32 = Mat()
        Core.addWeighted(dch[1], cfg.chromaWeight, dch[2], cfg.chromaWeight, 0.0, dist32)
        Core.addWeighted(dist32, 1.0, dch[0], cfg.lightnessWeight, 0.0, dist32)
        val dist = Mat(); dist32.convertTo(dist, CvType.CV_8UC1)
        if (valid != null) { val inv = Mat(); Core.bitwise_not(valid, inv); dist.setTo(Scalar(0.0), inv) }
        debug?.invoke("dist", dist)
        // Glare and uneven light raise the distance smoothly over large areas, while pieces stand out
        // sharply. Remove everything broader than a piece cluster (morphological opening = "what is
        // left when piece-sized bumps are erased") so one threshold works across the whole photo.
        run {
            val ds = Mat(); Imgproc.resize(dist, ds, small.size(), 0.0, 0.0, Imgproc.INTER_AREA)
            val k = ((max(sw, sh) * cfg.glareScale).toInt() or 1).toDouble()
            // Erode with the table-candidate minimum only: use a min filter, then smooth, then never exceed dist.
            val floor = Mat(); Imgproc.erode(ds, floor, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(k, k)))
            Imgproc.dilate(floor, floor, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(k, k)))
            Imgproc.GaussianBlur(floor, floor, Size(0.0, 0.0), k / 4)
            val floorFull = Mat(); Imgproc.resize(floor, floorFull, dist.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
            Core.subtract(dist, floorFull, dist)
        }
        // Hue evidence, fused with the distance: a strongly coloured pixel whose hue is clearly not the
        // wood's (blue sky, pink blossom, green) is piece even where the light makes the distance weak;
        // a saturated pixel in the wood's own hue band is more likely table (grain, lit wood).
        if (cfg.hueFusion) {
            val hsv = Mat(); Imgproc.cvtColor(blurred, hsv, Imgproc.COLOR_BGR2HSV)
            val ch = ArrayList<Mat>(); Core.split(hsv, ch)
            val sat = Mat(); Core.compare(ch[1], Scalar(120.0), sat, Core.CMP_GT)
            val hist = Mat()
            Imgproc.calcHist(listOf(ch[0]), org.opencv.core.MatOfInt(0), sat, hist, org.opencv.core.MatOfInt(180), org.opencv.core.MatOfFloat(0f, 180f))
            val hv = FloatArray(180); hist.get(0, 0, hv)
            var h0 = 0; var best = -1f
            for (i in 0 until 180) { var v = 0f; for (d in -2..2) v += hv[(i + d + 180) % 180]; if (v > best) { best = v; h0 = i } }
            val dh = Mat(); Core.absdiff(ch[0], Scalar(h0.toDouble()), dh)
            val wrap = Mat(); Core.absdiff(dh, Scalar(180.0), wrap)
            Core.min(dh, wrap, dh)
            val far = Mat(); Core.compare(dh, Scalar(cfg.hueFar), far, Core.CMP_GT)
            val coloured = Mat(); Core.compare(ch[1], Scalar(70.0), coloured, Core.CMP_GT)
            Core.bitwise_and(far, coloured, far)
            val near = Mat(); Core.compare(dh, Scalar(cfg.hueNear), near, Core.CMP_LE)
            val strong = Mat(); Core.compare(ch[1], Scalar(110.0), strong, Core.CMP_GT)
            Core.bitwise_and(near, strong, near)
            if (valid != null) { Core.bitwise_and(far, valid, far) }
            Core.add(dist, Scalar(cfg.hueBonus), dist, far)
            Core.subtract(dist, Scalar(cfg.huePenalty), dist, near)
            debug?.invoke("hue", far)
        }
        debug?.invoke("flat", dist)

        val tmp = Mat()
        val otsu = Imgproc.threshold(dist, tmp, 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)
        val t = otsu.coerceIn(cfg.minThreshold, cfg.maxThreshold)
        val mask = Mat(); Imgproc.threshold(dist, mask, t, 255.0, Imgproc.THRESH_BINARY)
        // Speckle (wood grain sparkling in glare) and hairline bridges go; kernel scales with the image.
        val ko = (max(3, max(w, h) / 700) or 1).toDouble()
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0)))
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(ko, ko)))
        debug?.invoke("mask", mask)

        val minArea = cfg.minAreaFraction * w * h
        // ---- Split touching pieces: cores (deep inside the mask) grown back with a watershed. ----
        val radius = bodyRadius(mask)
        val split = if (cfg.splitTouching && radius > 3) splitTouching(blurred, mask, radius, cfg) else mask
        debug?.invoke("split", split)
        val out = blobsOf(split, valid, minArea)
        val rawOut = if (split === mask) out else blobsOf(mask, valid, minArea)
        return Result(out, radius, rawOut, mask, minArea)
    }

    /** Connected regions of a piece mask, with whether each is cut off by the photo's (or the rectified area's) edge. */
    fun blobsOf(m: Mat, valid: Mat?, minArea: Double): List<Blob> {
    val w = m.cols(); val h = m.rows()
        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(m, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE)
        val list = ArrayList<Blob>()
        for (c in contours) {
            val area = Imgproc.contourArea(c)
            if (area < minArea) continue
            val r = Imgproc.boundingRect(c)
            var touches = r.x <= 1 || r.y <= 1 || r.x + r.width >= w - 2 || r.y + r.height >= h - 2
            if (!touches && valid != null) {
                // Touching the edge of the rectified area counts as touching the border.
                val grown = Mat(); Imgproc.erode(valid.submat(r), grown, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(7.0, 7.0)))
                val local = Mat.zeros(r.size(), CvType.CV_8UC1)
                Imgproc.drawContours(local, listOf(c), 0, Scalar(255.0), 1, Imgproc.LINE_8, Mat(), 0, org.opencv.core.Point(-r.x.toDouble(), -r.y.toDouble()))
                val inv = Mat(); Core.bitwise_not(grown, inv)
                Core.bitwise_and(local, inv, local)
                touches = Core.countNonZero(local) > 0
            }
            list.add(Blob(c, area, touches))
        }
        return list
    }

    /**
     * Inscribed radius of a typical piece body, from the mask alone: the distance transform's ridge
     * values. Knob heads and necks give low ridges, bodies the high ones, so a high percentile of the
     * local maxima is the body radius even when every piece touches another.
     */
    private fun bodyRadius(mask: Mat): Double {
        // Work on a copy about 1640 px on its long side: small enough to be fast, big enough for piece bodies to register.
        val f = max(1, Math.round(max(mask.cols(), mask.rows()) / 1640.0).toInt())
        val small = Mat(); Imgproc.resize(mask, small, Size(), 1.0 / f, 1.0 / f, Imgproc.INTER_NEAREST)
        // Fill pinholes (print colours close to the table) so they don't cut the bodies.
        Imgproc.morphologyEx(small, small, Imgproc.MORPH_CLOSE, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0)))
        val dt = Mat(); Imgproc.distanceTransform(small, dt, Imgproc.DIST_L2, 5)
        val dil = Mat(); Imgproc.dilate(dt, dil, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(7.0, 7.0)))
        val n = dt.rows() * dt.cols()
        val a = FloatArray(n); dt.get(0, 0, a)
        val b = FloatArray(n); dil.get(0, 0, b)
        val peaks = ArrayList<Float>()
        for (i in 0 until n) if (a[i] > 2f && a[i] >= b[i]) peaks.add(a[i])
        if (peaks.size < 5) return 0.0
        peaks.sort()
        return peaks[(peaks.size * 0.8).toInt()].toDouble() * f
    }

    /**
     * A piece's body is much wider than the contact between two touching pieces. Points deeper inside
     * the mask than half a typical piece's inscribed radius are "cores", one per piece (knob heads are
     * too thin to form one). A watershed on the image grows the cores back; where two pieces touch, the
     * fronts meet on the dark gap between them.
     */
    private fun splitTouching(bgr: Mat, mask: Mat, radius: Double, cfg: Config): Mat {
        // Pinholes inside pieces would break the cores apart.
        val solid = Mat()
        val kh = ((radius * 0.25).toInt().coerceAtLeast(3) or 1).toDouble()
        Imgproc.morphologyEx(mask, solid, Imgproc.MORPH_CLOSE, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(kh, kh)))
        val dt = Mat(); Imgproc.distanceTransform(solid, dt, Imgproc.DIST_L2, 5)
        val cores = Mat(); Imgproc.threshold(dt, cores, radius * cfg.coreDepth, 255.0, Imgproc.THRESH_BINARY)
        val cores8 = Mat(); cores.convertTo(cores8, CvType.CV_8UC1)
        val markers = Mat()
        val n = Imgproc.connectedComponents(cores8, markers, 8, CvType.CV_32S)
        if (n <= 2) return mask
        // Labels: cores 2.., sure table 1, unknown 0.
        Core.add(markers, Scalar(1.0), markers)
        val grow = (radius * 0.5).toInt().coerceAtLeast(3) or 1
        val near = Mat(); Imgproc.dilate(mask, near, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(grow.toDouble(), grow.toDouble())))
        val unknown = Mat(); Core.subtract(near, cores8, unknown)
        markers.setTo(Scalar(0.0), unknown)
        Imgproc.watershed(bgr, markers)
        val pieces = Mat(); Core.compare(markers, Scalar(1.5), pieces, Core.CMP_GT)
        // Final shapes come from the detailed mask (only pinholes closed), not the smoothed one.
        val fine = Mat(); Imgproc.morphologyEx(mask, fine, Imgproc.MORPH_CLOSE, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0)))
        Core.bitwise_and(pieces, fine, pieces)
        // Watershed lines are 1 px wide; make sure neighbours are not diagonally connected.
        Imgproc.erode(pieces, pieces, Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, Size(3.0, 3.0)))
        return pieces
    }
}
