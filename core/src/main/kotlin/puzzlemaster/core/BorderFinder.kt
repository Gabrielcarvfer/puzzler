package puzzlemaster.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.RotatedRect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Finds edge and corner pieces.
 *
 * 1. Pieces are boxed by a DETR detector fine-tuned on jigsaw pieces
 *    (pablo-moreira/puzzle-piece-detection, CC-BY-4.0), run in overlapping tiles.
 * 2. Inside each box the piece outline is cut out with GrabCut.
 * 3. Flat-side rule: take the tightest rotated rectangle around the outline. A flat side lies on its
 *    rectangle edge along (almost) its whole length; a tab only touches at the knob tip and a blank
 *    leaves a gap (the keyhole) in the middle. 1 flat side = edge piece, 2 adjacent = corner.
 */
class BorderFinder(detrOnnx: ByteArray, private val env: OrtEnvironment = OrtEnvironment.getEnvironment(), options: OrtSession.SessionOptions = OrtSession.SessionOptions()) : AutoCloseable {
    data class Config(
        val minScore: Float = 0.6f,
        /** Fraction of a rectangle edge the outline must cover to count as a flat side. */
        val flatCoverage: Double = 0.85,
        /** Distance from the rectangle edge (fraction of the rectangle's short side) still counted as "on" it. */
        val flatTolerance: Double = 0.05,
        /** Tiles across the long side of the image (more = smaller pieces still detected). */
        val tilesAcross: Int = 3,
    )

    class Found(
        val box: Rect,
        val score: Float,
        /** Outline in image coordinates (null if it couldn't be cut out). */
        val outline: Array<Point>?,
        val rect: RotatedRect?,
        /** Rectangle corners (4 points) and which of the 4 edges (corner i -> i+1) are flat. */
        val rectPts: Array<Point>?,
        val flat: BooleanArray,
        /** Straight sides with table outside them (image coordinates). */
        val flatSegments: List<Pair<Point, Point>> = emptyList(),
    ) {
        val flatCount get() = flatSegments.size
        /** Two flat sides meeting at roughly a right angle. */
        val isCorner get() = flatSegments.size >= 2 && flatSegments.indices.any { i -> flatSegments.indices.any { j ->
            i < j && run {
                val (a, b) = flatSegments[i]; val (c, d) = flatSegments[j]
                val cos = abs(((b.x - a.x) * (d.x - c.x) + (b.y - a.y) * (d.y - c.y)) / (hypot(b.x - a.x, b.y - a.y) * hypot(d.x - c.x, d.y - c.y)))
                cos < 0.3
            } } }
        val isEdge get() = flatCount >= 1
    }

    private val session = env.createSession(detrOnnx, options)
    private val n = 640
    /** Debug: write per-piece crops and masks here. */
    var debugDir: java.io.File? = null
    /** Flat sides found by the last [cutOut] call (image coordinates). */
    private var lastFlat: List<Pair<Point, Point>> = emptyList()

    /** Detector boxes over the whole image (overlapping tiles, merged with NMS). */
    fun detect(bgr: Mat, cfg: Config = Config()): List<Pair<Rect, Float>> {
        val w = bgr.cols(); val h = bgr.rows()
        // ~3 tiles across the long side keeps each tile well under DETR's 100 detections.
        val tile = min(min(w, h), max(w, h) / cfg.tilesAcross)
        val step = (tile * 0.8).toInt().coerceAtLeast(1)
        val raw = ArrayList<Pair<Rect, Float>>()
        var y = 0
        while (true) {
            var x = 0
            val ty = min(y, h - tile)
            while (true) {
                val tx = min(x, w - tile)
                raw.addAll(runTile(bgr, Rect(tx, ty, tile, tile), cfg))
                if (tx + tile >= w) break
                x += step
            }
            if (ty + tile >= h) break
            y += step
        }
        // NMS
        raw.sortByDescending { it.second }
        val out = ArrayList<Pair<Rect, Float>>()
        for (r in raw) if (out.none { iou(it.first, r.first) > 0.4 }) out.add(r)
        return out
    }

    private fun runTile(bgr: Mat, roi: Rect, cfg: Config): List<Pair<Rect, Float>> {
        val small = Mat(); Imgproc.resize(bgr.submat(roi), small, Size(n.toDouble(), n.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val b = ByteArray(n * n * 3); small.get(0, 0, b)
        val data = FloatArray(3 * n * n)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f); val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        for (i in 0 until n * n) for (c in 0 until 3) {
            val v = (b[i * 3 + (2 - c)].toInt() and 0xFF) / 255f // BGR -> RGB
            data[c * n * n + i] = (v - mean[c]) / std[c]
        }
        val out = ArrayList<Pair<Rect, Float>>()
        OnnxTensor.createTensor(env, FloatBuffer.wrap(data), longArrayOf(1, 3, n.toLong(), n.toLong())).use { t ->
            session.run(mapOf("pixels" to t)).use { res ->
                val scores = (res[0] as OnnxTensor).floatBuffer
                val boxes = (res[1] as OnnxTensor).floatBuffer
                for (q in 0 until scores.remaining()) {
                    val s = scores.get(q)
                    if (s < cfg.minScore) continue
                    val cx = boxes.get(q * 4); val cy = boxes.get(q * 4 + 1); val bw = boxes.get(q * 4 + 2); val bh = boxes.get(q * 4 + 3)
                    val x0 = roi.x + (cx - bw / 2) * roi.width; val y0 = roi.y + (cy - bh / 2) * roi.height
                    // Drop boxes cut by the tile border (the overlapping tile sees them whole) unless at the image border.
                    val cut = (x0 < roi.x + 2 && roi.x > 0) || (y0 < roi.y + 2 && roi.y > 0) ||
                        (x0 + bw * roi.width > roi.x + roi.width - 2 && roi.x + roi.width < bgr.cols()) ||
                        (y0 + bh * roi.height > roi.y + roi.height - 2 && roi.y + roi.height < bgr.rows())
                    if (cut) continue
                    out.add(Rect(x0.toInt(), y0.toInt(), (bw * roi.width).toInt(), (bh * roi.height).toInt()) to s)
                }
            }
        }
        return out
    }

    /** Full pipeline: boxes -> outlines -> flat sides. */
    fun find(bgr: Mat, cfg: Config = Config()): List<Found> {
        val boxes = detect(bgr, cfg)
        if (boxes.isEmpty()) return emptyList()
        val medSide = boxes.map { min(it.first.width, it.first.height).toDouble() }.sorted()[boxes.size / 2]
        @Suppress("UNUSED_VARIABLE") val unused = medSide
        return boxes.map { (box, s) ->
            lastFlat = emptyList()
            val outline = cutOut(bgr, box)
            Found(box, s, outline, null, null, BooleanArray(4), lastFlat)
        }
    }

    /** For each rectangle edge: fraction of its length covered by outline points lying on it. */
    private fun flatSides(outline: Array<Point>, rp: Array<Point>, shortSide: Double, cfg: Config): BooleanArray {
        val tol = cfg.flatTolerance * shortSide
        val bins = 40
        return BooleanArray(4) { e ->
            val a = rp[e]; val b = rp[(e + 1) % 4]
            val len = hypot(b.x - a.x, b.y - a.y)
            val ux = (b.x - a.x) / len; val uy = (b.y - a.y) / len
            val covered = BooleanArray(bins)
            for (p in outline) {
                val dx = p.x - a.x; val dy = p.y - a.y
                val along = dx * ux + dy * uy
                val across = abs(-dx * uy + dy * ux)
                if (across <= tol && along >= 0 && along <= len) covered[min(bins - 1, (along / len * bins).toInt())] = true
            }
            // Ignore the rounded corners (first/last bin).
            (1 until bins - 1).count { covered[it] }.toDouble() / (bins - 2) >= cfg.flatCoverage
        }
    }

    /**
     * Outline inside the (enlarged) box: the box's outer ring is table, so pixels far (in Lab colour)
     * from the ring's colours are piece; Otsu picks the threshold; keep the blob at the box centre.
     */
    private fun cutOut(bgr: Mat, box: Rect): Array<Point>? {
        val padX = (box.width * 0.4).toInt(); val padY = (box.height * 0.4).toInt()
        val x0 = max(0, box.x - padX); val y0 = max(0, box.y - padY)
        val x1 = min(bgr.cols(), box.x + box.width + padX); val y1 = min(bgr.rows(), box.y + box.height + padY)
        if (x1 - x0 < 8 || y1 - y0 < 8) return null
        val crop = bgr.submat(Rect(x0, y0, x1 - x0, y1 - y0))
        val f = min(1.0, 200.0 / max(crop.cols(), crop.rows()))
        val small = Mat(); Imgproc.resize(crop, small, Size(), f, f, Imgproc.INTER_AREA)
        Imgproc.GaussianBlur(small, small, Size(3.0, 3.0), 0.0)
        val lab = Mat(); Imgproc.cvtColor(small, lab, Imgproc.COLOR_BGR2Lab)
        val w = lab.cols(); val h = lab.rows()
        val px = ByteArray(w * h * 3); lab.get(0, 0, px)
        fun at(i: Int, c: Int) = (px[i * 3 + c].toInt() and 0xFF).toDouble()
        // Table colours from the outer ring (k-means-free: keep a few ring samples as exemplars).
        val ring = ArrayList<DoubleArray>()
        val rw = max(2, (min(w, h) * 0.06).toInt())
        for (y in 0 until h) for (x in 0 until w) {
            if (x >= rw && y >= rw && x < w - rw && y < h - rw) continue
            if ((x + y) % 3 != 0) continue
            val i = y * w + x; ring.add(doubleArrayOf(at(i, 0), at(i, 1), at(i, 2)))
        }
        // Robust table model: median of the ring (pieces poking into the ring are a minority).
        val med = DoubleArray(3) { c -> ring.map { it[c] }.sorted()[ring.size / 2] }
        val dist = Mat(h, w, CvType.CV_8UC1)
        val db = ByteArray(w * h)
        for (i in 0 until w * h) {
            // Lightness differences (shadows) count half as much as colour differences.
            val d = 0.5 * abs(at(i, 0) - med[0]) + 1.5 * (abs(at(i, 1) - med[1]) + abs(at(i, 2) - med[2]))
            db[i] = min(255.0, d).toInt().toByte()
        }
        dist.put(0, 0, db)
        val bin = Mat(); val otsuT = Imgproc.threshold(dist, bin, 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU).toInt()
        Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_CLOSE, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0)))
        Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0)))
        // Fill holes (print colours close to the table).
        run {
            val ff = bin.clone(); val m2 = Mat.zeros(h + 2, w + 2, CvType.CV_8UC1)
            Imgproc.floodFill(ff, m2, Point(0.0, 0.0), Scalar(255.0))
            val inv = Mat(); Core.bitwise_not(ff, inv); Core.bitwise_or(bin, inv, bin)
        }
        debugDir?.let { org.opencv.imgcodecs.Imgcodecs.imwrite(java.io.File(it, "mask_${box.x}_${box.y}.png").path, bin); org.opencv.imgcodecs.Imgcodecs.imwrite(java.io.File(it, "crop_${box.x}_${box.y}.png").path, small) }
        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(bin, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE)
        val centre = Point(((box.x + box.width / 2.0) - x0) * f, ((box.y + box.height / 2.0) - y0) * f)
        val c = contours.filter { Imgproc.pointPolygonTest(MatOfPoint2f(*it.toArray()), centre, false) >= 0 }.maxByOrNull { Imgproc.contourArea(it) }
            ?: contours.maxByOrNull { Imgproc.contourArea(it) } ?: return null
        // A blob reaching the crop border runs into neighbouring pieces: its straight cut there would
        // look like a flat side, so give no verdict.
        val pts = c.toArray()
        val area = Imgproc.contourArea(c)
        val boxArea = box.width * f * box.height * f
        if (area < 0.4 * boxArea) return null
        // Straight stretches about one side long, not on the crop border, with table just outside.
        val side = min(box.width, box.height) * f / 1.25
        val approx = MatOfPoint2f(); Imgproc.approxPolyDP(MatOfPoint2f(*pts), approx, max(1.5, side * 0.035), true)
        val poly = approx.toArray()
        val segs = ArrayList<Pair<Point, Point>>()
        val cx = pts.map { it.x }.average(); val cy = pts.map { it.y }.average()
        for (k in poly.indices) {
            val a = poly[k]; val b = poly[(k + 1) % poly.size]
            val len = hypot(b.x - a.x, b.y - a.y)
            if (len < 0.6 * side || len > 1.25 * side) continue
            fun nearBorder(p: Point) = p.x < 3 || p.y < 3 || p.x > w - 4 || p.y > h - 4
            if (nearBorder(a) && nearBorder(b)) continue
            // outward normal (away from blob centre)
            var nx = -(b.y - a.y) / len; var ny = (b.x - a.x) / len
            val mx = (a.x + b.x) / 2; val my = (a.y + b.y) / 2
            if ((mx - cx) * nx + (my - cy) * ny < 0) { nx = -nx; ny = -ny }
            // table check at 3 points along the side, a few px outside
            var table = 0
            for (t in listOf(0.25, 0.5, 0.75)) {
                val sx = (a.x + (b.x - a.x) * t + nx * max(3.0, side * 0.06)).toInt()
                val sy = (a.y + (b.y - a.y) * t + ny * max(3.0, side * 0.06)).toInt()
                if (sx in 0 until w && sy in 0 until h && (db[sy * w + sx].toInt() and 0xFF) < otsuT) table++
            }
            // Bounding-box test for this side: no part of the piece may stick out beyond its line.
            val outward = pts.maxOf { (it.x - a.x) * nx + (it.y - a.y) * ny }
            if (outward > side * 0.13) continue
            if (table >= 3) segs.add(Point(x0 + a.x / f, y0 + a.y / f) to Point(x0 + b.x / f, y0 + b.y / f))
        }
        lastFlat = segs
        return pts.map { Point(x0 + it.x / f, y0 + it.y / f) }.toTypedArray()
    }

    private fun iou(a: Rect, b: Rect): Double {
        val ix = max(0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
        val iy = max(0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
        val i = ix.toDouble() * iy
        return i / (a.width.toDouble() * a.height + b.width.toDouble() * b.height - i)
    }

    override fun close() = session.close()

    @Suppress("unused") private val keep = CvType.CV_8UC1
}
