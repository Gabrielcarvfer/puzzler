package puzzlemaster.core

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import kotlin.math.exp

/**
 * Learned placement model (see ml/train_locator.py): straightened piece + outline mask ->
 * probability for every (cell, rotation) of this puzzle. Produces [BoxLocator.Placement]s so the
 * existing pair logic can use it. Input preparation mirrors the training script exactly.
 */
class NetLocator(onnx: ByteArray, val cols: Int, val rows: Int) {
    private val net: Net = Dnn.readNetFromONNX(org.opencv.core.MatOfByte(*onnx))

    /** [photoBgr]: the photo the piece was detected in (raw colours). */
    fun locate(photoBgr: Mat, piece: Piece): BoxLocator.Placement? {
        val h = Imgproc.getPerspectiveTransform(
            MatOfPoint2f(*piece.cornerIdx.map { Point(piece.xs[it], piece.ys[it]) }.toTypedArray()),
            MatOfPoint2f(Point(32.0, 32.0), Point(96.0, 32.0), Point(96.0, 96.0), Point(32.0, 96.0)),
        )
        val patch = Mat(); Imgproc.warpPerspective(photoBgr, patch, h, Size(128.0, 128.0))
        val wo = MatOfPoint2f(); Core.perspectiveTransform(MatOfPoint2f(*Array(piece.xs.size) { Point(piece.xs[it], piece.ys[it]) }), wo, h)
        val mask = Mat.zeros(Size(128.0, 128.0), CvType.CV_8UC1)
        Imgproc.fillPoly(mask, listOf(MatOfPoint(*wo.toArray())), Scalar(255.0))
        Imgproc.erode(mask, mask, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0)))
        val grey = Mat(128, 128, patch.type(), Scalar(128.0, 128.0, 128.0)); patch.copyTo(grey, mask)
        val small = Mat(); Imgproc.resize(grey, small, Size(64.0, 64.0), 0.0, 0.0, Imgproc.INTER_AREA)
        val msmall = Mat(); Imgproc.resize(mask, msmall, Size(64.0, 64.0), 0.0, 0.0, Imgproc.INTER_AREA)
        // NCHW float input: (B, G, R, mask), each /255 - 0.5.
        val input = FloatArray(4 * 64 * 64)
        val pb = ByteArray(64 * 64 * 3); small.get(0, 0, pb)
        val mb = ByteArray(64 * 64); msmall.get(0, 0, mb)
        for (i in 0 until 64 * 64) {
            for (c in 0 until 3) input[c * 4096 + i] = (pb[i * 3 + c].toInt() and 0xFF) / 255f - 0.5f
            input[3 * 4096 + i] = (mb[i].toInt() and 0xFF) / 255f - 0.5f
        }
        val blob = Mat(intArrayOf(1, 4, 64, 64), CvType.CV_32F); blob.put(intArrayOf(0, 0, 0, 0), input)
        net.setInput(blob)
        val out = net.forward()
        val n = cols * rows * 4
        val logits = FloatArray(n); out.reshape(1, 1).get(0, 0, logits)
        val mx = logits.max()
        var sum = 0.0
        val p = DoubleArray(n) { exp((logits[it] - mx).toDouble()).also { v -> sum += v } }
        // Marginalise rotation for the cell decision.
        val cellP = DoubleArray(cols * rows) { c -> (p[c * 4] + p[c * 4 + 1] + p[c * 4 + 2] + p[c * 4 + 3]) / sum }
        val order = cellP.indices.sortedByDescending { cellP[it] }
        val best = order[0]
        val second = order.firstOrNull { o -> kotlin.math.abs(o % cols - best % cols) + kotlin.math.abs(o / cols - best / cols) > 0 } ?: best
        val rot = (0 until 4).maxBy { p[best * 4 + it] }
        val alts = order.take(3).map { Triple(it % cols + 0.5, it / cols + 0.5, cellP[it]) }
        return BoxLocator.Placement(piece, best % cols + 0.5, best / cols + 0.5, rot, cellP[best], cellP[best] - cellP[second], alts)
    }
}
