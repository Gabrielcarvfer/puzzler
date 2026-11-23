package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Test
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.imgcodecs.Imgcodecs
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

/** Every piece-sized outline that was rejected in one session photo, with the corner candidates considered. */
class CornerDebugTest {
    @Test
    fun rejectedPieces() {
        val dir = figureDir() ?: return
        val photos = File("../session/photos").listFiles { f -> f.name.endsWith(".jpg") }?.sorted() ?: return
        OpenCV.loadLocally()
        val tiles = ArrayList<BufferedImage>()
        val reasons = HashMap<String, Int>()
        var accepted = 0
        for (idx in (System.getenv("PM_PHOTOS_IDX") ?: "12,30,60").split(",").map { it.trim().toInt() }) {
            val f = photos.getOrNull(idx) ?: continue
            val meta = File(f.path.removeSuffix(".jpg") + ".txt").readLines().associate { l -> l.substringBefore(" ") to l.substringAfter(" ") }
            val g = meta.getValue("gravity").split(" ").map { it.toFloat() }.toFloatArray()
            val raw = Imgcodecs.imdecode(MatOfByte(*f.readBytes()), Imgcodecs.IMREAD_COLOR or Imgcodecs.IMREAD_IGNORE_ORIENTATION)
            val up = Mat()
            when (meta.getValue("rotation").toInt()) { 90 -> Core.rotate(raw, up, Core.ROTATE_90_CLOCKWISE); 180 -> Core.rotate(raw, up, Core.ROTATE_180); 270 -> Core.rotate(raw, up, Core.ROTATE_90_COUNTERCLOCKWISE); else -> raw.copyTo(up) }
            val view = Rectifier.squareUp(up, g, meta.getValue("focalPerLongSide").toDouble() * maxOf(up.cols(), up.rows()), maxTilt = 70.0, maxOut = 1.0)
            val factor = maxOf(1, Math.round(maxOf(view.image.cols(), view.image.rows()) / 2700.0).toInt())
            val pipe = LivePipeline()
            val det = PieceDetector(PieceDetector.Config(edgeSamples = 96))
            pipe.debugReject = { img, xs, ys, why ->
                reasons.merge(why.take(2), 1, Int::plus)
                val p = det.analyzeContour(xs, ys, img, 0)
                val (cx, cy) = det.debugContour ?: (xs to ys)
                val s = 420.0 / maxOf(img.width, img.height)
                val bi = BufferedImage(420, 420, BufferedImage.TYPE_INT_RGB)
                val src = BufferedImage(img.width, img.height, BufferedImage.TYPE_INT_RGB); src.setRGB(0, 0, img.width, img.height, img.pixels, 0, img.width)
                val gr = bi.createGraphics(); gr.drawImage(src, 0, 0, (img.width * s).toInt(), (img.height * s).toInt(), null)
                gr.color = Color.MAGENTA
                for (i in cx.indices) gr.fillRect((cx[i] * s).toInt() - 1, (cy[i] * s).toInt() - 1, 3, 3)
                gr.color = Color.BLACK
                for (i in xs.indices) gr.fillRect((xs[i] * s).toInt(), (ys[i] * s).toInt(), 1, 1)
                gr.color = Color.YELLOW
                if (p == null || true) for (i in det.debugCandidates) if (i < cx.size) gr.drawOval((cx[i] * s).toInt() - 5, (cy[i] * s).toInt() - 5, 10, 10)
                gr.color = Color.RED
                p?.cornerIdx?.forEach { i -> gr.fillOval((p.xs[i] * s).toInt() - 4, (p.ys[i] * s).toInt() - 4, 8, 8) }
                gr.color = Color.GREEN; gr.drawString("$why  cand ${det.debugCandidates.size}", 4, 12)
                tiles.add(bi)
            }
            val out = pipe.processHiRes(view.image, view.valid, LivePipeline.Lean(view.nadirX, view.nadirY, view.focal), factor)
            accepted += out.pieces.size
        }
        println("CORNERS accepted $accepted, rejected ${tiles.size}: $reasons")
        saveMontage(tiles.take(12), 4, File(dir, "corner_rejects.png"))
    }
}
