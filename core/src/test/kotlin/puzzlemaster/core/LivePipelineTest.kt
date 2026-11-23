package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Test
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File

class LivePipelineTest {
    companion object {
        fun draw(img: Mat, out: LivePipeline.Out): Mat {
            val vis = img.clone()
            val th = maxOf(2, img.cols() / 600)
            for (p in out.pieces) {
                val poly = MatOfPoint(*Array(p.xs.size) { Point(p.xs[it], p.ys[it]) })
                val flat = out.border[p.id]
                val col = when { flat == null -> Scalar(255.0, 230.0, 0.0); flat.size >= 2 -> Scalar(0.0, 0.0, 255.0); else -> Scalar(0.0, 140.0, 255.0) }
                Imgproc.polylines(vis, listOf(poly), true, col, if (flat == null) th else th * 2)
                flat?.forEach { e -> Imgproc.line(vis, Point(e.startX, e.startY), Point(e.endX, e.endY), Scalar(255.0, 0.0, 255.0), th * 2) }
                for (k in p.cornerIdx) Imgproc.circle(vis, Point(p.xs[k], p.ys[k]), th * 2, Scalar(0.0, 0.0, 255.0), -1)
                Imgproc.putText(vis, "${p.id}", Point(p.centroidX - 10, p.centroidY + 8), Imgproc.FONT_HERSHEY_SIMPLEX, th * 0.4, Scalar(255.0, 255.0, 255.0), th)
            }
            for ((xs, ys) in out.loose) Imgproc.polylines(vis, listOf(MatOfPoint(*Array(xs.size) { Point(xs[it], ys[it]) })), true, Scalar(255.0, 255.0, 255.0), 1)
            if (System.getenv("PM_REASONS") != null) for ((x, y, why) in out.rejects) Imgproc.putText(vis, why, Point(x - 30, y), Imgproc.FONT_HERSHEY_SIMPLEX, th * 0.35, Scalar(0.0, 0.0, 255.0), th)
            for (m in out.pairs) {
                Imgproc.line(vis, Point(m.a.midX, m.a.midY), Point(m.b.midX, m.b.midY), Scalar(0.0, 255.0, 0.0), th * 2)
                Imgproc.circle(vis, Point(m.a.midX, m.a.midY), th * 4, Scalar(0.0, 255.0, 0.0), -1)
                Imgproc.circle(vis, Point(m.b.midX, m.b.midY), th * 4, Scalar(0.0, 255.0, 0.0), -1)
            }
            return vis
        }
    }

    @Test
    fun realPhotos() {
        val dir = figureDir() ?: return
        OpenCV.loadLocally()
        val names = (System.getenv("PM_LIVE") ?: "real/frame_20261122_150344.png,real/frame_20261122_150029.png,real/hd_20261122_152527.png,border/border1.jpg").split(",")
        val pipe = LivePipeline()
        val scorer = MateScorer()
        for (name in names) {
            val f = File("src/test/resources/$name").takeIf { it.exists() } ?: continue
            val img = Imgcodecs.imread(f.path)
            TableSegmenter.segment(img, debug = { n, m -> val sm = Mat(); Imgproc.resize(m, sm, org.opencv.core.Size(), 1000.0 / m.cols(), 1000.0 / m.cols()); Imgcodecs.imwrite(File(dir, "live_${f.nameWithoutExtension}_$n.jpg").path, sm) })
            val out = pipe.process(img)
            println("LIVE ${f.name} ${img.cols()}x${img.rows()}: blobs ${out.blobs}, pieces ${out.pieces.size}, border ${out.border.size} (corners ${out.border.values.count { it.size >= 2 }}), pairs ${out.pairs.size} | seg ${out.segMs} ms, pieces ${out.pieceMs} ms, match ${out.matchMs} ms")
            for (m in out.pairs) println("   pair #${m.a.pieceId}.${m.a.index} - #${m.b.pieceId}.${m.b.index} gap %.4f runner-up %.4f picture %.3f".format(m.gap, m.runnerUp, m.picture))
            // Distribution of all fits (for calibration).
            val edges = out.pieces.flatMap { it.edges }
            val fits = ArrayList<Triple<Edge, Edge, MateScorer.Mate>>()
            for (i in edges.indices) for (j in i + 1 until edges.size) scorer.fit(edges[i], edges[j])?.let { fits.add(Triple(edges[i], edges[j], it)) }
            fits.sortBy { it.third.gap }
            println("   ${fits.size} candidate fits; best gaps: " + fits.take(12).joinToString { "#${it.first.pieceId}.${it.first.index}-#${it.second.pieceId}.${it.second.index} %.3f".format(it.third.gap) })
            Imgcodecs.imwrite(File(dir, "live_${f.nameWithoutExtension}.jpg").path, draw(img, out))
        }
    }

    /** Replays frames saved by the app (frame + gravity + focal), exactly as the phone processes them. */
    @Test
    fun replayPhoneFrames() {
        val dir = figureDir() ?: return
        val files = File("src/test/resources/" + (System.getenv("PM_CAPDIR") ?: "livecap")).listFiles { f -> f.name.endsWith(".jpg") }?.sorted() ?: return
        OpenCV.loadLocally()
        val only = System.getenv("PM_CAP")
        for (f in files) {
            if (only != null && f.nameWithoutExtension != only) continue
            val meta = File(f.path.removeSuffix(".jpg") + ".txt").readLines().associate { l -> l.substringBefore(" ") to l.substringAfter(" ") }
            val g = meta.getValue("gravity").split(" ").map { it.toFloat() }.toFloatArray()
            val img = Imgcodecs.imread(f.path)
            val view = Rectifier.squareUp(img, g, meta.getValue("focal").toDouble(), maxTilt = 70.0, maxOut = 1.15)
            TableSegmenter.segment(view.image, view.valid, debug = { n, m -> Imgcodecs.imwrite(File(dir, "replay_${f.nameWithoutExtension}_$n.png").path, m) })
            val t0 = System.currentTimeMillis()
            val out = LivePipeline().process(view.image, view.valid, LivePipeline.Lean(view.nadirX, view.nadirY, view.focal))
            val size = if (out.pieces.isEmpty()) 0.0 else out.pieces.map { it.scale }.sorted()[out.pieces.size / 2]
            println("REPLAY ${f.name}: tilt %.0f, blobs ${out.blobs}, pieces ${out.pieces.size} (size %.0f px), border ${out.border.size} (corners ${out.border.values.count { it.size >= 2 }}), pairs ${out.pairs.size}, ${System.currentTimeMillis() - t0} ms; rejects ${out.rejects.groupingBy { it.third.take(2) }.eachCount()}, body radius %.0f".format(view.tiltDeg, size, out.bodyRadius))
            for (m in out.pairs) println("   pair #${m.a.pieceId}-#${m.b.pieceId} gap %.4f runner-up %.4f".format(m.gap, m.runnerUp))
            Imgcodecs.imwrite(File(dir, "replay_${f.nameWithoutExtension}.jpg").path, draw(view.image, out))
        }
    }
}

class HiResPipelineTest {
    /** The coarse-to-fine path must find about as many pieces as the direct full-resolution path. */
    @Test
    fun hiResMatchesDirect() {
        val f = File("src/test/resources/map").listFiles { x -> x.name.endsWith(".jpg") }?.sorted()?.getOrNull(4) ?: return
        OpenCV.loadLocally()
        val full = Imgcodecs.imread(f.path)
        val pipe = LivePipeline()
        val t0 = System.currentTimeMillis()
        val hi = pipe.processHiRes(full, null, null, 3)
        val t1 = System.currentTimeMillis()
        val size = if (hi.pieces.isEmpty()) 0.0 else hi.pieces.map { it.scale }.sorted()[hi.pieces.size / 2]
        println("HIRES rejects ${hi.rejects.groupingBy { it.third.take(2) }.eachCount()} blobs ${hi.blobs} bodyRadius ${hi.bodyRadius}")
        println("HIRES ${full.cols()}x${full.rows()} factor 3: ${hi.pieces.size} pieces (size %.0f px), ${hi.loose.size} unanalysed outlines, ${t1 - t0} ms (segment ${hi.segMs}, pieces ${hi.pieceMs})".format(size))
        figureDir()?.let { Imgcodecs.imwrite(File(it, "hires.jpg").path, LivePipelineTest.draw(full, hi)) }
    }
}
