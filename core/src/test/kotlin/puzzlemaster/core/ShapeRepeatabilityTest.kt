package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Test
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.hypot

/**
 * Can a side's shape identify it? The same physical pieces appear in several table photos. Each
 * side measured in photo A is compared with every same-type side in photo B: if shape is usable for
 * a catalogue, its own re-measurement must be the closest (rank 1), clearly ahead of other pieces.
 */
class ShapeRepeatabilityTest {
    /** Same edge, scaled so the photo's median piece is 100 px. */
    private fun norm(e: Edge, k: Double, id: Int) = Edge(id, e.index, e.type, e.length * k, DoubleArray(e.cx.size) { e.cx[it] * k }, DoubleArray(e.cy.size) { e.cy[it] * k },
        e.colors, e.startX, e.startY, e.endX, e.endY, e.midX, e.midY)

    /** The edge that would mate perfectly with [e] (to compare two same-type edges with the mate scorer). */
    private fun mateOf(e: Edge): Edge {
        val n = e.cx.size
        return Edge(-1 - e.pieceId, e.index, if (e.type == EdgeType.TAB) EdgeType.BLANK else EdgeType.TAB, e.length,
            DoubleArray(n) { e.length - e.cx[n - 1 - it] }, DoubleArray(n) { -e.cy[n - 1 - it] }, e.colors, e.startX, e.startY, e.endX, e.endY, e.midX, e.midY)
    }

    @Test
    fun sameSideMeasuredTwice() {
        val photos = File("src/test/resources/map").listFiles { f -> f.name.endsWith(".jpg") }?.sorted()?.let { listOf(it[0], it[1], it[4]) } ?: return
        OpenCV.loadLocally()
        val scale = (System.getenv("PM_SCALE") ?: "0.8").toDouble()
        val pipe = LivePipeline()
        val pieces = ArrayList<List<Piece>>(); val grays = ArrayList<Mat>()
        for (f in photos) {
            val full = Imgcodecs.imread(f.path)
            val img = Mat(); Imgproc.resize(full, img, Size(), scale, scale, Imgproc.INTER_AREA)
            val out = pipe.process(img)
            val med = if (out.pieces.isEmpty()) 0.0 else out.pieces.map { it.scale }.sorted()[out.pieces.size / 2]
            println("SHAPE ${f.name}: ${out.pieces.size} cleanly outlined pieces, median size %.0f px".format(med))
            pieces.add(out.pieces)
            val g = Mat(); Imgproc.resize(full, g, Size(), 0.25, 0.25, Imgproc.INTER_AREA); Imgproc.cvtColor(g, g, Imgproc.COLOR_BGR2GRAY); grays.add(g)
            full.release(); img.release(); System.gc()
        }
        val orb = ORB.create(8000); val bf = BFMatcher.create(Core.NORM_HAMMING, false)
        val kd = grays.map { g -> val k = MatOfKeyPoint(); val d = Mat(); orb.detectAndCompute(g, Mat(), k, d); k.toArray() to d }
        val scorer = MateScorer(MateScorer.Config(maxLengthDiff = 1.0, maxBulgeDiff = 1.0, maxPositionDiff = 1.0))
        val selfGaps = ArrayList<Double>(); val otherBest = ArrayList<Double>(); val ranks = ArrayList<Int>(); var pool = 0
        for (i in photos.indices) for (j in photos.indices) {
            if (i == j || pieces[i].isEmpty() || pieces[j].isEmpty()) continue
            val knn = ArrayList<MatOfDMatch>(); bf.knnMatch(kd[i].second, kd[j].second, knn, 2)
            val good = knn.map { it.toArray() }.filter { it.size == 2 && it[0].distance < 0.75 * it[1].distance }.map { it[0] }
            if (good.size < 30) continue
            val k = scale / 0.25
            val h = Calib3d.findHomography(MatOfPoint2f(*good.map { val p = kd[i].first[it.queryIdx].pt; Point(p.x * k, p.y * k) }.toTypedArray()),
                MatOfPoint2f(*good.map { val p = kd[j].first[it.trainIdx].pt; Point(p.x * k, p.y * k) }.toTypedArray()), Calib3d.RANSAC, 8.0)
            if (h.empty()) continue
            val ka = 100.0 / pieces[i].map { it.scale }.sorted()[pieces[i].size / 2]
            val kb = 100.0 / pieces[j].map { it.scale }.sorted()[pieces[j].size / 2]
            val edgesB = pieces[j].flatMap { p -> p.edges.filter { it.type != EdgeType.FLAT }.map { norm(it, kb, p.id) } }
            for (pa in pieces[i]) {
                val q = MatOfPoint2f(); Core.perspectiveTransform(MatOfPoint2f(Point(pa.centroidX, pa.centroidY)), q, h)
                val c = q.toArray()[0]
                val pb = pieces[j].minBy { hypot(it.centroidX - c.x, it.centroidY - c.y) }
                if (hypot(pb.centroidX - c.x, pb.centroidY - c.y) > 0.25 * pb.scale) continue
                for (ea in pa.edges) {
                    if (ea.type == EdgeType.FLAT) continue
                    // The same side in photo B: nearest edge tip after mapping.
                    val mq = MatOfPoint2f(); Core.perspectiveTransform(MatOfPoint2f(Point(ea.midX, ea.midY)), mq, h)
                    val m = mq.toArray()[0]
                    val eb = pb.edges.minBy { hypot(it.midX - m.x, it.midY - m.y) }
                    if (eb.type != ea.type || hypot(eb.midX - m.x, eb.midY - m.y) > 0.25 * pb.scale) continue
                    val probe = mateOf(norm(ea, ka, 100000 + pa.id))
                    val self = scorer.fit(probe, norm(eb, kb, pb.id))?.gap ?: continue
                    val others = edgesB.filter { it.type == ea.type && !(it.pieceId == pb.id && it.index == eb.index) }.mapNotNull { scorer.fit(probe, it)?.gap }
                    if (others.isEmpty()) continue
                    selfGaps.add(self); otherBest.add(others.min()); ranks.add(1 + others.count { it < self }); pool += others.size
                }
            }
        }
        if (selfGaps.isEmpty()) { println("SHAPE no pieces seen cleanly in two photos"); return }
        fun pct(a: List<Double>, p: Double) = a.sorted()[((a.size - 1) * p).toInt()]
        println("SHAPE ${selfGaps.size} sides measured in two photos, each compared with on average ${pool / selfGaps.size} other same-type sides")
        println("SHAPE same side twice:      median gap %.2f%% of edge length (25%%-75%%: %.2f-%.2f)".format(100 * pct(selfGaps, 0.5), 100 * pct(selfGaps, 0.25), 100 * pct(selfGaps, 0.75)))
        println("SHAPE closest OTHER side:   median gap %.2f%% (25%%-75%%: %.2f-%.2f)".format(100 * pct(otherBest, 0.5), 100 * pct(otherBest, 0.25), 100 * pct(otherBest, 0.75)))
        println("SHAPE own re-measurement ranked 1st: ${ranks.count { it == 1 }}/${ranks.size}, in top 3: ${ranks.count { it <= 3 }}/${ranks.size}, median rank ${ranks.sorted()[ranks.size / 2]}")
    }
}
