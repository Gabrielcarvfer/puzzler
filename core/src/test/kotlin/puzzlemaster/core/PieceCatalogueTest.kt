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
 * Catalogue test on the six table photos: every photo is one "sighting" of the pieces in it.
 * Ground truth for "is this the same piece?" comes from registering the photos to each other.
 */
class PieceCatalogueTest {
    @Test
    fun recognisesPiecesAcrossPhotos() {
        val photos = File("src/test/resources/map").listFiles { f -> f.name.endsWith(".jpg") }?.sorted() ?: return
        OpenCV.loadLocally()
        val scale = 0.8
        val pipe = LivePipeline(); val cat = PieceCatalogue()
        val orb = ORB.create(8000); val bf = BFMatcher.create(Core.NORM_HAMMING, false)
        val kd = ArrayList<Pair<Array<org.opencv.core.KeyPoint>, Mat>>()
        val origin = HashMap<Int, Triple<Int, Double, Double>>() // entry -> (photo, x, y) of first sighting
        fun homography(i: Int, j: Int): Mat? {
            val knn = ArrayList<MatOfDMatch>(); bf.knnMatch(kd[i].second, kd[j].second, knn, 2)
            val good = knn.map { it.toArray() }.filter { it.size == 2 && it[0].distance < 0.75 * it[1].distance }.map { it[0] }
            if (good.size < 30) return null
            val k = scale / 0.25
            val h = Calib3d.findHomography(MatOfPoint2f(*good.map { val p = kd[i].first[it.queryIdx].pt; Point(p.x * k, p.y * k) }.toTypedArray()),
                MatOfPoint2f(*good.map { val p = kd[j].first[it.trainIdx].pt; Point(p.x * k, p.y * k) }.toTypedArray()), Calib3d.RANSAC, 8.0)
            return h.takeIf { !it.empty() }
        }
        var right = 0; var wrong = 0; var missed = 0; var unverifiable = 0
        for ((pi, f) in photos.withIndex()) {
            val full = Imgcodecs.imread(f.path)
            val img = Mat(); Imgproc.resize(full, img, Size(), scale, scale, Imgproc.INTER_AREA)
            val g = Mat(); Imgproc.resize(full, g, Size(), 0.25, 0.25, Imgproc.INTER_AREA); Imgproc.cvtColor(g, g, Imgproc.COLOR_BGR2GRAY)
            val k = MatOfKeyPoint(); val d = Mat(); orb.detectAndCompute(g, Mat(), k, d); kd.add(k.toArray() to d)
            val pieces = pipe.process(img).pieces
            val t0 = System.currentTimeMillis()
            val sightings = cat.observe(img, pieces)
            val ms = System.currentTimeMillis() - t0
            for (s in sightings) {
                // Where would an already-catalogued piece at this spot have come from?
                fun sameSpot(e: Int): Boolean? {
                    val (op, ox, oy) = origin[e] ?: return null
                    if (op == pi) return false
                    val h = homography(op, pi) ?: return null
                    val q = MatOfPoint2f(); Core.perspectiveTransform(MatOfPoint2f(Point(ox, oy)), q, h)
                    return hypot(q.toArray()[0].x - s.piece.centroidX, q.toArray()[0].y - s.piece.centroidY) < 0.3 * s.piece.scale
                }
                if (s.isNew) {
                    if (origin.keys.any { sameSpot(it) == true }) missed++
                    origin[s.entry.id] = Triple(pi, s.piece.centroidX, s.piece.centroidY)
                } else when (sameSpot(s.entry.id)) { true -> right++; false -> wrong++; null -> unverifiable++ }
            }
            println("CAT ${f.name}: ${pieces.size} clean pieces -> ${sightings.count { it.isNew }} new, ${sightings.count { !it.isNew }} recognised ($ms ms)")
            full.release(); img.release(); System.gc()
        }
        println("CAT catalogue: ${cat.entries.size} pieces; seen once ${cat.entries.count { it.sightings == 1 }}, twice ${cat.entries.count { it.sightings == 2 }}, 3+ times ${cat.entries.count { it.sightings >= 3 }}")
        println("CAT recognition: correct $right, wrong $wrong, could not check $unverifiable; seen before but not recognised $missed")
        // Duplicate merging: every merge must join two entries that really are the same spot on the table.
        val before = cat.entries.size
        val merges = cat.mergeDuplicates()
        var mOk = 0; var mBad = 0; var mUnknown = 0
        for ((keep, gone) in merges) {
            val a = origin[keep]; val b = origin[gone]
            if (a == null || b == null) { mUnknown++; continue }
            val h = if (a.first == b.first) null else homography(a.first, b.first)
            if (a.first == b.first) { mBad++; continue } // two different pieces in the same photo
            if (h == null) { mUnknown++; continue }
            val q = MatOfPoint2f(); Core.perspectiveTransform(MatOfPoint2f(Point(a.second, a.third)), q, h)
            if (hypot(q.toArray()[0].x - b.second, q.toArray()[0].y - b.third) < 60) mOk++ else mBad++
        }
        println("CAT merging: $before -> ${cat.entries.size} entries; merges correct $mOk, wrong $mBad, could not check $mUnknown (duplicates known to exist: $missed)")
        val pairs = cat.solve()
        println("CAT pairs on averaged shapes: ${pairs.size}: " + pairs.take(12).joinToString { "#${it.a.id}(x${it.a.sightings})-#${it.b.id}(x${it.b.sightings}) %.2f%% vs %.2f%%".format(100 * it.gap, 100 * it.runnerUp) })
    }
}
