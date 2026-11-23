package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Assert.assertTrue
import org.junit.Test
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint
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
 * Real-photo test of [FeatureLocator]. Without labels, correctness is measured by consistency: the
 * same physical piece seen in two different photos of the table must be placed on the same cell.
 */
class FeatureLocatorTest {
    @Test
    fun samePieceSameCellAcrossPhotos() {
        val boxFile = File("src/test/resources/box/box.jpg").takeIf { it.exists() } ?: return
        val photos = File("src/test/resources/map").listFiles { f -> f.name.endsWith(".jpg") }?.sorted()?.let { listOf(it[0], it[1], it[4]) } ?: return
        OpenCV.loadLocally()
        val scale = (System.getenv("PM_SCALE") ?: "0.6").toDouble()
        val poster = BoxLocator.fromPhoto(Imgcodecs.imread(boxFile.path), listOf(Point(254.0, 430.0), Point(7954.0, 410.0), Point(7980.0, 5948.0), Point(238.0, 5932.0)), cell = (System.getenv("PM_CELL") ?: "128").toInt()).poster
        val t0 = System.currentTimeMillis()
        val loc = FeatureLocator(poster, 38, 26)
        println("FEAT poster ${poster.cols()}x${poster.rows()}, ${loc.posterFeatures} features, built in ${System.currentTimeMillis() - t0} ms")
        val results = ArrayList<FeatureLocator.Result>()
        val grays = ArrayList<Mat>()
        for (f in photos) {
            val full = Imgcodecs.imread(f.path)
            val img = Mat(); Imgproc.resize(full, img, Size(), scale, scale, Imgproc.INTER_AREA)
            val r = loc.locate(img, FeatureLocator.Config(minNcc = (System.getenv("PM_NCC") ?: "0.15").toDouble(), minInliers = (System.getenv("PM_INL") ?: "5").toInt()))
            results.add(r)
            println("FEAT ${f.name}: ${img.cols()}x${img.rows()}, ${r.features} features, ${r.matches} matches, ${r.pieces.size} pieces, ${r.pairs.size} loose pairs | detect ${r.detectMs} match ${r.matchMs} group ${r.groupMs} ms")
            val g = Mat(); Imgproc.resize(full, g, Size(), 0.25, 0.25, Imgproc.INTER_AREA); Imgproc.cvtColor(g, g, Imgproc.COLOR_BGR2GRAY); grays.add(g)
            figureDir()?.let { dir ->
                for (p in r.pieces) {
                    Imgproc.polylines(img, listOf(MatOfPoint(*p.quad)), true, Scalar(0.0, 255.0, 0.0), 3)
                    Imgproc.putText(img, "${p.row + 1}.${p.col + 1}", Point(p.cx - 30, p.cy), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255.0, 255.0, 255.0), 2)
                }
                for (pr in r.pairs) Imgproc.line(img, Point(pr.a.cx, pr.a.cy), Point(pr.b.cx, pr.b.cy), Scalar(255.0, 0.0, 255.0), 5)
                Imgcodecs.imwrite(File(dir, "featloc_${f.nameWithoutExtension}.jpg").path, img)
            }
            full.release(); img.release(); System.gc()
        }
        // Cross-photo consistency.
        val orb = ORB.create(8000); val bf = BFMatcher.create(Core.NORM_HAMMING, false)
        val kd = grays.map { g -> val k = MatOfKeyPoint(); val d = Mat(); orb.detectAndCompute(g, Mat(), k, d); k.toArray() to d }
        var tot = 0; var same = 0; var near = 0
        for (i in photos.indices) for (j in i + 1 until photos.size) {
            val knn = ArrayList<MatOfDMatch>(); bf.knnMatch(kd[i].second, kd[j].second, knn, 2)
            val good = knn.map { it.toArray() }.filter { it.size == 2 && it[0].distance < 0.75 * it[1].distance }.map { it[0] }
            if (good.size < 30) continue
            val k = scale / 0.25
            val h = Calib3d.findHomography(MatOfPoint2f(*good.map { val p = kd[i].first[it.queryIdx].pt; Point(p.x * k, p.y * k) }.toTypedArray()),
                MatOfPoint2f(*good.map { val p = kd[j].first[it.trainIdx].pt; Point(p.x * k, p.y * k) }.toTypedArray()), Calib3d.RANSAC, 8.0)
            if (h.empty()) continue
            val a = results[i].pieces; val b = results[j].pieces
            if (a.isEmpty() || b.isEmpty()) continue
            val proj = MatOfPoint2f(); Core.perspectiveTransform(MatOfPoint2f(*a.map { Point(it.cx, it.cy) }.toTypedArray()), proj, h)
            val tol = 0.45 * loc.cell * a.map { it.scale }.sorted()[a.size / 2]
            for ((pa, q) in a.zip(proj.toArray())) {
                val pb = b.minBy { kotlin.math.hypot(it.cx - q.x, it.cy - q.y) }
                if (kotlin.math.hypot(pb.cx - q.x, pb.cy - q.y) > tol) continue
                tot++
                val d = kotlin.math.abs(pa.col - pb.col) + kotlin.math.abs(pa.row - pb.row)
                if (d == 0) same++ else if (d == 1) near++
                if (d != 0) println("FEAT   mismatch: (${pa.row + 1},${pa.col + 1}) n=${pa.inliers} ncc=%.2f  vs  (${pb.row + 1},${pb.col + 1}) n=${pb.inliers} ncc=%.2f".format(pa.ncc, pb.ncc))
            }
        }
        println("FEAT consistency: $tot pieces seen in two photos; same cell $same (${percent(same.toDouble() / tot)}), neighbour cell $near, elsewhere ${tot - same - near}")
        assertTrue("located pieces must be consistent across photos", same >= 0.9 * tot)
    }

    /** Frames saved by the app (with gravity + focal length), processed exactly as on the phone. */
    @Test
    fun replayPhoneFrames() {
        val dir = figureDir() ?: return
        val boxFile = File("src/test/resources/box/box.jpg").takeIf { it.exists() } ?: return
        val files = File("src/test/resources/" + (System.getenv("PM_CAPDIR") ?: "livecap")).listFiles { f -> f.name.endsWith(".jpg") }?.sorted() ?: return
        OpenCV.loadLocally()
        val cellPx = (System.getenv("PM_CELL") ?: "64").toInt()
        val loc = FeatureLocator(BoxLocator.fromPhoto(Imgcodecs.imread(boxFile.path), listOf(Point(254.0, 430.0), Point(7954.0, 410.0), Point(7980.0, 5948.0), Point(238.0, 5932.0)), cell = cellPx).poster, 38, 26)
        val cfg = FeatureLocator.Config(maxFeatures = 30000, minNcc = (System.getenv("PM_NCC") ?: "0.15").toDouble(), minInliers = (System.getenv("PM_INL") ?: "5").toInt())
        for (f in files) {
            val meta = File(f.path.removeSuffix(".jpg") + ".txt").readLines().associate { l -> l.substringBefore(" ") to l.substringAfter(" ") }
            val g = meta.getValue("gravity").split(" ").map { it.toFloat() }.toFloatArray()
            val img = Imgcodecs.imread(f.path)
            val view = Rectifier.squareUp(img, g, meta.getValue("focal").toDouble(), maxTilt = 75.0, maxOut = 1.15)
            val r = loc.locate(view.image, cfg)
            val edge = r.pieces.count { it.row == 0 || it.col == 0 || it.row == 25 || it.col == 37 }
            println("PHONE ${f.name}: tilt %.0f, ${view.image.cols()}x${view.image.rows()}, ${r.features} features, ${r.matches} matches, ${r.pieces.size} pieces ($edge edge), ${r.pairs.size} pairs, piece size %.0f px | ${r.detectMs + r.matchMs + r.groupMs} ms".format(view.tiltDeg, if (r.pieces.isEmpty()) 0.0 else loc.cell * r.pieces.map { it.scale }.sorted()[r.pieces.size / 2]))
            val vis = view.image.clone()
            for (p in r.pieces) {
                val e = p.row == 0 || p.col == 0 || p.row == 25 || p.col == 37
                Imgproc.polylines(vis, listOf(MatOfPoint(*p.quad)), true, if (e) Scalar(0.0, 140.0, 255.0) else Scalar(255.0, 230.0, 0.0), 4)
                Imgproc.putText(vis, "${p.row + 1}.${p.col + 1}", Point(p.cx - 34, p.cy + 8), Imgproc.FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255.0, 255.0, 255.0), 2)
            }
            for (pr in r.pairs) Imgproc.line(vis, Point(pr.a.cx, pr.a.cy), Point(pr.b.cx, pr.b.cy), Scalar(255.0, 0.0, 255.0), 8)
            Imgcodecs.imwrite(File(dir, "phone_${f.nameWithoutExtension}.jpg").path, vis)
            // Evidence sheet: each located piece (warped upright, 3x3 cells around it) next to the box picture there.
            val k = 56.0 / loc.cell; val sz = 168
            val posterBig = BoxLocator.fromPhoto(Imgcodecs.imread(boxFile.path), listOf(Point(254.0, 430.0), Point(7954.0, 410.0), Point(7980.0, 5948.0), Point(238.0, 5932.0)), cell = 56).poster
            val padded = Mat(); Core.copyMakeBorder(posterBig, padded, 56, 56, 56, 56, Core.BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0))
            val tiles = r.pieces.sortedByDescending { it.inliers }.take(24)
            if (tiles.isNotEmpty()) {
                val sheet = Mat.zeros(((tiles.size + 3) / 4) * (sz + 6), 4 * (2 * sz + 14), view.image.type())
                for ((i, p) in tiles.withIndex()) {
                    val a = p.a; val det = a[0] * a[4] - a[1] * a[3]
                    val i0 = a[4] / det; val i1 = -a[1] / det; val i3 = -a[3] / det; val i4 = a[0] / det
                    val i2 = -(i0 * a[2] + i1 * a[5]); val i5 = -(i3 * a[2] + i4 * a[5])
                    val ox = (p.col - 1) * loc.cell; val oy = (p.row - 1) * loc.cell
                    val m = Mat(2, 3, org.opencv.core.CvType.CV_64F); m.put(0, 0, k * i0, k * i1, k * (i2 - ox), k * i3, k * i4, k * (i5 - oy))
                    val patch = Mat(); Imgproc.warpAffine(view.image, patch, m, Size(sz.toDouble(), sz.toDouble()))
                    val x0 = (i % 4) * (2 * sz + 14); val y0 = (i / 4) * (sz + 6)
                    patch.copyTo(sheet.submat(org.opencv.core.Rect(x0, y0, sz, sz)))
                    padded.submat(org.opencv.core.Rect(p.col * 56, p.row * 56, sz, sz)).copyTo(sheet.submat(org.opencv.core.Rect(x0 + sz + 4, y0, sz, sz)))
                    Imgproc.rectangle(sheet, Point(x0 + 56.0, y0 + 56.0), Point(x0 + 112.0, y0 + 112.0), Scalar(0.0, 255.0, 255.0), 1)
                    Imgproc.rectangle(sheet, Point(x0 + sz + 4 + 56.0, y0 + 56.0), Point(x0 + sz + 4 + 112.0, y0 + 112.0), Scalar(0.0, 255.0, 255.0), 1)
                    Imgproc.putText(sheet, "${i + 1}: ${p.row + 1}.${p.col + 1} n${p.inliers}", Point(x0 + 3.0, y0 + 14.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.45, Scalar(255.0, 255.0, 255.0), 1)
                }
                Imgcodecs.imwrite(File(dir, "evidence_${f.nameWithoutExtension}.jpg").path, sheet)
            }
        }
    }
}
