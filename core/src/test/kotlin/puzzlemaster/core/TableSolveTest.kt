package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Test
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/**
 * Offline "table solve": overlapping high-resolution photos -> stitched map -> every free-lying piece
 * catalogued (the same piece in several photos is recognised by its position on the map and its
 * side shapes are averaged) -> pairs solved -> one figure showing where the pairs are.
 * Photos are read from src/test/resources/<PM_PHOTOS or "map">.
 */
class TableSolveTest {
    @Test
    fun solveTable() {
        val dir = figureDir() ?: return
        val photos = File("src/test/resources/" + (System.getenv("PM_PHOTOS") ?: "map")).listFiles { f -> f.name.endsWith(".jpg") }?.sorted() ?: return
        if (photos.size < 2) return
        OpenCV.loadLocally()
        val reg = MosaicTest.register(photos) // quarter-resolution registration + mosaic.jpg
        val scale = 0.8; val q = 0.25 / scale // analysis coords -> registration coords
        val pipe = LivePipeline(); val cat = PieceCatalogue()
        class Seen(val photo: Int, val x: Double, val y: Double, val dirs: Array<DoubleArray>, val outline: Array<Point>, val sideMid: Array<Point>)
        val seen = HashMap<Int, Seen>() // entry id -> best sighting (for hints and drawing), in analysis coords of its photo
        fun toRef(i: Int): Mat? = reg.toRef[i]
        for ((pi, f) in photos.withIndex()) {
            val hi = toRef(pi) ?: continue
            val full = Imgcodecs.imread(f.path)
            val img = Mat(); Imgproc.resize(full, img, Size(), scale, scale, Imgproc.INTER_AREA)
            full.release()
            val pieces = pipe.process(img).pieces
            // Tell the catalogue where already-known pieces should be in this photo.
            val hInv = hi.inv()
            for ((id, s) in seen) {
                val hs = toRef(s.photo) ?: continue
                val m = Mat(); Core.gemm(hInv, hs, 1.0, Mat(), 0.0, m) // photo s -> photo pi (registration coords)
                val src = MatOfPoint2f(Point(s.x * q, s.y * q), Point((s.x + 50) * q, s.y * q))
                val dst = MatOfPoint2f(); Core.perspectiveTransform(src, dst, m)
                val d = dst.toArray()
                val rot = atan2(d[1].y - d[0].y, d[1].x - d[0].x); val c = cos(rot); val sn = sin(rot)
                cat.hintPosition(cat.entries[id], d[0].x / q, d[0].y / q, Array(4) { k -> doubleArrayOf(c * s.dirs[k][0] - sn * s.dirs[k][1], sn * s.dirs[k][0] + c * s.dirs[k][1]) })
            }
            val sightings = cat.observe(img, pieces)
            for (s in sightings) {
                val p = s.piece
                if (s.isNew || seen[s.entry.id] == null) {
                    val dirs = Array(4) { DoubleArray(2) }; val mids = Array(4) { Point() }
                    for (k in 0 until 4) {
                        val e = p.edges[k]; val dx = (e.startX + e.endX) / 2 - p.centroidX; val dy = (e.startY + e.endY) / 2 - p.centroidY; val l = hypot(dx, dy)
                        dirs[s.map[k]] = doubleArrayOf(dx / l, dy / l); mids[s.map[k]] = Point(e.midX, e.midY)
                    }
                    seen[s.entry.id] = Seen(pi, p.centroidX, p.centroidY, dirs, Array(p.xs.size) { Point(p.xs[it], p.ys[it]) }, mids)
                }
            }
            println("TABLE ${f.name}: ${pieces.size} free pieces -> ${sightings.count { it.isNew }} new, ${sightings.count { !it.isNew }} seen before")
            img.release(); System.gc()
        }
        val pairs = cat.solve()
        println("TABLE catalogue ${cat.entries.size} pieces (seen 2+ times: ${cat.entries.count { it.sightings >= 2 }}, 3+: ${cat.entries.count { it.sightings >= 3 }}); pairs ${pairs.size}")
        for ((i, p) in pairs.withIndex()) println("TABLE pair ${i + 1}: #${p.a.id} (seen ${p.a.sightings}x) + #${p.b.id} (seen ${p.b.sightings}x): gap %.2f%%, runner-up %.2f%%".format(100 * p.gap, 100 * p.runnerUp))

        // Figure: mosaic with every catalogued piece outlined and the pairs numbered.
        val canvas = Imgcodecs.imread(File(dir, "mosaic.jpg").path)
        fun toCanvas(photo: Int, pts: Array<Point>): Array<Point> {
            val m = Mat(); Core.gemm(reg.canvasT, toRef(photo), 1.0, Mat(), 0.0, m)
            val dst = MatOfPoint2f(); Core.perspectiveTransform(MatOfPoint2f(*pts.map { Point(it.x * q, it.y * q) }.toTypedArray()), dst, m)
            return dst.toArray()
        }
        for ((id, s) in seen) {
            val n = cat.entries[id].sightings
            Imgproc.polylines(canvas, listOf(MatOfPoint(*toCanvas(s.photo, s.outline))), true, if (n >= 2) Scalar(255.0, 230.0, 0.0) else Scalar(200.0, 200.0, 200.0), if (n >= 2) 2 else 1)
        }
        val cols = listOf(Scalar(0.0, 255.0, 0.0), Scalar(0.0, 220.0, 255.0), Scalar(255.0, 0.0, 255.0), Scalar(0.0, 128.0, 255.0), Scalar(255.0, 255.0, 0.0))
        for ((i, p) in pairs.withIndex()) {
            val sa = seen[p.a.id] ?: continue; val sb = seen[p.b.id] ?: continue
            val c = cols[i % cols.size]
            val oa = toCanvas(sa.photo, sa.outline); val ob = toCanvas(sb.photo, sb.outline)
            Imgproc.polylines(canvas, listOf(MatOfPoint(*oa)), true, c, 5); Imgproc.polylines(canvas, listOf(MatOfPoint(*ob)), true, c, 5)
            val ma = toCanvas(sa.photo, arrayOf(sa.sideMid[p.sideA]))[0]; val mb = toCanvas(sb.photo, arrayOf(sb.sideMid[p.sideB]))[0]
            Imgproc.line(canvas, ma, mb, c, 4)
            for (m in listOf(ma, mb)) { Imgproc.circle(canvas, m, 22, Scalar(0.0, 0.0, 0.0), -1); Imgproc.putText(canvas, "${i + 1}", Point(m.x - 12, m.y + 11), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, c, 3) }
        }
        Imgcodecs.imwrite(File(dir, "table_pairs.jpg").path, canvas)
    }
}
