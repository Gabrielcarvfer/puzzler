package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.util.Random
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/**
 * Box placement on synthetic data (no photos needed): a random "printed" box picture, and table photos
 * made by cutting pieces out of it, rotating, scaling and scattering them on a wood-coloured table.
 * The true box cell of every piece is known, so placements and pairs can be checked exactly.
 */
class BoxMapTest {
    companion object {
        const val COLS = 40; const val ROWS = 25; const val CELL = 60
        lateinit var poster: Mat
        lateinit var placer: BoxPlacer

        @BeforeClass @JvmStatic fun setUp() {
            OpenCV.loadLocally()
            poster = randomPrint(COLS * CELL, ROWS * CELL, 1)
            placer = BoxPlacer(poster)
        }

        /** Dense random shapes in many colours: plenty of distinct features everywhere, like a busy print. */
        fun randomPrint(w: Int, h: Int, seed: Long): Mat {
            val r = Random(seed)
            val m = Mat(h, w, CvType.CV_8UC3, Scalar(200.0, 190.0, 180.0))
            repeat(w * h / 120) {
                val c = Scalar(r.nextInt(256).toDouble(), r.nextInt(256).toDouble(), r.nextInt(256).toDouble())
                val p = Point(r.nextInt(w).toDouble(), r.nextInt(h).toDouble())
                when (r.nextInt(3)) {
                    0 -> Imgproc.circle(m, p, 2 + r.nextInt(9), c, -1)
                    1 -> Imgproc.line(m, p, Point(p.x + r.nextInt(31) - 15, p.y + r.nextInt(31) - 15), c, 1 + r.nextInt(3))
                    else -> Imgproc.rectangle(m, p, Point(p.x + 2 + r.nextInt(10), p.y + 2 + r.nextInt(10)), c, -1)
                }
            }
            return m
        }
    }

    /** A piece placed on the table: box cells [c0, r0] .. [c0 + wCells - 1, r0], all moved together. */
    class Placed(val col: Int, val row: Int, val wCells: Int, val angle: Double, val x: Double, val y: Double)

    private val scale = 4.0  // piece size on the table photo: 240 px

    /** Renders [pieces] on a table; returns the photo and each piece's outline (one outline per [Placed]). */
    private fun table(pieces: List<Placed>, w: Int = 2400, h: Int = 2400): Pair<Mat, List<Pair<DoubleArray, DoubleArray>>> {
        val photo = Mat(h, w, CvType.CV_8UC3, Scalar(40.0, 110.0, 170.0)) // warm wood (BGR)
        val outlines = ArrayList<Pair<DoubleArray, DoubleArray>>()
        for (p in pieces) {
            val a = Math.toRadians(p.angle)
            val cx = (p.col + p.wCells / 2.0) * CELL; val cy = (p.row + 0.5) * CELL
            val m = Mat(2, 3, CvType.CV_64F)
            m.put(0, 0, scale * cos(a), -scale * sin(a), p.x - scale * (cos(a) * cx - sin(a) * cy),
                scale * sin(a), scale * cos(a), p.y - scale * (sin(a) * cx + cos(a) * cy))
            val warped = Mat(); Imgproc.warpAffine(poster, warped, m, Size(w.toDouble(), h.toDouble()))
            val corners = listOf(Point(p.col * CELL.toDouble(), p.row * CELL.toDouble()), Point((p.col + p.wCells) * CELL.toDouble(), p.row * CELL.toDouble()),
                Point((p.col + p.wCells) * CELL.toDouble(), (p.row + 1) * CELL.toDouble()), Point(p.col * CELL.toDouble(), (p.row + 1) * CELL.toDouble()))
            val mapped = MatOfPoint2f(); Core.transform(MatOfPoint2f(*corners.toTypedArray()), mapped, m)
            val q = mapped.toArray()
            val mask = Mat.zeros(h, w, CvType.CV_8UC1)
            Imgproc.fillPoly(mask, listOf(MatOfPoint(*q.map { Point(it.x, it.y) }.toTypedArray())), Scalar(255.0))
            warped.copyTo(photo, mask)
            outlines.add(DoubleArray(4) { q[it].x } to DoubleArray(4) { q[it].y })
        }
        return photo to outlines
    }

    /** 14 single pieces scattered at random angles, none of them neighbours on the box. */
    private fun scattered(seed: Long): List<Placed> {
        val r = Random(seed)
        val cells = HashSet<Pair<Int, Int>>()
        val out = ArrayList<Placed>()
        var k = 0
        while (out.size < 14) {
            val c = 2 + r.nextInt(COLS - 4); val rr = 2 + r.nextInt(ROWS - 4)
            if ((-1..1).any { dc -> (-1..1).any { dr -> (c + dc) to (rr + dr) in cells } }) continue
            cells.add(c to rr)
            out.add(Placed(c, rr, 1, r.nextDouble() * 360, 220.0 + (k % 4) * 520, 220.0 + (k / 4) * 520)); k++
        }
        return out
    }

    private fun cellOf(sx: Double, sy: Double, g: BoxPlacer.Grid) = Math.floor((sx - g.offX) / g.cw).toInt() to Math.floor((sy - g.offY) / g.ch).toInt()

    @Test
    fun placesEveryPieceInItsOwnCell() {
        val pieces = scattered(11)
        val (photo, outlines) = table(pieces)
        val map = BoxMap(placer)
        map.observe("p1", photo, outlines, CELL * scale, cleanCount = outlines.size)
        val g = map.cleanGrid()
        val cells = map.cells(g)
        val truth = pieces.map { it.col to it.row }.toSet()
        assertTrue("no piece may land in a wrong cell: ${cells.keys - truth}", (cells.keys - truth).isEmpty())
        assertTrue("at least 12 of 14 pieces placed, got ${cells.size}", cells.size >= 12)
        // The fitted grid matches the true one (60 px cells, no offset).
        assertEquals(CELL.toDouble(), g.cw, 1.0); assertEquals(CELL.toDouble(), g.ch, 1.0)
        assertTrue(abs(g.offX) < 6 && abs(g.offY) < 6)
    }

    @Test
    fun neighboursOnTheBoxBecomePairsAndJoinedOnesAreRecognised() {
        // Background pieces so the grid gets fitted, plus two box neighbours (22, 11) and (23, 11):
        // in photo 1 they lie apart (a pair to suggest), in photo 2 they are joined (already done).
        val base = scattered(12).filter { it.col !in 17..25 || it.row !in 9..13 }
        val apart = listOf(Placed(22, 11, 1, 30.0, 300.0, 2150.0), Placed(23, 11, 1, 200.0, 2150.0, 2150.0))
        val (photo1, out1) = table(base + apart)
        val mapApart = BoxMap(placer)
        mapApart.observe("p1", photo1, out1, CELL * scale, cleanCount = out1.size)
        val g1 = mapApart.cleanGrid(); val c1 = mapApart.cells(g1)
        val a = c1[22 to 11]; val b = c1[23 to 11]
        assertTrue("both neighbours placed", a != null && b != null)
        assertFalse("pieces lying apart are not joined", mapApart.joined(a!!, b!!, g1) || mapApart.joined(b, a, g1))

        // Joined: one two-cell piece group, moved together. Its outline holds both pieces, so the
        // voting has to split it into two cells.
        val (photo2, out2) = table(base + listOf(Placed(22, 11, 2, 75.0, 1250.0, 2150.0)))
        val mapJoined = BoxMap(placer)
        mapJoined.observe("p2", photo2, out2, CELL * scale, cleanCount = out2.size - 1)
        val g2 = mapJoined.cleanGrid(); val c2 = mapJoined.cells(g2)
        val ja = c2[22 to 11]; val jb = c2[23 to 11]
        assertTrue("both halves of a joined pair are placed: ${c2.keys}", ja != null && jb != null)
        assertTrue("pieces lying joined as on the box are recognised as joined", mapJoined.joined(ja!!, jb!!, g2) || mapJoined.joined(jb, ja, g2))
    }

    @Test
    fun saveAndLoadKeepEverything() {
        val (photo, outlines) = table(scattered(13))
        val map = BoxMap(placer)
        map.observe("p1", photo, outlines, CELL * scale, cleanCount = outlines.size, tag = 90)
        val f = File.createTempFile("boxmap", ".bin"); f.deleteOnExit()
        map.save(f)
        val back = BoxMap(placer)
        assertTrue(back.load(f))
        assertEquals(map.all.size, back.all.size)
        for ((x, y) in map.all.zip(back.all)) {
            assertEquals(x.photo, y.photo); assertEquals(x.tag, y.tag); assertEquals(x.inliers, y.inliers); assertEquals(x.clean, y.clean)
            assertEquals(x.boxPts.size, y.boxPts.size)
            assertEquals(x.boxX, y.boxX, 1e-9)
        }
        val g = map.cleanGrid()
        assertEquals(map.cells(g).keys, back.cells(back.cleanGrid()).keys)
        assertFalse("a file of another kind is refused", BoxMap(placer).load(File.createTempFile("other", ".bin").also { it.writeText("hello"); it.deleteOnExit() }))
    }

    @Test
    fun parallelPlacementMatchesSequential() {
        val photos = (21L..26L).map { table(scattered(it)) }
        val seq = BoxMap(placer)
        for ((i, p) in photos.withIndex()) seq.observe("p$i", p.first, p.second, CELL * scale, cleanCount = p.second.size)
        val par = BoxMap(placer)
        val pool = java.util.concurrent.Executors.newFixedThreadPool(3)
        photos.mapIndexed { i, p -> pool.submit { par.addAll(par.locate("p$i", p.first, p.second, CELL * scale, cleanCount = p.second.size)) } }.forEach { it.get() }
        pool.shutdown()
        assertEquals(seq.all.size, par.all.size)
        assertEquals(seq.cells(seq.cleanGrid()).keys, par.cells(par.cleanGrid()).keys)
    }

    @Test
    fun findsTheBoxPictureCornersOnATiltedPhoto() {
        val pic = randomPrint(1600, 1000, 5)
        val photo = Mat(2400, 3200, CvType.CV_8UC3, Scalar(70.0, 70.0, 70.0))
        val truth = listOf(Point(520.0, 380.0), Point(2700.0, 300.0), Point(2850.0, 1950.0), Point(400.0, 2080.0))
        val h = Imgproc.getPerspectiveTransform(MatOfPoint2f(Point(0.0, 0.0), Point(1600.0, 0.0), Point(1600.0, 1000.0), Point(0.0, 1000.0)), MatOfPoint2f(*truth.toTypedArray()))
        val warped = Mat(); Imgproc.warpPerspective(pic, warped, h, photo.size())
        val mask = Mat.zeros(photo.size(), CvType.CV_8UC1)
        Imgproc.fillPoly(mask, listOf(MatOfPoint(*truth.toTypedArray())), Scalar(255.0))
        warped.copyTo(photo, mask)
        val found = BoxLocator.detectCorners(photo)
        assertTrue("corners found", found != null)
        for ((f, t) in found!!.zip(truth)) assertTrue("corner $f vs $t", hypot(f.x - t.x, f.y - t.y) < 0.015 * photo.cols())
    }
}
