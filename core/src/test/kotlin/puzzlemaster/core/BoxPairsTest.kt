package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Test
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.DMatch
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.features2d.FlannBasedMatcher
import org.opencv.features2d.SIFT
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * "The box proposes, the shape confirms": every catalogued piece is located on the box picture by
 * its printed picture; pieces that land in neighbouring cells are candidate pairs; the two sides
 * that would face each other must then be tab-against-blank and fit in shape.
 */
class BoxPairsTest {
    class Loc(val e: PieceCatalogue.Entry, val col: Int, val row: Int, val inliers: Int, /** entry side index facing N, E, S, W */ val facing: IntArray, val offCentre: Double)

    @Test
    fun boxProposesShapeConfirms() {
        val boxFile = File("src/test/resources/box/box.jpg").takeIf { it.exists() } ?: return
        val photos = File("../session/photos").listFiles { f -> f.name.endsWith(".jpg") }?.sorted() ?: return
        OpenCV.loadLocally()
        val cell = 64; val cols = 38; val rows = 26
        val poster = BoxLocator.fromPhoto(Imgcodecs.imread(boxFile.path), listOf(Point(254.0, 430.0), Point(7954.0, 410.0), Point(7980.0, 5948.0), Point(238.0, 5932.0)), cell = cell).poster
        val gray = Mat(); Imgproc.cvtColor(poster, gray, Imgproc.COLOR_BGR2GRAY)
        val pkM = MatOfKeyPoint(); val pd = Mat()
        SIFT.create(0, 3, 0.02, 10.0, 1.6).detectAndCompute(gray, Mat(), pkM, pd)
        val pk = pkM.toArray()
        val flann = FlannBasedMatcher.create(); flann.add(listOf(pd)); flann.train()

        val saved = File("../session/replay_catalogue.bin")
        val reuse = saved.exists() && System.getenv("PM_REPLAY") == null
        val session = ScanSession()
        val crops = HashMap<Int, Mat>()
        session.debugHook = { image, pass ->
            for (sg in pass.sightings) {
                val pc = sg.piece
                val r = (pc.scale * 0.95).toInt()
                val x0 = (pc.centroidX - r).toInt().coerceIn(0, image.cols() - 2); val y0 = (pc.centroidY - r).toInt().coerceIn(0, image.rows() - 2)
                val w = minOf(2 * r, image.cols() - x0); val h = minOf(2 * r, image.rows() - y0)
                if (w > 8 && h > 8) {
                    val c = Mat(); Imgproc.resize(image.submat(org.opencv.core.Rect(x0, y0, w, h)), c, org.opencv.core.Size(240.0, 240.0))
                    val k = 240.0 / (2 * r)
                    Imgproc.polylines(c, listOf(org.opencv.core.MatOfPoint(*Array(pc.xs.size) { Point((pc.xs[it] - x0) * k, (pc.ys[it] - y0) * k) })), true, org.opencv.core.Scalar(255.0, 230.0, 0.0), 1)
                    crops[sg.entry.id] = c
                }
            }
        }
        if (reuse) session.catalogue.load(saved)
        else for (f in photos) {
            val meta = File(f.path.removeSuffix(".jpg") + ".txt").readLines().associate { l -> l.substringBefore(" ") to l.substringAfter(" ") }
            session.process(f.readBytes(), meta.getValue("rotation").toInt(), meta.getValue("gravity").split(" ").map { it.toFloat() }.toFloatArray(), meta.getValue("focalPerLongSide").toDouble())
        }
        val cat = session.catalogue
        if (!reuse) cat.mergeDuplicates()
        println("BOXPAIRS ${photos.size} photos, catalogue ${cat.entries.size} pieces (measured 3+ times: ${cat.entries.count { it.sightings >= 3 }})")

        if (!reuse) cat.save(saved)
        // Piece proportions tell the cutting grid: square-ish = 38x26, oblong (1.12) = 40x25.
        val asp = cat.entries.filter { it.sightings >= 3 }.map { e -> val a = e.sides[0].length + e.sides[2].length; val b = e.sides[1].length + e.sides[3].length; maxOf(a, b) / minOf(a, b) }.sorted()
        if (asp.size >= 5) println("BOXPAIRS piece long/short side: median %.3f (25%%-75%%: %.3f-%.3f) over ${asp.size} pieces measured 3+ times; 38x26 grid would give 1.03, 40x25 would give 1.12".format(asp[asp.size / 2], asp[asp.size / 4], asp[asp.size * 3 / 4]))

        class Hit(val e: PieceCatalogue.Entry, val tx: Double, val ty: Double, val th: Double, val inliers: Int)
        val hits = ArrayList<Hit>()
        for (e in cat.entries) {
            if (e.descriptors.rows() < 6) continue
            val knn = ArrayList<MatOfDMatch>(); flann.knnMatch(e.descriptors, knn, 2)
            val good = ArrayList<DMatch>()
            for (k in knn) { val m = k.toArray(); if (m.size == 2 && m[0].distance < 0.8f * m[1].distance) good.add(m[0]) }
            if (good.size < 4) continue
            val from = MatOfPoint2f(*good.map { e.keypoints[it.queryIdx] }.toTypedArray())
            val to = MatOfPoint2f(*good.map { pk[it.trainIdx].pt }.toTypedArray())
            val inl = Mat()
            val a = Calib3d.estimateAffinePartial2D(from, to, inl, Calib3d.RANSAC, 4.0, 2000, 0.99, 10)
            if (a.empty()) continue
            val n = Core.countNonZero(inl)
            val sc = hypot(a.get(0, 0)[0], a.get(1, 0)[0])
            if (n < 5 || sc < cell * 0.75 || sc > cell * 1.35) continue
            hits.add(Hit(e, a.get(0, 2)[0], a.get(1, 2)[0], atan2(a.get(1, 0)[0], a.get(0, 0)[0]), n))
        }
        println("BOXPAIRS located on the box (5+ supporting matches): ${hits.size} of ${cat.entries.size}")
        val pw = poster.cols().toDouble(); val ph = poster.rows().toDouble()
        // The box print need not show exactly the puzzle's area, so the cutting grid may be shifted and
        // slightly scaled against it. Fit period and phase per axis so that piece centres fall on cell centres.
        fun fitAxis(v: List<Double>, nominal: Double): DoubleArray {
            var best = doubleArrayOf(nominal, 0.0); var bs = -1e9
            var p = nominal * 0.93
            while (p <= nominal * 1.07) {
                var ph0 = 0.0
                while (ph0 < 1.0) {
                    var sc = 0.0
                    for (x in v) sc += cos(2 * PI * (x / p - ph0 - 0.5))
                    if (sc > bs) { bs = sc; best = doubleArrayOf(p, ph0) }
                    ph0 += 0.02
                }
                p += nominal * 0.002
            }
            return doubleArrayOf(best[0], best[1], bs / v.size)
        }
        val fx = fitAxis(hits.map { it.tx }, pw / 40); val fy = fitAxis(hits.map { it.ty }, ph / 25)
        println("BOXPAIRS fitted grid: cell %.1f x %.1f px (nominal %.1f x %.1f), shift %.2f / %.2f cells, agreement %.2f / %.2f (1 = all centres exactly on cell centres, 0 = none)".format(fx[0], fy[0], pw / 40, ph / 25, fx[1], fy[1], fx[2], fy[2]))
        for ((gc, gr, fitted) in listOf(Triple(38, 26, false), Triple(40, 25, false), Triple(40, 25, true))) {
            val cw = if (fitted) fx[0] else pw / gc; val chh = if (fitted) fy[0] else ph / gr
            val offX = if (fitted) fx[1] * cw else 0.0; val offY = if (fitted) fy[1] * chh else 0.0
            val locs = ArrayList<Loc>()
            val dirs = arrayOf(doubleArrayOf(0.0, -1.0), doubleArrayOf(1.0, 0.0), doubleArrayOf(0.0, 1.0), doubleArrayOf(-1.0, 0.0))
            for (hgt in hits) {
                val col = Math.floor((hgt.tx - offX) / cw).toInt(); val row = Math.floor((hgt.ty - offY) / chh).toInt()
                if (col !in 0 until gc || row !in 0 until gr) continue
                val e = hgt.e; val th = hgt.th
                val facing = IntArray(4) { d -> (0 until 4).maxBy { sd -> val x = cos(th) * e.sideDirs[sd][0] - sin(th) * e.sideDirs[sd][1]; val y = sin(th) * e.sideDirs[sd][0] + cos(th) * e.sideDirs[sd][1]; x * dirs[d][0] + y * dirs[d][1] } }
                if (facing.toSet().size != 4) continue
                locs.add(Loc(e, col, row, hgt.inliers, facing, hypot((hgt.tx - offX) / cw - (col + 0.5), (hgt.ty - offY) / chh - (row + 0.5))))
            }
            var innerFlat = 0; var rimOk = 0; var rimBad = 0
            for (l in locs) {
                val out = booleanArrayOf(l.row == 0, l.col == gc - 1, l.row == gr - 1, l.col == 0)
                for (d in 0 until 4) { val flat = l.e.sides[l.facing[d]].type == EdgeType.FLAT; if (out[d]) { if (flat) rimOk++ else rimBad++ } else if (flat) innerFlat++ }
            }
            val sheetRows = ArrayList<Mat>()
            var cand = 0; var typeOk = 0; val fits = ArrayList<Double>(); var rejected = 0
            for (i in locs.indices) for (j in i + 1 until locs.size) {
                val a = locs[i]; val b = locs[j]
                val dc = b.col - a.col; val dr = b.row - a.row
                if (abs(dc) + abs(dr) != 1) continue
                val d = if (dr == -1) 0 else if (dc == 1) 1 else if (dr == 1) 2 else 3
                val sa = a.facing[d]; val sb = b.facing[(d + 2) % 4]
                val ta = a.e.sides[sa].type; val tb = b.e.sides[sb].type
                cand++
                val compat = (ta == EdgeType.TAB && tb == EdgeType.BLANK) || (ta == EdgeType.BLANK && tb == EdgeType.TAB)
                if (fitted) {
                    val ca = crops[a.e.id]; val cb = crops[b.e.id]
                    if (ca != null && cb != null) {
                        val row = Mat.zeros(240, 760, ca.type())
                        ca.copyTo(row.submat(org.opencv.core.Rect(0, 0, 240, 240))); cb.copyTo(row.submat(org.opencv.core.Rect(250, 0, 240, 240)))
                        // Box picture around the two cells (5 x 5 cells), cells outlined.
                        val c0 = minOf(a.col, b.col) - 2; val r0 = minOf(a.row, b.row) - 2
                        val x = (offX + c0 * cw).toInt().coerceIn(0, poster.cols() - (5 * cw).toInt() - 1); val y = (offY + r0 * chh).toInt().coerceIn(0, poster.rows() - (5 * chh).toInt() - 1)
                        val box = Mat(); Imgproc.resize(poster.submat(org.opencv.core.Rect(x, y, (5 * cw).toInt(), (5 * chh).toInt())), box, org.opencv.core.Size(240.0, 240.0))
                        val kx = 240.0 / (5 * cw); val ky = 240.0 / (5 * chh)
                        for ((l, colr) in listOf(a to org.opencv.core.Scalar(0.0, 255.0, 0.0), b to org.opencv.core.Scalar(255.0, 0.0, 255.0)))
                            Imgproc.rectangle(box, Point((offX + l.col * cw - x) * kx, (offY + l.row * chh - y) * ky), Point((offX + (l.col + 1) * cw - x) * kx, (offY + (l.row + 1) * chh - y) * ky), colr, 2)
                        box.copyTo(row.submat(org.opencv.core.Rect(510, 0, 240, 240)))
                        Imgproc.rectangle(row, Point(1.0, 1.0), Point(238.0, 238.0), org.opencv.core.Scalar(0.0, 255.0, 0.0), 2)
                        Imgproc.rectangle(row, Point(251.0, 1.0), Point(488.0, 238.0), org.opencv.core.Scalar(255.0, 0.0, 255.0), 2)
                        Imgproc.putText(row, "${sheetRows.size + 1}: r${a.row + 1} c${a.col + 1} + r${b.row + 1} c${b.col + 1}${if (compat) "" else "  (sides disagree)"}", Point(6.0, 20.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, org.opencv.core.Scalar(255.0, 255.0, 255.0), 2)
                        sheetRows.add(row)
                    }
                }
                if (!compat) continue
                typeOk++
                val g = cat.mateGap(a.e, sa, b.e, sb)
                if (g == null) rejected++ else fits.add(g)
            }
            if (fitted && sheetRows.isNotEmpty()) {
                val sheet = Mat.zeros(((sheetRows.size + 1) / 2) * 250, 2 * 770, sheetRows[0].type())
                sheetRows.forEachIndexed { k, r -> r.copyTo(sheet.submat(org.opencv.core.Rect((k % 2) * 770, (k / 2) * 250, 760, 240))) }
                figureDir()?.let { Imgcodecs.imwrite(File(it, "box_pairs.jpg").path, sheet) }
                println("BOXPAIRS sheet with ${sheetRows.size} pairs written")
            }
            val dup = locs.groupBy { it.row * gc + it.col }.filter { it.value.size > 1 }
            println("BOXPAIRS grid ${gc}x$gr${if (fitted) " fitted" else ""}: placed ${locs.size}; mean distance from cell centre %.2f cells; inward straight sides $innerFlat; rim sides straight $rimOk / not $rimBad; neighbour candidates $cand, tab-against-blank $typeOk; shape-rejected $rejected; cells claimed twice ${dup.size}".format(locs.map { it.offCentre }.average()) + "; shape fits (percent of edge length): " + fits.sorted().joinToString(" ") { "%.1f".format(100 * it) })
        }
    }

    @Test
    fun boxPlacerOnSavedCatalogue() {
        val boxFile = File("src/test/resources/box/box.jpg").takeIf { it.exists() } ?: return
        val saved = File("../session/replay_catalogue.bin").takeIf { it.exists() } ?: return
        OpenCV.loadLocally()
        val poster = BoxLocator.fromPhoto(Imgcodecs.imread(boxFile.path), listOf(Point(254.0, 430.0), Point(7954.0, 410.0), Point(7980.0, 5948.0), Point(238.0, 5932.0)), cell = 64).poster
        val t0 = System.currentTimeMillis()
        val placer = BoxPlacer(poster)
        val t1 = System.currentTimeMillis()
        val cat = PieceCatalogue(); cat.load(saved)
        val hits = cat.entries.mapNotNull { placer.locate(it) }
        val grid = placer.fitGrid(hits)
        val placed = placer.place(hits, grid)
        val pairs = placer.pairs(placed)
        println("PLACER poster ${poster.cols()}x${poster.rows()}, ${placer.posterFeatures} features, built in ${t1 - t0} ms; ${cat.entries.size} entries located in ${System.currentTimeMillis() - t1} ms")
        println("PLACER located ${hits.size}, placed ${placed.size} (one per cell), grid cell %.1f x %.1f offset %.1f / %.1f px fitted=${grid.fitted}; pairs ${pairs.size}, sides agree ${pairs.count { it.sidesAgree }}".format(grid.cw, grid.ch, grid.offX, grid.offY))
        for (p in pairs) println("PLACER   r${p.rowA + 1} c${p.colA + 1} + r${p.rowB + 1} c${p.colB + 1}  agree=${p.sidesAgree} support ${p.support}")
        org.junit.Assert.assertTrue(pairs.isNotEmpty())
    }

    /** Why do most catalogued pieces get no position, and what do looser settings cost in precision? */
    @Test
    fun coverage() {
        val boxFile = File("src/test/resources/box/box.jpg").takeIf { it.exists() } ?: return
        val saved = File("../session/replay_catalogue.bin").takeIf { it.exists() } ?: return
        OpenCV.loadLocally()
        val poster = BoxLocator.fromPhoto(Imgcodecs.imread(boxFile.path), listOf(Point(254.0, 430.0), Point(7954.0, 410.0), Point(7980.0, 5948.0), Point(238.0, 5932.0)), cell = 64).poster
        val placer = BoxPlacer(poster)
        val cat = PieceCatalogue(); cat.load(saved)
        val nf = cat.entries.map { it.descriptors.rows() }.sorted()
        println("COVER ${cat.entries.size} entries; picture features per piece: median ${nf[nf.size / 2]}, 25%-75%: ${nf[nf.size / 4]}-${nf[nf.size * 3 / 4]}; pieces with fewer than 20: ${nf.count { it < 20 }}")
        for ((minIn, ratio) in listOf(5 to 0.8f, 4 to 0.8f, 5 to 0.85f, 4 to 0.85f, 4 to 0.9f, 3 to 0.85f)) {
            val hits = cat.entries.mapNotNull { placer.locate(it, minIn, ratio) }
            val grid = placer.fitGrid(hits)
            val placed = placer.place(hits, grid)
            val pairs = placer.pairs(placed)
            // Precision proxies: how close centres fall to cell centres, and how often facing sides agree.
            val off = hits.map { h -> val fx = (h.tx - grid.offX) / grid.cw; val fy = (h.ty - grid.offY) / grid.ch; hypot(fx - Math.floor(fx) - 0.5, fy - Math.floor(fy) - 0.5) }
            println("COVER min matches $minIn, ratio $ratio: located ${hits.size}, one per cell ${placed.size}, mean distance from cell centre %.2f, pairs ${pairs.size}, sides agree ${pairs.count { it.sidesAgree }}".format(off.average()))
        }
    }
}
