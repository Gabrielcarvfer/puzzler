package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Test
import org.opencv.core.Point
import org.opencv.imgcodecs.Imgcodecs
import java.io.File

/**
 * Detection benchmark on the phone's own photos, scored against the box picture.
 * A catalogue piece located on the box is a correctly detected, correctly measured piece; the number of
 * distinct box cells reached is the main score. Entries landing in an already-taken cell are duplicates
 * (or wrong placements). Run with PM_BENCH=<variant>[,<variant>...]; PM_BENCH_STEP picks every n-th photo.
 */
class BenchTest {
    private fun variant(name: String): Pair<ScanSession.Config, LivePipeline.Config> {
        val s = ScanSession.Config(); val p = LivePipeline.Config(); val seg = TableSegmenter.Config()
        return when (name) {
            "base" -> s to p
            "csmooth2" -> s to p.copy(cornerSmooth = 2)
            "csmooth4" -> s to p.copy(cornerSmooth = 4)
            "csmooth8" -> s to p.copy(cornerSmooth = 8)
            "minpx120" -> s.copy(minPiecePx = 120.0) to p
            "minpx180" -> s.copy(minPiecePx = 180.0) to p
            "search3200" -> s.copy(searchSize = 3200.0) to p
            "search2200" -> s.copy(searchSize = 2200.0) to p
            "split1.3" -> s to p.copy(splitAbove = 1.3)
            "split1.6" -> s to p.copy(splitAbove = 1.6)
            "neckwide" -> s to p.copy(neckRounds = listOf(0.06, 0.11, 0.16, 0.21, 0.26), neckWidth = 0.28)
            "necknarrow" -> s to p.copy(neckRounds = listOf(0.06, 0.11, 0.16), neckWidth = 0.18)
            "quality0.15" -> s to p.copy(minQuality = 0.15)
            "quality0.35" -> s to p.copy(minQuality = 0.35)
            "thr14" -> s to p.copy(segmentation = seg.copy(minThreshold = 14.0))
            "thr22" -> s to p.copy(segmentation = seg.copy(minThreshold = 22.0))
            "glare0.15" -> s to p.copy(segmentation = seg.copy(glareScale = 0.15))
            "glare0.3" -> s to p.copy(segmentation = seg.copy(glareScale = 0.3))
            "refine0" -> s to p.copy(refineRange = 0.0)
            "nohires" -> s to p.copy(hiResRefine = false)
            else -> error("unknown variant $name")
        }
    }

    @Test
    fun bench() {
        val names = System.getenv("PM_BENCH")?.split(",") ?: return
        val step = System.getenv("PM_BENCH_STEP")?.toInt() ?: 2
        OpenCV.loadLocally()
        val onPhone = File("../session/export/on_phone.txt").readLines().map { it.trim() }.toSet()
        val photos = File("../session/photos").listFiles { f -> f.name in onPhone }!!.sorted().filterIndexed { i, _ -> i % step == 0 }
        val poster = BoxLocator.fromPhoto(Imgcodecs.imread("src/test/resources/box/box.jpg"), listOf(Point(254.0, 430.0), Point(7954.0, 410.0), Point(7980.0, 5948.0), Point(238.0, 5932.0)), cell = 64).poster
        val placer = BoxPlacer(poster)
        val out = File("../test/bench.tsv")
        if (!out.exists()) out.writeText("variant\tphotos\tpieces/photo\tmeasurable/photo\tcatalogue\tlocated\tcells\tduplicates\tpairs\tagree\tcentre\tms/photo\n")
        for (name in names) {
            val (sc, pc) = variant(name)
            val session = ScanSession(PieceCatalogue(), sc, pc)
            val map = BoxMap(placer)
            val crops = HashMap<BoxMap.Sighting, org.opencv.core.Mat>()
            var current = ""
            session.debugHook = { img, pass ->
                val outl = pass.out.pieces.map { it.xs to it.ys } + pass.out.loose
                val before = map.all.size
                map.observe(current, img, outl, pass.pieceSize, pass.out.pieces.size)
                if (System.getenv("PM_SHEET") != null) for (sg in map.all.subList(before, map.all.size)) {
                    val r = org.opencv.imgproc.Imgproc.boundingRect(org.opencv.core.MatOfPoint(*Array(sg.xs.size) { org.opencv.core.Point(sg.xs[it], sg.ys[it]) }))
                    val rx = r.x.coerceAtLeast(0); val ry = r.y.coerceAtLeast(0); val rr = org.opencv.core.Rect(rx, ry, minOf(r.x + r.width, img.cols()) - rx, minOf(r.y + r.height, img.rows()) - ry)
                    val c = img.submat(rr).clone()
                    org.opencv.imgproc.Imgproc.polylines(c, listOf(org.opencv.core.MatOfPoint(*Array(sg.xs.size) { org.opencv.core.Point(sg.xs[it] - rr.x, sg.ys[it] - rr.y) })), true, org.opencv.core.Scalar(0.0, 255.0, 0.0), 3)
                    crops[sg] = c
                }
            }
            val why = HashMap<String, Int>(); var blobs = 0
            var pieces = 0; var meas = 0; val t0 = System.currentTimeMillis()
            for ((i, f) in photos.withIndex()) {
                val meta = File(f.path.removeSuffix(".jpg") + ".txt").readLines().associate { l -> l.substringBefore(" ") to l.substringAfter(" ") }
                current = f.nameWithoutExtension
                val pass = session.process(f.readBytes(), meta.getValue("rotation").toInt(), meta.getValue("gravity").split(" ").map { it.toFloat() }.toFloatArray(), meta.getValue("focalPerLongSide").toDouble())
                pieces += pass.out.pieces.size; meas += pass.measurable; blobs += pass.out.blobs
                for (r in pass.out.rejects) why.merge(r.third.substringBefore(' ').trimEnd('0','1','2','3','4','5','6','7','8','9','.'), 1, Int::plus)
                if ((i + 1) % 10 == 0) session.catalogue.mergeDuplicates()
            }
            val ms = (System.currentTimeMillis() - t0) / photos.size
            val cat = session.catalogue; cat.mergeDuplicates()
            val hits = cat.entries.mapNotNull { placer.locate(it) }
            val grid = placer.fitGrid(hits)
            val cells = hits.map { Math.floor((it.tx - grid.offX) / grid.cw).toInt() to Math.floor((it.ty - grid.offY) / grid.ch).toInt() }
            val placed = placer.place(hits, grid); val pairs = placer.pairs(placed)
            val off = hits.map { h -> val fx = (h.tx - grid.offX) / grid.cw; val fy = (h.ty - grid.offY) / grid.ch; kotlin.math.hypot(fx - Math.floor(fx) - 0.5, fy - Math.floor(fy) - 0.5) }
            val line = listOf(name, photos.size, "%.1f".format(pieces.toDouble() / photos.size), "%.1f".format(meas.toDouble() / photos.size), cat.entries.size, hits.size,
                cells.toSet().size, hits.size - cells.toSet().size, pairs.size, pairs.count { it.sidesAgree }, "%.3f".format(off.average()), ms).joinToString("\t")
            println("BENCH $line")
            println("BENCH   blobs/photo %.1f; rejected per photo: ".format(blobs.toDouble() / photos.size) + why.entries.sortedByDescending { it.value }.joinToString { "${it.key} %.1f".format(it.value.toDouble() / photos.size) })
            out.appendText(line + "\n")
            val mg = map.grid()
            val mp = map.pairs(mg)
            val strong = map.spots.filter { it.support >= 6 }
            val moff = strong.map { sp -> val fx = (sp.boxX - mg.offX) / mg.cw; val fy = (sp.boxY - mg.offY) / mg.ch; kotlin.math.hypot(fx - Math.floor(fx) - 0.5, fy - Math.floor(fy) - 0.5) }
            fun cen(l: List<BoxMap.Spot>) = l.map { sp -> val fx = (sp.boxX - mg.offX) / mg.cw; val fy = (sp.boxY - mg.offY) / mg.ch; kotlin.math.hypot(fx - Math.floor(fx) - 0.5, fy - Math.floor(fy) - 0.5) }.let { if (it.isEmpty()) 0.0 else it.average() }
            println("BENCH   boxmap centre: clean-only spots %.3f (${strong.count { sp -> sp.sightings.all { it.clean } }}), loose-only %.3f (${strong.count { sp -> sp.sightings.none { it.clean } }}), mixed %.3f".format(cen(strong.filter { sp -> sp.sightings.all { it.clean } }), cen(strong.filter { sp -> sp.sightings.none { it.clean } }), cen(strong.filter { sp -> sp.sightings.any { it.clean } && sp.sightings.any { !it.clean } })))
            val cg = map.cleanGrid()
            for ((mv, ms) in listOf(3 to 0.25, 4 to 0.3, 5 to 0.4)) {
                val cells = map.cells(cg, mv, ms)
                val strongCells = cells.values.filter { it.votes >= 6 }
                val keys = strongCells.map { it.col to it.row }.toSet()
                var np = 0; for (k in keys) { if ((k.first + 1) to k.second in keys) np++; if (k.first to (k.second + 1) in keys) np++ }
                // Purity: share of a sighting's points in its best cell (1 = clearly one piece).
                println("BENCH   cells(votes>=$mv share>=$ms): cells ${cells.size}, with 6+ votes ${strongCells.size}, seen in 2+ photos ${strongCells.count { c -> c.seen.map { it.first.photo }.toSet().size >= 2 }}, neighbour pairs $np; grid %.1f x %.1f off %.1f/%.1f".format(cg.cw, cg.ch, cg.offX, cg.offY))
            }
            if (System.getenv("PM_SHEET") != null) {
                val cells = map.cells(cg).values.filter { it.votes >= 6 }.sortedBy { c -> c.seen.any { it.first.clean } }
                val tiles = ArrayList<org.opencv.core.Mat>()
                for (c in cells.take(48)) {
                    val sg = c.best
                    val crop = crops[sg] ?: continue
                    val t = org.opencv.core.Mat(200, 400, org.opencv.core.CvType.CV_8UC3, org.opencv.core.Scalar(0.0, 0.0, 0.0))
                    val k = 200.0 / maxOf(crop.cols(), crop.rows())
                    val a1 = org.opencv.core.Mat(); org.opencv.imgproc.Imgproc.resize(crop, a1, org.opencv.core.Size(crop.cols() * k, crop.rows() * k))
                    a1.copyTo(t.submat(org.opencv.core.Rect(0, 0, a1.cols(), a1.rows())))
                    // Box: the cell with a margin of 0.4 cell, cell outlined.
                    val x0 = (cg.offX + (c.col - 0.4) * cg.cw).toInt().coerceIn(0, poster.cols() - 1); val y0 = (cg.offY + (c.row - 0.4) * cg.ch).toInt().coerceIn(0, poster.rows() - 1)
                    val w = minOf((1.8 * cg.cw).toInt(), poster.cols() - x0); val h = minOf((1.8 * cg.ch).toInt(), poster.rows() - y0)
                    val b = org.opencv.core.Mat(); org.opencv.imgproc.Imgproc.resize(poster.submat(org.opencv.core.Rect(x0, y0, w, h)), b, org.opencv.core.Size(190.0, 190.0 * h / w))
                    val bh = minOf(b.rows(), 200)
                    b.submat(0, bh, 0, 190).copyTo(t.submat(org.opencv.core.Rect(205, 0, 190, bh)))
                    val kx = 190.0 / w; val ky = 190.0 / w
                    org.opencv.imgproc.Imgproc.rectangle(t, org.opencv.core.Point(205 + (cg.offX + c.col * cg.cw - x0) * kx, (cg.offY + c.row * cg.ch - y0) * ky), org.opencv.core.Point(205 + (cg.offX + (c.col + 1) * cg.cw - x0) * kx, (cg.offY + (c.row + 1) * cg.ch - y0) * ky), org.opencv.core.Scalar(255.0, 0.0, 255.0), 2)
                    org.opencv.imgproc.Imgproc.putText(t, "${tiles.size + 1}${if (c.seen.any { it.first.clean }) " clean" else ""}", org.opencv.core.Point(4.0, 18.0), 0, 0.6, org.opencv.core.Scalar(0.0, 255.0, 255.0), 2)
                    tiles.add(t)
                }
                val rows = (tiles.size + 3) / 4
                val sheet = org.opencv.core.Mat(rows * 205, 4 * 405, org.opencv.core.CvType.CV_8UC3, org.opencv.core.Scalar(40.0, 40.0, 40.0))
                for ((i, t) in tiles.withIndex()) t.copyTo(sheet.submat(org.opencv.core.Rect((i % 4) * 405, (i / 4) * 205, 400, 200)))
                Imgcodecs.imwrite("../test/boxmap_sheet.jpg", sheet)
            }
            println("BENCH   boxmap: spots ${map.spots.size}, support>=6 ${strong.size} -> cells ${map.occupiedCells(mg)} (support>=12: ${map.occupiedCells(mg, 12)}), seen in 2+ photos ${strong.count { sp -> sp.sightings.map { it.photo }.toSet().size >= 2 }}, centre %.3f, pairs ${mp.size}; grid %.1f x %.1f off %.1f/%.1f".format(if (moff.isEmpty()) 0.0 else moff.average(), mg.cw, mg.ch, mg.offX, mg.offY))

            cat.clear()
        }
    }

    /** Montage of rejected outlines (PM_REJECTS=<reason prefix>) to see what the detector refuses. */
    @Test
    fun rejectMontage() {
        val want = System.getenv("PM_REJECTS") ?: return
        OpenCV.loadLocally()
        val onPhone = File("../session/export/on_phone.txt").readLines().map { it.trim() }.toSet()
        val photos = File("../session/photos").listFiles { f -> f.name in onPhone }!!.sorted().filterIndexed { i, _ -> i % 25 == 3 }
        val session = ScanSession()
        val tiles = ArrayList<java.awt.image.BufferedImage>()
        session.pipeline.debugReject = { img, xs, ys, why ->
            if (why.startsWith(want) && tiles.size < 64) {
                val t = 220
                val bi = java.awt.image.BufferedImage(img.width, img.height, java.awt.image.BufferedImage.TYPE_INT_RGB)
                bi.setRGB(0, 0, img.width, img.height, img.pixels, 0, img.width)
                val out = java.awt.image.BufferedImage(t, t, java.awt.image.BufferedImage.TYPE_INT_RGB)
                val g = out.createGraphics()
                val k = t.toDouble() / maxOf(img.width, img.height)
                g.drawImage(bi, 0, 0, (img.width * k).toInt(), (img.height * k).toInt(), null)
                g.color = java.awt.Color.GREEN; g.stroke = java.awt.BasicStroke(2f)
                val poly = java.awt.Polygon(IntArray(xs.size) { (xs[it] * k).toInt() }, IntArray(ys.size) { (ys[it] * k).toInt() }, xs.size)
                g.drawPolygon(poly); g.color = java.awt.Color.YELLOW; g.drawString(why, 4, 14); g.dispose()
                tiles.add(out)
            }
        }
        for (f in photos) {
            val meta = File(f.path.removeSuffix(".jpg") + ".txt").readLines().associate { l -> l.substringBefore(" ") to l.substringAfter(" ") }
            session.process(f.readBytes(), meta.getValue("rotation").toInt(), meta.getValue("gravity").split(" ").map { it.toFloat() }.toFloatArray(), meta.getValue("focalPerLongSide").toDouble())
        }
        val cols = 8; val rows = (tiles.size + cols - 1) / cols
        val sheet = java.awt.image.BufferedImage(cols * 220, maxOf(1, rows) * 220, java.awt.image.BufferedImage.TYPE_INT_RGB)
        val g = sheet.createGraphics()
        for ((i, t) in tiles.withIndex()) g.drawImage(t, (i % cols) * 220, (i / cols) * 220, null)
        g.dispose()
        javax.imageio.ImageIO.write(sheet, "jpg", File("../test/rejects_$want.jpg"))
        println("REJECTS ${tiles.size} tiles")
    }

    /** Throughput with 1 vs N analysing threads, same photos, same code path as the app (PM_PAR=N). */
    @Test
    fun parallelSpeed() {
        val par = System.getenv("PM_PAR")?.toInt() ?: return
        OpenCV.loadLocally()
        val onPhone = File("../session/export/on_phone.txt").readLines().map { it.trim() }.toSet()
        val photos = File("../session/photos").listFiles { f -> f.name in onPhone }!!.sorted().take(24)
        val poster = Imgcodecs.imread("../test/patches/poster64.png")
        val placer = BoxPlacer(poster)
        for (threads in listOf(1, par)) {
            val session = ScanSession()
            val map = BoxMap(placer)
            session.analysisHook = { img, pass -> map.addAll(map.locate("x", img, pass.out.pieces.map { it.xs to it.ys } + pass.out.loose, pass.pieceSize, pass.out.pieces.size)) }
            val pool = java.util.concurrent.Executors.newFixedThreadPool(threads)
            val t0 = System.currentTimeMillis()
            val jobs = photos.map { f -> pool.submit {
                val meta = File(f.path.removeSuffix(".jpg") + ".txt").readLines().associate { l -> l.substringBefore(" ") to l.substringAfter(" ") }
                session.process(f.readBytes(), meta.getValue("rotation").toInt(), meta.getValue("gravity").split(" ").map { it.toFloat() }.toFloatArray(), meta.getValue("focalPerLongSide").toDouble())
            } }
            jobs.forEach { it.get() }
            val ms = System.currentTimeMillis() - t0
            pool.shutdown()
            println("PAR threads $threads: ${photos.size} photos in $ms ms (%.0f ms/photo); catalogue ${session.catalogue.entries.size}, box sightings ${map.all.size}".format(ms.toDouble() / photos.size))
        }
    }
}
