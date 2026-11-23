package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Test
import org.opencv.core.Point
import org.opencv.imgcodecs.Imgcodecs
import java.io.File

/** Builds the phone's box map (files/boxmap.bin) from its stored photos, exactly as the app does (PM_EXPORT=1). */
class BoxMapExportTest {
    @Test
    fun export() {
        if (System.getenv("PM_EXPORT") == null) return
        OpenCV.loadLocally()
        val onPhone = File("../session/export/on_phone.txt").readLines().map { it.trim() }.toSet()
        val photos = File("../session/photos").listFiles { f -> f.name in onPhone }!!.sorted()
        val poster = Imgcodecs.imread("../test/patches/poster64.png").takeIf { !it.empty() }
            ?: BoxLocator.fromPhoto(Imgcodecs.imread("src/test/resources/box/box.jpg"), listOf(Point(254.0, 430.0), Point(7954.0, 410.0), Point(7980.0, 5948.0), Point(238.0, 5932.0)), cell = 64).poster
        val map = BoxMap(BoxPlacer(poster))
        val session = ScanSession()
        var name = ""; var rot = 0
        session.debugHook = { img, pass ->
            val m = pass.back
            map.observe(name, img, pass.out.pieces.map { it.xs to it.ys } + pass.out.loose, pass.pieceSize, pass.out.pieces.size, rot) { x0, y0 ->
                var x = x0; var y = y0
                if (m != null) { val wz = m[6] * x + m[7] * y + m[8]; val nx = (m[0] * x + m[1] * y + m[2]) / wz; y = (m[3] * x + m[4] * y + m[5]) / wz; x = nx }
                Point(x / pass.frameW, y / pass.frameH)
            }
        }
        for (f in photos) {
            val meta = f.path.removeSuffix(".jpg").let { File("$it.txt") }.readLines().associate { l -> l.substringBefore(" ") to l.substringAfter(" ") }
            name = f.nameWithoutExtension; rot = meta.getValue("rotation").toInt()
            session.process(f.readBytes(), rot, meta.getValue("gravity").split(" ").map { it.toFloat() }.toFloatArray(), meta.getValue("focalPerLongSide").toDouble())
        }
        map.save(File("../session/export/boxmap.bin"))
        val g = map.cleanGrid()
        val cells = map.cells(g).filterValues { it.votes >= 6 }
        var np = 0; var nj = 0
        for ((k, c) in cells) for (d in listOf(1 to 0, 0 to 1)) { val o = cells[(k.first + d.first) to (k.second + d.second)] ?: continue; np++; if (map.joined(c, o, g) || map.joined(o, c, g)) nj++ }
        println("BOXMAP ${photos.size} photos: ${map.all.size} sightings, ${cells.size} cells with 6+ votes, $np pairs, of which already joined on the table $nj; catalogue-only path had ${session.catalogue.entries.size} entries")
        val back = BoxMap(map.placer); println("BOXMAP reload ok=${back.load(File("../session/export/boxmap.bin"))} sightings ${back.all.size}")
    }
}
