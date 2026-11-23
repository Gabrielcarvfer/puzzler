package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Test
import java.io.BufferedOutputStream
import java.io.DataOutputStream
import java.io.File

/**
 * Rebuilds the phone's catalogue from the photos it has already taken (../session/photos), with the
 * same code the phone runs, and writes it in the phone's file formats to ../session/export:
 * catalogue.bin and thumbs.bin (which photo each piece was last seen in, and its outline there).
 */
class ExportForPhoneTest {
    @Test
    fun export() {
        if (System.getenv("PM_EXPORT") == null) return
        // Only photos the phone still has (it keeps the newest ones), so every piece's last-seen photo can be shown there.
        val onPhone = File("../session/export/on_phone.txt").takeIf { it.exists() }?.readLines()?.map { it.trim() }?.toSet()
        val photos = File("../session/photos").listFiles { f -> f.name.endsWith(".jpg") && (onPhone == null || f.name in onPhone) }?.sorted() ?: return
        OpenCV.loadLocally()
        class Seen(val photo: String, val rotation: Int, val outline: FloatArray)
        val seen = HashMap<Int, Seen>()
        val session = ScanSession()
        var current = "" ; var rot = 0
        session.debugHook = { _, pass ->
            for (s in pass.sightings) {
                val p = s.piece; val m = pass.back
                val step = maxOf(1, p.xs.size / 80)
                val o = FloatArray(((p.xs.size + step - 1) / step) * 2)
                var q = 0; var t = 0
                while (t < p.xs.size) {
                    var x = p.xs[t]; var y = p.ys[t]
                    if (m != null) { val wz = m[6] * x + m[7] * y + m[8]; val nx = (m[0] * x + m[1] * y + m[2]) / wz; y = (m[3] * x + m[4] * y + m[5]) / wz; x = nx }
                    o[q++] = (x / pass.frameW).toFloat(); o[q++] = (y / pass.frameH).toFloat(); t += step
                }
                seen[s.entry.id] = Seen(current, rot, o)
            }
        }
        for ((i, f) in photos.withIndex()) {
            val meta = File(f.path.removeSuffix(".jpg") + ".txt").readLines().associate { l -> l.substringBefore(" ") to l.substringAfter(" ") }
            current = f.nameWithoutExtension; rot = meta.getValue("rotation").toInt()
            session.process(f.readBytes(), rot, meta.getValue("gravity").split(" ").map { it.toFloat() }.toFloatArray(), meta.getValue("focalPerLongSide").toDouble())
            if ((i + 1) % 10 == 0) session.catalogue.mergeDuplicates()
        }
        val cat = session.catalogue
        cat.mergeDuplicates()
        val out = File("../session/export").apply { mkdirs() }
        cat.save(File(out, "catalogue.bin"))
        DataOutputStream(BufferedOutputStream(File(out, "thumbs.bin").outputStream())).use { o ->
            o.writeInt(cat.entries.size)
            for (e in cat.entries) {
                val t = seen[e.id]
                o.writeUTF(t?.photo ?: ""); o.writeInt(t?.rotation ?: 0)
                val ol = t?.outline ?: FloatArray(0)
                o.writeInt(ol.size); for (v in ol) o.writeFloat(v)
            }
        }
        // How much do the extra views help placement on the box?
        val boxFile = File("src/test/resources/box/box.jpg")
        if (boxFile.exists()) {
            val poster = BoxLocator.fromPhoto(org.opencv.imgcodecs.Imgcodecs.imread(boxFile.path), listOf(org.opencv.core.Point(254.0, 430.0), org.opencv.core.Point(7954.0, 410.0), org.opencv.core.Point(7980.0, 5948.0), org.opencv.core.Point(238.0, 5932.0)), cell = 64).poster
            val placer = BoxPlacer(poster)
            for (first in listOf(true, false)) {
                val hits = cat.entries.mapNotNull { placer.locate(it, firstViewOnly = first) }
                val grid = placer.fitGrid(hits); val placed = placer.place(hits, grid); val pairs = placer.pairs(placed)
                val off = hits.map { h -> val fx = (h.tx - grid.offX) / grid.cw; val fy = (h.ty - grid.offY) / grid.ch; kotlin.math.hypot(fx - Math.floor(fx) - 0.5, fy - Math.floor(fy) - 0.5) }
                println("EXPORT ${if (first) "first view only" else "all views"}: located ${hits.size}, one per cell ${placed.size}, mean distance from cell centre %.2f, pairs ${pairs.size} (sides agree ${pairs.count { it.sidesAgree }})".format(if (off.isEmpty()) 0.0 else off.average()))
            }
            println("EXPORT pieces with more than one view: ${cat.entries.count { it.extraViews.isNotEmpty() }}")
        }
        println("EXPORT ${photos.size} photos -> ${cat.entries.size} catalogue pieces (${cat.entries.count { it.sightings >= 3 }} measured 3+ times), last-seen photo known for ${cat.entries.count { seen[it.id] != null }}")
    }
}
