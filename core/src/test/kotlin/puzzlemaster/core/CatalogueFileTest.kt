package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Test
import java.io.File

/** Examines a catalogue pulled from the phone (../session/phone/catalogue.bin). */
class CatalogueFileTest {
    @Test
    fun analyse() {
        val f = File("../session/phone/catalogue.bin").takeIf { it.exists() && it.length() > 100 } ?: return
        OpenCV.loadLocally()
        val cat = PieceCatalogue()
        if (!cat.load(f)) { println("CATFILE could not be read"); return }
        val e = cat.entries
        println("CATFILE ${e.size} entries; sightings: once ${e.count { it.sightings == 1 }}, 2-4 ${e.count { it.sightings in 2..4 }}, 5-14 ${e.count { it.sightings in 5..14 }}, 15+ ${e.count { it.sightings >= 15 }}")
        println("CATFILE entries with a usable picture (8+ features): ${e.count { it.descriptors.rows() >= 8 }}; edge pieces (a flat side): ${e.count { en -> en.sides.any { it.type == EdgeType.FLAT } }}")
        val sig = e.groupingBy { en -> val c = en.sides.joinToString("") { it.type.name.take(1) }; (0 until 4).map { c.substring(it) + c.substring(0, it) }.min() }.eachCount()
        println("CATFILE side-type patterns: $sig")
        for ((tight, loose) in listOf(0.016 to 0.03, 0.022 to 0.035, 0.028 to 0.04)) {
            val c2 = PieceCatalogue(PieceCatalogue.Config(mergeTight = tight, mergeLoose = loose)); c2.load(f)
            val m = c2.mergeDuplicates()
            val strict = c2.solve()
            val c3 = PieceCatalogue(PieceCatalogue.Config(mergeTight = tight, mergeLoose = loose, maxGap = 0.022, maxRatio = 0.9)); c3.load(f); c3.mergeDuplicates()
            val relaxed = c3.solve()
            println("CATFILE merge limits %.1f%%/%.1f%%: ${m.size} merges -> ${c2.entries.size} entries (measured 5+ times: ${c2.entries.count { it.sightings >= 5 }}); pairs: strict ${strict.size}, relaxed ${relaxed.size}".format(tight * 100, loose * 100) +
                (if (relaxed.isNotEmpty()) " best gaps " + relaxed.take(6).joinToString { "%.2f%%(x${it.a.sightings},x${it.b.sightings})".format(it.gap * 100) } else ""))
        }
    }
}
