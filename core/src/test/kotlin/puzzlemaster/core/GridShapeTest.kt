package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Test
import java.io.File

/** Are the pieces square or oblong? Tells which grid the puzzle was cut in (e.g. 38x26 vs 40x25). */
class GridShapeTest {
    @Test
    fun pieceAspect() {
        val f = File("../session/phone/catalogue.bin").takeIf { it.exists() } ?: return
        OpenCV.loadLocally()
        val cat = PieceCatalogue()
        if (!cat.load(f)) { println("GRID catalogue unreadable"); return }
        for (minS in listOf(1, 3, 6)) {
            val r = cat.entries.filter { it.sightings >= minS }.map { e ->
                val a = e.sides[0].length + e.sides[2].length; val b = e.sides[1].length + e.sides[3].length
                maxOf(a, b) / minOf(a, b)
            }.sorted()
            if (r.size < 5) continue
            println("GRID pieces measured $minS+ times (${r.size}): long side / short side median %.3f (25%%-75%%: %.3f-%.3f)".format(r[r.size / 2], r[r.size / 4], r[r.size * 3 / 4]))
        }
        println("GRID expected: 38x26 grid -> 1.03, 40x25 grid -> 1.12")
    }
}
