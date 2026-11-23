package puzzlemaster.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import puzzlemaster.core.synth.SyntheticPuzzle
import kotlin.math.PI
import kotlin.random.Random

class MatchTrackerTest {
    /** Detects a 30-piece scene once; frames are then simulated with flicker, false matches and removals. */
    @Test
    fun locksAtMostFiveConfidentPairsAndReleasesPickedUpPieces() {
        val p = SyntheticPuzzle(6, 5, seed = 21)
        val rnd = Random(21)
        val img = p.background(1600, 1200, RgbImage.argb(28, 66, 42), rnd)
        val pls = ArrayList<SyntheticPuzzle.Placement>()
        var k = 0
        for (r in 0 until p.rows) for (c in 0 until p.cols) {
            val (cx, cy) = p.pieceCenter(r, c)
            val tx = 140.0 + (k % 6) * 260; val ty = 120.0 + (k / 6) * 240
            pls.add(SyntheticPuzzle.Placement(r, c, rnd.nextDouble(2 * PI), 0.9, tx, ty, cx, cy)); k++
        }
        for (pl in pls) p.render(img, pl, rnd)
        val pieces = PieceDetector().detect(img).pieces
        val truthMatches = EdgeMatcher().matchInView(pieces)
        assertEquals(30, pieces.size)

        val tracker = MatchTracker()
        val removed = HashSet<Int>()
        var firstGroups: List<Pair<Int, Int>> = emptyList()
        for (f in 0 until 60) {
            // Each frame: a piece is missed 10% of the time, a true match is missed 40% of the time,
            // and a couple of random false matches appear.
            val visible = pieces.filter { it.id !in removed && rnd.nextDouble() > 0.1 }
            val ids = visible.map { it.id }.toSet()
            val ms = truthMatches.filter { it.a.pieceId in ids && it.b.pieceId in ids && rnd.nextDouble() > 0.4 }.toMutableList()
            repeat(2) {
                val a = visible.random(rnd); val b = visible.random(rnd)
                if (a.id != b.id) ms.add(EdgeMatch(a.edges[0], b.edges[1], 0.1))
            }
            tracker.update(visible, ms)
            val groups = tracker.groups
            assertTrue("at most 5 groups", groups.size <= 5)
            if (f == 15) {
                assertEquals("5 groups locked once evidence accumulates", 5, groups.size)
                firstGroups = groups.map { it.a to it.b }
                // Lock must be stable: pick up both pieces of group 1.
                val trackToPiece = tracker.activeTracks.associate { it.id to it.piece.id }
                removed.add(trackToPiece.getValue(groups[0].a)); removed.add(trackToPiece.getValue(groups[0].b))
            }
            if (f in 16..20) assertEquals("group kept while pieces briefly missing", firstGroups[0], tracker.groups.first { it.slot == 1 }.let { it.a to it.b })
        }
        val final = tracker.groups
        assertEquals(5, final.size)
        assertTrue("picked-up pair released and slot refilled", final.none { (it.a to it.b) == firstGroups[0] })
        // Every locked pair must be a real match (false matches never accumulate).
        val trackToPiece = tracker.activeTracks.associate { it.id to it.piece.id }
        val realPairs = truthMatches.map { setOf(it.a.pieceId, it.b.pieceId) }.toSet()
        for (g in final) assertTrue("locked pair is real", setOf(trackToPiece.getValue(g.a), trackToPiece.getValue(g.b)) in realPairs)
        println("tracker: final groups ${final.map { "${it.slot}:%.2f".format(it.stats.probability) }}")
    }
}
