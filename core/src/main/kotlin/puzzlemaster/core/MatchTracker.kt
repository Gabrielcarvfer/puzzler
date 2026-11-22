package puzzlemaster.core

import kotlin.math.hypot

/**
 * Temporal layer on top of per-frame matching.
 *
 * Pieces are tracked across frames by position. For every pair of tracked pieces we count how often
 * they were both visible and how often the per-frame matcher said they fit; the ratio (with a prior) is
 * the pair's probability. Up to [Config.maxLocked] confident pairs are *locked*: they stay on screen
 * even when a frame misses them, until one of the two pieces disappears for a while (picked up), or
 * the user dismisses them. Then the next best pair takes the free slot.
 */
class MatchTracker(val cfg: Config = Config()) {
    data class Config(
        val maxLocked: Int = 5,
        /** Frames a track survives without being seen (covers hands / brief occlusion). */
        val maxMissingFrames: Int = 10,
        /** Minimum frames both pieces must have been seen together before locking. */
        val minCoVisible: Int = 4,
        val lockProbability: Double = 0.6,
        /** Track association radius as a fraction of piece size. */
        val associationRadius: Double = 0.6,
    )

    class Track(val id: Int, var piece: Piece) {
        var missing = 0
        var seenFrames = 0
        val x get() = piece.centroidX
        val y get() = piece.centroidY
    }

    class PairStats {
        var coVisible = 0
        var hits = 0.0
        var lastScore = 1.0
        /** Matched edge tip positions relative to each track's centroid (latest frame they matched). */
        var offAx = 0.0; var offAy = 0.0; var offBx = 0.0; var offBy = 0.0
        /** Probability with a Beta(1,2) prior so a single lucky frame isn't trusted. */
        val probability get() = (hits + 1) / (coVisible + 3)
    }

    /** A pair shown to the user. Track ids, not frame piece ids. */
    class Group(val slot: Int, val a: Int, val b: Int, val stats: PairStats) {
        var ax = 0.0; var ay = 0.0; var bx = 0.0; var by = 0.0
        var aVisible = true; var bVisible = true
    }

    private val tracks = LinkedHashMap<Int, Track>()
    private val pairs = HashMap<Long, PairStats>()
    private val locked = ArrayList<Group>()
    private val dismissed = HashSet<Long>()
    private var nextTrack = 1
    var frame = 0; private set

    val activeTracks: Collection<Track> get() = tracks.values
    val groups: List<Group> get() = locked

    private fun key(a: Int, b: Int): Long = if (a < b) a.toLong() shl 32 or b.toLong() else b.toLong() shl 32 or a.toLong()

    /** Feeds one frame. Returns the track id for each piece id of this frame. */
    fun update(pieces: List<Piece>, matches: List<EdgeMatch>): Map<Int, Int> {
        frame++
        val assigned = associate(pieces)
        val visible = assigned.values.toSet()

        // Co-visibility for every pair of visible tracks.
        val vis = visible.sorted()
        for (i in vis.indices) for (j in i + 1 until vis.size) pairs.getOrPut(key(vis[i], vis[j])) { PairStats() }.coVisible++
        val pieceById = pieces.associateBy { it.id }
        for (m in matches) {
            val ta = assigned[m.a.pieceId] ?: continue
            val tb = assigned[m.b.pieceId] ?: continue
            if (ta == tb) continue
            val st = pairs.getOrPut(key(ta, tb)) { PairStats() }
            st.hits += 1.0
            st.lastScore = m.score
            val (ea, eb) = if (ta < tb) m.a to m.b else m.b to m.a
            val pa = pieceById.getValue(ea.pieceId); val pb = pieceById.getValue(eb.pieceId)
            st.offAx = ea.midX - pa.centroidX; st.offAy = ea.midY - pa.centroidY
            st.offBx = eb.midX - pb.centroidX; st.offBy = eb.midY - pb.centroidY
        }

        // Release groups whose pieces are gone.
        locked.removeAll { g -> tracks[g.a] == null || tracks[g.b] == null }
        // Fill free slots with the most probable pairs that don't reuse a locked piece.
        if (locked.size < cfg.maxLocked) {
            val busy = locked.flatMap { listOf(it.a, it.b) }.toMutableSet()
            val candidates = pairs.entries
                .filter { (k, s) -> s.coVisible >= cfg.minCoVisible && s.probability >= cfg.lockProbability && k !in dismissed }
                .sortedByDescending { it.value.probability }
            for ((k, s) in candidates) {
                if (locked.size >= cfg.maxLocked) break
                val a = (k shr 32).toInt(); val b = (k and 0xFFFFFFFFL).toInt()
                if (a in busy || b in busy || tracks[a] == null || tracks[b] == null) continue
                val slot = (1..cfg.maxLocked).first { sl -> locked.none { it.slot == sl } }
                locked.add(Group(slot, a, b, s))
                busy.add(a); busy.add(b)
            }
            locked.sortBy { it.slot }
        }
        for (g in locked) {
            val ta = tracks.getValue(g.a); val tb = tracks.getValue(g.b)
            g.ax = ta.x + g.stats.offAx; g.ay = ta.y + g.stats.offAy
            g.bx = tb.x + g.stats.offBx; g.by = tb.y + g.stats.offBy
            g.aVisible = ta.missing == 0; g.bVisible = tb.missing == 0
        }
        return assigned
    }

    /** User dismissed a group (e.g. "Next"): never lock that pair again. */
    fun dismiss(slot: Int? = null) {
        val gone = if (slot == null) locked.toList() else locked.filter { it.slot == slot }
        for (g in gone) dismissed.add(key(g.a, g.b))
        locked.removeAll(gone.toSet())
    }

    fun reset() { tracks.clear(); pairs.clear(); locked.clear(); dismissed.clear() }

    private fun associate(pieces: List<Piece>): Map<Int, Int> {
        val result = HashMap<Int, Int>()
        val free = tracks.values.toMutableList()
        // Greedy nearest assignment, closest pairs first.
        val cand = ArrayList<Triple<Double, Piece, Track>>()
        for (p in pieces) for (t in free) {
            val d = hypot(p.centroidX - t.x, p.centroidY - t.y)
            if (d < cfg.associationRadius * maxOf(p.scale, t.piece.scale)) cand.add(Triple(d, p, t))
        }
        cand.sortBy { it.first }
        val usedT = HashSet<Int>()
        for ((_, p, t) in cand) {
            if (p.id in result || t.id in usedT) continue
            result[p.id] = t.id; usedT.add(t.id); t.piece = p; t.missing = 0; t.seenFrames++
        }
        for (p in pieces) if (p.id !in result) {
            val t = Track(nextTrack++, p); t.seenFrames = 1
            tracks[t.id] = t; result[p.id] = t.id
        }
        val it = tracks.values.iterator()
        while (it.hasNext()) {
            val t = it.next()
            if (t.id in usedT || result.containsValue(t.id)) continue
            t.missing++
            if (t.missing > cfg.maxMissingFrames) {
                it.remove()
                pairs.keys.removeAll { k -> (k shr 32).toInt() == t.id || (k and 0xFFFFFFFFL).toInt() == t.id }
            }
        }
        return result
    }
}
