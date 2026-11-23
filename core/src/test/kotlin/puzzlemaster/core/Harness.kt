package puzzlemaster.core

import puzzlemaster.core.synth.SyntheticPuzzle
import kotlin.math.hypot

/** Ground-truth identity of a detected edge: piece (r,c) and side 0=top,1=right,2=bottom,3=left. */
data class TrueEdge(val r: Int, val c: Int, val side: Int)

fun SyntheticPuzzle.neighbour(e: TrueEdge): TrueEdge? = when (e.side) {
    0 -> if (e.r > 0) TrueEdge(e.r - 1, e.c, 2) else null
    1 -> if (e.c < cols - 1) TrueEdge(e.r, e.c + 1, 3) else null
    2 -> if (e.r < rows - 1) TrueEdge(e.r + 1, e.c, 0) else null
    else -> if (e.c > 0) TrueEdge(e.r, e.c - 1, 1) else null
}

/** Maps each detected edge to the true side whose (transformed) midpoint is closest. */
fun SyntheticPuzzle.identifyEdges(piece: Piece, pl: SyntheticPuzzle.Placement): Map<Edge, TrueEdge> {
    val k = corners(pl.r, pl.c).map { pl.toImage(it.first, it.second) }
    val mids = (0 until 4).map { s -> ((k[s].first + k[(s + 1) % 4].first) / 2) to ((k[s].second + k[(s + 1) % 4].second) / 2) }
    return piece.edges.associateWith { e ->
        val mx = (e.startX + e.endX) / 2; val my = (e.startY + e.endY) / 2
        val s = (0 until 4).minBy { hypot(mids[it].first - mx, mids[it].second - my) }
        TrueEdge(pl.r, pl.c, s)
    }
}

fun SyntheticPuzzle.trueType(e: TrueEdge): EdgeType {
    if (neighbour(e) == null) return EdgeType.FLAT
    val (xs, ys) = side(e.r, e.c, e.side)
    val n = xs.size
    val dx = xs[n - 1] - xs[0]; val dy = ys[n - 1] - ys[0]
    val len = hypot(dx, dy)
    var out = 0.0; var inn = 0.0
    for (i in 0 until n) {
        val across = (-(xs[i] - xs[0]) * dy + (ys[i] - ys[0]) * dx) / len // + = interior
        if (-across > out) out = -across
        if (across > inn) inn = across
    }
    return if (out > inn) EdgeType.TAB else EdgeType.BLANK
}

fun percent(x: Double) = "%.1f%%".format(x * 100)

/** Directory for test figures, or null when disabled. */
fun figureDir(): java.io.File? = System.getenv("PM_DEBUG")?.takeIf { it.isNotEmpty() }?.let { java.io.File(it).apply { mkdirs() } }

/** Draws a piece's detected outline (edge types), corners and an optional caption onto [img]. */
fun renderPieceFigure(img: RgbImage, piece: Piece?, caption: String, highlight: Edge? = null): java.awt.image.BufferedImage {
    val s = 2
    val bi = java.awt.image.BufferedImage(img.width * s, img.height * s, java.awt.image.BufferedImage.TYPE_INT_RGB)
    val src = java.awt.image.BufferedImage(img.width, img.height, java.awt.image.BufferedImage.TYPE_INT_RGB)
    src.setRGB(0, 0, img.width, img.height, img.pixels, 0, img.width)
    val g = bi.createGraphics()
    g.setRenderingHint(java.awt.RenderingHints.KEY_ANTIALIASING, java.awt.RenderingHints.VALUE_ANTIALIAS_ON)
    g.drawImage(src, 0, 0, img.width * s, img.height * s, null)
    if (piece != null) {
        val n = piece.xs.size
        for (e in 0 until 4) {
            val edge = piece.edges[e]
            g.color = when (edge.type) { EdgeType.FLAT -> java.awt.Color(200, 200, 200); EdgeType.TAB -> java.awt.Color(255, 160, 0); EdgeType.BLANK -> java.awt.Color(0, 200, 255) }
            g.stroke = java.awt.BasicStroke(if (edge === highlight) 6f else 2.5f)
            var i = piece.cornerIdx[e]
            while (i != piece.cornerIdx[(e + 1) % 4]) { val j = (i + 1) % n; g.drawLine((piece.xs[i] * s).toInt(), (piece.ys[i] * s).toInt(), (piece.xs[j] * s).toInt(), (piece.ys[j] * s).toInt()); i = j }
        }
        g.color = java.awt.Color.RED
        for (k in piece.cornerIdx) g.fillOval((piece.xs[k] * s).toInt() - 5, (piece.ys[k] * s).toInt() - 5, 10, 10)
    }
    g.color = java.awt.Color.WHITE
    g.drawString(caption, 4, 14)
    return bi
}

/** Tiles figures into one image. */
fun saveMontage(figs: List<java.awt.image.BufferedImage>, columns: Int, file: java.io.File) {
    if (figs.isEmpty()) return
    val cw = figs.maxOf { it.width }; val ch = figs.maxOf { it.height }
    val rows = (figs.size + columns - 1) / columns
    val out = java.awt.image.BufferedImage(cw * minOf(columns, figs.size), ch * rows, java.awt.image.BufferedImage.TYPE_INT_RGB)
    val g = out.createGraphics()
    figs.forEachIndexed { i, f -> g.drawImage(f, (i % columns) * cw, (i / columns) * ch, null) }
    javax.imageio.ImageIO.write(out, "png", file)
}
