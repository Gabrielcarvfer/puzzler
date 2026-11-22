package puzzlemaster.core

/** Sub-pixel outline refinement at full resolution. */
object OutlineRefiner {
    fun luminance(img: RgbImage): FloatArray =
        FloatArray(img.width * img.height) { val q = img.pixels[it]; 0.299f * ((q shr 16) and 0xFF) + 0.587f * ((q shr 8) and 0xFF) + 0.114f * (q and 0xFF) }

    /** Moves outline points (every ~1 px) to the max luminance gradient within +-1.5 working pixels along the normal. */
    fun refine(px: DoubleArray, py: DoubleArray, lum: FloatArray, w: Int, h: Int, range: Double): Pair<DoubleArray, DoubleArray> {
        val per = PieceDetector.perimeter(px, py)
        val n = per.toInt().coerceIn(100, 4000)
        val (xs, ys) = PieceDetector.resampleClosed(px, py, n)
        val ox = DoubleArray(n); val oy = DoubleArray(n)
        fun lumAt(x: Double, y: Double): Float {
            val ix = x.toInt().coerceIn(0, w - 2); val iy = y.toInt().coerceIn(0, h - 2)
            val tx = (x - ix).toFloat().coerceIn(0f, 1f); val ty = (y - iy).toFloat().coerceIn(0f, 1f)
            val a = lum[iy * w + ix] * (1 - tx) + lum[iy * w + ix + 1] * tx
            val b = lum[(iy + 1) * w + ix] * (1 - tx) + lum[(iy + 1) * w + ix + 1] * tx
            return a * (1 - ty) + b * ty
        }
        for (i in 0 until n) {
            val a = (i - 2 + n) % n; val b = (i + 2) % n
            var nx = -(ys[b] - ys[a]); var ny = xs[b] - xs[a]
            val l = kotlin.math.hypot(nx, ny).coerceAtLeast(1e-9); nx /= l; ny /= l
            var best = 0.0; var bestT = 0.0
            var t = -range
            while (t <= range) {
                val g = kotlin.math.abs(lumAt(xs[i] + nx * (t + 0.5), ys[i] + ny * (t + 0.5)) - lumAt(xs[i] + nx * (t - 0.5), ys[i] + ny * (t - 0.5))).toDouble()
                // Slight preference for small moves.
                val sc = g * (1 - 0.15 * kotlin.math.abs(t) / range)
                if (sc > best) { best = sc; bestT = t }
                t += 0.5
            }
            ox[i] = xs[i] + nx * bestT; oy[i] = ys[i] + ny * bestT
        }
        // Light smoothing to remove jitter from the snap.
        return PieceDetector.smoothClosed(ox, oy, 2)
    }

}
