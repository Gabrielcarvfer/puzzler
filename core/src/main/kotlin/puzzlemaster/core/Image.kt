package puzzlemaster.core

/** Simple ARGB (0xAARRGGBB) image, platform independent so it runs on Android and in JVM tests. */
class RgbImage(val width: Int, val height: Int, val pixels: IntArray = IntArray(width * height)) {
    init {
        require(pixels.size == width * height) { "pixel buffer size mismatch" }
    }

    operator fun get(x: Int, y: Int): Int = pixels[y * width + x]
    operator fun set(x: Int, y: Int, argb: Int) {
        pixels[y * width + x] = argb
    }

    /** Average color of a (2r+1)^2 patch, clamped to the image. Returns floats r,g,b into [out] at [offset]. */
    fun samplePatch(cx: Double, cy: Double, r: Int, out: FloatArray, offset: Int) {
        val ix = cx.toInt()
        val iy = cy.toInt()
        var sr = 0f
        var sg = 0f
        var sb = 0f
        var n = 0
        for (y in (iy - r)..(iy + r)) {
            if (y < 0 || y >= height) continue
            for (x in (ix - r)..(ix + r)) {
                if (x < 0 || x >= width) continue
                val p = pixels[y * width + x]
                sr += (p shr 16) and 0xFF
                sg += (p shr 8) and 0xFF
                sb += p and 0xFF
                n++
            }
        }
        if (n == 0) n = 1
        out[offset] = sr / n
        out[offset + 1] = sg / n
        out[offset + 2] = sb / n
    }

    companion object {
        fun argb(r: Int, g: Int, b: Int): Int =
            (0xFF shl 24) or (r.coerceIn(0, 255) shl 16) or (g.coerceIn(0, 255) shl 8) or b.coerceIn(0, 255)
    }
}
