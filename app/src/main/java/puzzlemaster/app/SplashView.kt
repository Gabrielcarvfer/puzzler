package puzzlemaster.app

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.BlurMaskFilter
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PathMeasure
import android.graphics.Typeface
import android.view.MotionEvent
import android.view.View
import android.view.animation.LinearInterpolator
import androidx.core.graphics.PathParser
import kotlin.math.min

/**
 * Launch screen: a light trail draws the Puzzler piece on a dark grid, the outline flares, the name
 * appears, then everything fades out. Tap to skip. [onDone] runs once when it has finished.
 */
class SplashView(context: Context, private val onDone: () -> Unit, private val onSkip: () -> Unit = {}) : View(context) {
    private val orange = 0xFFFF7A1A.toInt()
    private val piece = PathParser.createPathFromPathData(
        "M34,34 H46 C47,34 46,31 46,28.5 A8,8 0 1,1 62,28.5 C62,31 61,34 62,34 H74 " +
            "V46 C74,47 77,46 79.5,46 A8,8 0 1,1 79.5,62 C77,62 74,61 74,62 V74 " +
            "H62 C61,74 62,71 62,68.5 A8,8 0 1,0 46,68.5 C46,71 47,74 46,74 H34 " +
            "V62 C34,61 37,62 39.5,62 A8,8 0 1,0 39.5,46 C37,46 34,47 34,46 Z",
    )
    private val scaled = Path()
    private val partial = Path()
    private var measure = PathMeasure()
    private var length = 0f
    private var unit = 1f
    private var t = 0f // 0..1 over the whole animation
    private var finished = false

    private val glowWide = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeJoin = Paint.Join.ROUND; strokeCap = Paint.Cap.ROUND; color = orange }
    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeJoin = Paint.Join.ROUND; strokeCap = Paint.Cap.ROUND; color = orange }
    private val core = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeJoin = Paint.Join.ROUND; strokeCap = Paint.Cap.ROUND; color = 0xFFFFE3C8.toInt() }
    private val grid = Paint().apply { color = 0xFF1C2633.toInt(); strokeWidth = 2f }
    private val head = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE }
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = orange; textAlign = Paint.Align.CENTER; typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD); letterSpacing = 0.35f
    }
    private val textGlow = Paint(text).apply { }
    private val anim = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = TOTAL_MS; interpolator = LinearInterpolator()
        addUpdateListener { t = it.animatedValue as Float; invalidate() }
        addListener(object : android.animation.AnimatorListenerAdapter() { override fun onAnimationEnd(a: android.animation.Animator) = finish() })
    }

    init { setLayerType(LAYER_TYPE_SOFTWARE, null); isClickable = true }

    override fun onAttachedToWindow() { super.onAttachedToWindow(); anim.start() }
    override fun onDetachedFromWindow() { anim.cancel(); super.onDetachedFromWindow() }

    private fun finish() { if (!finished) { finished = true; onDone() } }

    override fun onTouchEvent(e: MotionEvent): Boolean { if (e.action == MotionEvent.ACTION_UP) { onSkip(); anim.cancel(); finish() }; return true }

    override fun onSizeChanged(w: Int, h: Int, ow: Int, oh: Int) {
        unit = min(w, h) / 108f * 0.9f
        val m = Matrix(); m.setScale(unit, unit)
        // Centre the piece (its body centre is at 54,54 on the icon grid, shifted like the icon).
        m.postTranslate(w / 2f - 54 * unit + 1.5f * unit, h * 0.42f - 54 * unit)
        scaled.reset(); piece.transform(m, scaled)
        measure = PathMeasure(scaled, true); length = measure.length
        glowWide.strokeWidth = 7f * unit; glowWide.maskFilter = BlurMaskFilter(5f * unit, BlurMaskFilter.Blur.NORMAL)
        line.strokeWidth = 2.2f * unit; line.maskFilter = BlurMaskFilter(1.2f * unit, BlurMaskFilter.Blur.SOLID)
        core.strokeWidth = 0.8f * unit
        text.textSize = 11f * unit
        textGlow.textSize = text.textSize; textGlow.maskFilter = BlurMaskFilter(3f * unit, BlurMaskFilter.Blur.NORMAL)
    }

    override fun onDraw(c: Canvas) {
        val ms = t * TOTAL_MS
        val fadeOut = ((ms - FADE_AT) / (TOTAL_MS - FADE_AT)).coerceIn(0f, 1f)
        val alpha = 1f - fadeOut
        c.drawColor(Color.argb((255 * alpha).toInt(), 7, 9, 13))
        // Grid, rising slowly into view.
        grid.alpha = (90 * min(1f, ms / 600f) * alpha).toInt()
        val step = 12 * unit
        var x = (width / 2f) % step; while (x < width) { c.drawLine(x, 0f, x, height.toFloat(), grid); x += step }
        var y = (height * 0.42f) % step; while (y < height) { c.drawLine(0f, y, width.toFloat(), y, grid); y += step }

        // Light trail: drawn along the outline over DRAW_MS.
        val drawn = ((ms - 250) / DRAW_MS).coerceIn(0f, 1f)
        val eased = 1 - (1 - drawn) * (1 - drawn)
        partial.reset(); measure.getSegment(0f, eased * length, partial, true)
        // Flare once the loop closes.
        val flare = if (drawn >= 1f) (1f - ((ms - 250 - DRAW_MS) / 800f).coerceIn(0f, 1f)) else 0f
        glowWide.alpha = ((70 + 150 * flare) * alpha).toInt()
        line.alpha = (255 * alpha).toInt(); core.alpha = (255 * alpha).toInt()
        c.drawPath(partial, glowWide); c.drawPath(partial, line); c.drawPath(partial, core)
        if (drawn in 0.001f..0.999f) {
            val pos = FloatArray(2); measure.getPosTan(eased * length, pos, null)
            head.maskFilter = BlurMaskFilter(4f * unit, BlurMaskFilter.Blur.NORMAL); head.color = orange
            c.drawCircle(pos[0], pos[1], 4.5f * unit, head)
            head.maskFilter = null; head.color = Color.WHITE
            c.drawCircle(pos[0], pos[1], 1.6f * unit, head)
        }

        // Name, typed in after the outline closes.
        val nameT = ((ms - 250 - DRAW_MS - 200) / 1000f).coerceIn(0f, 1f)
        if (nameT > 0f) {
            val word = "PUZZLER"
            val shown = word.substring(0, (nameT * word.length).toInt().coerceIn(1, word.length))
            val ty = height * 0.42f + 46 * unit
            // Keep the full word's layout so the letters don't shift as they appear.
            val full = text.measureText(word)
            val left = width / 2f - full / 2f
            text.textAlign = Paint.Align.LEFT; textGlow.textAlign = Paint.Align.LEFT
            textGlow.alpha = (200 * alpha).toInt(); text.alpha = (255 * alpha).toInt()
            c.drawText(shown, left, ty, textGlow); c.drawText(shown, left, ty, text)
        }
    }

    companion object {
        private const val DRAW_MS = 2800f
        private const val FADE_AT = 5000f
        private const val TOTAL_MS = 5600L
    }
}
