package puzzlemaster.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CameraCharacteristics
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.Gravity
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import puzzlemaster.core.PieceCatalogue
import puzzlemaster.core.Rectifier
import java.io.File
import java.util.concurrent.Executors

/**
 * Always-on live scanning. Each pass: camera frame -> squared up with the gravity sensor ->
 * [LivePipeline] outlines the pieces lying free -> [PieceCatalogue] recognises or adds each one and
 * refines its side shapes -> pairs are solved on the catalogue and highlighted when in view.
 */
class MainActivity : ComponentActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var status: TextView
    private var camera: Camera? = null
    private val worker = Executors.newSingleThreadExecutor()
    /** Photos are taken on their own thread so that processing never holds up the camera. */
    private val captureExecutor = Executors.newSingleThreadExecutor()
    /** A kept photo waiting on disk to be processed. */
    private class Shot(val file: File, val rotation: Int, val gravity: FloatArray, val focalPerLongSide: Double) {
        /** Name under which the photo is kept in the session folder. */
        val name: String = "photo_%08d".format(System.currentTimeMillis() / 100 % 100000000)
    }
    /** The photo being processed right now (worker thread only). */
    /** The photo being processed on this thread (photos are processed in parallel). */
    private val shotOnThread = ThreadLocal<Shot?>()
    private val currentShot: Shot? get() = shotOnThread.get()
    private val queue = java.util.concurrent.LinkedBlockingDeque<Shot>(60)
    private var lastKeptAt = 0L
    private var shotNo = 0
    private val recentSharpness = ArrayDeque<Double>()
    @Volatile private var taken = 0
    @Volatile private var kept = 0
    @Volatile private var processed = 0
    @Volatile private var gyroRate = 0f
    @Volatile private var line2 = "Sweep the camera over the pieces."
    /** False while the user has stopped photo-taking (queued photos are still processed). */
    /** Off at start: photos are taken only after the user taps Start photos. */
    @Volatile private var capturing = false
    private lateinit var stopButton: Button
    private fun syncStopButton() { stopButton.text = if (capturing) "Stop photos" else "Start photos" }
    private val catalogue by lazy { PieceCatalogue() }
    private val session by lazy {
        puzzlemaster.core.ScanSession(catalogue).also { it.debugHook = { image, pass -> rememberThumbs(image, pass) }; it.analysisHook = { image, pass -> mapPieces(image, pass) } }
    }
    /** A small picture of every catalogued piece (latest sighting) and where its four sides are in it. */
    private class Thumb(
        /** Close-up and side positions; null for pieces known only from a saved catalogue. */
        val bmp: android.graphics.Bitmap?, val sides: Array<android.graphics.PointF>?,
        /** The photo it was last seen in, its rotation, and the piece's outline there (x, y pairs, 0..1 of the upright photo). */
        val photo: String, val rotation: Int, val outline: FloatArray,
        /** The exact piece within the outline (x, y, 0..1), when the outline may hold several touching pieces. */
        val marker: FloatArray? = null,
    )
    /** A pair from the box map: where each piece was seen, and their box cells. */
    private class ShownPair(val a: Thumb, val b: Thumb, val colA: Int, val rowA: Int, val colB: Int, val rowB: Int, val votes: Int)
    private val thumbs = java.util.concurrent.ConcurrentHashMap<Int, Thumb>()

    private fun rememberThumbs(image: Mat, pass: puzzlemaster.core.ScanSession.Pass) {
        for (s in pass.sightings) {
            try {
                val p = s.piece
                val r = (p.scale * 0.95).toInt()
                val x0 = (p.centroidX - r).toInt().coerceIn(0, image.cols() - 2); val y0 = (p.centroidY - r).toInt().coerceIn(0, image.rows() - 2)
                val w = minOf(2 * r, image.cols() - x0); val h = minOf(2 * r, image.rows() - y0)
                if (w < 8 || h < 8) continue
                val small = Mat(); Imgproc.resize(image.submat(org.opencv.core.Rect(x0, y0, w, h)), small, org.opencv.core.Size(THUMB.toDouble(), THUMB.toDouble()))
                val rgba = Mat(); Imgproc.cvtColor(small, rgba, Imgproc.COLOR_BGR2RGBA)
                val bmp = android.graphics.Bitmap.createBitmap(THUMB, THUMB, android.graphics.Bitmap.Config.ARGB_8888)
                org.opencv.android.Utils.matToBitmap(rgba, bmp)
                val kx = THUMB.toFloat() / w; val ky = THUMB.toFloat() / h
                val sides = Array(4) { android.graphics.PointF() }
                for (i in 0 until 4) sides[s.map[i]] = android.graphics.PointF(((p.edges[i].midX - x0) * kx).toFloat(), ((p.edges[i].midY - y0) * ky).toFloat())
                // Outline in the upright photo (the analysed image may have been squared up).
                val m = pass.back
                val step = maxOf(1, p.xs.size / 80)
                val o = FloatArray(((p.xs.size + step - 1) / step) * 2)
                var q = 0; var t = 0
                while (t < p.xs.size) {
                    var x = p.xs[t]; var y = p.ys[t]
                    if (m != null) { val wz = m[6] * x + m[7] * y + m[8]; val nx = (m[0] * x + m[1] * y + m[2]) / wz; y = (m[3] * x + m[4] * y + m[5]) / wz; x = nx }
                    o[q++] = (x / pass.frameW).toFloat(); o[q++] = (y / pass.frameH).toFloat(); t += step
                }
                val shot = currentShot
                thumbs[s.entry.id] = Thumb(bmp, sides, shot?.name ?: "", shot?.rotation ?: 0, o)
            } catch (e: Throwable) { Log.w(TAG, "thumb failed", e) }
        }
    }

    /** Opens the "where is it" view for a found pair: the table photo(s) the two pieces were last seen in. */
    private fun showWhere(pairIndex: Int) {
        val p = solved.getOrNull(pairIndex) ?: return
        val a = p.a; val b = p.b
        Thread {
            fun load(t: Thumb): android.graphics.Bitmap? {
                val f = File(sessionDir, t.photo + ".jpg")
                if (t.photo.isEmpty() || !f.exists()) return null
                val bmp = android.graphics.BitmapFactory.decodeFile(f.path, android.graphics.BitmapFactory.Options().apply { inSampleSize = 4 }) ?: return null
                if (t.rotation == 0) return bmp
                return android.graphics.Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, android.graphics.Matrix().apply { postRotate(t.rotation.toFloat()) }, true)
            }
            val panels = ArrayList<WherePanel>()
            // A ring around the exact piece (the outline may hold touching pieces).
            fun marks(t: Thumb): List<FloatArray> {
                val m = t.marker ?: return listOf(t.outline)
                return listOf(t.outline, FloatArray(64) { k -> val ang = (k / 2) * Math.PI * 2 / 32; if (k % 2 == 0) (m[0] + 0.045 * Math.cos(ang)).toFloat() else (m[1] + 0.034 * Math.sin(ang)).toFloat() })
            }
            if (a.photo == b.photo) load(a)?.let { panels.add(WherePanel(it, marks(a) + marks(b))) }
            else { load(a)?.let { panels.add(WherePanel(it, marks(a))) }; load(b)?.let { panels.add(WherePanel(it, marks(b))) } }
            runOnUiThread {
                if (panels.isEmpty()) Toast.makeText(this, "The photos for this pair are no longer stored.", Toast.LENGTH_SHORT).show()
                else overlay.where = WhereView("Pair ${pairIndex + 1}: where the pieces were last seen. Tap to close.", panels, pairIndex)
            }
        }.start()
    }

    /** Pictures of the found pairs (best fits first), for the strip at the top of the screen. */
    private fun publishPairs() {
        overlay.pairThumbs = emptyList()
    }
    /** Every photo used is kept here with its sensor data, so sessions can be replayed on a computer. */
    private val sessionDir by lazy { File(getExternalFilesDir(null), "session").apply { mkdirs() } }
    private val catalogueFile by lazy { File(filesDir, "catalogue.bin") }
    @Volatile private var ready = false
    @Volatile private var resetRequested = false
    private var solved: List<ShownPair> = emptyList()
    /** False until the saved list has been loaded, so restoring it at start-up doesn't sound like a find. */
    @Volatile private var pairsAnnounceReady = false
    /** Every piece-sized outline (clean or not) placed on the box by its print; pairs come from here. */
    private var boxMap: puzzlemaster.core.BoxMap? = null
    private val boxMapFile by lazy { File(filesDir, "boxmap.bin") }

    /** Places this photo's outlines on the box (worker thread, inside session.process). */
    private fun mapPieces(image: Mat, pass: puzzlemaster.core.ScanSession.Pass) {
        val bm = boxMap ?: return
        val shot = currentShot ?: return
        val m = pass.back
        try {
            val tm = System.currentTimeMillis()
            val found = bm.locate(shot.name, image, pass.out.pieces.map { it.xs to it.ys } + pass.out.loose, pass.pieceSize, pass.out.pieces.size, shot.rotation) { x0, y0 ->
                var x = x0; var y = y0
                if (m != null) { val wz = m[6] * x + m[7] * y + m[8]; val nx = (m[0] * x + m[1] * y + m[2]) / wz; y = (m[3] * x + m[4] * y + m[5]) / wz; x = nx }
                org.opencv.core.Point(x / pass.frameW, y / pass.frameH)
            }
            bm.addAll(found)
            Log.i(TAG, "MAP ${found.size} placed, ${System.currentTimeMillis() - tm} ms (${pass.out.pieces.size + pass.out.loose.size} outlines): ${bm.lastTiming}")
        } catch (e: Throwable) { Log.w(TAG, "box map failed", e) }
    }
    /** Box-picture matcher (built at start-up from files/box_poster.png) and what it has found so far. */
    @Volatile private var placer: puzzlemaster.core.BoxPlacer? = null
    /** Placement per catalogue piece, with how many views it had when tried (a new view means another try). */
    private val hitCache = HashMap<Int, Pair<puzzlemaster.core.BoxPlacer.Hit?, Int>>()
    @Volatile private var boxPatches: List<android.graphics.Bitmap> = emptyList()
    @Volatile private var placedCount = 0
    private lateinit var pairsPanel: android.widget.ScrollView
    private lateinit var pairsList: LinearLayout
    private lateinit var pairsButton: Button
    private val contextCache = HashMap<String, android.graphics.Bitmap>()

    private val thumbsFile by lazy { File(filesDir, "thumbs.bin") }

    /** Saves the catalogue together with where each piece was last seen, so the Pairs list works after a restart. */
    private fun saveAll() {
        catalogue.save(catalogueFile)
        try { boxMap?.save(boxMapFile) } catch (e: Throwable) { Log.w(TAG, "saving box map failed", e) }
        try {
            val tmp = File(thumbsFile.path + ".tmp")
            java.io.DataOutputStream(java.io.BufferedOutputStream(tmp.outputStream())).use { o ->
                o.writeInt(catalogue.entries.size)
                for (e in catalogue.entries) {
                    val t = thumbs[e.id]
                    o.writeUTF(t?.photo ?: ""); o.writeInt(t?.rotation ?: 0)
                    val ol = t?.outline ?: FloatArray(0)
                    o.writeInt(ol.size); for (v in ol) o.writeFloat(v)
                }
            }
            tmp.renameTo(thumbsFile)
        } catch (e: Throwable) { Log.w(TAG, "saving piece locations failed", e) }
    }

    /** Counterpart of [saveAll]; entries are in the same order as in the catalogue file. */
    private fun loadThumbs() {
        if (!thumbsFile.exists()) return
        try {
            java.io.DataInputStream(java.io.BufferedInputStream(thumbsFile.inputStream())).use { i ->
                val n = i.readInt()
                if (n != catalogue.entries.size) { Log.w(TAG, "piece locations don't match the catalogue ($n vs ${catalogue.entries.size})"); return }
                for (e in catalogue.entries) {
                    val photo = i.readUTF(); val rot = i.readInt()
                    val ol = FloatArray(i.readInt()) { i.readFloat() }
                    if (photo.isNotEmpty() && ol.size >= 6) thumbs[e.id] = Thumb(null, null, photo, rot, ol)
                }
            }
        } catch (e: Throwable) { Log.w(TAG, "loading piece locations failed", e) }
    }

    /** Pairs = neighbouring box cells that were each seen with enough matched print. */
    private fun resolvePairs() {
        val pl = placer; val bm = boxMap
        if (pl == null || bm == null) { solved = emptyList(); return }
        // Photos add to the map from other threads: take a consistent view of it.
        val (grid, cells) = synchronized(bm) { val g = bm.cleanGrid(); g to bm.cells(g).filterValues { it.votes >= 6 } }
        fun thumbOf(c: puzzlemaster.core.BoxMap.Cell): Thumb {
            val sg = c.best
            val o = FloatArray(sg.xs.size * 2) { if (it % 2 == 0) sg.xs[it / 2].toFloat() else sg.ys[it / 2].toFloat() }
            val mk = bm.markerOf(sg, c.col, c.row, grid)
            return Thumb(null, null, sg.photo, sg.tag, o, floatArrayOf(mk.x.toFloat(), mk.y.toFloat()))
        }
        val pairs = ArrayList<ShownPair>()
        for ((k, c) in cells) for (d in listOf(1 to 0, 0 to 1)) {
            val o = cells[(k.first + d.first) to (k.second + d.second)] ?: continue
            if (bm.joined(c, o, grid) || bm.joined(o, c, grid)) continue
            pairs.add(ShownPair(thumbOf(c), thumbOf(o), c.col, c.row, o.col, o.row, minOf(c.votes, o.votes)))
        }
        pairs.sortByDescending { it.votes }
        val patches = pairs.map { pr ->
            val m = pl.patch(pr.colA, pr.rowA, pr.colB, pr.rowB, grid, 300)
            val rgba = Mat(); Imgproc.cvtColor(m, rgba, Imgproc.COLOR_BGR2RGBA)
            android.graphics.Bitmap.createBitmap(300, 300, android.graphics.Bitmap.Config.ARGB_8888).also { org.opencv.android.Utils.matToBitmap(rgba, it) }
        }
        val before = solved.size
        placedCount = cells.size
        boxPatches = patches
        solved = pairs
        if (pairs.size > before && before >= 0 && pairsAnnounceReady) announcePairs()
        pairsAnnounceReady = true
        Log.i(TAG, "BOXMAP sightings ${bm.all.size}, cells ${cells.size}, pairs ${pairs.size}")
        runOnUiThread { pairsButton.text = "Pairs (${pairs.size})"; if (pairsPanel.visibility == android.view.View.VISIBLE) rebuildPairsPanel() }
    }

    /**
     * Decodes only the part of a stored photo that is needed: [u0]..[v1] in 0..1 of the upright photo.
     * Returns the upright region and the upright photo size.
     */
    private fun regionUpright(t: Thumb, u0: Float, v0: Float, u1: Float, v1: Float, maxSide: Int): android.graphics.Bitmap? {
        val f = File(sessionDir, t.photo + ".jpg")
        if (t.photo.isEmpty() || !f.exists()) return null
        @Suppress("DEPRECATION") val dec = android.graphics.BitmapRegionDecoder.newInstance(f.path, false) ?: return null
        try {
            val w = dec.width; val h = dec.height
            fun raw(u: Float, v: Float): FloatArray = when (t.rotation) {
                90 -> floatArrayOf(v, 1 - u); 180 -> floatArrayOf(1 - u, 1 - v); 270 -> floatArrayOf(1 - v, u); else -> floatArrayOf(u, v)
            }
            val a = raw(u0, v0); val b = raw(u1, v1)
            val r = android.graphics.Rect((minOf(a[0], b[0]) * w).toInt().coerceIn(0, w - 1), (minOf(a[1], b[1]) * h).toInt().coerceIn(0, h - 1),
                (maxOf(a[0], b[0]) * w).toInt().coerceIn(1, w), (maxOf(a[1], b[1]) * h).toInt().coerceIn(1, h))
            if (r.width() < 4 || r.height() < 4) return null
            var sample = 1; while (maxOf(r.width(), r.height()) / (sample * 2) >= maxSide) sample *= 2
            val bmp = dec.decodeRegion(r, android.graphics.BitmapFactory.Options().apply { inSampleSize = sample }) ?: return null
            if (t.rotation == 0) return bmp
            return android.graphics.Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, android.graphics.Matrix().apply { postRotate(t.rotation.toFloat()) }, true).also { if (it !== bmp) bmp.recycle() }
        } finally { dec.recycle() }
    }

    /** Upright size of a stored photo (from its header only). */
    private fun uprightSize(t: Thumb): IntArray? {
        val f = File(sessionDir, t.photo + ".jpg"); if (!f.exists()) return null
        val o = android.graphics.BitmapFactory.Options().apply { inJustDecodeBounds = true }
        android.graphics.BitmapFactory.decodeFile(f.path, o)
        if (o.outWidth <= 0) return null
        return if (t.rotation % 180 == 0) intArrayOf(o.outWidth, o.outHeight) else intArrayOf(o.outHeight, o.outWidth)
    }

    /** A wide cut-out of the table photo around a piece (so it can be found by its surroundings), piece outlined. */
    private fun contextOf(t: Thumb, color: Int): android.graphics.Bitmap? {
        val key = t.photo + "/" + t.outline.contentHashCode() + "/" + t.marker.contentHashCode() + "/" + color
        synchronized(contextCache) { contextCache[key]?.let { return it } }
        val size = uprightSize(t) ?: return null
        val W = size[0].toFloat(); val H = size[1].toFloat()
        val n = t.outline.size / 2
        var cx = 0f; var cy = 0f
        for (i in 0 until n) { cx += t.outline[2 * i] * W; cy += t.outline[2 * i + 1] * H }
        cx /= n; cy /= n
        var r = 0f
        for (i in 0 until n) r = maxOf(r, kotlin.math.hypot(t.outline[2 * i] * W - cx, t.outline[2 * i + 1] * H - cy))
        val mk = t.marker
        if (mk != null) { r = minOf(r, 0.06f * maxOf(W, H)); cx = mk[0] * W; cy = mk[1] * H }
        val half = (r * 3.6f).coerceAtLeast(40f).coerceAtMost(minOf(W, H) / 2)
        val x0 = (cx - half).coerceIn(0f, W - 2 * half); val y0 = (cy - half).coerceIn(0f, H - 2 * half)
        val part = regionUpright(t, x0 / W, y0 / H, (x0 + 2 * half) / W, (y0 + 2 * half) / H, CONTEXT) ?: return null
        val out = android.graphics.Bitmap.createBitmap(CONTEXT, CONTEXT, android.graphics.Bitmap.Config.RGB_565)
        val c = android.graphics.Canvas(out)
        c.drawBitmap(part, null, android.graphics.Rect(0, 0, CONTEXT, CONTEXT), android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG))
        part.recycle()
        val k = CONTEXT / (2 * half)
        val path = android.graphics.Path()
        for (i in 0 until n) { val x = (t.outline[2 * i] * W - x0) * k; val y = (t.outline[2 * i + 1] * H - y0) * k; if (i == 0) path.moveTo(x, y) else path.lineTo(x, y) }
        path.close()
        val paint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply { style = android.graphics.Paint.Style.STROKE; strokeWidth = 3f; this.color = color }
        c.drawPath(path, paint)
        if (mk != null) {
            // Dashed ring around (not over) the exact piece: the outline can hold several touching pieces.
            paint.strokeWidth = 4f; paint.color = Color.WHITE
            paint.pathEffect = android.graphics.DashPathEffect(floatArrayOf(14f, 10f), 0f)
            c.drawCircle((mk[0] * W - x0) * k, (mk[1] * H - y0) * k, r * 1.5f * k, paint)
        }
        synchronized(contextCache) { if (contextCache.size > 60) contextCache.clear(); contextCache[key] = out }
        return out
    }

    /** Pairs marked done (by their box cells), kept across restarts; they move to the bottom section. */
    private fun pairKey(p: ShownPair) = "${p.colA},${p.rowA}-${p.colB},${p.rowB}"
    private val donePairs: MutableSet<String> by lazy { java.util.Collections.synchronizedSet(HashSet(prefs.getStringSet("donePairs", emptySet())!!)) }
    private fun saveDone() = prefs.edit().putStringSet("donePairs", HashSet(donePairs)).apply()
    private var showDone = false
    private val panelGen = java.util.concurrent.atomic.AtomicInteger()

    /**
     * Fills the Pairs list: pairs still to do first, each with both pieces in their surroundings and the
     * box patch; then a collapsible "Done" section. Previews are made on a background thread; a newer
     * refresh cancels an older one.
     */
    private fun rebuildPairsPanel() {
        val gen = panelGen.incrementAndGet()
        val pairs = solved; val patches = boxPatches
        val open = pairs.indices.filter { pairKey(pairs[it]) !in donePairs }
        val done = pairs.indices.filter { pairKey(pairs[it]) in donePairs }
        pairsList.removeAllViews()
        pairsList.addView(Button(this).apply {
            text = "Clear list (I moved the pieces)"
            setOnClickListener {
                android.app.AlertDialog.Builder(this@MainActivity)
                    .setTitle("Clear the list?")
                    .setMessage("This forgets all catalogued pieces and pairs, so the next sweep starts fresh with the pieces where they are now.")
                    .setPositiveButton("Clear") { _, _ -> resetRequested = true; donePairs.clear(); saveDone() }
                    .setNegativeButton("Cancel", null).show()
            }
        }, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { leftMargin = 16; topMargin = 8 })
        fun text(t: String, size: Float = 16f) = TextView(this).apply { text = t; setTextColor(Color.WHITE); textSize = size; setPadding(20, 18, 20, 8) }
        pairsList.addView(text(if (pairs.isEmpty()) "No pairs yet. Pieces placed on the box picture so far: $placedCount.\nPairs appear when two placed pieces are neighbours. Pieces with detailed print (blossom, buildings, trees) can be placed; plain sky cannot."
            else "${open.size} pair(s) to do · ${done.size} done · $placedCount pieces placed.\nCyan = piece A, orange = piece B; the dashed ring surrounds the exact piece. Tap the photos to zoom in.", 15f))
        val doneHeader = Button(this).apply { text = if (showDone) "Hide done (${done.size})" else "Show done (${done.size})"; setOnClickListener { showDone = !showDone; rebuildPairsPanel() } }
        val doneBox = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        val openBox = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        pairsList.addView(openBox)
        if (done.isNotEmpty()) { pairsList.addView(doneHeader, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { leftMargin = 16; topMargin = 24 }); pairsList.addView(doneBox) }
        val order = open.map { it to openBox } + (if (showDone) done.map { it to doneBox } else emptyList())
        Thread {
            for ((n, entry) in order.withIndex()) {
                if (panelGen.get() != gen) return@Thread
                val (i, box) = entry
                val p = pairs[i]
                val ca = contextOf(p.a, 0xFF3CE8FF.toInt()); val cb = contextOf(p.b, 0xFFFF7A1A.toInt())
                val patch = patches.getOrNull(i)
                val isDone = box === doneBox
                runOnUiThread {
                    if (panelGen.get() != gen) return@runOnUiThread
                    val head = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
                    head.addView(text("${if (isDone) "✓" else "${n + 1}."}  A: row ${p.rowA + 1}, col ${p.colA + 1}   B: row ${p.rowB + 1}, col ${p.colB + 1}", 15f), LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
                    head.addView(Button(this).apply {
                        text = if (isDone) "Undo" else "Done"
                        setOnClickListener { if (isDone) donePairs.remove(pairKey(p)) else donePairs.add(pairKey(p)); saveDone(); rebuildPairsPanel() }
                    }, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { rightMargin = 16 })
                    box.addView(head)
                    val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(12, 0, 12, 0) }
                    for (b in listOf(ca, cb)) {
                        val iv = android.widget.ImageView(this).apply { adjustViewBounds = true; setPadding(6, 6, 6, 6) }
                        if (b != null) iv.setImageBitmap(b)
                        row.addView(iv, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
                    }
                    row.setOnClickListener { openPair(p) }
                    box.addView(row)
                    if (ca == null || cb == null) box.addView(text("(a photo of this pair is no longer stored)", 13f))
                    if (patch != null && !isDone) {
                        val iv = android.widget.ImageView(this).apply { setImageBitmap(patch); adjustViewBounds = true; setPadding(18, 0, 18, 24) }
                        box.addView(iv, LinearLayout.LayoutParams(resources.displayMetrics.widthPixels / 2, WRAP_CONTENT))
                    }
                }
            }
        }.start()
    }

    /** Full-screen viewer over the Pairs list: the pair's photos, zoomable, with marks that can be hidden. */
    private fun openPair(p: ShownPair) {
        val panel = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setBackgroundColor(0xFA05080C.toInt()); isClickable = true; setPadding(0, 110, 0, 40) }
        val bar = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(16, 0, 16, 12) }
        val views = ArrayList<ZoomImageView>()
        val bitmaps = ArrayList<android.graphics.Bitmap>()
        fun close() { root.removeView(panel); bitmaps.forEach { it.recycle() } }
        bar.addView(Button(this).apply { text = "Close"; setOnClickListener { close() } }, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        bar.addView(Button(this).apply { text = "Marks off"; setOnClickListener { val on = !(views.firstOrNull()?.showMarks ?: true); views.forEach { it.showMarks = on }; text = if (on) "Marks off" else "Marks on" } }, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        bar.addView(Button(this).apply {
            text = if (pairKey(p) in donePairs) "Undo done" else "Done"
            setOnClickListener { if (pairKey(p) in donePairs) donePairs.remove(pairKey(p)) else donePairs.add(pairKey(p)); saveDone(); close(); rebuildPairsPanel() }
        }, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        panel.addView(bar)
        val photos = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        panel.addView(photos, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f))
        panel.addView(TextView(this).apply { text = "Pinch or double-tap to zoom, drag to look around."; textSize = 13f; setPadding(24, 8, 24, 0) })
        root.addView(panel, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
        Thread {
            fun mark(t: Thumb, color: Int): ZoomImageView.Mark {
                val n = t.outline.size / 2
                var cx = 0f; var cy = 0f; for (i in 0 until n) { cx += t.outline[2 * i]; cy += t.outline[2 * i + 1] }
                val m = t.marker
                return ZoomImageView.Mark(t.outline, m?.get(0) ?: (cx / n), m?.get(1) ?: (cy / n), 0.045f, color)
            }
            val groups = if (p.a.photo == p.b.photo) listOf(p.a to listOf(mark(p.a, 0xFF3CE8FF.toInt()), mark(p.b, 0xFFFF7A1A.toInt())))
                else listOf(p.a to listOf(mark(p.a, 0xFF3CE8FF.toInt())), p.b to listOf(mark(p.b, 0xFFFF7A1A.toInt())))
            for ((t, marks) in groups) {
                val bmp = regionUpright(t, 0f, 0f, 1f, 1f, 2048) ?: continue
                bitmaps.add(bmp)
                runOnUiThread {
                    val v = ZoomImageView(this, bmp, marks)
                    views.add(v)
                    photos.addView(v, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f).apply { setMargins(8, 8, 8, 8) })
                }
            }
            if (groups.none { File(sessionDir, it.first.photo + ".jpg").exists() })
                runOnUiThread { Toast.makeText(this, "The photos for this pair are no longer stored.", Toast.LENGTH_SHORT).show() }
        }.start()
    }
    private var solvedAtSize = -1
    private val announced = HashSet<Long>()
    private val tone = ToneGenerator(AudioManager.STREAM_ALARM, 90)
    private var torchOn = false
    private val zoomSteps = floatArrayOf(1f, 2f, 3f, 5f)
    private var zoomIdx = 0
    @Volatile private var gravity = floatArrayOf(0f, 0f, 9.8f)
    private var pass = 0
    private var imageCapture: androidx.camera.core.ImageCapture? = null
    @Volatile private var paused = false
    /** Consecutive passes in which every piece in view already had 3+ measurements. */
    private var doneStreak = 0

    private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else status.text = "Camera permission is required."
    }

    private val gyroListener = object : SensorEventListener {
        override fun onSensorChanged(e: SensorEvent) { gyroRate = kotlin.math.sqrt(e.values[0] * e.values[0] + e.values[1] * e.values[1] + e.values[2] * e.values[2]) }
        override fun onAccuracyChanged(s: Sensor?, a: Int) {}
    }

    private fun refreshStatus() = runOnUiThread {
        status.text = (if (capturing) "Taking photos: just sweep over the pieces, no need to hold still." else if (queue.size > 0) "Photos stopped. Processing what was taken…" else "Photos off. Tap Start photos, then sweep over the pieces.") + "\nPhotos: $taken taken · $kept sharp · $processed processed" +
            (if (queue.size > 0) " (${queue.size} waiting)" else "") + "\n" + line2
    }

    private val gravityListener = object : SensorEventListener {
        override fun onSensorChanged(e: SensorEvent) { gravity = floatArrayOf(e.values[0], e.values[1], e.values[2]) }
        override fun onAccuracyChanged(s: Sensor?, a: Int) {}
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        if (!OpenCVLoader.initLocal()) Log.e(TAG, "OpenCV failed to load")
        root = FrameLayout(this)
        previewView = PreviewView(this).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
        overlay = OverlayView(this)
        overlay.onPairTap = { showWhere(it) }
        status = TextView(this).apply {
            setTextColor(Color.WHITE); setBackgroundColor(0xCC05080C.toInt()); textSize = 14f
            setPadding(24, 48, 24, 16); text = "Starting…"
        }
        val light = Button(this).apply { text = "Light" }
        light.setOnClickListener { torchOn = !torchOn; camera?.cameraControl?.enableTorch(torchOn); light.text = if (torchOn) "Light ●" else "Light" }
        val zoom = Button(this).apply { text = "Zoom 1×" }
        zoom.setOnClickListener {
            zoomIdx = (zoomIdx + 1) % zoomSteps.size
            val info = camera?.cameraInfo?.zoomState?.value
            val z = zoomSteps[zoomIdx].coerceIn(info?.minZoomRatio ?: 1f, info?.maxZoomRatio ?: 1f)
            camera?.cameraControl?.setZoomRatio(z)
            zoom.text = "Zoom ${zoomSteps[zoomIdx].toInt()}×"
        }
        val stop = Button(this).apply { text = "Start photos" }
        stopButton = stop
        stop.setOnClickListener {
            capturing = !capturing
            syncStopButton()
            if (capturing) captureNext()
            refreshStatus()
        }
        pairsList = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        pairsPanel = android.widget.ScrollView(this).apply { setBackgroundColor(0xF205080C.toInt()); visibility = android.view.View.GONE; addView(pairsList) }
        pairsButton = Button(this).apply { text = "Pairs (0)" }
        pairsButton.setOnClickListener {
            if (pairsPanel.visibility == android.view.View.VISIBLE) pairsPanel.visibility = android.view.View.GONE
            else { pairsPanel.visibility = android.view.View.VISIBLE; rebuildPairsPanel() }
        }
        val help = Button(this).apply { text = "?" }
        help.setOnClickListener { showWizard(0) }
        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER
            setBackgroundColor(0xCC05080C.toInt()); setPadding(16, 12, 16, 48)
            addView(light, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
            addView(zoom, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
            addView(stop, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1.3f))
            addView(pairsButton, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1.2f))
            addView(help, LinearLayout.LayoutParams(0, WRAP_CONTENT, 0.5f))
        }
        root.addView(previewView, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
        root.addView(overlay, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
        root.addView(status, FrameLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT, Gravity.TOP))
        root.addView(pairsPanel, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT).apply { topMargin = 320; bottomMargin = 230 })
        root.addView(bar, FrameLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT, Gravity.BOTTOM))
        setContentView(root)
        // Tron styling for every button and text, including views added later (panels, wizard).
        root.viewTreeObserver.addOnGlobalLayoutListener { tronify(root) }
        // Launch animation on top of everything; removed when it ends or is tapped.
        lateinit var splash: SplashView
        val theme = playTheme()
        splash = SplashView(this, onDone = { root.post { root.removeView(splash) } }, onSkip = { stopTheme(theme) })
        root.post { root.addView(splash, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT)) }
        if (!prefs.getBoolean("wizardDone", false) || !File(filesDir, "box_poster.png").exists()) showWizard(0)
        worker.execute {
            val posterFile = File(filesDir, "box_poster.png")
            val poster = if (posterFile.exists()) Imgcodecs.imread(posterFile.path) else Mat()
            if (poster.empty()) Log.w(TAG, "no box picture on this phone: pairs are off")
            else { val t0 = System.currentTimeMillis(); placer = puzzlemaster.core.BoxPlacer(poster); Log.i(TAG, "box picture ready: ${placer!!.posterFeatures} features, ${System.currentTimeMillis() - t0} ms") }
            val ok = catalogue.load(catalogueFile)
            if (ok) loadThumbs()
            placer?.let { pl -> boxMap = puzzlemaster.core.BoxMap(pl).also { bm -> Log.i(TAG, "box map loaded=${bm.load(boxMapFile)} sightings ${bm.all.size}") } }
            Log.i(TAG, "catalogue loaded=$ok entries=${catalogue.entries.size}, last-seen photos known for ${thumbs.size}")
            resolvePairs(); publishPairs()
            cacheDir.listFiles { f -> f.name.startsWith("shot_") }?.forEach { it.delete() }
            ready = true
            refreshStatus()
            processLoop()
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        else permission.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera(resume: Boolean = true) {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val aspect = AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
            val previewBuilder = Preview.Builder()
                .setResolutionSelector(ResolutionSelector.Builder().setAspectRatioStrategy(aspect).build())
            // Full-resolution stills (the sensor's high-resolution mode when the phone offers it):
            // pieces are big enough to measure without zooming in.
            val captureBuilder = androidx.camera.core.ImageCapture.Builder()
                .setResolutionSelector(
                    ResolutionSelector.Builder().setAspectRatioStrategy(aspect)
                        .setAllowedResolutionMode(ResolutionSelector.PREFER_HIGHER_RESOLUTION_OVER_CAPTURE_RATE)
                        .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY).build(),
                )
                .setCaptureMode(androidx.camera.core.ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            val capture = captureBuilder.build()
            imageCapture = capture
            provider.unbindAll()
            val preview = previewBuilder.build().also { it.surfaceProvider = previewView.surfaceProvider }
            camera = provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture)
            val info = camera?.cameraInfo?.zoomState?.value
            camera?.cameraControl?.setZoomRatio(zoomSteps[zoomIdx].coerceIn(info?.minZoomRatio ?: 1f, info?.maxZoomRatio ?: 8f))
            if (torchOn) camera?.cameraControl?.enableTorch(true)
            Log.i(TAG, "camera bound, zoom=${zoomSteps[zoomIdx]}")
            if (resume) captureNext()
        }, ContextCompat.getMainExecutor(this))
    }

    /** One pass = one still. The next still is taken as soon as the previous one has been processed. */
    /** Shoots continuously. A photo is kept only if it is sharp; kept photos wait in a short queue. */
    /**
     * Plays assets/sounds/theme.m4a if it is in the build (it is kept out of the repository), or
     * returns null. The player releases itself when the sound ends.
     */
    private fun playTheme(): android.media.MediaPlayer? = try {
        assets.openFd("sounds/theme.m4a").use { fd ->
            android.media.MediaPlayer().apply {
                setAudioAttributes(android.media.AudioAttributes.Builder().setUsage(android.media.AudioAttributes.USAGE_MEDIA).setContentType(android.media.AudioAttributes.CONTENT_TYPE_MUSIC).build())
                setDataSource(fd.fileDescriptor, fd.startOffset, fd.length)
                setOnCompletionListener { it.release() }
                prepare(); start()
            }
        }
    } catch (e: Exception) { null }

    private fun stopTheme(p: android.media.MediaPlayer?) { try { p?.stop(); p?.release() } catch (_: Exception) {} }

    /** Theme for a newly found pair; falls back to a beep without the sound file. Never overlaps itself. */
    @Volatile private var pairSound: android.media.MediaPlayer? = null
    private fun announcePairs() = runOnUiThread {
        if (pairSound != null) return@runOnUiThread
        val p = playTheme()
        if (p == null) { tone.startTone(ToneGenerator.TONE_PROP_BEEP2, 250); return@runOnUiThread }
        pairSound = p
        p.setOnCompletionListener { it.release(); pairSound = null }
    }

    /** Neon look: outlined glowing buttons, monospace glowing text, dark panels. Applied once per view. */
    private fun tronify(v: android.view.View) {
        if (v.getTag(R.id.tronified) == true) { if (v is android.view.ViewGroup) for (i in 0 until v.childCount) tronify(v.getChildAt(i)); return }
        val orange = 0xFFFF7A1A.toInt()
        val mono = android.graphics.Typeface.create(android.graphics.Typeface.MONOSPACE, android.graphics.Typeface.BOLD)
        when (v) {
            is Button -> {
                v.background = android.graphics.drawable.GradientDrawable().apply { setColor(0x2AFF7A1A); setStroke(3, orange); cornerRadius = 10f }
                v.setTextColor(orange); v.typeface = mono; v.isAllCaps = true; v.letterSpacing = 0.08f
                v.setShadowLayer(8f, 0f, 0f, orange); v.stateListAnimator = null
                // Same height everywhere: one line, text shrinks to fit instead of wrapping.
                v.maxLines = 1; v.setPadding(8, 0, 8, 0)
                v.setAutoSizeTextTypeUniformWithConfiguration(9, 15, 1, android.util.TypedValue.COMPLEX_UNIT_SP)
                v.layoutParams?.let { it.height = (52 * resources.displayMetrics.density).toInt(); v.layoutParams = it }
            }
            is TextView -> {
                v.typeface = mono
                // Neon orange text, like the splash screen's name.
                v.setTextColor(orange)
                v.setShadowLayer(7f, 0f, 0f, orange)
            }
            is android.view.ViewGroup -> for (i in 0 until v.childCount) tronify(v.getChildAt(i))
        }
        v.setTag(R.id.tronified, true)
    }

    @Volatile private var referenceRequested = false
    private lateinit var root: FrameLayout
    private var wizardView: android.view.View? = null
    private val prefs by lazy { getSharedPreferences("pm", MODE_PRIVATE) }

    /** Step-by-step instructions; step 1 takes the reference photo of the box picture. */
    private fun showWizard(step: Int) {
        wizardView?.let { root.removeView(it) }
        val hasRef = File(filesDir, "box_poster.png").exists()
        val steps = listOf(
            "Welcome to Puzzler" to "Puzzler finds puzzle pieces that go together and shows you where they are on your table.\n\nIt takes three steps:\n1. A reference photo of the box picture.\n2. Sweeping the camera over your pieces.\n3. Reading the list of pairs.",
            "1. Reference photo" to "Lay the puzzle box flat with the picture facing up, in good light.\n\nHold the phone straight above it so the picture fills most of the screen, hold still, and tap \"Take reference photo\".\n\nThe app uses this picture to recognise every piece by its print and work out where it belongs." +
                (if (hasRef) "\n\nA reference photo is already set. You can keep it." else ""),
            "2. Sweep over the pieces" to "Point the camera at your pieces from about 20–30 cm, at any angle, and move slowly across the table, like a scanner. No need to hold still: blurry photos are thrown away.\n\nPass over the same area from a couple of different angles: some pieces are only recognised from a better view.\n\nPieces with detailed print (blossom, buildings, trees) are found first. Plain sky is hard.\n\nThe phone beeps when it finds new pairs.",
            "3. Pairs" to "Tap \"Pairs\" for the list. For each pair you see both pieces in their surroundings on the table: the dashed ring surrounds the exact piece, and the box patch shows where they go.\n\nTap a pair to zoom into its photos. Tap Done once you've placed it: it moves to the bottom.\n\nAfter you move pieces around, tap \"Clear list\" at the top of the list and sweep again.\n\nYou can open these instructions again with the ? button.",
        )
        val (title, body) = steps[step]
        val panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL; setBackgroundColor(0xF0101018.toInt()); setPadding(48, 140, 48, 80)
            isClickable = true
        }
        panel.addView(TextView(this).apply { text = title; textSize = 26f; setTextColor(0xFFFF7A1A.toInt()); setPadding(0, 0, 0, 30) })
        panel.addView(TextView(this).apply { text = body; textSize = 18f; setTextColor(Color.WHITE) }, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f))
        val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        fun btn(t: String, f: () -> Unit) = Button(this).apply { text = t; setOnClickListener { f() } }.also { buttons.addView(it, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)) }
        if (step > 0) btn("Back") { showWizard(step - 1) }
        if (step == 1) {
            btn("Take reference photo") { closeWizard(); takeReference() }
            if (hasRef) btn("Keep current") { showWizard(2) }
        } else if (step < steps.size - 1) btn("Next") { showWizard(step + 1) }
        else btn("Start") { closeWizard(); prefs.edit().putBoolean("wizardDone", true).apply() }
        panel.addView(buttons)
        wizardView = panel
        root.addView(panel, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
    }

    private fun closeWizard() { wizardView?.let { root.removeView(it) }; wizardView = null }

    private fun takeReference() {
        status.text = "Hold the phone above the box picture… taking the reference photo"
        capturing = true
        referenceRequested = true
        captureNext()
    }

    /** Lets the user check the detected corners of the box picture, then builds the reference. */
    private fun onReferencePhoto(bytes: ByteArray, rotation: Int) {
        capturing = false
        var bmp = android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size, android.graphics.BitmapFactory.Options().apply { inSampleSize = 4 }) ?: return
        if (rotation != 0) bmp = android.graphics.Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, android.graphics.Matrix().apply { postRotate(rotation.toFloat()) }, true)
        val small = Mat(); org.opencv.android.Utils.bitmapToMat(bmp, small); Imgproc.cvtColor(small, small, Imgproc.COLOR_RGBA2BGR)
        val found = puzzlemaster.core.BoxLocator.detectCorners(small)
        val init = found?.map { android.graphics.PointF(it.x.toFloat(), it.y.toFloat()) } ?: listOf(
            android.graphics.PointF(bmp.width * 0.1f, bmp.height * 0.1f), android.graphics.PointF(bmp.width * 0.9f, bmp.height * 0.1f),
            android.graphics.PointF(bmp.width * 0.9f, bmp.height * 0.9f), android.graphics.PointF(bmp.width * 0.1f, bmp.height * 0.9f))
        val editor = CornerEditor(this, bmp, init)
        val panel = FrameLayout(this).apply { isClickable = true; setBackgroundColor(Color.BLACK) }
        panel.addView(editor, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT).apply { topMargin = 260; bottomMargin = 240 })
        panel.addView(TextView(this).apply {
            text = "Drag the four cyan circles onto the corners of the picture (not the box edge), then tap Use this photo."
            textSize = 17f; setTextColor(Color.WHITE); setPadding(32, 90, 32, 0)
        }, FrameLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT, Gravity.TOP))
        val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(16, 0, 16, 60) }
        buttons.addView(Button(this).apply { text = "Retake"; setOnClickListener { root.removeView(panel); takeReference() } }, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        buttons.addView(Button(this).apply { text = "Use this photo"; setOnClickListener {
            val k = 4.0 // the editor shows a quarter-size decode
            val corners = editor.corners.map { org.opencv.core.Point(it.x * k, it.y * k) }
            root.removeView(panel)
            status.text = "Preparing the reference picture…"
            worker.execute { buildReference(bytes, rotation, corners) }
        } }, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        panel.addView(buttons, FrameLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT, Gravity.BOTTOM))
        root.addView(panel, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
    }

    /** Worker thread: squares up the box picture, saves it, and restarts placement with it. */
    private fun buildReference(bytes: ByteArray, rotation: Int, corners: List<org.opencv.core.Point>) {
        try {
            var full = Imgcodecs.imdecode(org.opencv.core.MatOfByte(*bytes), Imgcodecs.IMREAD_COLOR)
            when (rotation) {
                90 -> Core.rotate(full, full, Core.ROTATE_90_CLOCKWISE)
                180 -> Core.rotate(full, full, Core.ROTATE_180)
                270 -> Core.rotate(full, full, Core.ROTATE_90_COUNTERCLOCKWISE)
            }
            val poster = puzzlemaster.core.BoxLocator.fromPhoto(full, corners, cell = 64).poster
            full.release()
            Imgcodecs.imwrite(File(filesDir, "box_poster.png").path, poster)
            File(filesDir, "reference.jpg").writeBytes(bytes)
            val pl = puzzlemaster.core.BoxPlacer(poster)
            placer = pl
            boxMap = puzzlemaster.core.BoxMap(pl); boxMapFile.delete()
            resolvePairs()
            Log.i(TAG, "REFERENCE set: ${poster.cols()}x${poster.rows()}, ${pl.posterFeatures} features")
            runOnUiThread { capturing = false; syncStopButton(); refreshStatus(); showWizard(2) }
        } catch (e: Throwable) {
            Log.e(TAG, "reference failed", e)
            runOnUiThread { Toast.makeText(this, "Could not use that photo: ${e.message}", Toast.LENGTH_LONG).show(); showWizard(1) }
        }
    }

    private fun captureNext() {
        val capture = imageCapture ?: return
        if (isFinishing || isDestroyed || paused || !capturing) return
        capture.takePicture(captureExecutor, object : androidx.camera.core.ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                try {
                    val g = gravity.copyOf(); val turning = gyroRate
                    val buf = image.planes[0].buffer
                    val bytes = ByteArray(buf.remaining()).also { buf.get(it) }
                    val rot = image.imageInfo.rotationDegrees
                    image.close()
                    if (referenceRequested) { referenceRequested = false; runOnUiThread { onReferencePhoto(bytes, rot) }; return }
                    taken++
                    // Sharpness: variance of the Laplacian on a quarter-size grayscale decode (fast).
                    val small = Imgcodecs.imdecode(org.opencv.core.MatOfByte(*bytes), Imgcodecs.IMREAD_REDUCED_GRAYSCALE_4)
                    val lap = Mat(); Imgproc.Laplacian(small, lap, CvType.CV_64F)
                    val mean = org.opencv.core.MatOfDouble(); val sd = org.opencv.core.MatOfDouble()
                    Core.meanStdDev(lap, mean, sd)
                    val sharp = sd.get(0, 0)[0] * sd.get(0, 0)[0]
                    small.release(); lap.release()
                    val median = synchronized(recentSharpness) {
                        recentSharpness.addLast(sharp); if (recentSharpness.size > 15) recentSharpness.removeFirst()
                        recentSharpness.sorted()[recentSharpness.size / 2]
                    }
                    // Keep photos that are at least as sharp as most recent ones and not taken mid-swing.
                    val ok = sharp >= 0.75 * median && sharp >= 60 && turning < 1.2f
                    val now = System.currentTimeMillis()
                    var queued = false
                    if (ok && ready && now - lastKeptAt >= 500 && queue.remainingCapacity() > 0) {
                        val f = File(cacheDir, "shot_${shotNo++}.jpg")
                        f.writeBytes(bytes)
                        queued = queue.offerLast(Shot(f, rot, g, focalPx(1)))
                        if (queued) { kept++; lastKeptAt = now } else f.delete()
                    }
                    Log.i(TAG, "SHOT sharp %.0f median %.0f gyro %.2f sharpEnough=$ok queued=$queued waiting ${queue.size}".format(sharp, median, turning))
                    refreshStatus()
                } catch (e: Throwable) { Log.e(TAG, "capture handling failed", e) } finally {
                    try { image.close() } catch (_: Throwable) {}
                    captureNext()
                }
            }
            override fun onError(exc: androidx.camera.core.ImageCaptureException) {
                Log.w(TAG, "capture failed: ${exc.message}")
                previewView.postDelayed({ captureNext() }, 800)
            }
        })
    }

    /**
     * Forgets every catalogued piece and pair (for when the pieces have been moved), including photos
     * still waiting to be processed. Photos already stored are kept. Runs on the worker thread.
     */
    private fun clearEverything() {
        while (true) { val s = queue.pollFirst() ?: break; s.file.delete() }
        catalogue.clear(); boxMap?.clear(); boxMapFile.delete(); solved = emptyList(); announced.clear(); hitCache.clear(); thumbs.clear()
        boxPatches = emptyList(); placedCount = 0; contextCache.clear()
        catalogueFile.delete(); thumbsFile.delete()
        overlay.pairThumbs = emptyList(); overlay.live = null
        line2 = "List cleared. Sweep over the pieces to build a new one."
        Log.i(TAG, "CLEARED by user")
        runOnUiThread { pairsButton.text = "Pairs (0)"; if (pairsPanel.visibility == android.view.View.VISIBLE) rebuildPairsPanel(); refreshStatus() }
    }

    /** Works through the kept photos; when it has caught up, it tidies the catalogue and solves. */
    /** Photos analysed at once (each needs a few hundred MB while in flight). */
    private val parallel = 3
    private val pool = Executors.newFixedThreadPool(parallel)
    private val inFlight = java.util.concurrent.atomic.AtomicInteger()
    private val sinceSolve = java.util.concurrent.atomic.AtomicInteger()

    /** Catalogue maintenance and pairs, while no photo is being committed. */
    private fun housekeeping() = session.exclusive {
        val t0 = System.currentTimeMillis()
        val merged = catalogue.mergeDuplicates()
        resolvePairs(); publishPairs(); saveAll()
        sinceSolve.set(0)
        Log.i(TAG, "TIDY merged ${merged.size} duplicates, catalogue ${catalogue.entries.size}, pairs ${solved.size}, ${System.currentTimeMillis() - t0} ms")
    }

    /** Hands kept photos to [parallel] worker threads; tidies up when it has caught up. */
    private fun processLoop() {
        while (!isDestroyed) {
            if (resetRequested && inFlight.get() == 0) { resetRequested = false; session.exclusive { clearEverything() }; sinceSolve.set(0) }
            if (inFlight.get() >= parallel || resetRequested) { Thread.sleep(50); continue }
            val shot = try { queue.pollFirst(400, java.util.concurrent.TimeUnit.MILLISECONDS) } catch (e: InterruptedException) { return }
            if (shot == null) {
                if (sinceSolve.get() > 0 && inFlight.get() == 0 && ready) {
                    housekeeping()
                    line2 = "Catalogue: ${catalogue.entries.size} pieces · ${solved.size} pair(s) found."
                    refreshStatus()
                }
                continue
            }
            inFlight.incrementAndGet()
            pool.execute {
                try { onStill(shot); sinceSolve.incrementAndGet() } catch (e: Throwable) { Log.e(TAG, "pass failed", e) } finally {
                    archive(shot)
                    synchronized(this) { processed++ }
                    inFlight.decrementAndGet()
                }
                if (sinceSolve.get() >= 10) housekeeping()
                refreshStatus()
            }
        }
    }


    /** Moves a processed photo into the session folder with its sensor data (up to [MAX_SESSION_PHOTOS]). */
    private fun archive(shot: Shot) {
        try {
            // Keep the newest photos: when full, the oldest one goes.
            val old = sessionDir.list()?.filter { it.endsWith(".jpg") }?.sorted() ?: emptyList()
            if (old.size >= MAX_SESSION_PHOTOS) for (o in old.take(old.size - MAX_SESSION_PHOTOS + 1)) { File(sessionDir, o).delete(); File(sessionDir, o.removeSuffix(".jpg") + ".txt").delete() }
            val name = shot.name
            val dst = File(sessionDir, "$name.jpg")
            if (!shot.file.renameTo(dst)) { shot.file.copyTo(dst, true); shot.file.delete() }
            File(sessionDir, "$name.txt").writeText("rotation ${shot.rotation}\ngravity ${shot.gravity[0]} ${shot.gravity[1]} ${shot.gravity[2]}\nfocalPerLongSide ${shot.focalPerLongSide}\n")
        } catch (e: Throwable) { Log.w(TAG, "archive failed", e); shot.file.delete() }
    }

    /** Focal length in pixels for an image whose long side is [longSide] px, including zoom. */
    private fun focalPx(longSide: Int): Double {
        var fmm = 4.7; var sensorMm = 6.4
        try {
            val c2 = Camera2CameraInfo.from(camera!!.cameraInfo)
            c2.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull()?.let { fmm = it.toDouble() }
            c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)?.let { sensorMm = maxOf(it.width, it.height).toDouble() }
        } catch (e: Throwable) { Log.w(TAG, "camera characteristics", e) }
        val zoom = camera?.cameraInfo?.zoomState?.value?.zoomRatio ?: 1f
        return fmm / sensorMm * longSide * zoom
    }

    private fun onStill(shot: Shot) {
        val t0 = System.currentTimeMillis()
        shotOnThread.set(shot)
        val r = session.process(shot.file.readBytes(), shot.rotation, shot.gravity, shot.focalPerLongSide)
        synchronized(this) { pass++ }
        val out = r.out; val sightings = r.sightings
        val byEntry = sightings.associateBy { it.entry.id }
        val seen = sightings.map { s -> SeenPiece(s.piece, s.entry.id, s.entry.sightings, out.border[s.piece.id] ?: emptyList()) }
        val pairs = ArrayList<SeenPair>()
        var fresh = false
        if (fresh) tone.startTone(ToneGenerator.TONE_PROP_BEEP2, 250)
        val ms = System.currentTimeMillis() - t0
        Log.i(TAG, "PASS ${r.frameW}x${r.frameH} tilt %.0f blobs ${out.blobs} clean ${out.pieces.size} measurable ${r.measurable} new ${sightings.count { it.isNew }} recognised ${sightings.count { !it.isNew }} catalogue ${catalogue.entries.size} solved ${solved.size} inView ${pairs.size} size %.0f | pipeline ${r.pipelineMs} observe ${r.observeMs} total $ms ms".format(r.tiltDeg, r.pieceSize))
        overlay.live = LiveView(seen, out.loose, pairs, r.back, r.frameW, r.frameH)
        val multi = catalogue.entries.count { it.sightings >= 2 }
        line2 = "Catalogue: ${catalogue.entries.size} pieces ($multi measured 2+ times) · ${solved.size} pair(s)" + when {
            solved.isNotEmpty() -> " · tap Pairs for the list"
            out.pieces.isNotEmpty() && r.measurable == 0 -> " · ➤ a little closer (pieces %.0f px, need 150)".format(r.pieceSize)
            else -> ""
        }
    }

    override fun onResume() {
        super.onResume()
        if (paused) { paused = false; captureNext() }
        val sm = getSystemService(SENSOR_SERVICE) as SensorManager
        sm.getDefaultSensor(Sensor.TYPE_GRAVITY)?.let { sm.registerListener(gravityListener, it, SensorManager.SENSOR_DELAY_GAME) }
        sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)?.let { sm.registerListener(gyroListener, it, SensorManager.SENSOR_DELAY_GAME) }
    }

    override fun onPause() {
        super.onPause()
        paused = true
        (getSystemService(SENSOR_SERVICE) as SensorManager).let { it.unregisterListener(gravityListener); it.unregisterListener(gyroListener) }
    }

    override fun onDestroy() { super.onDestroy(); worker.shutdownNow(); captureExecutor.shutdown() }

    companion object {
        private const val TAG = "PuzzleMaster"
        private const val MAX_SESSION_PHOTOS = 400
        private const val THUMB = 200
        private const val CONTEXT = 520
    }
}
