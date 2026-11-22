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
import puzzlemaster.core.LivePipeline
import puzzlemaster.core.Rectifier
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.hypot

/**
 * Always-on live scanning. Each pass: camera frame -> squared up with the gravity sensor ->
 * [LivePipeline] (outlines, edge/corner pieces, pairs that fit) -> drawn over the preview.
 * A pair is shown once two passes agree on it, and announced with a beep.
 */
class MainActivity : ComponentActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var status: TextView
    private var camera: Camera? = null
    private val worker = Executors.newSingleThreadExecutor()
    private val pipeline by lazy { LivePipeline() }
    private val tone = ToneGenerator(AudioManager.STREAM_ALARM, 90)
    private var torchOn = false
    private val zoomSteps = floatArrayOf(1f, 2f, 3f, 5f)
    private var zoomIdx = 0
    @Volatile private var gravity = floatArrayOf(0f, 0f, 9.8f)
    private var pass = 0

    /** Pairs seen over recent passes (rectified coordinates). */
    private class Tracked(var ax: Double, var ay: Double, var bx: Double, var by: Double, var hits: Int, var lastPass: Int, val index: Int)
    private val tracked = ArrayList<Tracked>()
    private var nextPairIndex = 0

    private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else status.text = "Camera permission is required."
    }

    private val gravityListener = object : SensorEventListener {
        override fun onSensorChanged(e: SensorEvent) { gravity = floatArrayOf(e.values[0], e.values[1], e.values[2]) }
        override fun onAccuracyChanged(s: Sensor?, a: Int) {}
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        if (!OpenCVLoader.initLocal()) Log.e(TAG, "OpenCV failed to load")
        val root = FrameLayout(this)
        previewView = PreviewView(this).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
        overlay = OverlayView(this)
        status = TextView(this).apply {
            setTextColor(Color.WHITE); setBackgroundColor(0x99000000.toInt()); textSize = 15f
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
            synchronized(tracked) { tracked.clear() }
        }
        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER
            setBackgroundColor(0x99000000.toInt()); setPadding(16, 8, 16, 48)
            addView(light, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
            addView(zoom, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        }
        root.addView(previewView, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
        root.addView(overlay, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
        root.addView(status, FrameLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT, Gravity.TOP))
        root.addView(bar, FrameLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT, Gravity.BOTTOM))
        setContentView(root)
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        else permission.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val aspect = AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
            val preview = Preview.Builder()
                .setResolutionSelector(ResolutionSelector.Builder().setAspectRatioStrategy(aspect).build())
                .build().also { it.surfaceProvider = previewView.surfaceProvider }
            // Big frames: edge matching needs pieces a few hundred pixels across.
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(
                    ResolutionSelector.Builder().setAspectRatioStrategy(aspect)
                        .setAllowedResolutionMode(ResolutionSelector.PREFER_HIGHER_RESOLUTION_OVER_CAPTURE_RATE)
                        .setResolutionStrategy(ResolutionStrategy(Size(2560, 1920), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build(),
                )
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build().also { it.setAnalyzer(worker) { img -> try { onFrame(img) } catch (e: Throwable) { Log.e(TAG, "pass failed", e) } finally { img.close() } } }
            provider.unbindAll()
            camera = provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
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

    private fun onFrame(img: ImageProxy) {
        val t0 = System.currentTimeMillis()
        val plane = img.planes[0]
        val w = img.width; val h = img.height
        val rgba = Mat(h, w, CvType.CV_8UC4)
        val buf = plane.buffer; val rs = plane.rowStride
        if (rs == w * 4) { val all = ByteArray(w * h * 4); buf.position(0); buf.get(all, 0, minOf(all.size, buf.remaining())); rgba.put(0, 0, all) }
        else { val row = ByteArray(w * 4); for (y in 0 until h) { buf.position(y * rs); buf.get(row, 0, w * 4); rgba.put(y, 0, row) } }
        val bgr0 = Mat(); Imgproc.cvtColor(rgba, bgr0, Imgproc.COLOR_RGBA2BGR)
        val up = Mat()
        when (img.imageInfo.rotationDegrees) {
            90 -> Core.rotate(bgr0, up, Core.ROTATE_90_CLOCKWISE)
            180 -> Core.rotate(bgr0, up, Core.ROTATE_180)
            270 -> Core.rotate(bgr0, up, Core.ROTATE_90_COUNTERCLOCKWISE)
            else -> bgr0.copyTo(up)
        }
        val g = gravity.copyOf()
        val f = focalPx(maxOf(up.cols(), up.rows()))
        val view = Rectifier.squareUp(up, g, f)
        val out = pipeline.process(view.image, view.valid, LivePipeline.Lean(view.nadirX, view.nadirY, view.focal))
        pass++
        val size = if (out.pieces.isEmpty()) 0.0 else out.pieces.map { it.scale }.sorted()[out.pieces.size / 2]
        val shown = vote(out, size)
        val ms = System.currentTimeMillis() - t0
        Log.i(TAG, "PASS ${up.cols()}x${up.rows()} tilt %.0f blobs ${out.blobs} pieces ${out.pieces.size} border ${out.border.size} pairs ${out.pairs.size} shown ${shown.size} size %.0f | seg ${out.segMs} piece ${out.pieceMs} match ${out.matchMs} total $ms ms".format(view.tiltDeg, size))
        overlay.live = LiveView(out, shown, view.back, up.cols(), up.rows())
        val corners = out.border.values.count { it.size >= 2 }
        runOnUiThread {
            status.text = "${out.pieces.size + out.loose.size} pieces (${out.pieces.size} analysed) · ${out.border.size} edge pieces ($corners corner) · ${shown.size} pair(s) · %.1fs · tilt %.0f°".format(ms / 1000.0, view.tiltDeg) + "\n" + when {
                out.pieces.isEmpty() -> "No pieces recognised in view yet."
                size < 170 -> "Pieces are too small to pair (%.0f px, need 170+). Zoom in on a small group.".format(size)
                else -> "Orange = edge piece, red C = corner, coloured line = these two fit."
            }
        }
        if (pass % 2 == 1) saveDebug(up, g, f, out)
    }

    /** A pair is shown after 2 passes agree on it, and kept while it was seen in the last 4 passes. */
    private fun vote(out: LivePipeline.Out, size: Double): List<ShownPair> = synchronized(tracked) {
        val tol = maxOf(20.0, size * 0.35)
        var fresh = false
        for (p in out.pairs) {
            val t = tracked.firstOrNull {
                (hypot(it.ax - p.a.midX, it.ay - p.a.midY) < tol && hypot(it.bx - p.b.midX, it.by - p.b.midY) < tol) ||
                    (hypot(it.ax - p.b.midX, it.ay - p.b.midY) < tol && hypot(it.bx - p.a.midX, it.by - p.a.midY) < tol)
            }
            if (t == null) tracked.add(Tracked(p.a.midX, p.a.midY, p.b.midX, p.b.midY, 1, pass, nextPairIndex++))
            else {
                t.hits++; t.lastPass = pass
                if (t.hits == 2) fresh = true
            }
        }
        tracked.removeAll { pass - it.lastPass > 4 }
        if (fresh) tone.startTone(ToneGenerator.TONE_PROP_BEEP2, 250)
        tracked.filter { it.hits >= 2 }.map { ShownPair(it.ax, it.ay, it.bx, it.by, it.index) }
    }

    /** Latest raw frame + sensor data, so a pass can be replayed offline (adb pull .../files/live). */
    private fun saveDebug(up: Mat, g: FloatArray, f: Double, out: LivePipeline.Out) {
        val dir = File(getExternalFilesDir(null), "live").apply { mkdirs() }
        Imgcodecs.imwrite(File(dir, "frame.jpg").path, up, org.opencv.core.MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 95))
        File(dir, "frame.txt").writeText("gravity ${g[0]} ${g[1]} ${g[2]}\nfocal $f\npieces ${out.pieces.size} pairs ${out.pairs.size}\n")
    }

    override fun onResume() {
        super.onResume()
        val sm = getSystemService(SENSOR_SERVICE) as SensorManager
        sm.getDefaultSensor(Sensor.TYPE_GRAVITY)?.let { sm.registerListener(gravityListener, it, SensorManager.SENSOR_DELAY_UI) }
    }

    override fun onPause() {
        super.onPause()
        (getSystemService(SENSOR_SERVICE) as SensorManager).unregisterListener(gravityListener)
    }

    override fun onDestroy() { super.onDestroy(); worker.shutdown() }

    companion object { private const val TAG = "PuzzleMaster" }
}
