package puzzlemaster.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import org.junit.Test
import java.io.File
import java.nio.FloatBuffer

class DecoderTimingTest {
    @Test
    fun timeDecoderVariants() {
        val dir = File(System.getenv("PM_MODELS") ?: return)
        if (System.getenv("PM_TIMING") == null && !File(dir, ".timing").exists()) return
        val env = OrtEnvironment.getEnvironment()
        val opts = ai.onnxruntime.OrtSession.SessionOptions().apply { setIntraOpNumThreads(1) }
        for (model in listOf("sam_mask_decoder_multi.onnx", "sam_mask_decoder_single.onnx")) env.createSession(File(dir, model).path, opts).use { dec ->
            for (orig in listOf(1024f, 256f)) {
                val emb = OnnxTensor.createTensor(env, FloatBuffer.allocate(256 * 64 * 64), longArrayOf(1, 256, 64, 64))
                val maskIn = OnnxTensor.createTensor(env, FloatBuffer.allocate(256 * 256), longArrayOf(1, 1, 256, 256))
                val has = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(0f)), longArrayOf(1))
                val size = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(orig, orig)), longArrayOf(2))
                val pc = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(500f, 500f, 0f, 0f)), longArrayOf(1, 2, 2))
                val pl = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(1f, -1f)), longArrayOf(1, 2))
                val inputs = mapOf("image_embeddings" to emb, "point_coords" to pc, "point_labels" to pl, "mask_input" to maskIn, "has_mask_input" to has, "orig_im_size" to size)
                repeat(3) { dec.run(inputs).close() }
                val t0 = System.nanoTime()
                repeat(20) { dec.run(inputs).close() }
                println("TIMING $model orig=$orig: %.1f ms/call".format((System.nanoTime() - t0) / 20e6))
            }
        }
    }
}
