package puzzlemaster.core

import ai.onnxruntime.OrtEnvironment
import org.junit.Test
import java.io.File

class ModelInfoTest {
    @Test
    fun printModelSignatures() {
        val dir = File(System.getenv("PM_MODELS") ?: return)
        val env = OrtEnvironment.getEnvironment()
        for (f in listOf("mobile_sam_image_encoder.onnx", "sam_mask_decoder_multi.onnx")) {
            env.createSession(File(dir, f).path).use { s ->
                println(f)
                s.inputInfo.forEach { (k, v) -> println("  in  $k ${v.info}") }
                s.outputInfo.forEach { (k, v) -> println("  out $k ${v.info}") }
            }
        }
    }
}
