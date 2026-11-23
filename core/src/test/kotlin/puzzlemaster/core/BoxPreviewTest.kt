package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Test
import org.opencv.imgcodecs.Imgcodecs
import java.io.File

class BoxPreviewTest {
    @Test
    fun preview() {
        val f = File("src/test/resources/box/box.jpg").takeIf { it.exists() } ?: return
        OpenCV.loadLocally()
        val m = Imgcodecs.imread(f.path, Imgcodecs.IMREAD_REDUCED_COLOR_4)
        println("BOX ${m.cols()}x${m.rows()}")
        figureDir()?.let { Imgcodecs.imwrite(File(it, "box_preview.jpg").path, m) }
    }
}
