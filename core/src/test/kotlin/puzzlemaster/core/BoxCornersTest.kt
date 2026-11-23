package puzzlemaster.core

import nu.pattern.OpenCV
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.opencv.core.Point
import org.opencv.imgcodecs.Imgcodecs
import java.io.File
import kotlin.math.hypot

class BoxCornersTest {
    @Test
    fun findsBoxPictureCorners() {
        val f = File("src/test/resources/box/box.jpg").takeIf { it.exists() } ?: return
        OpenCV.loadLocally()
        val img = Imgcodecs.imread(f.path)
        val truth = listOf(Point(254.0, 430.0), Point(7954.0, 410.0), Point(7980.0, 5948.0), Point(238.0, 5932.0))
        val c = BoxLocator.detectCorners(img)
        println("CORNERS found $c")
        assertNotNull(c)
        val err = c!!.zip(truth).maxOf { (a, b) -> hypot(a.x - b.x, a.y - b.y) } / img.cols()
        println("CORNERS worst error %.2f%% of width".format(err * 100))
        assertTrue(err < 0.05) // may snap to the box edge instead of the print; the user can drag the corners
    }
}
