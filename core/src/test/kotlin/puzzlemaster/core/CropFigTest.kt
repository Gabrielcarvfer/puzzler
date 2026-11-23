package puzzlemaster.core
import org.junit.Test
import java.io.File
import javax.imageio.ImageIO
class CropFigTest {
    @Test fun crop() {
        val dir = figureDir() ?: return
        val f = File(dir, "hough.jpg").takeIf { it.exists() } ?: return
        val bi = ImageIO.read(f)
        ImageIO.write(bi.getSubimage(bi.width / 4, bi.height / 3, bi.width / 2, bi.height / 3), "jpg", File(dir, "hough_crop.jpg"))
    }
}
