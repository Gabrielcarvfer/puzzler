import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins { id("org.jetbrains.kotlin.jvm") }

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}
kotlin { compilerOptions { jvmTarget.set(JvmTarget.JVM_17) } }

dependencies {
    // ONNX Runtime: same ai.onnxruntime API on JVM (tests) and Android (onnxruntime-android in :app).
    compileOnly("com.microsoft.onnxruntime:onnxruntime:1.20.0")
    testImplementation("org.openpnp:opencv:4.9.0-0")
    // OpenCV Java API: desktop jar here, the Android AAR in :app provides it at runtime.
    compileOnly("org.openpnp:opencv:4.9.0-0")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.20.0")
    testImplementation("junit:junit:4.13.2")
    // OpenCV (desktop, natives bundled) for mosaic/mapping experiments in tests.
    testImplementation("org.openpnp:opencv:4.9.0-0")
}

tasks.test {
    maxHeapSize = "3g"
    testLogging { showStandardStreams = true; events("passed", "failed") }
    // Test figures (overlays of what the detector/matcher saw) are written here.
    val figures = rootProject.layout.projectDirectory.dir("test").asFile
    doFirst { figures.mkdirs() }
    environment("PM_DEBUG", figures.absolutePath)
    environment("PM_MODELS", rootProject.layout.projectDirectory.dir("models").asFile.absolutePath)
}
