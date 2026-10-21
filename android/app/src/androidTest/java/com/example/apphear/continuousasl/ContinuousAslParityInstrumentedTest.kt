package com.example.apphear.continuousasl

import androidx.test.platform.app.InstrumentationRegistry
import com.example.apphear.BuildConfig
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/** Synthetic model/export parity only; no participant recordings or accuracy evaluation. */
class ContinuousAslParityInstrumentedTest {
    @Test fun syntheticCpuLogitsMatchExportAndCausalPrefixes() {
        assumeTrue(BuildConfig.CONTINUOUS_ASL_EXPERIMENT_ENABLED)
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        fun floats(name: String): FloatArray {
            val bytes = instrumentation.context.assets.open("continuous_asl/$name").use { it.readBytes() }
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            return FloatArray(buffer.remaining()).also { buffer.get(it) }
        }
        val flat = floats("synthetic_features.f32")
        val expected = floats("synthetic_logits.f32")
        val frames = (0 until 17).map { flat.copyOfRange(it*378, (it+1)*378) }
        ContinuousAslModel(instrumentation.targetContext).use { model ->
            val actual = model.logits(frames)
            for (t in actual.indices) for (c in actual[t].indices) {
                val e = expected[t*2289+c]
                assertTrue("Native ONNX parity at $t/$c", abs(actual[t][c]-e) <= 0.0001f+0.00003f*abs(e))
            }
            for (t in frames.indices) {
                val causal = model.logits(frames.take(t+1)).last()
                for (c in causal.indices) assertTrue(abs(causal[c]-actual[t][c]) < 0.0002f)
            }
        }
    }
}
