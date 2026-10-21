package com.example.apphear.continuousasl

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import com.example.apphear.perceptionvnext.PhoneSegmentationFeatureBuilder
import org.json.JSONArray
import org.json.JSONObject
import java.nio.FloatBuffer
import java.security.MessageDigest

/** Load on IO; infer/close on the serialized camera owner. Uses the existing CPU ORT dependency. */
internal class ContinuousAslModel(context: Context) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val options: OrtSession.SessionOptions
    private val session: OrtSession
    val vocabulary: List<String>
    val policy: ContinuousAslPolicy
    private val classes: Int

    init {
        val manifest = JSONObject(context.assets.open("continuous_asl/manifest.json").bufferedReader().use { it.readText() })
        require(manifest.getString("schema_version") == "apphear.continuous_asl.android_research.v1")
        require(manifest.getString("feature_schema") == PhoneSegmentationFeatureBuilder.SCHEMA)
        require(manifest.getInt("feature_width") == PhoneSegmentationFeatureBuilder.FEATURE_WIDTH)
        require(manifest.getInt("blank_id") == 0 && manifest.getInt("unknown_id") == 1)
        require(!manifest.getBoolean("automatic_commit"))
        require(manifest.getString("input_name") == "features" && manifest.getString("output_name") == "logits")
        require(manifest.getString("source_checkpoint_sha256") ==
            "14d0e4669674265ea2be8a3161fa6699e4f813daf4b97b384863dce1daa038ff")
        classes = manifest.getInt("classes")
        val vocabBytes = context.assets.open("continuous_asl/vocabulary.json").use { it.readBytes() }
        require(digest(vocabBytes) == manifest.getString("vocabulary_sha256"))
        val tokens = JSONArray(String(vocabBytes, Charsets.UTF_8))
        vocabulary = (0 until tokens.length()).map { tokens.getString(it) }
        require(vocabulary.size == classes && vocabulary.distinct().size == classes)
        require(vocabulary.take(2) == listOf("<BLANK>", "<UNK>"))
        policy = ContinuousAslPolicy(manifest.getInt("context_observations"),
            manifest.getDouble("minimum_interval_ms"), manifest.getDouble("maximum_gap_ms"),
            manifest.getInt("maximum_draft_tokens"), manifest.getInt("maximum_confirmed_characters"))
        val bytes = context.assets.open("continuous_asl/causal_gloss_ctc_fp32.onnx").use { it.readBytes() }
        require(digest(bytes) == manifest.getString("model_sha256"))
        options = OrtSession.SessionOptions()
        try {
            options.setIntraOpNumThreads(manifest.getInt("threads"))
            options.setInterOpNumThreads(1)
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            session = environment.createSession(bytes, options)
            try {
                val input = session.inputInfo["features"]?.info as? TensorInfo
                val output = session.outputInfo["logits"]?.info as? TensorInfo
                require(input?.shape?.contentEquals(longArrayOf(1, -1, 378)) == true)
                require(output?.shape?.contentEquals(longArrayOf(1, -1, classes.toLong())) == true)
            } catch (failure: Throwable) { session.close(); throw failure }
        } catch (failure: Throwable) { options.close(); throw failure }
    }

    fun logits(frames: List<FloatArray>): Array<FloatArray> {
        require(frames.isNotEmpty() && frames.size <= policy.context)
        require(frames.all { it.size == 378 && it.all(Float::isFinite) })
        val flat = FloatArray(frames.size*378)
        frames.forEachIndexed { i, frame -> frame.copyInto(flat, i*378) }
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(flat), longArrayOf(1, frames.size.toLong(), 378)).use { input ->
            session.run(mapOf("features" to input)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val values = (result[0].value as Array<Array<FloatArray>>)[0]
                require(values.size == frames.size && values.all { it.size == classes && it.all(Float::isFinite) })
                return values.map { it.copyOf() }.toTypedArray()
            }
        }
    }

    fun predictLast(frames: List<FloatArray>): Int {
        val values = logits(frames).last()
        var best = 0
        for (i in 1 until values.size) if (values[i] > values[best]) best = i
        return best
    }

    override fun close() { session.close(); options.close() }

    private fun digest(bytes: ByteArray) = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it) }
}
