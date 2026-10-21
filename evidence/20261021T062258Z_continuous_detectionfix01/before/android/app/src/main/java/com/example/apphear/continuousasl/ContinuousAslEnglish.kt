package com.example.apphear.continuousasl

import java.util.Locale

internal data class ContinuousAslOutputPolicy(val stableTokenMs: Double, val segmentIdleMs: Double) {
    init { require(stableTokenMs > 0 && segmentIdleMs > stableTokenMs) }
}

/** Delays publication, never changes CTC token order or suppresses a short nonblank peak.
 * A run is settled when it ends, or after the configured hold. A blank rearms repeats.
 * This is a display policy, not a calibrated sign/no-sign or confidence decision. */
internal class ContinuousAslStableDecoder(private val classes: Int, private val holdMs: Double) {
    private var previous = 0
    private var started = 0.0
    private var emitted = true
    init { require(classes > 2 && holdMs > 0) }
    fun reset() { previous = 0; emitted = true }
    fun accept(token: Int, timestamp: Double): Int? {
        require(token in 0 until classes && timestamp.isFinite()) { "CONTINUOUS_ASL_TOKEN_RANGE" }
        if (token != previous) {
            val completed = finish()
            previous = token; started = timestamp; emitted = token == 0
            return completed
        }
        return if (!emitted && timestamp - started >= holdMs) finish() else null
    }
    fun finish(): Int? = if (emitted) null else previous.also { emitted = true }
}

internal enum class ContinuousAslEnglishKind { PHRASE, WORD_EQUIVALENTS, PARTIAL }
internal data class ContinuousAslEnglishResult(
    val text: String, val kind: ContinuousAslEnglishKind, val untranslated: Int,
)

/** Exact, local engineering rules. No inferred referents, universal ASL reorder rule,
 * casing-based pseudo-translation, web request, or claims of qualified linguistic review.
 * An unsupported literal label stays visible, in order, as [untranslated: LABEL]. */
internal class ContinuousAslEnglish(
    private val words: Map<String, String>, private val phrases: Map<List<String>, String>,
) {
    init {
        require(words.isNotEmpty())
        require(words.all { (label, english) -> label.isNotBlank() && english.isNotBlank() && !label.startsWith("<") })
        require(phrases.all { (labels, english) -> labels.isNotEmpty() && labels.all { it.isNotBlank() && !it.startsWith("<") } && english.isNotBlank() })
    }
    fun render(labels: List<String>): ContinuousAslEnglishResult {
        require(labels.none { it == "<BLANK>" }) { "CONTINUOUS_ASL_BLANK_IN_TEXT" }
        if (labels.isEmpty()) return ContinuousAslEnglishResult("", ContinuousAslEnglishKind.WORD_EQUIVALENTS, 0)
        phrases[labels]?.let { return ContinuousAslEnglishResult(it, ContinuousAslEnglishKind.PHRASE, 0) }
        var missing = 0
        val text = labels.joinToString(" ") { label -> words[label] ?: "[untranslated: $label]".also { missing++ } }
        val capitalized = text.replaceFirstChar { if (it.isLowerCase()) it.titlecase(Locale.ENGLISH) else it.toString() }
        return ContinuousAslEnglishResult(capitalized,
            if (missing == 0) ContinuousAslEnglishKind.WORD_EQUIVALENTS else ContinuousAslEnglishKind.PARTIAL, missing)
    }
}
