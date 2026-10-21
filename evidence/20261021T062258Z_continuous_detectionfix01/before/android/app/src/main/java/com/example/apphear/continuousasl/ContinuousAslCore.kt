package com.example.apphear.continuousasl

import com.example.apphear.perceptionvnext.PhoneRawObservation
import com.example.apphear.perceptionvnext.PhoneSegmentationFeatureBuilder

internal data class ContinuousAslPolicy(
    val context: Int, val minimumIntervalMs: Double, val maximumGapMs: Double,
    val maximumDraftTokens: Int, val maximumConfirmedCharacters: Int,
) {
    init {
        require(context == 61 && minimumIntervalMs > 0 && maximumGapMs > minimumIntervalMs)
        require(maximumDraftTokens > 0 && maximumConfirmedCharacters > 0)
    }
}

internal enum class ContinuousAslPhase { INACTIVE, LISTENING, EDITING, FULL, ERROR }
internal data class ContinuousAslSnapshot(
    val revision: Long = 0, val phase: ContinuousAslPhase = ContinuousAslPhase.INACTIVE,
    val draft: String = "", val english: String = "", val observations: Int = 0,
    val message: String = "Select Continuous word recognition in Settings",
    val translationStatus: String = "Limited offline English · experimental",
)

/** Standard greedy CTC, also used by parity tests. */
internal class ContinuousAslGreedyDecoder(private val classes: Int, private val blank: Int = 0) {
    private var previous = -1
    fun reset() { previous = -1 }
    fun accept(token: Int): Int? {
        require(token in 0 until classes) { "CONTINUOUS_ASL_TOKEN_RANGE" }
        val result = token.takeIf { it != previous && it != blank }
        previous = token
        return result
    }
}

/** Camera-executor confined. Automatic display only: no recordings, WORD/history commits or TTS. */
internal class ContinuousAslCore(
    private val vocabulary: List<String>, private val policy: ContinuousAslPolicy,
    private val predictLast: (List<FloatArray>) -> Int,
    private val outputPolicy: ContinuousAslOutputPolicy,
    private val translator: ContinuousAslEnglish,
    private val publish: (ContinuousAslSnapshot) -> Unit = {},
) {
    private val features = PhoneSegmentationFeatureBuilder()
    private val decoder = ContinuousAslStableDecoder(vocabulary.size, outputPolicy.stableTokenMs)
    private val context = ArrayDeque<FloatArray>()
    private val draft = mutableListOf<String>()
    private val segment = mutableListOf<String>()
    private val completed = mutableListOf<ContinuousAslEnglishResult>()
    private var lastSource: Double? = null
    private var lastSelected: Double? = null
    private var lastNonblank: Double? = null
    private var active = false
    @Volatile var snapshot = ContinuousAslSnapshot(); private set

    init { require(vocabulary.size > 2 && vocabulary.take(2) == listOf("<BLANK>", "<UNK>")) }
    fun policyTextLimit() = policy.maximumConfirmedCharacters

    fun select(enabled: Boolean) {
        if (enabled == active) return
        active = enabled
        draft.clear(); segment.clear(); completed.clear(); lastNonblank = null
        resetTemporal()
        emit(if (active) ContinuousAslPhase.LISTENING else ContinuousAslPhase.INACTIVE,
            if (active) "Sign to display text automatically. Editing is optional." else "Continuous mode off",
            observations = 0)
    }

    fun pause() {
        finishSegment(); resetTemporal()
        if (active) emit(if (isFull()) ContinuousAslPhase.FULL else ContinuousAslPhase.LISTENING,
            "Paused; reopen recognition to continue")
    }

    private fun resetTemporal() {
        context.clear(); features.reset(); decoder.reset()
        lastSelected = null; lastSource = null
    }

    fun consume(observation: PhoneRawObservation) {
        if (!active || snapshot.phase != ContinuousAslPhase.LISTENING) return
        try {
            val timestamp = observation.timestampMs
            val old = lastSource
            require(timestamp.isFinite() && (old == null || timestamp > old)) { "CONTINUOUS_ASL_NON_CHRONOLOGICAL_TIMESTAMP" }
            if (old != null && timestamp-old > policy.maximumGapMs) {
                finishSegment(); resetTemporal()
            }
            lastSource = timestamp
            val visible = (0..1).any { observation.handObserved[it] && observation.handJointValid[it].any { valid -> valid } }
            if (!visible) {
                // Close only the already observed CTC run, then abstain. This is not a no-sign classifier.
                decoder.finish()?.let(::append)
                finishIfIdle(timestamp)
                resetTemporal(); lastSource = timestamp
                emit(if (isFull()) ContinuousAslPhase.FULL else ContinuousAslPhase.LISTENING,
                    "Show your hands to continue · displayed text is experimental")
                return
            }
            if (lastSelected?.let { timestamp-it < policy.minimumIntervalMs } == true) return
            lastSelected = timestamp
            context.addLast(features.derive(observation))
            if (context.size > policy.context) context.removeFirst()
            val predicted = predictLast(context.toList())
            if (predicted != 0) lastNonblank = timestamp
            decoder.accept(predicted, timestamp)?.let(::append)
            finishIfIdle(timestamp)
            emit(if (isFull()) ContinuousAslPhase.FULL else ContinuousAslPhase.LISTENING,
                if (isFull()) "Output limit reached. Edit or clear to continue." else "Automatic text · experimental recognition",
                observations = snapshot.observations+1)
        } catch (failure: Exception) {
            resetTemporal()
            emit(ContinuousAslPhase.ERROR, "Continuous recognition unavailable. Clear to retry or select WORD.")
            throw failure
        }
    }

    fun clear() {
        draft.clear(); segment.clear(); completed.clear(); lastNonblank = null; resetTemporal()
        emit(if (active) ContinuousAslPhase.LISTENING else ContinuousAslPhase.INACTIVE,
            "Text cleared", observations = 0)
    }

    fun beginEdit(): ContinuousAslSnapshot? {
        if (!active || snapshot.english.isBlank() || snapshot.phase == ContinuousAslPhase.ERROR ||
            snapshot.phase == ContinuousAslPhase.EDITING) return null
        resetTemporal()
        emit(ContinuousAslPhase.EDITING, "Editing text · recognition paused")
        return snapshot
    }

    fun cancelEdit(expected: Long) {
        if (!active || snapshot.phase != ContinuousAslPhase.EDITING || snapshot.revision != expected) return
        resetTemporal(); finishSegment()
        emit(if (isFull()) ContinuousAslPhase.FULL else ContinuousAslPhase.LISTENING,
            "Automatic text · editing optional")
    }

    fun applyEdit(expected: Long, edited: String): Boolean {
        if (!active || snapshot.phase != ContinuousAslPhase.EDITING || snapshot.revision != expected) return false
        val text = edited.trim()
        if (text.length > policy.maximumConfirmedCharacters) return false
        draft.clear(); segment.clear(); completed.clear(); lastNonblank = null; resetTemporal()
        if (text.isNotBlank()) completed.add(ContinuousAslEnglishResult(text, ContinuousAslEnglishKind.PHRASE, 0))
        emit(if (isFull()) ContinuousAslPhase.FULL else ContinuousAslPhase.LISTENING, "Text edited · continue signing")
        return true
    }

    private fun append(token: Int) {
        draft.add(vocabulary[token]); segment.add(vocabulary[token])
    }
    private fun finishIfIdle(timestamp: Double) {
        if (lastNonblank?.let { timestamp-it >= outputPolicy.segmentIdleMs } == true) finishSegment()
    }
    private fun finishSegment() {
        if (segment.isNotEmpty()) completed.add(translator.render(segment.toList()))
        segment.clear(); lastNonblank = null
    }
    private fun parts() = completed + if (segment.isEmpty()) emptyList() else listOf(translator.render(segment.toList()))
    // Stop after a complete literal token; never cut an annotation label in half to fit a limit.
    private fun isFull() = draft.size >= policy.maximumDraftTokens ||
        parts().joinToString(" ") { it.text }.length >= policy.maximumConfirmedCharacters

    private fun emit(phase: ContinuousAslPhase, message: String, observations: Int = snapshot.observations) {
        val rendered = parts()
        val untranslated = rendered.sumOf { it.untranslated }
        val status = when {
            untranslated > 0 -> "Partial English · $untranslated untranslated labels shown in brackets"
            rendered.any { it.kind == ContinuousAslEnglishKind.WORD_EQUIVALENTS } ->
                "English word equivalents · ASL word order retained"
            else -> "Limited offline English rules · not linguistically validated"
        }
        snapshot = ContinuousAslSnapshot(snapshot.revision+1, phase, draft.joinToString(" "),
            rendered.joinToString(" ") { it.text }, observations, message, status)
        publish(snapshot)
    }
}