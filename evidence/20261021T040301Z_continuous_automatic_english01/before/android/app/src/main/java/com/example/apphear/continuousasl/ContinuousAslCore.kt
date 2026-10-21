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

internal enum class ContinuousAslPhase { INACTIVE, LISTENING, REVIEW, FULL, ERROR }

internal data class ContinuousAslSnapshot(
    val revision: Long = 0, val phase: ContinuousAslPhase = ContinuousAslPhase.INACTIVE,
    val draft: String = "", val confirmed: String = "", val observations: Int = 0,
    val message: String = "Select Continuous word recognition in Settings",
)

/** CTC repeats are collapsed before blank removal, including across inference calls. */
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

/** Camera-executor confined. No recordings, interpolation, auto-commit or language conversion. */
internal class ContinuousAslCore(
    private val vocabulary: List<String>, private val policy: ContinuousAslPolicy,
    private val predictLast: (List<FloatArray>) -> Int,
    private val publish: (ContinuousAslSnapshot) -> Unit = {},
) {
    private val features = PhoneSegmentationFeatureBuilder()
    private val decoder = ContinuousAslGreedyDecoder(vocabulary.size)
    private val context = ArrayDeque<FloatArray>()
    private val draft = mutableListOf<String>()
    private var lastSource: Double? = null
    private var lastSelected: Double? = null
    private var active = false
    @Volatile var snapshot = ContinuousAslSnapshot(); private set

    init { require(vocabulary.size > 2 && vocabulary.take(2) == listOf("<BLANK>", "<UNK>")) }
    fun policyTextLimit() = policy.maximumConfirmedCharacters

    fun select(enabled: Boolean) {
        if (enabled == active) return
        active = enabled
        draft.clear()
        resetTemporal()
        emit(if (active) ContinuousAslPhase.LISTENING else ContinuousAslPhase.INACTIVE,
            if (active) "Show your hands and sign. Review the draft before use." else "Continuous mode off",
            confirmed = "", observations = 0)
    }

    fun pause() {
        resetTemporal()
        if (active) emit(ContinuousAslPhase.LISTENING, "Paused; reopen recognition to continue")
    }

    fun resetTemporal() {
        context.clear(); features.reset(); decoder.reset()
        lastSelected = null; lastSource = null
    }

    fun consume(observation: PhoneRawObservation) {
        if (!active || snapshot.phase != ContinuousAslPhase.LISTENING) return
        try {
            val timestamp = observation.timestampMs
            val old = lastSource
            require(old == null || timestamp > old) { "CONTINUOUS_ASL_NON_CHRONOLOGICAL_TIMESTAMP" }
            if (old != null && timestamp-old > policy.maximumGapMs) resetTemporal()
            lastSource = timestamp
            val visible = (0..1).any { observation.handObserved[it] && observation.handJointValid[it].any { valid -> valid } }
            if (!visible) {
                // Positive-only training supplies no calibrated no-sign decision. With no
                // observed hand we abstain explicitly and break CTC/feature continuity.
                resetTemporal(); lastSource = timestamp
                emit(ContinuousAslPhase.LISTENING, "Show your hands to continue")
                return
            }
            if (lastSelected?.let { timestamp-it < policy.minimumIntervalMs } == true) return
            lastSelected = timestamp
            context.addLast(features.derive(observation))
            if (context.size > policy.context) context.removeFirst()
            val token = decoder.accept(predictLast(context.toList()))
            if (token != null) draft.add(vocabulary[token])
            val full = draft.size >= policy.maximumDraftTokens
            emit(if (full) ContinuousAslPhase.FULL else ContinuousAslPhase.LISTENING,
                if (full) "Draft full. Review or clear to continue." else "Experimental draft • review before use",
                observations = snapshot.observations+1)
        } catch (failure: Exception) {
            resetTemporal()
            emit(ContinuousAslPhase.ERROR, "Continuous recognition unavailable. Clear to retry or select WORD.")
            throw failure
        }
    }

    fun clear() {
        draft.clear(); resetTemporal()
        emit(if (active) ContinuousAslPhase.LISTENING else ContinuousAslPhase.INACTIVE,
            "Draft cleared", confirmed = "", observations = 0)
    }

    fun beginReview(): ContinuousAslSnapshot? {
        if (!active || draft.isEmpty() || snapshot.phase == ContinuousAslPhase.ERROR) return null
        resetTemporal()
        emit(ContinuousAslPhase.REVIEW, "Review and edit the experimental draft")
        return snapshot
    }

    fun cancelReview(expected: Long) {
        if (!active || snapshot.phase != ContinuousAslPhase.REVIEW || snapshot.revision != expected) return
        resetTemporal()
        emit(if (draft.size >= policy.maximumDraftTokens) ContinuousAslPhase.FULL else ContinuousAslPhase.LISTENING,
            "Draft kept. Review before use.")
    }

    fun confirm(expected: Long, edited: String): Boolean {
        if (!active || snapshot.phase != ContinuousAslPhase.REVIEW || snapshot.revision != expected) return false
        val text = edited.trim()
        val confirmed = listOf(snapshot.confirmed, text).filter { it.isNotBlank() }.joinToString(" ")
        if (text.isEmpty() || confirmed.length > policy.maximumConfirmedCharacters) return false
        draft.clear(); resetTemporal()
        emit(ContinuousAslPhase.LISTENING, "Edited text confirmed", confirmed = confirmed)
        return true
    }

    private fun emit(phase: ContinuousAslPhase, message: String,
                     confirmed: String = snapshot.confirmed, observations: Int = snapshot.observations) {
        snapshot = ContinuousAslSnapshot(snapshot.revision+1, phase, draft.joinToString(" "),
            confirmed, observations, message)
        publish(snapshot)
    }
}
