package com.example.apphear.continuousasl

import com.example.apphear.data.settings.RecognitionModeSettings
import com.example.apphear.fingerspelling.RecognitionMode
import com.example.apphear.perceptionvnext.PhoneRawObservation
import org.junit.Assert.*
import org.junit.Test

class ContinuousAslCoreTest {
    private val words = listOf("<BLANK>", "<UNK>", "HELLO", "WORLD")
    private val policy = ContinuousAslPolicy(61, 100.0, 1000.0, 128, 4096)
    private fun observation(t: Double, visible: Boolean = true) = PhoneRawObservation(
        (t/10).toInt(), t, 480, 360,
        Array(2) { FloatArray(42) { 10f } }, Array(2) { BooleanArray(21) { visible } },
        Array(2) { FloatArray(21) { 1f } }, floatArrayOf(1f, 1f),
        booleanArrayOf(visible, visible), booleanArrayOf(visible, visible),
        Array(2) { floatArrayOf(20f, 20f, 15f) }, FloatArray(16) { 10f },
        FloatArray(8) { 1f }, BooleanArray(8) { true }, true)

    @Test fun ctcCollapsesBeforeRemovingBlank() {
        val d = ContinuousAslGreedyDecoder(4)
        assertEquals(listOf(2, 2, 3), listOf(2, 2, 0, 2, 3, 3, 0).mapNotNull(d::accept))
    }

    @Test fun buildGatesCannotRestoreUnsupportedModes() {
        assertEquals(listOf(RecognitionMode.WORD), RecognitionModeSettings.availableModes(false, false))
        assertEquals(RecognitionMode.WORD, RecognitionModeSettings.resolve("CONTINUOUS_WORD", true, false))
        assertEquals(RecognitionMode.CONTINUOUS_WORD, RecognitionModeSettings.resolve("CONTINUOUS_WORD", false, true))
        assertEquals(listOf(RecognitionMode.WORD, RecognitionMode.CONTINUOUS_WORD),
            RecognitionModeSettings.availableModes(false, true))
    }

    @Test fun noHandAbstainsAndNeverCallsModel() {
        var calls = 0
        val c = ContinuousAslCore(words, policy, { calls++; 2 })
        c.select(true); c.consume(observation(0.0, false)); c.consume(observation(100.0, false))
        assertEquals(0, calls); assertEquals("", c.snapshot.draft)
        assertEquals("", c.snapshot.confirmed)
    }

    @Test fun emitsImmediatelyAndBoundsCausalContextWithoutPadding() {
        val sizes = mutableListOf<Int>()
        val c = ContinuousAslCore(words, policy, { frames ->
            sizes.add(frames.size); assertEquals(378, frames.last().size); 2
        })
        c.select(true)
        repeat(100) { c.consume(observation(it*100.0)) }
        assertEquals(1, sizes.first()); assertEquals(61, sizes.last())
        assertEquals("HELLO", c.snapshot.draft)
        assertEquals("", c.snapshot.confirmed)
    }

    @Test fun realCadenceAndHandReentryDoNotInterpolate() {
        var calls = 0
        val c = ContinuousAslCore(words, policy, { calls++; 2 })
        c.select(true)
        c.consume(observation(0.0)); c.consume(observation(40.0)); c.consume(observation(100.0))
        assertEquals(2, calls); assertEquals("HELLO", c.snapshot.draft)
        c.consume(observation(110.0, false)); c.consume(observation(120.0))
        assertEquals(3, calls); assertEquals("HELLO HELLO", c.snapshot.draft)
    }

    @Test fun gapResetsTemporalStateAndDuplicateTimestampFailsClosed() {
        val sizes = mutableListOf<Int>()
        val c = ContinuousAslCore(words, policy, { sizes.add(it.size); 2 })
        c.select(true)
        c.consume(observation(0.0)); c.consume(observation(100.0)); c.consume(observation(1200.0))
        assertEquals(listOf(1, 2, 1), sizes)
        try { c.consume(observation(1200.0)); fail("Duplicate timestamp accepted") }
        catch (_: IllegalArgumentException) { }
        assertEquals(ContinuousAslPhase.ERROR, c.snapshot.phase)
        c.clear(); assertEquals("", c.snapshot.draft)
    }

    @Test fun reviewFreezesInputAndOnlyExplicitEditedTextIsConfirmed() {
        val c = ContinuousAslCore(words, policy, { 2 })
        c.select(true); c.consume(observation(0.0))
        val before = c.beginReview()!!
        c.consume(observation(100.0))
        assertSame(before, c.snapshot)
        assertFalse(c.confirm(before.revision-1, "stale"))
        assertTrue(c.confirm(before.revision, "Hello, edited by user"))
        assertEquals("Hello, edited by user", c.snapshot.confirmed)
        assertEquals("", c.snapshot.draft)
    }

    @Test fun modeExitAndPauseInvalidatePendingConfirmation() {
        val c = ContinuousAslCore(words, policy, { 2 })
        c.select(true); c.consume(observation(0.0))
        val before = c.beginReview()!!
        c.pause(); assertFalse(c.confirm(before.revision, "old"))
        c.select(false); assertEquals("", c.snapshot.draft)
        assertEquals("", c.snapshot.confirmed)
        c.consume(observation(100.0)); assertEquals("", c.snapshot.draft)
    }

    @Test fun fullDraftStopsRatherThanSilentlyTruncatingOrCommitting() {
        var token = 2
        val c = ContinuousAslCore(words, policy.copy(maximumDraftTokens = 2), { token.also { token = 5-it } })
        c.select(true)
        c.consume(observation(0.0)); c.consume(observation(100.0)); c.consume(observation(200.0))
        assertEquals(ContinuousAslPhase.FULL, c.snapshot.phase)
        assertEquals("HELLO WORLD", c.snapshot.draft); assertEquals("", c.snapshot.confirmed)
    }
}
