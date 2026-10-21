package com.example.apphear.continuousasl

import com.example.apphear.data.settings.RecognitionModeSettings
import com.example.apphear.fingerspelling.RecognitionMode
import com.example.apphear.perceptionvnext.PhoneRawObservation
import org.junit.Assert.*
import org.junit.Test
import kotlin.random.Random

class ContinuousAslCoreTest {
    private val words = listOf("<BLANK>", "<UNK>", "IX-1p", "WANT", "WATER", "NOT", "KNOW", "IX-3p:j")
    private val policy = ContinuousAslPolicy(61, 100.0, 1000.0, 128, 4096)
    private val output = ContinuousAslOutputPolicy(200.0, 1200.0)
    private val english = ContinuousAslEnglish(mapOf("IX-1p" to "I", "WANT" to "want", "WATER" to "water", "NOT" to "not", "KNOW" to "know"),
        mapOf(listOf("IX-1p", "WANT", "WATER") to "I want water.", listOf("IX-1p", "NOT", "KNOW") to "I do not know."))
    private fun core(p: ContinuousAslPolicy = policy, predict: (List<FloatArray>) -> Int) =
        ContinuousAslCore(words, p, predict, output, english)
    private fun start(c: ContinuousAslCore) { c.select(true); c.resume() }
    private fun observation(t: Double, visible: Boolean = true) = PhoneRawObservation(
        (t/10).toInt(), t, 480, 360,
        Array(2) { FloatArray(42) { 10f } }, Array(2) { BooleanArray(21) { visible } },
        Array(2) { FloatArray(21) { 1f } }, floatArrayOf(1f, 1f),
        booleanArrayOf(visible, visible), booleanArrayOf(visible, visible),
        Array(2) { floatArrayOf(20f, 20f, 15f) }, FloatArray(16) { 10f },
        FloatArray(8) { 1f }, BooleanArray(8) { true }, true)

    @Test fun delayedPublicationPreservesGreedyCtcIncludingShortPeaksAndBlankRepeats() {
        val random = Random(17)
        repeat(500) {
            val stream = List(100) { random.nextInt(words.size) }
            val greedy = ContinuousAslGreedyDecoder(words.size)
            val stable = ContinuousAslStableDecoder(words.size, 200.0)
            val expected = stream.mapNotNull(greedy::accept)
            val actual = stream.mapIndexedNotNull { i, token -> stable.accept(token, i*100.0) }.toMutableList()
            stable.finish()?.let(actual::add)
            assertEquals(expected, actual)
        }
    }

    @Test fun buildGatesCannotRestoreUnsupportedModes() {
        assertEquals(listOf(RecognitionMode.WORD), RecognitionModeSettings.availableModes(false, false))
        assertEquals(RecognitionMode.WORD, RecognitionModeSettings.resolve("CONTINUOUS_WORD", true, false))
        assertEquals(RecognitionMode.CONTINUOUS_WORD, RecognitionModeSettings.resolve("CONTINUOUS_WORD", false, true))
    }

    @Test fun englishAppearsAutomaticallyWithoutEditingOrConfirmation() {
        val stream = ArrayDeque(listOf(2, 3, 4, 0))
        val c = core { stream.removeFirst() }
        start(c)
        repeat(4) { c.consume(observation(it*100.0)) }
        assertEquals("IX-1p WANT WATER", c.snapshot.draft)
        assertEquals("I want water.", c.snapshot.english)
        assertEquals(ContinuousAslPhase.LISTENING, c.snapshot.phase)
    }

    @Test fun sustainedRunSettlesOnceWithoutWaitingForFullReceptiveField() {
        val sizes = mutableListOf<Int>()
        val c = core { sizes.add(it.size); 2 }
        start(c); c.consume(observation(0.0)); c.consume(observation(100.0))
        assertEquals("", c.snapshot.english)
        c.consume(observation(200.0)); assertEquals("I", c.snapshot.english)
        (3 until 100).forEach { c.consume(observation(it*100.0)) }
        assertEquals(1, sizes.first()); assertEquals(61, sizes.last())
        assertEquals("IX-1p", c.snapshot.draft)
    }

    @Test fun noHandAbstainsAndNeverCallsModel() {
        var calls = 0
        val c = core { calls++; 2 }
        start(c); c.consume(observation(0.0, false)); c.consume(observation(100.0, false))
        assertEquals(0, calls); assertEquals("", c.snapshot.draft); assertEquals("", c.snapshot.english)
    }

    @Test fun cadenceAndBriefHandLossCannotPublishUnsettledTokens() {
        var calls = 0
        val c = core { calls++; 2 }
        start(c); c.consume(observation(0.0)); c.consume(observation(40.0)); c.consume(observation(100.0))
        assertEquals(2, calls); assertEquals("", c.snapshot.draft)
        c.consume(observation(110.0, false)); assertEquals("", c.snapshot.draft)
        c.consume(observation(120.0)); c.consume(observation(220.0, false))
        assertEquals(3, calls); assertEquals("", c.snapshot.draft); assertEquals("", c.snapshot.english)
        repeat(3) { c.consume(observation(300.0+it*100.0)) }
        assertEquals(6, calls); assertEquals("IX-1p", c.snapshot.draft)
    }

    @Test fun blankSeparatesRealRepeatWhileSustainedNonblankDoesNotRepeat() {
        val stream = ArrayDeque(listOf(4, 4, 4, 4, 0, 4, 0))
        val c = core { stream.removeFirst() }
        start(c); repeat(7) { c.consume(observation(it*100.0)) }
        assertEquals("WATER WATER", c.snapshot.draft)
        assertEquals("Water water", c.snapshot.english)
    }

    @Test fun gapDropsUnsettledRunAndDuplicateTimestampFailsClosed() {
        val sizes = mutableListOf<Int>()
        val c = core { sizes.add(it.size); 2 }
        start(c); c.consume(observation(0.0)); c.consume(observation(100.0)); c.consume(observation(1200.0))
        assertEquals(listOf(1, 2, 1), sizes); assertEquals("", c.snapshot.draft)
        try { c.consume(observation(1200.0)); fail("Duplicate timestamp accepted") } catch (_: IllegalArgumentException) { }
        assertEquals(ContinuousAslPhase.ERROR, c.snapshot.phase)
        c.clear(); assertEquals(ContinuousAslPhase.LISTENING, c.snapshot.phase)
    }

    @Test fun idleFinalizesDisplaySegmentWithoutRewritingItWithNextSegment() {
        var token = 0
        val c = core { token }
        start(c)
        listOf(2, 3, 4, 0).forEachIndexed { i, t -> token = t; c.consume(observation(i*100.0)) }
        (4..15).forEach { c.consume(observation(it*100.0)) }
        token = 3; c.consume(observation(1600.0)); token = 0; c.consume(observation(1700.0))
        assertEquals("I want water. Want", c.snapshot.english)
    }

    @Test fun optionalEditingFreezesInputAndReplacesRatherThanDuplicatesText() {
        val c = core { 2 }
        start(c); repeat(3) { c.consume(observation(it*100.0)) }
        val before = c.beginEdit()!!
        c.consume(observation(300.0)); assertSame(before, c.snapshot)
        assertFalse(c.applyEdit(before.revision-1, "stale"))
        assertTrue(c.applyEdit(before.revision, "Edited English"))
        assertEquals("Edited English", c.snapshot.english); assertEquals("", c.snapshot.draft)
        repeat(3) { c.consume(observation(400.0+it*100.0)) }
        assertEquals("Edited English I", c.snapshot.english)
    }

    @Test fun canceledEditAndPauseKeepTextButInvalidateStaleEdit() {
        val c = core { 2 }
        start(c); repeat(3) { c.consume(observation(it*100.0)) }
        val before = c.beginEdit()!!
        c.cancelEdit(before.revision); assertEquals("I", c.snapshot.english)
        assertFalse(c.applyEdit(before.revision, "stale"))
        val second = c.beginEdit()!!; c.pause(); assertFalse(c.applyEdit(second.revision, "stale"))
        c.select(false); assertEquals("", c.snapshot.english); assertEquals("", c.snapshot.draft)
    }

    @Test fun unknownAndSpatialLabelsAreNeverErasedOrTurnedIntoInventedEnglish() {
        val stream = ArrayDeque(listOf(1, 7, 0))
        val c = core { stream.removeFirst() }
        start(c); repeat(3) { c.consume(observation(it*100.0)) }
        assertEquals("[untranslated: <UNK>] [untranslated: IX-3p:j]", c.snapshot.english)
        assertTrue(c.snapshot.translationStatus.contains("2 untranslated"))
    }

    @Test fun fullOutputStopsModelAndCanBeClearedWithoutAnEdit() {
        var calls = 0
        var token = 2
        val c = core(policy.copy(maximumDraftTokens = 2)) { calls++; token.also { token = if (token == 2) 3 else 2 } }
        start(c); repeat(6) { c.consume(observation(it*100.0)) }
        assertEquals(3, calls); assertEquals(ContinuousAslPhase.FULL, c.snapshot.phase)
        assertEquals("IX-1p WANT", c.snapshot.draft)
        c.clear(); assertEquals("", c.snapshot.english); assertEquals(ContinuousAslPhase.LISTENING, c.snapshot.phase)
    }

    @Test fun overlongEditIsRejectedAndEmptyEditCanClearText() {
        val c = core(policy.copy(maximumConfirmedCharacters = 10)) { 2 }
        start(c); repeat(3) { c.consume(observation(it*100.0)) }
        val before = c.beginEdit()!!
        assertFalse(c.applyEdit(before.revision, "This is too long"))
        assertEquals("I", c.snapshot.english)
        assertTrue(c.applyEdit(before.revision, "")); assertEquals("", c.snapshot.english)
    }

    @Test fun selectingOwnershipDoesNotResumeAnUnstartedActivity() {
        var calls = 0
        val c = core { calls++; 2 }
        c.select(true)
        repeat(3) { c.consume(observation(it*100.0)) }
        assertEquals(0, calls); assertEquals("", c.snapshot.draft)
        c.resume(); c.consume(observation(300.0))
        assertEquals(1, calls)
    }

    @Test fun missingHandsKeepAlreadySettledTextButDiscardAnotherPendingRun() {
        var calls = 0
        val c = core { calls++; 2 }
        start(c); repeat(3) { c.consume(observation(it*100.0)) }
        assertEquals("IX-1p", c.snapshot.draft)
        c.consume(observation(300.0, false)); c.consume(observation(400.0))
        c.consume(observation(410.0, false)); c.consume(observation(500.0, false))
        assertEquals(4, calls)
        assertEquals("IX-1p", c.snapshot.draft); assertEquals("I", c.snapshot.english)
    }

    @Test fun shortCtcPeaksStillPublishOnObservedTransitionsAndBlankRearmer() {
        val stream = ArrayDeque(listOf(2, 0, 2, 3, 0))
        val c = core { stream.removeFirst() }
        start(c); repeat(5) { c.consume(observation(it*100.0)) }
        assertEquals("IX-1p IX-1p WANT", c.snapshot.draft)
        assertEquals("I I want", c.snapshot.english)
    }

    @Test fun pauseBlocksLateCallbacksAndExplicitResumeStartsFreshContext() {
        val sizes = mutableListOf<Int>()
        val c = core { sizes.add(it.size); 2 }
        start(c); repeat(3) { c.consume(observation(it*100.0)) }
        c.pause()
        val paused = c.snapshot
        assertNull(c.beginEdit())
        repeat(3) { c.consume(observation(300.0+it*100.0)) }
        assertEquals(listOf(1, 2, 3), sizes); assertSame(paused, c.snapshot)
        c.select(true); c.consume(observation(600.0))
        assertSame(paused, c.snapshot); assertEquals(3, sizes.size)
        c.resume(); c.consume(observation(700.0))
        assertEquals(listOf(1, 2, 3, 1), sizes)
        assertEquals("I", c.snapshot.english)
    }

    @Test fun resumeCannotEnableAnInactiveRecognitionOwner() {
        var calls = 0
        val c = core { calls++; 2 }
        start(c); c.consume(observation(0.0)); c.select(false); c.resume()
        c.consume(observation(100.0))
        assertEquals(1, calls); assertEquals(ContinuousAslPhase.INACTIVE, c.snapshot.phase)
        assertEquals("", c.snapshot.english)
    }

    @Test fun errorSurvivesPauseAndResumeUntilExplicitClear() {
        var calls = 0
        val c = core { calls++; 2 }
        start(c); c.consume(observation(0.0))
        try { c.consume(observation(0.0)); fail("Duplicate timestamp accepted") } catch (_: IllegalArgumentException) { }
        assertEquals(ContinuousAslPhase.ERROR, c.snapshot.phase)
        val errorMessage = c.snapshot.message
        c.pause(); assertEquals(ContinuousAslPhase.ERROR, c.snapshot.phase)
        c.resume(); c.consume(observation(100.0))
        assertEquals(ContinuousAslPhase.ERROR, c.snapshot.phase)
        assertEquals(errorMessage, c.snapshot.message); assertEquals(1, calls)
        c.clear(); c.consume(observation(200.0)); assertEquals(2, calls)
    }

    @Test fun fullOutputSurvivesPauseResumeAndClearWhilePausedDoesNotResume() {
        var calls = 0
        var token = 2
        val c = core(policy.copy(maximumDraftTokens = 2)) { calls++; token.also { token = if (token == 2) 3 else 2 } }
        start(c); repeat(3) { c.consume(observation(it*100.0)) }
        assertEquals(ContinuousAslPhase.FULL, c.snapshot.phase)
        val text = c.snapshot.english
        c.pause(); assertEquals(ContinuousAslPhase.FULL, c.snapshot.phase)
        c.resume(); c.consume(observation(300.0))
        assertEquals(3, calls); assertEquals(ContinuousAslPhase.FULL, c.snapshot.phase)
        assertEquals(text, c.snapshot.english)
        c.pause(); c.clear(); c.consume(observation(400.0))
        assertEquals(3, calls); assertEquals("", c.snapshot.english)
        c.resume(); c.consume(observation(500.0)); assertEquals(4, calls)
    }

    @Test fun pauseInvalidatesEditAndOldDismissCannotCancelResumedInput() {
        val sizes = mutableListOf<Int>()
        val c = core { sizes.add(it.size); 2 }
        start(c); repeat(3) { c.consume(observation(it*100.0)) }
        val edit = c.beginEdit()!!
        c.pause(); c.consume(observation(300.0))
        assertEquals(3, sizes.size)
        c.resume(); c.cancelEdit(edit.revision)
        assertFalse(c.applyEdit(edit.revision, "stale text"))
        c.consume(observation(400.0))
        assertEquals(listOf(1, 2, 3, 1), sizes)
        assertEquals("I", c.snapshot.english)
        assertEquals(ContinuousAslPhase.LISTENING, c.snapshot.phase)
    }
}
