package com.example.apphear.continuousasl

import com.google.gson.JsonParser
import java.io.File
import org.junit.Assert.*
import org.junit.Test

/** Engineering contract checks using the actual packaged rules, not ASL accuracy evidence. */
class ContinuousAslEnglishAssetTest {
    private val config = JsonParser.parseString(File("src/continuousAsl/assets/continuous_asl_output/english.json").readText()).asJsonObject
    private val words = config.getAsJsonObject("word_equivalents").entrySet().associate { it.key to it.value.asString }
    private val phrases = config.getAsJsonArray("phrases").map { it.asJsonObject }.associate {
        it.getAsJsonArray("gloss").map { label -> label.asString } to it.get("english").asString
    }
    private val renderer = ContinuousAslEnglish(words, phrases)

    @Test fun exactPackagedRulesProduceEnglishWithoutAnEdit() {
        assertEquals("I want water.", renderer.render(listOf("IX-1p", "WANT", "WATER")).text)
        assertEquals("I do not understand.", renderer.render(listOf("IX-1p", "NOT", "UNDERSTAND")).text)
        assertEquals("Thank you.", renderer.render(listOf("THANK-YOU")).text)
        assertEquals("Good morning.", renderer.render(listOf("GOOD", "MORNING")).text)
    }

    @Test fun annotationMorphologyAndUnknownsArePreservedWithoutGuessing() {
        val labels = listOf("IX-3p:j", "(1h)WANT", "WANT_2", "<UNK>")
        val result = renderer.render(labels)
        assertEquals(labels.joinToString(" ") { "[untranslated: $it]" }, result.text)
        assertEquals(4, result.untranslated)
        assertEquals(ContinuousAslEnglishKind.PARTIAL, result.kind)
    }

    @Test fun unsupportedWordOrderIsNotPresentedAsSentenceTranslation() {
        val result = renderer.render(listOf("WANT", "WATER", "IX-1p"))
        assertEquals("Want water I", result.text)
        assertEquals(ContinuousAslEnglishKind.WORD_EQUIVALENTS, result.kind)
    }

    @Test fun packagedRulesUseOnlyLiteralModelLabelsAndDiscloseLackOfReview() {
        val vocabulary = JsonParser.parseString(File("../../packages/continuous_asl_research_01/android_assets/continuous_asl/vocabulary.json").readText())
            .asJsonArray.map { it.asString }.drop(2).toSet()
        assertTrue(vocabulary.containsAll(words.keys))
        assertTrue(vocabulary.containsAll(phrases.keys.flatten()))
        assertFalse(config.get("qualified_linguistic_review").asBoolean)
        assertEquals(config.getAsJsonArray("phrases").size(), phrases.size)
        assertFalse(words.containsKey("<UNK>"))
        assertFalse(words.containsKey("<BLANK>"))
    }
}
