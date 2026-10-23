package com.example.apphear

import com.example.apphear.domain.texttoasl.*
import com.google.gson.JsonParser
import org.junit.Assert.*
import org.junit.Test
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest

class AslVocabulary647Test {
    private val base = AslWordMotionFixesTest().frozenCatalogForExpansion()
    private val root = Path.of("src/aslVocabulary647/assets")
    private val metadata = Files.readAllBytes(root.resolve(AslVocabulary647.MANIFEST))
    private val manifest = JsonParser.parseString(String(metadata)).asJsonObject
    private val words = manifest.getAsJsonArray("entries").map { raw -> val e = raw.asJsonObject
        AslWebVocabularyExpansion.Word(e.get("id").asString, e.getAsJsonArray("english").map { it.asString },
            e.get("animationName").asString, e.get("durationMs").asLong, e.get("sourceArchiveMember").asString,
            e.get("sourceSha256").asString, e.get("motionSha256").asString, e.get("sourceLanguage").asString,
            e.get("license").asString, e.get("linguisticValidationStatus").asString) }
    private val aliases = manifest.getAsJsonArray("aliases").map { raw -> val e = raw.asJsonObject
        AslVocabulary647.Alias(e.get("signId").asString, e.getAsJsonArray("english").map { it.asString }) }
    private val expanded = AslVocabulary647.merge(base.getAssets(), words, aliases)
    private val letters = ('a'..'z').associateWith { base.findLetter(it)!!.copy(assetPath = AslVocabulary647.PATH) }
    private val catalog = AslAnimationCatalog.fromAssets(expanded, letters)
    private val planner = AslTranslationPlanner(catalog)

    @Test fun all215AdditionalInputsAreDistinctLexicalActionsAndNotSpelling() {
        assertEquals(432, base.getAssets().size)
        assertEquals(647, expanded.size)
        for (word in words) {
            assertNull(base.findByAlias(word.id))
            val plan = planner.plan(" ${word.id.uppercase()}! ")
            assertEquals(listOf(word.id), plan.queue.map { it.id })
            assertTrue(plan.fingerspelledWords.isEmpty())
            assertFalse(plan.queue.single().metadata.isLinguisticallyValidated)
            assertEquals(word.animationName, plan.queue.single().replacement!!.animationName)
        }
    }

    @Test fun aliasesDoNotAddEntriesOrMotionsAndAllPreviousBindingsArePreserved() {
        assertEquals(51, aliases.sumOf { it.english.size })
        for (old in base.getAssets()) {
            val updated = expanded.single { it.id == old.id }
            assertEquals(old.replacement, updated.replacement)
            assertEquals(old.metadata, updated.metadata)
            assertTrue(updated.aliases.containsAll(old.aliases))
        }
        for (alias in aliases) for (input in alias.english) {
            assertEquals(listOf(alias.signId), planner.plan(input).queue.map { it.id })
            assertFalse(catalog.findByAlias(input)!!.metadata.isLinguisticallyValidated)
        }
        for (id in listOf("why", "where", "you", "your_yours", "today")) {
            assertTrue(expanded.single { it.id == id }.replacement!!.animationName.endsWith("_3DLEX_ASL_PREVIEW"))
        }
    }

    @Test fun oldAndNewWordsAndNamesUseOneResidentAvatarWithoutClaimingSentenceTranslation() {
        val word = words.first().id
        val plan = planner.plan("HELLO $word NICOLE TODAY")
        assertEquals(listOf("hello", word, "letter_n", "letter_i", "letter_c", "letter_o", "letter_l", "letter_e", "today"), plan.queue.map { it.id })
        assertEquals(1, plan.queue.map { it.assetPath }.distinct().size)
        assertEquals(AslVocabulary647.PATH, plan.queue.first().assetPath)
        assertTrue(plan.queue.all(AslSingleAvatarPolicy::permits))
        assertTrue(plan.translationNotice!!.contains("not a validated ASL sentence"))
        assertNull(catalog.findByAlias("are")); assertNull(catalog.findByAlias("is"))
    }

    @Test fun frozenPackageHashMatchesAndInvalidCountsOrMappingsFailClosed() {
        val h = MessageDigest.getInstance("SHA-256").digest(metadata).joinToString("") { "%02x".format(it) }
        assertEquals(AslVocabulary647.MANIFEST_SHA256, h)
        val digest = MessageDigest.getInstance("SHA-256")
        Files.newInputStream(root.resolve(AslVocabulary647.PATH)).use { stream ->
            val buffer = ByteArray(64*1024)
            while (true) { val count = stream.read(buffer); if (count < 0) break; digest.update(buffer, 0, count) }
        }
        assertEquals(AslVocabulary647.SHA256, digest.digest().joinToString("") { "%02x".format(it) })
        assertTrue(runCatching { AslVocabulary647.merge(base.getAssets(), words.drop(1), aliases) }.isFailure)
        assertTrue(runCatching { AslVocabulary647.merge(base.getAssets(), words, aliases.drop(1)) }.isFailure)
        assertTrue(runCatching { AslVocabulary647.merge(base.getAssets(), listOf(words[1])+words.drop(1), aliases) }.isFailure)
        assertTrue(runCatching { AslVocabulary647.merge(base.getAssets(), listOf(words[0].copy(sourceLanguage="NGT"))+words.drop(1), aliases) }.isFailure)
        val badAliases = listOf(aliases[0].copy(signId="missing")) + aliases.drop(1)
        assertTrue(runCatching { AslVocabulary647.merge(base.getAssets(), words, badAliases) }.isFailure)
    }
}
