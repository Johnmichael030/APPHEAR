package com.example.apphear

import com.example.apphear.domain.texttoasl.*
import com.google.gson.JsonParser
import org.junit.Assert.*
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest

class AslWordMotionFixesTest {
    private val base = AslWebVocabularyExpansionTest().frozenCatalogForMotionChecks()
    private val root = Path.of("src/aslWordMotionFixes/assets")
    private val metadata = Files.readAllBytes(root.resolve(AslWordMotionFixes.MANIFEST))
    private val manifest = JsonParser.parseString(String(metadata)).asJsonObject
    private val overrides = manifest.getAsJsonArray("overrides").map { raw ->
        val e = raw.asJsonObject
        AslWordMotionFixes.Override(e.get("replacesId").asString,
            AslWebVocabularyExpansion.Word(e.get("id").asString, e.getAsJsonArray("english").map { it.asString },
                e.get("animationName").asString, e.get("durationMs").asLong, e.get("sourceArchiveMember").asString,
                e.get("sourceSha256").asString, e.get("motionSha256").asString, e.get("sourceLanguage").asString,
                e.get("license").asString, e.get("linguisticValidationStatus").asString))
    }
    private val assets = AslWordMotionFixes.merge(base.getAssets(), overrides)
    private val letters = ('a'..'z').associateWith { base.findLetter(it)!!.copy(assetPath = AslWordMotionFixes.PATH) }
    private val catalog = AslAnimationCatalog.fromAssets(assets, letters)
    internal fun frozenCatalogForExpansion(): AslAnimationCatalog = catalog
    private val planner = AslTranslationPlanner(catalog)
    private fun hash(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    @Test fun onlyFiveRequestedMotionsChangeWithoutNewWordsOrAliases() {
        assertEquals(432, assets.size)
        assertEquals(setOf("why", "where", "you", "your_yours", "today"), overrides.map { it.targetId }.toSet())
        for (old in base.getAssets()) {
            val updated = assets.single { it.id == old.id }
            assertEquals(old.aliases, updated.aliases)
            assertEquals(old.displayText, updated.displayText)
            if (overrides.any { it.targetId == old.id }) {
                assertNotEquals(old.replacement!!.animationName, updated.replacement!!.animationName)
                assertFalse(updated.metadata.isLinguisticallyValidated)
            } else {
                assertEquals(old.copy(assetPath = AslWordMotionFixes.PATH), updated)
            }
        }
        for ((input, id) in mapOf("WHY!" to "why", "WHERE?" to "where", "you" to "you", "your" to "your_yours", "yours" to "your_yours", "TODAY" to "today")) {
            val plan = planner.plan(input)
            assertEquals(listOf(id), plan.queue.map { it.id })
            assertTrue(plan.fingerspelledWords.isEmpty())
            assertTrue(plan.queue.single().replacement!!.animationName.endsWith("_3DLEX_ASL_PREVIEW"))
        }
    }

    @Test fun unavailableOrRejectedMotionsAndEnglishCopulasAreNotInvented() {
        for (id in listOf("from", "what")) {
            assertEquals(base.findByAlias(id)!!.replacement, catalog.findByAlias(id)!!.replacement)
        }
        for (text in listOf("are", "is")) {
            assertNull(catalog.findByAlias(text))
            val plan = planner.plan(text)
            assertEquals(text.map { "letter_$it" }, plan.queue.map { it.id })
            assertEquals(listOf(text), plan.fingerspelledWords)
        }
        val mixed = planner.plan("YOU NICOLE TODAY")
        assertEquals(1, mixed.queue.map { it.assetPath }.distinct().size)
        assertEquals(AslWordMotionFixes.PATH, mixed.queue.first().assetPath)
        assertTrue(mixed.translationNotice!!.contains("not a validated ASL sentence"))
    }

    @Test fun frozenModelAndAll458PreviousActionsAndGeometryArePreserved() {
        assertEquals(AslWordMotionFixes.MANIFEST_SHA256, hash(metadata))
        fun fileHash(path: Path): String {
            val digest = MessageDigest.getInstance("SHA-256")
            Files.newInputStream(path).use { stream ->
                val buffer = ByteArray(64 * 1024)
                while (true) { val count = stream.read(buffer); if (count < 0) break; digest.update(buffer, 0, count) }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }
        assertEquals(AslWordMotionFixes.SHA256, fileHash(root.resolve(AslWordMotionFixes.PATH)))
        assertEquals(AslWebVocabularyExpansion.SHA256,
            fileHash(Path.of("src/aslWebVocabulary/assets").resolve(AslWebVocabularyExpansion.PATH)))
        // Complete channel/geometry/original-binary comparison runs in the independent
        // standard-library verifier, without retaining two full GLBs in the test JVM.
        assertEquals(458, manifest.get("originalActions").asInt)
        assertEquals(458 + AslWordMotionFixes.REPLACEMENTS, manifest.get("totalActions").asInt)
    }

    @Test fun wrongTargetDuplicateMissingWrongLanguageOrFakeReviewFailsClosed() {
        val first = overrides.first()
        for (bad in listOf(first.copy(targetId = "what"), first.copy(motion = first.motion.copy(sourceLanguage = "NGT")),
                first.copy(motion = first.motion.copy(reviewStatus = "REVIEWED")), first.copy(motion = first.motion.copy(animationName = "fake")))) {
            assertTrue(runCatching { AslWordMotionFixes.merge(base.getAssets(), listOf(bad) + overrides.drop(1)) }.isFailure)
        }
        assertTrue(runCatching { AslWordMotionFixes.merge(base.getAssets(), overrides.drop(1)) }.isFailure)
        assertTrue(runCatching { AslWordMotionFixes.merge(base.getAssets(), listOf(overrides[1]) + overrides.drop(1)) }.isFailure)
    }
}
