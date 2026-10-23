package com.example.apphear

import com.example.apphear.domain.texttoasl.*
import com.google.gson.JsonParser
import org.junit.Assert.*
import org.junit.Test
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest

class AslWebVocabularyExpansionTest {
    private fun json(path: Path) = JsonParser.parseString(String(Files.readAllBytes(path))).asJsonObject
    private val originalRoot = Path.of("src/aslOriginalWords/assets")
    private val webRoot = Path.of("src/aslWebVocabulary/assets")
    private val registry = json(webRoot.resolve(AslWebVocabularyExpansion.MANIFEST))
    private val original = json(originalRoot.resolve(OriginalWordBatchCatalog.MANIFEST))
    private val oldRegistry = json(originalRoot.resolve(AslFingerspellingBundle.MANIFEST))
    private val frozen = json(Path.of("src/main/assets/asl_animations/manifest.json")).getAsJsonArray("clips")
        .associate { it.asJsonObject.get("id").asString to it.asJsonObject }
    private val originalWords = (original.getAsJsonArray("replacements") + original.getAsJsonArray("newClips")).map { raw ->
        val e = raw.asJsonObject
        val source = frozen[e.get("id").asString]
        AslAnimationAsset(e.get("id").asString, (e.get("displayText") ?: source!!.get("displayText")).asString,
            e.get("asset").asString, AslAnimationAsset.Type.WORD,
            (e.getAsJsonArray("aliases") ?: source!!.getAsJsonArray("aliases")).map { it.asString },
            AslAnimationAsset.Replacement(null, e.get("animationName").asString, e.get("durationMs").asLong),
            SignDictionaryMetadata(avatarProfile = AslSingleAvatarPolicy.profile))
    }
    private val oldWords = oldRegistry.getAsJsonArray("wordAdditions").map { raw ->
        val e = raw.asJsonObject
        AslVocabularyExpansion.Word(e.get("id").asString, e.getAsJsonArray("english").map { it.asString },
            e.get("animationName").asString, e.get("durationMs").asLong, e.get("sourceAsset").asString,
            e.get("sourceSha256").asString, e.get("reference").asString, e.get("meaning").asString,
            e.get("license").asString, e.get("variantOf")?.asString, e.get("construction")?.asString ?: "RETARGETED_PREVIEW")
    }
    private val aliasRoot = json(Path.of("src/main/assets/asl_animations/existing_aliases_asl.json"))
    private val aliases = aliasRoot.getAsJsonArray("aliases") + aliasRoot.getAsJsonArray("webVocabularyAliases")
    private val aliased = aliases.fold(originalWords) { base, raw ->
        val e = raw.asJsonObject
        val path = e.get("asset").asString
        ExistingAslAlias(e.get("signId").asString, e.getAsJsonArray("english").map { it.asString }, path,
            e.get("sha256").asString, e.get("reference").asString).apply(base, hash(Files.readAllBytes(originalRoot.resolve(path))))
    }
    private val base = AslVocabularyExpansion.merge(aliased.map { it.copy(assetPath = AslFingerspellingBundle.PATH) }, oldWords)
    private val words = registry.getAsJsonArray("entries").map { raw ->
        val e = raw.asJsonObject
        AslWebVocabularyExpansion.Word(e.get("id").asString, e.getAsJsonArray("english").map { it.asString },
            e.get("animationName").asString, e.get("durationMs").asLong, e.get("sourceArchiveMember").asString,
            e.get("sourceSha256").asString, e.get("motionSha256").asString, e.get("sourceLanguage").asString,
            e.get("license").asString, e.get("linguisticValidationStatus").asString)
    }
    private val letters = AslFingerspellingBundle.letters(oldRegistry.getAsJsonArray("letters").map { raw ->
        val e = raw.asJsonObject
        AslFingerspellingBundle.Letter(e.get("letter").asString.single(), e.get("animationName").asString, e.get("durationMs").asLong)
    }).mapValues { (_, a) -> a.copy(assetPath = AslWebVocabularyExpansion.PATH) }
    private val merged = AslWebVocabularyExpansion.merge(base, words)
    private val catalog = AslAnimationCatalog.fromAssets(merged, letters)
    internal fun frozenCatalogForMotionChecks(): AslAnimationCatalog = catalog
    private val planner = AslTranslationPlanner(catalog)
    private fun hash(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    @Test fun addsExactly200DistinctLexicalEntriesWithoutCountingAliasesOrAlphabet() {
        assertEquals(232, base.size)
        assertEquals(432, merged.size)
        assertEquals(200, words.size)
        for (word in words) {
            assertNull(AslAnimationCatalog.fromAssets(base).findByAlias(word.id))
            val plan = planner.plan("  ${word.id.uppercase()}!  ")
            assertEquals(listOf(word.id), plan.queue.map { it.id })
            assertTrue(plan.fingerspelledWords.isEmpty())
            assertFalse(plan.queue.single().metadata.isLinguisticallyValidated)
            assertNotEquals(TextToAslPlan.Mode.VALIDATED_PHRASE, plan.mode)
        }
    }

    @Test fun realPackageHashesAndAllOriginalActionsArePreserved() {
        assertEquals(AslWebVocabularyExpansion.MANIFEST_SHA256, hash(Files.readAllBytes(webRoot.resolve(AslWebVocabularyExpansion.MANIFEST))))
        val bytes = Files.readAllBytes(webRoot.resolve(AslWebVocabularyExpansion.PATH))
        assertEquals(AslWebVocabularyExpansion.SHA256, hash(bytes))
        fun gltf(b: ByteArray): com.google.gson.JsonObject {
            val size = java.nio.ByteBuffer.wrap(b).order(java.nio.ByteOrder.LITTLE_ENDIAN).getInt(12)
            return JsonParser.parseString(String(b, 20, size)).asJsonObject
        }
        val old = Files.readAllBytes(originalRoot.resolve(AslFingerspellingBundle.PATH))
        assertEquals(AslFingerspellingBundle.SHA256, hash(old))
        val a = gltf(old); val b = gltf(bytes)
        for (field in listOf("meshes", "skins", "nodes", "materials", "textures", "images")) assertEquals(a.get(field), b.get(field))
        val actions = b.getAsJsonArray("animations")
        assertEquals(458, actions.size())
        a.getAsJsonArray("animations").forEachIndexed { i, action -> assertEquals(action, actions[i]) }
        val names = actions.map { it.asJsonObject.get("name").asString }
        words.forEach { assertTrue(names.contains(it.animationName)) }
    }

    @Test fun aliasesOldWordsNewWordsAndNamesUseOneAvatarWithoutSentenceValidation() {
        assertEquals(listOf("apple", "car", "mother", "letter_n", "letter_i", "letter_c", "letter_o", "letter_l", "letter_e"),
            planner.plan("APPLE AUTOMOBILE MOMMY NICOLE").queue.map { it.id })
        assertEquals(listOf("buy", "help", "sick", "teacher", "student", "work", "road", "father"),
            planner.plan("purchase assist unwell instructor pupil labour street daddy").queue.map { it.id })
        val plan = planner.plan("HELLO TEA NICOLE")
        assertEquals(1, plan.queue.map { it.assetPath }.distinct().size)
        assertEquals(AslWebVocabularyExpansion.PATH, plan.queue.first().assetPath)
        assertTrue(plan.queue.all(AslSingleAvatarPolicy::permits))
        assertTrue(plan.translationNotice!!.contains("not a validated ASL sentence"))
        assertEquals(listOf("nicole"), plan.fingerspelledWords)
    }

    @Test fun incompatibleProvenanceCountsDuplicatesAndCollisionsFailClosed() {
        val first = words.first()
        for (bad in listOf(first.copy(sourceLanguage = "NGT"), first.copy(license = "unknown"),
                first.copy(reviewStatus = "REVIEWED"), first.copy(sourceMember = "../apple.glb"),
                first.copy(english = listOf("hello")), first.copy(animationName = "fake"),
                first.copy(sourceSha256 = "bad"), first.copy(durationMs = 0))) {
            assertTrue(runCatching { AslWebVocabularyExpansion.merge(base, listOf(bad) + words.drop(1)) }.isFailure)
        }
        assertTrue(runCatching { AslWebVocabularyExpansion.merge(base, words.drop(1)) }.isFailure)
        assertTrue(runCatching { AslWebVocabularyExpansion.merge(base, listOf(words[1]) + words.drop(1)) }.isFailure)
        assertTrue(runCatching { AslWebVocabularyExpansion.merge(base.drop(1), words) }.isFailure)
        val conflict = base.map { if (it.id == "hello") it.copy(aliases = it.aliases + first.id) else it }
        assertTrue(runCatching { AslWebVocabularyExpansion.merge(conflict, words) }.isFailure)
    }
}
