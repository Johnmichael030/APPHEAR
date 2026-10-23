package com.example.apphear.domain.texttoasl

import java.util.Locale

/** Offline licensed motion previews. Deliberately separate from the reviewed dictionary importer. */
object AslWebVocabularyExpansion {
    const val PATH = "asl_animations/web_vocabulary_200/avatar.glb"
    const val MANIFEST = "asl_animations/web_vocabulary_200/manifest.json"
    // Frozen package identities are generated only after offline conversion and validation.
    const val SHA256 = "6b4dbc4fce99ff6d396e3bc8bf2c792adef4f1fe1deb7bfa709306096084e9cb"
    const val MANIFEST_SHA256 = "252933a4c6710f6d944fd5f814624fbebdc85c8c86c86bf76b66d42cb7277016"

    data class Word(
        val id: String,
        val english: List<String>,
        val animationName: String,
        val durationMs: Long,
        val sourceMember: String,
        val sourceSha256: String,
        val motionSha256: String,
        val sourceLanguage: String,
        val license: String,
        val reviewStatus: String,
    )

    fun merge(base: List<AslAnimationAsset>, words: List<Word>): List<AslAnimationAsset> {
        require(base.count { it.type in setOf(AslAnimationAsset.Type.WORD, AslAnimationAsset.Type.PHRASE) } == 232)
        require(words.size == 200)
        require(words.map { it.sourceSha256 }.distinct().size == 200)
        require(words.map { it.motionSha256 }.distinct().size == 200)
        val additions = words.map { previewAsset(it, PATH) }
        val result = base.map { it.copy(assetPath = PATH) } + additions
        require(result.all(AslSingleAvatarPolicy::permits))
        AslAnimationCatalog.fromAssets(result) // Atomic ID, display-text and alias collision check.
        return result
    }

    /** Shared validation for a licensed lexical motion; caller pins the complete bundle. */
    fun previewAsset(word: Word, path: String): AslAnimationAsset {
            require(path in setOf(PATH, AslWordMotionFixes.PATH, AslVocabulary647.PATH))
            require(word.id.matches(Regex("[a-z]+")))
            require(word.english == listOf(word.id)) // Publisher's ASL lexical label, no invented synonyms.
            require(word.animationName == "SIGN_${word.id.uppercase(Locale.US)}_3DLEX_ASL_PREVIEW")
            require(word.durationMs in 500..12_500)
            require(word.sourceMember == "ANIM_GLB/${word.id}.glb")
            require(listOf(word.sourceSha256, word.motionSha256).all { it.matches(Regex("[a-f0-9]{64}")) })
            require(word.sourceLanguage == "ASL" && word.license == "CC-BY-4.0" && word.reviewStatus == "UNREVIEWED")
            return AslAnimationAsset(word.id, word.id, path, AslAnimationAsset.Type.WORD, word.english,
                AslAnimationAsset.Replacement(null, word.animationName, word.durationMs),
                SignDictionaryMetadata(avatarProfile = AslSingleAvatarPolicy.profile,
                    source = "https://osf.io/mqzyk/ — ${word.sourceMember}", license = word.license,
                    meaning = word.id, gloss = listOf(word.id.uppercase(Locale.US)),
                    references = listOf("https://aclanthology.org/2024.signlang-1.33/", "https://osf.io/mqzyk/"),
                    category = "3D-LEX ASL lexical preview",
                    usageNotes = "Unreviewed retargeted motion. Facial grammar and sentence translation are not validated.",
                    status = SignDictionaryMetadata.Status.MAPPED))
    }
}
