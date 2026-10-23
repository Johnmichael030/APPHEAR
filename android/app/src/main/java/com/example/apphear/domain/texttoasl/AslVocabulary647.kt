package com.example.apphear.domain.texttoasl

/** Frozen licensed lexical preview expansion; aliases never count as additional signs. */
object AslVocabulary647 {
    const val PATH = "asl_animations/vocabulary_647/avatar.glb"
    const val MANIFEST = "asl_animations/vocabulary_647/manifest.json"
    const val SHA256 = "7840f979713d878f273a71975c38e1d12dbf3856151af531cbed8daf70fd0024"
    const val MANIFEST_SHA256 = "25b864eb411c76693ac338e6f0062c7fffa13a03143e0c46ceccc9f96ee5bcc9"
    const val NEW_ENTRIES = 215
    const val TOTAL_ENTRIES = 647
    const val NEW_ALIASES = 51
    data class Alias(val signId: String, val english: List<String>)

    fun merge(base: List<AslAnimationAsset>, words: List<AslWebVocabularyExpansion.Word>,
              aliases: List<Alias>): List<AslAnimationAsset> {
        require(base.count { it.type in setOf(AslAnimationAsset.Type.WORD, AslAnimationAsset.Type.PHRASE) } == 432)
        require(words.size == NEW_ENTRIES)
        require(words.map { it.sourceSha256 }.distinct().size == NEW_ENTRIES)
        require(words.map { it.motionSha256 }.distinct().size == NEW_ENTRIES)
        require(aliases.sumOf { it.english.size } == NEW_ALIASES)
        var result = base.map { it.copy(assetPath = PATH) } + words.map { AslWebVocabularyExpansion.previewAsset(it, PATH) }
        AslAnimationCatalog.fromAssets(result)
        for (alias in aliases) {
            require(alias.english.isNotEmpty() && alias.english.all { it.isNotBlank() })
            val target = result.single { it.id == alias.signId }
            require(target.type in setOf(AslAnimationAsset.Type.WORD, AslAnimationAsset.Type.PHRASE))
            result = result.map { if (it.id == target.id) it.copy(aliases = (it.aliases + alias.english).distinct()) else it }
        }
        require(result.all(AslSingleAvatarPolicy::permits))
        AslAnimationCatalog.fromAssets(result) // Reject collisions atomically; never repair silently.
        require(result.count { it.type in setOf(AslAnimationAsset.Type.WORD, AslAnimationAsset.Type.PHRASE) } == TOTAL_ENTRIES)
        return result
    }
}
