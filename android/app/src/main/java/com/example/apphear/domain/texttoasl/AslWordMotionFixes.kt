package com.example.apphear.domain.texttoasl

/** Five licensed motion substitutions. Catalog size and all English aliases are unchanged. */
object AslWordMotionFixes {
    const val PATH = "asl_animations/word_motion_fixes/avatar.glb"
    const val MANIFEST = "asl_animations/word_motion_fixes/manifest.json"
    const val SHA256 = "244ba8daf6371124dac2da6a985fff0dfaead8ee92e672fcfe93ebf5d5e24616"
    const val MANIFEST_SHA256 = "918f65803b7da726f34f6ca39de034f6a731566df4a5933d7a9fbbd79417fa56"
    const val REPLACEMENTS = 5
    const val TOTAL_ENTRIES = 432
    data class Override(val targetId: String, val motion: AslWebVocabularyExpansion.Word)
    private val allowedOverrides = mapOf("why" to "why", "where" to "where", "you" to "you",
        "your_yours" to "your", "today" to "today")

    fun merge(base: List<AslAnimationAsset>, overrides: List<Override>): List<AslAnimationAsset> {
        require(base.count { it.type in setOf(AslAnimationAsset.Type.WORD, AslAnimationAsset.Type.PHRASE) } == 432)
        require(overrides.size == REPLACEMENTS)
        require(overrides.map { it.targetId }.toSet() == allowedOverrides.keys)
        val motions = overrides.map { it.motion }
        require(motions.map { it.sourceSha256 }.distinct().size == motions.size)
        require(motions.map { it.motionSha256 }.distinct().size == motions.size)
        val replacements = overrides.associate { entry ->
            require(allowedOverrides[entry.targetId] == entry.motion.id)
            val old = base.single { it.id == entry.targetId }
            val replacement = AslWebVocabularyExpansion.previewAsset(entry.motion, PATH)
            entry.targetId to replacement.copy(id = old.id, displayText = old.displayText, aliases = old.aliases)
        }
        val result = base.map { replacements[it.id] ?: it.copy(assetPath = PATH) }
        require(result.all(AslSingleAvatarPolicy::permits))
        AslAnimationCatalog.fromAssets(result)
        require(result.count { it.type in setOf(AslAnimationAsset.Type.WORD, AslAnimationAsset.Type.PHRASE) } == TOTAL_ENTRIES)
        return result
    }
}
