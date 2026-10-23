package com.example.apphear.ui.texttoasl

import android.content.Context
import com.example.apphear.domain.texttoasl.*
import org.json.JSONObject
import java.security.MessageDigest

/** Hash-pinned opt-in substitutions within the preserved 432-entry catalog. */
class AslWordMotionFixesLoader(private val context: Context) {
    fun load(base: AslAnimationCatalog): AslAnimationCatalog {
        val bytes = context.assets.open(AslWordMotionFixes.MANIFEST).use { it.readBytes() }
        val hash = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        require(hash == AslWordMotionFixes.MANIFEST_SHA256)
        val root = JSONObject(String(bytes, Charsets.UTF_8))
        require(root.getInt("schemaVersion") == 1 && root.getString("packageId") == "3dlex-asl-word-motion-fixes-v1")
        require(root.getString("language") == "ASL" && root.getString("avatarProfile") == "ORIGINAL_HUMAN")
        require(root.getString("asset") == AslWordMotionFixes.PATH && root.getString("sha256") == AslWordMotionFixes.SHA256)
        require(root.getString("baselineSha256") == AslWebVocabularyExpansion.SHA256 && root.getInt("originalActions") == 458)
        require(root.getString("status") == "UNREVIEWED_RETARGETED_PREVIEW" && root.getInt("qualifiedAslValidations") == 0)
        require(root.getInt("newLexicalEntries") == 0 && root.getInt("totalLexicalEntries") == AslWordMotionFixes.TOTAL_ENTRIES)
        require(root.getInt("replacementMotions") == AslWordMotionFixes.REPLACEMENTS)
        require(AslAssetDigests.of(context, AslWordMotionFixes.PATH).sha256 == AslWordMotionFixes.SHA256)
        fun word(e: JSONObject): AslWebVocabularyExpansion.Word {
            val forms = e.getJSONArray("english")
            return AslWebVocabularyExpansion.Word(e.getString("id"), (0 until forms.length()).map(forms::getString),
                e.getString("animationName"), e.getLong("durationMs"), e.getString("sourceArchiveMember"),
                e.getString("sourceSha256"), e.getString("motionSha256"), e.getString("sourceLanguage"),
                e.getString("license"), e.getString("linguisticValidationStatus"))
        }
        val overrides = root.getJSONArray("overrides")
        val assets = AslWordMotionFixes.merge(base.getAssets(),
            (0 until overrides.length()).map { i -> val e = overrides.getJSONObject(i)
                AslWordMotionFixes.Override(e.getString("replacesId"), word(e)) })
        val letters = ('a'..'z').associateWith { requireNotNull(base.findLetter(it)).copy(assetPath = AslWordMotionFixes.PATH) }
        return AslAnimationCatalog.fromAssets(assets, letters)
    }
}
