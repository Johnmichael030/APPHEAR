package com.example.apphear.ui.texttoasl

import android.content.Context
import com.example.apphear.domain.texttoasl.AslAnimationAsset
import com.example.apphear.domain.texttoasl.AslAnimationCatalog
import com.example.apphear.domain.texttoasl.AslFingerspellingBundle
import com.example.apphear.domain.texttoasl.AslWebVocabularyExpansion
import org.json.JSONObject
import java.security.MessageDigest

/** Opt-in, fail-closed loader; never upgrades previews to reviewed dictionary entries. */
class AslWebVocabularyLoader(private val context: Context) {
    fun load(base: List<AslAnimationAsset>, letters: Map<Char, AslAnimationAsset>): AslAnimationCatalog {
        val bytes = context.assets.open(AslWebVocabularyExpansion.MANIFEST).use { it.readBytes() }
        val hash = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        require(hash == AslWebVocabularyExpansion.MANIFEST_SHA256)
        val root = JSONObject(String(bytes, Charsets.UTF_8))
        require(root.getInt("schemaVersion") == 1 && root.getString("packageId") == "3dlex-asl-preview-200-v1")
        require(root.getString("language") == "ASL" && root.getString("avatarProfile") == "ORIGINAL_HUMAN")
        require(root.getString("asset") == AslWebVocabularyExpansion.PATH && root.getString("sha256") == AslWebVocabularyExpansion.SHA256)
        require(root.getString("baselineSha256") == AslFingerspellingBundle.SHA256)
        require(root.getString("status") == "UNREVIEWED_RETARGETED_PREVIEW" && root.getInt("qualifiedAslValidations") == 0)
        require(root.getInt("newLexicalEntries") == 200 && root.getInt("totalLexicalEntries") == 432 && root.getInt("originalActions") == 258)
        require(AslAssetDigests.of(context, AslWebVocabularyExpansion.PATH).sha256 == AslWebVocabularyExpansion.SHA256)
        val raw = root.getJSONArray("entries")
        val words = (0 until raw.length()).map { i ->
            val e = raw.getJSONObject(i)
            val forms = e.getJSONArray("english")
            AslWebVocabularyExpansion.Word(e.getString("id"), (0 until forms.length()).map(forms::getString),
                e.getString("animationName"), e.getLong("durationMs"), e.getString("sourceArchiveMember"),
                e.getString("sourceSha256"), e.getString("motionSha256"), e.getString("sourceLanguage"),
                e.getString("license"), e.getString("linguisticValidationStatus"))
        }
        require(letters.keys == ('a'..'z').toSet())
        val catalog = AslAnimationCatalog.fromAssets(AslWebVocabularyExpansion.merge(base, words),
            letters.mapValues { (_, letter) -> letter.copy(assetPath = AslWebVocabularyExpansion.PATH) })
        val selected = if (com.example.apphear.BuildConfig.ASL_WORD_MOTION_FIXES_ENABLED)
            AslWordMotionFixesLoader(context).load(catalog) else catalog
        return if (com.example.apphear.BuildConfig.ASL_VOCABULARY_647_ENABLED)
            AslVocabulary647Loader(context).load(selected) else selected
    }
}
