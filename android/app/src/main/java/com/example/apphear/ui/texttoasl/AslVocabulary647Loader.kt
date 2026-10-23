package com.example.apphear.ui.texttoasl

import android.content.Context
import com.example.apphear.domain.texttoasl.*
import org.json.JSONObject
import java.security.MessageDigest

/** Separate default-off expansion. Frozen 432 entries and five motion fixes are its base. */
class AslVocabulary647Loader(private val context: Context) {
    fun load(base: AslAnimationCatalog): AslAnimationCatalog {
        val bytes = context.assets.open(AslVocabulary647.MANIFEST).use { it.readBytes() }
        val hash = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        require(hash == AslVocabulary647.MANIFEST_SHA256)
        val root = JSONObject(String(bytes, Charsets.UTF_8))
        require(root.getInt("schemaVersion") == 1 && root.getString("packageId") == "3dlex-asl-vocabulary647-v1")
        require(root.getString("language") == "ASL" && root.getString("avatarProfile") == "ORIGINAL_HUMAN")
        require(root.getString("asset") == AslVocabulary647.PATH && root.getString("sha256") == AslVocabulary647.SHA256)
        require(root.getString("baselineSha256") == AslWordMotionFixes.SHA256 && root.getInt("originalActions") == 463)
        require(root.getString("status") == "UNREVIEWED_RETARGETED_PREVIEW" && root.getInt("qualifiedAslValidations") == 0)
        require(root.getInt("newLexicalEntries") == 215 && root.getInt("totalLexicalEntries") == 647 && root.getInt("newAliases") == 51)
        require(AslAssetDigests.of(context, AslVocabulary647.PATH).sha256 == AslVocabulary647.SHA256)
        val raw = root.getJSONArray("entries")
        val words = (0 until raw.length()).map { i -> val e = raw.getJSONObject(i); val forms = e.getJSONArray("english")
            AslWebVocabularyExpansion.Word(e.getString("id"), (0 until forms.length()).map(forms::getString),
                e.getString("animationName"), e.getLong("durationMs"), e.getString("sourceArchiveMember"),
                e.getString("sourceSha256"), e.getString("motionSha256"), e.getString("sourceLanguage"),
                e.getString("license"), e.getString("linguisticValidationStatus")) }
        val rawAliases = root.getJSONArray("aliases")
        val aliases = (0 until rawAliases.length()).map { i -> val e = rawAliases.getJSONObject(i); val forms = e.getJSONArray("english")
            require(e.getString("status") == "UNREVIEWED_PREVIEW_ALIAS")
            AslVocabulary647.Alias(e.getString("signId"), (0 until forms.length()).map(forms::getString)) }
        val assets = AslVocabulary647.merge(base.getAssets(), words, aliases)
        val letters = ('a'..'z').associateWith { requireNotNull(base.findLetter(it)).copy(assetPath = AslVocabulary647.PATH) }
        return AslAnimationCatalog.fromAssets(assets, letters)
    }
}
