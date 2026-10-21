package com.example.apphear.data.settings

import android.content.Context
import com.example.apphear.BuildConfig
import com.example.apphear.fingerspelling.RecognitionMode

/** Default WORD. Unsupported/stale choices safely resolve to WORD in nonexperimental builds. */
class RecognitionModeSettings(context: Context) {
    private val prefs = context.getSharedPreferences("recognition_mode", Context.MODE_PRIVATE)
    val choices get() = availableModes(BuildConfig.FINGERSPELL_EXPERIMENT_ENABLED,
        BuildConfig.CONTINUOUS_ASL_EXPERIMENT_ENABLED)
    val selectionAvailable get() = choices.size > 1
    fun current() = resolve(prefs.getString("selected", null),
        BuildConfig.FINGERSPELL_EXPERIMENT_ENABLED, BuildConfig.CONTINUOUS_ASL_EXPERIMENT_ENABLED)
    fun select(mode: RecognitionMode) {
        require(mode in choices)
        prefs.edit().putString("selected", mode.name).apply()
    }
    companion object {
        fun availableModes(fingerspell: Boolean, continuous: Boolean) = buildList {
            add(RecognitionMode.WORD)
            if (fingerspell) add(RecognitionMode.FINGERSPELL)
            if (continuous) add(RecognitionMode.CONTINUOUS_WORD)
        }
        fun resolve(saved: String?, enabled: Boolean, continuous: Boolean = false) =
            availableModes(enabled, continuous).firstOrNull { it.name == saved } ?: RecognitionMode.WORD
    }
}
