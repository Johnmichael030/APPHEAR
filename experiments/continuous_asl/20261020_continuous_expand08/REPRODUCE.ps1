# Completed-run read-only verification; does not train or evaluate held-out TEST.
$repo='D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite'
Set-Location -LiteralPath $repo
& "$repo\experiments\continuous_asl\20261020_continuous_expand08\Validate-Completed.ps1"

# Historical execution commands (DO NOT run another copy while an active driver
# or feature worker remains). Logs/config/checkpoint RNG state preserve detail.
# & "$repo\experiments\continuous_asl\20261020_continuous_expand07\Run-OverfitV2.ps1"
# & "$repo\experiments\continuous_asl\20261020_continuous_expand09\Run-PilotV2.ps1"
# & "$repo\experiments\continuous_asl\20261020_continuous_expand08\Run-FinishAndTrain.ps1"
# The completed wrappers reuse status/checkpoint evidence rather than refitting.
# If a main model is interrupted after a full epoch, the wrapper supplies--resume
# automatically from checkpoints/latest.json, using the same frozen manifest.
# Never modify a completed model or frozen dataset to request a fresh run.
