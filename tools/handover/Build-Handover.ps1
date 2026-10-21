$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$fingerspellPackage = (Join-Path $projectRoot 'packages\fingerspelling_continuous_v2_1_01').Replace('\','/')
if (-not (Test-Path -LiteralPath (Join-Path $fingerspellPackage 'android_assets'))) {throw 'Run git lfs pull and verify_checkout.py first.'}
$buildArguments = @('testDebugUnitTest','assembleDebug','assembleDebugAndroidTest','--console=plain','--no-daemon','--max-workers=2',
 '-Pandroid.useAndroidX=true','-Pfingerspell=true','-PfingerspellContinuous=true',"-PfingerspellArtifactDir=$fingerspellPackage",
 '-PrecognitionRearCamera=true','-PrecognitionCoordination=true','-PcontextPhraseAssist=true','-PcontextPhraseAssistProvider=fake',
 '-PaslAssetPilot=true','-PaslSharedCharacter=true','-PaslMoreWords=true','-PaslCharacterSelection=true','-PaslNewIdle=true',
 '-PaslAlphabetFallback=true','-PaslOriginalWords=true','-PcontinuousAsl=true',
 '-PSUPABASE_URL=','-PSUPABASE_ANON_KEY=','-PAPPHEAR_SUPABASE_URL=','-PAPPHEAR_SUPABASE_ANON_KEY=',
 '-PAPPHEAR_ELEVENLABS_MODEL_ID=','-PAPPHEAR_ELEVENLABS_VOICE_ID=','-PAPPHEAR_ELEVENLABS_API_KEY=',
 '-PAPPHEAR_CONTEXT_ASSIST_API_KEY=','-PAPPHEAR_GEMINI_API_KEY=')
Push-Location (Join-Path $projectRoot 'android')
try {
 & '.\gradlew.bat' @buildArguments
 if ($LASTEXITCODE -ne 0) {throw 'Handover build failed; preserve installed app and application data.'}
} finally {Pop-Location}
