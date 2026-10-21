# AppHear continuous ASL developer handover

Date: 21 October 2026, Asia/Shanghai. This delivery shares the existing verified APK and source; it does not deploy a newly trained model. Repository visibility must remain private. Share collaborator access separately.

## Clone and verify Git LFS

Use Git LFS and a short Windows path, such as C:\src\APPHEAR. Install Git LFS locally, clone this private repository, then run `git lfs pull`, `git lfs fsck` and `python .\tools\handover\verify_checkout.py` from its root. LFS retrieves actual binary assets; pointer files alone are insufficient. An ordinary GitHub source ZIP may contain pointers. The receiver needs repository access.

The APK is directly under `apk/AppHear-continuous-refined-20261021.apk`. GitHub's file view offers Download raw file. Version `1.0-continuous-refined-20261021`, application ID `com.example.apphear`, minimum API 26, target 36, 852,132,332 bytes (812.7 MiB). SHA256 `a547e867134f2be80aa1a54265b3c6f3eac524588de55670c009ee1995a8493c`. It is the existing credentials-free debug build, not a newly trained app model. Settings > Recognition mode > CONTINUOUS WORD · experimental enables the included mode.

All included models, checkpoints and app assets live at their original source paths with Git LFS, not in a new release archive. The verifier checks their materialized bytes against handover/CONTENT_MANIFEST.json. It reports missing or pointer-only files instead of silently skipping them. No data extraction or participant processing occurs.

## Source map

- `android/`: complete reviewed Android source/resources/build wrapper, unit/instrumentation tests and UI integration. Large binary payloads are materialized by Git LFS.
- `android/app/src/main/java/com/example/apphear/continuousasl/`: model runner, causal378 features, CTC core/decoder, streaming state, finite English renderer.
- `RecognitionActivity.kt` and Settings/recognition-mode code in the same app: camera ownership, lifecycle and mode selection.
- `packages/continuous_asl_research_01/`: exact exported epoch-12 model, manifest, literal vocabulary and synthetic parity fixtures.
- `tools/continuous_asl/`: feature generation, CTC training/evaluation/export and research scripts.
- `experiments/continuous_asl/`: preserved research reports/configuration/scripts. Binary weights are materialized by Git LFS at their original directories; source media/features/catalogs and reserved TEST payloads are omitted.
- `evidence/20261021T062258Z_continuous_detectionfix01/` and `docs/continuous-asl-*.md`: packaged APK validation, lifecycle/missing-hand fixes, replay evidence, English renderer limitations.
- `experiments/continuous_asl/20261021T080638Z_finetune_vocab01/`: latest completed A fine-tune; selected research checkpoint `A/checkpoints/epoch_002.pt`, SHA256`a190e96b035e74b374c651a48ca56178c1ce593666938e0825d63368556f51da`. **Not the APK model.**

Root `requirements.txt` describes the broader older pipeline. Existing research used Torch2.8.0+cu128/CUDA12.8, NumPy and ONNXRuntime1.22.0 on RTX3050Laptop4GiB; obtain supported wheels in a separate environment. No environment or CUDA installer is bundled. Android uses checked-in Gradle8.13/AGP8.11.2/Kotlin2.0.21 and its TOML dependency versions. Use JDK17 and Android SDK36 with build-tools35.0.0 as in the historical successful build. Dependency downloads and a new build on the receiving machine are not tested by this upload.

## Build and verify

After running git lfs pull, configure your own SDK path via local.properties or ANDROID_HOME. Do not copy someone else's signing keys or service credentials. From repository root run `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\handover\Build-Handover.ps1`. It runs unit tests and assembles debug APKs with the captured feature flags and empty cloud credentials; it does not install or access a phone. A new debug signing key may prevent updating an existing installation; do not clear application data as a workaround.

The supplied APK was already built and passed301hosttests and APK signature verification in its original evidence. This upload verifies hashes, source scope and archive integrity; it does not repeat a build or claim fresh phone validation. Synthetic ONNX/checkpoint parity fixtures are engineering evidence, not ASL accuracy measurements.

## Current behavior and limitations

The existing model is causal CTC,378features,48channels,RF61,2,287literal target labels plus blank0/UNK1. Preserve exact IDs, morphology, features and normalization. Do not confuse this schema with WORD378, fingerspell405 or local recorder3.0. The renderer provides75literal equivalents and40finite patterns; it is not general ASL-to-English translation.

Pause stops inference, asynchronous load resumes only while STARTED, and missing hands discard unsettled tokens. Keep lifecycle/no-hand/blank-separated-repeat safeguards. Do not reintroduce forced publication on loss of tracking. WORD/FINGERSPELL and avatar assets remain unchanged.

Latest A:1,500TRAIN/80known-signer family-heldDEV,10fine-tuningepochs/82.106s,1.214GiBpeakallocatedVRAM. TRAIN WER88.40%â†’85.70%; fullsequenceDEV95.67%â†’94.64%; streamingDEV95.67%â†’95.67%; exact0/80. No classes added. New-class data admission is blocked; no qualified linguistic review or independent no-sign/unseen-signer validation is established. Do not promote A or repeat failed framing/history variants blindly. See MISSION_REPORT.md and PROMOTION_DECISION.md for the frozen gates.

## Data access and reproduction limits

Training needs separately permissioned dataset access and verified feature files. Historical absolute Windows paths are provenance, not portable live paths. Frozen manifests/results are preserved, not silently rewritten for a new machine; introduce an explicit resolver/new experimental copy if authorized. The older read-only validators that require omitted data will correctly fail until dependencies and legitimate data access are restored. Keep ASLLRP3 literal occurrence and NCSLGR main-gloss pools separate. English translations are not gloss sequences.

No participant media, derived features, consent documents, private recording metadata, raw archives, environment/cache or reserved TEST payload was added. Existing encrypted experiments backup remains unchanged and needs its separately held recovery key. The receiving developer must obtain dataset access directly from the publisher rather than redistributing restricted media. [ASLLRP terms](https://www.bu.edu/asllrp/dai-terms.html) and [SignBank terms](https://www.bu.edu/asllrp/signbank-terms.pdf) require attribution and restrict commercial/redistribution uses. Cite CarolNeidle/DimitrisMetaxas, ASLLRP SignBank, Boston/Rutgers Universities, and the applicable corpus publication. Model files do not grant commercial or dataset redistribution permission.

The smallest quality-improvement step is the qualified12eventTRAIN review and permissioned repeated natural signs across independent recordings/signers, plus consented no-sign/boundary evaluation. Resolve local permission/verified-label records and RIT timing before admitting new classes. Do not use the reserved17TEST for selection/tuning.
