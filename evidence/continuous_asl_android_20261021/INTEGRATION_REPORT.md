# Continuous-ASL Android integration result

**PASS host build, tests, export parity and asset integrity; PARTIAL device readiness; BLOCKED recognition-quality/production acceptance.**

The user-requested **CONTINUOUS WORD · experimental** mode is present in Settings in the opt-in debug APK. It uses the selected full-task checkpoint through an independent small FP32 inference export. Candidate01 observations and the existing 378-feature builder feed bounded causal CTC drafts. Explicit review/edit is required before text is used or spoken. WORD/FINGERSPELL remain separate; default/release flags are off.

- JVM tests: **284**, failures **0**, errors **0**; includes nine new continuous tests and existing feature/owner regressions.
- Debug and Android-test APK builds: **PASS**. Final log: `build_final.log`.
- Synthetic desktop PyTorch/ONNX and causal-prefix parity: **PASS**. This is numerical evidence, not model evaluation.
- Android native parity test: compiled, **not run**. No phone installation, application-data deletion, camera recording or participant processing was performed.
- Existing source asset files unchanged: **566**. Frozen fingerspelling package entries unchanged: **33**. Selected continuous source checkpoint SHA unchanged.
- No fitting, data acquisition, reserved TEST use or avatar/model replacement.

## Size

Only model/vocabulary/manifest are packaged for this feature: **813,449 bytes (0.776 MiB)**. Their compressed/stored APK payload is **791,456 bytes (0.755 MiB)**, excluding ZIP directory/alignment/code overhead. Synthetic fixtures are only in the test APK.

Training checkpoint size is 2,349,195 bytes; the runtime asset package is 65.37% smaller. Optimizer/RNG/data/logs are not shipped. No new inference dependency is added. Extra quantization was unnecessary and was not performed.

Complete debug APK: **853,580,528 bytes (814.04 MiB)**. Most existing payload is avatar animation assets; no avatar compression or removal occurred. The 0.776-MiB figure describes this recognition addition, not the whole application.

APK: `D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\android\app\build\outputs\apk\debug\app-debug.apk`

APK SHA256: `bbef4b37d7067d092a726c55ab9fe98a4761d793442deaa992a1c0633ac93296`

## Review

| Layer | Result | Remaining limitation |
|---|---|---|
| Requested Settings + recognition integration | PASS | Available in this built opt-in debug APK; not yet installed. |
| Exclusive owner, review, lifecycle and input guards | PASS host tests/source review | Physical-device switching/backpressure/TTS smoke pending. |
| Model export and contract | PASS numerical/host evidence | Native Android synthetic parity test awaits a device/emulator. |
| Product accuracy, no-sign behavior, cadence/latency | BLOCKED | Full-task held-out WER remains 95.67%; no accepted streaming/unseen-signer evidence. |

No outstanding critical host-build issue was found. The explicit review gate does not make the recognizer accurate. Recommendation: **local experimental engineering smoke only; do not promote**. Device parity and F22 behavior/timing must be checked before wider use. No permission to upload new participant data or to publish restricted models/data is inferred.

Implementation/design/size details: `docs/continuous-asl-android-research-20261021.md`.
Rollback: disable `continuousAsl` when building, then choose WORD. A stale continuous preference resolves to WORD in a build without this flag. No existing model or GLB needs restoration.

One pre-existing Windows-1252 ellipsis byte in RecognitionActivity was normalized to UTF-8 for source editing; its original bytes are in `before/`. An initial unsigned PowerShell-wrapper invocation was blocked by the machine's script policy. The equivalent direct Gradle command succeeded without changing persistent execution policy.
