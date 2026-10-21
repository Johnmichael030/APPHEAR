# Continuous word recognition: opt-in Android integration

Date: 2026-10-21, Asia/Shanghai.

## Scope and decisions

The user requested continuous-ASL recognition in Settings and the recognition screen, then asked to reduce its packaged size if necessary. This adds a third mode, `CONTINUOUS_WORD`, beside WORD and FINGERSPELL in a debug build with `-PcontinuousAsl=true`. Default/release flags stay off. This is research UI/runtime integration, not promotion of a successful recognizer.

Continuous recognition means causal gloss-sequence inference over genuine current camera observations. A draft is tentative literal ASL gloss output. Confirmed text is the text a user explicitly reviewed/edited and accepted. Neither is an automatic ASL-to-English translation.

The architecture skill was used to resolve these choices against the user's existing authorization: preserve the WORD and fingerspelling owners, reuse Candidate01 and the verified feature math, choose the already-selected full-task checkpoint rather than the small in-sample control, require explicit draft review, and build/test locally. No new fit, threshold tuning, TEST use, installation, account change, dataset download or avatar change is part of this feature.

## Model and feature contract

- Source: `experiments/continuous_asl/20261020T143609Z_learning_diagnostic01/candidate_checkpoints/epoch_012.pt`.
- SHA256: `14d0e4669674265ea2be8a3161fa6699e4f813daf4b97b384863dce1daa038ff`.
- Frozen selected source model: 48-channel causal CTC, 2,287 literal target labels plus blank 0 and UNK 1. Its held-out WER is 95.67%; it is unreliable.
- `packages/continuous_asl_research_01` is a separate inference export. Original checkpoints and manifests are never overwritten.
- Raw Candidate01 observations enter the existing `PhoneSegmentationFeatureBuilder` unchanged. The resulting `apphear.perception_vnext.segmentation_features.v1` 378 floats implement the same math used by the training wrapper `apphear.continuous_asl.candidate01_causal378.v1`. No feature block is removed and no new feature layout is invented.
- TRAIN mean/std and clamp are embedded in the exported graph, matching the original PyTorch forward pass. The Android adapter must not normalize twice.
- The manifest pins schemas, shape, checkpoint/model/vocabulary hashes, class order, input/output names, CPU threads and operational limits. Invalid/missing packages fail closed while the selected mode continues to block WORD commits.

## Causal inference and ownership

Exactly one recognizer consumes each Candidate01 observation. Continuous mode takes priority in the existing `ownsObservation` callback; it returns ownership even if the model is missing or failed. No S2/WORD commit or contextual automatic fingerspell routing runs in that mode. Candidate01 perception, S2 admission/segmentation, recognition-v2, the frozen WORD models and the fingerspelling models remain untouched.

The Android session samples the first genuine observation at least 100 ms after the previous selected observation. Feature differences are computed from those selected observations, never from sliced stateful feature vectors. Actual timestamps are retained; there is no interpolation, fabricated landmark, resampling or dropped feature block. Physical camera cadence may be lower than training cadence; live accuracy and throughput remain unverified.

The model's exact receptive field is 61 observations. The adapter immediately evaluates each selected observation using up to 61 actual prior observations; it does not wait for a 61-frame window to fill. The latest logit is decoded with stateful greedy CTC (collapse IDs before removing blank). No new language model, beam search, calibration or confidence threshold is added. Literal UNK remains visible as `<UNK>` rather than being turned into an invented word.

No observed valid hand causes an explicit abstention and temporal/CTC reset. Long timestamp gaps reset context. Duplicate/backward timestamps or nonfinite inference fail closed. These are operational safeguards, not calibrated false-activation thresholds. Drafts stop at 128 tokens instead of silently discarding or committing output. Confirmation text is bounded to 4,096 characters. Pause/lens/session changes invalidate temporal state; camera rebinding clears the experimental draft and confirmation buffer. Review freezes input, and revision/generation checks reject stale confirmations.

All model loading is on IO. Inference runs on the existing serialized camera worker with CPU ONNX Runtime, two intra-op threads and one inter-op thread. Session/options close only after the camera worker drains. No GPU delegate, camera queue or new runtime dependency is added. Future phone timing must report p50/p95 and delivered cadence; host build/parity success is not a latency measurement.

## UI

Settings → Recognition mode → **CONTINUOUS WORD · experimental**.

The recognition screen shows an experimental badge, DRAFT GLOSSES, a review button, a reliability explanation and a separate CONFIRMED text area. Long drafts are visually limited to three lines; Review contains the complete editable draft. Clear affects only this mode. Hear opens review for tentative output, or speaks already-confirmed text. No draft automatically enters the WORD sentence builder, Context Assist, saved history or TTS. The bottom speech-input control keeps its existing behavior.

## Measured package size

| File | Bytes |
|---|---:|
| FP32 inference ONNX | 777,490 |
| Literal vocabulary | 34,441 |
| Manifest/policy | 1,518 |
| **Runtime assets total** | **813,449 (0.776 MiB)** |

The original training checkpoint is 2,349,195 bytes. The runtime package is approximately 65.37% smaller, mainly because optimizer/RNG/training state is absent. Dataset files, training features, old models, logs and research reports are not app assets. Synthetic parity fixtures are in the test APK only. The app already uses ONNX Runtime 1.22.0 for Candidate01, so this feature introduces no second inference library.

Additional lossy compression, FP16 or INT8 quantization is unnecessary for this sub-1-MiB addition. It would require independent numerical and decoding validation. Keep the verified FP32 export. Existing GLB/avatar assets are outside this feature's size changes. Actual APK entries and overall APK size are recorded after the completed build in `evidence/continuous_asl_android_20261021/VALIDATION.json`.

## Checks and limits

- Export parity compares PyTorch with CPU ONNX over synthetic lengths 1/2/17/61/97 and every causal prefix. Checks use `atol=3e-5, rtol=3e-5`; maximum absolute error was 6.1035e-5 and all argmax IDs matched. This is numerical parity, not recognition evidence.
- Existing feature-builder golden tests remain applicable because its code/math did not change.
- New JVM tests cover CTC repetition/blank order, mode gates, no-hand abstention, immediate inference/context bounds, real cadence, gap/timestamp failures, explicit editing, stale confirmation and draft limits.
- `ContinuousAslParityInstrumentedTest` compares Android CPU logits with a synthetic fixture. It requires an authorized device/emulator; compiling it is not a claim that it ran.
- Frozen assets and checkpoint hashes are verified separately. Validation artifacts and logs remain local.

The saved accuracy report remains authoritative about recognition quality. Production promotion, reliable no-sign behavior, unseen-signer recognition, qualified annotation review, boundary behavior and phone end-to-end latency remain unverified/blocked. No deployment or accuracy claim follows from compilation.

## Build and test

Use the existing `tools/android/build_working_baseline.ps1` with `-Configuration fake -OriginalWords -ContinuousAsl`. The new switch forwards `-PcontinuousAsl=true`; all existing flags/assets stay in their owners. On machines enforcing signed PowerShell scripts, invoke the equivalent `gradlew.bat` command recorded in the evidence commands file; no persistent execution-policy change is required.

Read-only package check: compare the source checkpoint SHA above and each asset SHA in `manifest.json`. Do not rerun the exclusive export into an existing package. To rebuild a future export, choose a new package identity and validate it separately.

Device testing after an authorized installation must check Settings persistence, camera overlay, edited confirmation/Hear, Clear, pause/resume, lens switch, missing/corrupt model, no hands, switching back to WORD/FINGERSPELL, delivered cadence/latency, and preserved existing application data. Recognition-quality acceptance remains a separate gate.

API references: [ONNX Runtime Java](https://onnxruntime.ai/docs/get-started/with-java.html), [session options](https://onnxruntime.ai/docs/api/java/ai/onnxruntime/OrtSession.SessionOptions.html), [tensor API](https://onnxruntime.ai/docs/api/java/ai/onnxruntime/OnnxTensor.html), [PyTorch ONNX export](https://docs.pytorch.org/docs/main/onnx_export.html).
