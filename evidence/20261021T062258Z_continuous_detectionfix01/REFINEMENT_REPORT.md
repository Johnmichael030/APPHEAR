# Continuous-ASL detection refinement and accuracy decision

Date: 2026-10-21, Asia/Shanghai. **Overall: PARTIAL.** Two detection/lifecycle software defects are fixed and a new APK is built. The recognizer is still unreliable; no improvement in held-out word error rate is claimed.

## What changed in the app

1. **No new token on missing-hand observations.** Previously, losing hand tracking forced an unfinished CTC run into the displayed text, even before its display hold elapsed. The app now discards the unfinished run and abstains. Already displayed text remains. Short peaks ending on a genuinely observed blank/token transition and real blank-separated repeats remain supported.
2. **Pause actually stops inference.** A suspended state now blocks late queued camera observations. Selecting ownership alone cannot resume the model. Activity start and asynchronous model-load completion explicitly resume only while the Activity is STARTED. Pauses preserve text and ERROR/FULL states, invalidate stale edits and reset temporal context before a valid resume.

Automatic experimental English display, optional editing and tap-to-Hear retain their existing behavior. This task did not expand the English dictionary, alter model weights, change thresholds, change feature math, change camera cadence, or modify WORD/FINGERSPELL behavior. Existing experimental mode gating remains: Settings > Recognition mode > CONTINUOUS WORD · experimental; release/default continuous flag remains off.

## Measured comparison with the previous APK behavior

The frozen packaged epoch-12 model and exact existing data/labels/vocabulary were replayed through both display policies. These are **cached-cadence diagnostic replays**, not live camera recordings, consented no-sign evaluation or unseen-signer product accuracy. The 80 sequences are from new recording families of known signers and have already been used for development. No reserved TEST labels/features/predictions were accessed or used for tuning.

| Scope / measure | Previous pending-token flush | Fixed pending-token discard |
|---|---:|---:|
| TRAIN replay utterances | 192 | 192 |
| TRAIN reference tokens | 1,464 | 1,464 |
| TRAIN gloss WER | 94.3306% | 93.9891% |
| TRAIN substitutions / deletions / insertions | 548 / 813 / 20 | 391 / 982 / 3 |
| TRAIN tokens published on no-hand observations | 186 | 0 |
| Held-out replay utterances | 80 | 80 |
| Held-out reference tokens | 485 | 485 |
| **Held-out gloss WER** | **95.6701%** | **95.6701%** |
| Held-out substitutions / deletions / insertions | 230 / 232 / 2 | 176 / 288 / 0 |
| Held-out tokens published on no-hand observations | 58 | 0 |
| Held-out exact sequences | 0 / 80 | 0 / 80 |

The app now follows its declared visibility/abstention contract. However, the fix exchanges many substitutions for deletions; it does **not** improve held-out WER. A token that is briefly observed and then lost before an observed CTC transition can now be missed. Counts of publication during missing tracking are not a calibrated false-activation rate: no labeled no-sign cohort is available, and hands-visible ordinary movement can still produce wrong words.

All 192 caches contain 7,847 observations, including 2,309 no-hand observations and 1,092 hand re-entries. The broader reset-on-dropout serving policy differs from missingness-containing training sequences: 646 of 5,538 visible-frame argmax IDs differ from continuous model context in this replay. That justified a separate experiment, not an untested deployment.

## Proposed changes that failed their gates

**Retaining missing-hand feature/model history:** both variants abstained on no hands and used identical labels, decoder and 200-ms display hold. Retaining history slightly reduced TRAIN replay WER to 93.5109%, but worsened the fixed 80-sequence result from 95.6701% to **97.7320%**, increasing empty outputs from 7 to 30 and deletions from 288 to 409. The paired gate required lower WER without more insertions on both partitions; it failed. This policy was **not integrated**. RF61 prefix/full-logit numerical checks and all 272 input feature hashes passed, so the unfavorable result was not hidden or repaired by changing data.

**Small framing-shift learning hypothesis:** the frozen 192-example learning checkpoint reproduced 10.587432% TRAIN WER. Predeclared normalized x/y shifts of +/-0.03 produced 11.338798%, 10.860656%, 10.928962% and 10.655738% WER. The worst deterioration was only 0.751 percentage points. Zero of four offsets met the predeclared 10-percentage-point sensitivity gate, which required three. Raw-to-feature synthetic parity and real mask/zero-preservation checks passed; 224 original files remained unchanged. The conditional augmentation fit was therefore not launched, and the gate was not relaxed.

The framing probe scored the existing checkpoint on RTX 3050 Laptop GPU in 13.639 measured scoring seconds. It did not train. The cached replay computations took 0.708 seconds for the initial 192-scope comparison, 0.510 seconds for paired context scoring, and 0.087 seconds for the held-out previous-APK/fix comparison. These exclude setup/imports and are not phone latency measurements.

**New fits: zero. New checkpoints: zero. Model promotions: zero.** Neither failed proposal justifies another longer same-data fit. These outcomes do not rule out other perception, scale, crop, representation or annotation problems.

## Vocabulary decision

**No new recognition classes were added.** The current output has 2,287 literal target labels plus blank 0 and UNK 1. These include spatial, classifier and variant annotations; they are not 2,287 interchangeable English words.

The exact frozen 1,500-utterance TRAIN set contains 12,423 target occurrences, three publisher signer IDs and 34 source-recording keys:

| Distinct supporting TRAIN utterances per label | Labels |
|---|---:|
| 1 | 1,182 |
| 2–4 | 626 |
| 5–9 | 246 |
| 10–19 | 124 |
| 20+ | 109 |

Thus 1,808 labels have fewer than five supporting utterances. Support counts distinct utterances; repeated tokens in one sentence do not supply independent examples. This is an association with limited data support, not proof that label sparsity alone causes the failure. Errors on well-supported labels remain severe in earlier audits.

Adding strings to vocabulary.json cannot teach recognition: output weights, real labeled sequences and evaluation must support each new class. The two label pools remain separate. No morphology stripping, invented gloss, fabricated example, changed exclusion or hidden label merge was used.

`VOCABULARY_SUPPORT.csv` contains the complete current list with model ID, distinct-utterance support, token frequency, signer/source coverage, 192-subset support and existing English-equivalent mapping. The English renderer still supports 75 literal equivalents and 40 finite patterns; increasing that lookup coverage would not improve the visual model.

## Validation, preservation and APK

| Check | Status |
|---|---|
| Host tests | PASS: 301 tests, zero failures/errors/skips; includes 22 continuous-core tests |
| Independent code review | PASS within scoped lifecycle/abstention fixes; no critical/high regression identified |
| APK and Android test APK build | PASS; APK build command took 150.288 seconds |
| APK signature/version inspection | PASS; Android APK v2 signature verifies |
| Frozen protection | PASS: 210 source/package/asset files hash-identical; all 528 previous APK asset entries retain sizes/CRCs; continuous assets also compared byte-for-byte |
| Experimental model recognition quality | PARTIAL: held-out WER unchanged at 95.67%, zero exact sequences |
| Further proposed training | BLOCKED by failed discriminator gate; no unnecessary fit launched |
| Qualified label/visibility review, consented no-sign/streaming acceptance | BLOCKED: no qualified reviewer or new participant permissions established |
| Current phone smoke | BLOCKED: ADB device list empty; no installation or camera capture occurred |

APK: `AppHear-continuous-refined-20261021.apk` in this evidence directory.

- Version: `1.0-continuous-refined-20261021`; application ID `com.example.apphear`; minimum Android API 26, target 36.
- Bytes: 852,132,332 (812.657 MiB).
- SHA256: `a547e867134f2be80aa1a54265b3c6f3eac524588de55670c009ee1995a8493c`.
- Existing model, vocabulary, English assets, avatar/GLB assets and permissions retained. No new runtime dependency.
- Credentials-free guest/offline debug build. Existing Android TTS is available; no cloud-service key is embedded.
- Previous APK remains at `before/AppHear-previous.apk`, SHA256 `97ad71071780935137206e89ac25b218206e7696fc236765f7c75d1215e0aefa`.
- Changed app files: ContinuousAslCore.kt, RecognitionActivity.kt, ContinuousAslCoreTest.kt and the debug version suffix in app/build.gradle.kts. Before/after snapshots and source_changes.diff are saved.

All earlier datasets, manifests, checkpoint selections, exclusions, research results and production assets remain preserved. The existing dataset quantity is retained; no new dataset download or larger acquisition target was introduced. The new APK was not installed on a phone.

## Smallest next step toward actual quality improvement

Reuse the existing 12-event TRAIN review package at `experiments/continuous_asl/20261020T141110Z_alignment_observability_review01/review_manifest.csv`. An authorized qualified ASL reviewer needs to determine whether the short publisher intervals, missing articulators and literal targets provide valid learning supervision. Existing computational joins passed, but linguistic/visual adequacy is unresolved; do not invent these judgments.

After permissions are established, collect natural continuous phrases that repeat a small set of existing reviewed signs across independent signers and recording sessions. Reuse `experiments/continuous_asl/20261020T004837_separate02/pilot_package_20261020T014259`, preserve annotation provenance and freeze signer/session boundaries before fitting. Include consented hands-visible no-sign movement, missed-sign and boundary annotations. This is a targeted support/quality step, not a demand for more unique vocabulary or another bulk download.

Before claiming improvement, require fixed-protocol independent gloss WER and exact sequences, missed signs, false activations per no-sign minute, boundary/repeat errors and measured camera-to-text p50/p95 latency/cadence. Current repeated development use cannot substitute for that evidence. No checkpoint should be promoted on the basis of these software fixes or TRAIN-only probes.

Exact verification/build/device commands are recorded in `commands_and_results.txt`. See `VALIDATION.json`, `RUNTIME_FIX.md`, `RUNTIME_REPLAY_RESULTS.json`, `continuity_replay/REPORT.md`, `continuity_replay/safety_before_after/REPORT.md`, `VOCABULARY_SUPPORT.csv` and the separate framing-control report.