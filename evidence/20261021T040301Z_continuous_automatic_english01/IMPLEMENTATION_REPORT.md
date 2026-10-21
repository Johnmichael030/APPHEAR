# AppHear automatic continuous-ASL output — implementation and APK

Date: 2026-10-21. Overall status: **PARTIAL**.

## Delivered behavior

Settings > Recognition mode > CONTINUOUS WORD · experimental enables automatic display. Settled CTC glosses appear without a review dialog. Supported literal patterns are rendered as English automatically. Edit text (optional) opens the current English text; Save edit replaces it and recognition resumes. Cancel retains it. Hear speaks the currently displayed text on an explicit tap without opening editing. Clear resets this mode. Long output can be scrolled.

The separate raw gloss display remains available. WORD, fingerspelling, model weights, feature math, camera ownership, original assets and frozen research results retain their existing contracts. No model was trained, selected or promoted. No recording, dataset acquisition, participant-data processing or TEST evaluation occurred.

## Acceptance status

| Requirement | Status | Evidence / limit |
|---|---|---|
| Stable automatic sequence display | PASS engineering / PARTIAL recognition | Settled CTC token runs are appended once. Randomized synthetic tests preserve greedy CTC order, short peaks and blank-separated repeats. This does not establish correct recognition or calibrated no-sign behavior. |
| Gloss-to-English stage | PARTIAL | Offline exact lookup: 75 literal word equivalents and 40 finite phrase templates. These are manually authored engineering rules, not a trained translator or qualified linguistic review. 2,212 of 2,287 literal target labels have no single-label mapping. |
| Automatic English display; optional editing | PASS host / PARTIAL device | No beginEdit/applyEdit call is required for automatic text. Revision/lifecycle guards protect optional edits. Hear uses a snapshot rather than requiring review. Full UI/live-camera device checks are pending. |
| Build and host checks | PASS | 293 JVM tests, zero failures/errors/skips. Full debug APK and existing Android test APK build successfully. APK signature verifies. |
| Frozen artifact preservation | PASS | 173 files hash-identical, including frozen continuous package, selected checkpoint and main assets. All 526 previous APK asset entries retain sizes and CRCs. Exact model/vocabulary/manifest bytes match the frozen source. Only two new output assets were added. |
| Reliable general recognition / translation | BLOCKED | Existing packaged checkpoint still has 95.67% WER on the historical 80-sequence held-out protocol; this is not unseen-signer/streaming acceptance. No independently evaluated English translator, qualified review, consented no-sign cohort or streaming quality evidence was added. |
| Current physical-device validation | BLOCKED | ADB reported no devices. No main-app install, camera use or phone data change was performed in this task. Earlier synthetic numerical model parity remains separate evidence. |

## How stabilization works

The new `continuous_asl_output/policy.json` is separate from the frozen model package. A nonblank CTC run is published when it ends or has lasted 200 ms, whichever happens first. A short nonblank peak is retained when its run ends; it is not discarded by an arbitrary minimum-duration confidence filter. Blanks rearm repeated labels. Missing hands close only an already observed run and then abstain. Long acquisition gaps or pauses discard an unsettled run rather than replaying it across a new session.

A 1,200-ms blank/no-hand idle interval finalizes the current display segment. This is a display heuristic, not a verified linguistic sentence boundary or sign/no-sign classifier. Current-segment English can update as a supported phrase becomes complete. Completed display segments remain fixed. The 128-token bound and character stop keep memory bounded; English stops after a complete label at the character limit rather than cutting a literal annotation label. Clear or optional editing resumes at the limit. User edits are limited to 4,096 characters.

Both timings are explicit, uncalibrated engineering settings, not thresholds optimized on reserved data. They have not been accepted for recognition-quality improvement. The 61-observation causal model context and genuine 100-ms sampling policy are unchanged. No padding, resampling, dropped feature block, new GPU delegate or runtime dependency was added.

The frozen manifest's `automatic_commit: false` remains intact: this checkpoint still cannot commit into the WORD sentence builder, history or production acceptance path. The newly authorized automatic-display policy is a separate UI layer; it does not override model eligibility or enable automatic speech/history.

## Limited English contract

Literal example: `IX-1p WANT WATER` -> `I want water.` Another packaged rule: `IX-1p NOT UNDERSTAND` -> `I do not understand.` These are synthetic engineering examples, not measured camera recognition or qualified ASL translation evidence.

An exact supported sequence uses its finite English phrase. Otherwise supported labels use lexical English equivalents in their original order and the UI explicitly states that word order is retained. Every unsupported label is shown as `[untranslated: LABEL]`, including `<UNK>`, classifier labels, unrecognized variants and unresolved spatial indices. No stripping of annotation morphology, inferred tense, guessed third-person referent, question inference or universal word reorder rule occurs. A mixture is marked Partial English. Raw glosses remain separately visible. There is no network translator and no claim of unrestricted English translation.

The existing reviewed Text-to-ASL grammar registry has zero phrase entries and provides no verified reverse translator. Reversing it, or treating English dataset translations as ASL gloss targets, would not supply a validated translation model. A general translator and linguistic evaluation are still needed before the complete product requirement can pass.

## APK

- File: `AppHear-continuous-auto-English-20261021.apk` in this evidence directory.
- Application ID: `com.example.apphear`.
- Version: `1.0-continuous-auto-20261021`, versionCode 1; debug, opt-in continuous mode.
- Minimum Android API 26; target API 36.
- Size: 852,132,332 bytes / 812.657 MiB.
- SHA256: `97ad71071780935137206e89ac25b218206e7696fc236765f7c75d1215e0aefa`.
- APK v2 signature verified; existing local debug certificate.
- Model assets unchanged: 813,449 bytes. New English/display assets: 7,813 bytes.
- The whole APK still contains existing avatar/application assets; no GLB or asset removal/compression was performed.
- Credentials-free guest/offline configuration; existing Android TTS remains available. Cloud-service keys are not embedded.
- Previous APK preserved at `before/AppHear-previous.apk` (SHA256 `bbef4b37d7067d092a726c55ab9fe98a4761d793442deaa992a1c0633ac93296`).

## Validation and next step

`build.log`: full build, success, 121.195 seconds. `asset_contract_tests.log`: final test run, success; 293 tests. Tests include randomized CTC equivalence, short peaks, long runs, missing hands, real cadence, gaps/duplicate timestamps, idle segment isolation, automatic English, unsupported labels, edit replacement/cancel/stale revisions, mode exit, output limits and actual packaged language rules.

`VALIDATION.json` records package/integrity results. `apk_signature.txt` and `apk_badging.txt` record independent APK inspection. All host/instrumented fixtures are synthetic; none is ASL accuracy evidence. Existing Android numerical parity test compiles, but was not rerun on hardware because no device was detected.

Next: install the delivered APK with data preserved and verify Settings persistence, automatic text without a dialog, optional edit/cancel, Hear/Clear, scrolling, pause/resume, lens switch and switching back to WORD/FINGERSPELL. No participant recording should be performed without applicable consent. Recognition quality, no-sign false activations, complete-sequence accuracy, boundary behavior and camera-to-text latency require separate consented streaming evaluation. General English translation requires an independently evaluated translator and qualified language review.

See `commands_and_results.txt` for exact build/integrity/device commands. No installation was performed by this task.