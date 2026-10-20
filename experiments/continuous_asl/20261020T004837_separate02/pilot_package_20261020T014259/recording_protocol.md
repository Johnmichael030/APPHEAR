# Streaming pilot recording protocol v1 — preparation only

**Do not start a camera, record, upload, annotate or extract identifiable participant data until the consent/permissions checklist has been established by the responsible human.** No participant has been enrolled or recorded by this package. It does not install an APK or change AppHear. All samples described below are planned, not acquired.

## First signer: small preflight, then full pilot

Use an adult qualified ASL signer whose proficiency is confirmed by a qualified reviewer. Record the qualification reference, not unnecessary personal credentials. A coordinator assigns a pseudonymous signer ID and separately audits the person's identity against known current-model/segmenter training, frozen isolated partitions and previous phone-design cohorts. Different IDs alone do not establish different people. Assign `PILOT_1_EVAL`; do not allocate these recordings to training/calibration or choose a checkpoint from them.

After permissions pass, collect one short session containing two diagnostic signs, one natural connected phrase and a visible-idle interval. Review glosses and source-time boundaries and validate the350/378 feature/annotation joins before collecting the remaining blocks. Keep this small preflight even if it exposes failure; mark unusable/unreviewed data explicitly. Do not discard failures to make the pilot look successful.

Use a tripod, landscape720p or better, nominal30FPS, ordinary indoor even lighting and a plain contrasting background. Include full face, shoulders, elbows, both hands and signing space; ASL facial information must not be cropped away under an anonymity claim. Avoid bystanders, identifying documents and spoken names. Disable audio unless separately consented and needed; do not collect GPS/IMEI. Use local storage with cloud backup/sync disabled for this collection. Record front/rear camera, mirroring, rotation, dimensions and actual timestamps. Preserve original media; the observation producer must document how it obtains `UPRIGHT_UNMIRRORED_SOURCE_RGB`.

Use an existing authorized continuous video capture method. The isolated Recorder's approximately3-second clips do not substitute for uninterrupted sessions. Do not build/install a new recorder or debug APK as part of this package. If the existing capture cannot export source timestamps or the required current-pipeline logs, record that blocker and stop the affected pipeline step. A separate camera video supports offline feature checks, not live normal-app latency. Never fabricate logs or source clocks.

Keep recordings as uninterrupted blocks of at most3minutes, sharing a stable session ID; each gets a unique recording ID. Preserve pre-roll, between-sign behavior and post-roll. Do not stitch isolated clips into purported natural continuous signing. Nominal30FPS is a recording setting, not a measured completed processing rate. Genuine dropped frames and slow processing remain visible in the frame/time index; do not interpolate observations or hide gaps.

## Diagnostic sequence prompts

These token instructions elicit repeat/rearm/motion conditions; **they are not ground truth**. The signer may decline an unnatural construction. Annotators record what was actually signed, including variation, omissions and uncertainty. The following tokens were found in the existing legacy236 vocabulary during preparation, but the exact current recognizer vocabulary must still be pinned before lexical scoring.

| ID | Requested diagnostic sequence | Condition |
|---|---|---|
| D01 | HELLO → HOME | Comfortable pause with hands lowered between signs; two repetitions |
| D02 | HELLO → HOME | Hands remain naturally up between signs; two repetitions |
| D03 | HELLO → HELLO | Genuine repeated sign with a natural release; two repetitions |
| D04 | SORRY, then hold a relaxed hand position | Separate lexical sign end from post-sign hold; two repetitions |
| D05 | GOOD → HAVE → KNOW | Sign at a comfortable rate, then repeat naturally; no metronome |

Record approximate instructions/block markers separately from reviewed sign boundaries. Do not score the prompted number of repetitions as performed signs. Classifiers, simultaneous constructions and fingerspelling are allowed in natural responses but require appropriate expert labels; uncertain/unsupported content remains explicitly unscored, not converted to English glosses.

## Ten natural connected-ASL prompts

The English descriptions below are topics only. Ask the signer to produce their own natural ASL, typically3–5 lexical signs where appropriate, without enforcing English word order or requiring named gloss tokens. Preserve facial/nonmanual grammar and coarticulation. Capture all responses, including retakes, and keep retake links.

| ID | Topic/instruction |
|---|---|
| P01 | Greet a friend and indicate you are going home. |
| P02 | Ask where school or work is. |
| P03 | Describe something you did today. |
| P04 | Ask a friend for help with an ordinary task. |
| P05 | Explain that you know or do not know where something is. |
| P06 | Describe water or another everyday object in the room. |
| P07 | Apologize and explain a small everyday misunderstanding. |
| P08 | Describe a plan for later today. |
| P09 | Repeat or clarify part of a message naturally. |
| P10 | Tell a short everyday message chosen by the signer. |

Also collect NARRATIVE01, a30–60-second natural account of a routine day, and NARRATIVE02, a30–60-second short explanation/conversation response. Allow natural pauses and signs outside the vocabulary; retain OOV status. Authentic ordered glosses and individual-sign boundaries must come from qualified review of the actual recording, never these English prompts. Optional English translations belong in a separate field/file and must never populate `glosses`.

## Negative and transition blocks

Per signer, aim for **15minutes of adjudicated no-sign exposure**, with approximately3minutes NO_VISIBLE_HANDS,4minutes VISIBLE_IDLE,4minutes FACE_ADJACENT non-sign activity, and4minutes ACCIDENTAL_MOTION. Split into short continuous blocks with breaks. Let the signer choose comfortable ordinary behavior; a block instruction is not proof every frame is negative. Review the whole interval and remove any actual signing/uncertainty from the negative denominator explicitly.

Separately record one short OOV-signing block and one incomplete/transition block. OOV is genuine signing, not hand absence; do not count it as no-sign minutes. Mixed/ambiguous transitions receive UNCERTAIN intervals until reviewed. Do not ask participants to expose private routines or uncomfortable motion. Instructional intent, model blank output and landmark absence are not negative truth.

## Metadata and annotation workflow

For every block: record recording/session/signer IDs, pilot role, prompt ID, retake parent if any, device/camera pseudonym, resolution, nominalFPS, source-clock origin/units, start/end source times, mirror/rotation semantics, environment/occlusion conditions, producer/config/model references and hashes, raw file hash, decoded-frame index, observed frame indices, and dropped/stalled periods. The full source frame index is strictly ordered with contiguous frame IDs; raw observations may be a genuine timestamp-matched subset. There is no timestamp rounding/deduplication repair.

Store source video and permissions in restricted local storage **outside this repository**. After explicit LOCAL_PROCESS authorization, use the existing Candidate01 producer and raw.v1 representation. This package does not launch extraction. Save full causal378 segmentation features for the observed stream. Recognition-v2 features are **single-sign windows**, not one window for a multi-sign phrase. Reference-boundary windows in this package are oracle diagnostic inputs only; they cannot pass automatic event selection or end-to-end recognition gates.

Two qualified reviewers label independently while blinded to predictions: actual gloss tokens, sign onsets/offsets, uncertainty, OOV and no-sign intervals. Retain both reviews and an adjudication reference; the validator consumes the finalized interval table without sorting or repairing it. All bounds are source-time half-open intervals `[start_ms,end_ms)`. It rejects overlapping sequential labels; complex simultaneous annotation needs a separately reviewed format rather than coercion into this schema. Unavailable glosses are represented by UNCERTAIN/OOV intervals and omitted from scorable sequences, with excluded duration/counts disclosed. Never invent a transcript to make validation pass.

## Second independent signer

Begin only after first-signer permission checks, source-time joins, observation/feature recipe checks and qualified annotation review succeed. Assign `PILOT_2_EVAL`; audit both real identity and aliases against signer1 and the historical registry. Repeat the same core prompts and negative categories, with the signer's own natural wording. Add one documented lighting/background variation within the consented setting; do not confound every condition with signer identity. Keep both evaluation-only, never use pilot2 to tune thresholds after seeing pilot1. Check per-signer results separately and preserve failures.

This two-signer pilot checks acquisition/annotation readiness, not adequate population accuracy. Thirty no-sign minutes with zero false commits is only the prior Poisson planning minimum; no false-activation or WER measurement occurs in this preparation. The proposed later gates remain in `../continuation_20261020T013509/acceptance_protocol.md`.
