# TRAIN-only alignment and temporal-observability review

Generated 2026-10-20T22:20:36.309225+08:00 (Asia/Shanghai).

Computational review is complete; qualified visual/linguistic adjudication is blocked. The user explicitly confirmed no qualified reviewer is available. No training, model inference, perception execution, video decoding, acquisition or cadence comparison occurred.

## Scope and evidence reuse

The prior READ_ONLY_FAILURE_AUDIT.md and commands_and_results.txt were read first. Existing exact feature reconstruction, metric and normalization checks were reused. The frozen fold SHA256 is 87442a6f1241860e369a5c8f0768d583b3a4058f5a9ad03dcc135532c419c72f; universe manifest SHA256 is 04fb7c6c26bc934d6618902ceeaa2ee69e19d0c12e5bfe922e5743c8518c0ebe.

Only the 1,500 internal TRAIN utterances were analyzed. IDs equal the frozen fold IDs and belong to the 1,706 global TRAIN universe. Shared train_dev manifest metadata was read for membership; only TRAIN publisher rows were retained for analysis. No held-out examples or predictions influenced selection. Prior pilot artifacts, including saved held-out reports, were hashed solely for preservation, not parsed as review evidence. No reserved TEST directory, payload or manifest was enumerated, opened, hashed or decoded.

All new writes are confined to D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T141110Z_alignment_observability_review01. Existing labels, identities, exclusions, manifests, features, checkpoints, model selection, production/Android/GLB assets and reports were not edited.

## Deterministic selection

Reconstructing cached masks produces A=880, B=1,440, C=6,492 and MIXED=3,611 over 12,423 TRAIN events. These categories are disjoint and exhaustive.

- A: no retained frame inside the inclusive event interval.
- B: retained frames exist, but neither hand has an observed flag on any such frame.
- C: retained frames exist, and every retained frame has at least one observed hand. This is an operational cached-mask control, not proof of complete articulator/linguistic observability. Both hands need not be observed.
- MIXED: some but not all retained frames have observed hands. These events remain unchanged in the dataset, outside this review selection.

Stable event key is sample_id:source_row:event_id. Ordering uses SHA256 of the prefix apphear-alignment-review-v1| plus that key. A anchors prioritize unused recording families then unused masters, then smallest hash. For each anchor, B/C prioritize the same family, then minimize the sum of absolute log2 ratios for inclusive duration in frames, distinct-utterance label support and sequence observation length. Unused master is a tie preference, followed by hash. Twelve unique utterances are required. No random seed or model outcomes are used. selection.json preserves exact IDs, candidate digest, algorithm and matching ratios.

The selection spans four families, 7 masters and 3 signer IDs. All B/C matches share their anchor family. All C matches have identical duration and target support to A. Two B matches have twice the A duration and different labels: KNOW versus IX-loc:j, REALLY versus (1h)HAVE; support ratios 19/24 and 62/58. Observation-length ratios range approximately 0.644–1.081. Masters need not differ within a family. These are approximate controls, not exchangeable causal matches. Hash selection is not representative sampling and does not establish rare-label behavior.

## Computational alignment

All 12,423 TRAIN labels, publisher event IDs/CSV rows, clip/master names, collection/utterance IDs and inclusive event/utterance endpoints match publisher CSV rows. Annotation type is AUTHENTIC_PUBLISHED_DOMINANT_HAND_OCCURRENCE_EVENTS. These occurrence intervals are not independently adjudicated whole-sign extents. Nondominant-only publisher omissions, literal morphology and Hidden flags remain unchanged.

Publisher master coordinates are inclusive; subtract utterance master start to obtain zero-based native clip coordinates. Native fps is 30000/1001. Inclusive duration is (end-start+1)*1000/fps; last-frame timestamp differs from end-exclusive display time. All TRAIN arrays have indices 0,3,6,... within their cached frame count. Maximum timestamp disagreement with frame_index*1000/fps is 3.64e-12 ms, floating-point scale. This confirms stored arithmetic, not independent visual alignment of source frame zero.

For the 12 selected records, feature timestamps, embedded transcripts and source origins agree with cached raw metadata. Feature hand-observed/body-available channels agree exactly with raw masks. Effective hand validity correctly equals raw_joint_valid AND hand_observed; body validity matches directly. Raw joint-valid flags can remain true when hand-observed is false by design. This is not a new mask bug or proof of physical hand presence. Prior exact full-feature reconstruction remains the wider evidence. Independent scalar interval checks agree with vectorized membership for all 12. No computational coordinate/join/mask discrepancy was found.

|ID|Group|Literal target|Native clip interval|Duration ms|Distinct TRAIN support|Kept frames|Any/both hand fraction|
|---|---|---|---|---:|---:|---:|---|
|R01|A|IX-loc:i|[31, 32]|66.733|218|0|N/A / N/A|
|R02|B|IX-loc:i|[47, 48]|66.733|218|1|0.0 / 0.0|
|R03|C|IX-loc:i|[33, 34]|66.733|218|1|1.0 / 1.0|
|R04|A|IX-loc:j|[113, 113]|33.367|24|0|N/A / N/A|
|R05|B|KNOW|[63, 64]|66.733|19|1|0.0 / 0.0|
|R06|C|IX-loc:j|[60, 60]|33.367|24|1|1.0 / 0.0|
|R07|A|(1h)HAVE|[175, 175]|33.367|58|0|N/A / N/A|
|R08|B|REALLY|[387, 388]|66.733|62|1|0.0 / 0.0|
|R09|C|(1h)HAVE|[300, 300]|33.367|58|1|1.0 / 0.0|
|R10|A|IX-1p|[32, 32]|33.367|361|0|N/A / N/A|
|R11|B|IX-1p|[18, 18]|33.367|361|1|0.0 / 0.0|
|R12|C|IX-1p|[33, 33]|33.367|361|1|1.0 / 1.0|

All selected durations are 33.367–66.733 ms, below the actual 100.1 ms cadence. A intervals contain one/two native frames but no retained frame; each lies between neighboring retained samples. B and C retain exactly one frame per selected event. Thus sampling phase can remove direct observations inside publisher intervals; this does not establish loss of all sign information.

|Group|Candidates|Selected|Mean duration ms|Mean label support|Kept frames|Any hand fraction|Body fraction|Context any hand fraction|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|A|880|4|41.708|165.25|0|N/A|N/A|0.6875|
|B|1440|4|58.392|165.0|4|0.0|1.0|0.6875|
|C|6492|4|41.708|165.25|4|1.0|1.0|1.0|

Body availability is 100% in B/C focal retained frames and neighboring context for all three groups. Context uses nearest two retained frames before and two after, within the same utterance. Mean contextual any-hand availability is 68.75% for A/B and 100% for C. A inside fractions are undefined, not zero. B has no effective refined-hand geometry inside the interval, while body/context survives. C may have only one observed hand/frame. event_level_findings.csv preserves exact masks, raw/effective joint counts, body validity, neighboring frames and Hidden flags. Hidden semantics remain unresolved.

## Qualified review

BLOCKED: user confirmed no qualified ASL reviewer is available. Zero source videos were viewed, decoded or semantically adjudicated. Empty visual flags mean NOT REVIEWED.

review_manifest.csv references original authorized TRAIN clips and hashes, raw/feature paths and hashes, publisher rows, literal labels, master/relative bounds, fps and times. It supplies +/-1 second original clip context bounded to the same utterance. No altered media/frame copies were created.

A later authorized qualified reviewer should inspect the focal interval followed by natural neighboring/utterance context. Document qualification/authorization, date, visible articulators, both hands where required, neighboring context, boundary plausibility, target semantics, rationale and direct evidence versus inference. Allowed multi-flags: OBSERVABLE, PARTIALLY_OBSERVABLE, NOT_OBSERVABLE, ANNOTATION_BOUNDARY_QUESTION, TARGET_SEMANTICS_QUESTION, INDETERMINATE. Never substitute a neighboring label. Record later review in a new version/directory, preserving this unresolved manifest.

## Hypotheses and decision

|Question|Classification|Evidence/limit|
|---|---|---|
|Cadence skips annotated intervals|CONFIRMED|880/12,423 TRAIN events (7.08%); selected A scalar checks pass. These are publisher intervals, not independently verified whole-sign extents.|
|Retained intervals lack direct refined-hand geometry|CONFIRMED|1,440 more events (11.59%); selected B masks/channels agree. Combined A/B: 2,320 (18.68%). Population counts come from full TRAIN metadata, not extrapolation of 12 examples.|
|Temporal/refinement loss contributes to recognition failure|POSSIBLE|Short spans and missing effective hand geometry support a limitation; surviving context and unresolved annotation semantics prevent a causal claim.|
|Physical occlusion/off-frame hands versus perception uncertainty explains B|BLOCKED|Cached masks cannot distinguish these; qualified source review unavailable.|
|Metadata/mask wiring error explains B|NOT_EVIDENCED|Selected raw-to-feature observed/valid/body checks pass under actual gating contract.|
|Publisher/extracted coordinates disagree arithmetically|NOT_EVIDENCED|All TRAIN CSV/offset/index/timestamp checks pass. Visual boundary correctness remains BLOCKED.|
|Controls verify positive cached-mask computation|CONFIRMED|Four C events and corresponding feature channels agree; full articulator adequacy unresolved.|
|Cadence-only comparison worth planning|POSSIBLE, conditional|Direct interval omissions support a targeted coverage/cost question; qualified review is still needed before treating cadence as a recognition remedy.|

Cheapest next step: qualified review of these frozen 12 references, especially whether one/two-frame publisher spans mark full signing extents or narrower events and whether B reflects occlusion or detector failure. Do not infer causation, population recognition improvement, source transfer or unseen-signer generalization.

## Cadence-only plan — not executed

If qualified review confirms meaningful temporal loss, seek explicit authorization for a new research directory comparing the existing 10 Hz policy with every native frame (~29.97 Hz) on the exact 12 TRAIN utterances in review_manifest.csv. Freeze their full original targets, IDs and media/annotation hashes. Change only cadence; keep coordinate origin, orientation, perception weights/settings, feature math/width, label/vocabulary conventions and normalization policy fixed. No fit is part of this comparison.

Use existing 10 Hz arrays as reference; new extraction outputs must be separate. Verify native index monotonicity, inclusive origin/counts, timestamps and unchanged literal joins; reject discrepancies, never trim or repair. Compare focal-event and full-utterance coverage, any/both-hand/effective-joint/body availability and context, extraction wall time, frames/second, storage and peak GPU/host memory. Keep analysis TRAIN-only, with no held-out selection, reserved TEST or production access. Higher cadence may not repair missing hands; measure it. Do not claim WER improvement from coverage.

The confirmed gradient-accumulation-tail defect remains a separate issue; this review establishes no connection. Do not bundle an optimizer correction, architecture change or label/vocabulary change with a claimed cadence-only ablation. No extraction/comparison/training is authorized by this plan itself.

## Hardware and timing

GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, 3962 MiB. Existing PyTorch 2.8.0+cu128 reports CUDA 12.8 available. NumPy 1.26.4 is installed; TensorFlow 2.20.0 is listed by metadata only, CUDA support unverified. CuPy/Numba are absent. No install/update occurred.

CPU: 16 logical threads; torch reports 8 CPU threads. RAM at inspection: 15.42 GiB total, 3.33 GiB available. CIM access was denied; GlobalMemoryStatusEx supplied RAM. Arrays were cached on disk at entry, not loaded in this new process; OS page-cache residency unknown. Cached indices/masks during analysis occupied 4,668,755 bytes.

CPU cached-array/annotation review took 4.088 s, including loading/hash/event reconstruction. Successful analysis including scoped hashes/exports: 6.294 s internally, 7.854 s shell wall. Hardware inspection: 5.486 s internally, 7.543 s shell wall. GPU numerical work was not used; no benchmark or speedup claim. Small per-utterance arrays and 12-event final work do not justify transfer/initialization overhead. All 12 memberships have independent scalar CPU verification. torch was used only for capability queries; no tensor/model was created.

First diagnostic attempt failed an assertion comparing raw joint validity directly to effective feature validity. Source inspection established the required AND-with-observed rule. Only this new script was corrected; the successful run passed. No project feature, label or mask was changed; the failed attempt wrote no CSV/report output.

## Integrity and limitations

SHA256, byte size and nanosecond-mtime before/after checks passed for 1637 existing files (127,812,376 bytes): all 76 previous pilot artifacts/checkpoints, all 21 prior audit artifacts, 1,500 TRAIN raw arrays, selected original features/videos/signer-provenance references, source annotations/manifests and three relevant pipeline source files. Selected reference count 36 means 12 each of feature/video/signer references; repeated provenance paths are deduplicated in actual hash totals. Pilot/audit membership has no additions/removals. Source hashes match the prior allowlist.

input_files_before.json and input_files_after.json declare exact scope; integrity_results.json records results. verify_review.py rechecks only these paths and new checksums. Production/Android/GLB assets were not written, but were not independently hashed; no whole-workspace integrity claim. Hashing selected video bytes for preservation is not decoding. No inference/training/perception, downloads, external services, global environment/GPU changes or Git commands occurred. No reserved TEST enumeration/access. Unlogged unrelated historical process behavior cannot be certified.

## Commands

Working directory: D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite

& 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261019T155428Z_cslr01\runtime\venv\Scripts\python.exe' -B 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T141110Z_alignment_observability_review01\inspect_hardware.py'
& 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261019T155428Z_cslr01\runtime\venv\Scripts\python.exe' -B 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T141110Z_alignment_observability_review01\review_cached.py'
& 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261019T155428Z_cslr01\runtime\venv\Scripts\python.exe' -B 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T141110Z_alignment_observability_review01\write_report.py'
& 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261019T155428Z_cslr01\runtime\venv\Scripts\python.exe' -B 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T141110Z_alignment_observability_review01\verify_review.py'

First three are provenance/construction commands, intentionally exclusive-create and not rerunnable over completed outputs. To validate this completed package, run only verify_review.py. commands_and_results.txt includes initial PowerShell, denied CIM queries and failed/successful analysis attempts.

PARTIAL — computational evidence improved, but qualified visual/annotation review remains unresolved
