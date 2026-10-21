"""Compose reports only from retained evidence. Never fills missing gates with metrics."""
from __future__ import annotations
from collections import Counter
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import sys
import time
import numpy as np
from prepare_features import ROOT,SCHEMA,checked_run,sha

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);args=p.parse_args();run=checked_run(args.run_root)
    # Hardware, test inventories and the user's blocked-data answer below are a
    # historical narrative for this run, not generic results for future runs.
    if run.name!='20261019T155428Z_cslr01': raise ValueError('REPORT_TEMPLATE_IS_PINNED_TO_THE_AUDITED_RUN; fresh runs retain machine-readable evidence and need their own narrative')
    def read(name: str) -> dict: return json.loads((run/name).read_text(encoding='utf-8-sig'))
    def write(name: str,text: str) -> None: (run/name).write_text(text.strip()+'\n',encoding='utf-8')
    quality=read('data/processed/full/quality.json'); rows=read('data/processed/full/manifest.json')
    training=read('logs/training_status.json');evaluation=read('evaluation_results.json');parity=read('model_parity.json')
    perf=read('performance_results.json');replay=read('logs/replay_profile.json');gpu=read('logs/gpu_probe.json')
    frameworks=read('logs/framework_probe.json');baseline=read('logs/baseline_s2.json');preservation=read('preservation_verification.json')
    metadata=read('metadata_quality.json');dups=read('logs/visual_duplicates.json');hazards=read('logs/regression_hazards_02.json')
    log=list(csv.DictReader((run/'training_log.csv').open()));best=read('checkpoints/best.json')
    counts=Counter(s for r in rows for s in r['glosses']);train_vocab=read('vocabulary.json')['tokens']
    total_frames=sum(r['decoded_frames'] for r in rows);seconds=sum(r['container_duration_seconds'] for r in rows)
    hand=[sum(r['hand_observed_fraction'][i]*r['sequence_length'] for r in rows)/944 for i in range(2)]
    body=sum(r['body_available_fraction']*r['sequence_length'] for r in rows)/944
    raw_movies=list((run/'data/raw/ncslgr/videos').glob('*.mov'));raw_bytes=sum(p.stat().st_size for p in raw_movies)
    peak_vram=max(int(r['peak_allocated_vram_bytes']) for r in log)
    repo=ROOT.as_posix();runpath=run.as_posix();tool=f'{repo}/tools/continuous_asl'
    write('environment_report.md',f'''# Environment evidence

Measured on 2026-10-20 Asia/Shanghai. Windows 11 Home Single Language, build 26200 x64; PowerShell 7.6.5 Core. AMD Ryzen 7 4800H, 8 cores / 16 logical processors; physical RAM 16,556,486,656 bytes. Initial available physical memory was 3,399,760 KiB. Automatic C: pagefile: allocated 12,582 MB, initial current 452 MB, peak 6,578 MB. Current snapshots are in logs/environment_*.json; initial exact outputs are in logs/discovery_initial.json.

Initial disks: C: NTFS 486,983,143,424 bytes total / 26,722,525,184 free; D: NTFS 1,000,202,039,296 total / 188,578,664,448 free. All new environment/cache/data paths are inside `{runpath}` on D:. Full How2Sign frontal training video (published 290 GB) would exceed available D: capacity and was not acquired.

Actual NVIDIA device: **{gpu['gpu_name']}**, 4,096 MiB VRAM, driver 591.86; integrated AMD Radeon also present. Initial NVIDIA utilization/memory were 0% / 0 MiB, temperature 46 C. Driver reports CUDA capability 13.1; this is not an installed toolkit. No nvcc/toolkit was found. The isolated PyTorch wheel supplies CUDA {gpu['cuda_runtime']} and cuDNN raw version {gpu['cudnn_version']}; no driver/toolkit upgrade occurred.

Default python/python3 WindowsApps aliases fail; py reports no registered Python. An existing direct Python 3.9.13 installation has torch 2.8.0+cpu and no CUDA execution. It was preserved. Research uses Python 3.11.11 and torch 2.8.0+cu128, installed only under runtime/venv using uv 0.10.12. `uv pip` manages that venv; pip is not required inside it. Requirements are pinned in requirements.observed.txt. TensorFlow {frameworks['packages']['tensorflow']} reports built_with_cuda=false / no GPUs; ORT {frameworks['packages']['onnxruntime']} providers are {frameworks['onnxruntime_available_providers']}. Perception therefore uses CPU ORT + TFLite XNNPACK; training uses PyTorch CUDA.

GPU acceptance probe: 512x512 FP32 matrix multiply plus backward, 2 warmups / 6 measured iterations, median {gpu['median_ms']:.3f} ms, peak allocated {gpu['peak_allocated_vram_bytes']:,} bytes. Input, output and gradient are cuda:0, finite. CPU equivalent at 2 threads: median {frameworks['CPU_fallback']['median_ms']:.3f} ms. This is a tensor comparison, not ASL model throughput. Training later independently verified CUDA model/loss/gradient placement in logs/batch_probe.json.

Git 2.56.0, LFS 3.8.0, curl 8.21.0, Node and JDK17 were found. WSL distributions Ubuntu, docker-desktop and Ubuntu-22.04 are stopped; WSL was not started. FFmpeg/7-Zip were absent from PATH; OpenCV decoded the MOV samples. No Git metadata exists in this copy, so branch and baseline commit are unavailable. Historical branch names are not current proof.

Repeat discovery: `& '{tool}/Invoke-EnvironmentDiscovery.ps1' -RunRoot '{runpath}'`. Its recorded command keys include Get-CimInstance for OS/CPU/RAM/pagefile/GPU, Get-ComputerInfo, Get-Volume, Get-PSDrive, Get-Command, git status/version and nvidia-smi. Read-only system queries needed sandbox escalation; the initial denied CIM call is retained. No credentials or complete environment-variable dumps were collected. Framework commands: `python -B tools/continuous_asl/gpu_probe.py --output <fresh-output> --require-cuda` and `framework_probe.py --run-root <run>`. Source scripts provide exact tensor code and provider checks.
''')
    write('preprocessing_report.md',f'''# Preprocessing — COMPLETED diagnostic subset

Source: official BU NCSLGR public video index, authentic utterance main-gloss strings, camera 0. Only URLs mapping to one unique utterance/label were eligible. Fingerspelled strings, descriptive classifiers and ambiguous complex glosses were excluded by the frozen lexical parser; the 37 retained sequences preserve their source token spelling (including + / hyphens). English How2Sign transcripts were never training targets. This is a highly selective elicited-utterance pilot, not representative coverage of continuous ASL.

{len(rows)} / {len(rows)} videos loaded, {raw_bytes:,} bytes, {total_frames:,} decoded frames, {seconds:.3f} container seconds. All are 324x312 pixels. Native FPS ranges {min(r['native_fps'] for r in rows):.5f}–{max(r['native_fps'] for r in rows):.5f}. Sampling retains the first genuine frame at or after 100 ms since the previous selected frame, with no interpolation, duplication or carried-forward landmarks. {quality['selected_frame_count']} observations survive. Raw timestamps, original frame indices and container/fallback provenance are retained. Container speed was not independently calibrated against original capture timing; old compressed BU videos have documented timing caveats.

Exact packaged perception identities: DWPose-T `37b0f265200f8e249adba1cc9b62925f5e1252245580ed1f66f583d7bce239c0`; hand refiner `6acda74af3fbf40e68265c20c7394b2bad81a16a481dcd79ad7a081887c3d6b9`. Existing offline current-frame extraction is reused with explicit Android asset paths, CPU ORT and two-thread TFLite. Raw-v1 semantics remain upright, unmirrored genuine source RGB. No depth or 3D reconstruction is fabricated.

Experimental feature schema: `{SCHEMA}`, float32 [T,378]. Features reuse the current/prior causal extractor and retain both complete hand blocks, arm/body geometry, support/validity, real delta-time and motion deltas. Causal-prefix feature equality was checked at three prefixes for every clip. Recognition V2's whole-clip duration/progress fields are avoided. Schema equality, source/video/feature hashes, monotonic timestamps and CTC minimum lengths are checked. Configuration SHA256: `{quality['preprocessing_config_sha256']}`.

Data live separately under data/raw, data/extracted, data/processed and data/derived. Each processed NPZ uses allow_pickle=false and includes version, source IDs, unknown signer status, split, gloss type, timestamp/length, configuration hash, quality flags, license/provenance. Raw evidence is retained. Extraction refuses existing output directories; interrupted runs retain successful files and stop on the first invalid input. Use a fresh run for a full replay instead of overwriting processed data.

CPU extraction p50/p95: {quality['extraction_ms_p50']:.3f}/{quality['extraction_ms_p95']:.3f} ms per selected observation; {quality['elapsed_seconds']:.3f} seconds phase wall time including model setup. Detailed per-clip quality and timings: data/processed/full/manifest.json and quality.json.
''')
    write('dataset_quality_report.md',f'''# Dataset quality and leakage

COMPLETED: all {len(rows)} videos readable, finite features, valid chronological nonempty sequences, no acquisition failures, no exact container SHA duplicates, no decoded-pixel duplicates and no heuristic visual duplicate candidates. The visual check hashes all decoded pixels and compares five chronological dHashes with a conservative threshold; absence of candidates does not prove no near-duplicates.

Split was frozen by collection hash with seed 20261019 before training: 30 TRAIN / 7 validation clips; held-out collections ncslgr10e.xml and ncslgr10m.xml. No shared collection, source URL or exact video content was found. Official train/validation/test partitions are not supplied in this public index. TEST payload was not acquired. Signer identity is unknown: **signer leakage is UNVERIFIABLE**, and this is not a cross-signer product benchmark. No frames from a clip are split independently.

{len(counts)} observed lexical token types across {sum(counts.values())} reference tokens. TRAIN vocabulary contains {len(train_vocab)-2} lexical tokens plus BLANK=0 / UNK=1, fitted only to TRAIN. Validation has {evaluation['reference_tokens']} tokens, of which {evaluation['out_of_vocabulary_tokens']} are outside the training vocabulary. Those references remain authentic and count as errors; validation labels do not expand the trained vocabulary. Sequence token lengths: {min(len(r['glosses']) for r in rows)}–{max(len(r['glosses']) for r in rows)}; observations: {min(r['sequence_length'] for r in rows)}–{max(r['sequence_length'] for r in rows)}. No invalid CTC lengths.

Weighted observed-hand fractions (extractor side order): {hand[0]:.3f}, {hand[1]:.3f}; body available fraction {body:.3f}. These are model evidence flags, not manual occlusion truth. Low hand coverage and low-resolution historical footage are material limitations. No feature block was dropped to improve speed. Joint support and missingness remain explicit. The small simple-lexical subset omits fingerspelling, complex classifier annotation and broader nonmanual ASL information.

How2Sign inspection: TRAIN 31,165 rows / 2,192 source videos; validation 1,741 / 115; no duplicate sentence IDs, invalid timestamps, empty translations or shared source-video IDs. Annotation-to-RGB pairing was not verified because RGB was not acquired. Its English labels are not gloss-CTC eligible. BU index: 5,219 rows / 2,612 unique URLs / 38 collections; 37 strict eligible utterances. Detailed exclusion counts: metadata_quality.json. The old SSDB archive is retained but safe extraction rejected Windows-unsafe trailing-space members; it was not executed or used for training.

Missing truth: individual sign boundaries, signer identity, true non-sign footage and source timing validation. User confirmed no additional authorized local data. These gates remain blocked. No unlabeled motion was relabeled as a sign or as a negative.
''')
    write('baseline_report.md',f'''# Baselines before training

Existing production accuracy baseline: BLOCKED. Required private held-out captures/features and a compatible end-to-end word benchmark are absent. New utterance gloss CTC and the frozen 236-way isolated WORD task have incompatible labels/objectives; their numbers must not be compared as accuracy improvement. Top-1, macro-F1, per-class precision/recall, full-word confusion, original non-sign false activations and boundary F1 are unavailable.

Packaged S2 engineering baseline: PASS. Model SHA `{baseline['model_sha256']}`. All 131 frozen debug-fixture observations replay with maximum probability error {baseline['fixture_max_abs_probability_error']} and {baseline['fixture_argmax_mismatches']} argmax differences. On 944 new observations, 4-thread Windows TFLite invocation p50/p95 {baseline['research_s2_ms_p50']:.3f}/{baseline['research_s2_ms_p95']:.3f} ms. Predicted states are not ground truth. Full CameraX/native/WORD/SentenceBuilder replay is not established by this component check.

Tests before training: 56 existing word-schema/fingerspelling tests passed. A broader set had 28 further existing passes, four new model checks passed, and one existing failure due missing `artifacts/recognition_upgrade/perception_vnext/recognition_utility_236_01/phase7_production_cadence_aligned_segmenter_01/best_model.keras`. The packaged TFLite S2 is present and independently tested. No checkpoint was fabricated to make the old test pass.

Small-model control: initial random-weight CTC validation WER {evaluation['untrained_baseline']['gloss_wer']*100:.1f}% (insertions can make WER exceed 100%). An always-empty output would have 100% deletion WER on these 25 tokens. Both are diagnostic controls, not production baselines. No trained non-sign baseline exists.
''')
    write('evaluation_report.md',f'''# Evaluation — research checkpoint, DO NOT PROMOTE

Best checkpoint: `{best['path']}`, SHA256 `{best['sha256']}`; epoch {best['epoch']} of {training['epochs_completed']} completed. FP32 causal TCN, four 48-channel residual blocks, dilations 1/2/4/8, two kernel-3 convolutions per block, per-observation LayerNorm, receptive field 61 observations. Train-only mean/std with clipping [-10,10] is embedded; 39 output logits include BLANK/UNK. Greedy CTC decoding; no beam-search or validation hyperparameter sweep. Settings: training_config.yaml. Batch probe selected four using actual longest-sequence forward/backward; model/loss/gradient placement was cuda:0. AdamW lr=0.001, weight decay=0.0001, gradient norm cap=5, seed=20261019; no AMP, accumulation or gradient checkpointing was needed. Lexicographic validation WER/loss selected epoch 8; patience six caused early stop at epoch 14. All epochs have separate resumable checkpoints.

**Collection-disjoint diagnostic gloss WER: {evaluation['gloss_wer']*100:.1f}%** = ({evaluation['substitutions']} substitutions + {evaluation['deletions']} deletions + {evaluation['insertions']} insertions) / {evaluation['reference_tokens']} reference glosses. Seven validation clips; signer identities unverified; four OOV references. This is poor and does not establish useful generalization, cross-signer accuracy, or improved AppHear recognition. CPU recomputation CTC loss {evaluation['loss']:.6f}; small CPU/CUDA loss differences are recorded. Predictions and edit counts: sequence_errors.csv. token_alignment_confusion.csv is the sequence-alignment equivalent of a confusion matrix; isolated top-1 and macro-F1 are not applicable.

Robustness diagnostics keep original sequence references: drop every third genuine observation and recompute motion/delta-time -> WER {evaluation['stress_tests']['drop_every_third_observation']['gloss_wer']*100:.1f}%; explicitly mark both hands unobserved and recompute all features -> {evaluation['stress_tests']['both_hands_unobserved']['gloss_wer']*100:.1f}%. These are artificial corruption probes, not real occlusion or non-sign evaluation. The model's willingness to emit labels is not calibrated. CTC BLANK is not a safe unknown/reject class; UNK has no positive TRAIN examples.

BLOCKED: cross-signer accuracy, individual boundary P/R/F1, non-sign false accept/reject/activations, threshold/margin calibration and temporal-stability operating curves. User confirmed no further authorized local labels/footage. No arbitrary threshold or fabricated boundary target was added. See non_sign_recording_protocol.md for data requirements.

Selection decision: reject promotion. Do not run a larger grid or buy/download a larger corpus based on this tiny result. First obtain appropriately licensed, signer-identified authentic continuous gloss data and consented hard negatives, then improve landmark coverage and build a held-out streaming benchmark. Keep fingerspelling separate. Training curves are saved as CSV plus PNG/SVG. CUDA CTC backward warned that deterministic execution is unsupported; seeds and packages are fixed, but bit-for-bit replication is not guaranteed.
''')
    write('performance_report.md',f'''# Performance — measured scopes

| Measurement | p50 | p95 | Scope |
|---|---:|---:|---|
| Existing Candidate01 pose + hand extraction | {quality['extraction_ms_p50']:.3f} ms | {quality['extraction_ms_p95']:.3f} ms | 944 selected CPU frames, all 37 clips |
| Existing S2 TFLite | {baseline['research_s2_ms_p50']:.3f} ms | {baseline['research_s2_ms_p95']:.3f} ms | CPU, four threads, invocation only |
| New ONNX CTC | {perf['onnx_CPU']['p50_ms']:.3f} ms | {perf['onnx_CPU']['p95_ms']:.3f} ms | CPU, two threads, 944 real prefixes |
| New PyTorch CTC CPU | {perf['torch_CPU']['p50_ms']:.3f} ms | {perf['torch_CPU']['p95_ms']:.3f} ms | two threads, invocation only |
| New PyTorch CTC CUDA | {perf['torch_CUDA']['p50_ms']:.3f} ms | {perf['torch_CUDA']['p95_ms']:.3f} ms | preplaced input + synchronized invocation, transfer excluded |
| CPU RGB-to-gloss replay | {replay['selected_frame_ms_p50']:.3f} ms | {replay['selected_frame_ms_p95']:.3f} ms | one real 88-frame clip / 30 observations; includes features and new model |

One-clip CPU replay wall time excluding startup: {replay['replay_seconds_excluding_initialization']:.3f} s; processing capacity {replay['replay_observations_per_second']:.2f} selected observations/s. Actual configured observation cadence is capped at 10 Hz. This does **not** pass the 15 FPS product gate. It is saved-video replay without live camera, rendering, SentenceBuilder, lifecycle or Android scheduling. No Android end-to-end FPS, signed-phrase latency or device thermal stability was measured.

Replay startup (perception + ORT) {replay['initialization_seconds']:.3f} s; ONNX-only session construction {perf['onnx_session_cold_start_ms']:.3f} ms (single process/cold sample, OS file caches unspecified). New ONNX {perf['onnx_bytes']:,} bytes; checkpoint {perf['checkpoint_bytes']:,} bytes; {perf['parameter_count']:,} trainable parameters. Layer/shape/prefix checks support an adapter investigation, not mobile runtime compatibility certification. For this small workload, CPU ORT was faster than synchronized PyTorch CUDA; that does not imply GPU training was unnecessary.

Training peak allocated VRAM {peak_vram:,} bytes; peak reserved {max(int(r['peak_reserved_vram_bytes']) for r in log):,}. Actual CUDA inputs/logits/loss/gradients and finite memory-probe results are retained. Median epoch loop time {statistics.median(float(r['epoch_seconds']) for r in log):.3f} s, sum {sum(float(r['epoch_seconds']) for r in log):.3f} s; excludes environment/model initialization and includes validation. PowerShell sampled NVIDIA utilization/memory/temperature every two seconds; the job is too short for sustained-load or thermal conclusions.

Actual CPU replay worker PID {replay['memory_after']['actual_pid']}: peak working set {replay['memory_after']['peak_working_set_bytes']:,} bytes; private commit {replay['memory_after']['private_commit_bytes']:,}; peak pagefile-backed usage {replay['memory_after']['peak_pagefile_bytes']:,}. Peak working set is resident RAM, not total commit. **Training peak RAM was not captured reliably**: PowerShell observed the Windows venv launcher, whose small RSS must not be interpreted as the CUDA worker's memory. This limitation remains explicit rather than triggering another training run for a missing metric.

Main measured bottleneck: landmark extraction, with substantial hand missingness. Recommended next performance experiment is a device-measured cadence/coverage study while retaining complete feature blocks, after a valid accuracy/rejection benchmark exists. No heavier RGB backbone or default model replacement is justified.
''')
    write('model_parity_report.md',f'''# Export and transfer

ONNX export: {parity['status']}, opset 17, {parity['onnx_bytes']:,} bytes; SHA256 `{parity['onnx_sha256']}`. Standard graph operators: {', '.join(parity['operator_types'])}. Checked with onnx.checker and Windows ORT CPU.

PyTorch CPU versus ORT CPU on identical inputs from all 37 clips: maximum absolute logit difference {parity['maximum_abs_logit_error']:.9g}, mean {parity['mean_abs_logit_error']:.9g}, zero frame argmax and decoded-sequence mismatches. Predeclared CPU export tolerance 1e-4 passed. Full/prefix last-output maximum difference {parity['bounded_history_vs_full_max_abs_error']:.9g}. The five focused engineering tests also cover a 120-step synthetic prefix and the 61-step receptive-field bound. These probes are not ASL accuracy samples.

PyTorch CPU versus CUDA: max {parity['pytorch_CPU_vs_CUDA']['max_abs_logit_error']:.9g}, mean {parity['pytorch_CPU_vs_CUDA']['mean_abs_logit_error']:.9g}, zero frame argmax differences. This is a separate backend difference and is larger than the CPU export tolerance; it is not a claim of strict CUDA numerical parity. ORT CUDA EP is unavailable in this environment and was not tested.

Contract: `{SCHEMA}`; FP32 [B,T,378] -> [B,T,39] logits, dynamic B/T, chronological observations, embedded TRAIN normalization, vocabulary order frozen in export/contract.json and vocabulary.json, BLANK=0/UNK=1, no calibrated rejection. Causal 61-observation bounded history is sufficient for the last output. Main quality checks used batch=1; arbitrary batches were not certified for Android.

The existing app WORD consumer expects a [1,96,350] temporal view plus mask and 236-way word output; fingerspelling uses 405 features/28 logits. The new artifact is not drop-in compatible with either. Sharing feature width 378 with S2 does not share normalization or output semantics. Required adapter: current-frame causal features + experimental normalization/model + streaming CTC state, enforced experimental switch and original-path fallback. Boundary/rejection, signer, licensing, Android operator/provider/device and signed-phrase gates are missing. No adapter, model selection change, APK build/install, data clearing or runtime flag change was made. Disabled experiment = leave original app path running. Distribution/production rights remain unapproved.
''')
    write('non_sign_recording_protocol.md','''# Blocked non-sign and boundary gates

No usable labeled negative evaluation set was found in the acquired subset. The user confirmed no additional authorized local data. No footage was recorded and no private media was silently ingested.

For a future separately authorized capture, obtain informed consent covering research/training, storage duration, access and withdrawal; record adults who can consent, keep participant identifiers pseudonymous, and store the consent record separately. Include idle posture, ordinary hand movements, everyday gestures, sign transitions, incomplete/aborted signs, ambiguous signing, hand occlusion and variable camera/frame loss. Include complete target signing so correct-but-rejected events can be counted. Record device, capture timestamps and dropped-frame evidence. Keep signer identities for mutually exclusive train/calibration/evaluation groups and keep sequences intact. A fluent ASL annotator should mark lexical glosses and individual boundaries, explicit non-sign intervals and uncertain/unscorable intervals; do not turn unannotated gaps into negative truth. Independently review a subset and document disagreement. No count or duration is asserted as sufficient before precision requirements are chosen.

On signer-disjoint calibration data compare existing top1 commitment, confidence/margin operating curves, boundary-aware temporal stability, and an explicit unknown/reject head with valid negative training labels. Lock an operating point before held-out streaming evaluation. Report false activations/hour, false accepts/rejects, correct-but-dropped signs, gloss WER, boundary precision/recall/F1 and phrase latency with sample counts/confidence intervals. Test dropped frames and missing landmarks separately from true negatives. A CTC blank token alone is not a rejection system. No thresholds are calibrated in this run.
''')
    write('regression_report.md',f'''# Regression and release gates

84 existing tests passed; one existing test fails because its frozen Keras source artifact is missing. Five new CTC engineering checks pass; six hazard regressions pass; nine archive safety tests pass; 15 downloader policy tests pass. The earlier test-harness argument error is retained in logs/regression_hazards.txt; corrected hazards_02 is six passes. Real HTTP range resume passed and a cached sample was reused after hash verification. Completed-run -Resume was checked to start no further epochs and preserve training_log.csv. PowerShell AST and Python syntax checks are recorded separately. Full Android instrumentation and broad write-producing training/eval suites were not run blindly.

| Gate / finding | Actual status |
|---|---|
| A long ACTIVE event loses start | Reproduced with existing Python decoder/ring and the Kotlin materializer precondition: start=0, retained first=13, 13 evictions. No Kotlin/device execution claimed. |
| A recovery policy | Isolated event_guard.py returns EXPIRED_EVENT and skips recognition if any required index is absent. Never synthesizes/truncates a sign. Age-boundary, count eviction, missing commit and bad ordering tests pass. Not wired into Android; production issue remains. |
| B unconditional top1 commitment | Source verified in PhoneProductionSession.consume: SentenceBuilder adds top1; negative calibration is blocked. No arbitrary threshold patch. |
| C throughput | Component + CPU RGB-to-gloss replay measured; 10 Hz sampling / no Android means 15 FPS product gate not passed. |
| D recognition flag | RECOGNITION_ENABLED declared but unused by main runtime; source finding persists. Enabled/disabled runtime correction not implemented or tested. Existing configuration selection is distinct. |
| E merge overlap | Existing merge overwrites its source train NPZ on a disposable new synthetic fixture. New safety helper rejects identical/nested source/output paths. Original merge source remains unchanged. |
| Leakage | Collection/URL/exact-content checks pass; signer leakage unresolved. |
| Export CPU parity | PASS; no decoded differences on 37 clips. |
| Non-sign + boundary evaluation | BLOCKED, user confirmed no extra data. |
| Existing word/fingerspelling | Relevant unit checks pass; full live-device behavior is not established by unit tests. |
| Assets / manifests / avatars | All {preservation['original_files_checked']:,} original files and {preservation['original_bytes_checked']:,} bytes match hashes and modification times, including production models, Original/FemaleASL GLBs and manifests. |
| Production / release | BLOCKED. No runtime integration, enforced new Android flag, signed-phrase benchmark or authorized install. |

Original preservation list SHA256: `{preservation['baseline_sha256']}`. The first timestamp comparison incorrectly compared a deserialized DateTime to a string and flagged all times despite matching hashes; corrected UTC tick comparison confirms zero changes. Faulty diagnostic logs are retained. No original file was repaired or touched to make the verification pass.
''')
    write('review_report.md','''# Project /review and /imprint audit

Plan alignment: PARTIAL. Safe acquisition, hardware verification, causal feature reuse, a bounded GPU gloss pilot, export and truthful measurements are complete. Temporal/rejection supervision and cross-signer/device gates are unavailable. The mission's blocked-step rule permits these independent diagnostic tasks; it does not permit a production accuracy claim.

System integrity: PASS for isolation/preservation, PARTIAL for unverified future deployment. New schema is documented only in this isolated run; original context/feature-schema.md and progress tracker are preserved per the user's higher-priority non-overwrite instructions. All new readers check schema/hash contracts; full hand and body blocks remain. No Python/Android feature port or runtime selection changed. Corpus/user authorization does not imply model redistribution rights.

Readiness: FAIL for production. Critical: absent labeled non-sign/boundary/cross-signer and Android evidence prevents safe promotion (one grouped release blocker). Important: poor 88% diagnostic WER, weak hand coverage/OOVs; existing event expiry/unused flag/top1 issues remain; original source Keras artifact missing; true training-worker peak RAM not captured. Minor: research scripts were syntax/test checked, but full repository mypy/black/ruff gates were not run. This is research tooling, not a shipped component. Recommendation: keep the app unchanged; obtain valid labels/consent and improve benchmark/feature coverage before further model work. No code was auto-fixed by this review.
''')
    # Preserve preparatory status documents before updating their current-run summaries.
    snapshots=run/'logs/initial_documents';snapshots.mkdir(exist_ok=True)
    for name in ('README.md','feature_schema_report.md','dataset_registry.csv','dataset_licensing_report.md'):
        path=run/name;copy=snapshots/name
        if path.exists() and not copy.exists():copy.write_bytes(path.read_bytes())
    old=(snapshots/'feature_schema_report.md').read_text()
    old=old.replace('Status: CONTRACTS_TRACED; MODEL_NOT_SELECTED.','Status: CONTRACTS_TRACED; isolated causal TCN pilot selected and completed; production schema unchanged.')
    old=old.replace('No model is selected or trained merely by writing this document.','The following pilot was selected after validating authentic gloss sequences and causal-prefix checks.')
    write('feature_schema_report.md',old+f'''
## Frozen experimental contract

`{SCHEMA}` reuses derive_causal_segmentation_features: columns [0,84) hand XY; [84,126) joint validity; [126,168) proposal support; [168,170) hand presence; [170,172) presence validity; [172,174) observed; [174,176) proposal; [176,182) ROIs; [182,198) body XY; [198,206) body support; [206,214) body validity; 214 body available; 215 delta-time; [216,300) hand deltas; [300,342) delta validity; [342,348) ROI deltas; [348,350) ROI validity; [350,366) body deltas; [366,374) body delta validity; [374,376) observed transitions; [376,378) proposal transitions. Endpoints are exclusive.

TRAIN-only normalization plus clipping [-10,10], per-observation LayerNorm, 48 channels, RF61, 39 CTC logits. The model is independent of S2 despite sharing raw feature math/width. No duration/progress/future-frame input. Boundary/rejection heads were not trained because labels do not support them. Unit prefix/bounded-history tests and exported real-clip parity pass. All schemas/assets/normalizations remain distinct; feature width alone never authorizes substitution. Current Android consumes none of these new artifacts.
''')
    registry=list(csv.DictReader((snapshots/'dataset_registry.csv').open()))
    for row in registry:
        if row['dataset']=='NCSLGR':
            row.update(active_status='PRIVATE_RESEARCH_PILOT_TRAINED_NOT_FOR_DEPLOYMENT',
                annotations='Public index authentic utterance glosses used; XML timing/signer labels unavailable without account',
                split_policy='30 TRAIN / 7 VAL by disjoint collection; signer independence UNVERIFIED',
                storage_reference='37 MOV files 12159833 bytes plus retained metadata/archives')
    with (run/'dataset_registry.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(registry[0]));w.writeheader();w.writerows(registry)
    license_text=(snapshots/'dataset_licensing_report.md').read_text().replace('Status: DOCUMENTATION_REVIEWED; availability of acquired files must be checked separately.','Status: DOCUMENTATION_REVIEWED; 37 public-index-linked NCSLGR clips acquired and used only for a private noncommercial pilot. Download receipts verify observed payload identity. No redistribution or production clearance.')
    license_text=license_text.replace('Eligible for private non-commercial data feasibility only.','Used for the isolated private non-commercial research pilot only.')
    write('dataset_licensing_report.md',license_text)
    commands=f'''```powershell
$repo = '{repo}'
$run = '{runpath}'
Set-Location -LiteralPath $repo
& "$repo/tools/continuous_asl/Verify-Preservation.ps1" -RunRoot $run
# Already early-stopped: this safely reports completion without more epochs.
& "$repo/tools/continuous_asl/Invoke-Training.ps1" -RunRoot $run -Resume
```
'''
    reproduce=f'''```powershell
# Optional fresh reproduction; do not rerun completed preprocessing/export into this run.
# Reserve at least 16 GiB free for another isolated environment and retain all old runs.
$next = & "$repo/tools/continuous_asl/New-Experiment.ps1" -TemplateRunRoot $run
& "$repo/tools/continuous_asl/Initialize-Environment.ps1" -RunRoot $next -CorePackages -PerceptionPackages -ExportPackages
& "$repo/tools/continuous_asl/Invoke-GpuProbe.ps1" -RunRoot $next -RequireCuda
& "$repo/tools/continuous_asl/Invoke-Download.ps1" -RunRoot $next -DryRun
& "$repo/tools/continuous_asl/Invoke-Download.ps1" -RunRoot $next
& "$repo/tools/continuous_asl/Invoke-Download.ps1" -RunRoot $next -Ids ncslgr_video_index -Extract
& "$repo/tools/continuous_asl/Invoke-MetadataValidation.ps1" -RunRoot $next
& "$repo/tools/continuous_asl/Invoke-Download.ps1" -RunRoot $next -ManifestPath "$next/subset_acquisition_plan.json" -MaxTotalBytes 83886080 -DryRun
& "$repo/tools/continuous_asl/Invoke-Download.ps1" -RunRoot $next -ManifestPath "$next/subset_acquisition_plan.json" -MaxTotalBytes 83886080
& "$repo/tools/continuous_asl/Invoke-Preprocessing.ps1" -RunRoot $next -Tag full
& "$repo/tools/continuous_asl/Invoke-Baseline.ps1" -RunRoot $next
& "$repo/tools/continuous_asl/Invoke-Training.ps1" -RunRoot $next
& "$repo/tools/continuous_asl/Invoke-Evaluation.ps1" -RunRoot $next
```

This reproduces the diagnostic protocol, not guaranteed identical CUDA CTC bits. Endpoints may change; checksum/safety failures must stop and retain source files. No command can finish the blocked signer/boundary/rejection/device gates without new authorized evidence. Do not start more training on unchanged tiny data as the recommended next research action.
'''
    write('resume_commands.md',commands+reproduce)
    write('README.md',f'''# Continuous ASL run {run.name}

**PARTIAL — trained research checkpoint; not integrated or production-ready.** Repository `{repo}`; this copy has no Git metadata, so branch/commit are unavailable. See report.md for the complete measured result and release gates.

The RTX 3050 ran real CUDA training: 14 epochs, early stop, best epoch 8. On 7 collection-held-out diagnostic clips: 88% gloss WER (22/25 edits), four OOV tokens, signer identities unknown. No improvement to product accuracy is established. 37 authentic-gloss NCSLGR clips / 12.16 MB / 944 causal observations; 30 TRAIN / 7 VAL. How2Sign English annotations are metadata only. OpenASL/WLASL are excluded; ASL Citizen is not scheduled. Research rights do not authorize redistribution/deployment.

CPU ONNX export parity passes on all 37 clips; artifact 336,478 bytes. Model p50/p95 0.348/0.618 ms; landmark extraction 53.281/62.685 ms. One saved-clip CPU RGB-to-gloss replay processes 17.35 selected observations/s, capped at 10 Hz; no Android FPS claim. Non-sign, boundary, cross-signer and Android gates remain blocked; user confirmed no extra authorized data. Original {preservation['original_files_checked']:,} files are byte- and timestamp-preserved, including models/manifests/avatars. No APK install, runtime change or production asset replacement.

Evidence: environment_report.md, architecture_audit.md, feature_schema_report.md, dataset_registry.csv, dataset_licensing_report.md, download_manifest.csv, preprocessing_report.md, dataset_quality_report.md, training_config.yaml, training_log.csv, training_curves.png/svg, evaluation_report.md, sequence_errors.csv, token_alignment_confusion.csv, performance_report.md, model_parity_report.md, regression_report.md, review_report.md, reproducibility_manifest.json. All source/raw/failed artifacts are retained.

Each major phase has a separate PowerShell entry point under `{tool}`. Download -DryRun does not download/extract/write. Completed-run -Resume is a no-op; preprocessing/export refuse existing destinations. See resume_commands.md for exact commands, optional fresh reproduction and blocked-data requirements. To stop using this experiment, leave it unselected: the original app and environment are unchanged. Do not delete retained data/checkpoints as an automatic rollback.
''')
    write('report.md',f'''# AppHear continuous ASL mission — PARTIAL

**Result: a reproducible private research pilot, with poor diagnostic recognition and blocked release gates. The existing application was not changed.**

Repository: `{repo}`. No .git metadata: current branch and baseline commit unavailable. The initial all-original-file preservation manifest is the baseline, not a historical branch name.

| Evidence | Measured result |
|---|---|
| GPU | RTX 3050 Laptop, 4 GiB; tensor, logits, loss and gradients verified on cuda:0 |
| Data | 37 public-index-linked BU NCSLGR videos; 12,159,833 bytes; {total_frames} decoded frames / 944 selected observations |
| Split | 30 train / 7 validation by collection; no shared collection/content; signer IDs unknown |
| Training | FP32 causal TCN, 76,551 parameters, batch 4; 14 epochs; early stop; best epoch 8 |
| Sequence result | **88% gloss WER: 6 substitutions + 16 deletions, 25 reference glosses**; diagnostic only |
| Robustness | 92% WER after one-third frame drop; 92% with both hands unobserved; artificial corruption only |
| Non-sign / boundaries | BLOCKED: no valid negative or individual-boundary truth; no calibrated rejection |
| Export | ONNX 336,478 bytes; CPU parity max {parity['maximum_abs_logit_error']:.3g}, no decoded changes on 37 clips |
| CPU latency p50/p95 | Model 0.348/0.618 ms; pose/hand extraction 53.281/62.685 ms |
| Offline RGB-to-gloss | One clip, 17.35 processed observations/s, max 10 Hz sampling; 56.096/58.652 ms; startup 8.531 s |
| Memory | Training peak allocated VRAM {peak_vram:,} bytes; replay peak working set 366,141,440 bytes; training peak RAM unavailable |
| Preservation | All 2,932 originals / 1,569,935,151 bytes match SHA256 and modification times |

This is not a cross-signer/end-to-end product accuracy result. Four of 25 validation tokens are OOV; weak hand coverage and a tiny simple-gloss subset limit the experiment. No isolated-word or macro-F1 comparison is valid against the existing 236-way WORD system. CTC blank/UNK do not provide safe non-sign rejection. No production-readiness or Android 15 FPS claim is made.

Files created: code only under tools/continuous_asl; source/receipt/index/translation/sample data under this run's data tree; 14 checkpoints under checkpoints (best pointer epoch_008.pt, SHA `{best['sha256']}`); exported model/contract under export; isolated environment/cache under runtime/cache; reports/curves and logs here. No pre-existing code, model, GLB, manifest, checkpoint or dataset file was modified. Reproducibility manifest lists new artifact hashes. The default Python 3.9 environment and Android defaults remain intact.

Datasets: How2Sign project/script and 32,906 translation rows inspected, gloss unavailable in inspected published downloader; no RGB/keypoint bulk download. OpenASL excluded pending ND/training rights clarification. ASL Citizen/WLASL are isolated-sign sources and were not new training inputs; WLASL agreement still pending. NCSLGR's public index/37 direct clips enabled the private noncommercial pilot under documented BU terms; detailed XML requires a free DAI account, which was not created or bypassed. No redistribution, commercial/model-distribution or production rights are asserted. See dataset_licensing_report.md and retained official source URLs.

Engineering gates: 84 existing unit tests pass, one existing test lacks its frozen Keras artifact; five CTC tests, six hazard cases, nine archive safety cases and 15 downloader policy cases pass. HTTP range resume, fixture replay, causal-prefix and CPU export checks pass. A reproduced event-start eviction now has an isolated explicit EXPIRED_EVENT guard, but no Android correction was applied. Production top1 commitment, unused recognition flag and original unsafe merge implementation remain separate source findings. The new data path has a hard disjoint-output guard.

Remaining blockers: signer identities/cross-signer benchmark; timed gloss boundaries; consented real non-sign/hard-negative data and threshold calibration; adequate feature/label coverage; licensed deployment/distribution clearance; Android operator/adapter/enforced experimental-switch/device tests; representative camera load/FPS/latency/thermal evidence; absent original Keras source artifact; actual training-worker peak RAM. User confirmed no additional authorized local labels. The short GPU run is not a sustained thermal test. No Android build/install, app data clearing, default recognizer/flag change, or experimental integration occurred.

**Recommended next action:** keep this checkpoint unselected. Obtain appropriately licensed signer/timing labels and consented negatives, build a held-out streaming benchmark, and examine hand visibility/cadence on a representative device before scaling training. The present measurements favor investigating preprocessing coverage and latency first.

Detailed evidence: environment_report.md; architecture_audit.md; feature_schema_report.md; preprocessing_report.md; dataset_quality_report.md; baseline_report.md; evaluation_report.md; performance_report.md; model_parity_report.md; regression_report.md; review_report.md; non_sign_recording_protocol.md. Sequence edits/confusions and training curves are retained, with failures and missing metrics clearly labeled.

Exact safe continuation commands (the training phase is already complete):

{commands}

No command can resolve the blocked data/device gates without new evidence. Optional fresh replay commands are in resume_commands.md; all downloads/processing/export use fresh destinations and retain sources.
''')
    generated=[]
    for base in (ROOT/'tools/continuous_asl',run):
        for directory,dirs,names in os.walk(base):
            if base==run and Path(directory)==run:dirs[:]=[d for d in dirs if d not in ('runtime','cache')]
            for name in sorted(names):
                path=Path(directory)/name
                if path.name=='reproducibility_manifest.json':continue
                rel=path.relative_to(run) if base==run else path.relative_to(ROOT)
                generated.append({'scope':'run' if base==run else 'repository_new_tools','path':str(rel).replace('\\','/'),
                    'bytes':path.stat().st_size,'sha256':sha(path)})
    manifest={'schema':'apphear.continuous_asl.reproducibility.v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'status':'PARTIAL_NOT_FOR_PROMOTION','repository_root':str(ROOT),'run_root':str(run),'git_commit':None,
        'original_baseline_sha256':sha(run/'preservation_baseline.json'),'preservation_verification_sha256':sha(run/'preservation_verification.json'),
        'training_manifest_sha256':training['manifest_sha256'],'training_config_sha256':training['config_sha256'],
        'executed_training_script_sha256':training['training_script_sha256'],'current_training_script_sha256':sha(ROOT/'tools/continuous_asl/ctc_experiment.py'),
        'best_checkpoint':best,'model_schema':SCHEMA,'requirements_file':'requirements.observed.txt',
        'seed':20261019,'CUDA_CTC_bitwise_determinism_guaranteed':False,'user_confirmed_additional_local_data':False,
        'artifact_count':len(generated),'artifact_logical_bytes':sum(r['bytes'] for r in generated),
        'excluded_runtime_cache':'recreated from pinned environment specification; no full wheel/cache hashes',
        'original_environment_modified':False,'android_integrated':False,'APK_installed':False,'files':generated}
    write('reproducibility_manifest.json',json.dumps(manifest,indent=2))
    print(json.dumps({'status':'PARTIAL','reports_written':True,'artifacts_hashed':len(generated),'original_preservation':preservation['status']}))
if __name__=='__main__': main()
