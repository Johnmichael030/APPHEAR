from pathlib import Path
from datetime import datetime,timezone,timedelta
import csv,hashlib,json
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[2]
def read(n):return json.loads((RUN/n).read_text(encoding='utf-8-sig'))
def csvread(n):
 with (RUN/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def text(n,s):
 with (RUN/n).open('x',encoding='utf-8') as f:f.write(s)
rows=csvread('event_level_findings.csv');groups=csvread('group_summary.csv')
sel=read('selection.json');hw=read('hardware_and_timing.json');it=read('integrity_results.json')
py=ROOT/'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
cmd=lambda name:f"& '{py}' -B '{RUN/name}'"
table=['|ID|Group|Literal target|Native clip interval|Duration ms|Distinct TRAIN support|Kept frames|Any/both hand fraction|','|---|---|---|---|---:|---:|---:|---|']
for r in rows:table.append(f"|{r['review_id']}|{r['group']}|{r['literal_target']}|[{r['relative_start_frame']}, {r['relative_end_frame']}]|{float(r['inclusive_duration_ms']):.3f}|{r['target_support_distinct_train_utterances']}|{r['retained_count']}|{r['any_hand_fraction'] or 'N/A'} / {r['both_hand_fraction'] or 'N/A'}|")
gt=['|Group|Candidates|Selected|Mean duration ms|Mean label support|Kept frames|Any hand fraction|Body fraction|Context any hand fraction|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for g in groups:gt.append(f"|{g['group']}|{g['candidate_events']}|4|{float(g['mean_duration_ms']):.3f}|{g['mean_target_support']}|{g['retained_frames_total']}|{g['any_hand_frame_fraction'] or 'N/A'}|{g['body_available_frame_fraction'] or 'N/A'}|{g['mean_context_any_hand_fraction']}|")
n_sources=len({r['source_group'] for r in rows});n_signers=len({r['signer_id'] for r in rows})
report=f"""# TRAIN-only alignment and temporal-observability review

Generated {datetime.now(timezone(timedelta(hours=8))).isoformat()} (Asia/Shanghai).

Computational review is complete; qualified visual/linguistic adjudication is blocked. The user explicitly confirmed no qualified reviewer is available. No training, model inference, perception execution, video decoding, acquisition or cadence comparison occurred.

## Scope and evidence reuse

The prior READ_ONLY_FAILURE_AUDIT.md and commands_and_results.txt were read first. Existing exact feature reconstruction, metric and normalization checks were reused. The frozen fold SHA256 is 87442a6f1241860e369a5c8f0768d583b3a4058f5a9ad03dcc135532c419c72f; universe manifest SHA256 is 04fb7c6c26bc934d6618902ceeaa2ee69e19d0c12e5bfe922e5743c8518c0ebe.

Only the 1,500 internal TRAIN utterances were analyzed. IDs equal the frozen fold IDs and belong to the 1,706 global TRAIN universe. Shared train_dev manifest metadata was read for membership; only TRAIN publisher rows were retained for analysis. No held-out examples or predictions influenced selection. Prior pilot artifacts, including saved held-out reports, were hashed solely for preservation, not parsed as review evidence. No reserved TEST directory, payload or manifest was enumerated, opened, hashed or decoded.

All new writes are confined to {RUN}. Existing labels, identities, exclusions, manifests, features, checkpoints, model selection, production/Android/GLB assets and reports were not edited.

## Deterministic selection

Reconstructing cached masks produces A=880, B=1,440, C=6,492 and MIXED=3,611 over 12,423 TRAIN events. These categories are disjoint and exhaustive.

- A: no retained frame inside the inclusive event interval.
- B: retained frames exist, but neither hand has an observed flag on any such frame.
- C: retained frames exist, and every retained frame has at least one observed hand. This is an operational cached-mask control, not proof of complete articulator/linguistic observability. Both hands need not be observed.
- MIXED: some but not all retained frames have observed hands. These events remain unchanged in the dataset, outside this review selection.

Stable event key is sample_id:source_row:event_id. Ordering uses SHA256 of the prefix apphear-alignment-review-v1| plus that key. A anchors prioritize unused recording families then unused masters, then smallest hash. For each anchor, B/C prioritize the same family, then minimize the sum of absolute log2 ratios for inclusive duration in frames, distinct-utterance label support and sequence observation length. Unused master is a tie preference, followed by hash. Twelve unique utterances are required. No random seed or model outcomes are used. selection.json preserves exact IDs, candidate digest, algorithm and matching ratios.

The selection spans four families, {n_sources} masters and {n_signers} signer IDs. All B/C matches share their anchor family. All C matches have identical duration and target support to A. Two B matches have twice the A duration and different labels: KNOW versus IX-loc:j, REALLY versus (1h)HAVE; support ratios 19/24 and 62/58. Observation-length ratios range approximately 0.644–1.081. Masters need not differ within a family. These are approximate controls, not exchangeable causal matches. Hash selection is not representative sampling and does not establish rare-label behavior.

## Computational alignment

All 12,423 TRAIN labels, publisher event IDs/CSV rows, clip/master names, collection/utterance IDs and inclusive event/utterance endpoints match publisher CSV rows. Annotation type is AUTHENTIC_PUBLISHED_DOMINANT_HAND_OCCURRENCE_EVENTS. These occurrence intervals are not independently adjudicated whole-sign extents. Nondominant-only publisher omissions, literal morphology and Hidden flags remain unchanged.

Publisher master coordinates are inclusive; subtract utterance master start to obtain zero-based native clip coordinates. Native fps is 30000/1001. Inclusive duration is (end-start+1)*1000/fps; last-frame timestamp differs from end-exclusive display time. All TRAIN arrays have indices 0,3,6,... within their cached frame count. Maximum timestamp disagreement with frame_index*1000/fps is {it['max_timestamp_coordinate_error_ms']:.3g} ms, floating-point scale. This confirms stored arithmetic, not independent visual alignment of source frame zero.

For the 12 selected records, feature timestamps, embedded transcripts and source origins agree with cached raw metadata. Feature hand-observed/body-available channels agree exactly with raw masks. Effective hand validity correctly equals raw_joint_valid AND hand_observed; body validity matches directly. Raw joint-valid flags can remain true when hand-observed is false by design. This is not a new mask bug or proof of physical hand presence. Prior exact full-feature reconstruction remains the wider evidence. Independent scalar interval checks agree with vectorized membership for all 12. No computational coordinate/join/mask discrepancy was found.

{chr(10).join(table)}

All selected durations are 33.367–66.733 ms, below the actual 100.1 ms cadence. A intervals contain one/two native frames but no retained frame; each lies between neighboring retained samples. B and C retain exactly one frame per selected event. Thus sampling phase can remove direct observations inside publisher intervals; this does not establish loss of all sign information.

{chr(10).join(gt)}

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

GPU: {hw['gpu_query']}. Existing PyTorch {hw['torch_version']} reports CUDA {hw['torch_compiled_cuda']} available. NumPy {hw['installed_numerical_libraries']['numpy']} is installed; TensorFlow {hw['installed_numerical_libraries']['tensorflow']} is listed by metadata only, CUDA support unverified. CuPy/Numba are absent. No install/update occurred.

CPU: {hw['cpu_logical_threads']} logical threads; torch reports {hw['torch_cpu_threads']} CPU threads. RAM at inspection: {hw['total_physical_memory_bytes']/2**30:.2f} GiB total, {hw['available_physical_memory_bytes']/2**30:.2f} GiB available. CIM access was denied; GlobalMemoryStatusEx supplied RAM. Arrays were cached on disk at entry, not loaded in this new process; OS page-cache residency unknown. Cached indices/masks during analysis occupied {hw['cached_array_ram_bytes']:,} bytes.

CPU cached-array/annotation review took {hw['cpu_cached_array_review_seconds']:.3f} s, including loading/hash/event reconstruction. Successful analysis including scoped hashes/exports: {it['elapsed_analysis_and_hardware_seconds']:.3f} s internally, 7.854 s shell wall. Hardware inspection: {hw['capability_inspection_seconds']:.3f} s internally, 7.543 s shell wall. GPU numerical work was not used; no benchmark or speedup claim. Small per-utterance arrays and 12-event final work do not justify transfer/initialization overhead. All 12 memberships have independent scalar CPU verification. torch was used only for capability queries; no tensor/model was created.

First diagnostic attempt failed an assertion comparing raw joint validity directly to effective feature validity. Source inspection established the required AND-with-observed rule. Only this new script was corrected; the successful run passed. No project feature, label or mask was changed; the failed attempt wrote no CSV/report output.

## Integrity and limitations

SHA256, byte size and nanosecond-mtime before/after checks passed for {it['existing_files_checked']} existing files ({it['existing_bytes_checked']:,} bytes): all 76 previous pilot artifacts/checkpoints, all 21 prior audit artifacts, 1,500 TRAIN raw arrays, selected original features/videos/signer-provenance references, source annotations/manifests and three relevant pipeline source files. Selected reference count 36 means 12 each of feature/video/signer references; repeated provenance paths are deduplicated in actual hash totals. Pilot/audit membership has no additions/removals. Source hashes match the prior allowlist.

input_files_before.json and input_files_after.json declare exact scope; integrity_results.json records results. verify_review.py rechecks only these paths and new checksums. Production/Android/GLB assets were not written, but were not independently hashed; no whole-workspace integrity claim. Hashing selected video bytes for preservation is not decoding. No inference/training/perception, downloads, external services, global environment/GPU changes or Git commands occurred. No reserved TEST enumeration/access. Unlogged unrelated historical process behavior cannot be certified.

## Commands

Working directory: {ROOT}

{cmd('inspect_hardware.py')}
{cmd('review_cached.py')}
{cmd('write_report.py')}
{cmd('verify_review.py')}

First three are provenance/construction commands, intentionally exclusive-create and not rerunnable over completed outputs. To validate this completed package, run only verify_review.py. commands_and_results.txt includes initial PowerShell, denied CIM queries and failed/successful analysis attempts.

PARTIAL — computational evidence improved, but qualified visual/annotation review remains unresolved
"""
text('ALIGNMENT_OBSERVABILITY_REPORT.md',report)
commands=f"""Working directory: {ROOT}
New writes only: {RUN}
First read previous READ_ONLY_FAILURE_AUDIT.md and commands_and_results.txt with Get-Content -LiteralPath.
Read both user attachment missions with Get-Content -LiteralPath.

Initial PowerShell:
$reviewStamp=(Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$reviewDir=Join-Path 'experiments\\continuous_asl' ($reviewStamp+'_alignment_observability_review01')
New-Item -ItemType Directory -Path $reviewDir -ErrorAction Stop
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
Result: NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, 3962 MiB
Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory,TotalVisibleMemorySize
Get-CimInstance Win32_ComputerSystem | Select-Object NumberOfLogicalProcessors
CIM queries: Access to a CIM resource was not available to the client; combined shell exit1.
Fallback: Python ctypes Windows GlobalMemoryStatusEx; no escalation/config changes.

Read-only source inspections:
Get-Content -LiteralPath 'experiments\\continuous_asl\\20261020T135119Z_read_only_failure_audit01\\analyze_read_only.py' -TotalCount 130
Get-Content -LiteralPath 'experiments\\continuous_asl\\20261020T133148Z_source_transfer_pilot01\\resolved_manifest.json' -TotalCount 90
Get-Content -LiteralPath 'pipeline\\perception_vnext\\segmentation\\learned_features.py' -TotalCount 220
Get-Content -LiteralPath 'pipeline\\perception_vnext\\recognition_schema.py' -TotalCount 220
A short isolated Python -B probe printed prior hash-manifest types, first manifest entry and fold keys; no payload/model execution.

{cmd('inspect_hardware.py')}
Exit0; shell7.543s; internal capability{hw['capability_inspection_seconds']:.6f}s.
{cmd('review_cached.py')}
First attempt exit1 at new-script line108;6.291s.
AssertionError comparing raw hand_joint_valid directly with effective feature validity.
Source contract: effective=raw.hand_joint_valid & raw.hand_observed[:,:,None].
Corrected only new review script. No old file changed; no CSV/report output existed yet.
{cmd('review_cached.py')}
Second attempt exit0; shell7.854s; internal{it['elapsed_analysis_and_hardware_seconds']:.6f}s.
A880/B1440/C6492/MIXED3611;12 unique utterances;12423 publisher joins;
12/12 scalar interval checks;0 coordinate/mask discrepancies.
1637 existing files unchanged;76pilot and21audit files unchanged.

Import-Csv -LiteralPath '{RUN/'event_level_findings.csv'}' | Select-Object review_id,group,literal_target,sample_id,relative_start_frame,relative_end_frame,target_support_distinct_train_utterances,sequence_observations,retained_count,any_hand_fraction,both_hand_fraction,publisher_hidden_flag,matching_cost | Format-Table -AutoSize
Get-Content -LiteralPath '{RUN/'selection.json'}'

{cmd('write_report.py')}
Creates report, command log and new-artifact checksum index.
{cmd('verify_review.py')}
Read-only validation; execution output is returned by the tool and can be reproduced.

User confirmed: No qualified reviewer available.
No training/inference/perception/media decode/acquisition/install/external service/Git changes.
No reserved TEST access/enumeration. All existing inputs preserved.
"""
text('commands_and_results.txt',commands)
lines=[f"{sha(p)}  {p.name}" for p in sorted(RUN.iterdir()) if p.is_file() and p.name!='artifact_hashes.sha256']
text('artifact_hashes.sha256','\n'.join(lines)+'\n')
print(json.dumps(dict(report=str(RUN/'ALIGNMENT_OBSERVABILITY_REPORT.md'),new_artifacts_hashed=len(lines),verdict='PARTIAL'),indent=2))

