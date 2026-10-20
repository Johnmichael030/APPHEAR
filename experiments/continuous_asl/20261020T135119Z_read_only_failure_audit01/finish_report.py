"""Write findings only after old-input integrity comparison; never execute a recognizer."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta
import csv,hashlib,json,math
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[2];PILOT=RUN.parent/'20261020T133148Z_source_transfer_pilot01'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(name,x):
 with (RUN/name).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2);f.write('\n')
def md(name,s):
 with (RUN/name).open('x',encoding='utf-8') as f:f.write(s.rstrip()+'\n')
def link(label,path,line):return f'[{label}]({Path(path).as_posix()}:{line})'
def main():
 a=read(RUN/'audit_summary.json');fs=read(RUN/'feature_summary.json');support=read(RUN/'target_support.json');cp=read(RUN/'checkpoint_metadata_CPU_only.json');stop=read(RUN/'stopping_reconciliation.json');fc=read(RUN/'feature_and_target_checks.json')
 state=read(PILOT/'training_status.json');rows=read(PILOT/'resolved_manifest.json');lookup={r['sample_id']:r for r in rows};fit=[r for r in rows if r['internal_role']=='TRAIN'];held=[r for r in rows if r['internal_role']=='HELD_OUT'];vocab=read(PILOT/'vocabulary.json')['tokens']
 # Validate the pre-existing artifact index against the pre-audit full pilot snapshot.
 first=read(RUN/'pilot_files_before.json');firstmap={r['path']:r for r in first};indexed=0
 for line in (PILOT/'artifact_hashes.sha256').read_text().splitlines():
  expected,name=line.split('  ',1);assert firstmap[str((PILOT/name).resolve())]['sha256']==expected;indexed+=1
 stats=[];scores={}
 for name in ['train1500','monitor','primary80','strong27','full206']:
  s=read(PILOT/f'selected_{name}.json');scores[name]=s;hyp=Counter(t for p in s['predictions'] for t in p['hypothesis']);ref=Counter(t for p in s['predictions'] for t in p['reference'])
  stats.append({'scope':name,'WER':s['gloss_WER'],'CTC_loss':s['CTC_loss'],'S':s['substitutions'],'D':s['deletions'],'I':s['insertions'],'reference_tokens':s['reference_tokens'],
   'blank_fraction':s['blank_argmax_fraction'],'empty_outputs':s['empty_hypotheses'],'unique_predicted':s['unique_predicted_tokens'],'hypothesis_length_histogram':s['hypothesis_length_histogram'],
   'top10_predictions':hyp.most_common(10),'top_reference_token':ref.most_common(1),'UNK_predictions':hyp['<UNK>'],'BLANK_predictions_in_decoded_text':hyp['<BLANK>'],
   'adjacent_duplicate_hypothesis_tokens':sum(x==y for p in s['predictions'] for x,y in zip(p['hypothesis'],p['hypothesis'][1:]))})
 save('prediction_collapse_summary.json',stats)
 # Descriptive associations; no causal inference and no data selection for another fit.
 features={r['sample_id']:r for r in fc};associations=[]
 for name in ['train1500','primary80','strong27']:
  score=scores[name]
  for criterion,fn in [('no_reference_interval_sampling_gaps',lambda z:z['missing_event_observation_count']==0),('some_reference_intervals_unsampled',lambda z:z['missing_event_observation_count']>0),
    ('any_hand_fraction_below50pct',lambda z:z['observed_any_hand_fraction']<.5),('any_hand_fraction_atleast50pct',lambda z:z['observed_any_hand_fraction']>=.5)]:
   ps=[p for p in score['predictions'] if fn(features[p['sample_id']])];refs=sum(p['reference_tokens'] for p in ps);errors=sum(p['substitutions']+p['deletions']+p['insertions'] for p in ps)
   associations.append({'scope':name,'condition':criterion,'utterances':len(ps),'reference_tokens':refs,'WER':errors/refs if refs else None,'causal_claim':False})
 save('feature_error_associations.json',associations)
 # Hash all explicit existing non-TEST inputs again. New report files are excluded.
 before=read(RUN/'input_files_before.json');after={};changed=[]
 for path,b in before.items():
  p=Path(path);s=p.stat();now={'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':sha(p)};after[path]=now
  if now!=b:changed.append(path)
 save('input_files_after.json',after)
 pilot_after=read(RUN/'pilot_files_after.json');pam={r['path']:r for r in pilot_after};pilotchanges=[p for p,b in firstmap.items() if pam.get(p)!=b];added=sorted(set(pam)-set(firstmap));removed=sorted(set(firstmap)-set(pam))
 integrity={'pilot_before_files':len(first),'pilot_after_files':len(pilot_after),'original_pilot_artifact_index_verified':indexed,'pilot_changed':pilotchanges,'pilot_added':added,'pilot_removed':removed,
  'referenced_nonTEST_inputs_hashed':len(before),'referenced_input_bytes':sum(v['bytes'] for v in before.values()),'referenced_inputs_changed':changed,'whole_workspace_hashed':False,
  'scope_limit':'Every file in target pilot and all explicit feature/raw/video/source-label/signer-provenance/code inputs checked before/after. Unrelated workspace and reserved TEST contents intentionally not opened.',
  'training_or_inference_executed':False,'packages_installed':False,'external_services_used':False}
 assert not (changed or pilotchanges or added or removed),integrity
 save('integrity_results.json',integrity)
 # The initial/selected/final checkpoint metadata were read on CPU only; no model instance.
 checkpoint_table='\n'.join(f"|{c['file']}|{c['epoch']}|{c['optimizer_step_min']}–{c['optimizer_step_max']}|{c['parameter_L2_change_from_initialization']:.4f}|{c['optimizer_groups'][0]['lr']}|" for c in cp)
 score_table='\n'.join(f"|{s['scope']}|{s['WER']*100:.4f}%|{s['S']}/{s['D']}/{s['I']}|{s['reference_tokens']}|{s['blank_fraction']*100:.3f}%|{s['empty_outputs']}|{s['unique_predicted']}|{s['UNK_predictions']}|" for s in stats)
 event_table='\n'.join(f"|{name}|{fs[name]['events']['target_events']}|{fs[name]['events']['events_no_kept_frame']} ({100*fs[name]['events']['events_no_kept_frame']/fs[name]['events']['target_events']:.2f}%)|{fs[name]['events']['sampled_events_no_observed_hand']} ({100*fs[name]['events']['sampled_events_no_observed_hand']/fs[name]['events']['target_events']:.2f}%)|" for name in ['TRAIN','PRIMARY80','STRONG27'])
 feature_table='\n'.join(f"|{name}|{fs[name]['observations']['min']:.0f}–{fs[name]['observations']['max']:.0f}|{fs[name]['target_tokens']['min']:.0f}–{fs[name]['target_tokens']['max']:.0f}|{fs[name]['CTC_slack']['min']:.0f}|{fs[name]['observed_any_hand_fraction']['mean']*100:.2f}%|{fs[name]['body_available_fraction']['mean']*100:.2f}%|{fs[name]['normalization_clip_fraction']['mean']*100:.4f}%|" for name in ['TRAIN','PRIMARY80','STRONG27'])
 zero=[z for z in fc if z['zero_hand_motion_sequence']];zero_text='; '.join(f"`{z['sample_id']}` ({z['observations']}observations,{z['target_tokens']}targets,hand-observed fraction{z['observed_any_hand_fraction']:.3f})" for z in zero)
 ex=read(RUN/'undersampled_event_examples.json')[:4];extext='\n'.join(f"- `{x['sample_id']}`: literal target `{x['gloss']}`, relative source frames{x['source_relative_frame_span']}, duration{x['publisher_span_ms']:.1f}ms, **zero kept frames in that annotated interval**." for x in ex)
 modelref=link('ctc_experiment.py',ROOT/'tools/continuous_asl/ctc_experiment.py',22);trainingref=link('train.py',PILOT/'train.py',100);feature_ref=link('learned_features.py',ROOT/'pipeline/perception_vnext/segmentation/learned_features.py',159);prep_ref=link('prepare_asllrp.py',RUN.parent/'20261020_continuous_expand06/tools/prepare_asllrp.py',78)
 now=datetime.now(timezone(timedelta(hours=8))).isoformat()
 md('READ_ONLY_FAILURE_AUDIT.md',f'''# Read-only failure audit: continuous-ASL source-transfer pilot

Generated {now} (Asia/Shanghai).

**PARTIAL — evidence narrowed, but a key diagnostic remains unresolved.**

The pilot is computationally intact and demonstrably underlearned. No critical vocabulary, blank-ID, WER, file-join, feature-recomputation or normalization defect was found. New read-only evidence reveals **missing direct temporal/hand observations for a material fraction of annotated sign intervals**, while the long-tail target task and held-out-based stopping rule leave full-TRAIN learnability unestablished. These facts narrow the problem; they do not isolate a single causal root cause or prove that more epochs would fix it.

One small implementation defect is confirmed: the final odd gradient-accumulation group is half-scaled. Its scale is too small to call it the primary cause without an ablation. No patch, model inference, backward call, optimizer step, package installation, download, GPU/system configuration change, Git command or write to an existing file occurred. Checkpoints were deserialized **onto CPU for metadata/tensor inspection only**; no model or optimizer instance was constructed. Cached-landmark feature math was recomputed without perception/model inference.

## 1. Evidence and early stopping reconciled

All four pilot Python files were inspected: common.py, train.py, finish.py and verify.py (the remaining implementation file), plus PILOT_REPORT.md, config, METRICS.csv, logs, manifests, vocabulary, normalization, initialization/epoch12/epoch14 metadata and saved predictions. The pre-existing75-artifact checksum index matches the full pre-audit76-file snapshot (including the index itself).

Fourteen complete epochs,188updates/epoch,2,632updates and21,000utterance presentations are consistent. The selected checkpoint is epoch12, the minimum logged primary80(WER,CTCloss) tuple. This is a local research pointer, not a promotion or proof of acceptance. Selected epoch12 fullTRAIN must not be confused with final epoch14's128-example monitor.

Epoch6 has472edit errors/485references, WER97.3196%; epoch12 has470/485, WER96.9072%. The two-edit gain is **{stop['epoch6_to12_improvement_percentage_points']:.4f}percentage points**, below the0.5point credible-reset threshold (at least3edit errors on485references). Epoch8,10,12and14 therefore produce four stale evaluations under the declared rule, even though epoch12 is a measurable local improvement. The epoch14NO_CREDIBLE_PRIMARY_GAIN stop is implemented consistently; it does **not** establish that TRAIN had saturated. See stopping_reconciliation.json and {link('train.py:129',PILOT/'train.py',129)}.

Optimization loss falls8.2549(epoch1)→3.7945(epoch14); monitor loss falls6.5485→3.5026 and monitorWER100%→85.346%. PrimaryCTC rises from its epoch2minimum6.5720 to8.0112at14; primaryWER only improves slightly. Parameters and optimizer states change measurably:

|Checkpoint|Epoch|Saved optimizer steps|Parameter L2 change from initialization|Learning rate|
|---|---|---|---|---|
{checkpoint_table}

All model tensors are finite; checkpoint vocabulary, normalization buffers, config hash and fold hash agree. Model has{cp[1]['model_parameter_count_excluding_norm_buffers']:,}trainable scalar parameters by state-tensor count excluding mean/std. Initialization is fresh random, not recognition-pretrained. Full checkpoint metadata and shapes are in checkpoint_metadata_CPU_only.json.

## 2. Model, targets, padding and decoding

|Question|Verified finding|Classification|
|---|---|---|
|Architecture|378→48linear projection;4residual causal blocks, dilations1/2/4/8, two width3convolutions each; per-frame LayerNorm/ReLU;48→2289output. RF61observations, no temporal downsampling, attention, recurrent layer or dropout|CONFIRMED|
|Feature basis|Current/backward raw-landmark segmentation features, including explicit validity/missingness, normalized image XY and deltas; no RGB appearance or language model branch|CONFIRMED; lexical adequacy POSSIBLE limitation|
|Initialization|Fresh seed20261020, default module initialization; initialoptimizer has0steps; selectedweights changed|CONFIRMED|
|Optimizer/loss|AdamWlr0.001,wd0.0001,betas0.9/0.999,eps1e-8,clip5,FP32; CTCLoss blank0,zero_infinityFalse,default mean reduction|CONFIRMED|
|Scheduler|No scheduler constructed or stepped; lr0.001in initial/12/14param groups|CONFIRMED|
|Batch|4utterances per microbatch,2microbatches per normal optimizer update|CONFIRMED; tail issue below|
|Targets|2,287whole literal ASL occurrence-gloss strings mapped to IDs2…2288; blank0/UNK1specials. Not English translation, subword or character training|CONFIRMED|
|Label convention|Same unchanged strings for TRAIN/eval; no lowercasing, punctuation splitting, canonicalization or whitespace WER normalization. All80/27references map exactly|CONFIRMED|
|BOS/EOS|None; CTCdoes not require them|No defect evidenced|
|Padding/lengths|Right-zero-pad raw feature arrays; pass exact valid input/target lengths toCTC. Model is causal with per-frameLayerNorm, so rightpadding cannot affect earlier real outputs. No explicit padding mask is needed by this architecture; this is a static architectural conclusion, not a new forward test|No defect evidenced|
|Blank/repeat decoder|Greedy argmax, suppressblank0, collapse adjacent raw repeated IDs; repetitions separated by blank remain valid repeated glosses|Loss/target/decoder contract compatible|
|Loss/metric weighting|Mean CTCnormalizes per target length; reported mean per utterance. WER pools edit errors/reference tokens. They optimize different quantities, so lowerloss need not imply lowerWER|CONFIRMED; not an implementation mismatch|
|Full206loss|126sequences have unsupported labels; aggregateCTCloss intentionallyNone, full206WER keeps literal references. It does not select checkpoint|Correctly scoped|

Sources: {modelref}, {link('CTC batch/decoder',ROOT/'tools/continuous_asl/ctc_experiment.py',79)}, {link('optimizer and initialization',PILOT/'train.py',63)}, {link('scoring',PILOT/'train.py',81)}. The frozen decoder source hash matches config. Raw per-frame logit sequences were not saved, so no new independent argmax replay was attempted.

### Confirmed small accumulation-tail defect

1,500/4=375microbatches, an odd count. At {link('train.py:109',PILOT/'train.py',109)}, every loss is divided by2; at110–113the final lone microbatch is still stepped. Its gradient is therefore half the average-of-available-microbatches scale. One of188updates/epoch is affected (0.532%); nominal utterance-gradient weighting shortfall is0.133%per epoch. Adam moments make the eventual parameter effect nonlinear, so this is **not** a proved0.133%performance effect. It is CONFIRMED as a small implementation issue, but **NOT EVIDENCED as the primary recognition failure**. No correction was executed. A pure arithmetic final-group divisor check, not a fit, is sufficient to demonstrate it; any later code correction belongs in a new authorized experiment.

## 3. Dataset/split integrity and feature→target provenance

All1,500TRAIN and206heldout IDs are exact members of the cited globalTRAIN universe. They match the frozen source-fold IDs. Raw video hashes, feature hashes, embedded feature metadata, cached raw metadata and publisher annotation joins agree. Every ordered target equals its event-label sequence, and each event's ID, label, source row, utterance/video/master identity and native frame interval matches the original local publisherCSV. Current signer IDs are publisher-query sidecar joins added after extraction; an absent signer in older embedded raw metadata is not silently replaced with visual identity inference.

All1,706feature sequences exactly reproduce from cached raw landmarks using the existing causal378function ({feature_ref}). Recomputed1,500TRAIN-only mean/std matches saved normalization bit-for-bit. No empty features/targets, nonfiniteX, shape mismatch, nonmonotonic timestamps, globalCTC length failure or exact duplicateXsequence was found. Native stored frame counts equal inclusive publisher utterance spans; media bytes match the stored hashes. No new video/perception inference was run, so visual boundary correctness is not independently certified.

|Group|Observation length|Target length|MinimumCTCslack|Mean fraction with any hand observed|Mean body availability|Mean fraction clipped beyond±10|
|---|---|---|---|---|---|---|
{feature_table}

One TRAIN sequence has zero hand-XY-delta energy: {zero_text}. It is a signal-quality flag, not proof of a corrupt file or a static actual sign. No all-zero/duplicate sequence was silently removed.

TRAIN/heldout overlap is zero for utterance IDs, master-recording IDs, collections, raw media hashes, decoded-pixel hashes and conservative recording families. Evaluation contains two already-seen TRAIN signers;3signers occur inTRAIN. All undated Ben masters stay together; date-shaped master prefixes remain conservative family proxies, not proven capture dates. Source/signer causal effects and true parent aliases cannot be fully separated. The pilot code reads only the cited universe filtered toglobalTRAIN({link('train.py:17',PILOT/'train.py',17)}); evaluation functions consume only these206heldout records and the128TRAIN monitor. Static code/data-flow and saved memberships provide no evidence of reservedTESTloading. No reservedTESTdataset files were opened by this audit. Absolute proof of unlogged historical process behavior is unavailable.

## 4. Newly measured temporal/visibility gaps

The nominal10Hz policy selects the first native frame at least100msafter the previous retained frame ({prep_ref}); actual median spacing is100.1ms. GlobalCTC feasibility only asks whether totalT can encodeU plus repeated labels. It does not ensure even one retained observation lies inside every sign interval.

|Group|Publisher target events|No retained native frame in event interval|Has retained frame(s), but no observed hand in them|
|---|---|---|---|
{event_table}

The two gap categories are disjoint. Together they affect2,320/12,423TRAIN intervals (18.68%) and89/485primary intervals (18.35%). These are exact cached-frame-index/observed-mask counts relative to **publisher** sign spans, not newly adjudicated ASL ground truth. Nearby frames or body/context may still contain evidence, and 'hand observed' is a detector state. The counts do not establish that each affected label is impossible to learn. Hidden-event flags are retained (647TRAIN,8primary); their physical/linguistic meaning was not independently verified here.

Examples using stable IDs only:

{extext}

This makes a temporal-evidence/preprocessing limitation **POSSIBLE**, with the missing direct observations themselves **CONFIRMED**. It is not evidence of a wrong feature/transcript file join. Byte-perfect feature reproduction can coexist with a task-inadequate sampling or landmark representation. Before another fit, a small qualified visual review of the existing referenced intervals is needed to distinguish true sampling/refinement loss, annotation-boundary semantics and contextual observability. No cadence, frame, target or feature was changed.

## 5. Prediction collapse and metric validity

An independent1DLevenshtein implementation checked5,469saved prediction examples across37scorefiles. Every reference matches the frozen transcript; every edit-distance total and aggregateWER matches. The same literal token unit is used on both sides, including classifier strings and whole fingerspelled words. Unnecessary participant names are not exposed in this report.

|Selected scope|WER|S/D/I|Reference tokens|Blank frame fraction|Empty outputs|Distinct predictions|UNK outputs|
|---|---|---|---|---|---|---|---|
{score_table}

The selected checkpoint is blank-dominated in frame space, but it is **not** an all-blank or one-token-only model:507TRAIN labels and57primary labels are emitted. Its primary dominant token share is{scores['primary80']['top_predicted_token_fraction']*100:.2f}%, not the≈99%IX-1pcollapse of an earlier model. Repeated decoded labels can be correct CTCbehavior when separated by blanks; no new raw-logit replay was performed. Detailed token counts, length histograms, unknown counts and repeated-hypothesis counts are in prediction_collapse_summary.json.

Learning is measurable—loss falls, parameter norms change, output diversity expands and monitorWER improves—but far below the declaredTRAIN50%screen. A high blank fraction alone is not a bug; deletion-heavy errors plus poor TRAINWER establish inadequate recognition. Primary/all27lack no vocabulary references, so unknown labels cannot explain their failure. feature_error_associations.json reports posthoc sampling/visibility associations without causal claims. Earlier full-task/nine-NCSLGRWER values remain non-equivalent comparisons in vocabulary, membership, normalization and evaluation IDs.

## 6. Ranked root-cause assessment

|Priority / suspected cause|Classification|Concrete evidence|Cheapest read-only confirmation/rejection test|
|---|---|---|---|
|1. Full-task learning is inadequate, and heldout-driven early stopping occurs before a TRAIN-learning prerequisite is met|CONFIRMED state; LIKELY protocol contribution; 'more epochs fixes it' remains POSSIBLE|TRAIN87.08%,monitor88.28%atselected12; final14monitor85.35%andlossstilldeclining; heldoutstaleness, not TRAINsaturation, stops14|Already reconciled loss/WER/optimizer/weight deltas and two-edit reset arithmetic. No read-only test can prove a longer run would succeed; keep that causal claim unresolved|
|2. Fine-grained long-tail supervision versus a small geometry-only CTCrepresentation|LIKELY contributor; not isolated|2,287labels over12,423tokens;1,182labels (51.68%)appear in one distinctTRAIN utterance;1,808(79.06%)appear in<5; image-coordinate segmentationfeatures project378→48; stronger-support27stillfails|Use saved errors/support strata and review whether indistinguishable landmark trajectories carry different target distinctions. Audit a small existing labeled visual sample before any new fit; do not merge/rename labels|
|3. Temporal/observed-hand evidence loss inside authentic target intervals|CONFIRMED observation gaps; POSSIBLE causal contributor|880TRAIN sign intervals have no kept frame;1,440more have no observed hand at kept frames. Same pipeline onbothsplits, actual100.1mscadence|Read-only native-frame-index versus event-span and hand-mask join is completed. Next cheapest unresolved test is qualified visual/annotation review on those existing intervals; no training required|

Other hypotheses: basic file corruption, malformed tensor lengths, blank-ID/vocabulary mismatch, wrong source join, a WER-normalization bug, decoder/CTCtarget incompatibility and normalization leakage are **NOT EVIDENCED** by these checks. A small accumulation-tail scaling defect is **CONFIRMED**, but its material role is **NOT EVIDENCED**. Source/signer effects and landmark information limits remain **POSSIBLE** and confounded. A semantic/visual data-label alignment defect is **POSSIBLE**, because provenance/hash/frame arithmetic cannot validate what the signer actually expressed. Insufficient epochs as the sole cause is **NOT EVIDENCED**.

## 7. Checkpoint disposition and minimal next experiment plan — not executed

Retain initialization,epoch12andepoch14 unchanged for forensic/reference diagnostics. Do not promote them or use selected12as a default warm start for new folds. New internal folds could overlap its1,500TRAIN IDs; a new fit must establish nonleaking initialization. The initial0-update checkpoint is reusable only when its exact vocabulary, TRAIN-only normalization and fold contract remain valid.

First do a **no-fit alignment/observability review** on a predeclared small existing set: four no-kept-frame events, four sampled/no-observed-hand events and four fully observed controls, selected fromTRAINby stable-IDhash and matched by event duration/support where feasible. Confirm publisher frame origins, hand visibility and literal event semantics with qualified ASL annotation. Do not use reservedTESTor alter existing labels. Frame-index evidence is already measured; visual/linguistic adjudication is the key unresolved diagnostic.

If that review confirms material sampling loss, the minimal corrective research experiment is a new-directory **cadence-only feature comparison** on those exact authorized source clips, holding targets, extraction math and frame origins fixed, then verifying event coverage and computational cost. No such extraction or fit was executed. If it does not confirm a material information defect, investigate TRAIN-only task learnability before expanding generalization claims; do not blindly add epochs or datasets.

Entry criteria for another training run:

1. A later explicit authorization; same frozen IDs or a newly reviewed manifest, exactlabel/schema/vocabulary/normalization hashes, zero known group/content overlap and noTESTselection.
2. The small visual/provenance review resolves frame origin and label semantics. Event-level coverage is reported; unobserved targets have an explicit adjudication/protocol decision, never silent trimming or label repair.
3. Isolate one changed variable. Handle the accumulation tail correctly in a new trainer and verify its divisor arithmetically; do not bundle that minor correction with a claimed cadence or architecture ablation without an appropriate control.
4. Predeclare a TRAIN-learning gate on a deterministic source-balanced monitor plus full selectedTRAIN: WER≤50%,≥10predicted labels and multi-token hypotheses on≥25%ofexamples. Failure means the transfer result is inconclusive. Existing8/64memorization is a wiring/learnability control, not full-task success.
5. Do not use an unpassed TRAINlearning screen to assert a source effect. Once that prerequisite is met, the80primary sequences must pass the predeclared≤90%WER/diversity/multi-token screen; report27as a subgroup and full206coverage context. With onlytwofamilyclusters, even a pass is exploratory.

Withhold source-transfer acceptance if TRAINlearning fails, lineage remains unresolved, there is group leakage, target visibility/compatibility is unresolved or the result depends on swapping a restricted subgroup for the full evaluation. Withhold all unseen-signer claims: evaluation signers already occur inTRAIN, and recording/source/signer factors are confounded. Production promotion remains blocked.

## 8. Integrity, commands and limitations

All76existing pilot files and5,198explicit nonTESTinputs (1,451,409,089input bytes; sourcefeatures,rawlandmarks,rawvideo,labels,signerprovenanceandcode) were hashed before/after. **Zero content/size/mtime changes; zero pilot additions or removals.** Original75artifactindex verified. The new timestamped report directory is excluded from comparison. No attempt was made to hash the entire unrelated workspace or reservedTESTcontents; this is a complete audit of the target pilot and its enumerated inputs, not a whole-machine proof.

PowerShell was used for initial/full-target hash snapshots and source/report inspection. Isolated Python-Bwas necessary for NPZ/math, independent edit distances and CPU-only trusted checkpoint metadata. No existing training or finalizer script was executed. No old verifier chain was run because its broad protected-input hashing could open reservedTESTcontents; equivalent target/control/source checks were performed on the explicit nonTESTscope here. Exact commands, output paths and scope are in commands_and_results.txt. All new writes remain under `{RUN}`.

Limitations: no model replay, no native video decode or qualified visual annotation, no causal ablation, no population confidence from two families, no authenticated capture-date lineage beyond conservative grouping, and no proof of unlogged historical behavior. Missing frame/hand observations are not automatically mislabeled samples. Target semantic adequacy and the cause of full-task underlearning remain unresolved.

**Final verdict: PARTIAL — evidence narrowed, but a key diagnostic remains unresolved.**
''')
 py=ROOT/'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
 md('commands_and_results.txt',f'''Strict read-only audit; all new writes confined to {RUN}
Working directory: {ROOT}

Initial PowerShell snapshot (executed before inspecting pilot content):
$auditPilot='{PILOT}'
$auditRun='{RUN}'
$auditBefore=@(Get-ChildItem -LiteralPath $auditPilot -Recurse -File | Sort-Object FullName | ForEach-Object {{ [PSCustomObject]@{{path=$_.FullName;bytes=$_.Length;last_write_utc=$_.LastWriteTimeUtc.ToString('o');sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}} }})
ConvertTo-Json -InputObject $auditBefore -Depth 5 | Set-Content -LiteralPath (Join-Path $auditRun 'pilot_files_before.json') -Encoding utf8
Result:76files,38083529bytes.

PowerShell inspections (read-only):
Get-Content -LiteralPath '{PILOT/'PILOT_REPORT.md'}'
Get-Content -LiteralPath '{PILOT/'common.py'}'
Get-Content -LiteralPath '{PILOT/'train.py'}'
Get-Content -LiteralPath '{PILOT/'finish.py'}'
Get-Content -LiteralPath '{PILOT/'verify.py'}'
Get-Content -LiteralPath '{PILOT/'config.json'}'
Get-Content -LiteralPath '{ROOT/'tools/continuous_asl/ctc_experiment.py'}'
Get-Content -LiteralPath '{ROOT/'pipeline/perception_vnext/segmentation/learned_features.py'}'
Get-Content -LiteralPath '{ROOT/'pipeline/perception_vnext/recognition_schema.py'}'
Get-Content -LiteralPath '{RUN.parent/'20261020_continuous_expand06/tools/prepare_asllrp.py'}'
Get-Content -LiteralPath '{RUN.parent/'20261020_continuous_expand06/tools/asllrp_catalog.py'}'
Additional rg keyword searches located schema/frame/timestamp/save logic and source line references. Source reads were line-numbered using Get-Content | ForEach-Object; no file was edited.

Exact isolated analysis:
& '{py}' -B '{RUN/'analyze_read_only.py'}'
Result:26.298s;1706cached-landmark feature recomputations,1500TRAINpublisherjoins,5469saved prediction checks;5198explicit nonTESTinputs hashed before. CPUcheckpoint metadata only; zero model forwards/backwards/optimizer/GPUcalls. analysis_console.txt is the captured output.

After-snapshot command (same PowerShell enumeration/hash expression, saved to pilot_files_after.json):
$auditAfter=@(Get-ChildItem -LiteralPath $auditPilot -Recurse -File | Sort-Object FullName | ForEach-Object {{ [PSCustomObject]@{{path=$_.FullName;bytes=$_.Length;last_write_utc=$_.LastWriteTimeUtc.ToString('o');sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}} }})
ConvertTo-Json -InputObject $auditAfter -Depth 5 | Set-Content -LiteralPath (Join-Path $auditRun 'pilot_files_after.json') -Encoding utf8

Report/integrity finalization (new files only):
& '{py}' -B '{RUN/'finish_report.py'}'
Checks5198nonTESTinput hashes again and both76filepilot snapshots; zerochanges. No existing train.py, finish.py or verifier chain executed.

No training, fine-tuning, resumption, model inference, package install, download, external service, GPU/system change, Git mutation, existing-file edit/delete/rename or reservedTESTdata access.
''')
 newfiles=sorted(p for p in RUN.iterdir() if p.is_file() and p.name!='artifact_hashes.sha256')
 md('artifact_hashes.sha256','\n'.join(f'{sha(p)}  {p.name}' for p in newfiles))
 print(json.dumps({'verdict':'PARTIAL','pilot_files_unchanged':len(first),'nonTEST_input_files_unchanged':len(before),'new_report':str(RUN/'READ_ONLY_FAILURE_AUDIT.md'),'training_executed':False,'model_inference_executed':False}))

if __name__=='__main__':main()
