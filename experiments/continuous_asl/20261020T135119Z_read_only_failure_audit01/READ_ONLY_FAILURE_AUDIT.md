# Read-only failure audit: continuous-ASL source-transfer pilot

Generated 2026-10-20T22:03:56.946058+08:00 (Asia/Shanghai).

**PARTIAL — evidence narrowed, but a key diagnostic remains unresolved.**

The pilot is computationally intact and demonstrably underlearned. No critical vocabulary, blank-ID, WER, file-join, feature-recomputation or normalization defect was found. New read-only evidence reveals **missing direct temporal/hand observations for a material fraction of annotated sign intervals**, while the long-tail target task and held-out-based stopping rule leave full-TRAIN learnability unestablished. These facts narrow the problem; they do not isolate a single causal root cause or prove that more epochs would fix it.

One small implementation defect is confirmed: the final odd gradient-accumulation group is half-scaled. Its scale is too small to call it the primary cause without an ablation. No patch, model inference, backward call, optimizer step, package installation, download, GPU/system configuration change, Git command or write to an existing file occurred. Checkpoints were deserialized **onto CPU for metadata/tensor inspection only**; no model or optimizer instance was constructed. Cached-landmark feature math was recomputed without perception/model inference.

## 1. Evidence and early stopping reconciled

All four pilot Python files were inspected: common.py, train.py, finish.py and verify.py (the remaining implementation file), plus PILOT_REPORT.md, config, METRICS.csv, logs, manifests, vocabulary, normalization, initialization/epoch12/epoch14 metadata and saved predictions. The pre-existing75-artifact checksum index matches the full pre-audit76-file snapshot (including the index itself).

Fourteen complete epochs,188updates/epoch,2,632updates and21,000utterance presentations are consistent. The selected checkpoint is epoch12, the minimum logged primary80(WER,CTCloss) tuple. This is a local research pointer, not a promotion or proof of acceptance. Selected epoch12 fullTRAIN must not be confused with final epoch14's128-example monitor.

Epoch6 has472edit errors/485references, WER97.3196%; epoch12 has470/485, WER96.9072%. The two-edit gain is **0.4124percentage points**, below the0.5point credible-reset threshold (at least3edit errors on485references). Epoch8,10,12and14 therefore produce four stale evaluations under the declared rule, even though epoch12 is a measurable local improvement. The epoch14NO_CREDIBLE_PRIMARY_GAIN stop is implemented consistently; it does **not** establish that TRAIN had saturated. See stopping_reconciliation.json and [train.py:129](D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T133148Z_source_transfer_pilot01/train.py:129).

Optimization loss falls8.2549(epoch1)→3.7945(epoch14); monitor loss falls6.5485→3.5026 and monitorWER100%→85.346%. PrimaryCTC rises from its epoch2minimum6.5720 to8.0112at14; primaryWER only improves slightly. Parameters and optimizer states change measurably:

|Checkpoint|Epoch|Saved optimizer steps|Parameter L2 change from initialization|Learning rate|
|---|---|---|---|---|
|initialization.pt|0|0–0|0.0000|0.001|
|epoch_012.pt|12|2256–2256|87.1358|0.001|
|epoch_014.pt|14|2632–2632|93.8056|0.001|

All model tensors are finite; checkpoint vocabulary, normalization buffers, config hash and fold hash agree. Model has186,801trainable scalar parameters by state-tensor count excluding mean/std. Initialization is fresh random, not recognition-pretrained. Full checkpoint metadata and shapes are in checkpoint_metadata_CPU_only.json.

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

Sources: [ctc_experiment.py](D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/tools/continuous_asl/ctc_experiment.py:22), [CTC batch/decoder](D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/tools/continuous_asl/ctc_experiment.py:79), [optimizer and initialization](D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T133148Z_source_transfer_pilot01/train.py:63), [scoring](D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T133148Z_source_transfer_pilot01/train.py:81). The frozen decoder source hash matches config. Raw per-frame logit sequences were not saved, so no new independent argmax replay was attempted.

### Confirmed small accumulation-tail defect

1,500/4=375microbatches, an odd count. At [train.py:109](D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T133148Z_source_transfer_pilot01/train.py:109), every loss is divided by2; at110–113the final lone microbatch is still stepped. Its gradient is therefore half the average-of-available-microbatches scale. One of188updates/epoch is affected (0.532%); nominal utterance-gradient weighting shortfall is0.133%per epoch. Adam moments make the eventual parameter effect nonlinear, so this is **not** a proved0.133%performance effect. It is CONFIRMED as a small implementation issue, but **NOT EVIDENCED as the primary recognition failure**. No correction was executed. A pure arithmetic final-group divisor check, not a fit, is sufficient to demonstrate it; any later code correction belongs in a new authorized experiment.

## 3. Dataset/split integrity and feature→target provenance

All1,500TRAIN and206heldout IDs are exact members of the cited globalTRAIN universe. They match the frozen source-fold IDs. Raw video hashes, feature hashes, embedded feature metadata, cached raw metadata and publisher annotation joins agree. Every ordered target equals its event-label sequence, and each event's ID, label, source row, utterance/video/master identity and native frame interval matches the original local publisherCSV. Current signer IDs are publisher-query sidecar joins added after extraction; an absent signer in older embedded raw metadata is not silently replaced with visual identity inference.

All1,706feature sequences exactly reproduce from cached raw landmarks using the existing causal378function ([learned_features.py](D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/pipeline/perception_vnext/segmentation/learned_features.py:159)). Recomputed1,500TRAIN-only mean/std matches saved normalization bit-for-bit. No empty features/targets, nonfiniteX, shape mismatch, nonmonotonic timestamps, globalCTC length failure or exact duplicateXsequence was found. Native stored frame counts equal inclusive publisher utterance spans; media bytes match the stored hashes. No new video/perception inference was run, so visual boundary correctness is not independently certified.

|Group|Observation length|Target length|MinimumCTCslack|Mean fraction with any hand observed|Mean body availability|Mean fraction clipped beyond±10|
|---|---|---|---|---|---|---|
|TRAIN|8–246|2–42|5|77.27%|99.52%|0.0254%|
|PRIMARY80|16–64|3–17|11|73.83%|99.29%|0.0653%|
|STRONG27|18–42|3–9|14|78.01%|99.54%|0.0391%|

One TRAIN sequence has zero hand-XY-delta energy: `asllrp3_f27e40aae77bb6f2` (28observations,5targets,hand-observed fraction0.357). It is a signal-quality flag, not proof of a corrupt file or a static actual sign. No all-zero/duplicate sequence was silently removed.

TRAIN/heldout overlap is zero for utterance IDs, master-recording IDs, collections, raw media hashes, decoded-pixel hashes and conservative recording families. Evaluation contains two already-seen TRAIN signers;3signers occur inTRAIN. All undated Ben masters stay together; date-shaped master prefixes remain conservative family proxies, not proven capture dates. Source/signer causal effects and true parent aliases cannot be fully separated. The pilot code reads only the cited universe filtered toglobalTRAIN([train.py:17](D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T133148Z_source_transfer_pilot01/train.py:17)); evaluation functions consume only these206heldout records and the128TRAIN monitor. Static code/data-flow and saved memberships provide no evidence of reservedTESTloading. No reservedTESTdataset files were opened by this audit. Absolute proof of unlogged historical process behavior is unavailable.

## 4. Newly measured temporal/visibility gaps

The nominal10Hz policy selects the first native frame at least100msafter the previous retained frame ([prepare_asllrp.py](D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020_continuous_expand06/tools/prepare_asllrp.py:78)); actual median spacing is100.1ms. GlobalCTC feasibility only asks whether totalT can encodeU plus repeated labels. It does not ensure even one retained observation lies inside every sign interval.

|Group|Publisher target events|No retained native frame in event interval|Has retained frame(s), but no observed hand in them|
|---|---|---|---|
|TRAIN|12423|880 (7.08%)|1440 (11.59%)|
|PRIMARY80|485|40 (8.25%)|49 (10.10%)|
|STRONG27|148|12 (8.11%)|8 (5.41%)|

The two gap categories are disjoint. Together they affect2,320/12,423TRAIN intervals (18.68%) and89/485primary intervals (18.35%). These are exact cached-frame-index/observed-mask counts relative to **publisher** sign spans, not newly adjudicated ASL ground truth. Nearby frames or body/context may still contain evidence, and 'hand observed' is a detector state. The counts do not establish that each affected label is impossible to learn. Hidden-event flags are retained (647TRAIN,8primary); their physical/linguistic meaning was not independently verified here.

Examples using stable IDs only:

- `asllrp3_00808136f6af5bbc`: literal target `IX-3p:i`, relative source frames[31, 32], duration66.7ms, **zero kept frames in that annotated interval**.
- `asllrp3_00db627b2bb6575f`: literal target `TWO`, relative source frames[47, 47], duration33.4ms, **zero kept frames in that annotated interval**.
- `asllrp3_00db627b2bb6575f`: literal target `DEAF`, relative source frames[268, 268], duration33.4ms, **zero kept frames in that annotated interval**.
- `asllrp3_0107eafd515f94d0`: literal target `IX-3p:i`, relative source frames[11, 11], duration33.4ms, **zero kept frames in that annotated interval**.

This makes a temporal-evidence/preprocessing limitation **POSSIBLE**, with the missing direct observations themselves **CONFIRMED**. It is not evidence of a wrong feature/transcript file join. Byte-perfect feature reproduction can coexist with a task-inadequate sampling or landmark representation. Before another fit, a small qualified visual review of the existing referenced intervals is needed to distinguish true sampling/refinement loss, annotation-boundary semantics and contextual observability. No cadence, frame, target or feature was changed.

## 5. Prediction collapse and metric validity

An independent1DLevenshtein implementation checked5,469saved prediction examples across37scorefiles. Every reference matches the frozen transcript; every edit-distance total and aggregateWER matches. The same literal token unit is used on both sides, including classifier strings and whole fingerspelled words. Unnecessary participant names are not exposed in this report.

|Selected scope|WER|S/D/I|Reference tokens|Blank frame fraction|Empty outputs|Distinct predictions|UNK outputs|
|---|---|---|---|---|---|---|---|
|train1500|87.0804%|1541/9258/19|12423|95.307%|262|507|0|
|monitor|88.2771%|153/840/1|1126|95.112%|19|121|0|
|primary80|96.9072%|157/312/1|485|92.321%|6|57|0|
|strong27|96.6216%|51/92/0|148|92.573%|2|28|0|
|full206|97.8818%|534/1221/1|1794|93.149%|14|101|0|

The selected checkpoint is blank-dominated in frame space, but it is **not** an all-blank or one-token-only model:507TRAIN labels and57primary labels are emitted. Its primary dominant token share is29.89%, not the≈99%IX-1pcollapse of an earlier model. Repeated decoded labels can be correct CTCbehavior when separated by blanks; no new raw-logit replay was performed. Detailed token counts, length histograms, unknown counts and repeated-hypothesis counts are in prediction_collapse_summary.json.

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

PowerShell was used for initial/full-target hash snapshots and source/report inspection. Isolated Python-Bwas necessary for NPZ/math, independent edit distances and CPU-only trusted checkpoint metadata. No existing training or finalizer script was executed. No old verifier chain was run because its broad protected-input hashing could open reservedTESTcontents; equivalent target/control/source checks were performed on the explicit nonTESTscope here. Exact commands, output paths and scope are in commands_and_results.txt. All new writes remain under `D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T135119Z_read_only_failure_audit01`.

Limitations: no model replay, no native video decode or qualified visual annotation, no causal ablation, no population confidence from two families, no authenticated capture-date lineage beyond conservative grouping, and no proof of unlogged historical behavior. Missing frame/hand observations are not automatically mislabeled samples. Target semantic adequacy and the cause of full-task underlearning remain unresolved.

**Final verdict: PARTIAL — evidence narrowed, but a key diagnostic remains unresolved.**
