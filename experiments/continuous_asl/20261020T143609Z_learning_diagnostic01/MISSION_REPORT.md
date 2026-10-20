# Fast continuous-ASL learning diagnostic and controlled CUDA candidate

Generated 2026-10-20T22:44:03.253985+08:00 (Asia/Shanghai).

Overall PARTIAL: execution and fixed-subset learnability passed; continuous recognition is not accepted. Candidate promotion:NO. No reserved TEST evaluation, production/Android changes, data acquisition, source relabeling or cadence feature regeneration occurred.

## Answers

1. Fixed subset learned:True. WER407.6923% → 18.5897%; CTC37.663582 → 0.396543.
2. Full candidate trained:True. Complete epochs:13.
3. Compatible held-out WER improved descriptively:True; credible gated improvement:False. See exact denominator/counts below.
4. Subset elapsed3.993s; peak96.68MiB allocated. Candidate duration/memory below.
5. Higher cadence was assessed from metadata only; no new landmarks/features or recognition comparison.
6. Qualified semantic/visual alignment review, streaming/no-sign/latency validity and unseen-signer generalization remain unresolved.
7. Safe to promote:NO. This is exploratory known-signer recording-family research.

## Gate A and preservation

Reused the latest alignment report, prior pilot summary, verified cached-feature and annotation evidence rather than redoing preprocessing. Located the source checkpoint via saved training_status.json, verified its SHA256 e2a6e0046bca4a3f97f33f023dc7346dd9d6226e19e9bd855357a00c8f1579bf. Strict load, vocabulary and normalization buffers passed. A disposable source-weight model performed one real TRAIN CUDA forward/backward/optimizer-step smoke; model state was discarded, never used to warm-start the subset or candidate. Source checkpoint bytes stayed unchanged.

All1,580 used feature files (TRAIN1500+compatible80) passed hash, schema-before-read, float32/378width, finite values, timestamp ordering, transcript metadata and CTC feasibility T>=U+adjacent repeats. All target IDs>=2; blank0/UNK1 remain specials. The existing batch helper's unknown fallback cannot silently affect these verified targets. Frozen IDs and literal targets match the globalTRAIN universe; master/family/content/signer constraints remain unchanged. No two-pool merge occurred.

Accumulation now forms groups of up to two microbatches and divides each loss by the actual group count, then clips/steps once. With375microbatches this gives187pairs plus one correctly scaled singleton. This is experiment-local. It fixes a small confirmed defect, not a claim of the recognition root cause.

## Gate B — fixed subset learnability

Sixteen deterministic, family-balanced TRAIN utterances, stable SHA ordering with prefix subset-20261020|, seed20261020. Exact IDs:subset_manifest.json. Full2,289-class output, saved TRAIN-only normalization, labels, model and optimizer were retained. No DEV/TEST example was used.

Predeclared bounds:200epochs/60s, measurements at1and every10epochs. Pass:WER<=20%,loss<=50%initial,and>=10percentage-point WER reduction (low-initial-WER safeguard documented in config). Passed at epoch60, with29/156errors,blank fraction0.825721,diversity96,empty outputs0. Initial random outputs caused insertion-heavy WER above100%; this is valid token WER, not accuracy. The transient all-blank state at epoch10 resolved without changing data or hyperparameters.

This proves the wiring can memorize these16existing examples. It does not establish full-task learning or transfer. Saved checkpoint:D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T143609Z_learning_diagnostic01\subset_checkpoint.pt; SHA256:9108584df8a9124c18383a28c351cecf9580f625e4b7beb431bc4a6177c97e53.

## Full candidate results

The candidate started from scratch, independent of smoke and subset weights, with the original 48-channel CausalCTC, seed20261020, FP32 AdamW lr0.001/weightdecay0.0001, microbatch4, accumulation2, clip5. It used identical frozen TRAIN1500/primary80/subgroup27, 2,287 literal gloss labels plus blank0/UNK1 and unchanged TRAIN-only normalization/378 features. The modeling/training-variable correction was the accumulation tail; cadence/features/labels/architecture/optimizer were unchanged.

Stopped WALL_BUDGET_PARTIAL_EPOCH after 13 complete epochs; selected epoch12 by the primary80 tuple (WER,CTC loss). FullTRAIN was evaluated every epoch; the same fixed128 monitor was derived from those predictions. Primary80 was evaluated at declared intervals. Strong27 never selected a checkpoint. Every complete epoch has a resumable model/optimizer/RNG checkpoint and logged hash.

|Scope|Utterances|Reference tokens|WER|CTC loss|S/D/I|Blank fraction|Token diversity|
|---|---:|---:|---:|---:|---|---:|---:|
|Selected TRAIN|1500|12423|88.4005%|3.902538|1526/9444/12|0.955894|450|
|Primary held-out|80|485|95.6701%|7.606348|136/328/0|0.931360|49|
|Support5 subgroup|27|148|95.2703%|6.454491|49/92/0|0.921751|24|

Primary80 frozen baseline:96.9072%,470/485 edits (157S/312D/1I). Candidate:95.6701%,464/485 edits. Absolute WER reduction:1.2371 percentage points; relative reduction:1.2766%. Positive means improvement; negative means worse. One edit is100/485=0.206186 percentage points. Credible improvement gate passed:False. FullTRAIN learning screen passed:False.

Data, label units, normalization, vocabulary, decoder/edit counter and primary membership are equivalent. Execution is not identical: fullTRAIN-per-epoch evaluation adds overhead and can consume the wall budget earlier. CUDA CTC backward remains nondeterministic in warn-only mode; even an equal epoch/seed comparison is not proof the tail correction alone caused a score difference. The earlier nine-NCSLGR85.96% baseline remains non-equivalent and is not used for the claim.

Selected experimental checkpoint:
D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T143609Z_learning_diagnostic01\candidate_checkpoints\epoch_012.pt
SHA256:14d0e4669674265ea2be8a3161fa6699e4f813daf4b97b384863dce1daa038ff

Candidate-stage elapsed including fit/evaluation/checkpoint I/O and final selected reload/scoring:188.337s. The field training_loop_seconds in decision.json includes finalization; it is not optimizer-only time. Complete-epoch fit time sums to75.484s; last logged complete-epoch elapsed is176.597s. Peak allocated:1,304,623,616bytes (1.215GiB); reserved:1,342,177,280bytes. See runtime_estimate_after_epoch1.json for the estimate made after the first epoch; estimates are not measured full-run duration.

The180s bound is a soft stop request checked before optimization groups. A group/evaluation/checkpoint/final scoring may finish after it. If partial-epoch work occurred, it is not selected and the last completed-epoch checkpoint is retained; incomplete work is not claimed as a complete epoch. No automatic second fit or continuation ran.

## Gate C — metadata-only cadence assessment

Current retained indices0,3,6,... were compared arithmetically with all native indices at29.97fps on the frozen12-event review manifest. Current focal frame total:8; native-policy expected total:17. All12intervals would include a native frame; the four A intervals regain index coverage. New hand observability, landmark quality and recognition outcomes are UNKNOWN: higher-cadence features/perception were not executed. Qualified review remains blocked and did not block the independent learning diagnostic.

## Example predictions

Full reference/hypothesis/error data:prediction_examples.csv and per-epoch JSONs. Literal labels are unchanged.

- asllrp3_013cf20f34d13a56: reference ["IX-1p", "(L)LIVE", "HOUSE", "IX-loc:i", "WITH", "MOTHER+FATHER", "TWO", "BROTHER"]; hypothesis [].
- asllrp3_086eb559c9e909a1: reference ["IX-3p:i", "ns-fs-MARY", "BORN", "BIG_2", "BABY", "(1h)WOW/AWFUL"]; hypothesis ["IX-2p", "NONE/NOTHING"].
- asllrp3_09bf85196934962b: reference ["ns-fs-MARY", "ACCEPT", "MANY", "DIFFERENT+", "CAR"]; hypothesis ["IX-3p:i", "ONE"].
- asllrp3_0a66b09f16ef3e3a: reference ["ns-fs-JOHN", "READ+", "MANY", "BOOK+"]; hypothesis ["IX-1p", "FINISH"].
- asllrp3_0b4efabf526e5ea0: reference ["IX-3p:i", "ns-fs-MARY", "ACCEPT", "MANY", "DO-DO+", "part:indef"]; hypothesis ["IX-1p", "part:indef"].

## Hardware and uncertainty

NVIDIA GeForce RTX 3050 Laptop GPU,total4.000GiB,free at startup3.222GiB; torch2.8.0+cu128,CUDA12.8. Device tensors and gradients were exercised. Allocated guard2.5GiB/free256MiB/disk10GiB. No OOM or skipped/invalid batches are hidden. No paired CPU speed benchmark was performed, so no CUDA speedup claim. CUDA CTC nondeterminism warning is preserved in run.log.

## Decision and next action

Do not promote. If TRAIN remains above the50%learning screen and held-out errors remain near baseline, this is insufficient full-task learning; a tiny-subset pass does not remedy a long-tail task. The tail correction is not established as a material recognition fix. If TRAIN improves while held-out loss worsens, that is an overfitting/generalization signal, not success. Do not merely repeat longer same-data fits.

Next useful research step: review saved fullTRAIN token-support/error strata and input representation adequacy against a predeclared compact TRAIN learning control; independently obtain qualified adjudication for the12short/hand-missing events. Any architecture/representation or cadence experiment must be separately controlled. No reserved TEST, new acquisition or production action is authorized by these suggestions. No unseen-signer claim: held-out recording families contain known signers and only two family clusters.

## Integrity, scripts and continuation limits

PASS: 1598declared existing/new-source input files have identical before/after SHA256,size,mtime; changed paths:0. input_integrity.json provides exact scope. Production was not modified or independently hashed; no whole-workspace proof. Previous manifests/checkpoints/features/exclusions remain untouched.

Model/optimizer/CPU+CUDA RNG, completed epoch, best/staleness/collapse state and elapsed budget are saved every complete epoch. CONTROL_REVIEW.md records independent source review. The in-script --resume-candidate branch was not exercised; its exclusive integrity-output filename conflicts after a finished/failed invocation and its checkpoint hash needs external verification. Do not use it blindly. Completed checkpoint state is resumable in a new authorized continuation harness/directory; partial timed-out work has not been persisted. This completed bounded run needs no resume. Preserve the frozen script/config hashes rather than modifying them now.

Construction/training command (already executed; do not repeat over completed outputs):
& 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261019T155428Z_cslr01\runtime\venv\Scripts\python.exe' -B 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T143609Z_learning_diagnostic01\run_mission.py' *> 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T143609Z_learning_diagnostic01\run.log'

Read-only validation command:
& 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261019T155428Z_cslr01\runtime\venv\Scripts\python.exe' -B 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T143609Z_learning_diagnostic01\verify_mission.py'

Detailed commands/results:commands_and_results.txt. Final artifact hash index:artifact_hashes.sha256. All new files remain in D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T143609Z_learning_diagnostic01.
