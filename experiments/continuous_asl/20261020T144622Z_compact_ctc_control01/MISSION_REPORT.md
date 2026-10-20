# Full-task CTC diagnosis and one compact learning control

Generated 2026-10-20T22:54:50.401721+08:00 (Asia/Shanghai).

**PARTIAL overall diagnosis; PASS numerical checks and compact learning.** One new96-utterance TRAIN-only CUDA fit ran. It passed every predeclared global/subgroup learning gate. No full-data fit, prior16-example rerun, held-out evaluation of the new control, acquisition, new feature extraction or promotion occurred.

## Confirmed failure signature from saved full-task evidence

The previous selected epoch12 checkpoint remains unchanged: SHA25614d0e4669674265ea2be8a3161fa6699e4f813daf4b97b384863dce1daa038ff. Historical metrics are preserved, not replaced by this easier-scope experiment.

|Saved scope|WER|S/D/I|Reference tokens|Hypothesis tokens|Hyp/ref length|Argmax blank|Distinct predictions|
|---|---:|---|---:|---:|---:|---:|---:|
|TRAIN1500|88.4005%|1526/9444/12|12423|2991|0.2408|95.5894%|450|
|Compatible held80|95.6701%|136/328/0|485|157|0.3237|93.1360%|49|

Deletions and short output dominate; this is not strictly a single-token collapse. TRAIN has236empty hypotheses; held80has5. Saved posterior probabilities/logits do not exist, so posterior confidence/entropy/calibration are UNKNOWN. Blank figures are argmax frame fractions, not mean blank probability.

All485held-out reference tokens have TRAIN-seen labels. Aligned held-out recall is21/251=8.37% for labels seen in20+distinct TRAIN utterances and0/234 for lower support. TRAIN20+recall is only15.68%. Thus rare/unseen labels cannot be the sole explanation. Long-tail pressure is substantial:1182/2287labels occur in one distinct TRAIN utterance and1808/2287in fewer than5. diagnosis_by_label.csv preserves every literal label's distinct-utterance support, token counts, aligned recall, deletions/substitutions and confusion alternatives. Alignments use the verified diagonal/substitute,delete,insert tie order; label recall is alignment-dependent, not a new ground-truth annotation.

All1580saved sequences are CTC feasible (minimum slack5);17adjacent-repeat targets are accounted for. No empty targets, blank/UNK targets, downsampling length mismatch, normalization/vocabulary/decoder drift or frozen-membership inconsistency was found. Existing exact feature reconstruction and source joins were reused; selected96feature files were rechecked for hash/schema/dtype/finite/timestamp/transcript/length integrity, not regenerated.

FullTRAIN WER continues100%epoch1→88.40%epoch12→84.61%epoch13; loss6.591→3.728. Held-out loss worsens6.627epoch1→7.606epoch12. These show incomplete full-task learning plus weak transfer, not training convergence or proof that further epochs solve it.

Cached observation-gap associations are inconsistent across splits: TRAIN gap-flagged WER91.35% versus79.16% without flag, while held-out reverses94.94% versus97.67%. Duration/support/source are confounded; no causal visibility claim follows. Missing-hand flags are detector states, not physical absence. Detailed strata, temporal lengths and per-utterance S/D/I/length ratios are in the diagnosis CSV/JSON files.

## Frozen compact design

The manifest was selected before this fit using only authorized internalTRAIN metadata/targets, never held-out outcomes. Seed prefix:compact96-seed20261020; model seed20261020. Six TRAIN families receive16utterances each. Greedy selection first covers unused combinations of family/input-length quartile/target-length quartile/observation category/rare-bearing status, then masters and underrepresented marginal categories, with stable SHA ties. Existing manifests and source labels were not edited.

Coverage:96utterances,all3available TRAIN signers,all34masters,404literal labels,756tokens. Among selected labels, full-TRAIN distinct-utterance support is68singletons,90with2–4support,82with5–9,78with10–19,and86with20+. Input lengths15–100observations; target lengths2–27tokens. Input quartile counts26/26/24/20; target counts26/25/23/22. There are33utterances with unsampled publisher intervals and54with sampled/no-observed-hand intervals; categories may overlap. No low-quality example or literal target was discarded.

Equal-family quotas overrepresent small families; signer distribution is16/64/16. This is coverage-balanced TRAIN learning evidence, not a random population sample or unseen-signer test. It covers404/2287labels, not the entire task or all ambiguous examples. The previous16-example experiment was not rerun.

Frozen subset SHA256:c7b78f1aef43b5501fce5fafbf69e91753e5d9d85cb594a3e18fc693ecfb5198. All subset IDs are in the original1500TRAIN; full2,289-class vocabulary, blank0/UNK1, saved fullTRAIN-only normalization,378features and original48-channel causal model remain unchanged. AdamW0.001/weightdecay0.0001,FP32,batch4,accumulation2,clip5. Actual-group-count scaling preserves the prior tail correction. No warm start, architecture/feature/cadence/label/optimizer change was bundled.

Predeclared limits:120epochs,1440optimizer steps or90second stop request. Evaluate initial,epoch1,and every5epochs. Pass requires WER<=30%,loss<=50%initial,>=20percentage-point WER reduction,and WER<=50%for every family,rare-bearing utterances and upper input-length quartile. Numerical/OOM/resource faults stop immediately. No new full training was authorized automatically.

## Measured compact learning

Passed at epoch30,optimizer steps360. All metrics below are in-sample TRAIN; they are not comparable held-out recognition scores.

|Epoch|Optimizer steps|WER|CTC loss|Argmax blank|Hyp/ref length|Distinct predicted labels|
|---|---:|---:|---:|---:|---:|---:|
|0|0|413.889%|42.0389|0.0000|4.1429|94|
|1|12|100.000%|10.8910|1.0000|0.0000|0|
|5|60|98.942%|5.8106|0.9963|0.0198|1|
|10|120|93.915%|5.0469|0.9683|0.1455|11|
|15|180|86.772%|3.9619|0.9476|0.2553|46|
|20|240|69.048%|2.5939|0.9157|0.4272|134|
|25|300|42.725%|1.3194|0.8801|0.6376|262|
|30|360|12.037%|0.5678|0.8225|0.9378|373|

Final errors:36S/51D/4I = 91/756. Empty outputs:0. CTC loss fell98.65%. Random initial WER>100% reflects insertion-heavy token sequences, not negative accuracy.

|Family|Utterances|Final WER|
|---|---:|---:|
|asllrp_dai_participant_2::2013-06-27|16|14.894%|
|asllrp_dai_participant_3::2011-12-01|16|12.745%|
|asllrp_dai_participant_3::2011-12-08|16|11.176%|
|asllrp_dai_participant_3::2012-01-27|16|6.154%|
|asllrp_dai_participant_3::2012-02-09|16|6.306%|
|asllrp_dai_participant_5::UNRESOLVED_ALL_UNDATED_MASTERS|16|20.134%|

Rare-bearing WER:13.889%; upper-input-quartile WER:17.722%. Worst family:20.134%. Thus the improvement is not confined to one easy family or short examples. Blank fraction remains82.25%while WER is12.04%; a high CTC blank fraction alone is not evidence of failed recognition.

Measured compact-stage elapsed:15.725s,including initial/final scoring and checkpoint I/O,but excluding Python/CUDA import and preparation/final integrity. Peak allocated:92,356,608bytes (88.08MiB); reserved:112.00MiB. GPU:NVIDIA GeForce RTX 3050 Laptop GPU,torch2.8.0+cu128,CUDA12.8. No OOM/nonfinite/invalid batch occurred. The90sbound is a soft group-boundary stop request; final scoring/hashing may add overhead. Here it passed well within budget. Diagnostic fit_seconds includes periodic evaluation/save overhead and is not pure optimizer time. No CPU/GPU speedup was measured or claimed. CUDA CTC nondeterminism warning remains in run.log.

Checkpoint:D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T144622Z_compact_ctc_control01\checkpoints\epoch_030.pt
SHA256:2ff4b2aa7231c4d3d5cf90dded03dcff224e8c311b5dc91893aecf991f5c4db7

The checkpoint contains model,optimizer,CPU/CUDA RNG,exact vocabulary/config/subset hashes,completed epoch and optimizer count. Partial checkpoints,if a budget stop occurs,include shuffled IDs and next group index at an optimizer boundary. No resume CLI was implemented or exercised; state is sufficient for a separately authorized continuation harness. This successful fit needs no continuation.

## Examples

Complete final examples:final_predictions.json. No labels were silently normalized or substituted.

- asllrp3_48507e7eb82da9b8: reference ["PEOPLE", "IX-1p", "1p-MEET+++:i/j/k/l", "PAST+NIGHT", "WOW", "(L-X)INTERESTING"]; hypothesis ["IX-1p", "1p-MEET+++:i/j/k/l", "PAST+NIGHT", "WOW", "(L-X)INTERESTING"].
- asllrp3_12c3a93500af51aa: reference ["PAST+NIGHT", "(5)ACCIDENT", "PCL:5\\long line of cars\\\"\"", "(1h)WOW/AWFUL"]; hypothesis ["IX-1p", "(5)ACCIDENT", "PCL:5\\long line of cars\\\"\"", "(1h)WOW/AWFUL"].
- asllrp3_1399c3f7e54b60cf: reference ["IX-3p:i", "BOOK", "READ+", "REALLY", "(1h)GOOD/THANK-YOU", "#NO", "LOUSY", "(1h)5\\wave no\\\"\"", "IX-1p", "NOT", "LIKE"]; hypothesis ["GAME", "BOOK", "READ+", "(1h)GOOD/THANK-YOU", "#NO", "LOUSY", "(1h)5\\wave no\\\"\"", "IX-1p", "NOT", "LIKE"].
- asllrp3_0541021d00521482: reference ["IX-1p", "LIKE", "IX-3p:i", "CHOCOLATE", "fs-CEREAL", "BUT", "NONE/NOTHING", "(25)WHY", "part:indef"]; hypothesis ["IX-1p", "LIKE", "IX-3p:i", "CHOCOLATE", "fs-CEREAL", "BUT", "NONE/NOTHING", "(25)WHY", "part:indef"].
- asllrp3_2debd9737dec3941: reference ["RAIN++", "(1h)WOW/AWFUL"]; hypothesis ["RAIN++", "(1h)WOW/AWFUL"].

## Evidence versus hypothesis

|Explanation|Assessment|
|---|---|
|Fundamentally invalid CTC/decoder/target mapping|NOT_EVIDENCED; saved checks and representative compact learning pass.|
|Inadequate full-task learning under the previous bound|CONFIRMED state; not fullTRAIN convergence. Compact success strengthens this diagnosis but does not identify the sufficient remedy.|
|Severe long-tail sparsity as the sole cause|NOT_SUPPORTED; common labels fail too. Sparse support remains a plausible contributor to optimization and transfer.|
|Optimization dynamics/task scale/effective exposure|LIKELY contributor, not causally isolated: same unchanged pipeline learns96varied examples in30passes, whereas full1500remains poor after13passes. Dataset size,interference and per-example exposure differ.|
|48-channel capacity/representation limit at full scale|POSSIBLE; compact success rules out inability to learn these404labels,not capacity/information adequacy for2287labels.|
|Missing input observability/annotation semantics|POSSIBLE and unresolved; compact includes flagged samples but cannot establish their visual/linguistic sufficiency.|
|New held-out/unseen-signer improvement|NOT_TESTED by this compact run; old held-out95.67%remains frozen.|

No single root cause is established. The strongest supported explanation is a scale-dependent full-task learning failure with long-tail/representation/optimization factors still interacting,rather than a demonstrated blank-ID or WER implementation defect.

## Next single experiment — recommendation only

Test **TRAIN-set size only:96→192 nested utterances**. Retain these exact96and add16deterministic unused TRAIN utterances per family under the same coverage rule. Keep full vocabulary,original1500-fitted normalization,model48,features,cadence,labels,optimizer,seed and30complete passes/evaluation schedule fixed. Use matched strata and report label-support/composition drift. Dataset size is the intended changed variable; total optimizer steps necessarily double360→720and must be reported as a consequence,not claimed matched compute.

This is a cheap scale probe before full training: if learning deteriorates sharply,it narrows the scaling/interference/capacity question; if192alsolearns,it still does not prove full-task transfer or that more full-data epochs suffice. A later matched-step control may be needed for causality. Do not change channels,blank bias,loss weights,labels or cadence simultaneously. No192fit or full new training ran now; any follow-up requires an explicit bounded compute allowance.

## Integrity, commands and disposition

PASS:200explicit input paths have unchanged SHA256,size,mtime. Preparation verified all90prior output hashes; Phase1 verified its38saved inputs; the control rechecked96features and frozen local copies/manifest/config. Exact merged scope:input_integrity.json. No existing checkpoint,label,manifest,feature,report,production/Android/GLB asset or frozen evaluation was modified. Production assets were not independently hashed; no whole-workspace claim. No TEST access,acquisition,install,Git change or competing GPU job.

decision.json records PASS for the compact gate. mission_decision.json records PARTIAL for unresolved overall cause. Candidate promotion:NO; no product accuracy or unseen-signer claim.

Executed preparation:
& 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261019T155428Z_cslr01\runtime\venv\Scripts\python.exe' -B 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T144622Z_compact_ctc_control01\control.py' --prepare
& 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261019T155428Z_cslr01\runtime\venv\Scripts\python.exe' -B 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T144622Z_compact_ctc_control01\verify_prepared.py'

Executed one fit:
& 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261019T155428Z_cslr01\runtime\venv\Scripts\python.exe' -B 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T144622Z_compact_ctc_control01\control.py' --train *> 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T144622Z_compact_ctc_control01\run.log'

Finalization:
& 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261019T155428Z_cslr01\runtime\venv\Scripts\python.exe' -B 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T144622Z_compact_ctc_control01\finish_compact.py'

To validate completed artifacts without retraining:
& 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261019T155428Z_cslr01\runtime\venv\Scripts\python.exe' -B 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T144622Z_compact_ctc_control01\verify_compact.py'

commands_and_results.txt records commands and results. Do not rerun preparation/training over completed outputs.
