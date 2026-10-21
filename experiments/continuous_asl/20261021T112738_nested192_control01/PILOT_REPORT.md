# Controlled 192-example continuous-ASL learning experiment

Generated 2026-10-21T11:43:21+08:00 (Asia/Shanghai). Exactly one new CUDA training run was completed.

**PASS bounded TRAIN learnability and Android CPU model parity; PARTIAL overall recognition; BLOCKED promotion, new participant collection and independent streaming acceptance.**

## Main finding

The nested 192-example control learned its TRAIN data: **10.59% WER**, versus 12.04% in the frozen 96-example control. Its one-time post-fit evaluation scored **101.03% WER** on 80 held-out sequences, with **0/80 exact matches**. This does not improve recognition or justify changing the app model.

There is no demonstrated learnability breakdown at 192 examples. The model can memorize a larger, varied subset. Generalization remains poor, including sequences whose labels were fitted. Label coverage, representation/annotation adequacy, recording-family distribution and optimization remain interacting explanations; no single cause is proven.

## Frozen design and execution

- Retained every original 96 TRAIN ID, literal target, feature path/hash and selection descriptor. Added exactly 16 unused authorized TRAIN utterances per family: 32 per family, six families, three known signers, 192 unique utterances total.
- Targets: **631 unique labels, 1,464 token occurrences**. Full output vocabulary remains **2,287 gloss labels + blank0/UNK1**. Only the ASLLRP3 literal dominant-realized-occurrence pool was used.
- Fixed 48-channel causal CTC, 378 features, original full-TRAIN normalization, seed20261020, FP32 AdamW lr0.001/weight_decay0.0001, batch4, accumulation2 with actual-tail scaling, clip5 and literal greedy CTC. No feature extraction or label changes.
- Initial model tensors were verified **byte-identical** to the 96-example initialization. Checkpoint container hashes differ because their configuration/subset metadata differ.
- Fixed **30 complete passes**; maximum720 updates/90-second soft stop. Measurements at0,1,5,10,15,20,25,30. No stop at an earlier successful learning gate; no held-out-based selection.
- Gate reused unchanged: TRAIN WER≤30%, CTC≤50%initial, WER reduction≥20percentage points, and each family/rare-bearing/upper-input-quartile WER≤50%. The final control passed all gates.
- Actual **720 optimizer steps / 5,760 utterance presentations**, versus360/2,880 in the 96-example control. This is matched passes, **not matched compute**. Larger vocabulary coverage and example composition also change with added data.
- Measured fit/evaluation/checkpoint stage: **29.991s**. Sum of30 optimizer-epoch durations **20.458s**, mean **0.682s/epoch**. First stage excludes imports/preparation and separate held-out scoring; per-epoch timing excludes periodic scoring/save.
- Peak PyTorch allocated **88.36MiB**, reserved **112.00MiB**. Separate nvidia-smi device-wide observed peak **261MiB**, including CUDA/context overhead. RTX3050 Laptop4GiB. No OOM, skipped batch or nonfinite failure.
- Early blank collapse resolved: epoch1 all blank, epoch5 WER97.40%, epoch15 WER74.86%, epoch25 WER20.29%, epoch30 WER10.59%. CTC CUDA backward remains nondeterministic in warn-only mode; no seed-equivalent causality claim.

## Results by scope

| Scope | Utterances | Reference tokens | WER | Exact sequences | CTC loss |
|---|---:|---:|---:|---:|---:|
| Frozen96_TRAIN | 96 | 756 | 12.037% | 48/96 | 0.567814 |
| New192_TRAIN | 192 | 1464 | 10.587% | 76/192 | 0.532527 |
| Original96_inside_new192 | 96 | 756 | 11.905% | 34/96 | 0.553030 |
| Added96_inside_new192 | 96 | 708 | 9.181% | 42/96 | 0.512024 |
| New192_fixed_heldout80 | 80 | 485 | 101.031% | 0/80 | 11.511077 |
| Original_support5_subgroup27 | 27 | 148 | 100.676% | 0/27 | 10.546657 |

The original96 portion remains at11.90% WER, close to its previous12.04%; exact sequence matches decrease48→34. The added96 reach9.18% WER. Aggregate10.59% should not be read as a proven causal quality gain because the denominators/composition and compute differ. New192 exact sequence accuracy is76/192=39.58%, versus48/96=50% in the old control.

Final TRAIN errors:100substitutions/43deletions/12insertions,155/1,464. No empty predictions;610distinct predicted labels;81.56%argmax blank frames. This blank fraction is not mean blank probability or evidence of collapse by itself.

## Held-out validity, coverage and limitations

- Epoch30 was frozen and hashed before the80evaluation. All80feature hashes/schemas/timestamps/target joins/CTC lengths passed. No recording family crosses the192/80boundary. Targets map exactly into the fixed vocabulary. The original stronger27 are reported as a subgroup and do not select the checkpoint.
- These are known signers and two held-out recording families. The80set has been development-exposed in earlier research. It is held out from this fit, **not a pristine final TEST or unseen-signer evaluation**.
- Model vocabulary covers485/485reference tokens. Actual192training targets cover**386/485(79.59%)**; only**22/80**sequences have all reference labels represented in192TRAIN. The99unsupported tokens limit interpretation of the complete80result.
- The fully supported22 still score**97.50%WER**,117/120edits and0/22exact. The remaining58 score102.19%WER. These descriptive strata retain the original80 score; no restricted subset replaces the main metric or tunes the model.
- The original27strong-support definition refers to support in1500TRAIN, not support in192TRAIN. Its100.68%WER is reported under that unchanged definition.
- The earlier1500-fit checkpoint scored95.67%WER on the same80 under the same label/decoder/normalization conventions, but used different training membership, target support, pass count and checkpoint-selection policy. The new101.03% is descriptively worse, not an equal-budget architecture comparison.
- New held-out output has338S/124D/28I and389hypothesis tokens, with0empty outputs. Substitutions now dominate. Merely suppressing blank or forcing longer predictions has no demonstrated benefit.

## Phone parity and measured model time

Both synthetic tests passed on Samsung **SM-A226B**. A separate camera-free package compiled the actual AppHear model-loader source and synthetic fixtures. The installed AppHear APK path stayed unchanged; no AppHear install/clear, participant capture or microphone use occurred.

**This tests the existing packaged epoch12 model, not promotion/export of the new192 checkpoint.** The harness uses contract constants rather than camera feature extraction, so it establishes CPU model numerical parity, not whole-app lifecycle/feature/camera parity.

| Synthetic input observations | Timed runs after3warmups | p50 model ms | p95 model ms |
|---|---:|---:|---:|
| 1 | 20 | 1.233 | 1.282 |
| 17 | 20 | 4.367 | 5.001 |
| 61 | 20 | 13.733 | 27.430 |

These are CPU inference+tensor/output-copy timings only. No camera cadence, missing-hand rate, streaming false activations, missed signs, sign boundaries or end-to-end latency was measured. Utterance-level metrics cannot fill those gaps.

## Data support, linguistic review and collection

The existing1500TRAIN pool still has**1,182singleton labels** and**1,808labels with fewer than5distinct supporting utterances**. Support counts use distinct utterances, not repeated token occurrences. The new control adds zero newly acquired examples; the earlier3,121eligible/3,025ever-fitted inventory remains unchanged.

- [LABEL_COLLECTION_PRIORITIES.csv](<D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261021T112738_nested192_control01/LABEL_COLLECTION_PRIORITIES.csv>) lists all1,808under-supported literal TRAIN labels with existing signer/family coverage and an explicit planning gap to5utterances. Five is a collection-planning target, not a validated accuracy threshold or instruction to collect every row.
- No qualified ASL reviewer is available, per the user. No consent/permission for new participant recordings has been established. No recording, upload, new annotation or invented gloss was performed.
- Smallest next step: qualified review of the existing12TRAIN events in [review_manifest.csv](<D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T141110Z_alignment_observability_review01/review_manifest.csv>). Reuse that frozen package to distinguish label/timing/visibility problems before another large fit.
- Then choose a small practical set of reviewed existing labels and collect natural continuous phrases with repeated occurrences across consented qualified signers. Reserve independent new signers and recording sessions for evaluation before collecting/training. Preserve authentic annotation provenance; do not translate English into fabricated gloss ground truth.
- Reuse the existing consented-pilot protocol/templates under `20261020T004837_separate02/pilot_package_20261020T014259`. No collection begins until permissions and a qualified annotator are established. Keep ASLLRP3/NCSLGR2 separate.

## Acceptance protocol still required

Before further model selection, freeze consenting signer/session splits and labels, capture conditions, decoder and the following definitions/thresholds. No thresholds were fit in this experiment.

| Measure | Required evidence | Current status |
|---|---|---|
| Continuous gloss WER + exact sequences | Authentic held-out utterance/stream references; per-signer/source results; frozen comparison with confidence limits | Only known-signer development80 diagnostic available |
| False activations | Unconfirmed/confirmed emissions per no-sign minute; idle, conversational motion and hands-visible negatives | BLOCKED: no consented no-sign set |
| Missed signs | Annotated reference gloss deletions/recall and missed events by signer/support | Token edit counts exist; genuine streaming event recall BLOCKED |
| Boundaries | Qualified onset/release labels; onset/end error distributions; split/merge/repeat counts | BLOCKED: qualified review unavailable |
| End-to-end latency | Capture→draft and sign-end→review-ready p50/p95, delivered cadence and missing observations on actual streams | Model-only synthetic time PASS; full pipeline BLOCKED |
| Unseen-signer accuracy | Signers absent from all training/selection; frozen final evaluation | BLOCKED: no verified independent signer pilot |

## Decision and exact artifacts

**Do not promote or automatically launch a larger fit.** The192learning control passes, while held-out transfer remains unsuccessful even for supported labels. The result narrows the failure: fundamental inability to learn192examples or a persistent all-blank decoder is not supported. Feature/label alignment adequacy and support across people/contexts need evidence before assuming additional epochs or a bigger model will solve it.

The next practical action is qualified review of existing examples and permissioned targeted repetition/evaluation collection. A later384nested learning probe is an option for a separately bounded study, not executed or automatically authorized here. Neither the app model nor its thresholds changed.

- New checkpoint: [epoch_030.pt](<D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261021T112738_nested192_control01/checkpoints/epoch_030.pt>)
- Checkpoint SHA256: `da4b47063e64f1eb5fb0dc974f3910befea4ff45c8542e0be9d43f6589dc57d0`.
- Frozen192manifest SHA256: `8a2234a3eafd0d336ebe888d7b8e391b17410540b62d702315d2f5cb0b3c5f48`.
- Logs: [run.log](<D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261021T112738_nested192_control01/run.log>), [epoch_timing.csv](<D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261021T112738_nested192_control01/epoch_timing.csv>), [gpu_monitor.csv](<D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261021T112738_nested192_control01/gpu_monitor.csv>).
- Predictions: [PREDICTION_EXAMPLES.csv](<D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261021T112738_nested192_control01/PREDICTION_EXAMPLES.csv>); [HELDOUT_SUPPORT_STRATA.csv](<D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261021T112738_nested192_control01/HELDOUT_SUPPORT_STRATA.csv>).
- Phone evidence: [phone_instrumentation.log](<D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261021T112738_nested192_control01/phone_instrumentation.log>); [PHONE_PARITY_RESULTS.json](<D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261021T112738_nested192_control01/PHONE_PARITY_RESULTS.json>).
- [control_input_integrity.json](<D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261021T112738_nested192_control01/control_input_integrity.json>) and [postfit_evaluation_integrity.json](<D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261021T112738_nested192_control01/postfit_evaluation_integrity.json>) preserve exact input scope. Prior experiments, exclusions, reserved17TEST, existing checkpoints/manifests and production assets remain unchanged.

Revalidate completed artifacts without fitting:

```powershell
$python = 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261019T155428Z_cslr01\runtime\venv\Scripts\python.exe'
& $python -B 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261021T112738_nested192_control01\verify192.py'
```

The initial training-launch path omitted one nested project-directory component and failed before Python opened the script. The corrected command completed the single fit. The first harness build lost a dotted PowerShell argument; quoting corrected it. The first device command referenced absent SDKbuild-tools36; discovery selected installed35. These setup failures caused no extra fit or model/data change and are recorded in commands_and_results.txt.
