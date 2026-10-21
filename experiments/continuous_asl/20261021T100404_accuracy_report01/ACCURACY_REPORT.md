# AppHear trained-data accuracy report

Created 2026-10-21T10:04:04+08:00 (Asia/Shanghai). Results are from saved experiments completed on 2026-10-20; no new training or evaluation inference was run.

## Overall result

**PARTIAL — the model can learn a compact training set, but reliable continuous-ASL recognition has not been established. Production promotion remains BLOCKED.**

- Latest compact learning control: **12.04% TRAIN WER** on 96 already-seen utterances. This is in-sample learning, not held-out accuracy.
- Latest full-task candidate: **88.40% TRAIN WER** and **95.67% held-out WER** on the compatible 80-sequence evaluation. It failed its learning and credible-improvement gates.
- An independent legacy isolated-sign benchmark has higher classification scores, but uses a different task and feature contract. It cannot be substituted for continuous recognition accuracy.

## What the metrics mean

**WER = (substitutions + deletions + insertions) / reference gloss tokens. Lower is better.** ASL glosses are literal annotation labels, including pointing, classifier and fingerspelling notation; they are not ordinary English word counts.

**Sequence accuracy = exactly matching whole gloss sequences / evaluated utterances. Higher is better.** The exact-match values below were recomputed from saved references and hypotheses. They do not measure temporal boundaries or live-app behavior.

Do not call `100 − WER` ordinary recognition accuracy: insertions, sequence lengths and evaluation scope matter, and WER can exceed 100%. Frame argmax blank fraction is a CTC diagnostic, not confidence or mean blank probability.

## Trained-data coverage

These are the existing frozen inventory totals, not a new raw-dataset recount. The subsequent 16- and 96-utterance controls reuse previously fitted ASLLRP3 examples and add no independent data.

| Label pool | Eligible utterances | Unique utterances ever fitted | Eligible labels | Labels ever fitted |
|---|---:|---:|---:|---:|
| ASLLRP3 | 2,119 | 2,119 | 2,739 | 2,739 |
| NCSLGR2 | 1,002 | 906 | 1,062 | 920 |

- Total eligible: **3,121 utterances**, 23,002 gloss-token occurrences.
- Ever fitted: **3,025 unique utterances**, 22,586 target-token occurrences, **3,256 distinct literal spellings** across the separate pools.
- Eligible display vocabulary: 3,370 distinct spellings; shared spellings do not prove semantic equivalence between the pools. Keep ASLLRP3 and NCSLGR2 supervision separate.
- The older inventory listed 96 NCSLGR2 utterances as prepared but not fitted. These are different from the later 96-example ASLLRP3 compact control.
- Repeated epochs/runs are not additional independent utterances. Some records ever fitted in older experiments are development records in newer splits; each evaluation must be interpreted against its own model initialization and frozen split.

## Latest full-task and compact-control results

| Scope | Utterances | Reference tokens | WER | Exact sequences | Sequence accuracy | CTC loss |
|---|---:|---:|---:|---:|---:|---:|
| Full-task TRAIN | 1,500 | 12,423 | 88.4005% | 15/1500 | 1.00% | 3.902538 |
| Primary held-out | 80 | 485 | 95.6701% | 0/80 | 0.00% | 7.606348 |
| Stronger-support subgroup | 27 | 148 | 95.2703% | 0/27 | 0.00% | 6.454491 |
| Compact TRAIN control | 96 | 756 | 12.0370% | 48/96 | 50.00% | 0.567814 |

The stronger-support 27 sequences are a subgroup of the 80, not a replacement evaluation. The compact control has no new held-out score and cannot be ranked against the full-task evaluation.

| Scope | Substitutions | Deletions | Insertions | Empty outputs | Argmax blank frames | Distinct predicted labels |
|---|---:|---:|---:|---:|---:|---:|
| Full-task TRAIN | 1,526 | 9,444 | 12 | 236 | 95.59% | 450 |
| Primary held-out | 136 | 328 | 0 | 5 | 93.14% | 49 |
| Stronger-support subgroup | 49 | 92 | 0 | 0 | 92.18% | 24 |
| Compact TRAIN control | 36 | 51 | 4 | 0 | 82.25% | 373 |

### Full-task candidate and matched baseline

- Frozen ASLLRP3 internal TRAIN: 1,500 utterances; 2,287 target gloss labels plus blank/UNK; causal 378-feature, 48-channel CTC; TRAIN-only normalization; fixed greedy decoding.
- Completed 13 full epochs; selected epoch 12 using the primary held-out evaluation. Evaluation families are disjoint under the saved guard, but signers are known. This is **not unseen-signer generalization**.
- Matched previous source-transfer baseline: **96.9072% WER**, 470/485 edits. New candidate: **95.6701% WER**, 464/485 edits. Difference: six fewer edits, **1.2371 percentage points**. The predeclared credible-improvement gate did not pass.
- Candidate TRAIN loss 3.902538 versus held-out loss 7.606348. Short output and deletions dominate. TRAIN itself is poorly learned, so poor transfer cannot be attributed solely to signer/source differences or rare labels.
- Measured candidate stage: **188.337 s**, including evaluation/checkpoint/finalization overhead; peak allocated GPU memory **1.215 GiB**. RTX 3050 Laptop GPU. The 180-second limit was a soft stop request; finalization extended elapsed time.

### Compact learning control

- 96 TRAIN utterances, six recording families, three known signers, 404 target labels and 756 reference tokens. Full 2,289-class output vocabulary was retained.
- Passed its predeclared learning gate at **epoch 30 / 360 optimizer steps**. WER fell from 413.89% random-initial output to 12.04%; CTC loss fell from 42.038942 to 0.567814.
- Final errors: **36 substitutions + 51 deletions + 4 insertions = 91/756**. No empty predictions. Worst-family WER 20.13%; rare-bearing WER 13.89%; upper input-length quartile WER 17.72%.
- Measured stage: **15.725 s** including initial/final scoring and checkpoint I/O; peak allocated **88.08 MiB**. Preparation and Python/CUDA import are excluded.
- This demonstrates compact TRAIN learnability; it does not prove full-vocabulary accuracy, held-out transfer or streaming readiness. A high blank fraction alone does not imply collapse: the successful compact result still has 82.25% blank argmax frames.

## Historical continuous-ASL results

**These rows are an experiment inventory, not a comparable leaderboard.** Data pools, DEV IDs, vocabulary, normalization and scope differ. The original nine-sequence 85.96% WER baseline is not equivalent to the later 80-sequence or 413-sequence evaluations. WER is the selected development/selection score, not untouched final TEST accuracy.

| Saved run | TRAIN utterances | TRAIN labels | Selection sequences | Completed epochs | Selected WER |
|---|---:|---:|---:|---:|---:|
| `20261019T155428Z_cslr01/logs` | 30 | 37 | 7 | 14 | 88.000% |
| `20261020_continuous_expand03/models/compact24` | 100 | 115 | 9 | 22 | 96.491% |
| `20261020_continuous_expand03/models/reference48` | 100 | 115 | 9 | 14 | 85.965% |
| `20261020_continuous_expand04/models/compact24` | 181 | 141 | 9 | 27 | 87.719% |
| `20261020_continuous_expand04/models/reference48` | 181 | 141 | 9 | 16 | 94.737% |
| `20261020_continuous_expand05/models/compact24` | 455 | 286 | 9 | 12 | 100.000% |
| `20261020_continuous_expand05/models/reference48` | 455 | 286 | 9 | 20 | 98.246% |
| `20261020_continuous_expand08/models/reference48` | 2119 | 2739 | 9 | 17 | 100.000% |
| `20261020_continuous_expand09/models/reference48` | 906 | 920 | 9 | 20 | 96.491% |
| `20261020T004837_separate02/experiment_B` | 30 | 37 | 7 | 11 | 92.000% |
| `20261020T102748Z_continuation_pilot01` | 2119 | 2739 | 9 | 3 | 100.000% |
| `20261020T104054Z_expand_eval01/pilot_fresh48_v2` | 1706 | 2517 | 413 | 5 | 97.991% |
| `20261020T123237Z_collapse_diagnosis01/pilot64` | 64 | 372 | 413 | 40 | 97.466% |
| `20261020T124813Z_label_support01/pilot497` | 497 | 423 | 92 | 20 | 98.390% |
| `20261020T133148Z_source_transfer_pilot01` | 1500 | 2287 | 80 | 14 | 96.907% |

The 64-example memorization pilot reached approximately 12.22% TRAIN WER but 97.47% DEV WER; that gap is consistent with memorization without useful transfer. Its restricted TRAIN scope differs from the latest 1,500-example candidate.

## Separate legacy isolated-sign benchmark

Archived Experiment A uses 236 classes, legacy feature schema 3.0 with 258 features, and official ASL Citizen signer-disjoint partitions. These top-1/F1 scores are classification metrics, **not continuous gloss-sequence WER or current Android accuracy**.

| Model | Validation top-1 | Validation macro-F1 | Previously frozen test top-1 | Previously frozen test macro-F1 |
|---|---:|---:|---:|---:|
| Unchanged legacy FP32 | 65.13% | 64.12% | 61.22% | 60.22% |
| Fresh masked TCN, five epochs | 19.45% | 15.49% | Not reported here | Not reported here |
| Warm-start primary, five epochs | 67.10% | 66.57% | 62.56% | 61.69% |
| Warm-start repeat, five epochs | 66.54% | 65.78% | 62.18% | 61.45% |

The archived isolated test aggregate covers 2,620 rows from 11 test signer IDs after model decisions were frozen. Values above are quoted from the existing isolated-sign report only; no TEST payload or per-example TEST predictions were opened or reevaluated. This is a separate historical test, not the protected 17-utterance continuous TEST set. Validation has only five signer clusters and correlated cadence views. No thresholds or models were selected for this report.

## Example saved predictions

Examples below preserve literal label conventions. The compact example is in-sample; the primary example is held-out. Neither is a live-camera measurement.

### Compact TRAIN control: `asllrp3_0541021d00521482`

Reference: `["IX-1p", "LIKE", "IX-3p:i", "CHOCOLATE", "fs-CEREAL", "BUT", "NONE/NOTHING", "(25)WHY", "part:indef"]`

Hypothesis: `["IX-1p", "LIKE", "IX-3p:i", "CHOCOLATE", "fs-CEREAL", "BUT", "NONE/NOTHING", "(25)WHY", "part:indef"]`

### Primary held-out: `asllrp3_013cf20f34d13a56`

Reference: `["IX-1p", "(L)LIVE", "HOUSE", "IX-loc:i", "WITH", "MOTHER+FATHER", "TWO", "BROTHER"]`

Hypothesis: `[]`

## Interpretation and readiness

| Finding | Status | Meaning |
|---|---|---|
| Saved metrics and selected checkpoint integrity | PASS | Four saved prediction sets independently reproduce edit counts/WER; selected candidate and compact checkpoint hashes match. |
| Compact TRAIN learning | PASS | The current pipeline learns a varied 96-example subset. |
| Full-task learning and held-out recognition | PARTIAL | High TRAIN/held-out error persists; credible improvement gate failed. |
| Single implementation root cause | UNKNOWN | Existing mapping/CTC checks pass; full-task scale, optimization, sparsity and representation remain interacting hypotheses. |
| Unseen-signer continuous accuracy | BLOCKED | The current source-transfer evaluation uses known signers. |
| Streaming false activations, missed signs, boundaries and latency | BLOCKED | No verified consented streaming/no-sign evaluation establishes these measures. Utterance WER is not a substitute. |
| Qualified linguistic/visual alignment review | BLOCKED | No qualified reviewer has been established. |
| Production promotion | BLOCKED | These research outcomes do not justify replacing production models. |

The strongest supported explanation is **scale-dependent full-task learning failure with poor transfer**, rather than a demonstrated blank-ID or WER bug. Long-tail support is a plausible contributor, not an established sole cause: common labels also fail in the saved audit. The compact success supports learnability of that subset only.

The existing next-step recommendation is a controlled nested 96-to-192 TRAIN-only scale probe with the same model, vocabulary, normalization and learning schedule, followed by an independently valid held-out protocol. This report authorizes and executes no additional training. Do not treat longer same-data training as a proven remedy.

## Evidence and preservation

For this report, **23 archived status-file hashes** were checked, four saved prediction sets were independently rescored, and the two latest selected checkpoints were SHA256-verified. This is a bounded evidence check, not a fresh full-dataset or feature audit. All prior results, checkpoints, manifests, exclusions and production assets remain unchanged.

- Inventory and original experiment history: [FULL_TRAINING_REPORT.md](<D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T134156Z_gloss_training_report01/FULL_TRAINING_REPORT.md>)
- Full-task candidate report: [MISSION_REPORT.md](<D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T143609Z_learning_diagnostic01/MISSION_REPORT.md>)
- Compact learning-control report: [MISSION_REPORT.md](<D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T144622Z_compact_ctc_control01/MISSION_REPORT.md>)
- Isolated-sign archived aggregate report: [isolated_sign_training_report.md](<D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T004837_separate02/isolated_sign_training_report.md>)
- Full trained gloss list: [ALL_TRAINED_GLOSS_LABELS.txt](<D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T134156Z_gloss_training_report01/ALL_TRAINED_GLOSS_LABELS.txt>)
- Structured recomputed accuracy evidence: [accuracy_evidence.json](<D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261021T100404_accuracy_report01/accuracy_evidence.json>)
- Exact source evidence hashes: [source_hashes.sha256](<D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261021T100404_accuracy_report01/source_hashes.sha256>)

### Selected checkpoint hashes

- Full-task candidate epoch 12: `14d0e4669674265ea2be8a3161fa6699e4f813daf4b97b384863dce1daa038ff`.
- Compact TRAIN control epoch 30: `2ff4b2aa7231c4d3d5cf90dded03dcff224e8c311b5dc91893aecf991f5c4db7`.

**Final decision: PASS report/evidence integrity; PARTIAL research recognition; BLOCKED continuous model promotion. No acquisition, new fitting, checkpoint selection, TEST tuning, production change or upload was performed.**
