# Experiment A — executed, modest repeated validation gain

Eligible existing data were used immediately without waiting for How2Sign. Exact TRAIN/VAL source hashes and tensor schemas: experiment_A/data_audit.json. Signer sets are disjoint and concordant with official ASL Citizen partitions. Class vocab236, feature schema3.0 [T,258], masks and valid sequence lengths4–96. All feature blocks remain; no geometric augmentation or negative relabeling. CPU mmap staging avoids loading the whole corpus into VRAM. No TEST payload was loaded for checkpoint/threshold/augmentation decisions. Metrics below are isolated-clip validation, not signed-phrase or current live-app accuracy.

| Candidate | Completed epochs | Validation top1 | Macro-F1 |
|---|---:|---:|---:|
| Unchanged legacy INT8 TFLite | no training | 64.28% | 63.01% |
| Unchanged legacy FP32, parity-gated Keras transfer | no training | 65.13% | 64.12% |
| Fresh masked TCN | 5 | 19.45% | 15.49% |
| Warm-start clip-objective, seed20261020 | 5 | 67.10% | 66.57% |
| Independent repeat, seed20261021 | 5 | 66.54% | 65.78% |

The scratch model was not competitive within the bounded run. A justified second candidate transfers the unchanged FP32 model, preserving exact causal block/LN/skip/head math. CPU Keras/PyTorch transfer maximum error1.55e-6, no argmax changes. Its new loss is NLL of mean real-frame class probability, matching the existing clip decision; learning rate1e-4, AdamW, microbatch4, accumulation8 (effective32), no AMP, no augmentation, gradient cap5. Each run is capped at5epochs/600s, with atomic per-epoch checkpoints, optimizer/RNG states and immutable numbered checkpoint files. Seeds differ only in the repeat; checkpoint selection uses validation macro-F1 then top1.

Against the matched FP32 baseline, primary top1 changes by 1.96 percentage points and macro-F1 by 2.45; repeat changes by 1.41/1.66 points. Primary paired signer bootstrap95% interval for top1 difference: [0.82, 2.62] points,2000 resamples. Only five signer clusters, correlated cadence views and validation checkpoint selection limit inference; no independent final-test claim. Two seeds support a modest validation effect, not production readiness.

Per-class precision/recall/F1, confusion matrices, per-signer outcomes and predictions are retained in validation JSONs and confusion CSVs. Positive-only confidence/coverage curves are in paired_comparison.json; no operating threshold was selected. False activations, real false rejection and hard negatives are unmeasured. New fingerspelling training is BLOCKED: no eligible temporal letter-sequence dataset was verified. Existing spelling paths/tests are preserved.

Primary checkpoint: `D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T004837_separate02/experiment_A/warmstart/checkpoints/epoch_004.pt`. Repeat checkpoint: `D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T004837_separate02/experiment_A/warmstart_repeat/checkpoints/epoch_003.pt`. Exact configurations, hashes, per-epoch time/worker RAM/VRAM and logs remain in their separate directories. Current production WORD expects350 features and is not directly comparable; its accuracy improvement is NOT MEASURED. No production checkpoint selected.


Final native-test evaluation was added after training/checkpoint/threshold decisions were frozen. On 2620 rows from 11 test signer IDs, disjoint from TRAIN/VAL: frozen FP32 top1 61.22% / macro-F1 60.22%; primary 62.56% / 61.69%; repeat 62.18% / 61.45%. Primary gains 1.34 top1 points (35 additional correct rows), repeat 0.95 points (25 additional correct rows). No post-test training or selection occurred. This is a legacy258 isolated-sign benchmark; current live-app/negative/boundary behavior remains unmeasured. See final_test_report.md and final_test_freeze.json.
