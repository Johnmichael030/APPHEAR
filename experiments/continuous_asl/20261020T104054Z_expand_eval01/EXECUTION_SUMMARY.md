# AppHear continuation outcome

**Data goal: PASS. Recognition readiness: PARTIAL. Production promotion: BLOCKED.**

The latest user instruction makes **2,000** the active data target. There are **3,121 verified eligible utterances**, including **96 newly admitted NCSLGR utterances**. Further acquisition has stopped. Earlier 5,000-target/1,879-shortfall fields in frozen intermediate artifacts are historical and superseded by `goal_update.json`; their bytes were preserved.

| Finding | Classification | Evidence and scope |
|---|---|---|
| Active 2,000-example target | PASS | 3,121 unique eligible IDs; 1,002 NCSLGR2 and 2,119 ASLLRP3, separate label pools |
| Additional authentic timed data | PASS | 96/98 deferred NCSLGR utterances; 2 overlapping intervals excluded without edits; two exact publisher EAF/media joins |
| Evaluation partition | PASS, scoped | 1,706 TRAIN / 413 DEV; three publisher TRAIN signers / one held-out signer; 39 / 8 source recordings; no ID/signer/source/raw/pixel hash intersection |
| Fresh CUDA pilot execution | PASS | Five completed epochs, saved/reloaded checkpoints, optimizer and RNG; RTX 3050; actual CUDA tensors and gradients |
| Recognition quality | PARTIAL | Selected epoch 4 TRAIN 97.6015% / DEV 97.9913% WER; learning gate failed, run stopped |
| RIT timing admission | BLOCKED | 278 exact annotation/media joins, but every container has two extra frames and native 60 fps; no authoritative crop/frame-origin adapter; zero eligible |
| Hard CTC/token mapping defect | No evidence found | Prior 50 checks reused; both controlled eight-TRAIN-sample probes memorize; this does not prove production correctness |
| Dominant cause of poor output | PARTIAL hypothesis | 98.40% blank frames and 280/283 emitted tokens IX-1p; frequent-token collapse/underfitting observed; precise optimization/data/feature cause unproven |
| Frozen artifacts | PASS | Both existing read-only validators passed initially and at closure; protected input hashes and 20,745 metadata records checked |
| Streaming false activation, boundaries, latency | BLOCKED for product claims | No new consented no-sign/streaming evaluation recordings; these utterance WER results cannot establish streaming behavior |
| Production readiness | BLOCKED | No checkpoint promoted; Android, thresholds, APK, GLB and reserved 17 TEST untouched |

The fit uses only the ASLLRP3 literal occurrence pool. The 3,121 count is an eligible catalog, **not** a merged training manifest. New 96 NCSLGR examples were not added to the already frozen pilot. Do not normalize away publisher label distinctions to improve scores.

All 413 new DEV utterances had historical TRAIN exposure in prior models. The new model was therefore initialized randomly, with new TRAIN-only vocabulary and normalization and no prior ASL recognition weights. This is a research development partition, not a pristine TEST or a multi-signer accuracy estimate. Its WER is not numerically comparable to the old 85.96% baseline on nine different NCSLGR DEV utterances.

`best_checkpoint_dev.json` retains the reused metric helper's generic `SIGNER UNVERIFIED` scope string. The actual new partition's publisher signer verification is documented in `signer_metadata.json`, `dataset_plan.json` and `EVALUATION_SPLIT_AUDIT.csv`; the saved prediction file was preserved rather than edited.

Acquisition accounting: 503,666,520 new receipt bytes; 4,556,263,744 including recorded prior payload; cap 107,374,182,400 bytes. Extracted archive members are not counted as a second network acquisition. Disk free at closure is recorded in `outcomes.json`.

Read `PILOT_RESULTS.md`, `LABEL_COMPATIBILITY.md`, `DATA_PROGRESS.csv`, `EVALUATION_SPLIT_AUDIT.csv`, `prediction_audit.csv` and `commands_and_results.txt`. The checksum index covers closed new artifacts; its own checksum and itself are not recursively included. The read-only verifier checks the index and frozen parent evidence without training, acquisition or TEST inference.

Next action: verify this package using the command below. No automatic continuation of the failed pilot is scheduled. If further model research is requested, the smallest useful test is a predeclared intermediate-size **TRAIN-only** learnability probe (for example 64 diverse utterances), preserving literal labels and changing one optimization variable at a time; require meaningful TRAIN learning before another generalization run. Eight-sample memorization alone is insufficient. No more data is needed to meet the active goal.

```powershell
& 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T104054Z_expand_eval01\Verify-Continuation.ps1'
```
