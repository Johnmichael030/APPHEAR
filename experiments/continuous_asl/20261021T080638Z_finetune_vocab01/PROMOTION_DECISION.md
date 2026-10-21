# Promotion decision

**DO NOT PROMOTE. Overall PARTIAL.** A is a real completed fine-tune, not a validated app-recognition improvement. Full-sequence DEV WER changes95.6701%→94.6392%, but exact sequences remain0/80, the paired descriptive interval includes no improvement, DEV CTC loss worsens, and current streaming WER stays95.6701%. Both recording families remain poorly recognized. B added0classes and C's predeclared gate failed.

The criterion for considering C was frozen before fitting: at least2percentage points of full-sequence DEV gain, no insertion-rate regression over2points, and no TRAIN WER regression over5points. A meets neither a meaningful streaming improvement nor the gain threshold. No thresholds were relaxed. Training stopped after10epochs for lack of gain; continued same-data fitting is not justified by these measurements.

Required independent promotion evidence is absent for new classes, unseen signers, consented no-sign behavior, whole-sign boundaries, adjacent repeats and live phone cadence/latency. No numerical threshold is selected retrospectively from these results. A later trial must predeclare its minimum useful change and independently verify retention, false activations, boundaries and device performance before export/deployment.

Retain the unchanged current app model and lifecycle/missing-hand safeguards. No APK was built/installed, renderer modified, threshold tuned, assets copied, Git operation performed or production checkpoint selected. Selected research artifact only: `A\checkpoints\epoch_002.pt`, SHA256 `a190e96b035e74b374c651a48ca56178c1ce593666938e0825d63368556f51da`.
