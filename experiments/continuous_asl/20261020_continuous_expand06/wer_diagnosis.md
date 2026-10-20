# Measured WER regression diagnosis

| Model | Own TRAIN WER | Shared100 TRAIN WER | Fixed9DEV WER | DEV blank frames |
|---|---:|---:|---:|---:|
| 100_reference48 | 90.69% | 90.69% | 85.96% | 95.58% |
| 181_reference48 | 83.39% | 82.79% | 94.74% | 97.91% |
| 181_compact24 | 79.74% | 77.53% | 87.72% | 97.91% |
| 455_reference48 | 80.98% | 87.85% | 98.25% | 94.65% |
| 455_compact24 | 99.87% | 100.00% | 100.00% | 99.77% |

All551 independent ELAN joins agree with existing target event units. No target normalization correction is justified. DEV OOV improves9→7→3, yet recognition regresses. TRAIN/DEV WER and blank fractions above quantify generalization and deletion behavior. They do not uniquely identify an optimizer or capacity cause.

The next predeclared test keeps the same causal48-channel model, baseline normalization and exact mapped head rows, and adapts at1e-4. New word rows start with bias-10. It must reproduce the baseline DEV predictions before fitting. Selection remains fixed9DEV WER then loss; no test tuning.

Signer independence, false activations, missed-sign events, visual boundary accuracy, and live Android latency remain unmeasured without independently verified annotations and consented no-sign recordings.
