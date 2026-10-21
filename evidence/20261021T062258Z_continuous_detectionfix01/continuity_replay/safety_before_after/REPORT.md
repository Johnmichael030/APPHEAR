# Old delivered no-hand flush versus new discard — fixed80

No fitting or tuning; previously development-exposed known-signer recording-family holdout. Cached observations do not reproduce intervening phone frames or establish live/no-sign false-activation rates.

|Policy|WER|SUB|DEL|INS|No-hand token publications|Exact|Empty outputs|
|---|---:|---:|---:|---:|---:|---:|---:|
|OLD_flush|95.6701%|230|232|2|58|0/80|7|
|NEW_discard|95.6701%|176|288|0|0|0/80|7|

The safety fix is not automatically an accuracy gain. Compare the full edit tradeoff above before describing user-facing recognition quality. Both policies have identical no-hand model-call abstention, weights, vocabulary, reset feature history,200ms settle policy and128-token limit. No oracle end flush was used. The paused-background lifecycle correction is separately tested and does not participate in these uninterrupted cached utterances.

The previous delivered APK model/vocabulary bytes match the frozen package exactly. Old source contains the pending-token flush; new replay uses discard. NEW reproduces the paired context-replay A counts exactly. All80 feature hashes and sealed parent config/results hashes remain unchanged; no reserved TEST payload was read.

Actual CPU vectorized inference/scoring duration: 0.086647s. See config.json, METRICS.csv, PREDICTION_EXAMPLES.csv and INTEGRITY.json.
