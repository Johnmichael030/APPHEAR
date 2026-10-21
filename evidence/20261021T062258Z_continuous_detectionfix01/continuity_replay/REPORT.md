# Paired missingness-context replay

Status: **BLOCKED_PAIRED_GATE**. Frozen epoch12; no fitting or app changes.

Both policies discard pending CTC on no hands and publish no token there. A resets feature/model context; B retains genuine missingness-containing cached features. Vocabulary, labels, stable200ms hold,128-token limit and end snapshot are identical. No oracle end flush.

|Partition|Policy|WER|SUB|DEL|INS|Exact sequences|
|---|---|---:|---:|---:|---:|---:|
|TRAIN192|A_reset_nohand|93.9891%|391|982|3|0/192|
|TRAIN192|B_retain_missing_features|93.5109%|117|1252|0|0/192|
|HELD80|A_reset_nohand|95.6701%|176|288|0|0/80|
|HELD80|B_retain_missing_features|97.7320%|65|409|0|0/80|

Gate requires lower B WER and no insertion-count increase in BOTH partitions. Gate details: `{"TRAIN192": {"WER_lower": true, "insertions_nonincrease": true}, "HELD80": {"WER_lower": false, "insertions_nonincrease": true}}`. Numerically equivalent RF61 prefixes pass the frozen tolerance on the longest TRAIN sequence. A exactly reproduces parent TRAIN192 edit/token counts.

Logical serving inference positions are visible observations only; logical model calls on no hands and published no-hand tokens are0 under both policies. Offline vectorized causal inference computes unused no-hand logits as an optimization, not as app behavior or device timing. All272 caches have0 timestamp gaps>1000ms; a gap would reset history/decoder, and unprovided raw rederivation is rejected rather than silently changing cached features.

Interpretation: changes in substitutions/deletions/insertions are descriptive. The80-sequence partition was already development-exposed and the same known signers appear in TRAIN, with guarded recording families held out. This is NOT independent/unseen-signer evidence, no production acceptance and no permission to merge label pools. No negative/no-sign ground truth is available, so false-activation rate cannot be measured here.

Preserved: all manifests, checkpoints, labels, vocabulary, normalization, app assets and reserved TEST payloads. See config.json/input_inventory.json for frozen hashes, PER_SOURCE.csv for source metrics, PREDICTION_EXAMPLES.csv/PREDICTIONS.json for literal hypotheses, PREFIX_PARITY.json and INTEGRITY.json.

Actual measured vectorized inference/scoring duration: 0.510352seconds; no phone latency claim.
