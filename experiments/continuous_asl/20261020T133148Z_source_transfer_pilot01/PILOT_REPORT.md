# Controlled known-signer source-transfer pilot

**PASS: frozen split verification, CUDA execution and checkpoint reload. PARTIAL: recognition. BLOCKED: promotion and causal/unseen-signer claims.** Exactly one fresh research fit ran. It stopped `NO_CREDIBLE_PRIMARY_GAIN` after **14 complete epochs** and2632optimizer updates. No acquisition, prior-fit rerun, TEST use, Android/APK/GLB change or checkpoint promotion occurred.

## Frozen design and controls

The authorized fold was copied byte-for-byte; SHA256 `87442a6f1241860e369a5c8f0768d583b3a4058f5a9ad03dcc135532c419c72f`. Source: `D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T131348Z_source_signer_audit01\folds\A_guarded_pooled_family_holdout.json`. InternalTRAIN1,500, full internal heldout206, primary compatible evaluation80, stronger-support subgroup27. The80/27IDs exactly match the frozen membership and recomputed TRAIN-only compatibility/support. No split or subgroup was reconstructed differently. Three TRAIN signers; two evaluation signers already inTRAIN. Thirty-four TRAIN masters and five held-out masters lie in disjoint conservative families (six/two); related undated Ben masters stay together. Dates remain conservative filename-prefix proxies, not independently verified capture timestamps.

The new resolved manifest preserves original globalTRAIN roles and adds an internal role; original manifests are unchanged. Feature hashes,378-float schema, finite values, increasing timestamps and CTC length constraints passed for all1,706used records. All pose/hand feature blocks and the unchanged causal48channel model/greedydecoder are retained. Only the ASLLRP3 literal realized-occurrence pool is used.

Fixed vocabulary:2,287literal TRAIN labels plus blank0/UNK1,2,289classes. All80and27references map exactly; no English translations or new label aliases. Vocabulary and mean/std fit only on1,500TRAIN. Fresh random initialization, seed20261020; no old recognition weights loaded. Initialization optimizer updates0. Initial checkpoint hash `2f0202eb7c24ef5cde79e6d8d4d9f5dc34831fac6091aeb42aa6dc4dc59b6ef2`; vocabulary hash `365b44d435c9def2bba6e0f347155c72bf2ca001bd365d5b10f55dbe91174715`; config hash `75ea1f17e277894195a4f3a1370343e5d579ffd31ffcc40e9f52d3b459aa08ed`; normalization hash `394a3862286b01b884d30952f767f8620969ea21c0384715254e4bddb2d41489`. All controlled-file hashes are in manifest_hashes.sha256.

AdamWlr0.001,weightdecay0.0001, batch4, accumulation2, FP32,gradientclip5; actual CUDA tensors and finite gradients confirmed in cuda_smoke.json without a smoke optimizer update. Fixed128-TRAIN monitor chosen by deterministic source-stratified IDs, entirely outside heldout. Primary80(WER,CTCloss) selects the local checkpoint; subgroup27 and full206 never select it.

Bounds:20epochs or180seconds;175second conservative stop request reserves flush time. Configured evaluations at1,2,4and every two epochs thereafter. Persistent blank/frequent-token collapse and four evaluations without≥0.5percentage-point credible primary improvement stop early. These controls were written before updates. Optimizer/RNG/config are saved with checkpoints; a mid-epoch budget stop, if any, preserves its order and pending gradients separately. No checkpoint is automatically promoted.

## Measured trajectory

|Epoch|Fixed TRAIN monitor WER|Primary80WER|Stronger27WER|PrimaryCTC|Primaryblank|Primarylabel diversity|
|---|---|---|---|---|---|---|
|1|100.000%|100.000%|100.000%|6.6278|1.0000|0|
|2|99.911%|100.000%|100.000%|6.5720|0.9996|1|
|4|98.934%|99.381%|99.324%|6.6169|0.9949|2|
|6|98.401%|97.320%|97.297%|6.8402|0.9747|9|
|8|96.892%|97.732%|97.297%|7.1996|0.9640|17|
|10|94.760%|97.526%|95.946%|7.2729|0.9665|30|
|12|88.277%|96.907%|96.622%|7.5633|0.9232|57|
|14|85.346%|97.113%|97.297%|8.0112|0.9301|48|

Selected checkpoint epoch12:

|Evaluation|Utterances|WER|CTCloss|Blankargmaxfraction|Labeldiversity|Emptyoutputs|
|---|---|---|---|---|---|---|
|train1500|1500|87.0804%|3.8853096224069597|0.953068|507|262|
|monitor|128|88.2771%|3.9109551068395376|0.951124|121|19|
|primary80|80|96.9072%|7.563339900970459|0.923209|57|6|
|strong27|27|96.6216%|6.487293190426296|0.925729|28|2|
|full206|206|97.8818%|N/A: unsupported full206targets|0.931488|101|14|

The27stronger-support sequences remain a subgroup of the80primary sequences, not a replacement. Their predictions are derived from the same full held-out decoding. The full206literal WER is diagnostic coverage context; CTC loss is not aggregated for it because126sequences have unsupported labels. Those labels were not silently mapped into a surrogate score. Loss for all compatible80and27is valid and reported. PREDICTION_EXAMPLES.csv includes all selected1,500TRAIN/206heldout pairs and subgroup flags. Every evaluation checkpoint has saved per-example predictions and blank counts. GROUP_RESULTS.csv reports both family and signer breakdowns; the80view remains74participant3and6participant2, so per-participant2 estimates are descriptive.

## Runtime, GPU and integrity

Measured epoch loop (fit, regular evaluation and checkpoint I/O): **88.502s**. Including selected-checkpoint reload and fullTRAIN validation: **96.487s**. Whole command including preflight/startup/initial evaluation: **114.268s**, exit0. Per-epoch fit and total durations are in METRICS.csv. These are wall-clock measurements, not estimates.

GPU: NVIDIA GeForce RTX 3050 Laptop GPU, total4,294,443,008bytes. Peak framework allocated **1,303,238,656bytes (1.214GiB)**; reserved1,340,080,128bytes. Allocated-memory cap2.5GiB and free-memory256MiB guards checked every microbatch, disk headroom10GiB. Fixed seeds and deterministic algorithms were used in warn-only mode; CUDACTC nondeterminism warning is retained, so bitwise cross-run reproducibility is not claimed. Reloaded selected hypotheses matched saved primary80predictions exactly.

Best local checkpoint: `D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T133148Z_source_transfer_pilot01\checkpoints\epoch_012.pt`

SHA256 `e2a6e0046bca4a3f97f33f023dc7346dd9d6226e19e9bd855357a00c8f1579bf`.

## Interpretation and next decision

TRAIN-learning screen passed: **False**. Primary transfer screen passed: **False**. The predeclared screen requires selected fullTRAIN and monitorWER≤50%,diversity≥10,multi-token output≥25%; primary80WER≤90%,diversity≥10,multi-token hypotheses≥20/80. These are research engineering gates, not production acceptance criteria.

Further training is not justified by this pilot. The TRAIN-learning screen failed, so poor held-out WER is inconclusive as a source-transfer diagnosis. Do not attribute the failure to recording or signer effects. Review existing feature/label alignment and the optimization/representation limitation before another fit; no automatic continuation is scheduled.

The old85.96%nine-NCSLGR baseline and prior413/92ASLLRPDEV experiments are **non-equivalent** in evaluation IDs,vocabulary,normalization-fit population and training membership. Decoder reuse alone does not make them comparable. No cross-run WER improvement claim is made; existing full-task results remain unchanged. See comparison_equivalence.csv. Within this frozen experiment, untrained predictions and the100%all-empty baseline provide transparent context.

This tests new recording families of known signers. It does not isolate a causal signer effect, source-dataset effect or unseen-signer generalization. Source/topic/session/label-frequency effects remain bundled, with only two held-out family clusters and uncertain parent lineage beyond the conservative guard.

Literal primary examples:

- `asllrp3_013cf20f34d13a56`: reference `["IX-1p", "(L)LIVE", "HOUSE", "IX-loc:i", "WITH", "MOTHER+FATHER", "TWO", "BROTHER"]` → hypothesis `["NORMAL"]`.
- `asllrp3_086eb559c9e909a1`: reference `["IX-3p:i", "ns-fs-MARY", "BORN", "BIG_2", "BABY", "(1h)WOW/AWFUL"]` → hypothesis `["IX-1p", "IX-1p", "NONE/NOTHING"]`.
- `asllrp3_09bf85196934962b`: reference `["ns-fs-MARY", "ACCEPT", "MANY", "DIFFERENT+", "CAR"]` → hypothesis `["IX-1p", "NONE/NOTHING"]`.
- `asllrp3_0a66b09f16ef3e3a`: reference `["ns-fs-JOHN", "READ+", "MANY", "BOOK+"]` → hypothesis `["IX-1p", "FINISH"]`.
- `asllrp3_0b4efabf526e5ea0`: reference `["IX-3p:i", "ns-fs-MARY", "ACCEPT", "MANY", "DO-DO+", "part:indef"]` → hypothesis `["IX-1p", "#NO+", "part:indef"]`.
