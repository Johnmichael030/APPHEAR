# AppHear continuous-ASL data and training report

Generated 2026-10-20T21:47:23.211133+08:00 (Asia/Shanghai), from frozen manifests, saved status files, predictions and checkpoints. No training, download, production change or TEST-content export was performed for this report.

## What “continuous words” means here

The dataset targets are **ASL gloss labels**, not a count of ordinary English words or reliably recognized signs. They include lexical signs, classifier descriptions, pointing/locus markers, one/two-hand variants, repetitions and whole fingerspelled words/names. Spelling/case/notation are preserved exactly. Blank and UNK are model-control tokens and are excluded from gloss totals.

There are **3,370 distinct eligible gloss spellings** across both separate pools, and **3,256 distinct spellings actually used as targets in completed continuous-ASL fits**. The display union deduplicates shared spellings; it does not prove semantic target compatibility or authorize mixing pools. The full eligible pool lists sum to3,801labels, of which431spellings occur in both, giving3,370distinct spellings.

## Data totals

|Pool|Eligible utterances|Eligible gloss labels|Unique utterances actually fitted|Fitted gloss labels|Prepared, not fitted|
|---|---|---|---|---|---|
|ASLLRP3|2,119|2,739|2,119|2,739|0|
|NCSLGR2|1,002|1,062|906|920|96|

- Eligible: **3,121unique utterances**, **23,002gloss-token occurrences** counting each utterance once.
- Actually fitted across completed runs: **3,025unique utterances**, **22,586target-token occurrences** counting each utterance once.
- Prepared but not fitted: **96NCSLGR utterances**, containing416token occurrences. They add142NCSLGR-pool labels,114spellings not already present anywhere in the fitted display union.
- Repeated epochs, warm continuations, smoke tests and diagnostic probes reuse data; their presentations are not additional independent examples.
- Actual fitting membership was reconstructed from all saved continuous run manifests and confirmed to equal the2,119ASLLRP3 plus906NCSLGR union. Historical target strings match the current eligible catalog for those IDs.

The current working ASLLRP split is1,706TRAIN/413DEV; all2,119were TRAIN in an older experiment. Thus “ever fitted” includes records now used as development data. New source-transfer evaluation was valid only because its model started fresh. None of these observations establishes pristine unseen-signer TEST accuracy.

## Complete lists and inventory

- ALL_ELIGIBLE_GLOSS_LABELS.txt — all3,370eligible literal spellings.
- ALL_TRAINED_GLOSS_LABELS.txt — all3,256spellings actually fitted.
- ASLLRP3_FULL_GLOSS_LIST.txt —2,739eligible pool labels.
- NCSLGR2_FULL_GLOSS_LIST.txt —1,062eligible pool labels (920used in completed NCSLGR fits).
- LATEST_PILOT_VOCABULARY.txt — exact latest model IDs:2,287TRAIN gloss labels plus blank/UNK,2,289classes.
- GLOSS_LABELS_AND_COUNTS.json — per-label pool membership, token frequencies and distinct-utterance support, separately for eligible and actually fitted data.
- UTTERANCE_DATA_INVENTORY.jsonl — all3,121stable IDs, ordered labels, source/signer metadata, feature hashes, fitted flag and saved run memberships.

The standalone HTML report includes a searchable complete3,370-label inventory. Report list row numbers are not model token IDs. Only LATEST_PILOT_VOCABULARY.txt gives the frozen current model mapping.

## Completed continuous research runs

Each row below is a saved completed main/bounded run. TRAIN count is actual fit membership; a model's output vocabulary can be larger than labels seen by a tiny smoke subset. Full status evidence, hashes, checkpoint paths, runtime and memory are in TRAINING_RUNS.json and TRAINING_STATUS_EVIDENCE.json. Smoke runs are included there separately, with4TRAIN/2DEV as specified by their saved trainer. Isolated-sign ExperimentA status files are retained as out-of-scope evidence and are excluded from continuous totals.

|Run|TRAIN utterances|TRAIN target labels|Selection eval utterances|Epochs completed|Selected eval WER|Stop status|
|---|---|---|---|---|---|---|
|20261019T155428Z_cslr01/logs|30|37|7|14|88.000%|EARLY_STOP|
|20261020_continuous_expand03/models/compact24|100|115|9|22|96.491%|EARLY_STOP|
|20261020_continuous_expand03/models/reference48|100|115|9|14|85.965%|EARLY_STOP|
|20261020_continuous_expand04/models/compact24|181|141|9|27|87.719%|EARLY_STOP|
|20261020_continuous_expand04/models/reference48|181|141|9|16|94.737%|EARLY_STOP|
|20261020_continuous_expand05/models/compact24|455|286|9|12|100.000%|EARLY_STOP|
|20261020_continuous_expand05/models/reference48|455|286|9|20|98.246%|EARLY_STOP|
|20261020_continuous_expand08/models/reference48|2119|2739|9|17|100.000%|EARLY_STOP|
|20261020_continuous_expand09/models/reference48|906|920|9|20|96.491%|WALL_BUDGET|
|20261020T004837_separate02/experiment_B|30|37|7|11|92.000%|EARLY_STOP|
|20261020T102748Z_continuation_pilot01|2119|2739|9|3|100.000%|NO_CREDIBLE_DEV_GAIN_INITIAL_WINDOW|
|20261020T104054Z_expand_eval01/pilot_fresh48_v2|1706|2517|413|5|97.991%|TRAIN_BLANK_OR_NO_LEARNABILITY_GATE|
|20261020T123237Z_collapse_diagnosis01/pilot64|64|372|413|40|97.466%|NO_MEANINGFUL_DEV_GAIN_GATE40|
|20261020T124813Z_label_support01/pilot497|497|423|92|20|98.390%|NO_MEANINGFUL_LEARNING_GATE20|
|20261020T133148Z_source_transfer_pilot01|1500|2287|80|14|96.907%|NO_CREDIBLE_PRIMARY_GAIN|

There are20saved continuous status records, including smoke runs, plus3isolated-sign status records outside these totals. Some exploratory overfit/cardinality probes use separate result files rather than training_status.json; they are preserved in DIAGNOSTIC_PROBES.json and add zero new data. Eight-sample memorization is not generalization.

Earlier experiments have different evaluation IDs, target pools, vocabulary and normalization. Their WER values are **not a comparable leaderboard**. The old85.96%nine-NCSLGR result cannot be ranked against the latest80-source-family result. The 48/24channel experiments only have architecture-comparison scope where their own frozen protocol establishes equivalence.

## Latest authorized source-transfer pilot

Frozen internalTRAIN1,500utterances with2,287gloss labels and12,423target tokens; primary80fully compatible held-out sequences; stronger27as a subgroup; full206heldout reported for coverage context. Only ASLLRP3 is used. Same known signers appear in TRAIN and evaluation; recording families are disjoint under the conservative guard. This is not unseen-signer generalization.

- Completed14epochs /2,632optimizer updates; stoppedNO_CREDIBLE_PRIMARY_GAIN.
- Selected epoch12: TRAINWER87.0804%; primary80WER96.9072%; stronger27WER96.6216%; full206WER97.8818%.
- Epoch loop88.502s; including reload/fullTRAIN validation96.487s. RTX3050Laptop, peak allocated1,303,238,656bytes (1.214GiB).
- Best local checkpoint: `D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T133148Z_source_transfer_pilot01\checkpoints\epoch_012.pt`.
- Checkpoint SHA256: `e2a6e0046bca4a3f97f33f023dc7346dd9d6226e19e9bd855357a00c8f1579bf`.
- Frozen source split SHA256: `87442a6f1241860e369a5c8f0768d583b3a4058f5a9ad03dcc135532c419c72f`.
- TRAIN-learning and transfer screens both failed. Further training is not justified by this run; poor heldout performance cannot establish a source effect while the TRAIN-learning screen fails. No checkpoint was promoted.

## Data contracts and remaining limits

ASLLRP3 uses literal dominant realized-occurrence labels; NCSLGR2 uses literal main-gloss labels. Both use the verified causal378feature schema in the current catalog, but shared feature dimensions or spellings do not make their supervision equivalent. NCSLGR legacy time calibration remains limited. The latest ASLLRP model uses the same causal48channelCTC family, TRAIN-only mean/std and vocabulary, fixed greedy decoding, blank0/UNK1, FP32 and research-only CUDA.

Known publisher signer metadata exists for ASLLRP. NCSLGR has988eligible records without explicit signer IDs; do not infer people from filenames or images. Independent source-dataset and signer effects remain unidentifiable with the current bridge/recording structure. Continuous streaming false-activation, boundary and end-to-end latency acceptance remain unverified; the current report describes utterance-supervised research.

How2Sign has no authenticated gloss supervision in the verified local package and is not counted as fitted gloss data. The RIT archive has a preserved timing/frame-origin blocker and contributes zero eligible training examples. Prior exclusions (six ASLLRP overlaps, frame-count mismatch, NCSLGR overlaps) and17reserved TEST utterances remain protected. TEST labels/predictions were not used to build these lists.

**Status: PASS data/report integrity; PARTIAL recognition; BLOCKED model promotion.** Current lists enumerate data and model targets; they are not a claim that the model correctly recognizes3,370words.
