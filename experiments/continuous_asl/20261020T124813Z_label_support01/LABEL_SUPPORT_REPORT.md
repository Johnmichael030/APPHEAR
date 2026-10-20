# Frozen label-support audit

**PASS bounded audit and scoped candidate validation. PARTIAL recognition. BLOCKED production promotion.** No new data acquisition or TEST tuning. The3,121eligible catalog, separate target pools, all prior manifests/exclusions/checkpoints/results and production assets remain unchanged.

The existing diagnostic verifier was located and run first; it also ran the preceding continuation verifier. Both passed, including46diagnostic artifacts,1,061parent artifacts and20,745protected metadata records. Existing feature/CTC/normalization/decoder checks were reused; no completed fit was repeated. New candidate feature files were hash/schema/timestamp/CTC checked before the one new fit.

## Distinct-utterance support

A label contributes once to support per TRAIN utterance, even when repeated in that utterance. Token occurrences are counted separately. Exact per-label counts, DEV-only labels, predicted occurrence counts, reference matches/substitutions/deletions and effective model TRAIN support are in TOKEN_SUPPORT.csv. No label normalization, ontology merge or partial target deletion occurred.

|Pool|TRAIN scope count|Token occurrences|Labels|Support1|Support2–4|Support5–9|Support10–19|Support20+|
|---|---|---|---|---|---|---|---|---|
|ASLLRP3|1706|14217|2517|1292 (51.33%)|678 (26.94%)|281 (11.16%)|144 (5.72%)|122 (4.85%)|
|NCSLGR2|1002|5549|1062|596 (56.12%)|286 (26.93%)|90 (8.47%)|44 (4.14%)|46 (4.33%)|

ASLLRP3 scope is the frozen1,706TRAIN split, with413same-pool DEV records, three/one publisher signers and39/8sources. TRAIN covers88.97% of DEV reference tokens, but only194/413utterances (46.97%) have every label supported. There are222DEV-only labels. Latest64fit effective support is much narrower:1,399/3,236DEVtokens (43.23%) are unseen in its64TRAIN examples. A full2519-class output vocabulary does not provide visual training for labels absent from those64targets.

NCSLGR2 scope is1,002eligible TRAIN-pool utterances;906were actually used in its historical reference fit. New eligibility must not be confused with completed training. Token and distinct support for the historical906fit are separately identified in TOKEN_SUPPORT.csv. Its oldnineDEV references have98.25% token coverage and8/9whole-utterance coverage under available1,002TRAIN labels, but incomplete signer metadata (988TRAINrecords unknown; two explicit signer IDs across14records), mixed source-group granularity (688keys, not688independent recordings) and tinyDEV make it less defensible for a new generalization pilot.

Main-gloss NCSLGR2 and literal dominant realized-occurrence ASLLRP3 are distinct supervision contracts. Sharing378features or some label spellings does not establish target equivalence. ASLLRP frame/event joins are verified; NCSLGR legacy timing retains its uncalibrated limits. Sequence/feature-length percentiles, minimumCTCslack, source/signer intersections and token-frequency top30s are in pool_details.json. No CTC-feasibility violations were found in reused full evidence or the new fit's checks. Signer-independent product or boundary accuracy remains unproven.

## Why12.22%TRAIN and97.47%DEV?

The old64fit's TRAIN CTC fell to0.3575 while DEV CTC rose to11.8044; it memorized343labels, not a generalizing recognizer. The saved alignment demonstrates errors even on supported labels:

|Effective64TRAIN support|DEV reference tokens|Correct|Reference-token error rate|
|---|---|---|---|
|Unseen|1399|0|100%|
|1utterance|609|3|99.51%|
|2–4|648|37|94.29%|
|5–9|305|8|97.38%|
|10–19|178|23|87.08%|
|20+|97|13|86.60%|

These are deterministic reference-token alignment errors, excluding insertions; complete WER retains insertions. They are associations with support, not causal estimates. Frequent-label errors are still severe, so rarity is **not** a sufficient explanation. The earlier full1706model errs on96.15% of1,687DEV tokens with20+TRAIN-utterance support, reinforcing that conclusion.

Saved64-model WER by whole-sequence support:

|Support universe|DEV group|Utterances|WER|
|---|---|---|---|
|64_effective_TRAIN|all_labels_seen|19|101.299%|
|64_effective_TRAIN|has_unseen_label|394|97.373%|
|1706_parent_TRAIN|all_labels_seen|194|97.430%|
|1706_parent_TRAIN|has_unseen_label|219|97.489%|

Best-supported explanation (moderate confidence): small effective training data and sparse fine-grained labels permit TRAIN memorization while visual/source/signer generalization remains poor. The singleDEV signer is genuinely unseen and sources are disjoint; source and signer shifts are confounded, and one held-out signer cannot characterize population performance. Cross-pool conventions explain why old85.96%baseline is not equivalent, but do **not** explain the within-ASLLRP64fit gap, since TRAIN/DEV share a contract. Prior audits found no mapping, blank-ID, decoder or normalization defect; genuine TRAIN sequences argue against a hard CTC inability to learn. Weak feature information, model capacity and optimization remain hypotheses, not ruled out.

## Frozen whole-sequence candidates

Thresholds5and10 and coverage gates were written before the candidate fit. Threshold means support in the original TRAIN universe. Keep a complete utterance only when every literal label clears it. Recompute support afterward; never claim threshold holds in retained data. All excluded IDs and below-threshold labels are in EXCLUDED_UTTERANCES.csv; original manifests were not edited.

|Candidate|Retained TRAIN|Actual labels|Excluded TRAIN|OriginalDEVtoken coverage|CompatibleDEV|Recursive closed-core TRAIN|
|---|---|---|---|---|---|---|
|ASLLRP3_support5|497|423|1209|72.96%|92|0|
|ASLLRP3_support10|199|175|1507|57.35%|31|0|
|NCSLGR2_support5|398|145|604|89.47%|5|258|
|NCSLGR2_support10|263|73|739|78.95%|3|0|

Selected≥5ASLLRP:497TRAIN/423labels/2,665tokens, three signers,37sources;92compatibleDEV records across all8DEVsources;22.28% of originalDEV utterances. The≥10alternative is199TRAIN/175labels/31DEV and loses one TRAIN signer's representation sharply. The≥5choice follows predeclared coverage gates, not prediction scores.

Important: after whole-sequence filtering,≥5candidate has90singleton labels and279labels below5retained examples. Recursively demanding≥5or≥10inside retained ASLLRP removes **all** utterances. The proposed experiment is a reduced-support **diagnostic**, not proof of an adequately supported closed vocabulary. NCSLGR's≥5recursivecore retains258utterances, but its tiny/incomplete signer DEV contract prevents choosing it over ASLLRP.

The frozen manifest contains497newTRAIN plus the original413DEV; compatible_dev.json is a separate evaluation view. OriginalDEV remains intact, including unsupported labels. New425class vocabulary and normalization are fit only on497TRAIN; no older recognition weights are reused. Unsupported originalDEVtargets map toUNK forCTCloss only; original literal references remain in WER. Restricted92DEV chooses the local checkpoint by(WER,loss), and original413DEV is reported alongside. Neither restricted-task score replaces a full-task baseline.

The completed pilot's compatibleDEV has **zero unsupported reference labels**, yet WER remains98.390%. Thus unseen labels cannot be the sole cause of failure. The final retained support is still uneven, so this is not proof that label sparsity is irrelevant. Rising DEV CTC with falling TRAIN CTC supports overfitting/source-or-signer transfer failure; those effects remain confounded. PILOT_COMPATIBLE_DEV_ERRORS.csv attributes new errors by retained distinct support. feature_availability_summary.json reports existing hand/body availability separately by split; these numerical checks do not constitute visual quality or linguistic alignment adjudication.

Manifest SHA256 `a735d19f84338a36146c11aeba7923cda6204c41153f617251ace9c1a5daafdf`. The one justified pilot was executed: see PILOT_RESULTS.md. It stopped `NO_MEANINGFUL_LEARNING_GATE20` after20epochs. Selected TRAIN WER43.677%, compatible92DEV98.390%, original413DEV98.053%. No promotion.
