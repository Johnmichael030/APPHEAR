# TRAIN-only source/signer audit

**PASS audit execution and frozen preservation. PARTIAL recording-transfer design. BLOCKED independent dataset/signer attribution and crossed experiments. No training was run or authorized.**

## Scope and membership

The current3,121eligible catalog is preserved. This inventory has **2,708applicable TRAIN utterances**:1,706ASLLRP3from the frozen current split, plus1,002NCSLGR2. The latter have explicit TRAIN membership in the historical906TRAIN manifest or the96new timed-TRAIN manifest; catalog tags such as TRAIN_POOL_NOT_FIT were not treated as sufficient proof by themselves. The413current ASLLRP DEV records are excluded from identity/fold construction, even though they were historicalTRAIN in older experiments. Reserved17TEST was not used.

SOURCE_SIGNER_COVERAGE.csv contains every audited stable utterance ID, corpus-scoped source/signer ID, literal ordered gloss sequence, token/observation counts, feature path/hash/status, applicable frozen TRAIN membership and annotation/signer provenance. Feature/finite/timestamp/CTC evidence was reused from the prior3,121-file audit; existence and metadata-based CTC feasibility were checked here. No video decoding, appearance-based identity inference, label editing, acquisition or new predictions occurred.

There are no missing source_group strings, but that does not prove complete underlying recording identity. ASLLRP has1,706records with publisher parent-master names and participant joins; NCSLGR has547EAF/full-source joins and455index-media IDs with uncertain underlying recording granularity. NCSLGR has988missing signer records. Its14known records have two corpus-scoped signer IDs (13and1utterances); the numerous remaining unknowns cannot be treated as one person.

## Identity and source structure

|ASLLRP TRAIN signer ID|Utterances|Published recording IDs|
|---|---|---|
|asllrp_dai_participant_2|754|10|
|asllrp_dai_participant_3|617|19|
|asllrp_dai_participant_5|335|10|

Every one of the39ASLLRP recording IDs contains exactly one known signer; all three signers appear in multiple recording IDs. No source contains multiple known signers. Cramér's V(recording,signer)=**1.0**. The recording-indicator design has rank39; adding signer indicators leaves rank39: **zero additional rank**. This proves the fixed recording and signer effects are perfectly aliased in this incidence structure. It does not prove either factor caused the model failure.

Dataset and recording are different levels. The ASLLRP TRAIN data have one source dataset and3known signers; NCSLGR has2explicit IDs and988unknowns. No verified same-person cross-dataset bridge is present. Dataset/label-pool and corpus-scoped identity are confounded; unknown IDs cannot establish a bridge. NCSLGR main-gloss and ASLLRP dominant-realized-occurrence targets remain separate. They share419literal spellings in currentTRAIN, which is insufficient to establish semantic annotation compatibility. Cross-dataset source transfer is **BLOCKED**.

The required source-by-signer table is SOURCE_SIGNER_CROSS_TAB.csv, with corpus and label-pool scope. Additional DATASET_SIGNER_CROSS_TAB.csv, SIGNER_BY_LABEL_POOL.csv and SOURCE_BY_LABEL_POOL.csv distinguish levels. SIGNER_LABEL_OVERLAP.csv describes literal target overlap without claiming semantic or person equivalence.

No multiple-source exact raw-hash aliases or duplicate decoded-pixel utterance hashes were found in this TRAIN inventory. Reuse of a full source hash by distinct utterances within one recording is expected. No identifier collisions across pool namespaces, multi-URL source keys or multi-collection source keys appeared. These checks cannot detect all re-encodings or unknown alternate views. Only current manifest view artifacts are audited; additional published camera views do not become training records automatically. Unresolved NCSLGR parent/view links block its crossed folds.

## Conservative recording-family guard

Distinct master filenames are not proof of independent recording ancestry. The Ben files form a numbered introduction-to-conclusion series; independence from a shared original is not established. Keep all335undated Ben records together. For date-coded masters, keep all masters with the same known signer/date-shaped prefix together. These are **conservative grouping proxies**, not inferred identities, authenticated capture dates or claims that every grouped file is physically one recording.

The guard gives8families:2for participant2,5for participant3,1unresolved Ben family. Preliminary filename-based A folds split these families and are explicitly marked PARTIAL_FILENAME_ONLY_GUARD_FAILED; their assignments are retained as completed audit evidence, not recommended for fitting. Entire masters, collections and conservative families remain together in the guarded folds. No adjacent utterance or known repeated view/content group crosses an assignment. Unknown cross-date re-encodings remain a limitation.

## Candidate folds

|Candidate|Internal TRAIN|Internal eval|All-label-compatible eval|Support≥5eval|Status|
|---|---|---|---|---|---|
|A_pooled_recording_holdout|1571|135|56|14|PARTIAL_FILENAME_ONLY_GUARD_FAILED|
|A_within_signer_2|684|70|13|2|PARTIAL_FILENAME_ONLY_GUARD_FAILED|
|B_signer_2|952|754|156|38|PASS_INTERNAL_GROUP_ASSIGNMENT|
|A_within_signer_3|587|30|14|3|PARTIAL_FILENAME_ONLY_GUARD_FAILED|
|B_signer_3|1089|617|213|66|PASS_INTERNAL_GROUP_ASSIGNMENT|
|A_within_signer_5|300|35|6|0|PARTIAL_FILENAME_ONLY_GUARD_FAILED|
|B_signer_5|1371|335|13|1|PASS_INTERNAL_GROUP_ASSIGNMENT|
|C_crossed|—|—|—|—|BLOCKED_INSUFFICIENT_IDENTIFIABILITY|
|NCS_crossed|—|—|—|—|BLOCKED_INSUFFICIENT_IDENTIFIABILITY|
|A_guarded_pooled_family_holdout|1500|206|80|27|PASS_CONSERVATIVE_GROUP_GUARD; PARTIAL_EXPLORATORY_SUPPORT|
|A_guarded_within_signer_2|722|32|3|0|PASS_CONSERVATIVE_GROUP_GUARD; PARTIAL_EXPLORATORY_SUPPORT|
|A_guarded_within_signer_3|443|174|53|2|PASS_CONSERVATIVE_GROUP_GUARD; PARTIAL_EXPLORATORY_SUPPORT|
|A_guarded_within_signer_5|—|—|—|—|BLOCKED_SINGLE_UNRESOLVED_PARENT_FAMILY|

Every actual fold has a separate folds/*.json membership manifest with a parent-manifest hash. All members come from currentASLLRP TRAIN. Full internal evaluation is retained; compatible/support5views include entire reference sequences only. EVAL_VIEW_EXCLUSIONS.csv records every auxiliary-view exclusion and its literal labels; no reference tokens were removed from full evaluation. CANDIDATE_FOLDS.csv reports signers/sources, vocabulary, token coverage, CTC slack, exact-content leakage checks and manifest hashes. LABEL_COVERAGE_BY_FOLD.csv reports distinct TRAIN support bins and token frequencies for both sides.

The guarded pooled fold has1,500TRAIN from34published masters/6families and206internal-eval utterances from5masters/2families. All evaluation signers occur inTRAIN; TRAIN includes three signers, eval two. TRAIN has2,287labels; eval607, with377shared. Full eval has1,794reference tokens; token coverage81.05%. Its80compatible utterances cover only27.03% of full-eval tokens and are imbalanced:74participant3,6participant2. The stronger-support view has27utterances. These restricted views control unknown/low-support labels partially; they do not match topic, sequence difficulty or frequency distributions. There are only two eval family clusters, so broad confidence claims are not supported.

A direct guarded split of the latest497TRAIN task would leave446TRAIN/51eval, but only24compatible records (23participant3,1participant2) and2support5records. That is too small for a multi-signer transfer screen. The larger currentTRAIN design is the better exploratory candidate; it still needs a TRAIN-learning gate before transfer WER is interpreted.

Leave-one-signer-out B folds also hold out all that person's recordings; they test **joint signer+recording transfer**, not an independent signer effect. A crossed C fold is impossible with these observed cells and was not fabricated. Ben has no safe within-person family holdout under the conservative guard. NCSLGR missing identities/recording lineage/target compatibility prevent an alternative crossed design.

## Existing model evidence, without new evaluation

Partition statistics reused from saved predictions and logs:

|Model|Partition|Utterances|WER|Empty outputs|Predicted labels|Blank argmax fraction|
|---|---|---|---|---|---|---|
|prior1706|TRAIN|1706|97.601%|436|11|0.9832|
|prior1706|FROZEN_DEV_REPORT_ONLY|413|97.991%|131|3|0.9840|
|prior64|TRAIN|64|12.219%|0|343|0.8358|
|prior64|FROZEN_DEV_REPORT_ONLY|413|97.466%|0|85|0.9377|
|prior497|TRAIN|497|43.677%|0|362|0.8651|
|prior497|FROZEN_DEV_REPORT_ONLY|413|98.053%|57|78|0.9598|

The latest497model's observed per-signer summaries:

|Model|Partition|Signer ID|Utterances|WER|Unseen reference tokens in actual fit|Dominant emission share|
|---|---|---|---|---|---|---|
|prior497|TRAIN|asllrp_dai_participant_2|262|48.846%|0|FRIEND (10.4%)|
|prior497|TRAIN|asllrp_dai_participant_3|199|32.837%|0|IX-1p (7.9%)|
|prior497|TRAIN|asllrp_dai_participant_5|36|62.753%|0|\WHAT\"" (5.9%)|
|prior497|FROZEN_DEV_REPORT_ONLY|asllrp_dai_participant_4|413|98.053%|875|IX-1p (29.0%)|

SAVED_MODEL_GROUP_ERRORS.csv includes per-recording and per-signer errors, output lengths/empties, label diversity, dominant-emission share and effective-fit support. It reports previously computed TRAIN resubstitution and frozenDEV separately. Per-group blank fractions are UNKNOWN because per-utterance frame logits were not saved; empty-output rates are not substituted for blank probability. Recoverable global1706TRAIN blank statistics were reused from its completed replay.

TRAIN scores show underfitting/memorization behavior, not source transfer. FrozenDEV uses one held-out signer and different recordings, so its high WER does not distinguish the factors. The compatible92DEV failed with zero unsupported labels in the last task, which excludes missing vocabulary as a sufficient explanation. Sparse support, source/recording/topic effects, signer variation and visual feature/model limitations remain competing explanations. Metadata correlations and current WER do not establish causality.

All old recognition models trained on some or all of the proposed internal holdout records. **Do not replay or warm-start those models and call it unseen-source performance.** A later source pilot needs a fresh recognition model/optimizer or separately proven nonleaking initialization, TRAIN-only normalization/vocabulary and the new frozen membership.

## Decision

Recommend one **source-recording-family transfer pilot**, exploratory and conditional on later explicit training authorization; see NEXT_EXPERIMENT_DECISION.md. It can ask whether failure persists with known signer identities and unseen families. Independent source-dataset/signer causal effects and a crossed comparison remain BLOCKED. No fit, checkpoint selection or promotion occurred here.
