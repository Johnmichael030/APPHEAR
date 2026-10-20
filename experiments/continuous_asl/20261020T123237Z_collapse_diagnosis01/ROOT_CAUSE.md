# Continuous-ASL recognition diagnosis

**PASS: bounded diagnosis and CUDA execution. PARTIAL: causal attribution and recognition. BLOCKED: promotion/product claims.** No acquisition, pool merge, production change or TEST tuning occurred. Active data goal remains 2,000, satisfied by the unchanged 3,121 eligible utterances.

## Verified failure mechanism

The selected five-epoch pilot checkpoint (epoch4) was replayed on all 1,706 TRAIN and 413 DEV records. Literal predictions and edit counts exactly matched saved results. DEV WER **97.9913%** has 218 substitutions, 2,953 deletions and no insertions / 3,236 tokens. Argmax blank frequency is **98.3956%**, mean blank probability **84.3033%**. There are 131 empty outputs, 281 one-token outputs and one two-token output. Only three distinct labels appear: 280 `IX-1p`, one `CAR`, two `ns-fs-JOHN`. `IX-1p` is only 178/3,236 reference tokens (5.50%), so 98.94% of emitted labels being this token is excessive concentration, not faithful reference frequency.

TRAIN also underfits: WER97.6015%, 98.3179% blank frames, 436 empty outputs, only11 distinct predicted tokens vs2,517 target labels; 862/1,337 emissions `IX-1p` (64.47%), compared with619/14,217 reference occurrences (4.35%). Only five TRAIN utterances have length3; almost all outputs are length0or1. The poor DEV score is not solely unseen-signer generalization: the large fit has not learned its TRAIN sequences well.

**Most likely root cause (moderate confidence): optimization remains in a blank/frequent-label solution in a sparse literal-gloss problem; unseen-signer and vocabulary-support limitations further constrain DEV.** The direct output mechanism is high-confidence measured evidence. An exact causal division between optimization exposure, model capacity, landmark information and label granularity is not established.

The ASLLRP full pool has2,739 literal labels,1,331 occurring once. New TRAIN has2517 labels, 1246 singletons, over14,217 tokens. Distinctions such as classifiers, spatial suffixes, hand variants, repetitions and whole fingerspelled words are preserved. Lack of samples per label is a measured constraint, not permission to collapse labels. CTC loss reduction alone did not demonstrate recognition.

## Critical engineering checks

| Check | Status | Evidence / limit |
|---|---|---|
| Frozen package | PASS | Requested Verify-Continuation.ps1 passed first:1,061 artifacts,20,745 protected metadata |
| Replay | PASS | Selected checkpoint hash verified; all2,119 new split predictions match saved predictions |
| Mapping | PASS | 2,519 unique vocab entries; blank0,UNK1; exact checkpoint/train-set mapping; no special token in literal targets; zero TRAIN UNK targets |
| Normalization | PASS | Mean/std recomputed from1,706 TRAIN feature arrays exactly match checkpoint buffers; no DEV fit |
| CTC length | PASS | All3,121 feature files hash/schema/finite/timestamp/length checked; same-length causal model has no downsampling; repeats included in minimumCTCsteps |
| Decoder | PASS | Same literal conventions in targets and greedy decoding; blank-separated duplicate test passes; old50 checks reused |
| Temporal contract | PASS scoped / PARTIAL external validity | ASLLRP events inside publisher utterance frame bounds; prior exact frame/count joins reused. No independent visual/linguistic frame audit was invented. NCSLGR legacy uncalibrated timing remains limited |
| Implementation bug | UNKNOWN as an absolute claim | No critical bug found in these targeted checks; no model correction was justified |

Full token distributions, normalization-sensitive labels and NFC collisions are retained in pool_details.json and pool_token_frequencies.csv. No strings were silently normalized or repaired. A shared literal spelling does not establish shared linguistic annotation semantics.

## Pool and evaluation decision

ASLLRP3 is the most defensible current research contract: literal realized dominant occurrences, known publisher IDs for all2,119 records,47 source recordings, four signers. The unchanged same-pool split is1,706TRAIN/413DEV,39/8 sources,three/one signers, with no sample/signer/collection/source/raw/pixel-hash overlap. All newDEV records were historicalTRAIN in older fits; this pilot uses fresh random weights. It remains development-exposed research data, not pristineTEST or a multi-signer reliability estimate.

NCSLGR2 remains separate:1,002 eligible utterances,1,062 labels,5,549 tokens,581singletonlabels; main-gloss rather than realized-occurrence contract. Only14 records have explicit signer IDs in this manifest (two IDs);988 unknown. Its source groups have mixed granularity:688 NCSLGR keys include utterance media and full recordings, so **do not interpret688 as independent source recordings**. It also has21 collection identifiers. POOL_COMPARISON.csv explicitly qualifies that count, with raw mechanical counts preserved separately. Legacy timing limitations and insufficient signer mapping make its current DEV contract less defensible. Publisher-name fields cannot be inferred from faces or filenames.

The two pools share 431 literal spellings, with 3370 in the union. Their target ontologies are not proven equivalent: merging is BLOCKED. Old85.96% baseline uses nine different NCSLGR DEV utterances, different target coverage/vocabulary/training regime. Its WER is not an equivalent comparison with the413-ASLLRP DEV. Reserved17TEST was not read for recognition/tuning.

357/3,236DEV reference tokens (11.03%) are absent from the full TRAIN vocabulary. In DEV CTC loss those targets useUNK; WER correctly keeps literal references. Thus DEV loss is a surrogate for unsupported labels, while literal WER includes unavoidable coverage errors. It cannot explain97.99% by itself. Do not editDEV labels to disguise the gap.

## Completed intermediate learnability pilot

Reused the preexisting64 source-stratified TRAIN monitor, preserving original features,48channels,2519-class vocabulary,1706TRAIN-only normalization,loss,decoder,optimizer and seed. No old recognition weights reused. Frozen64TRAIN/413DEV manifest SHA256 `74316382a1198b7815154d06829c39e06fcba9742dfaa16548450f08f6724c54`. This changes training subset/repeated exposure as a diagnostic; it is not a controlled proof that one hyperparameter causes full-fit failure.

Actual RTX3050CUDA: **40epochs / 320optimizer updates**, stopped `NO_MEANINGFUL_DEV_GAIN_GATE40`. Selected local checkpoint epoch40: TRAIN WER **12.2186%**, DEV **97.4660%**. Final measured epoch40 TRAIN WER **12.2186%**, DEV **97.4660%**, TRAIN token diversity343 / DEV85. The final TRAIN state is diagnostic, not a replacement for DEV-based selection. The selected local pointer is not a production selection.

This intermediate result, together with the earlier eight-sample memorization, tests learnability beyond a tiny fixture. It must not be presented as held-out accuracy. DEV unsupported by64TRAIN: 1399/3236 tokens; full vocabulary includes other1706TRAIN labels but the64fit does not see their target examples. Any DEV failure therefore mixes generalization and limited subset support.

See NEXT_PILOT_DECISION.md for the measured stopping decision. Checkpoint SHA256 `89170961d30b2c413a439bfd9288139d99b9a14b506dd06a738f0c09fad6ca36`. No checkpoint was promoted. Streaming boundaries, no-sign false activation and end-to-end latency remain BLOCKED without consented streaming evidence.
