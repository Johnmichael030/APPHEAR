# Continuous ASL fine tuning and vocabulary expansion

Date: 2026-10-21. Overall **PARTIAL**. The original model was successfully fine-tuned, but the candidate does not pass the recognition or promotion gates. No new vocabulary class was admitted or learned. Production and the reserved TEST set remain unchanged.

| Protocol on identical development records | Frozen epoch 12 | Selected A epoch 2 |
|---|---:|---:|
| Full-sequence greedy WER | 95.6701% | 94.6392% |
| Substitutions / deletions / insertions | 136 / 328 / 0 | 132 / 327 / 0 |
| Exact sequences | 0/80 | 0/80 |
| Mean sequence CTC loss | 7.606348 | 7.718292 |
| Argmax blank fraction | 93.1360% | 93.1789% |
| Current streaming replay WER | 95.6701% | 95.6701% |
| Streaming substitutions / deletions / insertions | 176 / 288 / 0 | 185 / 279 / 0 |
| Full TRAIN greedy WER | 88.4005% (reused) | 85.6959% |

The full-sequence candidate gains five correct aligned tokens, from 21 to 26 of 485. Nine utterances improve, four worsen, and 67 have the same edit count. The paired utterance-bootstrap descriptive interval for candidate-minus-baseline WER is [-2.510, 0.411] percentage points. It includes zero and ignores correlations among utterances in only two recording families. This is development-exposed, known-signer evidence, not an independent test or unseen-signer claim. The unchanged aggregate streaming score conceals changed predictions and a substitution/deletion tradeoff.

## Actual execution

A restored epoch-12 model weights and AdamW moments, changed LR from 0.001 to 0.0001, and kept architecture, normalization, vocabulary, frozen 1,500 TRAIN / 80 DEV split, features and greedy metric fixed. Batch 4, accumulation 2, FP32, seed 20261020. A disposable one-update smoke was saved and then discarded by reloading the source exactly. The main fit completed 10 epochs / 1,880 optimizer updates and stopped under its predeclared lack-of-gain rule, within the 12-epoch / 180-second bound.

Measured training plus epoch-evaluation wall time: 82.106 s. Sum of fit-loop time: 72.771 s. Epoch fit times: 6.975–8.874 s. Selected-checkpoint full TRAIN and replay scoring occurred afterward and is excluded from those training times. Peak PyTorch allocated VRAM: 1.214 GiB; reserved: 1.248 GiB. Sampled device-reported peak: 1427 MiB. WDDM sampling and allocator measurements are separate quantities. See A/METRICS.csv for measured frame throughput and A/gpu_memory.csv for samples.

Selected research checkpoint: `A/checkpoints/epoch_002.pt`, SHA256 `a190e96b035e74b374c651a48ca56178c1ce593666938e0825d63368556f51da`. It is source epoch 12 plus two fine-tuning epochs, not the final epoch-10 checkpoint. Checkpoints include optimizer, RNG and configuration identity. The A driver supports last-completed-epoch recovery and refuses to repeat a completed run. Resume-monitor output would restart its GPU CSV; no resume was required here. A new comparison must use a new directory.

## Recognition diagnosis

**PASS computational contract:** all 1,580 permitted feature files retain their hashes, embedded labels, schema, strictly increasing timestamps and CTC-feasible lengths. Blank 0 / UNK 1, all 2,289 class IDs and frozen mean/std remain correct. No current unknown-label mapping or CTC infeasibility explains these results.

**PARTIAL recognition:** the model emits 158 hypothesis tokens for 485 references; 93.18% of raw argmax frames are blank. It emits 49 distinct labels, with IX-1p accounting for 51/158 tokens. This is severe deletion and limited diversity, not complete empty-output or single-token collapse. Continued fitting lowers TRAIN loss while DEV CTC loss increases. TRAIN WER remains 85.70%, so this full task is also poorly fitted; the explanation cannot be pure overfitting alone.

Sparse supervision matters: 1,182/2,287 labels have one distinct TRAIN utterance and 1,808 have fewer than five. However, frequent-label recall is also only 25/251 for support >=20, while all other buckets together achieve 1/234. Rare labels are not the sole failure. The clean literal pool is kept separate from NCSLGR main glosses; adding names or combining feature-compatible pools would not repair this.

Prior TRAIN-only observability evidence found 880/12,423 published occurrences with no retained sample, plus 1,440 with no observed hand inside the retained interval. This task's DEV publisher intervals have 19/485 with zero samples and 100/485 with one. The publisher spans are not qualified physical whole-sign boundaries. Surviving body/context information, unresolved visual annotation quality, source distribution and model capacity/optimization can all contribute; no single causal root cause has been established.

Cached-logit beam tests use no English prior. Baseline beam8 reaches 94.6392% WER; A beam8 reaches 94.8454%, worse than A greedy. Thus beam search is not a consistent remedy. Baseline logits were saved before A; the beam probe itself ran after A, rather than before fitting as requested. This ordering limitation is recorded explicitly. No production decoder or confidence threshold was changed.

## Vocabulary and detection

463 distinct proposed new labels are inventoried, including local HELLO and 462 RIT literals; **zero have passed all data gates**. This is a proposal, not 463 learned words. HELLO has 60 potential isolated recordings from one signer/session, incompatible schema 3.0 and no established consent/review/independent validation. The user supplied the dataset root as the records location; the renewed recursive search found no completed consent/review record. All 480 reviewer_notes fields are empty. The thesis describes future consent and additional written permission, rather than supplying executed records. Participant media were not decoded or processed.

The append-only head, old logits/IDs, checkpoint serialization, new-row CTC gradient and expanded AdamW step pass synthetic engineering tests. No synthetic labels or model were distributed as learned signs. B remains blocked and its future real-data training path is not end-to-end validated. C was not run: A's 1.0309 pp improvement fails the predeclared 2 pp promising screen and streaming WER is unchanged.

No improvement in physical boundary timing, adjacent repeats, genuine no-sign false activations or phone camera latency can be claimed. DEV contains no adjacent repeated-label sequences, no consented no-sign cohort, and no independent boundary review. Host CPU synthetic forward timings are in HOST_INFERENCE_LATENCY.csv and are not phone/end-to-end measurements. No app build/install or asset update occurred. Tensor payload is 750,228 bytes, selected resumable checkpoint 2,349,195 bytes; APK asset impact is zero.

## Decision and next action

**Do not promote or continue the same-data fit.** Preserve the existing app model. Obtain the actual consent/permission records, qualified annotation provenance and an independent new-class recording session before admitting local HELLO. For continuous expansion, resolve the RIT publisher crop/time origin without guessing or trimming frames. Review the already selected TRAIN observability events with a qualified ASL reviewer, then design a single justified feature/cadence or label-support experiment. More unique classes alone is not the current remedy.

Run `Verify-Completed.ps1` or the direct Python verifier from commands_and_results.txt for read-only integrity checks. `train_expansion_B.py` without --train performs admission checks only and currently exits 2. It must remain blocked until real records satisfy every gate. No reserved TEST evaluation is authorized by this result.
