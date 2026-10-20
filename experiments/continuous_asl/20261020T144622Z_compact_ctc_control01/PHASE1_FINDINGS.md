# Phase 1 saved-prediction CTC diagnosis

PASS — no critical numerical, token-mapping, manifest or evaluation inconsistency found in the scoped saved evidence. A compact TRAIN-only learning control is justified; recognition quality is not accepted.

Confirmed facts:
- TRAIN: WER 88.4005%; CTC 3.902538; S/D/I 1526/9444/12; argmax blank 95.5894%; hypothesis/reference tokens 2991/12423 (0.2408); diversity 450; empty 236/1500.
- HELDOUT80: WER 95.6701%; CTC 7.606348; S/D/I 136/328/0; argmax blank 93.1360%; hypothesis/reference tokens 157/485 (0.3237); diversity 49; empty 5/80.

TRAIN vocabulary: 2287 literal glosses plus blank0/UNK1. TRAIN distinct-utterance support bands: {'20+': 109, '5-9': 246, '2-4': 626, '10-19': 124, '1': 1182}. Held-out labels are all represented in TRAIN; unseen labels cannot explain this held-out failure. No pool merge occurred.

The dominant selected error is deletion/short output. This is more precise than a single-token collapse: TRAIN predicts 450 distinct labels, held-out 49. Blank posterior probability is UNKNOWN because only greedy argmax frame counts were saved, not logits or probability arrays.

The full-task trajectory is still learning its TRAIN set within the bound: TRAIN WER falls from 100% at epoch1 to 88.40% at epoch12 and 84.61% at epoch13; TRAIN CTC falls from6.591 to3.728. Held-out CTC worsens from6.627 (epoch1) to7.606 (epoch12) despite a small WER change. These facts suggest underlearning plus poor transfer; they do not establish a single cause or justify unbounded continuation.

Per-label aligned recall and support strata are in diagnosis_by_label.csv and diagnosis_summary.json. TRAIN/held-out rare-label correlations are observational. Source families are disjoint, available signers are known, and family/signer structure is confounded. No unseen-signer conclusion is possible.

All1580 references exactly match resolved literal targets and the saved prediction CSV. The vocabulary equals sorted TRAIN literals after two specials; all targets have IDs>=2. Decoder removes blank0 and collapses repeated IDs before label lookup. No model downsampling; every saved input length meets U+adjacent repeats. Saved normalization/vocabulary/decoder hashes match the prior validated configuration. Prior feature schema/finite/strict-timestamp/hash/annotation checks were reused, not recomputed.

Hypotheses: long-tail support, optimization time/blank attractor, the 48-channel representation/capacity, and missing input observability may interact. A representative compact TRAIN control is the smallest authorized discriminator. It cannot by itself establish transfer or causally separate these explanations. Keep architecture, features, normalization, labels, optimizer and decoder unchanged; keep the corrected accumulation tail. No further fit was executed by this phase.

Preservation: no feature reconstruction, checkpoint/model loading, inference, training, acquisition, or reserved TEST access. Only explicit saved TRAIN/primary-held-out inputs were read. Integrity details are in diagnosis_input_hashes.json.
