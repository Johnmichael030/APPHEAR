# Feature contracts and learning-task decision

Status: CONTRACTS_TRACED; MODEL_NOT_SELECTED.

Tasks are distinct: isolated sign classification, continuous gloss recognition, sign-language translation, temporal segmentation, fingerspelling, and non-sign rejection. This run prioritizes authentic continuous ASL gloss sequences with temporal/rejection supervision where available. English sentence translations may be inspected as metadata, never relabeled as glosses.

| Existing contract | Consumer | Temporal/input semantics |
|---|---|---|
| normalized258 schema3.0 | legacy TCN/runtime | older hand+pose3D/visibility lineage |
| segmentation_features.v1 width378 | current S2 | previous genuine observation, explicit missingness;61-row causal history |
| recognition.v2 width350 | current WORD | maximum96 observations plus boolean mask;236-way probabilities |
| FingerspellingFeatureAdapter.v1 width405 | V2.1 spelling | timestamps, geometry, evidence flags;28 logits including SPACE/BLANK |

How2Sign OpenPose body/face/hand keypoints are not these contracts: joint order, confidence, missingness, coordinate spaces and extraction models differ. They cannot directly replace S2 or Recognition V2. Any adaptation must use a new versioned experimental schema, frozen train-only normalization and parity fixtures.

Candidate starting point, conditional on validated authentic gloss data: a compact causal TCN, CTC gloss head, and separately masked boundary/sign-vs-non-sign heads only when annotation semantics support them. Blank is an alignment symbol, not automatically a calibrated unknown/reject class. Unannotated gaps remain UNKNOWN rather than being silently assigned non-sign.

No model is selected or trained merely by writing this document. Data availability, leakage, CTC feasibility, real CPU/GPU timings and negative-label validity gate that selection. No full RGB backbone, new language mix, fingerspelling merger, production schema migration, or Android promotion is authorized automatically.

