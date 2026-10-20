# Feature contracts and learning-task decision

Status: CONTRACTS_TRACED; isolated causal TCN pilot selected and completed; production schema unchanged.

Tasks are distinct: isolated sign classification, continuous gloss recognition, sign-language translation, temporal segmentation, fingerspelling, and non-sign rejection. This run prioritizes authentic continuous ASL gloss sequences with temporal/rejection supervision where available. English sentence translations may be inspected as metadata, never relabeled as glosses.

| Existing contract | Consumer | Temporal/input semantics |
|---|---|---|
| normalized258 schema3.0 | legacy TCN/runtime | older hand+pose3D/visibility lineage |
| segmentation_features.v1 width378 | current S2 | previous genuine observation, explicit missingness;61-row causal history |
| recognition.v2 width350 | current WORD | maximum96 observations plus boolean mask;236-way probabilities |
| FingerspellingFeatureAdapter.v1 width405 | V2.1 spelling | timestamps, geometry, evidence flags;28 logits including SPACE/BLANK |

How2Sign OpenPose body/face/hand keypoints are not these contracts: joint order, confidence, missingness, coordinate spaces and extraction models differ. They cannot directly replace S2 or Recognition V2. Any adaptation must use a new versioned experimental schema, frozen train-only normalization and parity fixtures.

Candidate starting point, conditional on validated authentic gloss data: a compact causal TCN, CTC gloss head, and separately masked boundary/sign-vs-non-sign heads only when annotation semantics support them. Blank is an alignment symbol, not automatically a calibrated unknown/reject class. Unannotated gaps remain UNKNOWN rather than being silently assigned non-sign.

The following pilot was selected after validating authentic gloss sequences and causal-prefix checks. Data availability, leakage, CTC feasibility, real CPU/GPU timings and negative-label validity gate that selection. No full RGB backbone, new language mix, fingerspelling merger, production schema migration, or Android promotion is authorized automatically.


## Frozen experimental contract

`apphear.continuous_asl.candidate01_causal378.v1` reuses derive_causal_segmentation_features: columns [0,84) hand XY; [84,126) joint validity; [126,168) proposal support; [168,170) hand presence; [170,172) presence validity; [172,174) observed; [174,176) proposal; [176,182) ROIs; [182,198) body XY; [198,206) body support; [206,214) body validity; 214 body available; 215 delta-time; [216,300) hand deltas; [300,342) delta validity; [342,348) ROI deltas; [348,350) ROI validity; [350,366) body deltas; [366,374) body delta validity; [374,376) observed transitions; [376,378) proposal transitions. Endpoints are exclusive.

TRAIN-only normalization plus clipping [-10,10], per-observation LayerNorm, 48 channels, RF61, 39 CTC logits. The model is independent of S2 despite sharing raw feature math/width. No duration/progress/future-frame input. Boundary/rejection heads were not trained because labels do not support them. Unit prefix/bounded-history tests and exported real-clip parity pass. All schemas/assets/normalizations remain distinct; feature width alone never authorizes substitution. Current Android consumes none of these new artifacts.
