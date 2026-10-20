# Baselines before training

Existing production accuracy baseline: BLOCKED. Required private held-out captures/features and a compatible end-to-end word benchmark are absent. New utterance gloss CTC and the frozen 236-way isolated WORD task have incompatible labels/objectives; their numbers must not be compared as accuracy improvement. Top-1, macro-F1, per-class precision/recall, full-word confusion, original non-sign false activations and boundary F1 are unavailable.

Packaged S2 engineering baseline: PASS. Model SHA `820023d15c98dab2fb8f21914906ccd637d32bce7b825cbd1d89a58c80cd4ed7`. All 131 frozen debug-fixture observations replay with maximum probability error 0.0 and 0 argmax differences. On 944 new observations, 4-thread Windows TFLite invocation p50/p95 0.399/0.643 ms. Predicted states are not ground truth. Full CameraX/native/WORD/SentenceBuilder replay is not established by this component check.

Tests before training: 56 existing word-schema/fingerspelling tests passed. A broader set had 28 further existing passes, four new model checks passed, and one existing failure due missing `artifacts/recognition_upgrade/perception_vnext/recognition_utility_236_01/phase7_production_cadence_aligned_segmenter_01/best_model.keras`. The packaged TFLite S2 is present and independently tested. No checkpoint was fabricated to make the old test pass.

Small-model control: initial random-weight CTC validation WER 260.0% (insertions can make WER exceed 100%). An always-empty output would have 100% deletion WER on these 25 tokens. Both are diagnostic controls, not production baselines. No trained non-sign baseline exists.
