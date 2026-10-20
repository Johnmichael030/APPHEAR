# Bounded pilot decision — executed

**PASS execution; PARTIAL recognition; BLOCKED promotion.** The critical gates passed and the small same-pool CUDA pilot was executed in this session. No second pool was mixed or fitted.

|Epoch|TRAIN WER|DEV WER|TRAIN CTC|DEV CTC|TRAIN blank|DEV blank|TRAIN unique|DEV unique|
|---|---|---|---|---|---|---|---|---|
|1|100.000%|100.000%|16.7802|17.4561|1.0000|1.0000|0|0|
|10|99.357%|100.000%|5.4020|8.3229|0.9966|0.9894|2|1|
|20|87.299%|98.455%|3.6586|9.1015|0.9613|0.9640|57|26|
|30|47.106%|98.362%|1.3660|10.2746|0.8888|0.9489|221|63|
|40|12.219%|97.466%|0.3575|11.8044|0.8358|0.9377|343|85|

Stopped: `NO_MEANINGFUL_DEV_GAIN_GATE40`. Bounds80epochs/90seconds, collapse gate20, meaningful-sequence and DEV improvement gate40; see immutable pilot64/protocol.json written before fitting. Checkpoints include optimizer/RNG/manifest/config; saved checkpoints and examples were reloaded and DEV hypotheses matched.

Measured fit loop 33.119s; with selected-checkpoint reload 37.048s. Wrapper wall 52.849s includes startup/initialDEV. Peak allocated VRAM 1,305,027,072bytes (1.215GiB), reserved 1,361,051,648bytes. Device NVIDIA GeForce RTX 3050 Laptop GPU. All fit tensors CUDA, VRAM/free memory gates checked each batch. Epoch durations and all evaluation details are saved; CUDA CTC nondeterminism warning retained.

Selected DEV-based checkpoint: `D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T123237Z_collapse_diagnosis01\pilot64\checkpoints\epoch_040.pt`

SHA256 `89170961d30b2c413a439bfd9288139d99b9a14b506dd06a738f0c09fad6ca36`.

Selected TRAIN/DEV losses: 0.357516/11.804400; WER 12.2186%/97.4660%. Full prediction lists with literal references, token counts, blank probabilities and length histograms: pilot64/best_train.json and best_dev.json. Per-epoch lists include the final state; no estimates substituted for measurements.

Example selected TRAIN predictions:

- `asllrp3_b8d81ee60519a130`: ref `["IDEA", "REALLY", "MORE", "TWO", "DIFFERENT", "SEPARATE", "ROAD", "SET-UP", "ON", "5\\let's see\\\"\"", "(5)FIRST-IN-LIST", "STORY", "SYSTEM", "HOW", "EXPRESS"]` → hyp `["IDEA", "REALLY", "MORE", "TWO", "DIFFERENT", "SEPARATE", "ROAD", "SET-UP", "ON", "5\\let's see\\\"\"", "(5)FIRST-IN-LIST", "STORY", "HOW", "EXPRESS"]`
- `asllrp3_81716c8cac04f3f5`: ref `["ABOUT", "(2h)alt.SEE", "fs-VISION", "SUPERIOR", "fs-TYRANNY", "fs-OF", "fs-HEGEMONY", "fs-OF", "(2h)alt.SEE", "SUPERIOR", "DISCUSS", "(1h)AREA", "(2h)WOW", "SUPERIOR"]` → hyp `["ABOUT", "(2h)alt.SEE", "fs-VISION", "SUPERIOR", "fs-TYRANNY", "fs-OF", "fs-OF", "(2h)WOW", "SUPERIOR"]`
- `asllrp3_091cd61d054e0934`: ref `["FUTURE", "VISUALIZE", "THAT", "(1h)PERSON", "IX-3p:i", "HIGH", "fs-PITCHED", "CHARACTER", "(2h)TALKwg"]` → hyp `["FUTURE", "VISUALIZE", "THAT", "(1h)PERSON", "IX-3p:i", "HIGH", "fs-PITCHED", "CHARACTER", "(2h)TALKwg"]`
- `asllrp3_f1babebcf7f26127`: ref `["FEEL", "DIE", "fs-ZOMBIE", "SCL:V\\person levitating\\\"\"", "(1h)WORLD", "SCL:V\\person levitating\\\"\"", "5\\hesitation\\\"\"", "TALK", "HEAR", "(1h)CANNOT", "HEAR", "5\\hesitation\\\"\""]` → hyp `["FEEL", "DIE", "fs-ZOMBIE", "SCL:V\\person levitating\\\"\"", "(1h)WORLD", "SCL:V\\person levitating\\\"\"", "HEAR", "(1h)CANNOT", "HEAR", "5\\hesitation\\\"\""]`

Next action: keep this pilot stopped. Reuse the measured TRAIN-learning trajectory to predeclare a single next experiment, rather than adding data or repeating a full fit. A practical next step is a frozen intermediate256TRAIN support audit (no TEST), quantifying rare-gloss and DEV coverage and preserving the same signer-disjoint DEV; launch another bounded fit only if meaningful label support and a specific exposure hypothesis are established. Do not expect64-example memorization to generalize to708DEV labels. Do not merge ontologies, prune difficult DEV records, tune onTEST, or promote weights.

The immediate evidence supports a learnability/coverage investigation, not a decoder threshold sweep or another long GPU run. The3,121eligible count and2,000goal remain unchanged. Production streaming acceptance remains blocked by absent evaluation data.
