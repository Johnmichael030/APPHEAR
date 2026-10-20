# Measured fresh research pilot

**Execution PASS; recognition PARTIAL; no promotion.** Five CUDA epochs completed. The predeclared learning gate stopped the run; it was not extended after failure.

Data: frozen 1,706 TRAIN / 413 DEV manifest SHA256 `04fb7c6c26bc934d6618902ceeaa2ee69e19d0c12e5bfe922e5743c8518c0ebe`. Single target pool; three TRAIN publisher signers, one DEV signer. 2,517 literal TRAIN glosses plus blank ID 0 and UNK ID 1. 357/3,236 DEV tokens are out of vocabulary (11.03% irreducible exact-label coverage limitation with this fixed vocabulary). DEV reference labels remain literal, not replaced with UNK for scoring.

Architecture/config: causal 48-channel CTC, receptive field 61, 378-dimensional features; fresh random initialization; FP32, AdamW lr 0.001, weight decay 0.0001, batch 4, accumulation 2, gradient clip 5, seed 20261020. New TRAIN-only normalization and vocabulary. Bounds: 12 epochs / 180 seconds with a five-epoch learning window and saved optimizer/RNG state. See the exact gate in `pilot_fresh48_v2/config.json`.

| Epoch | Optimization loss | Fixed 64-TRAIN monitor WER | DEV WER | Epoch seconds | DEV emitted tokens |
|---|---|---|---|---|---|
| 1 | 8.2240 | 100.0000% | 100.0000% | 12.815 | 0 |
| 2 | 6.5980 | 99.6785% | 99.7219% | 11.197 | 22 |
| 3 | 6.3201 | 98.3923% | 98.5476% | 11.092 | 178 |
| 4 | 6.0841 | 98.5531% | 97.9913% | 12.101 | 283 |
| 5 | 5.8843 | 98.0707% | 98.4240% | 11.287 | 214 |

Selected epoch 4: full TRAIN WER **97.6015%**, loss **5.821992**; DEV WER **97.9913%**, loss **6.801273**; 218 substitutions, 2,953 deletions, 0 insertions / 3,236 reference tokens. 131/413 hypotheses empty; 98.3956% argmax frames blank. Emitted token counts: `{'IX-1p': 280, 'CAR': 1, 'ns-fs-JOHN': 2}`. This is frequent-token collapse with limited learning, not working general recognition. Epoch 5 full TRAIN WER is recorded separately; it is not the selected epoch's TRAIN score.

Measured runtime: epoch loop 58.500s; fit/reload/full-TRAIN diagnostics 80.841s; successful wrapper wall 100.971s including startup and logging. 8,530 TRAIN presentations; epoch-loop throughput 145.81 utterance presentations/s (includes DEV/monitor evaluation, not pure forward throughput). Per-epoch durations above are measured, not estimates.

RTX 3050 Laptop 4 GiB: framework peak allocated 1,303,643,648 bytes (1.214 GiB), reserved 1,361,051,648 bytes. Sampled whole-device memory peak 1447 MiB, from 200 ms `nvidia-smi` logs; sampling can miss peaks. Unknown utilization values remain in the raw CSV. Actual input/logits/loss CUDA device and finite-gradient smoke are saved. Checkpoint reload reproduced selected DEV predictions.

Checkpoint: `D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T104054Z_expand_eval01\pilot_fresh48_v2\checkpoints\epoch_004.pt`

SHA256: `3d66372e84f94bc42b0bdaa61702013601d3b7b1ac160f76a1e5a9bf147c25de`

Trainer SHA256: `2da43ed0fca5542cf3e4259aac97718d70e0e2854a13ebdb9e382cd78f030d62`.

The first new driver failed during smoke logging after `zero_grad(set_to_none=True)`, before any optimizer update. V2 captures gradient device before clearing it. Failure and original script are retained in a different output directory; no completed fit was repeated or overwritten. This is a diagnostic logging fix, not evidence of a prior model bug.

Prediction examples, publisher literal conventions:

- `asllrp3_2cf7b65be141761c`: reference `["IX-1p", "(2h)NOT-YET", "(P)PAY", "fs-RENT"]`; hypothesis `["IX-1p"]`.
- `asllrp3_b63ce006168d3721`: reference `["(1h)GOOD/THANK-YOU", "WEATHER", "IX-loc:i", "ARRIVE", "ns-BOSTON", "(2h)NOT-YET", "HURRY", "part:indef"]`; hypothesis `["IX-1p"]`.
- `asllrp3_47786393c384a039`: reference `["POSS-1p", "FAMILY", "IX-3p:i", "EVERY-YEAR", "VACATION+", "part:indef", "(2h)#NO+MORE", "(1h)part:indef"]`; hypothesis `["IX-1p"]`.
- `asllrp3_91bcefffdcfcd1e9`: reference `["ns-fs-BOB", "IX-loc:i", "HAVE", "BIG", "#DOG_2", "#NO", "IX-3p:i", "(1h)HAVE", "SMALL"]`; hypothesis `["IX-1p"]`.
- `asllrp3_8e3dc299ba22f98a`: reference `["ns-fs-BOB", "IX-loc:i", "(2h)NOT-YET", "GO-OUT", "#PO", "(1h)part:indef"]`; hypothesis `["IX-1p"]`.
- `asllrp3_4ff45ebbb9c63b38`: reference `["FARM", "SCL:bent-B\\flat land\\\"\"", "(2h)HORSE", "SCL:crvd-V\\horse galloping\\\"\"", "(1h)VOMIT/HATE", "IN", "RAIN"]`; hypothesis `["IX-1p"]`.

All 413 reference/hypothesis pairs are in `prediction_audit.csv` and the selected DEV JSON. The generic inherited `SIGNER UNVERIFIED` metric scope string is superseded for partition provenance by the new publisher-backed split audit; it does not change measured predictions.

## Controlled TRAIN-only cardinality probe

Eight fixed new-TRAIN utterances from three TRAIN signers and eight source groups; same features, literal targets, TRAIN normalization, matching initialized backbone/common output rows. Only output vocabulary cardinality changed: 2,519 versus 62 classes. Both fresh fits memorized: full vocabulary 2.7778% TRAIN WER after 100 updates / 3.040s; restricted vocabulary 1.3889% after 90 updates / 2.458s. Saved predictions/checkpoints/curves verify these figures. No DEV or TEST, no new examples, no generalization claim. CUDA CTC nondeterminism and CuBLAS workspace warnings are retained; these small results are not a deterministic causal estimate for the full fit.

This argues against an absolute CTC/token-mapping inability to learn. It does not establish the cause of full-data underfitting or justify merging label pools. A larger TRAIN-only learnability test is the smallest next research step if requested. The active data goal is already satisfied.

The old baseline 85.96% and old main 100% use the original nine NCSLGR DEV utterances and cannot be ranked numerically against this new ASLLRP3 413-utterance DEV. No new TEST scoring or production selection occurred.
