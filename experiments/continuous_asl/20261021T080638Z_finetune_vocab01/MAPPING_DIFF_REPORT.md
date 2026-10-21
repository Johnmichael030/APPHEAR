# Class mapping and head compatibility

**PASS old mapping preservation; BLOCKED real expansion.** MAPPING_DIFF.csv lists all2,289 original IDs, including blank0 and UNK1. Every current-model ID and literal label is unchanged. The selected A head remains [2289,48], with frozen normalization.

The synthetic extension tests appended three unit labels in memory, checked exact old parameter rows and zero old-logit difference, serialization roundtrip and optimizer moments. A separate synthetic test performs an actual expanded AdamW step. Those models were not saved or used as recognition evidence. Only added rows are initialized; old logits are initially preserved, but the normalization denominator changes when classes are appended. B must demonstrate real old/new retention after actual training before any export or packaging.

Invalid or unsupported admission is rejected; blank/UNK targets are not permitted. Literal morphology, classifiers, spatial indices, punctuation and source conventions are not stripped or merged. No English alias was inserted as a recognition target. No production inference/export path was changed. The future B path requires real-data end-to-end testing after its current data blockers are resolved.
