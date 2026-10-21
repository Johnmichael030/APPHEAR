# Recognition vocabulary decision

**PASS audit; no new recognition classes added.**

The current model has 2,287 literal target labels plus blank0 andUNK1. These are ASLLRP occurrence labels, including variants and spatial/classifier notation; they are not 2,287 interchangeable English words.

In the frozen1,500TRAIN utterances, 1,182 labels occur in only one distinct utterance and 1,808 occur in fewer than five. TRAIN has3publisher signer IDs. Repeated token occurrences within one utterance do not count as additional support.

Adding classes now would dilute already sparse supervision. The model output matrix, vocabulary IDs and genuine labeled examples must be trained/evaluated together; adding strings or English aliases cannot teach visual recognition. No target/label normalization, ontology merger, fabricated examples or unsupported semantics were introduced.

VOCABULARY_SUPPORT.csv contains the complete current literal label list, model IDs, distinct-utterance support, occurrence frequencies, signer/source coverage and existing English-equivalent mapping. A planning gap to5 is informational, not a validated quality threshold.

Next data step: qualified review of the existing12TRAIN event package, then consented natural phrases repeating a small chosen set of existing labels across independent signers/sessions, with predeclared no-sign and unseen-signer evaluation. No qualified reviewer or new participant permissions are currently established. English display coverage remains a separate75-label/40-pattern stage and was not expanded or presented as a recognition improvement.
