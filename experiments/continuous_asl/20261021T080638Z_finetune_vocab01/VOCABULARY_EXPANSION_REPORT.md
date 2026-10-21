# Vocabulary expansion evidence and blockers

**PARTIAL engineering; BLOCKED real new-class training.** The model still contains 2,287 literal labels plus blank 0 / UNK 1. New classes added: **0**. Verified admitted examples per new class: **none**. There are 510 proposal rows and 463 distinct proposed new labels; potential metadata occurrences must not be counted as eligible training examples.

| Source | Actual evidence | Admission decision |
|---|---|---|
| Local recorder | 480 JSON records and matching MP4s; 8 prompts x60; one signer, one session per prompt; HELLO is the sole new literal among these prompts | BLOCKED consent/permission records, qualified gloss/target review, genuine 378-feature extraction and independent validation |
| Top-level local videos | 232 named MP4s, 14,394,510 bytes; dictionary/demo-style isolated examples | UNKNOWN rights and authoritative targets; filenames are not labels |
| RIT SignStream3 | 278 media records; 276 after two existing overlap exclusions; 462 proposed new literals | BLOCKED 60fps/crop-origin contract: +2 native frames vs inclusive publisher span; no invented offset, trim or zero padding |
| ASLLRP3 | Existing 2,119 eligible examples already assigned 1,500 internal TRAIN /206 internal held /413 external DEV | No reassignment of held families to supply new classes |
| NCSLGR2 | 1,002 eligible under a different main-gloss annotation contract | Separate pool; no union on the basis of 378-dimensional features |
| How2Sign | Existing English translations | Not authentic gloss sequence supervision; no repeat acquisition |
| DawnSignPress | Official directory is visible, but publisher dataset document excludes its download | No media acquired; visibility is not authorization |

User follow-up supplied `D:\ASL VIDEOS DATASETS` as the record location. LOCAL_PERMISSION_SCAN.json lists the renewed read-only search across 1,223 files: no permission/consent/review-named records; 480 empty reviewer_notes fields; no consent/license/verified-label metadata keys found. The thesis's ethical section describes future approval and additional written consent; it is not an executed participant release. This does not prove that records cannot exist elsewhere. No participant pixel data, recordings or old schema arrays were used in fitting or perception.

Plan ranking uses actual recorder prompts, literal published occurrence counts, available sessions, missing support and implementation cost. RIT labels such as personal-name tokens and unusual morphology are retained literally; their spelling is not an English translation or a claim of broad user usefulness. All require target-contract review. Forty existing under-supported labels are included for collection/replay planning; no synthetic examples were created.

Implementation: vocabulary_extension.py preserves all old IDs and backbone/output rows, appends sorted approved literals, initializes only added rows with seeded Kaiming-uniform weights and median existing-label bias, and extends Adam moments with zeros only for the appended rows. Old raw logits remain exact at initialization, but extra logits necessarily alter the softmax denominator. Tests include roundtrip serialization, repeat/blank conventions via the canonical decoder, finite new-row CTC gradients and an actual synthetic AdamW update. Synthetic test classes are not actual trained vocabulary additions and no synthetic checkpoint was saved.

B admission requires reviewed permission/target/temporal evidence paths and hashes, disjoint recording families, duplicate/exclusion guards, frozen reviewed TRAIN/VAL records, literal label conversion, exact schema/hash/embedded labels, strictly increasing timestamps and CTC feasibility. The minimum 3 TRAIN /1 independent VAL utterances per new class is a technical smoke threshold, not evidence of generalization. New-data replay uses genuine old TRAIN examples, not held-out records. The future B driver is not proven on real expanded data; it currently fails safely before model fitting.

Authoritative source checks reused in this mission: [ASLLRP terms](https://www.bu.edu/asllrp/dai-terms.html), [SignBank terms](https://www.bu.edu/asllrp/signbank-terms.pdf), [publisher dataset description](https://www.bu.edu/asllrp/about-datasets.pdf), [annotation report](https://www.bu.edu/asllrp/rpt24/asllrpr24.pdf). Research permission, redistribution and production-use rights are distinct. No new dataset payload was downloaded.
