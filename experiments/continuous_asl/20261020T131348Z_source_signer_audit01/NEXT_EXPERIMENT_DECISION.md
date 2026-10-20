# Next experiment decision — no training authorized

**Recommendation: SOURCE-TRANSFER PILOT, restricted to conservative recording families within ASLLRP3.** Audit execution PASS; experimental identifiability PARTIAL; independent dataset-vs-signer attribution and crossed design BLOCKED. This document is a design for later approval, not an instruction to launch a fit now.

Use `D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T131348Z_source_signer_audit01\folds\A_guarded_pooled_family_holdout.json`

Manifest SHA256 `87442a6f1241860e369a5c8f0768d583b3a4058f5a9ad03dcc135532c419c72f`.

Internal TRAIN1,500, eval206. TRAIN has3known signers and34published masters across6conservative families. Eval has2known signers already inTRAIN and5masters across2families. Keep every member of each master/collection/family on one side. No frozen externalDEV or TEST participation in assignment, fitting or selection.

Held-out family proxies, selected by metadata SHA256 ordering before group WER inspection:

- `asllrp_dai_participant_2`: `asllrp_dai_participant_2::2013-06-25`.
- `asllrp_dai_participant_3`: `asllrp_dai_participant_3::2012-02-14`.

All undated Ben files remain together inTRAIN. Their independent parent-recording ancestry is unresolved. Date-shaped prefixes are conservative provenance groups, not verified capture timestamps. No facial identity or master-name person inference was used.

**Question:** can an otherwise identical recognizer learn complete sequences from other families of known signers and transfer to these unseen families? Failure here, after a meaningful TRAIN-learning gate, would show that an unseen signer is not necessary for the failure. Success would motivate a separate controlled joint signer/source study; it would not prove the old DEV gap was caused by signer variation.

**Label controls:** preserve literal ASLLRP3 targets; fit vocabulary and normalization only on1,500internalTRAIN. Report full206eval WER with its81.05%token coverage, all80whole-sequence compatible-eval WER (zero reference OOV), and27support≥5whole-sequence WER. Report reference-error support strata and insertions separately. All IDs/views are frozen in the manifest. Compatible eval is74participant3/6participant2, so participant2 results remain descriptive; no group removal to improve a score. Do not compare these restricted views directly to the old full-task WER or85.96%nine-NCSLGRbaseline.

**Smallest controlled run if later authorized:** one fresh48channel causalCTC model with unchanged378features/decoder/loss/optimizer family; bounded smoke-first execution, not a seed/LR sweep. No older ASL recognition weights may be loaded because they saw these internal-eval records. Save seed/config/TRAIN-only vocab+normalization/optimizer/RNG/checkpoints and hashes. Use the internal compatible view for local diagnostic selection only; leave full-taskDEV/TEST frozen. Limit the first run to a declared short epoch/wall/VRAM budget before execution.

**Predeclared screen for a future run:** require TRAIN WER≤50%, at least10distinct predicted labels and multi-token output on at least25% of a fixedTRAIN monitor before interpreting transfer. Then require compatible-eval WER≤90%, at least10distinct nonblank predicted labels, and multi-token hypotheses on at least25% of80compatible examples. Also report full206WER, support5view and both family-level errors. Stop on nonfinite/resource failure or persistent collapse. These are engineering screening criteria, not statistical acceptance or production accuracy targets; two family clusters cannot support a population confidence claim. A failed TRAIN gate makes the source-transfer inference inconclusive rather than proving a source effect.

**Remaining confounders:** source family includes date/session/background/topic/utterance/annotation and label-frequency shifts; source and signer fixed effects are rank-aliased; unknown parent lineage and re-encodings cannot be fully excluded by hashes. Source-dataset transfer needs verified same-person bridges and compatible target contracts across datasets, neither present. An independently crossed factorial experiment would require shared, verified recording-condition/session definitions across multiple signers, with matched supervision and permissions; never manufacture those relationships.

The smaller497-task family split has only24compatible examples and two with≥5TRAIN support, so it is not the recommended multi-signer screen. Within-one-person guarded alternatives are included but have one held-out family each and limited shared-label coverage. No new data acquisition is required for this exploratory design; resolving lineage or planning a genuine crossed capture is the smallest next **metadata/data-design** step if independent causal attribution is required.

**No model training was executed in this task. Wait for a later explicit training authorization.** The immediate runnable command is the read-only verifier listed in commands_and_results.txt.
