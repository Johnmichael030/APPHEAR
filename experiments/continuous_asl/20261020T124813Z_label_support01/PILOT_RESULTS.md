# One controlled restricted-task CUDA pilot

**PASS execution; PARTIAL recognition; BLOCKED promotion.** Fresh425class/48channel model, samecausal378schema, no dropped feature blocks or downsampling. TRAIN497, compatibleDEV92, originalDEV413; all original split boundaries and exclusions preserved. Choice≥5parent support was frozen without scores. This is not a clean one-factor ablation: membership, vocabulary and TRAIN-fitted normalization change together for the new task.

Protocol saved before fitting: pilot497/protocol.json. Seed20261020, FP32,AdamWlr.001wd.0001, batch4accumulation2,clip5;30epoch/90second bound, collapse gate10, meaningful-learning/DEV gate20, compatibleDEVpatience2 thereafter. Only one new fit executed. No prior weights loaded. All checkpoints have optimizer/RNG/config/manifest; selected predictions reproduced after checkpoint reload. CUDACTCnondeterminism warning retained.

|Epoch|TRAINWER|Compatible92DEVWER|Original413DEVWER|TRAINCTC|CompatibleDEVCTC|TRAINblank|CompatibleDEVblank|TRAINlabels|CompatibleDEVlabels|
|---|---|---|---|---|---|---|---|---|---|---|
|1|100.000%|100.000%|100.000%|5.676|5.625|1.0000|1.0000|0|0|
|5|94.522%|98.592%|98.022%|4.842|5.700|0.9626|0.9714|13|9|
|10|88.255%|98.994%|98.208%|3.738|6.417|0.9415|0.9682|90|23|
|15|73.058%|98.592%|98.424%|2.641|7.450|0.9135|0.9682|220|21|
|20|43.677%|98.390%|98.053%|1.680|8.502|0.8651|0.9541|362|44|

Stopped `NO_MEANINGFUL_LEARNING_GATE20` at20epochs /1260updates. Fit loop77.559s; with reload85.785s; wrapper99.272s. GPUNVIDIA GeForce RTX 3050 Laptop GPU, peakallocated74,494,976bytes (0.069GiB),reserved77,594,624bytes. Per-epoch actual fit/evaluation durations and memory gates logged. No estimate substituted for a measurement.

Selected epoch20 by compatibleDEV only. TRAINWER43.6773%,CTC1.679546,blank0.865057,diversity362; compatibleDEVWER98.3903%,CTC8.501956,blank0.954093,diversity44; originalDEVWER98.0532%,CTC10.698586,blank0.959776,diversity78.

Checkpoint `D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T124813Z_label_support01\pilot497\checkpoints\epoch_020.pt`

SHA256 `2e5e6606a1942c800217a5c1ae419c1232291e6242576ce130effc9a7deb7ed6`.

Full originalDEV and compatibleDEV results are separate. EVALUATION_CONTEXT.csv also rescored saved older predictions on the identical92IDs, without retraining or tuning. Restricted-label WER must not be compared directly with the full-vocabulary baseline or nine-utterance85.96%NCSLGRbaseline. The original full-task results remain frozen.

Literal compatibleDEV examples:

- `asllrp3_91bcefffdcfcd1e9`: `["ns-fs-BOB", "IX-loc:i", "HAVE", "BIG", "#DOG_2", "#NO", "IX-3p:i", "(1h)HAVE", "SMALL"]` → `["POSS-3p:i", "IX-1p"]`
- `asllrp3_efdd3f97599a2a2c`: `["DRESS/CLOTHES", "IX-loc:i", "ON", "#SALE", "IX-loc:i", "NOT", "#NO"]` → `["IX-1p", "WANT"]`
- `asllrp3_ef05548bd671c9b7`: `["POSS-1p", "FRIEND", "WANT+NEG", "BUY", "PLAID", "(2h)SHIRT", "WANT+NEG"]` → `["SNOW", "WANT"]`
- `asllrp3_c061590467f3a38b`: `["TEACH+AGENT", "IX-loc:i", "IX-3p-pl-arc", "HEAR/LISTEN", "IX-3p:i", "WHO"]` → `["IX-1p"]`
- `asllrp3_99ce19a46f4b03a1`: `["TEACH+AGENT", "IX-loc:i", "START", "NICE/CLEAN", "TIME+"]` → `["IX-1p"]`
- `asllrp3_a4c90a286bc87eaf`: `["MOTHER", "IX-loc:i", "SICK", "QMwg", "HOW"]` → `["IX-1p", "NOT"]`

All hypotheses, references, lengths, token counts and blank fractions are in pilot497/best_*.json and epoch JSONs. NoTESTaccess or promotion.

Next action: keep this bounded run stopped. Review its support-stratified and signer/source error evidence before any longer fit. A useful next no-fit diagnostic is whether supported, correctly acquired TRAIN labels transfer within unseen source recordings versus the unseen signer, using a predeclared additional TRAIN-only source split. That can distinguish memorization/source dependence from a signer effect more directly. It requires no new data or TEST access. Do not infer that label filtering solved generalization, or tune thresholds against theseDEV scores.
