# Fixed192 framing discriminator — no training launched

Date: 2026-10-21 (Asia/Shanghai). **PASS computational validity; PARTIAL recognition diagnosis; conditional training NOT JUSTIFIED.**

The frozen 192-example checkpoint reproduced its historical TRAIN result exactly. Small constant feature-space translations changed WER by less than one percentage point. The predeclared sensitivity gate failed, so the planned augmentation fit was not executed. No gate was relaxed after seeing results.

## Frozen design and measured results

The exact configuration was saved in config.json before model import/loading/inference. Only the existing 192 authorized TRAIN examples from ASLLRP3 were scored; no new model was fitted or selected. Model48,2,289 classes, blank0/UNK1, original TRAIN-only normalization, literal labels and greedy decoding were unchanged.

| Feature-space offset | TRAIN WER | Worsening from baseline | S / D / I | Exact sequences |
|---|---:|---:|---:|---:|
| Unmodified baseline | 10.587432% | 0.000000 pp | 100 /43 /12 | 76/192 |
| x +0.03 | 11.338798% | 0.751366 pp | 107 /48 /11 | 73/192 |
| x -0.03 | 10.860656% | 0.273224 pp | 110 /38 /11 | 73/192 |
| y +0.03 | 10.928962% | 0.341530 pp | 103 /45 /12 | 76/192 |
| y -0.03 | 10.655738% | 0.068306 pp | 99 /45 /12 | 76/192 |

Predeclared gate: baseline WER agreement within1e-6, computational checks PASS, and at least3of4 offsets worsening WER by at least0.10 absolute (10percentage points). Actual: **0of4**. Therefore `eligible_for_one_fit:false`.

Measured scoring time for the baseline plus four offsets: **13.639 seconds** on NVIDIA GeForce RTX3050 Laptop4GiB. This excludes Python imports/preparation. Total command time and peak diagnostic VRAM were not separately instrumented. Zero training epochs, zero optimizer updates and zero new checkpoints occurred. Torch2.8.0+cu128/CUDA12.8 were used; complete environment.json is preserved.

## Computational validity and preservation

- Reused read-only verify192.py:216 original input paths,80 historical held-out feature hashes and65 output hashes PASS. This only revalidated prior evidence; it did not rerun its training or held-out inference.
- Synthetic independent raw-observation→378feature parity:13 synthetic frames with varying image sizes and missing hand/body/proposal states; six translations. Maximum absolute difference1.1920929e-7, below the predeclared3e-7 tolerance. Synthetic results are engineering checks, not recognition evidence.
- Feature transform changes only valid current hand/body XY and present ROI center XY. Invalid geometry remains zero. Confidence, binary masks, dt, ROI size and every backward-delta channel remain exactly unchanged. No clipping, feature reduction, fabricated observation or time resampling occurs.
- Supplementary actual-feature checks passed for all192 TRAIN utterances /7,847 observations and all four offsets: invalid zeros/nonposition channels/input arrays and feature files preserved.
- 224 explicitly snapshotted original paths retain SHA256,size andmtime. No existing experiment, manifest, model, label, feature file, exclusion, production integration or avatar was edited. No reserved17TEST path was enumerated, opened, hashed or evaluated by this new experiment.
- No held-out inference or candidate selection occurred. The previously development-exposed80set and original27subgroup remain unchanged; they were not used to choose this TRAIN-only sensitivity gate.

## Interpretation and limits

The specific hypothesis that a constant3% coordinate shift causes a large failure of this memorizing192 model is **not supported**. This is not a claim that physical framing, pose scale, crops, missing articulators, lighting, domain differences or signer generalization are solved. Feature-space translation does not rerun image perception or simulate those effects.

Historical evidence remains unchanged: this192 model achieves10.59%TRAIN WER but101.03%WER on the development-exposed known-signer80set; the22fully label-supported sequences still have97.50%WER. That large transfer gap is far greater than the small perturbation effect observed here. No model-quality improvement is claimed.

There is no defensible augmentation training candidate under the frozen gate, so the conditional30epoch/720step fit is blocked by its own evidence requirement. Do not run `--fit-if-gated`, expand offsets, lower the gate, or launch another longer same-data fit merely to obtain a training log.

## Vocabulary and next action

Adding output-class names is not supported by this evidence. The current model already has2,287 literal gloss labels; the1500TRAIN population contains1,182 labels seen in one utterance. This192 subset actually fits631 labels,414 of them in only one selected utterance. All1,433 saved192TRAIN hypothesis tokens and all389 saved historical80hypothesis tokens belong to its fitted631-label set, so simple vocabulary masking cannot correct those existing errors. These counts came from the earlier read-only review, not new held-out inference.

Preserve ASLLRP3 andNCSLGR2 as distinct supervision contracts. The practical next data step remains qualified adjudication of existing ambiguous/poorly observed TRAIN events and permissioned repetition of existing labels across independent recordings/signers, including no-sign evaluation. No qualified reviewer or new participant permission is currently established; no consent or ground truth was invented.

Independent Android lifecycle/abstention fixes are outside this owned research directory and should be reported separately from these unchanged WER results. Read-only verification below is the only next command needed for this completed discriminator.

## Artifact identities

- Frozen192 subset SHA256: `8a2234a3eafd0d336ebe888d7b8e391b17410540b62d702315d2f5cb0b3c5f48`.
- Vocabulary SHA256: `365b44d435c9def2bba6e0f347155c72bf2ca001bd365d5b10f55dbe91174715`.
- Normalization SHA256: `394a3862286b01b884d30952f767f8620969ea21c0384715254e4bddb2d41489`.
- Original initialization checkpoint SHA256: `e2dd03d4f4b84aa395be4cfba0239723678a2610f2e5da73c1df97d12dd3560a`.
- Evaluated frozen192 epoch30 checkpoint SHA256: `da4b47063e64f1eb5fb0dc974f3910befea4ff45c8542e0be9d43f6589dc57d0`.
- New checkpoint: **NONE**.

See SENSITIVITY.csv, sensitivity_predictions/, synthetic_translation_parity.json, actual_feature_mask_checks.json, config.json, integrity_results.json, run.log and artifact_hashes.sha256. Exact commands are in commands_and_results.txt. These are diagnostic results, not independent streaming acceptance or unseen-signer accuracy.