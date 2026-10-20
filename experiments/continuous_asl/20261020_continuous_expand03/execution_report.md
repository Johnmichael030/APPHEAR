# Continuous ASL expansion and executed bounded comparison

**Execution: PASS. Overall readiness: PARTIAL.** All 223 eligible public-index candidates are now acquired, decoded, feature-compatible and unique. Both predeclared GPU runs completed and their selected checkpoints were frozen before custom test acquisition. Completed work was reused; no retraining, checkpoint selection or preprocessing of completed samples was repeated during this resume.

## Actual acquisition and preprocessing

The 79-clip TRAIN/DEV download checkpoint contained 47,233,931 bytes. This continuation added 14 custom test recordings (21,477,062 bytes) and 93 additional catalog recordings (36,346,021 bytes): **107 additional unique raw recordings since that checkpoint**. The active experiment now contains 186 new raw recordings / 105,057,014 bytes; 37 earlier raw recordings are reused and never counted as new.

All 5,219 source-index rows were audited. All 223 eligible candidate recordings passed source/annotation joins, decode and raw/full-decoded-pixel uniqueness checks. New v2 preparation produced 93 feature sets and reused 130 verified feature/decode records. There were 0 failed downloads, 0 preprocessing failures and 0 content duplicates. Total catalog observations: 7,186; original source target tokens: 1,162.

The conservative acquired-raw total including the unchanged How2Sign archive is 1,895,291,702 / 2,147,483,648 bytes, leaving 252,191,946 bytes. Additional NCSLGR files were bounded to 8 MiB each, 1 GiB per active run, 2 GiB aggregate with How2Sign and 10 GiB disk headroom. Existing archive verification was reused without rehashing/re-extracting it.

| Catalog role | Unique recordings | Use |
|---|---:|---|
| TRAIN_POOL_NOT_FIT | 181 | Contains original 100 fit recordings plus 81 new future-pool additions; v2 additions were not fitted |
| PRIOR_DEV_QUARANTINE | 19 | Old 7 development recordings plus 12 additional recordings from the same quarantined collections; no fitting |
| FROZEN_DEV_REFERENCE | 9 | Original development-based selection unchanged |
| FROZEN_TEST_EXPOSED_LOCKED | 14 | Acquired after selection freeze and evaluated once; now exposed, cannot be relabeled as a fresh future test |

The completed comparison remains 100 TRAIN / 9 DEV, with 30 TRAIN feature sets reused from the prior experiment, 70 new TRAIN recordings and 9 new DEV recordings. Splits are source-collection-disjoint, with source/media/decode overlap checks. Signer IDs are unknown, so they are not signer-independent or official corpus partitions.

Initial TRAIN/DEV preparation took 165.959 s; custom test preparation took 56.197 s. Catalog v2 preparation took 195.088 s, including cached validation and 2947 new perception observations; extraction-only p50/p95 = 52.669 / 79.302 ms per observed frame. These measured CPU preprocessing times are separate from model fitting. Published compressed media may play at half speed: source/container timestamps are retained, but real-world sign durations are unverified.

## Completed matched CUDA comparison

RTX 3050 Laptop 4 GiB; FP32, seed 20261020, TRAIN-only 117-token vocabulary and identical normalization, batch 4 / accumulation 2, AdamW lr 0.001 / weight decay 0.0001, clip 5, greedy CTC blank 0. Selection used DEV WER then DEV loss, patience 10, bounded 30 epochs / 180 seconds per model. Effective TCN receptive field **61 observations**. CUDA input/logits/loss/gradient/optimizer state are recorded in epoch logs.

| Model | Epochs / selected | DEV WER; S/D/I (57 refs) | Fit+reload s | Epoch median s | Peak allocated VRAM MiB | Worker peak MiB |
|---|---:|---|---:|---:|---:|---:|
| reference48 | 14 / 4 | 85.96%; 10/39/0 | 5.7732 | 0.3803 | 68.08 | 1101.59 |
| compact24 | 22 / 12 | 96.49%; 6/49/0 | 7.9807 | 0.3295 | 66.77 | 1096.02 |

Both runs stopped at their registered patience limit. Times are actual measured fit/DEV/checkpoint/final-reload loops, excluding interpreter startup and initial setup. VRAM figures are PyTorch allocator peaks; whole-GPU physical peaks were not independently sampled. No OOM or nonfinite-training failure was recorded. CUDA CTC backward warned about nondeterminism; fixed seeds do not imply bitwise reproduction.

| Model | TRAIN observations / complete epoch second | TRAIN+DEV observations / complete epoch second | Epoch min / max s |
|---|---:|---:|---|
| reference48 | 7541.38 | 8601.46 | 0.3440 / 0.7264 |
| compact24 | 8529.08 | 9728.01 | 0.3153 / 0.6675 |

Throughput derives from actual completed epoch durations and actual precomputed-feature observations (3,059 TRAIN + 430 DEV per epoch), including DEV/checkpoint overhead. It is not source RGB, camera or Android throughput. Per-epoch durations and throughput are saved in comparison_completed/epoch_throughput.csv.

## Frozen diagnostic test and model-only timing

The 14-utterance / 86-reference-token custom test is one source collection, with unknown signer clustering. It was downloaded only after both checkpoint decisions were frozen; acquisition timestamps were checked against the freeze. It was not used for fitting, early stopping, threshold tuning or model selection.

| Model | TEST WER; S/D/I | CPU model-only p50 / p95 ms |
|---|---|---|
| reference48 | 97.67%; 23/61/0 | 1.956 / 2.472 |
| compact24 | 98.84%; 13/72/0 | 1.336 / 1.660 |

Unchanged historical 39-token baseline: TEST WER 98.84% (9/76/0 S/D/I), 50 OOV tokens. Its restricted vocabulary differs from the matched 117-token pair, so it is only a historical diagnostic contrast. Matched models have 9 OOV DEV tokens / 57 references and 9 OOV TEST tokens / 86 references.

**Compact24 remains unselected.** DEV WER worsened by 10.53 percentage points. Same-input 2-thread CPU median model latency fell 31.72%, but TEST WER worsened by 1.16 points. Timing used the same 9 DEV sequences, 5 warmups and 20 repeats per sequence, 180 measurements/model. These measurements exclude perception, buffering, decoder delivery and Android integration. No joint accuracy/speed improvement, real-world accuracy or 15 completed FPS claim is supported.

## Checkpoints and frozen evidence

- reference48: `D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020_continuous_expand03\models\reference48\checkpoints\epoch_004.pt`
  SHA256 `489ec5883ca440364ab10690ee5cfb9d0b0b0e6942df8f477509ea4ba6faf99c`; configuration `models/reference48/config.json`; status `models/reference48/training_status.json`; every epoch retains a resumable optimizer checkpoint and a separate model CSV log.
- compact24: `D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020_continuous_expand03\models\compact24\checkpoints\epoch_012.pt`
  SHA256 `845e72314bcfbf5dd70dc5fa8fa59b324d72c5a5f3a68bda2cd988ef1f32823a`; configuration `models/compact24/config.json`; status `models/compact24/training_status.json`; every epoch retains a resumable optimizer checkpoint and a separate model CSV log.

Plan/config/script/manifest/checkpoint hashes match model_selection_freeze.json. Reloaded predictions match the selected-epoch evidence. Completed status verification after resume does not launch training. The early RF121 draft arithmetic error and its failed pre-fit focused check remain preserved; training_plan.v2.json corrected the value to 61 before fitting. The subsequent 6 focused checks passed. No broad repository-suite green claim is made.

## Labels, remaining blockers and permissions

Source provenance is the official BU NCSLGR index, joined by exact collection/utterance/camera/source URL and index row. Ordered main-gloss strings, case, locus punctuation, repetitions, grammar particles and HOLD targets are retained verbatim. This metric is literal source-main-gloss-token WER, not standardized lexical-only gloss accuracy. No English translation was used as a gloss label. Independent sign boundaries and signer IDs are null rather than inferred.

The remaining index rows are explicitly excluded: {"COMPLEX_GLOSS_TOKEN_BOUNDARIES_NEED_STRUCTURED_ANNOTATIONS": 613, "OTHER_CAMERA": 3337, "SHARED_VIDEO_NEEDS_AUTHENTIC_UTTERANCE_TIMING": 1046}. Shared recording URLs need authentic utterance timing; quoted/classifier/fingerspelling descriptions need structured annotations before defensible tokenization. Other-camera views are not credited as extra utterances. The next practical data step is an authorized official DAI structured XML export and per-utterance signer/timing mapping. Do not guess export URLs or participant identities. See source_review.md for checked NCSLGR/How2Sign/OpenASL/ChicagoFSWild access and task constraints. No further verified, fitting gloss payload is available under this mission's current supervision/access evidence.

How2Sign remains the verified Validation archive with English sentences, not gloss supervision. Its official Validation ID and earlier HTTP_404 URL/error remain recorded in execution_summary.json and original evidence. The official 31 GB TRAIN archive is outside the 2 GiB cap and was not downloaded. Existing archive integrity evidence was reused. BU terms permit the reviewed private research use; keep media, derivatives and checkpoints local and do not infer production/distribution permission.

Continuous streaming acceptance remains blocked by absent authentic per-sign timing, consented no-sign recordings, reliable signer-disjoint partitions and normal-app end-to-end latency measurements. The separate consented pilot still has no real participant consent/data; synthetic pilot fixtures are not model evidence. Do not tune thresholds against this exposed custom test or the frozen 2,620-row isolated test.

Initial custom test acquisition was not executed when approval review hit a usage limit at approximately 02:14 local. A later fresh approval at 09:58 local succeeded; it is no longer an acquisition blocker. No completed artifacts were removed.

## Preservation and validation

Fresh validation passed all 223 source/feature joins, timestamps, CTC feasibility, exact uniqueness, catalog/fitted-role joins, post-freeze TEST timing, frozen hashes, 4 preservation guard SHA256+mtime checks and 3338 protected size/mtime checks. Original full-hash evidence was reused; this is not a new full content rehash of all original files. Production models, thresholds, Android assets, old experiments and frozen isolated results are unchanged. No dependency was reinstalled.

Primary artifacts: execution_summary.json, readiness_validation.json, preservation_final.json, expanded_catalog_v2/manifest.json, expanded_catalog_v2/manifest.csv, expanded_catalog_v2/index_inventory.csv, expanded_catalog_v2/quality.json, comparison_completed/comparison.json, final_results.json, artifact_manifest.json. Source HTTP receipts, exact byte/hash joins and acquisition ledgers remain under data/raw and logs.

## Exact PowerShell commands

```powershell
Set-Location -LiteralPath 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite'
$run = 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020_continuous_expand03'
$py = 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261019T155428Z_cslr01\runtime\venv\Scripts\python.exe'
& "$run/Invoke-Research.ps1" -Phase Status
& $py -B "$run/tools/verify_completed.py" --artifacts
& "$run/Invoke-Research.ps1" -Phase Verify
```

Historical commands used for the completed runs (now safely return FROZEN_COMPLETE_NO_TRAINING):

```powershell
& "$run/Invoke-Research.ps1" -Phase Reference
& "$run/Invoke-Research.ps1" -Phase Candidate
```

Completed catalog orchestration (do not repeat successful downloads/preparation):

```powershell
& "$run/Acquire-Catalog.ps1"
& $py -B "$run/tools/expand_catalog.py" prepare
```

No incomplete training step remains in this frozen comparison. A future expanded-data experiment needs a new versioned output, a declared evaluation protocol and reserved evaluation data; signer-independent/streaming claims additionally need the structured signer/timing annotations and consented negative recordings described above. The 181-recording training pool is additive preparation, not evidence that 181 recordings were fitted.
