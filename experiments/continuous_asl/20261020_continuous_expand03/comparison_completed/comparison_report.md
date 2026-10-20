# Completed continuous-model comparison — executed, compact candidate unselected

Both predeclared runs were already complete when the resume request arrived. Checkpoint, status, plan, script and manifest hashes match the pre-TEST selection freeze. No training or selection was repeated. No training process was active at the resume inspection.

Same100TRAIN/9DEV source-collection-disjoint utterances,117-token TRAIN-only vocabulary, identical normalization arrays/decoding/seed and hyperparameters; only the planned channel width differs. Signer IDs are unknown; no signer-independent claim. Receptive field61.

| Model | Epochs / selected | DEV WER (S/D/I;57refs) | Fit+reload seconds | Median epoch seconds | Peak allocated VRAM MiB | Parameters |
|---|---:|---|---:|---:|---:|---:|
| reference48 | 14 / 4 | 85.96% (10/39/0) | 5.7732 | 0.3803 | 68.08 | 80373 |
| compact24 | 22 / 12 | 96.49% (6/49/0) | 7.9807 | 0.3295 | 66.77 | 26421 |

Candidate DEV WER regressed by10.53percentage points. Both DEV sets have9OOV tokens/57references. Both stopped at registered patience10; neither exhausted its30epoch/180second bound. Referencebestepoch4 and compactbestepoch12 were frozen before customTEST acquisition.

The surprisingly short fit times are measured small-dataset GPU loops, not estimates or complete command wall times. Per-epoch CSV includes data movement, DEV and checkpoint work. Derived throughput uses actual observation counts and complete epoch time; it excludes source RGB decoding/perception and cannot establish live15FPS. GPU input/logits/loss/gradients/Adam state were all recorded cuda:0. Peak process working set and allocator/free-memory samples are in comparison.json; no OOM/nonfinite failure was recorded.

TRAIN+DEV preparation took165.96s; new observed-frame extraction p50/p95=49.40/54.76ms. Old30feature sets and their decode evidence were reused. Frozen artifacts remain unchanged.

CustomTEST was acquired only after the recorded selection freeze. It has14utterances/86reference tokens from one previously untrained collection, with unknown signer clustering. These are exploratory diagnostic results, not current-app or signer-independent performance.

| Model | TEST WER | S/D/I | CPU model p50/p95 ms |
|---|---:|---|---|
| reference48 | 97.67% | 23/61/0 | 1.956/2.472 |
| compact24 | 98.84% | 13/72/0 | 1.336/1.660 |

Compact CPU median model latency is31.72% lower, but TEST WER is worse by1.16points. Timing uses the same9DEV input sequences,2CPU threads,5warmups/20repeats per sequence (180measurements/model). It excludes perception and normal-app delivery. No claim of joint speed/accuracy improvement or15completedFPS on Android.

The unchanged older39-token baseline scores98.84% WER on the same customTEST, with50OOV reference tokens. Its restricted vocabulary differs from the matched117-token comparison, so that contrast is not a controlled architecture ablation. Its checkpoint is unchanged.

Full predictions, losses, uncertainty limits, checkpoint paths and SHA-256 values are in comparison.json and ../final_results.json. Authentic source index strings are retained verbatim, including grammar/locus/HOLD notation; this is literal source-main-gloss-token transcription. Individual boundaries/no-sign exposure truth and normal-app latency remain unavailable.

Exact safe commands: Invoke-Research.ps1 -Phase Status; -Phase Reference -Resume and -Phase Candidate -Resume only verify frozen completed checkpoints and return FROZEN_COMPLETE_NO_TRAINING. Do not launch a new fit on these frozen decisions.
