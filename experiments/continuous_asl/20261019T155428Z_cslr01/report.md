# AppHear continuous ASL mission — PARTIAL

**Result: a reproducible private research pilot, with poor diagnostic recognition and blocked release gates. The existing application was not changed.**

Repository: `D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite`. No .git metadata: current branch and baseline commit unavailable. The initial all-original-file preservation manifest is the baseline, not a historical branch name.

| Evidence | Measured result |
|---|---|
| GPU | RTX 3050 Laptop, 4 GiB; tensor, logits, loss and gradients verified on cuda:0 |
| Data | 37 public-index-linked BU NCSLGR videos; 12,159,833 bytes; 2798 decoded frames / 944 selected observations |
| Split | 30 train / 7 validation by collection; no shared collection/content; signer IDs unknown |
| Training | FP32 causal TCN, 76,551 parameters, batch 4; 14 epochs; early stop; best epoch 8 |
| Sequence result | **88% gloss WER: 6 substitutions + 16 deletions, 25 reference glosses**; diagnostic only |
| Robustness | 92% WER after one-third frame drop; 92% with both hands unobserved; artificial corruption only |
| Non-sign / boundaries | BLOCKED: no valid negative or individual-boundary truth; no calibrated rejection |
| Export | ONNX 336,478 bytes; CPU parity max 2.86e-06, no decoded changes on 37 clips |
| CPU latency p50/p95 | Model 0.348/0.618 ms; pose/hand extraction 53.281/62.685 ms |
| Offline RGB-to-gloss | One clip, 17.35 processed observations/s, max 10 Hz sampling; 56.096/58.652 ms; startup 8.531 s |
| Memory | Training peak allocated VRAM 70,403,584 bytes; replay peak working set 366,141,440 bytes; training peak RAM unavailable |
| Preservation | All 2,932 originals / 1,569,935,151 bytes match SHA256 and modification times |

This is not a cross-signer/end-to-end product accuracy result. Four of 25 validation tokens are OOV; weak hand coverage and a tiny simple-gloss subset limit the experiment. No isolated-word or macro-F1 comparison is valid against the existing 236-way WORD system. CTC blank/UNK do not provide safe non-sign rejection. No production-readiness or Android 15 FPS claim is made.

Files created: code only under tools/continuous_asl; source/receipt/index/translation/sample data under this run's data tree; 14 checkpoints under checkpoints (best pointer epoch_008.pt, SHA `278f616ab5b6f9abe8584c124c542c221ff8e5ac641126eeddbff195cbc78180`); exported model/contract under export; isolated environment/cache under runtime/cache; reports/curves and logs here. No pre-existing code, model, GLB, manifest, checkpoint or dataset file was modified. Reproducibility manifest lists new artifact hashes. The default Python 3.9 environment and Android defaults remain intact.

Datasets: How2Sign project/script and 32,906 translation rows inspected, gloss unavailable in inspected published downloader; no RGB/keypoint bulk download. OpenASL excluded pending ND/training rights clarification. ASL Citizen/WLASL are isolated-sign sources and were not new training inputs; WLASL agreement still pending. NCSLGR's public index/37 direct clips enabled the private noncommercial pilot under documented BU terms; detailed XML requires a free DAI account, which was not created or bypassed. No redistribution, commercial/model-distribution or production rights are asserted. See dataset_licensing_report.md and retained official source URLs.

Engineering gates: 84 existing unit tests pass, one existing test lacks its frozen Keras artifact; five CTC tests, six hazard cases, nine archive safety cases and 15 downloader policy cases pass. HTTP range resume, fixture replay, causal-prefix and CPU export checks pass. A reproduced event-start eviction now has an isolated explicit EXPIRED_EVENT guard, but no Android correction was applied. Production top1 commitment, unused recognition flag and original unsafe merge implementation remain separate source findings. The new data path has a hard disjoint-output guard.

Remaining blockers: signer identities/cross-signer benchmark; timed gloss boundaries; consented real non-sign/hard-negative data and threshold calibration; adequate feature/label coverage; licensed deployment/distribution clearance; Android operator/adapter/enforced experimental-switch/device tests; representative camera load/FPS/latency/thermal evidence; absent original Keras source artifact; actual training-worker peak RAM. User confirmed no additional authorized local labels. The short GPU run is not a sustained thermal test. No Android build/install, app data clearing, default recognizer/flag change, or experimental integration occurred.

**Recommended next action:** keep this checkpoint unselected. Obtain appropriately licensed signer/timing labels and consented negatives, build a held-out streaming benchmark, and examine hand visibility/cadence on a representative device before scaling training. The present measurements favor investigating preprocessing coverage and latency first.

Detailed evidence: environment_report.md; architecture_audit.md; feature_schema_report.md; preprocessing_report.md; dataset_quality_report.md; baseline_report.md; evaluation_report.md; performance_report.md; model_parity_report.md; regression_report.md; review_report.md; non_sign_recording_protocol.md. Sequence edits/confusions and training curves are retained, with failures and missing metrics clearly labeled.

Exact safe continuation commands (the training phase is already complete):

```powershell
$repo = 'D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite'
$run = 'D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261019T155428Z_cslr01'
Set-Location -LiteralPath $repo
& "$repo/tools/continuous_asl/Verify-Preservation.ps1" -RunRoot $run
# Already early-stopped: this safely reports completion without more epochs.
& "$repo/tools/continuous_asl/Invoke-Training.ps1" -RunRoot $run -Resume
```


No command can resolve the blocked data/device gates without new evidence. Optional fresh replay commands are in resume_commands.md; all downloads/processing/export use fresh destinations and retain sources.
