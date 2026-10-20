# Bounded continuation after the DEV-WER diagnosis

**PASS execution and preservation; PARTIAL recognition. Stopped after the initial three-epoch window because DEV did not improve.**

The user explicitly authorized this isolated research pilot after the completed engineering checks found no critical implementation bug or data-integrity failure. It continued the frozen epoch-17 weights, AdamW state and RNG state on the exact original 2,119 TRAIN / 9 DEV manifest. No acquisition or preprocessing was repeated. The prior epoch-2 selection and every earlier artifact remain unchanged.

Only the continuation budget and diagnostic stop protocol changed. Architecture48/RF61,378features, batch4/accumulation2, AdamW lr0.001/weight-decay0.0001, gradient clip5, FP32, literal vocabulary2741, original TRAIN normalization and greedyCTC blank0/UNK1 stayed fixed. New directory contains all outputs.

| Cumulative epoch | TRAIN optimization CTC loss | Fixed64 TRAIN monitor WER | DEV WER | DEV S/D/I | Epoch seconds |
|---|---:|---:|---:|---:|---:|
| 18 | 3.052328 | 73.60% | 100.00% | 14/43/0 | 10.2894 |
| 19 | 2.884185 | 77.28% | 100.00% | 13/44/0 | 7.9258 |
| 20 | 2.745125 | 72.64% | 100.00% | 12/45/0 | 8.2455 |

The initial fixed64 monitor had 78.72%WER. Full2,119TRAIN WER improved from frozen epoch17:71.9532% to epoch20:65.2266% (2863S/8479D/42I over17453tokens). These are TRAIN diagnostics, not generalization.

The new pilot selected epoch18 on its own DEV WER/loss tuple: full TRAIN WER70.1713%, DEV100% (14S/43D/0I). The original selected epoch2 is not changed. New local-best checkpoint SHA256: `a0673a7cf56b147445f0123884bc1cda0261c2c34602b5ac33f2b26b0715953b`. The final epoch20 checkpoint is also retained, with SHA256 `782a748a9b6ac495a2198a329a9e2864bf23344c136f6f43e75a13e5f59bec82`.

Predeclared gates: first3additional epochs; maximum8additional epochs/120seconds, wall checked between epochs. Extend beyond3only if DEV improves at least2of57tokens versus the epoch17 parent and TRAIN monitor loss does not worsen. Stop on nonfinite loss/gradients, VRAM/resource guard, TRAIN blank collapse, or lack of TRAIN learning after2epochs. DEV remained100%, so the continuation gate failed and status is NO_CREDIBLE_DEV_GAIN_INITIAL_WINDOW. It was not extended to8epochs. The two-token gate is a research heuristic, not statistical significance on this tiny dependent DEV.

Actual CUDA epoch loop: 26.4644s; fit/reload/fullTRAINdiagnostics: 46.6959s. Complete process wall time including Python startup/data checks is separately recorded: 62.9377s. Median epoch: 8.2455s. Peak PyTorch allocated: 1305171456bytes (1.2155GiB); reserved: 1361051648bytes. Whole-device sampled occupancy peak: 1447.0MiB over296samples at200ms; device samples can miss brief peaks and differ from allocator memory. No OOM occurred.

Inputs/logits/loss are logged on cuda:0, optimizer state is inherited on CUDA, gradients are finite, and new optimizer steps completed. Each new checkpoint contains optimizer and RNG state. The selected new checkpoint was reloaded from disk and reproduced its DEV hypotheses exactly. Fixed seeds do not guarantee bitwise CUDA CTC backward determinism; the framework warning is preserved in the console log.

Frozen manifest SHA256: `adebeb9e2f20179eb59b7f79805ea9226ad6c9c0594e6cba193c771947cdaf40`. All2,119unique TRAIN utterances were consumed in each of3epochs (6,357presentations). The nine DEV examples are unchanged. The reserved17TEST set was not used. The other906eligible pool was not mixed in; all3,025eligible utterances and the six overlap/one frame-count exclusions remain preserved. Final metadata checks passed for20745protected/indexed paths.

## Prediction example

Literal references/hypotheses below use the existing conventions; all nine pairs for every epoch are saved in dev_epoch_*.json.

| Reference | New selected epoch18 hypothesis |
|---|---|
| `IX-3p:i LEARN+AGENT MOTHERwg CAR BUY FINISH IX-3p:i` | `EXPRESS LAPTOP` |
| `LEARN+AGENT IX-loc:i HOLD MOTHER FUTURE BUY CAR` | `PLAY FRIEND` |
| `IX-3p:i LEARN+AGENT POSS-3p:i FATHERwg HOUSE WHICH BUY IX-loc-arc` | `TYPE IX-3p-pl-arc` |

Interpretation: more optimization improved TRAIN without any DEV token-error reduction. This is consistent with the diagnosis of weak early fitting plus source/label/evaluation mismatch, but does not uniquely identify a cause. The same old DEV is a valid fixed literal-token diagnostic; signer-independent generalization remains unverified. Source scope, OOV tokens and feature-observation differences are quantified in the linked diagnostic directory. No checkpoint beats the frozen85.96%DEV baseline and none is promoted.

Next action: finish signer/source mapping and establish a same-convention independent evaluation protocol before another optimization loop. Do not use the reserved17TEST set for that tuning. No automatic longer training is recommended after this failed DEV gate.

Exact execution command (completed; wrapper verifies the completed checkpoint on repeat):

```powershell
& 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T102748Z_continuation_pilot01\Run-Pilot.ps1'
```

Read-only verification:

```powershell
& 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T102748Z_continuation_pilot01\Verify-Pilot.ps1'
```

Full diagnostic report: `../20261020T051631Z_devwer_diagnosis01/DIAGNOSIS.md`. The epoch2 checkpoint remains frozen and must not be promoted. Production integration, thresholds, Android app data, APKs, avatars and GLB assets are unchanged.
