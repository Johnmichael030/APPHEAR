# Immediate bounded pilot while ASLLRP features continue

Freeze all906 eligible NCSLGR TRAIN utterances (455 prior +451 newly verified
timed utterances), plus the unchanged original9DEV. No ASLLRP row, overlapping
publisher span, protected collection or reserved TEST enters fitting. This
prevents the CPU acquisition lane from changing the active training manifest.

Require clean overfit diagnostic v2: TRAIN WER≤5%, loss ratio≤0.25, CUDA
optimization and checkpoint reload parity, with all global overlap IDs excluded.
Keep the unchanged48-channel causal TCN/RF61/378features/greedyCTC/blank0,
FP32 AdamW0.001/wd0.0001, batch4/accumulate2, clip5, seed20261020. Fit this
TRAIN-only vocabulary and normalization. No architecture, decoder, label
canonicalization, blank penalty or production threshold changes.

Run one-epoch four-TRAIN/two-DEV smoke, then maximum30epochs/120seconds,
DEV patience10; selection remains exact9DEV WER then loss. Measure initial and
per-epoch TRAIN WER on a deterministic source-collection-stratified64-example
monitor; also measure full TRAIN WER for the final and selected checkpoints.
These are fitting diagnostics, never held-out evaluation evidence.

Predeclared continuation gate: final-epoch full TRAIN WER≤80%, monitor CTC
loss≤50% of its initial value, finite optimization and save/reload parity.
This is only a minimal learning signal, not acceptable recognition accuracy.
If it fails, block the queued main run and investigate measured predictions,
vocabulary/frequency/missingness and final-versus-DEV-selected behavior before
spending further GPU time. No blind long rerun or assumption that lower LR fixes
the problem. ASLLRP acquisition/feature validation continues independently.
