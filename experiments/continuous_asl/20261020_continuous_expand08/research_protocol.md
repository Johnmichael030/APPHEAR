# Bounded main experiment after the overfit gate

Freeze the largest verified, deduplicated, training-eligible publisher label
pool in the canonical manifest. Prefer ASLLRP3 on a size tie. Keep NCSLGR2 and
ASLLRP3 conventions as separate training pools: their publisher documents
different glossing conventions and dominant-hand scope. Do not canonicalize
labels or combine these pools to manufacture a2,000-example training count.
Reference original media/features directly; do not copy videos.

The ASLLRP3 task is the publisher's literal dominant-realized occurrence-gloss
sequence, including hidden events, complete fingerspelled-word labels and
compounds. Solely nondominant events are not supplied in that CSV and must not
be invented. This is not comprehensive ASL coverage or translation supervision.

Require the eligible eight-utterance overfit diagnostic v2 to PASS, including
CUDA optimization, loss reduction, decoded TRAIN WER≤5%, no overlapping
publisher spans, and checkpoint reload parity. The original numerical v1 is
preserved but cannot unlock this run because its subset contained a globally
overlapping publisher utterance. The unexecuted warm-start draft is superseded.

Keep the same causal48-channel TCN, RF61,378features,10Hz real-frame sampling,
greedy CTC, FP32, AdamW lr0.001/wd0.0001, clip5, batch4/accumulate2 and seed
20261020. Fit vocabulary and mean/std on this frozen TRAIN pool only. No lower
learning rate or architecture change is presumed to solve the regression.

Run a one-epoch four-TRAIN/two-DEV CUDA smoke with save/reload parity, then one
bounded main run: maximum60epochs/600seconds, DEV patience15. This extends the
bounded optimization window for the larger corpus; the small-subset diagnostic
proves learnability, not that this budget or more data guarantees accuracy.
Select by the exact original9DEV WER then loss. Preserve its literal targets,
feature hashes and decoding. Stop cleanly on memory, disk or numerical failures.

The fixed9DEV contains57 reference tokens from the older NCSLGR source, with
incomplete signer metadata. For ASLLRP3 training this is a limited cross-corpus,
cross-annotation-version diagnostic, not a within-ASLLRP or signer-independent
benchmark. Report OOV counts and all S/D/I. Never adjust DEV targets, add a new
DEV population, or use reserved17TEST in fitting/model selection.

Record actual unique TRAIN consumed, repeated utterance presentations, epoch
times, full fit/reload duration, command wall time, CPU model latency, GPU
allocation/reservation and sampled device memory, hashes and failures. Compute
selected-checkpoint TRAIN WER only after model selection is frozen. Poor results
are reported as measured; no further tuning or model promotion follows here.
